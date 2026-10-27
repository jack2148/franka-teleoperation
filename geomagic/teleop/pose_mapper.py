"""Sample-driven Geomagic mapping with validated transitions and a warm path.

Expose translation, tool-Z rotation, full relative 3D orientation, or a combined
6DoF pose from common activation anchors. No MuJoCo ownership.
"""

from dataclasses import dataclass
import math
import os
from typing import Optional

import numpy as np

from geomagic.vendor.geomagic_core.controller import PositionTeleopController
from geomagic.vendor.geomagic_core.frames import FrameMap
from geomagic.vendor.geomagic_core.pose_mapping import AnchoredPoseMapper
from geomagic.teleop.runtime_rotation import (validate_rotation_matrix, relative_quaternion, continuous_rotvec)
from geomagic.teleop.runtime_profile import get_profiler
from geomagic.vendor.geomagic_core.rotations import rotation_vector_to_matrix
from geomagic.vendor.geomagic_core.state import GeomagicState


# Frame map from import_v2/config/mapping.yaml remains unchanged.
# Dataset collection contract: rotation signs [+1, -1, +1].
# Mapping gains do not impose standalone task workspace bounds.
M_POSITION = ((0., 0., 1.), (1., 0., 0.), (0., 1., 0.))
POSITION_SCALE = (0.7, 0.7, 0.7)
ROTATION_SCALE = 0.4
ROTATION_CONTROL_SIGN = (1., -1., 1.)


def _map_relative_vector(vector):
    return ROTATION_SCALE * np.asarray(ROTATION_CONTROL_SIGN) * (np.asarray(M_POSITION) @ vector)


MODES = ("position", "rotation_z", "rotation", "6dof")
POSITION_MODES = ("position", "6dof")
ROTATION_MODES = ("rotation_z", "rotation", "6dof")
MODE_LABELS = {"position": "Position", "rotation_z": "Rotation Z", "rotation": "Rotation 3D", "6dof": "6DoF"}
if get_profiler():
    relative_quaternion = get_profiler().wrap('rotation.quaternion_relative', relative_quaternion)
    continuous_rotvec = get_profiler().wrap('rotation.continuous_log', continuous_rotvec)
    rotation_vector_to_matrix = get_profiler().wrap('rotation.exp', rotation_vector_to_matrix)
    validate_rotation_matrix = get_profiler().wrap('rotation.actual_validation', validate_rotation_matrix)
    _map_relative_vector = get_profiler().wrap('rotation.frame_mapping', _map_relative_vector)


def _copy(value):
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CartesianTarget:
    position: np.ndarray
    rotation: np.ndarray
    mode: str
    fresh: bool
    active: bool
    clutched: bool
    reanchored: bool
    reason: str
    mapper_updated: bool
    synchronized: bool = False


class GeomagicPoseMapper:
    """Consume accepted samples once; check receive-age on EVERY control tick.

    First attach/session/reconnect/mode switch requires a fresh sample sequence
    and the reference activation gate. Intent is discarded on transition edges;
    every new anchor uses ACTUAL pose. Repeated sequences cannot advance mapping.
    """

    def __init__(self, initial_position, initial_rotation, *, mode="position",
                 activation_samples=50, stale_timeout_s=0.050):
        if mode not in MODES:
            raise ValueError("mode must be position, rotation_z, rotation (3D) or 6dof")
        if type(activation_samples) is not int or activation_samples < 2:
            raise ValueError("activation_samples must be an integer >= 2")
        if not math.isfinite(stale_timeout_s) or stale_timeout_s <= 0:
            raise ValueError("stale_timeout_s must be finite and positive")
        self.mode = mode
        self._reference_mapping = os.environ.get("MIMIC_GEOMAGIC_REFERENCE_MAPPING") == "1"
        self.activation_samples = activation_samples
        self.stale_timeout_s = stale_timeout_s
        self.mapper_updates = 0
        self.reset(initial_position, initial_rotation)

    def reset(self, position, rotation):
        """Explicit episode reset from the caller's authoritative robot pose."""
        value = np.asarray(position, dtype=float)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError("initial position must be a finite 3-vector")
        self._position = value.copy()
        self._rotation = validate_rotation_matrix(rotation).copy()
        self._latest: Optional[GeomagicState] = None
        self._lost = False
        self._last_buttons = None
        self._pending_sync = True
        self.mapper_updates = 0
        self._rebuild()

    def _rebuild(self):
        self._relative_vector = np.zeros(3)
        self._device_anchor_quaternion = None
        # GEOMAGIC_MAPPING_LIMITS = DISABLED: no Geomagic-side step or
        # excursion clamp, no absolute workspace box, no tool offset. The FR3
        # side (condition_target, DLS, joint limits) bounds actual motion.
        mapper = AnchoredPoseMapper(
            FrameMap("geomagic_base", "fr3_world", np.array(M_POSITION)),
            POSITION_SCALE, stale_timeout_s=self.stale_timeout_s,
            position_enabled=self.mode in POSITION_MODES,
            orientation_enabled=self.mode in ROTATION_MODES, orientation_scale=ROTATION_SCALE,
            rotation_control_sign=ROTATION_CONTROL_SIGN,
        )
        self._controller = PositionTeleopController(
            mapper, self._position, deadman_button_index=0, clutch_button_index=1,
            activation_min_consecutive_samples=self.activation_samples,
            initial_target_rotation=self._rotation,
        )

    def synchronize_actual(self, position, rotation, *, rebuild=False):
        """Discard intent once; preserve the freshness gate unless rebuilt."""
        position = np.asarray(position, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("actual position must be a finite 3-vector")
        self._position = position.copy()
        self._rotation = validate_rotation_matrix(rotation).copy()
        if rebuild:
            self._rebuild()
            self._last_buttons = None
        else:
            self._controller.target = self._position.copy()
            self._controller.target_rotation = (
                self._rotation.copy() if self.mode in ROTATION_MODES else None)
            self._controller.active = False
            self._controller.mapper.deactivate()
        self._pending_sync = False
        self._relative_vector = np.zeros(3)
        self._device_anchor_quaternion = None

    def set_mode(self, mode, *, actual_position=None, actual_rotation=None):
        if mode not in MODES:
            raise ValueError("mode must be position, rotation_z, rotation (3D) or 6dof")
        if mode == self.mode:
            return False
        self.mode = mode
        if actual_position is not None and actual_rotation is not None:
            self.synchronize_actual(actual_position, actual_rotation, rebuild=True)
        else:
            self._pending_sync = True  # caller supplies actual pose on next tick
            self._rebuild()
        return True

    def _snapshot(self, *, fresh=False, active=False, clutched=False,
                  reanchored=False, reason="held", updated=False, synchronized=False):
        return CartesianTarget(_copy(self._position), _copy(self._rotation),
                               self.mode, fresh, active, clutched, reanchored,
                               reason, updated, synchronized)

    def update(self, sample, *, now, actual_position, actual_rotation,
               session_changed=False):
        """session_changed is an event, to be acknowledged by the network caller.

        Pass client.latest_state(), not poll()'s optional new-packet return.
        A missing/invalid/stale state disables updates immediately. Session IDs
        are also compared directly, so restart works for synthetic sources.
        """
        actual = np.asarray(actual_position, dtype=float)
        if actual.shape != (3,) or not np.isfinite(actual).all():
            raise ValueError("actual position must be finite and 3D")
        actual_rotation = validate_rotation_matrix(actual_rotation)
        if not math.isfinite(now):
            raise ValueError("now must be finite server monotonic time")
        reset = bool(session_changed or self._pending_sync)
        new = False
        if sample is not None:
            if self._latest is None or sample.session_id != self._latest.session_id:
                reset = True
                new = True
            elif sample.sequence > self._latest.sequence:
                # Detect interruption even if caller missed all stale ticks.
                if sample.timestamp - self._latest.timestamp > self.stale_timeout_s:
                    reset = True
                new = True
            if new:
                self._latest = sample
        current = self._latest
        fresh = False
        if sample is not None and current is not None:
            age = now - current.timestamp
            fresh = bool(
                sample.valid and current.valid and current.source_valid
                and current.units == "m" and current.frame_id == "geomagic_base"
                and len(current.position_xyz) == 3
                and current.orientation_xyzw is not None
                and len(current.orientation_xyzw) == 4
                and current.pose_is_finite() and current.orientation_is_usable()
                and math.isfinite(age) and -0.01 <= age <= self.stale_timeout_s
            )
        synchronized = False
        if reset or (not fresh and not self._lost):
            self.synchronize_actual(actual, actual_rotation, rebuild=True)
            synchronized = True
        if not fresh:
            self._lost = True
            return self._snapshot(reason="stale_or_invalid", synchronized=synchronized)
        self._lost = False
        if not new:
            return self._snapshot(fresh=True, active=self._controller.active,
                                  clutched=current.white_button, reason="held_sample",
                                  synchronized=synchronized)
        buttons = (current.grey_button, current.white_button)
        if self._last_buttons is not None and buttons != self._last_buttons:
            self.synchronize_actual(actual, actual_rotation)
            synchronized = True
        self._last_buttons = buttons
        # Active-interval hot path: the wrapper has already checked freshness,
        # finite pose, sequence and all safety edges. Avoid repeated controller,
        # FrameMap and SO(3) validation/serialization of the same intermediates.
        if ((not self._reference_mapping or self.mode == "rotation_z") and self._controller.active
                and not current.white_button and self._controller.mapper.anchored):
            pure = self._controller.mapper
            if self.mode in POSITION_MODES:
                self._position = pure.slave_anchor + np.asarray(POSITION_SCALE) * (
                    np.asarray(M_POSITION) @ (np.asarray(current.position_xyz) - pure.master_anchor))
                pure.target = self._controller.target = self._position.copy()
            if self.mode in ROTATION_MODES:
                if self._device_anchor_quaternion is None:
                    from geomagic.vendor.geomagic_core.rotations import matrix_to_quaternion_xyzw
                    self._device_anchor_quaternion = matrix_to_quaternion_xyzw(pure.master_anchor_rotation)
                relative = relative_quaternion(current.orientation_xyzw, self._device_anchor_quaternion)
                self._relative_vector = continuous_rotvec(relative, self._relative_vector)
                vector = _map_relative_vector(self._relative_vector)
                if self.mode == "rotation_z":
                    # Conjugating the mapped spatial rotation by the fixed
                    # robot anchor expresses its continuous log in tool axes.
                    # Its local Z component is anchor[:,2] dot spatial log.
                    yaw = float(pure.slave_anchor_rotation[:, 2] @ vector)
                    self._rotation = pure.slave_anchor_rotation @ rotation_vector_to_matrix((0., 0., yaw))
                else:
                    self._rotation = rotation_vector_to_matrix(vector) @ pure.slave_anchor_rotation
                if not np.isfinite(self._rotation).all():
                    raise ValueError("mapped rotation is non-finite")
                pure.target_rotation = self._controller.target_rotation = self._rotation.copy()
            self.mapper_updates += 1
            return self._snapshot(fresh=True, active=True, reason="active", updated=True,
                                  synchronized=synchronized)
        result = self._controller.update(current, actual, now, actual_rotation)
        self.mapper_updates += 1
        if result.reanchored:
            # The validated controller normally anchors to its held command.
            # Replace that anchor with actual at the activation/reanchor edge;
            # do not count or process the sample a second time.
            self.synchronize_actual(actual, actual_rotation)
            self._controller.mapper.activate(
                current, actual, now=now, held_target_xyz=actual,
                slave_actual_rotation=actual_rotation,
                held_target_rotation=actual_rotation,
            )
            self._controller.active = True
            self._controller.mapper.clutch.update(current.white_button)
            synchronized = True
        else:
            self._position = result.target_xyz.copy()
        if not result.reanchored and self.mode in ROTATION_MODES and result.target_rotation is not None:
            # Preserve the validated spatial delta, frame/sign convention and
            # left multiplication of the held FR3 anchor. All three rotation
            # components survive; the controller already handles hold/reanchor.
            self._rotation = result.target_rotation.copy()
        return self._snapshot(fresh=True, active=result.teleop_active,
                              clutched=result.clutch_pressed,
                              reanchored=result.reanchored, reason=result.reason,
                              updated=True, synchronized=synchronized)
