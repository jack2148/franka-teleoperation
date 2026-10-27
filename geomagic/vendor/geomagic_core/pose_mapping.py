# Vendored from geomagic_franka_teleop_import_v2/src/teleop/pose_mapping.py
# Changes: state import namespace; frames omits the unused force method.
# GEOMAGIC_MAPPING_LIMITS = DISABLED: the reference per-sample position step,
# anchor-relative position excursion, per-sample rotation step and
# anchor-relative rotation excursion clamps are removed (see PROVENANCE.json).
# Robot-side FR3 conditioning, DLS and joint limits remain authoritative.
"""Runtime-anchored position and orientation master-to-slave mapping."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Mapping, Optional, Sequence

import numpy as np

from .state import GeomagicState
from .clutch import Clutch
from .frames import FrameMap
from .rotations import (
    matrix_to_rotation_vector,
    orthonormalize_rotation_matrix,
    quaternion_xyzw_to_matrix,
    rotation_vector_to_matrix,
    validate_rotation_matrix,
)
from .scaling import scale_delta


class UnsafeSample(RuntimeError):
    """Raised when continuing motion would violate a Phase-1 safety rule."""


def rotation_control_sign_from_config(config: Mapping[str, object]) -> np.ndarray:
    """Read the separate relative-rotation control convention from config."""
    value = config.get("rotation_control_sign", {"x": 1, "y": 1, "z": 1})
    if isinstance(value, Mapping):
        values = [value.get(axis) for axis in "xyz"]
    else:
        values = value
    signs = np.asarray(values, dtype=float)
    if signs.shape != (3,) or not np.isfinite(signs).all() or not np.all(np.isin(signs, (-1.0, 1.0))):
        raise ValueError("rotation_control_sign must contain exactly x/y/z signs of +1 or -1")
    return signs


@dataclass(frozen=True)
class MappingResult:
    target_xyz: np.ndarray
    mapped_delta_xyz: np.ndarray
    master_anchor_xyz: np.ndarray
    slave_anchor_xyz: np.ndarray
    target_rotation: Optional[np.ndarray]
    mapped_rotation_vector: np.ndarray
    master_anchor_rotation: Optional[np.ndarray]
    slave_anchor_rotation: Optional[np.ndarray]
    sample_age_s: float
    clutched: bool
    reanchored: bool = False
    workspace_clamped: bool = False
    step_limited: bool = False
    orientation_excursion_limited: bool = False
    orientation_step_limited: bool = False


class AnchoredPoseMapper:
    def __init__(
        self,
        frame_map: FrameMap,
        position_scale: Sequence[float],
        stale_timeout_s: float,
        workspace_min_m: Optional[Sequence[float]] = None,
        workspace_max_m: Optional[Sequence[float]] = None,
        orientation_enabled: bool = False,
        orientation_scale: float = 1.0,
        position_enabled: bool = True,
        rotation_control_sign: Optional[Sequence[float]] = None,
    ) -> None:
        self.frame_map = frame_map
        self.scale = np.asarray(position_scale, dtype=float)
        scale_delta(np.zeros(3), self.scale)
        self.stale_timeout = float(stale_timeout_s)
        self.workspace_min = np.asarray(
            workspace_min_m if workspace_min_m is not None else (-np.inf,) * 3,
            dtype=float,
        )
        self.workspace_max = np.asarray(
            workspace_max_m if workspace_max_m is not None else (np.inf,) * 3,
            dtype=float,
        )
        self.orientation_enabled = bool(orientation_enabled)
        self.position_enabled = bool(position_enabled)
        self.orientation_scale = float(orientation_scale)
        self.rotation_control_sign = np.ones(3) if rotation_control_sign is None \
            else np.asarray(rotation_control_sign, dtype=float)
        if (not np.isfinite(self.stale_timeout)
                or self.stale_timeout <= 0
                or self.workspace_min.shape != (3,) or self.workspace_max.shape != (3,)
                or np.any(self.workspace_min >= self.workspace_max)
                or not np.isfinite(self.orientation_scale) or self.orientation_scale <= 0.0
                or self.rotation_control_sign.shape != (3,)
                or not np.isfinite(self.rotation_control_sign).all()
                or not np.all(np.isin(self.rotation_control_sign, (-1.0, 1.0)))):
            raise ValueError("invalid mapper configuration")
        self.clutch = Clutch()
        self.master_anchor: Optional[np.ndarray] = None
        self.slave_anchor: Optional[np.ndarray] = None
        self.target: Optional[np.ndarray] = None
        self.master_anchor_rotation: Optional[np.ndarray] = None
        self.slave_anchor_rotation: Optional[np.ndarray] = None
        self.target_rotation: Optional[np.ndarray] = None

    @property
    def anchored(self) -> bool:
        return self.master_anchor is not None and self.slave_anchor is not None

    def activate(
        self,
        master: GeomagicState,
        slave_actual_xyz: Sequence[float],
        now: Optional[float] = None,
        held_target_xyz: Optional[Sequence[float]] = None,
        slave_actual_rotation: Optional[Sequence[Sequence[float]]] = None,
        held_target_rotation: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        self._validate_sample(master, time.monotonic() if now is None else now)
        slave = np.asarray(slave_actual_xyz, dtype=float)
        if slave.shape != (3,) or not np.isfinite(slave).all():
            raise UnsafeSample("invalid FR3 anchor")
        target = slave if held_target_xyz is None else np.asarray(held_target_xyz, dtype=float)
        if (target.shape != (3,) or not np.isfinite(target).all()
                or np.any(target < self.workspace_min) or np.any(target > self.workspace_max)):
            raise UnsafeSample("invalid or out-of-workspace held target")
        self.master_anchor = np.asarray(master.position_xyz, dtype=float)
        self.slave_anchor = target.copy()
        self.target = target.copy()
        if self.orientation_enabled:
            if not master.orientation_is_usable():
                raise UnsafeSample("invalid master orientation anchor")
            if slave_actual_rotation is None:
                raise UnsafeSample("missing FR3 orientation anchor")
            actual_rotation = validate_rotation_matrix(slave_actual_rotation)
            target_rotation = actual_rotation if held_target_rotation is None \
                else validate_rotation_matrix(held_target_rotation)
            self.master_anchor_rotation = quaternion_xyzw_to_matrix(
                master.orientation_xyzw)  # type: ignore[arg-type]
            self.slave_anchor_rotation = target_rotation.copy()
            self.target_rotation = target_rotation.copy()
        else:
            self.master_anchor_rotation = None
            self.slave_anchor_rotation = None
            self.target_rotation = None
        self.clutch = Clutch()

    def deactivate(self) -> None:
        self.master_anchor = self.slave_anchor = self.target = None
        self.master_anchor_rotation = None
        self.slave_anchor_rotation = None
        self.target_rotation = None

    def _validate_sample(self, sample: GeomagicState, now: float) -> float:
        age = now - sample.timestamp
        if not sample.valid or not sample.pose_is_finite():
            raise UnsafeSample("invalid master pose")
        if self.orientation_enabled and not sample.orientation_is_usable():
            raise UnsafeSample("invalid master orientation")
        if not np.isfinite(age) or age < -0.01 or age > self.stale_timeout:
            raise UnsafeSample(f"stale master sample ({age:.4f} s)")
        return max(age, 0.0)

    def update(
        self,
        master: GeomagicState,
        slave_actual_xyz: Sequence[float],
        clutch_pressed: bool,
        now: Optional[float] = None,
        slave_actual_rotation: Optional[Sequence[Sequence[float]]] = None,
    ) -> MappingResult:
        now = time.monotonic() if now is None else now
        age = self._validate_sample(master, now)
        if not self.anchored or self.target is None:
            raise UnsafeSample("startup anchor required")
        slave_actual = np.asarray(slave_actual_xyz, dtype=float)
        if slave_actual.shape != (3,) or not np.isfinite(slave_actual).all():
            raise UnsafeSample("invalid FR3 pose")
        if self.orientation_enabled:
            if slave_actual_rotation is None:
                raise UnsafeSample("missing FR3 orientation")
            validate_rotation_matrix(slave_actual_rotation)
        event = self.clutch.update(clutch_pressed)
        if event.held:
            return self._result(age, np.zeros(3), clutched=True)
        if event.released:
            self.master_anchor = np.asarray(master.position_xyz, dtype=float)
            # Keep the commanded target exactly continuous on release. The
            # controller continues driving actual pose toward this held target.
            self.slave_anchor = self.target.copy()
            if self.orientation_enabled:
                self.master_anchor_rotation = quaternion_xyzw_to_matrix(
                    master.orientation_xyzw)  # type: ignore[arg-type]
                self.slave_anchor_rotation = self.target_rotation.copy()  # type: ignore[union-attr]
            return self._result(age, np.zeros(3), clutched=False, reanchored=True)

        mapped = np.zeros(3)
        workspace_clamped = False
        if self.position_enabled:
            master_delta = np.asarray(master.position_xyz) - self.master_anchor
            # No per-sample step or anchor-relative excursion clamp: the target
            # is the scaled, frame-mapped relative device translation.
            mapped = scale_delta(self.frame_map.map_delta(master_delta), self.scale)
            desired_unclamped = self.slave_anchor + mapped
            # Optional absolute workspace box (unbounded unless configured).
            desired = np.clip(desired_unclamped, self.workspace_min, self.workspace_max)
            workspace_clamped = bool(not np.allclose(desired, desired_unclamped))
            self.target = desired
            if workspace_clamped:
                # Rebase at the boundary so outward master travel cannot accumulate
                # into a hidden displacement/dead zone.
                self.master_anchor = np.asarray(master.position_xyz, dtype=float)
                self.slave_anchor = desired.copy()

        mapped_rotation_vector = np.zeros(3)
        if self.orientation_enabled:
            assert self.master_anchor_rotation is not None
            assert self.slave_anchor_rotation is not None
            assert self.target_rotation is not None
            current_master_rotation = quaternion_xyzw_to_matrix(
                master.orientation_xyzw)  # type: ignore[arg-type]
            # Spatial delta in device-base coordinates: R_current = delta * R_anchor.
            device_delta = current_master_rotation @ self.master_anchor_rotation.T
            mapped_delta = self.frame_map.map_rotation_delta(device_delta)
            mapped_rotation_vector = (matrix_to_rotation_vector(mapped_delta)
                                      * self.orientation_scale
                                      * self.rotation_control_sign)
            scaled_delta = rotation_vector_to_matrix(mapped_rotation_vector)
            # No per-sample or anchor-relative angular clamp. The mapped delta
            # is expressed in FR3 world, so it left-multiplies the held slave
            # anchor. Right multiplication would be a local-frame delta.
            self.target_rotation = orthonormalize_rotation_matrix(
                scaled_delta @ self.slave_anchor_rotation)
        # step/orientation limit flags remain in MappingResult for interface
        # compatibility and are always False now.
        return self._result(
            age, mapped, clutched=False, workspace_clamped=workspace_clamped,
            mapped_rotation_vector=mapped_rotation_vector)

    def _result(
        self,
        age: float,
        mapped_delta: np.ndarray,
        clutched: bool,
        reanchored: bool = False,
        workspace_clamped: bool = False,
        step_limited: bool = False,
        mapped_rotation_vector: Optional[np.ndarray] = None,
        orientation_excursion_limited: bool = False,
        orientation_step_limited: bool = False,
    ) -> MappingResult:
        return MappingResult(
            target_xyz=self.target.copy(),  # type: ignore[union-attr]
            mapped_delta_xyz=np.asarray(mapped_delta, dtype=float).copy(),
            master_anchor_xyz=self.master_anchor.copy(),  # type: ignore[union-attr]
            slave_anchor_xyz=self.slave_anchor.copy(),  # type: ignore[union-attr]
            target_rotation=None if self.target_rotation is None else self.target_rotation.copy(),
            mapped_rotation_vector=np.zeros(3) if mapped_rotation_vector is None
            else np.asarray(mapped_rotation_vector, dtype=float).copy(),
            master_anchor_rotation=None if self.master_anchor_rotation is None
            else self.master_anchor_rotation.copy(),
            slave_anchor_rotation=None if self.slave_anchor_rotation is None
            else self.slave_anchor_rotation.copy(),
            sample_age_s=age,
            clutched=clutched,
            reanchored=reanchored,
            workspace_clamped=workspace_clamped,
            step_limited=step_limited,
            orientation_excursion_limited=orientation_excursion_limited,
            orientation_step_limited=orientation_step_limited,
        )
