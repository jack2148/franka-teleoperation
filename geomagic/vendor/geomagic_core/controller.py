# Vendored from geomagic_franka_teleop_import_v2/src/teleop/controller.py
# Changes: state import namespace; frames omits the unused force method.
"""Deadman, clutch, freshness, and re-anchoring state machine."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence

import numpy as np

from .state import GeomagicState
from .pose_mapping import AnchoredPoseMapper, MappingResult, UnsafeSample
from .rotations import validate_rotation_matrix


@dataclass(frozen=True)
class TeleopResult:
    target_xyz: np.ndarray
    mapped_delta_xyz: np.ndarray
    target_rotation: Optional[np.ndarray]
    mapped_rotation_vector: np.ndarray
    master_anchor_xyz: Optional[np.ndarray]
    slave_anchor_xyz: Optional[np.ndarray]
    master_anchor_rotation: Optional[np.ndarray]
    slave_anchor_rotation: Optional[np.ndarray]
    sample_age_s: float
    teleop_active: bool
    deadman_pressed: bool
    clutch_pressed: bool
    reanchored: bool
    workspace_clamped: bool
    step_limited: bool
    orientation_excursion_limited: bool
    orientation_step_limited: bool
    stale_event: bool
    invalid_state: bool
    reason: str


class PositionTeleopController:
    """Hold a Cartesian target unless all position-teleop gates are satisfied."""

    def __init__(
        self,
        mapper: AnchoredPoseMapper,
        initial_target_xyz: Sequence[float],
        deadman_button_index: int,
        clutch_button_index: int,
        activation_min_consecutive_samples: int,
        initial_target_rotation: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        target = np.asarray(initial_target_xyz, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("initial_target_xyz must be a finite 3-vector")
        if deadman_button_index < 0 or clutch_button_index < 0:
            raise ValueError("button indexes must be non-negative")
        if deadman_button_index == clutch_button_index:
            raise ValueError("deadman and clutch must use different buttons")
        if activation_min_consecutive_samples < 2:
            raise ValueError("activation_min_consecutive_samples must be at least 2")
        self.mapper = mapper
        self.target = target.copy()
        if mapper.orientation_enabled:
            if initial_target_rotation is None:
                raise ValueError("initial_target_rotation is required when orientation is enabled")
            self.target_rotation: Optional[np.ndarray] = validate_rotation_matrix(
                initial_target_rotation).copy()
        else:
            self.target_rotation = None
        self.deadman_index = int(deadman_button_index)
        self.clutch_index = int(clutch_button_index)
        self.required_samples = int(activation_min_consecutive_samples)
        self.active = False
        self._fresh_count = 0
        self._last_sequence: Optional[int] = None
        self._last_timestamp: Optional[float] = None
        self._stream_was_fresh = False

    def _button(self, sample: GeomagicState, index: int) -> bool:
        return len(sample.buttons) > index and bool(sample.buttons[index])

    def _held(
        self,
        sample_age_s: float,
        deadman: bool,
        clutch: bool,
        stale_event: bool,
        invalid_state: bool,
        reason: str,
    ) -> TeleopResult:
        return TeleopResult(
            target_xyz=self.target.copy(),
            mapped_delta_xyz=np.zeros(3),
            target_rotation=None if self.target_rotation is None else self.target_rotation.copy(),
            mapped_rotation_vector=np.zeros(3),
            master_anchor_xyz=None if self.mapper.master_anchor is None
            else self.mapper.master_anchor.copy(),
            slave_anchor_xyz=None if self.mapper.slave_anchor is None
            else self.mapper.slave_anchor.copy(),
            master_anchor_rotation=None if self.mapper.master_anchor_rotation is None
            else self.mapper.master_anchor_rotation.copy(),
            slave_anchor_rotation=None if self.mapper.slave_anchor_rotation is None
            else self.mapper.slave_anchor_rotation.copy(),
            sample_age_s=sample_age_s,
            teleop_active=False,
            deadman_pressed=deadman,
            clutch_pressed=clutch,
            reanchored=False,
            workspace_clamped=False,
            step_limited=False,
            orientation_excursion_limited=False,
            orientation_step_limited=False,
            stale_event=stale_event,
            invalid_state=invalid_state,
            reason=reason,
        )

    def update(
        self,
        sample: Optional[GeomagicState],
        slave_actual_xyz: Sequence[float],
        now: float,
        slave_actual_rotation: Optional[Sequence[Sequence[float]]] = None,
    ) -> TeleopResult:
        actual = np.asarray(slave_actual_xyz, dtype=float)
        if actual.shape != (3,) or not np.isfinite(actual).all():
            raise UnsafeSample("invalid FR3 pose")
        if self.mapper.orientation_enabled:
            if slave_actual_rotation is None:
                raise UnsafeSample("missing FR3 orientation")
            validate_rotation_matrix(slave_actual_rotation)

        age = math.inf if sample is None else now - sample.timestamp
        finite_pose = bool(sample is not None and sample.pose_is_finite())
        orientation_valid = bool(
            not self.mapper.orientation_enabled
            or (sample is not None and sample.orientation_is_usable()))
        fresh = bool(
            sample is not None and sample.valid and finite_pose and orientation_valid
            and math.isfinite(age) and -0.01 <= age <= self.mapper.stale_timeout)
        if not fresh:
            stale = bool(sample is not None and math.isfinite(age)
                         and age > self.mapper.stale_timeout)
            stale_event = stale and self._stream_was_fresh
            self._stream_was_fresh = False
            self._fresh_count = 0
            self._last_sequence = None
            self._last_timestamp = None
            self.active = False
            self.mapper.deactivate()
            return self._held(max(age, 0.0), False, False, stale_event,
                              not stale, "stale" if stale else "invalid")

        assert sample is not None
        self._stream_was_fresh = True
        if sample.sequence != self._last_sequence:
            gap_ok = (self._last_timestamp is None
                      or sample.timestamp - self._last_timestamp <= self.mapper.stale_timeout)
            self._fresh_count = self._fresh_count + 1 if gap_ok else 1
            self._last_sequence = sample.sequence
            self._last_timestamp = sample.timestamp

        deadman = self._button(sample, self.deadman_index)
        clutch = self._button(sample, self.clutch_index)
        if not deadman:
            if self.active:
                self.active = False
                self.mapper.deactivate()
            return self._held(max(age, 0.0), False, clutch, False, False, "deadman_released")
        if self._fresh_count < self.required_samples:
            return self._held(max(age, 0.0), True, clutch, False, False, "freshness_gate")

        if not self.active:
            self.mapper.activate(
                sample, actual, now=now, held_target_xyz=self.target,
                slave_actual_rotation=slave_actual_rotation,
                held_target_rotation=self.target_rotation)
            self.active = True
            return TeleopResult(
                target_xyz=self.target.copy(), mapped_delta_xyz=np.zeros(3),
                target_rotation=None if self.target_rotation is None
                else self.target_rotation.copy(), mapped_rotation_vector=np.zeros(3),
                master_anchor_xyz=self.mapper.master_anchor.copy(),
                slave_anchor_xyz=self.mapper.slave_anchor.copy(),
                master_anchor_rotation=None if self.mapper.master_anchor_rotation is None
                else self.mapper.master_anchor_rotation.copy(),
                slave_anchor_rotation=None if self.mapper.slave_anchor_rotation is None
                else self.mapper.slave_anchor_rotation.copy(), sample_age_s=max(age, 0.0),
                teleop_active=True, deadman_pressed=True, clutch_pressed=clutch,
                reanchored=True, workspace_clamped=False, step_limited=False,
                orientation_excursion_limited=False, orientation_step_limited=False,
                stale_event=False, invalid_state=False, reason="activated",
            )

        mapped: MappingResult = self.mapper.update(
            sample, actual, clutch, now=now,
            slave_actual_rotation=slave_actual_rotation)
        self.target = mapped.target_xyz.copy()
        self.target_rotation = None if mapped.target_rotation is None \
            else mapped.target_rotation.copy()
        return TeleopResult(
            target_xyz=self.target.copy(), mapped_delta_xyz=mapped.mapped_delta_xyz.copy(),
            target_rotation=None if self.target_rotation is None else self.target_rotation.copy(),
            mapped_rotation_vector=mapped.mapped_rotation_vector.copy(),
            master_anchor_xyz=mapped.master_anchor_xyz.copy(),
            slave_anchor_xyz=mapped.slave_anchor_xyz.copy(),
            master_anchor_rotation=None if mapped.master_anchor_rotation is None
            else mapped.master_anchor_rotation.copy(),
            slave_anchor_rotation=None if mapped.slave_anchor_rotation is None
            else mapped.slave_anchor_rotation.copy(),
            sample_age_s=mapped.sample_age_s, teleop_active=True,
            deadman_pressed=True, clutch_pressed=clutch,
            reanchored=mapped.reanchored, workspace_clamped=mapped.workspace_clamped,
            step_limited=mapped.step_limited, stale_event=False,
            orientation_excursion_limited=mapped.orientation_excursion_limited,
            orientation_step_limited=mapped.orientation_step_limited,
            invalid_state=False, reason="clutched" if clutch else "active",
        )
