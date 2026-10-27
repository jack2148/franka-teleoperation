"""Pose deadband after the frozen mapper, before Cartesian conditioning.

The mapper stays memoryless. Each moving segment uses fixed input/output
anchors, subtracts an ON-radius in pose space, and ramps the gate continuously.
No input velocity is integrated. Safety reanchors reset both channels together.
"""
from dataclasses import dataclass, asdict, replace
import json
import math
import os
from pathlib import Path

import numpy as np

from geomagic.teleop.pose_mapper import POSITION_SCALE, ROTATION_SCALE, POSITION_MODES, ROTATION_MODES
from geomagic.vendor.geomagic_core.rotations import (
    matrix_to_rotation_vector, rotation_vector_to_matrix,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config/input_deadband.json"


@dataclass(frozen=True)
class DeadbandConfig:
    enabled: bool = False
    translation_on_m: float = .0002  # device displacement, before mapping gain
    translation_off_m: float = .0001
    rotation_on_rad: float = math.radians(.5)
    rotation_off_rad: float = math.radians(.25)
    dwell_s: float = .030
    quiet_dwell_s: float = .050
    engage_ramp_s: float = .030
    tuning_status: str = "TUNING_PENDING"

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("deadband enabled must be boolean")
        for on, off in ((self.translation_on_m, self.translation_off_m),
                        (self.rotation_on_rad, self.rotation_off_rad)):
            if not all(math.isfinite(x) for x in (on, off)) or not on > off > 0:
                raise ValueError("deadband requires finite ON > OFF > 0")
        for x in (self.dwell_s, self.quiet_dwell_s, self.engage_ramp_s):
            if not math.isfinite(x) or x <= 0:
                raise ValueError("deadband timing must be finite and positive")
        if self.quiet_dwell_s < self.engage_ramp_s:
            raise ValueError("quiet dwell must allow the engagement ramp to finish")

    @classmethod
    def from_environment(cls):
        path = Path(os.environ.get("MIMIC_GEOMAGIC_DEADBAND_CONFIG", CONFIG_PATH))
        values = json.loads(path.read_text())
        enabled = os.environ.get("MIMIC_GEOMAGIC_DEADBAND")
        if enabled is not None:
            if enabled not in ("0", "1"):
                raise ValueError("MIMIC_GEOMAGIC_DEADBAND must be 0 or 1")
            values["enabled"] = enabled == "1"
        return cls(**values)


class _PoseChannel:
    def __init__(self, on, off, config, *, rotation=False):
        self.on, self.off, self.config, self.rotation = on, off, config, rotation
        self.activations = self.suppressed = 0
        self.value = None
        self.active = False

    def reset(self, raw):
        self.value = raw.copy()
        self.input_anchor = raw.copy()
        self.output_anchor = raw.copy()
        self.intent = raw.copy()
        self.active = False
        self.candidate_since = self.engaged_at = self.last_motion = None
        self.last_time = None

    def delta(self, a, b):
        return matrix_to_rotation_vector(a @ b.T) if self.rotation else a - b

    def compose(self, vector, anchor):
        return rotation_vector_to_matrix(vector) @ anchor if self.rotation else anchor + vector

    def update(self, raw, now):
        if not math.isfinite(now):
            raise ValueError("deadband timestamp must be finite")
        if self.value is None:
            self.reset(raw)
        if self.last_time is not None and now < self.last_time:
            raise ValueError("deadband timestamp moved backwards without reanchor")
        self.last_time = now
        if not self.active:
            excursion = float(np.linalg.norm(self.delta(raw, self.input_anchor)))
            if excursion <= self.off:
                self.candidate_since = None
            elif excursion >= self.on and self.candidate_since is None:
                self.candidate_since = now
            if (self.candidate_since is not None
                    and now - self.candidate_since >= self.config.dwell_s - 1e-12):
                self.active = True
                self.activations += 1
                self.engaged_at = self.last_motion = now
                vector = self.delta(raw, self.input_anchor)
                size = float(np.linalg.norm(vector))
                self.intent = self.compose(vector * (max(0., size - self.off) / size),
                                           self.input_anchor)
            else:
                self.suppressed += 1
                return self.value.copy()
        else:
            vector = self.delta(raw, self.intent)
            size = float(np.linalg.norm(vector))
            if size > self.off + 1e-12:
                # Continuous play operator, not quantized accepted poses:
                # move to the nearest pose within OFF of the current input.
                # Even slow sustained motion advances without rate-dependent
                # staircase steps; every value remains a pose, not a speed.
                self.intent = self.compose(vector * ((size - self.off) / size), self.intent)
                self.last_motion = now
            else:
                self.suppressed += 1

        # Continuous radial boundary subtraction, anchored once per segment.
        # The gate ramp accounts for displacement accrued during the dwell,
        # so crossing a threshold cannot expose a hidden pose jump.
        vector = self.delta(self.intent, self.input_anchor)
        size = float(np.linalg.norm(vector))
        residual = vector * (max(0., size - self.on) / size) if size else vector
        u = min(1., max(0., (now - self.engaged_at) / self.config.engage_ramp_s))
        alpha = u * u * (3. - 2. * u)
        self.value = self.compose(alpha * residual, self.output_anchor)
        if now - self.last_motion >= self.config.quiet_dwell_s - 1e-12:
            # Retain the final intentional pose; never reanchor to lagging
            # actual state on a noise-gate transition (only safety edges do).
            self.active = False
            self.input_anchor = self.intent.copy()
            self.output_anchor = self.value.copy()
            self.candidate_since = None
        return self.value.copy()

    def status(self):
        return {"state": "ACTIVE" if self.active else "INACTIVE",
                "activation_count": self.activations,
                "suppressed_samples": self.suppressed}


class PoseInputDeadband:
    def __init__(self, config=None):
        self.config = config or DeadbandConfig()
        # Convert device-space thresholds to mapped pose units with the same
        # gains as the mapper; the frozen translation map is orthonormal.
        self.translation = _PoseChannel(POSITION_SCALE[0] * self.config.translation_on_m,
                                        POSITION_SCALE[0] * self.config.translation_off_m, self.config)
        self.rotation = _PoseChannel(ROTATION_SCALE * self.config.rotation_on_rad,
                                     ROTATION_SCALE * self.config.rotation_off_rad,
                                     self.config, rotation=True)

    def reset(self, position, rotation):
        self.translation.reset(np.asarray(position))
        self.rotation.reset(np.asarray(rotation))

    def update(self, target, *, now):
        if (not self.config.enabled or target.synchronized or target.reanchored
                or not target.fresh or not target.active or target.clutched):
            self.reset(target.position, target.rotation)
            return target
        # Count/filter accepted device samples only, never held control ticks.
        if not target.mapper_updated:
            return replace(target, position=self.translation.value.copy(),
                           rotation=self.rotation.value.copy())
        # Independent gates: a quiet channel never suppresses the other axis set.
        position = (self.translation.update(target.position, now) if target.mode in POSITION_MODES
                    else self.translation.value.copy())
        rotation = (self.rotation.update(target.rotation, now) if target.mode in ROTATION_MODES
                    else self.rotation.value.copy())
        return replace(target, position=position, rotation=rotation)

    def status(self):
        return {"config": asdict(self.config), "translation": self.translation.status(),
                "rotation": self.rotation.status()}
