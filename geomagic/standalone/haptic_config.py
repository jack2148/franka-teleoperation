"""Engineering defaults for Geomagic contact-cue release, in device units."""
from dataclasses import dataclass
import math


RETREAT_SPEED_THRESHOLD = 0.005  # m/s; deliberately above tiny hand tremor
RETREAT_CONFIRM_MS = 15.0


@dataclass(frozen=True)
class RetreatConfig:
    speed_threshold_mps: float = RETREAT_SPEED_THRESHOLD
    confirm_ms: float = RETREAT_CONFIRM_MS

    def __post_init__(self):
        for value in (self.speed_threshold_mps, self.confirm_ms):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError('retreat threshold and confirmation must be finite and positive')
