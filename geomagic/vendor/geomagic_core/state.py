"""Pure Python Geomagic state, adapted from the read-only source snapshot.

Provenance: geomagic_franka_teleop_import_v2/src/device/geomagic_state.py.
Retains metre positions, XYZW orientation, (grey, white) buttons, validity,
sequence, and local monotonic receive timestamp. Network provenance fields and
button accessors are additive. No device, ROS, force, or robot interface is copied.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple


Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]
QUATERNION_NORM_TOLERANCE = 0.05


@dataclass(frozen=True)
class GeomagicState:
    """A snapshot; timestamp is SERVER receive time, never source clock time.

    The receiver returns a copy with valid=False when stale. source_valid
    preserves the transmitted validity bit even for such a stale copy.
    """

    position_xyz: Vector3
    orientation_xyzw: Optional[Quaternion]
    buttons: Tuple[bool, ...]
    timestamp: float
    valid: bool
    frame_id: str = "geomagic_base"
    units: str = "m"
    sequence: int = 0
    session_id: str = ""
    source_timestamp: Optional[float] = None
    source_clock_domain: str = ""
    source_valid: bool = True

    @property
    def grey_button(self) -> bool:
        return bool(self.buttons[0]) if self.buttons else False

    @property
    def white_button(self) -> bool:
        return bool(self.buttons[1]) if len(self.buttons) > 1 else False

    @property
    def server_receive_monotonic_time(self) -> float:
        return self.timestamp

    def pose_is_finite(self) -> bool:
        values = self.position_xyz
        if self.orientation_xyzw is not None:
            values += self.orientation_xyzw
        return all(math.isfinite(value) for value in values)

    def orientation_is_usable(
        self, tolerance: float = QUATERNION_NORM_TOLERANCE
    ) -> bool:
        if self.orientation_xyzw is None:
            return False
        norm = math.hypot(*self.orientation_xyzw)
        return math.isfinite(norm) and abs(norm - 1.0) <= tolerance
