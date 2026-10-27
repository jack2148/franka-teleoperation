"""The only layer allowed to perform device-to-FR3 axis remapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .rotations import validate_rotation_matrix


@dataclass(frozen=True)
class FrameMap:
    master_frame: str
    slave_frame: str
    position_matrix: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.position_matrix, dtype=float)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("position_axis_map must be a finite 3x3 matrix")
        if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-8):
            raise ValueError("position_axis_map must be orthonormal")
        if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-8):
            raise ValueError("position_axis_map must be a proper rotation (det = +1)")
        object.__setattr__(self, "position_matrix", matrix)

    @classmethod
    def from_config(cls, config: dict) -> "FrameMap":
        matrix = config.get("position_axis_map")
        semantic = config.get("device_to_fr3")
        if matrix is not None and semantic is not None:
            raise ValueError("configure only one of position_axis_map or device_to_fr3")
        if semantic is not None:
            if not isinstance(semantic, dict):
                raise ValueError("device_to_fr3 must be a mapping")
            source_index = {"x": 0, "y": 1, "z": 2}
            rows = []
            for output_axis in ("x", "y", "z"):
                entry = semantic.get(output_axis)
                if not isinstance(entry, dict):
                    raise ValueError(f"device_to_fr3.{output_axis} is required")
                source = str(entry.get("source_axis") or "").lower()
                sign = float(entry.get("sign", 0.0))
                if source not in source_index or sign not in (-1.0, 1.0):
                    raise ValueError(
                        f"device_to_fr3.{output_axis} needs source_axis x/y/z and sign +/-1")
                row = [0.0, 0.0, 0.0]
                row[source_index[source]] = sign
                rows.append(row)
            matrix = rows
        if matrix is None:
            raise ValueError("mapping calibration required: device_to_fr3 is unset")
        return cls(
            master_frame=str(config.get("master_frame") or ""),
            slave_frame=str(config.get("slave_frame") or ""),
            position_matrix=np.asarray(matrix, dtype=float),
        )

    def map_delta(self, master_delta: Sequence[float]) -> np.ndarray:
        delta = np.asarray(master_delta, dtype=float)
        if delta.shape != (3,):
            raise ValueError("master delta must have shape (3,)")
        return self.position_matrix @ delta

    @property
    def rotation_frame_matrix(self) -> np.ndarray:
        """Proper Geomagic-base to FR3-world orientation-frame rotation."""
        return self.position_matrix.copy()

    def diagnostics(self) -> dict[str, object]:
        """Return startup-safe pose-frame matrix checks for console/log output."""
        position = self.position_matrix
        rotation = self.rotation_frame_matrix
        return {
            "M_position": position.tolist(),
            "det_M_position": float(np.linalg.det(position)),
            "M_rotation_frame": rotation.tolist(),
            "det_M_rotation_frame": float(np.linalg.det(rotation)),
            "orthonormal": bool(np.allclose(position.T @ position, np.eye(3), atol=1e-8)),
        }

    def map_force_to_device(self, force_fr3_world: Sequence[float]) -> np.ndarray:
        """Map an FR3-world force into the native HDAPI device force frame.

        Force is a polar vector, exactly like the position delta, so it uses the
        same ``position_matrix`` and simply travels in the opposite direction:
        ``position_matrix`` takes device -> FR3, therefore its inverse takes
        FR3 -> device. The matrix is orthonormal, so the inverse is the
        transpose. Rotation uses relative-rotation conjugation because it is
        an orientation operation; it is not a force-mapping sign rule.

        The pose mapping was corrected after the original haptic validation.
        This method is deliberately unchanged, but its physical force-feedback
        convention must be recalculated and revalidated before reflection is
        enabled for the corrected pose map.

        No extra sign is applied here. The caller passes the force acting *on
        the robot*; the operator must feel that same physical direction, so a
        wall opposing FR3 +X motion yields device -Z, i.e. physically backward.
        """
        force = np.asarray(force_fr3_world, dtype=float)
        if force.shape != (3,) or not np.isfinite(force).all():
            raise ValueError("force_fr3_world must be a finite 3-vector")
        return self.position_matrix.T @ force

    def map_rotation_delta(self, master_delta_rotation: Sequence[Sequence[float]]) -> np.ndarray:
        """Change a device-base relative rotation into the FR3 world basis.

        The configured frame transform is a proper rotation.  Conjugation
        applies it to a relative rotation matrix without mapping absolute
        orientation: ``R_fr3_rel = M R_geo_rel M.T``.  Its small-angle
        spatial rotation-vector equivalent is ``omega_fr3 = M omega_geo``.
        """
        rotation = validate_rotation_matrix(master_delta_rotation)
        mapping = self.rotation_frame_matrix
        mapped = mapping @ rotation @ mapping.T
        return validate_rotation_matrix(mapped)
