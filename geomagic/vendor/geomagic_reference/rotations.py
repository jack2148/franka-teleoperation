"""Explicit rotation conversions used at ROS, mapping, and MuJoCo boundaries."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def normalize_quaternion_xyzw(quaternion: Sequence[float]) -> np.ndarray:
    """Return a finite unit quaternion in explicit ROS/SciPy ``xyzw`` order."""
    value = np.asarray(quaternion, dtype=float)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError("quaternion_xyzw must be a finite 4-vector")
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        raise ValueError("quaternion norm is zero")
    return value / norm


def quaternion_xyzw_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """Convert ROS/SciPy scalar-last ``xyzw`` to an active 3x3 rotation."""
    x, y, z, w = normalize_quaternion_xyzw(quaternion)
    return np.array([
        [1.0 - 2.0 * (y*y + z*z), 2.0 * (x*y - z*w), 2.0 * (x*z + y*w)],
        [2.0 * (x*y + z*w), 1.0 - 2.0 * (x*x + z*z), 2.0 * (y*z - x*w)],
        [2.0 * (x*z - y*w), 2.0 * (y*z + x*w), 1.0 - 2.0 * (x*x + y*y)],
    ])


def validate_rotation_matrix(rotation: Sequence[Sequence[float]]) -> np.ndarray:
    value = np.asarray(rotation, dtype=float)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    if not np.allclose(value @ value.T, np.eye(3), atol=1e-7):
        raise ValueError("rotation matrix must be orthonormal")
    if not np.isclose(np.linalg.det(value), 1.0, atol=1e-7):
        raise ValueError("rotation matrix determinant must be +1")
    return value


def orthonormalize_rotation_matrix(rotation: Sequence[Sequence[float]]) -> np.ndarray:
    """Project a finite 3x3 matrix onto SO(3) to remove numeric drift."""
    value = np.asarray(rotation, dtype=float)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    left, _, right_t = np.linalg.svd(value)
    projected = left @ right_t
    if np.linalg.det(projected) < 0.0:
        left[:, -1] *= -1.0
        projected = left @ right_t
    return validate_rotation_matrix(projected)


def matrix_to_quaternion_xyzw(rotation: Sequence[Sequence[float]]) -> np.ndarray:
    """Convert an active rotation matrix to canonical scalar-last ``xyzw``."""
    matrix = validate_rotation_matrix(rotation)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale
    value = normalize_quaternion_xyzw((x, y, z, w))
    # q and -q are equivalent. Canonicalizing only the serialized form keeps
    # logs stable; all control and comparisons use matrices/relative angles.
    return -value if value[3] < 0.0 else value


def rotation_vector_to_matrix(rotation_vector: Sequence[float]) -> np.ndarray:
    vector = np.asarray(rotation_vector, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("rotation_vector must be a finite 3-vector")
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    axis = vector / angle
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def matrix_to_rotation_vector(rotation: Sequence[Sequence[float]]) -> np.ndarray:
    """Return the shortest axis-angle vector, robust through the 180 deg limit."""
    x, y, z, w = matrix_to_quaternion_xyzw(rotation)
    vector = np.array((x, y, z), dtype=float)
    sine_half = float(np.linalg.norm(vector))
    if sine_half < 1e-12:
        return np.zeros(3)
    angle = 2.0 * math.atan2(sine_half, max(0.0, float(w)))
    return vector * (angle / sine_half)


def rotation_distance_rad(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> float:
    """Geodesic distance; invariant to the serialized quaternion sign."""
    a = validate_rotation_matrix(left)
    b = validate_rotation_matrix(right)
    return float(np.linalg.norm(matrix_to_rotation_vector(a @ b.T)))


def limited_rotation(rotation: Sequence[Sequence[float]], max_angle_rad: float) -> tuple[np.ndarray, bool]:
    if not math.isfinite(max_angle_rad) or max_angle_rad <= 0.0:
        raise ValueError("max_angle_rad must be positive")
    vector = matrix_to_rotation_vector(rotation)
    angle = float(np.linalg.norm(vector))
    if angle <= max_angle_rad:
        return validate_rotation_matrix(rotation).copy(), False
    return rotation_vector_to_matrix(vector * (max_angle_rad / angle)), True
