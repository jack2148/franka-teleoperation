"""Boundary validation and continuous anchor-relative SO(3) logarithm.

The log selects the equivalent rotation vector nearest the preceding input log.
This is branch selection, never integration of device velocity/target increments.
Identical input retains an identical finite target; all state resets on reanchor.
"""
import math
import numpy as np

_IDENTITY = np.eye(3)
_ORTH_TOL = np.full((3, 3), 1e-7) + _IDENTITY * 1e-5


def validate_rotation_matrix(rotation):
    """Same finite/orthogonality/determinant checks without generic isclose."""
    r = np.asarray(rotation, dtype=float)
    if r.shape != (3, 3) or not np.isfinite(r).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    if np.any(np.abs(r @ r.T - _IDENTITY) > _ORTH_TOL):
        raise ValueError("rotation matrix must be orthonormal")
    if abs(float(np.linalg.det(r)) - 1.) > 1.01e-5:
        raise ValueError("rotation matrix determinant must be +1")
    return r


def relative_quaternion(current, anchor):
    """Normalized XYZW quaternion of current @ anchor.T, checked once/sample."""
    q = np.asarray(current, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("device quaternion must be finite XYZW")
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("device quaternion norm is zero")
    x, y, z, w = q / norm
    a, b, c, d = anchor
    return np.array((-w*a+x*d-y*c+z*b, -w*b+x*c+y*d-z*a,
                     -w*c-x*b+y*a+z*d, w*d+x*a+y*b+z*c))


def continuous_rotvec(quaternion, previous):
    q = np.asarray(quaternion)
    if q[3] < 0:
        q = -q
    size = float(np.linalg.norm(q[:3]))
    if size < 1e-12:
        length = float(np.linalg.norm(previous))
        if length < 1e-12:
            return np.zeros(3)
        # Identity at a full turn retains the established branch/axis.
        return previous / length * (2 * math.pi * round(length / (2 * math.pi)))
    axis = q[:3] / size
    angle = 2 * math.atan2(size, max(0., float(q[3])))
    turns = round((float(axis @ previous) - angle) / (2 * math.pi))
    return axis * (angle + 2 * math.pi * turns)
