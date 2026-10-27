"""Workspace scaling kept separate from frame remapping."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def scale_delta(delta: Sequence[float], scale: Sequence[float]) -> np.ndarray:
    delta_array = np.asarray(delta, dtype=float)
    scale_array = np.asarray(scale, dtype=float)
    if delta_array.shape != (3,) or scale_array.shape != (3,):
        raise ValueError("delta and scale must each contain three values")
    if not np.isfinite(scale_array).all() or np.any(scale_array <= 0.0):
        raise ValueError("position_scale values must be finite and positive")
    return delta_array * scale_array

