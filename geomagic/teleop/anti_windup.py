"""Geomagic-only follower leash, after shared DLS/rate clipping, before gravity.

Initial engineering values: >1.49x the per-joint observed maxima in high-clearance
physical traces from 2026-10-07, versus 1.62--2.04 rad blocked backlogs. See
reports/follower_anti_windup_20261008 for provenance and replay validation.
"""
from dataclasses import dataclass

import numpy as np


SOFT_LEASH_RAD = (0.06, 0.06, 0.06, 0.08, 0.12, 0.09, 0.11)
HARD_LEASH_RAD = (0.09, 0.09, 0.09, 0.12, 0.18, 0.135, 0.165)


@dataclass(frozen=True)
class AntiWindupConfig:
    soft_rad: tuple = SOFT_LEASH_RAD
    hard_rad: tuple = HARD_LEASH_RAD
    enabled: bool = True

    def __post_init__(self):
        soft, hard = np.asarray(self.soft_rad), np.asarray(self.hard_rad)
        if (type(self.enabled) is not bool or soft.shape != (7,) or hard.shape != (7,)
                or not np.isfinite(soft).all() or not np.isfinite(hard).all()
                or np.any(soft <= 0) or np.any(hard <= soft)):
            raise ValueError('anti-windup requires seven finite 0 < soft < hard limits and bool enabled')


class JointCommandAntiWindup:
    """Limit unrealized logical commands; never gate on contact/haptic state.

    The helper candidate already includes joint-rate and joint-position limits.
    Attenuate only increments sharing the sign of the existing logical error.
    Return the exact original candidate when inactive, preserving free-space
    numerical behavior. Hard projection also catches movement of actual q away
    from a held command. Transition synchronization remains the caller's job.
    """
    def __init__(self, config=None):
        self.config = config if config is not None else AntiWindupConfig()
        self.soft = np.asarray(self.config.soft_rad, dtype=float)
        self.hard = np.asarray(self.config.hard_rad, dtype=float)
        self.reset()

    def reset(self):
        self.backlog_increasing_block_count = 0
        self.hard_leash_hit_count = 0
        self.synchronize()

    def synchronize(self):
        self.tracking_error = np.zeros(7)
        self.joint_mask = np.zeros(7, dtype=bool)
        self.hard_mask = np.zeros(7, dtype=bool)
        self.active = False

    def apply(self, previous, actual, candidate, lower, upper):
        error = previous - actual
        step = candidate - previous
        self.joint_mask = (error * step > 0) & (np.abs(error) > self.soft)
        if not self.config.enabled:
            self.synchronize()
            self.tracking_error = candidate - actual
            return candidate
        if np.any(self.joint_mask):
            gain = np.clip((self.hard - np.abs(error)) / (self.hard - self.soft), 0., 1.)
            # Unwinding components have gain 1, even outside the hard region.
            adjusted = previous + np.where(self.joint_mask, gain * step, step)
            self.backlog_increasing_block_count += int(np.count_nonzero(self.joint_mask))
        else:
            adjusted = candidate
        low = np.maximum(lower, actual - self.hard)
        high = np.minimum(upper, actual + self.hard)
        if np.any(low > high):
            raise ValueError('actual q is too far outside joint limits to satisfy the hard leash')
        self.hard_mask = (adjusted < low) | (adjusted > high)
        if np.any(self.hard_mask):
            adjusted = np.clip(adjusted, low, high)
            self.hard_leash_hit_count += int(np.count_nonzero(self.hard_mask))
        self.active = bool(np.any(self.joint_mask) or np.any(self.hard_mask))
        self.tracking_error = adjusted - actual
        return adjusted

    def diagnostics(self):
        """Only materialize lists when existing low-rate diagnostics request them."""
        return dict(tracking_error_per_joint=self.tracking_error.tolist(),
                    max_tracking_error=float(np.max(np.abs(self.tracking_error))),
                    anti_windup_active=self.active,
                    anti_windup_joint_mask=self.joint_mask.tolist(),
                    backlog_increasing_block_count=self.backlog_increasing_block_count,
                    hard_leash_hit_count=self.hard_leash_hit_count)
