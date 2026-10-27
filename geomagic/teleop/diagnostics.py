"""Opt-in, bounded tracking statistics, separate from collection samples.

Timings cover every physics tick, limiter flags every DLS update, and detailed
pose/residual metrics approximately 100 Hz simulation time. p95 uses a bounded
4096-value window. No file writes occur on the control tick.
"""

from collections import defaultdict, deque
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from geomagic.vendor.geomagic_core.rotations import quaternion_xyzw_to_matrix


def angular_change_rad(a, b):
    relative = a @ b.T
    sine = .5 * np.linalg.norm((relative[2, 1] - relative[1, 2],
                               relative[0, 2] - relative[2, 0],
                               relative[1, 0] - relative[0, 1]))
    cosine = .5 * (np.trace(relative) - 1.)
    return float(math.atan2(sine, np.clip(cosine, -1., 1.)))


class RotationCommandDiagnostics:
    """Latest control-update derivatives, separate from learning data.

    Device derivative uses source HDAPI time when available, otherwise receive
    time. Target/actual derivatives use elapsed simulation time. Reanchor
    discontinuities are excluded and marked with zero derivatives.
    """
    def __init__(self):
        self.previous = self.device_previous = self.anchor = None
        self.sequence = self.session = None
        self.values = dict.fromkeys(("device_relative_rotation_rad", "device_angular_velocity_rad_s",
                                     "raw_target_angular_velocity_rad_s",
                                     "deadband_target_angular_velocity_rad_s",
                                     "conditioned_target_angular_velocity_rad_s",
                                     "actual_angular_velocity_rad_s"), 0.)

    def update(self, pipeline, sample, sim_time):
        raw, filtered, c = pipeline.last_raw_target, pipeline.last_target, pipeline.controller
        _, actual = c.actual_pose()
        current = (sim_time, raw.rotation.copy(), filtered.rotation.copy(),
                   c.target_rotation.copy(), actual.copy())
        edge = raw.synchronized or raw.reanchored
        if self.previous is not None and not edge:
            elapsed = sim_time - self.previous[0]
            if elapsed > 0:
                for key, r, before in zip(("raw_target_angular_velocity_rad_s",
                                           "deadband_target_angular_velocity_rad_s",
                                           "conditioned_target_angular_velocity_rad_s",
                                           "actual_angular_velocity_rad_s"),
                                          current[1:], self.previous[1:]):
                    self.values[key] = angular_change_rad(r, before) / elapsed
        else:
            for key in self.values:
                self.values[key] = 0.
        self.previous = current
        if sample is not None and sample.orientation_is_usable():
            if (sample.session_id, sample.sequence) != (self.session, self.sequence) or edge:
                device = quaternion_xyzw_to_matrix(sample.orientation_xyzw)
                t = (sample.source_timestamp if sample.source_clock_domain == "server.monotonic.hdapi"
                     and sample.source_timestamp is not None else sample.timestamp)
                if edge or self.anchor is None or sample.session_id != self.session:
                    self.anchor = device.copy()
                    self.device_previous = None
                    self.values["device_angular_velocity_rad_s"] = 0.
                self.values["device_relative_rotation_rad"] = angular_change_rad(device, self.anchor)
                if self.device_previous is not None and t > self.device_previous[0]:
                    self.values["device_angular_velocity_rad_s"] = (
                        angular_change_rad(device, self.device_previous[1]) / (t - self.device_previous[0]))
                self.device_previous = (t, device)
                self.session, self.sequence = sample.session_id, sample.sequence
        return dict(self.values)


def angular_error_deg(a, b):
    # Validated mapper/conditioner and MuJoCo rotations: only the angle is
    # needed here. trace(a @ b.T) == sum(a * b), without another SO(3) check.
    cosine = (float(np.sum(a * b)) - 1.) / 2.
    return math.degrees(math.acos(max(-1., min(1., cosine))))


class TrackingDiagnostics:
    def __init__(self, *, capacity=4096, report_path=None, clock=time.perf_counter, lightweight=False):
        self.lightweight = lightweight
        self.capacity, self.report_path, self.clock = capacity, report_path, clock
        self.values = {}
        self.counts = defaultdict(int)
        self.previous_start = None
        self.previous_finish = None
        self.cached = {}
        self.last_publish = -math.inf
        self.anti_windup = {}
        self.rotation_command = RotationCommandDiagnostics()

    @classmethod
    def from_environment(cls):
        mode = os.environ.get('MIMIC_GEOMAGIC_DIAGNOSTICS_MODE')
        if mode not in (None, 'off', 'production', 'detailed'):
            raise ValueError('diagnostics mode must be off, production or detailed')
        if mode == 'off' or (mode is None and os.environ.get("MIMIC_GEOMAGIC_DIAGNOSTICS") != "1"):
            return None
        return cls(report_path=os.environ.get("MIMIC_GEOMAGIC_DIAGNOSTICS_REPORT"),
                   lightweight=mode == 'production')

    def begin(self, dt):
        self.start = self.clock()
        if self.previous_finish is not None:
            previous_group = "active" if self.active else "inactive"
            self.add(previous_group, "between_ticks_ms", (self.start - self.previous_finish) * 1000)
        self.wall_dt = (None if self.previous_start is None
                        else self.start - self.previous_start)
        self.previous_start = self.start
        self.dt = dt
        self.control_dt = None
        self.stages = dict.fromkeys(("state_update", "mapping", "conditioning", "dls", "env_step"), 0.)
        self.overhead = 0.
        self.dls_calls = 0
        self.flags = {"translation_speed_limited": False, "angular_speed_limited": False}
        self.active = False

    def add(self, group, name, value):
        value = float(value)
        if not math.isfinite(value):
            self.counts[f"{group}.{name}.nonfinite"] += 1
            return
        key = f"{group}.{name}"
        if key not in self.values:
            self.values[key] = [0, 0., -math.inf, deque(maxlen=self.capacity)]
        stat = self.values[key]
        stat[0] += 1
        stat[1] += value
        stat[2] = max(stat[2], value)
        stat[3].append(value)

    def capture(self, pipeline, *, controller_updated=False, control_dt=None,
                servo_q_target=None):
        before = self.clock()
        c, target = pipeline.controller, pipeline.last_target
        raw_target = pipeline.last_raw_target
        self.active = bool(target.fresh and target.active and not target.clutched)
        group = "active" if self.active else "inactive"
        self.control_dt = control_dt
        self.counts[f"{group}.ticks"] += 1
        self.counts[f"{group}.mapper_updates"] += int(target.mapper_updated)
        self.counts[f"{group}.dls_calls"] += self.dls_calls
        input_group = "new_input" if target.mapper_updated else "held_input"
        self.counts[f"{group}.{input_group}.ticks"] += 1
        self.counts[f"{group}.{input_group}.dls_calls"] += self.dls_calls
        self.counts[f"{group}.control_updates"] += int(controller_updated)
        self.add(group, "dls_calls_per_tick", self.dls_calls)
        sampled = (not self.lightweight or target.synchronized or target.reanchored
                   or (controller_updated and self.counts[f"{group}.control_updates"] % 5 == 0)
                   or (not self.active and self.counts[f"{group}.ticks"] % 10 == 0))
        if sampled and (controller_updated or target.synchronized or target.reanchored):
            metrics = self.rotation_command.update(pipeline, pipeline.mapper._latest, float(c.data.time))
            for name, value in metrics.items():
                self.add(group, name, value)
        if self.dls_calls and sampled:
            for name, value in self.flags.items():
                self.add(group, name, value)
            raw = c.diagnostics["qdot_total"]
            limited = np.abs(raw) > c.helpers.MAX_JOINT_SPEED
            self.add(group, "joint_speed_limited", np.any(limited))
            for i, flag in enumerate(limited, 1):
                self.add(group, f"joint{i}_speed_limited", flag)
        # Sample on control-update count, not physics parity: an edge may shift
        # the 500 Hz phase. No expensive pose/residual work on other ticks.
        detailed = (target.synchronized or target.reanchored
                    or (self.active and controller_updated
                        and self.counts[f"{group}.control_updates"] % 5 == 0)
                    or (not self.active and self.counts[f"{group}.ticks"] % 10 == 0))
        if not detailed:
            self.overhead += self.clock() - before
            return
        self.anti_windup = c.anti_windup.diagnostics()
        p, r = c.actual_pose()  # Pre-physics actual, same state used by IK.
        for name, value in {
            "raw_position_error_mm": 1000 * np.linalg.norm(raw_target.position - p),
            "conditioned_position_error_mm": 1000 * np.linalg.norm(c.target_position - p),
            "raw_conditioned_position_error_mm": 1000 * np.linalg.norm(raw_target.position - c.target_position),
            "raw_orientation_error_deg": angular_error_deg(raw_target.rotation, r),
            "conditioned_orientation_error_deg": angular_error_deg(c.target_rotation, r),
            "raw_conditioned_orientation_error_deg": angular_error_deg(raw_target.rotation, c.target_rotation),
            "deadband_position_error_mm": 1000 * np.linalg.norm(target.position - p),
            "deadband_orientation_error_deg": angular_error_deg(target.rotation, r),
            "deadband_conditioned_orientation_error_deg": angular_error_deg(target.rotation, c.target_rotation),
        }.items():
            self.add(group, name, value)
        applied = c.hold_q_target if servo_q_target is None else servo_q_target
        for i, error in enumerate(applied - c.data.qpos[c.qpos_indices], 1):
            self.add(group, f"joint{i}_target_minus_actual_rad", error)
            self.add(group, f"joint{i}_absolute_error_rad", abs(error))
        d = c.diagnostics
        # Stale helper diagnostics must not be counted on an edge/no-IK tick.
        if self.dls_calls:
            raw = d["qdot_total"]
            clipped = np.clip(raw, -c.helpers.MAX_JOINT_SPEED, c.helpers.MAX_JOINT_SPEED)
            for key in ("jacobian_condition", "task_residual_norm", "null_task_leak"):
                self.add(group, key, d[key])
            # DLS compromise versus additional error caused by clipping.
            jp, jr = c.jacp[:, c.dof_indices], c.jacr[:, c.dof_indices]
            desired_p = c.helpers.POSITION_KP * (c.target_position - p)
            desired_r = c.helpers.ORIENTATION_KP * c.helpers.matrix_to_rotvec(c.target_rotation @ r.T)
            for label, qdot in (("before_clip", raw), ("after_clip", clipped)):
                self.add(group, f"linear_residual_{label}_m_s", np.linalg.norm(jp @ qdot - desired_p))
                self.add(group, f"angular_residual_{label}_rad_s", np.linalg.norm(jr @ qdot - desired_r))
        self.overhead += self.clock() - before

    def finish(self, env_step_s):
        before = self.clock()
        self.stages["env_step"] = env_step_s
        group = "active" if self.active else "inactive"
        if self.lightweight and self.counts[f"{group}.ticks"] % 11:
            self.previous_finish = self.clock()
            return
        elapsed = before - self.start
        self.add(group, "physics_dt_ms", self.dt * 1000)
        if self.control_dt is not None:
            self.add(group, "control_dt_ms", self.control_dt * 1000)
        if self.wall_dt is not None:
            self.add(group, "wall_loop_dt_ms", self.wall_dt * 1000)
        for key, duration in self.stages.items():
            self.add(group, key + "_ms", duration * 1000)
        self.add(group, "other_ms", max(0., elapsed - sum(self.stages.values()) - self.overhead) * 1000)
        self.overhead += self.clock() - before
        self.add(group, "diagnostics_ms", self.overhead * 1000)
        self.add(group, "control_work_ms", (self.clock() - self.start) * 1000)
        self.previous_finish = self.clock()

    def summary(self):
        return {
            "percentile_window": self.capacity,
            "lightweight": self.lightweight,
            "sampling": ("pre-physics actual; counters every physics tick; timing samples every 11 ticks; "
                         "limiter/rotation samples every 5 control updates; pose/residual about 100 Hz simulation; "
                         "mean/max/count over sampled values, p95 recent window") if self.lightweight else
                        ("pre-physics actual; timings every physics tick; limiter fractions "
                         "every DLS update; detailed pose/residual about 100 Hz simulation; "
                         "mean/max/count lifetime, p95 recent window"),
            "counts": dict(self.counts),
            "rotation_command": dict(self.rotation_command.values),
            "anti_windup": dict(self.anti_windup),
            "metrics": {name: {"count": s[0], "mean": s[1] / s[0],
                               "p95": float(np.percentile(s[3], 95)), "max": s[2]}
                        for name, s in sorted(self.values.items())},
        }

    def publish(self, *, force=False):
        # Called by UI/status/shutdown, never by finish/control loop.
        now = self.clock()
        if force or now - self.last_publish >= 2.:
            self.cached = self.summary()
            if self.report_path:
                path = Path(self.report_path).expanduser()
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(path.suffix + ".tmp")
                temporary.write_text(json.dumps(self.cached, indent=2, allow_nan=False) + "\n")
                temporary.replace(path)
            self.last_publish = now
        return self.cached
