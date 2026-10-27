"""Explicit offline Square validation. Steps simulation; never opens hardware/UDP.

python -B -m geomagic.teleop.validate_gravity_timing --output /tmp/geo_validation
Benchmarks use production runner/actions/post-step numeric diagnostics, without
rendering, file I/O on ticks, UDP decoding, or UI publishing. All dynamics dt is
simulation elapsed time; wall time is used only for pacing/input freshness.
"""
import argparse
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import mujoco
import numpy as np

from geomagic.teleop.synthetic import sample
from geomagic.teleop.diagnostics import angular_error_deg
from geomagic.teleop.robot_helpers import MIMIC_ROOT
from geomagic.teleop.input_deadband import DeadbandConfig


REPRESENTATIVE_Q = {'home': [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853], 'rotation_start': [-6.649529698260463e-11, 0.013543322143224845, -5.461744327415895e-06, -1.5836772576008744, -0.0008300551914630847, 1.5685787651671363, -0.785300374161692], 'middle': [0.08070756479620642, -0.2811478193160239, -0.0033975561970914113, -2.1817447568724213, -0.8315710816923857, 2.427721453930734, -1.796289210645956], 'end': [0.09319044031860241, -0.1151001733748088, 0.0282484249926648, -1.9206815154911534, -0.46458618689767794, 1.9370511915448623, -1.4530993378869521]}

@dataclass
class LocalStatus:
    session_changed: bool = False


class OfflineClient:
    """Accepted-state source only; no socket/thread/device or MuJoCo ownership."""
    def __init__(self):
        self.state = None
        self.changed = False

    def poll(self):
        pass

    def latest_state(self):
        # Match production latest_state(): immutable dataclass copy per poll.
        return None if self.state is None else replace(self.state)

    def status(self):
        return LocalStatus(self.changed)

    def acknowledge_session_change(self):
        self.changed = False

    def close(self):
        pass


def make_backend(model, data, *, mode="rotation", now=None):
    from geomagic.leader.teleop_backend import GeomagicTeleopBackend
    now = now or [0.]
    backend = GeomagicTeleopBackend(model, data, teleop_mode=mode,
                                  activation_samples=2, monotonic=lambda: now[0],
                                  deadband_config=DeadbandConfig(enabled=False))
    backend.client = OfflineClient()
    return backend, now


def square_model():
    model = mujoco.MjModel.from_xml_path(str(MIMIC_ROOT / "environments/mimicgen_square/scene.xml"))
    return model, mujoco.MjData(model)


def reset(model, data, q=None):
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    data.ctrl[model.actuator("fr3_gripper").id] = 0
    if q is not None:
        data.qpos[:7] = q
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)


def hold_cycles(model, data, q, cycles=10, seconds=3., support=True):
    reset(model, data, q)
    backend, now = make_backend(model, data)
    c = backend.pipeline.controller
    initial_q = data.qpos[c.qpos_indices].copy()
    initial_p, initial_r = c.actual_pose()
    sequence = 0
    per_cycle = []
    for cycle in range(cycles):
        # Genuine deadman engage/release mapping edges, with zero intended motion.
        for grey in (True, True, False):
            now[0] = float(data.time)
            backend.client.state = sample(sequence, time=now[0], grey=grey)
            sequence += 1
            backend.update_command(.001)
        np.testing.assert_array_equal(c.hold_q_target, data.qpos[c.qpos_indices])
        if not support:
            backend.hold_q_target = c.hold_q_target.copy()
        hold = backend.hold_q_target.copy()
        before_q = data.qpos[c.qpos_indices].copy()
        before_p, before_r = c.actual_pose()
        for _ in range(round(seconds / model.opt.timestep)):
            # Inactive supported action is constant, with no intent tracking/IK.
            backend.apply_controller(model, data, .001)
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)  # Measurement only; no per-tick extra forward.
        after_p, after_r = c.actual_pose()
        np.testing.assert_array_equal(backend.hold_q_target, hold)
        per_cycle.append({"cycle": cycle + 1,
                          "joint_drift_rad": (data.qpos[c.qpos_indices] - before_q).tolist(),
                          "xyz_drift_mm": ((after_p - before_p) * 1000).tolist(),
                          "orientation_drift_deg": angular_error_deg(after_r, before_r)})
    final_p, final_r = c.actual_pose()
    return {"cycles": cycles, "seconds_per_hold": seconds, "support": support,
            "joint_drift_rad": (data.qpos[c.qpos_indices] - initial_q).tolist(),
            "xyz_drift_mm": ((final_p - initial_p) * 1000).tolist(),
            "orientation_drift_deg": angular_error_deg(final_r, initial_r),
            "per_cycle": per_cycle}


def tracking_trial(model, data, device_vector, seconds=5.):
    reset(model, data)
    backend, now = make_backend(model, data)
    c = backend.pipeline.controller
    initial_p, initial_r = c.actual_pose()
    rows, joint_limited, angular_limited, updates = [], 0, 0, 0
    seq = 0
    for tick in range(round(seconds / model.opt.timestep)):
        now[0] = float(data.time)
        if tick % 2 == 0:
            # First two new samples establish actual anchor; move over 0.5 sim s.
            fraction = min(1., max(0., (data.time - .004) / .5))
            backend.client.state = sample(seq, time=now[0], rotvec=device_vector * fraction)
            seq += 1
        old_updates = backend.control_updates
        old_r = c.target_rotation.copy()
        backend.update_command(.001)
        if backend.control_updates != old_updates:
            updates += 1
            raw = c.diagnostics["qdot_total"]
            joint_limited += int(np.any(np.abs(raw) > c.helpers.MAX_JOINT_SPEED))
            err = np.linalg.norm(c.helpers.matrix_to_rotvec(backend.pipeline.last_target.rotation @ old_r.T))
            angular_limited += int(min(c.helpers.TARGET_ROTATION_KP * err,
                                      math.sqrt(2 * c.helpers.MAX_TARGET_ANGULAR_ACCEL * err))
                                   > c.helpers.MAX_TARGET_ANGULAR_SPEED)
        target = backend.pipeline.last_target
        p, r = c.actual_pose()
        rows.append((data.time, angular_error_deg(target.rotation, initial_r),
                     angular_error_deg(c.target_rotation, initial_r), angular_error_deg(r, initial_r),
                     angular_error_deg(target.rotation, r), angular_error_deg(c.target_rotation, r),
                     np.linalg.norm(p - initial_p) * 1000, *(1000 * (p - initial_p))))
        backend.apply_controller(model, data, .001)
        mujoco.mj_step(model, data)
    a = np.asarray(rows)
    return {"device_rotation_deg": float(np.linalg.norm(device_vector) * 180 / math.pi),
            "duration_sim_s": seconds, "control_updates": updates,
            "raw_angle_final_deg": float(a[-1, 1]), "conditioned_angle_final_deg": float(a[-1, 2]),
            "actual_angle_final_deg": float(a[-1, 3]),
            "raw_actual_error_final_deg": float(a[-1, 4]),
            "conditioned_actual_error_final_deg": float(a[-1, 5]),
            "raw_actual_error_max_deg": float(a[:, 4].max()),
            "conditioned_actual_error_max_deg": float(a[:, 5].max()),
            "position_error_max_mm": float(a[:, 6].max()),
            "position_error_final_mm": float(a[-1, 6]),
            "xyz_final_mm": a[-1, 7:10].tolist(),
            "angular_saturation_fraction": angular_limited / updates,
            "joint_saturation_fraction": joint_limited / updates}, a


def benchmark(*, diagnostics, paced, seconds=3., baseline=False):
    # Baseline modules, if requested, are loaded by a fresh subprocess BEFORE
    # calling this function. No runtime global swapping during a simulation.
    sys.path.insert(0, str(MIMIC_ROOT))
    from runtime.live_runner import MimicLiveRunner, STEPS_PER_DATASET_SAMPLE
    from geomagic.leader.teleop_backend import GeomagicTeleopBackend
    client, now = OfflineClient(), [0.]
    def initialize(b):
        b.client = client
        b._clock = lambda: now[0]
        b.pipeline.mapper.activation_samples = 2
        b.reset_controller_state()
    env = {"MIMIC_GEOMAGIC_DIAGNOSTICS": "1" if diagnostics else "0"}
    with patch.dict(os.environ, env), patch.object(GeomagicTeleopBackend, "initialize", initialize):
        r = MimicLiveRunner(input_backend="geomagic", task="mimicgen_square", image_size=(16, 16))
    r.set_teleop_mode("rotation")
    # No close-time report file: diagnostic summary remains in memory only.
    if r.backend.tracking:
        r.backend.tracking.report_path = None
    count, seq, dls, frames = 0, 0, 0, 0
    dls_fn = r.backend.pipeline.controller.helpers.compute_joint_target
    def counted(*args, **kwargs):
        nonlocal dls
        dls += 1
        return dls_fn(*args, **kwargs)
    start_sim = float(r.data.time)
    start = time.perf_counter()
    next_deadline = start + .001
    next_input = 0.
    with patch.object(r.backend.pipeline.controller.helpers, "compute_joint_target", counted):
        while count < round(seconds * 1000):
            wall = time.perf_counter() - start
            elapsed_sim = float(r.data.time) - start_sim
            input_time = wall if paced else elapsed_sim
            now[0] = input_time
            if input_time >= next_input:
                vector = np.array((.2, -.15, .25)) * math.sin(2 * math.pi * .4 * input_time)
                client.state = sample(seq, time=input_time, rotvec=vector)
                seq += 1
                next_input = (math.floor(input_time / .002 + 1e-9) + 1) * .002
            _, observation = r.step_once(capture_observation=count % STEPS_PER_DATASET_SAMPLE == 0)
            _ = r.post_step_state()
            frames += int(observation is not None)
            count += 1
            if paced:
                remaining = next_deadline - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                next_deadline += .001
                if time.perf_counter() - next_deadline > (.005 if baseline else .020):
                    next_deadline = time.perf_counter() + .001
    wall = time.perf_counter() - start
    sim = float(r.data.time) - start_sim
    summary = r.backend.tracking.summary() if diagnostics else None
    r.close()
    return {"diagnostics": diagnostics, "paced": paced, "baseline": baseline,
            "wall_s": wall, "simulation_s": sim, "physics_steps": count,
            "physics_hz_wall": count / wall, "control_updates": dls,
            "control_hz_wall": dls / wall, "input_hz_wall": seq / wall,
            "rtf": sim / wall, "dataset_frames": frames, "diagnostics_summary": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--baseline-root", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    baseline = args.baseline_root is not None
    if baseline:
        import importlib.util
        for name in ("geomagic.teleop.diagnostics", "geomagic.teleop.fr3_target_controller",
                     "geomagic.teleop.pipeline", "geomagic.leader.teleop_backend"):
            path = args.baseline_root / (name.replace(".", "/") + ".py")
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
    result = {}
    if not args.benchmark_only:
        model, data = square_model()
        result["holds"] = {}
        for name, q in REPRESENTATIVE_Q.items():
            result["holds"][name] = hold_cycles(model, data, q)
        result["bare_hold_home"] = hold_cycles(model, data, REPRESENTATIVE_Q["home"], support=False)
        result["tracking"] = {}
        for name, v in (("10deg", (0., np.deg2rad(10), 0.)),
                        ("30deg", (0., np.deg2rad(30), 0.)),
                        ("mixed", np.deg2rad((15., -20., 25.)))):
            info, rows = tracking_trial(model, data, np.asarray(v))
            result["tracking"][name] = info
            np.savez(args.output / f"tracking_{name}.npz", values=rows,
                     columns=np.array(["sim_s", "raw_deg", "conditioned_deg", "actual_deg", "raw_error_deg",
                                       "conditioned_error_deg", "position_error_mm", "x_mm", "y_mm", "z_mm"]))
    result["benchmarks"] = []
    for paced in (False, True):
        for diagnostics in (False, True):
            trials = [benchmark(diagnostics=diagnostics, paced=paced, baseline=baseline) for _ in range(3)]
            result["benchmarks"].extend(trials)
    (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "benchmarks"}, indent=2))
    for row in result["benchmarks"]:
        print({k: v for k, v in row.items() if k != "diagnostics_summary"})


if __name__ == "__main__":
    main()
