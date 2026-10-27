"""Offline Square step/hold evidence; no viewer, sockets, Touch, or recording."""
import argparse
import json
import math
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from geomagic.teleop.diagnostics import angular_change_rad
from geomagic.teleop.input_deadband import DeadbandConfig
from geomagic.teleop.synthetic import sample
from geomagic.teleop.validate_gravity_timing import square_model, reset, make_backend
from geomagic.vendor.geomagic_core.rotations import matrix_to_rotation_vector


def trial(hz=500, degrees=30., axis=(1., 0., 0.), *, step=True, deadband=False):
    model, data = square_model()
    reset(model, data)
    backend, now = make_backend(model, data)
    backend.control_period = 1. / hz
    if deadband:
        from geomagic.teleop.input_deadband import PoseInputDeadband
        backend.pipeline.deadband = PoseInputDeadband(DeadbandConfig.from_environment())
    c = backend.pipeline.controller
    _, initial = c.actual_pose()
    vector = np.asarray(axis) / np.linalg.norm(axis) * math.radians(degrees)
    rows = []
    sequence = 0
    stride = round(.001 ** -1 / hz)
    input_stop = .008 if step else .508
    for tick in range(round((input_stop + 5.004) / .001) + 1):
        now[0] = tick * .001
        if tick % stride == 0:
            fraction = (float(now[0] >= .008) if step
                        else min(1., max(0., (now[0] - .008) / .5)))
            backend.client.state = sample(sequence, time=now[0], rotvec=vector * fraction)
            sequence += 1
        backend.update_command(.001)
        _, actual = c.actual_pose()
        rotations = (backend.pipeline.last_raw_target.rotation,
                     backend.pipeline.last_target.rotation, c.target_rotation, actual)
        angles = [math.degrees(angular_change_rad(r, initial)) for r in rotations]
        velocities = ([0.] * 4 if not rows else
                      [math.degrees(angular_change_rad(r, before)) / .001
                       for r, before in zip(rotations, previous)])
        error = math.degrees(angular_change_rad(rotations[0], actual))
        rows.append((now[0], *angles, *velocities, error))
        previous = tuple(r.copy() for r in rotations)
        backend.apply_controller(model, data, .001)
        mujoco.mj_step(model, data)
    rows = np.asarray(rows)
    static = rows[:, 0] >= input_stop + .004
    settled = rows[:, 0] >= input_stop + 4.
    checkpoints = {}
    for label, time in (("first", input_stop), ("one_second", input_stop + 1.),
                        ("five_seconds", input_stop + 5.)):
        row = rows[round(time / .001)]
        checkpoints[label] = dict(zip(("sim_s", "raw_deg", "deadband_deg", "conditioned_deg",
                                      "actual_deg", "raw_deg_s", "deadband_deg_s",
                                      "conditioned_deg_s", "actual_deg_s", "raw_actual_error_deg"),
                                     map(float, row)))
    return {"hz": hz, "input_deg": degrees, "axis": list(axis), "step": step,
            "deadband": deadband, "checkpoints": checkpoints,
            "static_max_raw_deg_s": float(rows[static, 5].max()),
            "settled_max_conditioned_deg_s": float(rows[settled, 7].max()),
            "settled_max_actual_deg_s": float(rows[settled, 8].max()),
            "final_raw_actual_error_deg": float(rows[-1, 9])}, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results, traces = [], {}
    for hz in (250, 500, 1000):
        result, rows = trial(hz)
        results.append(result)
        traces[f"step_30_x_{hz}"] = rows
    for degrees, axis, label in ((5., (1., 0., 0.), "step_5_x"),
                                 (30., (0., 1., 0.), "step_30_y"),
                                 (30., (0., 0., 1.), "step_30_z"),
                                 (30., (1., -2., 3.), "step_30_mixed")):
        result, rows = trial(500, degrees, axis)
        results.append(result)
        traces[label] = rows
    for enabled in (False, True):
        result, rows = trial(500, step=False, deadband=enabled)
        results.append(result)
        traces[f"ramp_30_x_deadband_{enabled}"] = rows
    result, rows = trial(500, step=True, deadband=True)
    results.append(result)
    traces["step_30_x_deadband_True"] = rows
    np.savez(args.output / "step_hold_traces.npz", **traces,
             columns=np.array(("sim_s", "raw_deg", "deadband_deg", "conditioned_deg",
                               "actual_deg", "raw_deg_s", "deadband_deg_s",
                               "conditioned_deg_s", "actual_deg_s", "raw_actual_error_deg")))
    (args.output / "step_hold.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(json.dumps(results, indent=2, allow_nan=False))
    # Existing plotting environment, without any package installation.
    subprocess.run(["/usr/bin/python3", "-B", "-m", "geomagic.teleop.plot_rotation_semantics",
                    str(args.output)], check=True)


if __name__ == "__main__":
    main()
