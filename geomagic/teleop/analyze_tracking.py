"""Offline saved-log analysis; never launches a live backend or collection.

--profile additionally times a synthetic, in-memory UDP path and isolated
MuJoCo steps on caller-owned model/data. It has no sockets, UI or devices.
"""
import argparse
from collections import deque
import json
from pathlib import Path
import time
from unittest.mock import patch

import mujoco
import numpy as np

from geomagic.bridge.protocol import GeomagicStatePacket, encode_packet
from geomagic.bridge.udp_state_client import GeomagicUDPStateClient
from geomagic.leader.teleop_backend import GeomagicTeleopBackend
from geomagic.teleop.diagnostics import TrackingDiagnostics
from geomagic.teleop.fr3_target_controller import FR3TargetController
from geomagic.teleop.robot_helpers import MIMIC_ROOT
from geomagic.vendor.geomagic_core.rotations import (
    matrix_to_rotation_vector, quaternion_xyzw_to_matrix,
)


def rotation(pose):
    return quaternion_xyzw_to_matrix(pose[[1, 2, 3, 0]])


def stats(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(values)), "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values))}


def saved_analysis(z, model, data):
    c = FR3TargetController(model, data)
    h = c.helpers
    active = z["teleop_active"] & (z["teleop_mode"] == "rotation")
    indices = np.flatnonzero(active)
    series = {name: [] for name in ("jacobian_condition", "null_task_leak",
              "linear_residual_before_clip_m_s", "linear_residual_after_clip_m_s",
              "angular_residual_before_clip_rad_s", "angular_residual_after_clip_rad_s",
              "dls_replay_ms", "command_replay_max_error_rad", "actual_pose_replay_error_mm")}
    joint_flags, angular_flags, translation_flags = [], [], []
    for number, i in enumerate(indices):
        if number % 1000 == 0:
            print(f"Reconstructing saved IK: {number}/{len(indices)}", flush=True)
        if i < 2:
            raise ValueError("need two pre-active observations for this recording layout")
        # Original worker logs qpos AFTER mj_step, whose cached site/Jacobian
        # correspond to its PRE-step qpos. On the next tick compute_joint_target
        # sees cached sites from i-2 and current qpos i-1. Reconstruct both.
        data.qpos[c.qpos_indices] = z["arm_qpos"][i - 2]
        mujoco.mj_forward(model, data)
        data.qpos[c.qpos_indices] = z["arm_qpos"][i - 1]
        p, r = c.actual_pose()
        series["actual_pose_replay_error_mm"].append(
            1000 * np.linalg.norm(p - z["actual_ee_pose"][i - 1, 4:]))
        target = z["target_ee_pose"][i]
        target_rotation = rotation(target)
        before = time.perf_counter()
        q, d = h.compute_joint_target(
            model, data, c.ee_site_id, c.dof_indices, c.qpos_indices,
            c.joint_lower, c.joint_upper, target[4:], target_rotation,
            z["qtarget"][i - 1], c.jacp, c.jacr, .001,
            teleop_mode="full_pose", enable_nullspace_posture=h.ENABLE_NULLSPACE_POSTURE,
            q_posture_reference=z["qtarget"][0])
        series["dls_replay_ms"].append((time.perf_counter() - before) * 1000)
        series["command_replay_max_error_rad"].append(np.max(np.abs(q - z["qtarget"][i])))
        series["jacobian_condition"].append(d["jacobian_condition"])
        series["null_task_leak"].append(d["null_task_leak"])
        raw = d["qdot_total"]
        clipped = np.clip(raw, -h.MAX_JOINT_SPEED, h.MAX_JOINT_SPEED)
        joint_flags.append(np.abs(raw) > h.MAX_JOINT_SPEED)
        desired_p = h.POSITION_KP * (target[4:] - p)
        desired_r = h.ORIENTATION_KP * matrix_to_rotation_vector(rotation(target) @ r.T)
        for label, qdot in (("before_clip", raw), ("after_clip", clipped)):
            series[f"linear_residual_{label}_m_s"].append(
                np.linalg.norm(c.jacp[:, c.dof_indices] @ qdot - desired_p))
            series[f"angular_residual_{label}_rad_s"].append(
                np.linalg.norm(c.jacr[:, c.dof_indices] @ qdot - desired_r))
        previous, raw_pose = z["target_ee_pose"][i - 1], z["command_ee_pose"][i]
        ep = np.linalg.norm(raw_pose[4:] - previous[4:])
        er = np.linalg.norm(matrix_to_rotation_vector(rotation(raw_pose) @ rotation(previous).T))
        translation_flags.append(min(h.POSITION_KP * ep, np.sqrt(2 * h.MAX_TARGET_LINEAR_ACCEL * ep)) > h.MAX_TARGET_LINEAR_SPEED)
        angular_flags.append(min(h.TARGET_ROTATION_KP * er, np.sqrt(2 * h.MAX_TARGET_ANGULAR_ACCEL * er)) > h.MAX_TARGET_ANGULAR_SPEED)
    joint_flags = np.asarray(joint_flags)
    position = z["command_ee_pose"][indices, 4:]
    conditioned = z["target_ee_pose"][indices, 4:]
    actual = z["actual_ee_pose"][indices, 4:]
    wall_dt = np.diff(z["wall_time"])
    return {
        "active_samples": len(indices),
        "limits_percent": {"translation": 100 * float(np.mean(translation_flags)),
                           "angular": 100 * float(np.mean(angular_flags)),
                           "joint_any": 100 * float(np.mean(np.any(joint_flags, axis=1))),
                           "per_joint": (100 * np.mean(joint_flags, axis=0)).tolist()},
        "raw_position_range_mm": (1000 * np.ptp(position, axis=0)).tolist(),
        "conditioned_position_range_mm": (1000 * np.ptp(conditioned, axis=0)).tolist(),
        "raw_conditioned_position_error_mm": stats(1000 * np.linalg.norm(position - conditioned, axis=1)),
        "conditioned_actual_position_error_mm": stats(1000 * np.linalg.norm(conditioned - actual, axis=1)),
        "qtarget_actual_abs_rad_per_joint": [stats(np.abs(z["qtarget"][indices, j] - z["arm_qpos"][indices, j])) for j in range(7)],
        "wall_dt_active_ms": stats(1000 * wall_dt[active[1:]]),
        "wall_dt_inactive_ms": stats(1000 * wall_dt[~z["teleop_active"][1:]]),
        "recomputed": {key: stats(value) for key, value in series.items()},
        "replay_note": "Cached site/Jacobian i-2 + actual q i-1 reproduce pre-step IK; command error validates reconstruction. Replay time is measured NOW, not stored live timing.",
    }


class MemorySocket:
    def __init__(self):
        self.pending = deque()
    def setblocking(self, value):
        pass
    def bind(self, address):
        self.address = address
    def getsockname(self):
        return self.address
    def recvfrom(self, size):
        if not self.pending:
            raise BlockingIOError
        return self.pending.popleft(), ("127.0.0.1", 1234)
    def close(self):
        pass


def profile(z, model, data, *, ticks=3000):
    # Backend production update/apply methods, no initialize(), no real socket.
    now, sock = [0.], MemorySocket()
    backend = GeomagicTeleopBackend(model, data, teleop_mode="rotation",
                                    activation_samples=2, monotonic=lambda: now[0])
    tracker = TrackingDiagnostics()
    backend.tracking = backend.pipeline.tracking = backend.pipeline.controller.tracking = tracker
    with patch("geomagic.bridge.udp_state_client.socket.socket", return_value=sock):
        backend.client = GeomagicUDPStateClient(monotonic=lambda: now[0])
    indices = np.flatnonzero(z["teleop_active"])
    sequence = 0
    for phase in ("inactive", "active"):
        for tick in range(ticks):
            i = int(indices[tick % len(indices)])
            # Load recorded actual states outside the timed control work. No
            # result is recorded as a new episode or carried into a live run.
            data.qpos[backend.fr3_qpos_indices] = z["arm_qpos"][i]
            mujoco.mj_forward(model, data)
            now[0] += .001
            if tick % 2 == 0:  # 500 Hz source, 1000 Hz logical control
                sock.pending.append(encode_packet(GeomagicStatePacket(
                    1, "offline-profile", sequence, now[0], "offline",
                    tuple(z["geomagic_position_m"][i].tolist()), tuple(z["geomagic_orientation_xyzw"][i].tolist()),
                    phase == "active", False, True)))
                sequence += 1
            backend.update_command(.001)
            backend.apply_controller(model, data, .001)
            before = time.perf_counter()
            mujoco.mj_step(model, data)
            backend.finish_diagnostic_tick(time.perf_counter() - before)
    report = tracker.summary()
    # No publish/report writes on the control loop; save only the final output.
    backend.client.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    # NpzFile is lazy and reloads/decompresses on EVERY key access. Cache once
    # so offline disk/ZIP work cannot be mistaken for controller timing.
    with np.load(args.episode / "original_1000hz.npz", allow_pickle=False) as archive:
        z = {key: archive[key] for key in archive.files}
    model = mujoco.MjModel.from_xml_path(str(MIMIC_ROOT / "environments/mimicgen_square/scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    report = {"episode": str(args.episode), "saved_log": saved_analysis(z, model, data)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.profile:
        print("Profiling offline control components (no device/collection)", flush=True)
        report["offline_profile"] = profile(z, model, data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Offline analysis saved: {args.output}")


if __name__ == "__main__":
    main()
