"""Gain-only regression through the real controller; no hardware or recording.

Step measurements hold actual model/data fixed to isolate the conditioner.
The real DLS still runs once/update; these are not actual-EE response predictions.
"""
import ast
from dataclasses import fields, replace
import itertools
import math
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import mujoco

from geomagic.bridge.udp_state_client import TransportStatus
from geomagic.leader.teleop_backend import GeomagicTeleopBackend
from geomagic.teleop.fr3_target_controller import FR3TargetController
from geomagic.teleop.input_deadband import DeadbandConfig
from geomagic.teleop.pipeline import GeomagicTargetPipeline
from geomagic.teleop.pose_mapper import CartesianTarget, MODES
from geomagic.teleop.robot_helpers import (
    BRIDGE_PATH, CONSTANTS, FUNCTIONS, GEOMAGIC_ORIENTATION_CONDITIONING_GAIN,
    robot_helpers,
)
from geomagic.teleop.synthetic import sample
from geomagic.teleop.validate_gravity_timing import square_model, reset


@pytest.fixture(scope="module")
def model():
    return square_model()[0]


def controller(model, gain):
    data = mujoco.MjData(model)
    reset(model, data)
    return FR3TargetController(model, data, orientation_conditioning_gain=gain)


def active_target(c, mode):
    p, r = c.actual_pose()
    return CartesianTarget(p, r, mode, True, True, False, False, "synthetic", True)


def orientation_step_response(model, gain, degrees, mode="rotation", seconds=3.):
    """Return crossings and limits with ONLY orientation conditioner gain changed."""
    c = controller(model, gain)
    h = c.helpers
    target = active_target(c, mode)
    c.compute(target, .002)  # Actual-state activation edge, no DLS.
    initial_r = target.rotation.copy()
    axis = initial_r[:, 2] if mode == "rotation_z" else np.array((1., 0., 0.))
    desired_r = h.rotvec_to_matrix(axis * math.radians(degrees)) @ initial_r
    target = replace(target, rotation=desired_r)
    rows = []
    speed_limited = acceleration_limited = snap_updates = 0
    for tick in range(round(seconds / .002)):
        previous_r = c.target_rotation.copy()
        previous_w = c.angular_velocity.copy()
        previous_q = c.hold_q_target.copy()
        error = np.linalg.norm(h.matrix_to_rotvec(desired_r @ previous_r.T))
        snap = error < h.ROTATION_SNAP_ERROR
        desired_speed = min(gain * error, h.MAX_TARGET_ANGULAR_SPEED,
                            math.sqrt(2 * h.MAX_TARGET_ANGULAR_ACCEL * error))
        if not snap:
            desired_w = h.matrix_to_rotvec(desired_r @ previous_r.T) * desired_speed / error
            acceleration_limited += int(np.linalg.norm(desired_w - previous_w)
                                        > h.MAX_TARGET_ANGULAR_ACCEL * .002 + 1e-12)
            speed_limited += int(min(gain * error, math.sqrt(2 * h.MAX_TARGET_ANGULAR_ACCEL * error))
                                 > h.MAX_TARGET_ANGULAR_SPEED + 1e-12)
        q = c.compute(target, .002)
        acceleration = np.linalg.norm(c.angular_velocity - previous_w) / .002
        if snap and np.linalg.norm(previous_w) > 0:
            snap_updates += 1
        assert np.isfinite(q).all() and np.isfinite(c.target_rotation).all()
        assert np.all(q >= c.joint_lower) and np.all(q <= c.joint_upper)
        assert np.max(np.abs(q - previous_q)) <= h.MAX_JOINT_SPEED * .002 + 1e-12
        assert np.linalg.norm(c.angular_velocity) <= h.MAX_TARGET_ANGULAR_SPEED + 1e-12
        if not snap:
            assert acceleration <= h.MAX_TARGET_ANGULAR_ACCEL + 1e-10
        angle = math.degrees(axis @ h.matrix_to_rotvec(c.target_rotation @ initial_r.T))
        rows.append((.002 * (tick + 1), angle, np.linalg.norm(c.angular_velocity), acceleration, snap))
    a = np.asarray(rows)
    assert np.max(a[:, 1]) <= degrees + 1e-8
    assert np.max(np.abs(a[-100:, 1] - degrees)) < 1e-8
    metrics = dict(gain=gain, step_deg=degrees, mode=mode,
                   raw_step_deg=degrees, final_conditioned_deg=float(a[-1, 1]),
                   peak_angular_velocity_rad_s=float(a[:, 2].max()),
                   peak_angular_velocity_deg_s=float(np.degrees(a[:, 2].max())),
                   angular_acceleration_limited=bool(acceleration_limited),
                   angular_velocity_limited=bool(speed_limited),
                   velocity_cap_reached=bool(a[:, 2].max() >= h.MAX_TARGET_ANGULAR_SPEED - 1e-10),
                   peak_regular_angular_acceleration_rad_s2=float(a[a[:, 4] == 0, 3].max()),
                   peak_including_existing_snap_acceleration_rad_s2=float(a[:, 3].max()),
                   final_snap_updates=snap_updates,
                   overshoot_deg=max(0., float(a[:, 1].max() - degrees)))
    for fraction, name in ((.1, "t10_ms"), (.5, "t50_ms"), (.9, "t90_ms")):
        metrics[name] = float(a[np.flatnonzero(a[:, 1] >= degrees * fraction)[0], 0] * 1000)
    return metrics, a


def test_geomagic_override_is_private_and_shared_omy_math_remains_four():
    old, new = robot_helpers(4.), robot_helpers(8.)
    assert GEOMAGIC_ORIENTATION_CONDITIONING_GAIN == 8.
    assert old.condition_target.__globals__ is not new.condition_target.__globals__
    for h, gain in ((old, 4.), (new, 8.)):
        assert h.TARGET_ROTATION_KP == gain
        assert h.condition_target.__globals__["TARGET_ROTATION_KP"] == gain
        assert h.POSITION_KP == 8. and h.ORIENTATION_KP == 4.
        assert h.compute_joint_target.__globals__["ORIENTATION_KP"] == 4.
        assert h.condition_target.__code__.co_filename == str(BRIDGE_PATH)
    # Compile unchanged shared source without a Geomagic override or ROS startup.
    tree = ast.parse(BRIDGE_PATH.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS
                or isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id in CONSTANTS]
    omy = {"np": np, "mujoco": mujoco}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(BRIDGE_PATH), "exec"), omy)
    assert omy["TARGET_ROTATION_KP"] == 4.
    p, r = np.zeros(3), np.eye(3)
    v, w = np.zeros(3), np.zeros(3)
    desired = old.rotvec_to_matrix(np.array((.01, 0., 0.)))
    for _ in range(1000):
        expected = omy["condition_target"](p, r, p, desired, v, w, .002)
        actual = old.condition_target(p, r, p, desired, v, w, .002)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
        p, r, v, w = actual[0], actual[1], actual[5], actual[6]
    assert old.TARGET_ROTATION_KP == 4. and new.TARGET_ROTATION_KP == 8.


@pytest.mark.parametrize("gain", (0., -1., float("nan"), float("inf")))
def test_invalid_conditioning_gain_is_rejected(gain):
    with pytest.raises(ValueError):
        robot_helpers(gain)


@pytest.mark.parametrize("mode", ("rotation_z", "rotation", "6dof"))
@pytest.mark.parametrize("degrees", (5., 15., 30.))
def test_step_comparison_through_real_controller(model, mode, degrees):
    old, _ = orientation_step_response(model, 4., degrees, mode)
    new, _ = orientation_step_response(model, 8., degrees, mode)
    assert new["t50_ms"] < old["t50_ms"] and new["t90_ms"] < old["t90_ms"]
    assert old["angular_acceleration_limited"] and new["angular_acceleration_limited"]


def test_unsaturated_discrete_response_halves_time_constant(model):
    # 0.001 rad: k=8 starts exactly at amax*dt, not above it. Threshold
    # crossings occur before the existing 1e-4 rad final snap.
    results = []
    for gain in (4., 8.):
        result, _ = orientation_step_response(model, gain, math.degrees(.001))
        assert not result["angular_velocity_limited"]
        assert not result["angular_acceleration_limited"]
        for fraction, name in ((.1, "t10_ms"), (.5, "t50_ms"), (.9, "t90_ms")):
            n = math.ceil(math.log(1 - fraction) / math.log(1 - gain * .002))
            assert result[name] == pytest.approx(n * 2, abs=1e-8)
        results.append(result)
    assert results[1]["t90_ms"] / results[0]["t90_ms"] == pytest.approx(.5, abs=.01)


@pytest.mark.parametrize("mode", ("position", "6dof"))
@pytest.mark.parametrize("millimeters", (5., 20., 50.))
def test_translation_only_is_bitwise_unchanged(model, mode, millimeters):
    old, new = controller(model, 4.), controller(model, 8.)
    target = active_target(old, mode)
    for c in (old, new):
        c.compute(target, .002)
    target = replace(target, position=target.position + np.array((millimeters / 1000, 0., 0.)))
    for _ in range(1000):
        np.testing.assert_array_equal(old.compute(target, .002), new.compute(target, .002))
        np.testing.assert_array_equal(old.target_position, new.target_position)
        np.testing.assert_array_equal(old.target_rotation, new.target_rotation)
        np.testing.assert_array_equal(old.linear_velocity, new.linear_velocity)


@pytest.mark.parametrize("old_mode,new_mode", tuple(itertools.permutations(MODES, 2)))
def test_all_mode_transitions_anchor_actual_with_zero_jump(model, old_mode, new_mode):
    data = mujoco.MjData(model)
    reset(model, data)
    pipe = GeomagicTargetPipeline(model, data, mode=old_mode, activation_samples=2)
    for i in range(20):
        s = sample(i, position=(.05, .03, .04) if i >= 2 else (0., 0., 0.),
                   rotvec=(.6, -.4, .3) if i >= 2 else (0., 0., 0.))
        pipe.step(s, now=s.timestamp, dt=.002, gripper_command=0)
    data.qpos[0] += .02
    mujoco.mj_forward(model, data)
    p, r = pipe.controller.actual_pose()
    q = data.qpos[pipe.controller.qpos_indices].copy()
    assert pipe.set_mode(new_mode)
    for i in (20, 21):
        s = sample(i, position=(.3, -.2, .4), rotvec=(.8, -.6, .5))
        pipe.step(s, now=s.timestamp, dt=.002, gripper_command=0)
        np.testing.assert_array_equal(pipe.last_target.position, p)
        np.testing.assert_array_equal(pipe.last_target.rotation, r)
        np.testing.assert_array_equal(pipe.controller.hold_q_target, q)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("edge", ("grey", "white"))
def test_button_release_reengage_discards_old_target(model, mode, edge):
    data = mujoco.MjData(model)
    reset(model, data)
    pipe = GeomagicTargetPipeline(model, data, mode=mode, activation_samples=2)
    for i in range(20):
        s = sample(i, position=(.1, .1, .1) if i >= 2 else (0., 0., 0.),
                   rotvec=(.6, .3, -.4) if i >= 2 else (0., 0., 0.))
        pipe.step(s, now=s.timestamp, dt=.002, gripper_command=0)
    data.qpos[0] += .02
    mujoco.mj_forward(model, data)
    p, r = pipe.controller.actual_pose()
    q = data.qpos[pipe.controller.qpos_indices].copy()
    for i in (20, 21, 22):
        gates = ({"grey": False} if edge == "grey" else {"white": True}) if i < 22 else {}
        s = sample(i, position=(.3, .3, .3), rotvec=(.8, .5, -.6), **gates)
        pipe.step(s, now=s.timestamp, dt=.002, gripper_command=0)
        np.testing.assert_array_equal(pipe.last_target.position, p)
        np.testing.assert_array_equal(pipe.last_target.rotation, r)
        np.testing.assert_array_equal(pipe.controller.hold_q_target, q)


def test_gain_metadata_is_truthful_without_changing_learning_vectors(model):
    data = mujoco.MjData(model)
    reset(model, data)
    capture = SimpleNamespace(settings={})
    with patch("geomagic.leader.teleop_backend.get_sync_capture", return_value=capture):
        b = GeomagicTeleopBackend(model, data, deadband_config=DeadbandConfig(enabled=False))
    status = TransportStatus(**{f.name: 0 for f in fields(TransportStatus)})
    b.client = SimpleNamespace(status=lambda: status)
    key = "geomagic_orientation_conditioning_gain"
    assert capture.settings[key] == b.input_metrics()["control_schedule"][key] == 8.
    assert b.pipeline.controller.teleop_command(0).joint_position.shape == (7,)
