"""Follower constraints, first-update unwind, transitions and Geomagic isolation."""
from dataclasses import replace
from unittest.mock import patch

import mujoco
import numpy as np
import pytest

from geomagic.teleop.anti_windup import AntiWindupConfig, JointCommandAntiWindup
from geomagic.teleop.fr3_target_controller import FR3TargetController
from geomagic.teleop.pose_mapper import CartesianTarget, MODES
from geomagic.teleop.robot_helpers import MIMIC_ROOT
from geomagic.teleop.validate_gravity_timing import OfflineClient
from geomagic.teleop.synthetic import sample
from geomagic.standalone.control import StandaloneControl
from geomagic.teleop.diagnostics import TrackingDiagnostics


LOW, HIGH = np.full(7, -10.), np.full(7, 10.)


@pytest.mark.parametrize('sign', [-1, 1])
def test_normal_following_then_long_block_then_first_update_unwind(sign):
    aw = JointCommandAntiWindup()
    q, actual, old = np.zeros(7), np.zeros(7), np.zeros(7)
    step = np.full(7, sign*.8*.002)
    for _ in range(500):
        actual = q.copy()
        candidate = q + step
        result = aw.apply(q, actual, candidate, LOW, HIGH)
        np.testing.assert_array_equal(result, candidate)
        assert not aw.active
        q = result
    actual = q.copy()
    old = q.copy()
    for _ in range(2000):
        old += step
        q = aw.apply(q, actual, q + step, LOW, HIGH)
        assert np.all(np.abs(q - actual) <= aw.hard + 1e-14)
    assert np.min(np.abs(old - actual)) > 3.
    assert np.all(np.abs(q - actual) > aw.soft)
    before = q.copy()
    result = aw.apply(q, actual, q - step, LOW, HIGH)
    np.testing.assert_array_equal(result, before - step)
    assert not aw.joint_mask.any()  # Reverse is fully accepted on this update.
    assert np.all(np.abs(result - actual) < np.abs(before - actual))


@pytest.mark.parametrize('joint', range(7))
@pytest.mark.parametrize('sign', [-1, 1])
def test_partial_blocking_only_attenuates_the_backlog_increasing_joint(joint, sign):
    aw = JointCommandAntiWindup()
    actual = np.zeros(7)
    previous = np.full(7, .005)
    previous[joint] = sign*(aw.soft[joint]+aw.hard[joint])/2
    delta = np.full(7, .001)
    delta[joint] = sign*.001
    result = aw.apply(previous, actual, previous+delta, LOW, HIGH)
    assert aw.joint_mask.tolist() == [i == joint for i in range(7)]
    np.testing.assert_array_equal(result[np.arange(7)!=joint], (previous+delta)[np.arange(7)!=joint])
    assert result[joint]-previous[joint] == pytest.approx(sign*.0005)


@pytest.mark.parametrize('error_factor', [.5, 1., 1.4, 2., 10.])
@pytest.mark.parametrize('sign', [-1, 1])
def test_unwinding_is_never_attenuated_even_beyond_hard_leash(error_factor, sign):
    aw = JointCommandAntiWindup()
    actual = np.zeros(7)
    previous = sign*aw.soft*error_factor
    step = -sign*np.full(7, .0016)
    result = aw.apply(previous, actual, previous+step, LOW, HIGH)
    assert not aw.joint_mask.any()
    # Emergency clipping may unwind more, but cannot diminish the reverse step.
    assert np.all(sign*(result-previous) <= -np.abs(step)+1e-14)
    assert np.all(np.abs(result-actual) <= aw.hard+1e-14)


def test_hard_leash_handles_actual_moving_away_and_keeps_joint_limits():
    aw = JointCommandAntiWindup()
    previous = np.full(7, .5)
    result = aw.apply(previous, np.zeros(7), previous.copy(), LOW, HIGH)
    np.testing.assert_allclose(result, aw.hard)
    assert aw.hard_leash_hit_count == 7 and aw.active
    actual = HIGH-.01
    candidate = HIGH.copy()
    result = aw.apply(actual, actual, candidate, LOW, HIGH)
    np.testing.assert_array_equal(result, HIGH)
    assert not aw.active


def test_disabled_baseline_is_exact_original_integration():
    aw = JointCommandAntiWindup(AntiWindupConfig(enabled=False))
    previous = np.full(7, 2.)
    candidate = previous+.0016
    assert aw.apply(previous, np.zeros(7), candidate, LOW, HIGH) is candidate
    assert not aw.active


@pytest.mark.parametrize('config', [dict(soft_rad=(0.,)*7), dict(hard_rad=(.01,)*7),
                                    dict(soft_rad=(.1,)*6), dict(hard_rad=(float('nan'),)*7),
                                    dict(enabled=1)])
def test_invalid_config_rejected(config):
    with pytest.raises(ValueError): AntiWindupConfig(**config)


def scene():
    model = mujoco.MjModel.from_xml_path(str(MIMIC_ROOT/'environments/mimicgen_square/scene.xml'))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    mujoco.mj_forward(model, data)
    return model, data


def target(controller, mode='position', **kwargs):
    p, r = controller.actual_pose()
    return replace(CartesianTarget(p, r, mode, True, True, False, False, 'active', True), **kwargs)


@pytest.mark.parametrize('mode', MODES)
def test_shared_real_dls_stage_bounded_for_all_modes_and_one_solve(mode):
    model, data = scene()
    controller = FR3TargetController(model, data)
    value = target(controller, mode)
    controller.compute(value, .002)
    requested = replace(value, position=value.position + np.array([.06, -.04, -.08]))
    if mode == 'rotation_z':
        requested = replace(requested, rotation=value.rotation @ controller.helpers.rotvec_to_matrix(np.array((0., 0., .4))))
    elif mode != 'position':
        requested = replace(requested, rotation=controller.helpers.rotvec_to_matrix(np.array((.2, -.3, .4))) @ value.rotation)
    q_before = data.qpos.copy()
    with patch.object(controller.helpers, 'compute_joint_target', wraps=controller.helpers.compute_joint_target) as solve:
        for _ in range(750):
            result = controller.compute(requested, .002)
            assert np.all(np.abs(result-data.qpos[controller.qpos_indices]) <= controller.anti_windup.hard+1e-14)
        assert solve.call_count == 750
    assert controller.anti_windup.backlog_increasing_block_count > 0
    np.testing.assert_array_equal(data.qpos, q_before)
    # Reverse the conditioned Cartesian request at the DLS boundary. Upstream
    # operator conditioning has its own preserved delay; once DLS asks for
    # unwind, this very first eligible increment must pass without attenuation.
    previous = controller.hold_q_target.copy()
    actual = data.qpos[controller.qpos_indices]
    reverse_p = (value.position if mode in ('rotation_z', 'rotation') else
                 value.position - (requested.position-value.position))
    reverse_r = (value.rotation if mode == 'position' else
                 value.rotation @ requested.rotation.T @ value.rotation)
    candidate, _ = controller.helpers.compute_joint_target(
        model, data, controller.ee_site_id, controller.dof_indices,
        controller.qpos_indices, controller.joint_lower, controller.joint_upper,
        reverse_p, reverse_r, previous, controller.jacp, controller.jacr, .002,
        teleop_mode='full_pose', enable_nullspace_posture=controller.helpers.ENABLE_NULLSPACE_POSTURE,
        q_posture_reference=controller.nullspace_posture_reference)
    reducing = (previous-actual)*(candidate-previous) < 0
    assert reducing.any()
    result = controller.anti_windup.apply(previous, actual, candidate,
                                         controller.joint_lower, controller.joint_upper)
    np.testing.assert_array_equal(result[reducing], candidate[reducing])
    assert not controller.anti_windup.joint_mask[reducing].any()


@pytest.mark.parametrize('edge', ['release', 'reengage', 'clutch', 'stale', 'mode', 'session'])
def test_stronger_actual_state_transition_reset_preserved(edge):
    model, data = scene()
    controller = FR3TargetController(model, data)
    value = target(controller)
    controller.compute(value, .002)
    controller.hold_q_target += np.array([.2, .2, .2, .2, .2, -.2, .2])
    controller.linear_velocity[:] = .03
    controller.angular_velocity[:] = .2
    if edge == 'release': value = replace(value, active=False)
    if edge == 'reengage': controller._enabled = False
    if edge == 'clutch': value = replace(value, clutched=True)
    if edge == 'stale': value = replace(value, fresh=False)
    if edge == 'mode': value = replace(value, mode='6dof')
    if edge == 'session': value = replace(value, synchronized=True)
    result = controller.compute(value, .002)
    np.testing.assert_array_equal(result, data.qpos[controller.qpos_indices])
    np.testing.assert_array_equal(controller.linear_velocity, 0)
    np.testing.assert_array_equal(controller.angular_velocity, 0)
    np.testing.assert_array_equal(controller.target_position, controller.actual_pose()[0])
    assert not controller.anti_windup.active


def test_gravity_servo_offset_does_not_enter_logical_leash():
    now = [0.]
    source = OfflineClient()
    control = StandaloneControl(client=source, monotonic=lambda: now[0])
    try:
        control.backend.pipeline.mapper.activation_samples = 2
        for tick in range(8):
            now[0] = tick*.001
            source.state = sample(tick, time=now[0])
            control.step()
        control.backend.gravity_support.offset[:] = .2  # Deliberately larger than soft limits.
        now[0] += .002
        source.state = sample(10, time=now[0])
        control.backend._last_control_sim_time = float(control.data.time)-.002
        control.backend.update_command(.002)
        controller = control.backend.pipeline.controller
        assert not controller.anti_windup.active
        np.testing.assert_allclose(control.backend.hold_q_target-controller.hold_q_target, .2)
    finally: control.close()


def test_omy_math_remains_unleashed_and_geomagic_hook_is_separate():
    model, data = scene()
    controller = FR3TargetController(model, data)
    h = controller.helpers
    previous = controller.hold_q_target + np.full(7, .3)
    p, r = controller.actual_pose()
    raw, _ = h.compute_joint_target(model, data, controller.ee_site_id,
        controller.dof_indices, controller.qpos_indices, controller.joint_lower,
        controller.joint_upper, p, r, previous, controller.jacp, controller.jacr, .002)
    assert np.max(np.abs(raw-data.qpos[controller.qpos_indices])) > .29
    limited = controller.anti_windup.apply(previous, data.qpos[controller.qpos_indices], raw,
                                           controller.joint_lower, controller.joint_upper)
    assert np.max(np.abs(limited-data.qpos[controller.qpos_indices])) <= .18+1e-14


def test_low_rate_optional_diagnostics_include_logical_error_and_counters():
    now = [0.]
    source = OfflineClient()
    control = StandaloneControl(client=source, monotonic=lambda: now[0])
    diagnostics = TrackingDiagnostics(lightweight=True)
    try:
        control.backend.pipeline.mapper.activation_samples = 2
        for tick in range(8):
            now[0] = tick*.001
            source.state = sample(tick, time=now[0])
            control.step()
        controller = control.backend.pipeline.controller
        controller.anti_windup.apply(controller.hold_q_target+.3,
            control.data.qpos[controller.qpos_indices], controller.hold_q_target+.301,
            controller.joint_lower, controller.joint_upper)
        diagnostics.begin(.001)
        diagnostics.counts['active.control_updates'] = 4
        diagnostics.capture(control.backend.pipeline, controller_updated=True, control_dt=.002)
        result = diagnostics.summary()['anti_windup']
        assert result['anti_windup_active'] and len(result['tracking_error_per_joint']) == 7
        assert len(result['anti_windup_joint_mask']) == 7
        assert result['backlog_increasing_block_count'] == 7
        assert result['hard_leash_hit_count'] > 0
    finally: control.close()


def test_mujoco_contact_can_push_a_movable_object_when_follower_tracks():
    # A minimal physical follower fixture isolates useful contact from the FR3
    # task's reachability. Seven logical commands, with one translating pusher.
    dummy = ''.join(f'<body pos="0 {i+1} 0"><joint name="j{i}" type="slide" axis="1 0 0"/>'
                    '<geom type="sphere" size=".01" mass="1" contype="0" conaffinity="0"/></body>'
                    for i in range(1, 7))
    actuators = ''.join(f'<position joint="j{i}" kp="500" kv="20"/>' for i in range(7))
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><option gravity="0 0 0" timestep=".001" integrator="implicitfast"/>
      <worldbody><body name="pusher"><joint name="j0" type="slide" axis="1 0 0"/>
      <geom type="sphere" size=".05" mass="1"/></body>{dummy}
      <body name="object" pos=".11 0 0"><joint name="object_slide" type="slide" axis="1 0 0" frictionloss=".2"/>
      <geom type="sphere" size=".05" mass=".1"/></body></worldbody>
      <actuator>{actuators}</actuator></mujoco>''')
    data = mujoco.MjData(model)
    aw = JointCommandAntiWindup()
    logical = np.zeros(7)
    contact_ticks = 0
    for tick in range(3000):
        if tick % 2 == 0:
            candidate = logical.copy()
            candidate[0] += .0001
            logical = aw.apply(logical, data.qpos[:7], candidate, LOW, HIGH)
            assert not aw.active
        data.ctrl[:] = logical
        mujoco.mj_step(model, data)
        contact_ticks += int(data.ncon > 0)
    assert contact_ticks > 500
    assert data.qpos[7] > .10  # Actual contacted object was pushed, not frozen.
    assert data.qpos[0] > .14
