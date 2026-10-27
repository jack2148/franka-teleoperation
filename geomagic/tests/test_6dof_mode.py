"""6DoF acceptance with accepted synthetic samples; no device or recording."""
from dataclasses import replace
import itertools
import math
from unittest.mock import patch

import numpy as np
import pytest

from geomagic.teleop.pose_mapper import (
    GeomagicPoseMapper, M_POSITION, MODES, POSITION_SCALE, ROTATION_SCALE,
)
from geomagic.teleop.input_deadband import DeadbandConfig
from geomagic.teleop.pipeline import GeomagicTargetPipeline
from geomagic.teleop.synthetic import sample
from geomagic.teleop.validate_gravity_timing import square_model, reset, make_backend
from geomagic.vendor.geomagic_core.rotations import (
    rotation_vector_to_matrix as Exp, matrix_to_rotation_vector as Log,
    matrix_to_quaternion_xyzw as Quat,
)

P = np.array((.4, .2, .3))
R = Exp((.5, -.7, .4))
D = Exp((-.3, .2, .4))
DEVICE_P = np.array((.03, -.02, .04))
M = np.asarray(M_POSITION)
DP = np.array((.02, .03, .04))
DR = np.array((.2, -.15, .3))


def state(i, dp=(0., 0., 0.), dr=(0., 0., 0.), **kwargs):
    return sample(i, position=DEVICE_P + dp, rotvec=Log(Exp(dr) @ D), **kwargs)


def update(mapper, s, p=P, r=R):
    return mapper.update(s, now=s.timestamp, actual_position=p, actual_rotation=r)


def activated(mode):
    mapper = GeomagicPoseMapper(P, R, mode=mode, activation_samples=2)
    update(mapper, state(0))
    target = update(mapper, state(1))
    assert target.active and target.reanchored
    np.testing.assert_array_equal(target.position, P)
    np.testing.assert_array_equal(target.rotation, R)
    return mapper


@pytest.mark.parametrize('reference', ('0', '1'))
@pytest.mark.parametrize('packets', (False, True))
def test_static_mixed_pose_for_1000_updates_does_not_integrate(monkeypatch, reference, packets):
    monkeypatch.setenv('MIMIC_GEOMAGIC_REFERENCE_MAPPING', reference)
    mapper = activated('6dof')
    fixed_sample = state(2, DP, DR)
    fixed = update(mapper, fixed_sample)
    for i in range(3, 1003):
        incoming = replace(fixed_sample, sequence=i, timestamp=i*.005) if packets else fixed_sample
        target = update(mapper, incoming)
        np.testing.assert_array_equal(target.position, fixed.position)
        np.testing.assert_array_equal(target.rotation, fixed.rotation)


@pytest.mark.parametrize('reference', ('0', '1'))
def test_translation_only_matches_position_and_rotation_only_matches_3d(monkeypatch, reference):
    monkeypatch.setenv('MIMIC_GEOMAGIC_REFERENCE_MAPPING', reference)
    translation = update(activated('6dof'), state(2, DP))
    position = update(activated('position'), state(2, DP))
    np.testing.assert_allclose(translation.position, position.position, atol=1e-14)
    np.testing.assert_allclose(translation.rotation, R, atol=1e-14)
    rotation = update(activated('6dof'), state(2, dr=DR))
    full_3d = update(activated('rotation'), state(2, dr=DR))
    np.testing.assert_allclose(rotation.rotation, full_3d.rotation, atol=1e-14)
    np.testing.assert_array_equal(rotation.position, P)


@pytest.mark.parametrize('reference', ('0', '1'))
def test_mixed_pose_preserves_all_axes_valid_so3_and_existing_scales(monkeypatch, reference):
    monkeypatch.setenv('MIMIC_GEOMAGIC_REFERENCE_MAPPING', reference)
    target = update(activated('6dof'), state(2, DP, DR))
    np.testing.assert_allclose(target.position, P + .7 * (M @ DP), atol=1e-14)
    expected_vector = .4 * np.array((1., -1., 1.)) * (M @ DR)
    np.testing.assert_allclose(target.rotation, Exp(expected_vector) @ R, atol=1e-14)
    assert np.all(np.abs(target.position - P) > 0)
    assert np.all(np.abs(Log(target.rotation @ R.T)) > .01)
    assert np.isfinite(target.position).all() and np.isfinite(target.rotation).all()
    np.testing.assert_allclose(target.rotation.T @ target.rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(target.rotation) == pytest.approx(1., abs=1e-12)
    assert np.linalg.norm(Quat(target.rotation)) == pytest.approx(1., abs=1e-12)
    assert POSITION_SCALE == (.7, .7, .7) and ROTATION_SCALE == .4
    scaled = update(activated('6dof'), state(2, (.1, 0., 0.), (math.radians(10), 0., 0.)))
    assert np.linalg.norm(scaled.position - P) == pytest.approx(.07, abs=1e-14)
    assert np.linalg.norm(Log(scaled.rotation @ R.T)) == pytest.approx(math.radians(4), abs=1e-14)


@pytest.mark.parametrize('channel', ('translation', 'rotation'))
def test_deadband_channels_are_independent_through_noise_motion_and_quiet(channel):
    model, data = square_model()
    reset(model, data)
    config = DeadbandConfig(enabled=True)
    combined = GeomagicTargetPipeline(model, data, mode='6dof', activation_samples=2, deadband_config=config)
    isolated = GeomagicTargetPipeline(model, data, mode='position' if channel=='translation' else 'rotation',
                                    activation_samples=2, deadband_config=config)
    p, r = combined.controller.actual_pose()
    values = []
    for i in range(180):
        if i < 2:
            dp, dr = np.zeros(3), np.zeros(3)
        else:
            # First deliberate motion, then quiet below-OFF noise, then new
            # motion after the gates close. The other channel stays below ON.
            motion = 1. if i < 60 else (1. + .0001*(-1)**i if i < 100 else 1.5)
            dp = DP * motion if channel=='translation' else np.array((.00001*(-1)**i, 0., 0.))
            dr = DR * motion if channel=='rotation' else np.array((.0001*(-1)**i, 0., 0.))
        s = sample(i, position=dp, rotvec=dr)
        both = combined.update_mapping(s, now=s.timestamp)
        single = isolated.update_mapping(s, now=s.timestamp)
        np.testing.assert_allclose(both.position, single.position, atol=1e-14)
        np.testing.assert_allclose(both.rotation, single.rotation, atol=1e-14)
        values.append(both)
    active = getattr(combined.deadband, channel)
    quiet = combined.deadband.rotation if channel=='translation' else combined.deadband.translation
    assert active.activations >= 2
    assert quiet.activations == 0
    if channel=='translation':
        assert np.linalg.norm(values[-1].position - p) > .01
        np.testing.assert_array_equal(values[-1].rotation, r)
    else:
        assert np.linalg.norm(Log(values[-1].rotation @ r.T)) > .01
        np.testing.assert_array_equal(values[-1].position, p)


def test_mixed_deadband_matches_two_independent_channels_and_settles_without_drift():
    model, data = square_model()
    reset(model, data)
    pipelines = {mode: GeomagicTargetPipeline(model, data, mode=mode, activation_samples=2,
                                            deadband_config=DeadbandConfig(enabled=True))
                 for mode in ('6dof', 'position', 'rotation')}
    for i in range(180):
        dp = np.zeros(3) if i<2 else DP*(1. if i<100 else 1.5)
        dr = np.zeros(3) if i<40 else DR*(1. if i<130 else 2.)
        s = sample(i, position=dp, rotvec=dr)
        targets = {mode: pipe.update_mapping(s, now=s.timestamp) for mode, pipe in pipelines.items()}
        np.testing.assert_allclose(targets['6dof'].position, targets['position'].position, atol=1e-14)
        np.testing.assert_allclose(targets['6dof'].rotation, targets['rotation'].rotation, atol=1e-14)
    combined = pipelines['6dof']
    assert combined.deadband.translation.activations >= 2
    assert combined.deadband.rotation.activations >= 2
    fixed = combined.last_target
    for i in range(180, 1180):
        target = combined.update_mapping(replace(s, sequence=i, timestamp=i*.005), now=i*.005)
        np.testing.assert_array_equal(target.position, fixed.position)
        np.testing.assert_array_equal(target.rotation, fixed.rotation)


def transition_measurement(old, new):
    model, data = square_model()
    reset(model, data)
    pipe = GeomagicTargetPipeline(model, data, mode=old, activation_samples=2)
    p, r = pipe.controller.actual_pose()
    q = data.qpos[pipe.controller.qpos_indices].copy()
    for i in range(3):
        s = state(i, DP if i==2 else np.zeros(3), DR if i==2 else np.zeros(3))
        pipe.step(s, now=s.timestamp, dt=.002, gripper_command=128)
    # Command, raw mapper, conditioned target and joint command all lag actual.
    pipe.mapper._position = pipe.controller.target_position = p + (.1, 0., 0.)
    pipe.mapper._rotation = pipe.controller.target_rotation = Exp((math.radians(60), 0., 0.)) @ r
    pipe.controller.hold_q_target = q + .2
    pipe.controller.linear_velocity[:] = .3
    pipe.controller.angular_velocity[:] = .4
    assert pipe.set_mode(new)
    targets = [pipe.last_raw_target, pipe.last_target]
    for i in (3, 4):
        s = state(i, DP * 2, DR * 2)
        pipe.step(s, now=s.timestamp, dt=.002, gripper_command=128)
        targets.extend((pipe.last_raw_target, pipe.last_target))
        np.testing.assert_array_equal(pipe.controller.target_position, p)
        np.testing.assert_array_equal(pipe.controller.target_rotation, r)
        np.testing.assert_array_equal(pipe.controller.hold_q_target, q)
        np.testing.assert_array_equal(pipe.controller.linear_velocity, 0)
        np.testing.assert_array_equal(pipe.controller.angular_velocity, 0)
    assert pipe.last_target.reanchored
    return {'transition': old + ' -> ' + new,
            'position_jump_m': max(float(np.linalg.norm(t.position-p)) for t in targets),
            'orientation_jump_rad': max(float(np.linalg.norm(Log(t.rotation@r.T))) for t in targets)}


@pytest.mark.parametrize('old,new', list(itertools.permutations(MODES, 2)))
def test_every_mode_switch_discards_pose_and_joint_backlog(old, new):
    measurement = transition_measurement(old, new)
    assert measurement['position_jump_m'] < 1e-12
    assert measurement['orientation_jump_rad'] < 1e-12


@pytest.mark.parametrize('edge', ('grey', 'white'))
def test_deadman_and_clutch_reanchor_both_channels_and_keep_gravity_hold(edge):
    model, data = square_model()
    reset(model, data)
    backend, now = make_backend(model, data, mode='6dof')
    def drive(i, **opts):
        now[0] = i*.005
        data.time = now[0]
        backend.client.state = state(i, **opts)
        backend.update_command(.001)
        return backend.pipeline.last_target
    drive(0)
    drive(1)
    moving = drive(2, dp=DP, dr=DR)
    controller = backend.pipeline.controller
    p, r = controller.actual_pose()
    q = data.qpos[controller.qpos_indices].copy()
    assert np.linalg.norm(moving.position-p) > .01
    assert np.linalg.norm(Log(moving.rotation@r.T)) > .01
    paused = drive(3, dp=DP, dr=DR, **({'grey':False} if edge=='grey' else {'white':True}))
    assert not backend.teleop_active
    np.testing.assert_array_equal(paused.position, p)
    np.testing.assert_array_equal(paused.rotation, r)
    np.testing.assert_array_equal(controller.hold_q_target, q)
    np.testing.assert_array_equal(backend.hold_q_target, backend.gravity_support.command(q))
    drive(4, dp=DP*3, dr=DR*3, **({'grey':False} if edge=='grey' else {'white':True}))
    resumed = drive(5, dp=DP*3, dr=DR*3)
    assert resumed.active and resumed.reanchored
    np.testing.assert_array_equal(resumed.position, p)
    np.testing.assert_array_equal(resumed.rotation, r)
    np.testing.assert_array_equal(controller.hold_q_target, q)
    next_target = drive(6, dp=DP*4, dr=DR*4)
    np.testing.assert_allclose(next_target.position, p + .7*(M@DP), atol=1e-14)
    expected_dr = Log(Exp(DR*4) @ Exp(DR*3).T)
    np.testing.assert_allclose(next_target.rotation, Exp(.4*np.array((1.,-1.,1.))*(M@expected_dr))@r, atol=1e-14)


def test_combined_target_uses_one_existing_conditioner_and_one_full_pose_dls():
    model, data = square_model()
    reset(model, data)
    pipe = GeomagicTargetPipeline(model, data, mode='6dof', activation_samples=2)
    for i in (0, 1):
        s = sample(i)
        pipe.step(s, now=s.timestamp, dt=.002, gripper_command=64)
    h = pipe.controller.helpers
    s = sample(2, position=DP, rotvec=DR)
    with patch.object(h, 'condition_target', wraps=h.condition_target) as conditioner, \
         patch.object(h, 'compute_joint_target', wraps=h.compute_joint_target) as dls:
        command = pipe.step(s, now=s.timestamp, dt=.002, gripper_command=64)
    assert conditioner.call_count == dls.call_count == 1
    np.testing.assert_array_equal(conditioner.call_args.args[2], pipe.last_target.position)
    np.testing.assert_array_equal(conditioner.call_args.args[3], pipe.last_target.rotation)
    assert dls.call_args.kwargs['teleop_mode'] == 'full_pose'
    assert command.joint_position.shape == (7,) and np.isfinite(command.joint_position).all()
    assert command.gripper_command == 64
