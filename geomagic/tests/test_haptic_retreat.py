"""Offline release semantics and device/world sign checks; no HDAPI or viewer."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from geomagic.standalone.haptic_config import RetreatConfig
from geomagic.standalone.haptics import ContactResult, RetreatRelease, contact_force
from geomagic.standalone.control import StandaloneControl
from geomagic.standalone.ui_haptics import UIContactHaptics, haptic_status, haptic_metadata
from geomagic.teleop.pose_mapper import M_POSITION
from geomagic.teleop.synthetic import sample
from geomagic.teleop.validate_gravity_timing import OfflineClient
from geomagic.leader.teleop_backend import GeomagicTeleopBackend
from runtime.live_runner import MimicLiveRunner


def contact(world=(0, 0, 1), ids=((1, 2),), *, cancelled=False):
    return ContactResult(np.zeros(3) if cancelled else np.array(world, dtype=float),
                         dict(ee_environment=len(ids)),
                         tuple(dict(normal_force_n=10., body_pair=('fr3_hand', str(pair)))
                               for pair in ids), contact_ids=ids)


NONE = contact(ids=())


def physical(sequence, stamp, position=(0, 0, 0), **kw):
    return replace(sample(sequence, time=stamp, position=position, **kw),
                   source_timestamp=stamp, source_clock_domain='server.monotonic.hdapi')


class Harness:
    def __init__(self, config=None):
        self.release = RetreatRelease(config)
        self.data = SimpleNamespace(qpos=np.zeros(7), qvel=np.zeros(7), ctrl=np.zeros(7))
        self.backend = SimpleNamespace(_latest_state=None, input_fresh=True,
                                       teleop_active=True, sync=None)

    def step(self, state, contacts=None, *, now=None, enabled=True):
        self.backend._latest_state = state
        self.backend.teleop_active = bool(state and state.grey_button)
        return contact_force(self.backend, self.data, contacts if contacts is not None else contact(),
                             now=state.timestamp if now is None else now,
                             enabled=enabled, release=self.release)


AXES = [(axis, sign) for axis in range(3) for sign in (-1, 1)]


@pytest.mark.parametrize('axis,sign', AXES)
@pytest.mark.parametrize('motion', ['inward', 'stationary', 'retreat', 'noise', 'tangent'])
def test_all_six_world_axes_use_device_velocity_and_device_force(axis, sign, motion):
    world = np.eye(3)[axis] * sign
    direction = np.asarray(M_POSITION).T @ world
    speed = {'inward': -.01, 'stationary': 0., 'retreat': .01, 'noise': .004,
             'tangent': 0.}[motion]
    velocity = direction * speed
    if motion == 'tangent':
        velocity = np.roll(direction, 1) * .02
    h = Harness()
    contacts = contact(world)
    np.testing.assert_allclose(h.step(physical(0, 0), contacts), direction)
    for i in range(1, 9):
        stamp = i * .002
        force = h.step(physical(i, stamp, position=velocity * stamp), contacts)
        if motion == 'retreat' and i == 8:
            np.testing.assert_array_equal(force, 0)
            assert h.release.retreat_detected and h.release.retreat_suppressed
        else:
            np.testing.assert_allclose(force, direction)
    assert h.release.retreat_score_mps == pytest.approx(speed)
    assert contacts.renderable  # Actual contact is never falsified by release.


def suppress(h, contacts=None):
    contacts = contacts if contacts is not None else contact()
    direction = np.asarray(M_POSITION).T @ (contacts.direction_world /
                                           np.linalg.norm(contacts.direction_world))
    for i in range(9):
        h.step(physical(i, i*.002, position=direction * .01 * i*.002), contacts)
    assert h.release.retreat_suppressed
    return direction * .00016


def test_latch_survives_stationary_inward_deadman_clutch_and_contact_end_rearms():
    h = Harness()
    last = suppress(h)
    episode = h.release.contact_episode_id
    for i, kwargs in enumerate(({}, dict(grey=False), {}, dict(white=True), {}), 9):
        np.testing.assert_array_equal(h.step(physical(i, i*.002, position=last, **kwargs)), 0)
        assert h.release.retreat_suppressed and h.release.contact_episode_id == episode
    np.testing.assert_array_equal(h.step(physical(14, .028, position=last - .0001)), 0)
    np.testing.assert_array_equal(h.step(physical(15, .030), NONE), 0)
    assert not h.release.contact_active and not h.release.retreat_suppressed
    np.testing.assert_allclose(h.step(physical(16, .032)), (0, 1, 0))
    assert h.release.contact_episode_id == episode + 1


def test_short_retreat_and_confirmation_interruption_do_not_release():
    h = Harness()
    for i in range(6):  # 10 ms retreat, shorter than 15 ms.
        h.step(physical(i, i*.002, (0, i*.00002, 0)))
    for i in range(6, 9):
        force = h.step(physical(i, i*.002, (0, .0001, 0)))
        assert np.linalg.norm(force) == pytest.approx(1.)
    for i in range(9, 15):  # A separate 12 ms retreat must not accumulate.
        force = h.step(physical(i, i*.002, (0, .0001+(i-8)*.00002, 0)))
        assert np.linalg.norm(force) == pytest.approx(1.)
    assert not h.release.retreat_suppressed


def test_held_sample_cannot_confirm_retreat_with_wall_clock_alone():
    h = Harness()
    h.step(physical(0, 0))
    state = physical(1, .002, (0, .00002, 0))
    h.step(state)
    for now in (.005, .010, .018, .021):
        assert np.linalg.norm(h.step(state, now=now)) == pytest.approx(1.)
        assert not h.release.retreat_detected and not h.release.retreat_suppressed


@pytest.mark.parametrize('gate', ['deadman', 'clutch', 'receive_stale', 'source_stale',
                                  'invalid', 'input_stale', 'disabled'])
def test_existing_safety_gates_remain_zero_and_break_confirmation(gate):
    h = Harness()
    for i in range(6):
        h.step(physical(i, i*.002, (0, i*.00002, 0)))
    state = physical(6, .012, (0, .00012, 0))
    now, enabled = .012, True
    if gate == 'deadman': state = replace(state, buttons=(False, False))
    if gate == 'clutch': state = replace(state, buttons=(True, True))
    if gate == 'receive_stale': now = .033
    if gate == 'source_stale': state = replace(state, source_timestamp=-.010)
    if gate == 'invalid': state = replace(state, valid=False)
    if gate == 'input_stale': h.backend.input_fresh = False
    if gate == 'disabled': enabled = False
    np.testing.assert_array_equal(h.step(state, now=now, enabled=enabled), 0)
    assert not h.release.retreat_suppressed
    h.backend.input_fresh = True
    for i in range(7, 13):
        assert np.linalg.norm(h.step(physical(i, i*.002, (0, i*.00002, 0)))) == pytest.approx(1.)


@pytest.mark.parametrize('discontinuity', ['session', 'clock', 'gap', 'backwards', 'sequence'])
def test_sample_discontinuities_cannot_trigger_retreat(discontinuity):
    h = Harness()
    for i in range(6):
        h.step(physical(i, i*.002, (0, i*.00002, 0)))
    state = physical(6, .012, (0, .01, 0))
    if discontinuity == 'session': state = replace(state, session_id='new')
    if discontinuity == 'clock': state = replace(state, source_clock_domain='other')
    if discontinuity == 'gap': state = replace(state, timestamp=.050, source_timestamp=.050)
    if discontinuity == 'backwards': state = replace(state, source_timestamp=.009)
    if discontinuity == 'sequence': state = replace(state, sequence=4)
    assert np.linalg.norm(h.step(state)) == pytest.approx(1.)
    assert not h.release.retreat_detected


def test_acquisition_clock_avoids_receive_burst_velocity_and_uses_physical_position():
    h = Harness()
    h.step(physical(0, 0))
    for i in range(1, 9):
        state = replace(physical(i, i*.002, (0, i*.000008, 0)), timestamp=.017+i*.0001)
        assert np.linalg.norm(h.step(state)) == pytest.approx(1.)
    assert h.release.retreat_score_mps == pytest.approx(.004)
    assert not h.release.retreat_suppressed


def test_new_geom_identity_rearms_while_old_contact_persists():
    h = Harness()
    last = suppress(h)
    old_episode = h.release.contact_episode_id
    new = contact(world=(1, 0, 0), ids=((1, 2), (3, 4)))
    np.testing.assert_allclose(h.step(physical(9, .018, last), new), (0, 0, 1))
    assert h.release.contact_episode_id == old_episode + 1
    assert not h.release.retreat_suppressed


def test_cancellation_zero_normal_force_and_direction_change_do_not_rearm_same_contact():
    h = Harness()
    last = suppress(h, contact(ids=((1, 2), (3, 4))))
    episode = h.release.contact_episode_id
    for i, current in enumerate((contact(cancelled=True, ids=((1, 2), (3, 4))),
                                replace(contact(), selected=()),
                                contact(world=(-1, 0, 0)), contact()), 9):
        np.testing.assert_array_equal(h.step(physical(i, i*.002, last), current), 0)
        assert h.release.contact_episode_id == episode
        assert h.release.retreat_suppressed


def test_retreat_uses_weighted_net_direction_for_multiple_contacts():
    h = Harness()
    net = contact(world=(2, -1, 3), ids=((1, 2), (3, 4)))
    suppress(h, net)
    assert h.release.retreat_score_mps == pytest.approx(.01)


def test_contact_ids_come_from_true_mujoco_contacts_even_at_zero_normal_force():
    import mujoco
    from geomagic.standalone.haptics import ContactSelector
    from geomagic.tests.test_standalone_geo import contact_model
    model = contact_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    selector = ContactSelector(model)
    original = selector.collect(data)
    assert original.contact_ids and original.renderable
    before = (data.ncon, data.qpos.copy(), data.qvel.copy())
    def zero_contact_force(model, data, index, output):
        output[:] = 0
    with patch('geomagic.standalone.haptics.mujoco.mj_contactForce', side_effect=zero_contact_force):
        zero = selector.collect(data)
    assert zero.contact_ids == original.contact_ids
    assert zero.counts == original.counts and not zero.renderable
    h = Harness()
    suppress(h, original)
    np.testing.assert_array_equal(h.step(physical(9, .018), zero), 0)
    assert h.release.contact_active and h.release.retreat_suppressed
    assert data.ncon == before[0]
    np.testing.assert_array_equal(data.qpos, before[1])
    np.testing.assert_array_equal(data.qvel, before[2])


def test_diagnostics_keep_contact_true_while_force_is_zero_and_metadata_stays_fixed():
    h = Harness()
    suppress(h)
    status = haptic_status(True, contacts=contact(), force=np.zeros(3), release=h.release)
    assert status['contact_active'] and status['renderable_contact']
    assert status['retreat_detected'] and status['retreat_suppressed']
    assert status['haptic_force_N'] == status['force_target_n'] == 0.
    assert status['contact_episode_id'] == 1
    assert haptic_metadata(True)['haptic_force_N'] == 1.  # Existing episode metadata.
    h.backend.sync = SimpleNamespace(preview_context=lambda: {})
    trace = {}
    contact_force(h.backend, h.data, contact(), enabled=True, now=.016,
                  release=h.release, diagnostics=trace)
    assert trace['contact'] and trace['contact_active'] and trace['retreat_suppressed']
    assert trace['device_force'] == [0., 0., 0.] and trace['haptic_force_N'] == 0.


@pytest.mark.parametrize('config', [dict(speed_threshold_mps=0), dict(confirm_ms=-1),
                                    dict(confirm_ms=float('nan')), dict(speed_threshold_mps=True)])
def test_invalid_configuration_is_rejected(config):
    with pytest.raises(ValueError): RetreatConfig(**config)


def test_configurable_threshold_and_confirmation():
    h = Harness(RetreatConfig(speed_threshold_mps=.02, confirm_ms=10))
    for i in range(9):
        assert np.linalg.norm(h.step(physical(i, i*.002, (0, i*.00002, 0)))) == pytest.approx(1.)
    for i in range(9, 14):
        force = h.step(physical(i, i*.002, (0, .00016+(i-8)*.00006, 0)))
    np.testing.assert_array_equal(force, 0)


@pytest.mark.parametrize('mode', ['position', 'rotation_z', 'rotation', '6dof'])
def test_both_launch_paths_share_confirm_latch_and_reset_in_every_mode(mode):
    now = [0.]
    source = OfflineClient()
    control = StandaloneControl(client=source, monotonic=lambda: now[0])
    sender = SimpleNamespace(send=lambda *a, **kw: True, close=lambda: None)
    ui = UIContactHaptics(control.backend, control.data, sender, clock=lambda: now[0])
    current = [contact()]
    try:
        control.backend.pipeline.mapper.activation_samples = 2
        control.backend.set_teleop_mode(mode)
        with patch.object(control.selector, 'collect', side_effect=lambda data: current[0]), \
             patch.object(ui.selector, 'collect', side_effect=lambda data: current[0]):
            for tick in range(50):
                now[0] = tick*.001
                if tick == 35: current[0] = NONE
                if tick == 38: current[0] = contact()
                position = (0, max(0, min(tick-8, 18))*.00001, 0)
                if tick % 2 == 0:
                    source.state = physical(tick, now[0], position)
                _, force = control.step()
                ui.step()
                np.testing.assert_array_equal(force, ui.force)
                assert control.haptic_release.diagnostics() == ui.haptic_release.diagnostics()
                if 26 <= tick < 35:
                    np.testing.assert_array_equal(force, 0)
                    assert ui.haptic_release.retreat_suppressed
                if tick >= 38:
                    assert np.linalg.norm(force) == pytest.approx(1.)
        assert control.haptic_release.contact_episode_id == 2
    finally:
        ui.close()
        control.close()


@pytest.mark.parametrize('mode', ['position', 'rotation_z', 'rotation', '6dof'])
def test_suppression_does_not_change_physics_actions_or_observations(mode):
    sources = [OfflineClient(), OfflineClient()]
    runners = []
    now = [0.]
    def initialize(backend):
        backend.client = sources[len(runners)]
        backend._clock = lambda: now[0]
        backend.pipeline.mapper.activation_samples = 2
    sender = SimpleNamespace(send=lambda *a, **kw: True, close=lambda: None)
    current = [contact()]
    suppressed_ticks = 0
    try:
        with patch.object(GeomagicTeleopBackend, 'initialize', initialize):
            for enabled in (False, True):
                runners.append(MimicLiveRunner(input_backend='geomagic', task='mimicgen_square',
                    image_size=(16, 16), haptic_sender=sender if enabled else None))
        for runner in runners: runner.set_teleop_mode(mode)
        haptics = runners[1].haptics
        haptics.clock = lambda: now[0]
        with patch.object(haptics.selector, 'collect', side_effect=lambda data: current[0]):
            for tick in range(160):
                now[0] = tick*.001
                if tick == 80: current[0] = NONE
                if tick == 100: current[0] = contact()
                if tick % 2 == 0:
                    for source in sources:
                        source.state = physical(tick, now[0], (0, min(tick, 40)*.00001, 0))
                left, right = [r.step_once(capture_observation=tick%50 == 0) for r in runners]
                np.testing.assert_array_equal(left[0], right[0])
                np.testing.assert_array_equal(runners[0].data.qpos, runners[1].data.qpos)
                np.testing.assert_array_equal(runners[0].data.qvel, runners[1].data.qvel)
                np.testing.assert_array_equal(runners[0].data.ctrl, runners[1].data.ctrl)
                if left[1] is not None:
                    assert left[1].observation_state.shape == right[1].observation_state.shape == (16,)
                    assert left[1].action.shape == right[1].action.shape == (8,)
                    np.testing.assert_array_equal(left[1].observation_state, right[1].observation_state)
                    np.testing.assert_array_equal(left[1].action, right[1].action)
                suppressed_ticks += int(haptics.haptic_release.retreat_suppressed)
                if 40 <= tick < 80:
                    assert haptics.haptic_release.retreat_suppressed
                    np.testing.assert_array_equal(haptics.force, 0)
                if tick >= 100:
                    assert not haptics.haptic_release.retreat_suppressed
                    assert np.linalg.norm(haptics.force) == pytest.approx(1.)
        assert suppressed_ticks > 40
    finally:
        for runner in runners: runner.close()
