"""Hardware-free UI/standalone equivalence, owner lifecycle and data contracts."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from geomagic.standalone.control import StandaloneControl
from geomagic.standalone.device import ForceSender, HapticOwner, OwnerLock
from geomagic.standalone.haptics import ContactResult, FORCE_PACKET, FORCE_N, HAPTIC_MODE
from geomagic.standalone.ui_haptics import UIContactHaptics, haptic_metadata
from geomagic.teleop.synthetic import sample
from geomagic.teleop.validate_gravity_timing import OfflineClient
from geomagic.leader.teleop_backend import GeomagicTeleopBackend
from geomagic.teleop.robot_helpers import MIMIC_ROOT
from runtime.live_runner import MimicLiveRunner, STEPS_PER_DATASET_SAMPLE, DATASET_HZ
from runtime.ui_worker import _physical_geomagic_preflight
from ui.experiment_console.backend.work_adapter.geomagic_device import GeomagicHapticSupervisor, GeomagicDeviceSupervisor
from ui.experiment_console.backend.work_adapter.runtime import WorkRuntime
from geomagic.tests.fake_teleop_geo_owner import send_from_work

ROOT = Path(__file__).resolve().parents[2]
MODES = ('position', 'rotation_z', 'rotation', '6dof')


class MemorySender:
    def __init__(self):
        self.force = np.zeros(3)
        self.closed = False

    def send(self, force, enabled=True, **kw):
        self.force = np.array(force)
        return True

    def close(self):
        self.force[:] = 0
        self.closed = True


def contacts(direction, rendered=True, self_only=False):
    counts = dict(ee_object=0, ee_environment=int(rendered), arm_environment=0,
                  self=int(self_only), other=0)
    return ContactResult(np.array(direction, dtype=float), counts,
                         ({'normal_force_n': 10.},) if rendered else ())


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('case', ['+X', '-X', '+Y', '-Y', '+Z', '-Z', 'multi', 'none',
                                  'self', 'cancel', 'deadman', 'clutch', 'stale', 'invalid'])
def test_same_contact_command_in_both_production_paths(mode, case):
    now = [0.]
    source = OfflineClient()
    control = StandaloneControl(client=source, monotonic=lambda: now[0])
    sender = MemorySender()
    ui = UIContactHaptics(control.backend, control.data, sender, clock=lambda: now[0])
    try:
        control.backend.pipeline.mapper.activation_samples = 2
        control.backend.set_teleop_mode(mode)
        for i in range(8):
            now[0] = i*.001
            source.state = sample(i, time=now[0])
            control.step()
        assert control.backend.teleop_active
        directions = {'+X': (1,0,0), '-X': (-1,0,0), '+Y': (0,1,0),
                      '-Y': (0,-1,0), '+Z': (0,0,1), '-Z': (0,0,-1),
                      'multi': (20,-10,5), 'cancel': (0,0,0), 'invalid': (np.nan,0,0)}
        contact = contacts(directions.get(case, (1,0,0)),
                           rendered=case not in {'none', 'self'}, self_only=case=='self')
        now[0] += .001
        source.state = sample(10, time=now[0], grey=case!='deadman', white=case=='clutch')
        if case == 'stale':
            now[0] += .021  # The accepted sample is held; receive timestamps never go backwards.
        with patch.object(control.selector, 'collect', return_value=contact), \
             patch.object(ui.selector, 'collect', return_value=contact):
            _, standalone_force = control.step()
            ui.step()
        np.testing.assert_array_equal(sender.force, standalone_force)
        if case in directions and case not in {'cancel', 'invalid'}:
            assert np.linalg.norm(sender.force) == pytest.approx(1.)
            # Frozen original transform: Fx_world→Fz_device, Fy→Fx, Fz→Fy.
            d = np.asarray(directions[case], dtype=float)
            np.testing.assert_allclose(sender.force, d[[1,2,0]]/np.linalg.norm(d), atol=1e-15)
        else:
            np.testing.assert_array_equal(sender.force, 0)
        ui.close()
        np.testing.assert_array_equal(sender.force, 0)
    finally:
        control.close()


@pytest.mark.parametrize('mode', MODES)
def test_haptics_do_not_change_any_physics_command_or_learning_sample(mode):
    sources = [OfflineClient(), OfflineClient()]
    now = [0.]
    runners = []
    def initialize(b):
        b.client = sources[len(runners)]
        b._clock = lambda: now[0]
        b.pipeline.mapper.activation_samples = 2
    try:
        with patch.object(GeomagicTeleopBackend, 'initialize', initialize):
            for enabled in (False, True):
                runners.append(MimicLiveRunner(input_backend='geomagic', task='mimicgen_square',
                    image_size=(16,16), haptic_sender=MemorySender() if enabled else None))
        for r in runners:
            r.set_teleop_mode(mode)
        runners[1].haptics.clock = lambda: now[0]
        frames = []
        collected = []
        for tick in range(500):
            now[0] = tick*.001
            if tick % 2 == 0:
                for source in sources:
                    source.state = sample(tick, time=now[0],
                        position=(.015*np.sin(tick*.01), .003, -.002),
                        rotvec=(.1*np.sin(tick*.01), -.03, .07))
            results = [r.step_once(capture_observation=tick%STEPS_PER_DATASET_SAMPLE==0) for r in runners]
            np.testing.assert_array_equal(results[0][0], results[1][0])
            np.testing.assert_array_equal(runners[0].data.qpos, runners[1].data.qpos)
            np.testing.assert_array_equal(runners[0].data.qvel, runners[1].data.qvel)
            if results[0][1] is not None:
                a,b = results[0][1], results[1][1]
                assert a.observation_state.shape == b.observation_state.shape == (16,)
                assert a.action.shape == b.action.shape == (8,)
                np.testing.assert_array_equal(a.observation_state, b.observation_state)
                assert a.control_step == b.control_step == tick
                frames.append(tick)
                collected.append(a)
        from runtime.recording import stack_observations, DatasetCameraRenderer
        import inspect
        assert inspect.signature(DatasetCameraRenderer).parameters['cameras'].default == ('operator', 'wrist')
        stacked = stack_observations(collected)
        assert stacked['observation_state'].shape == (10,16)
        assert stacked['action'].shape == (10,8)
        np.testing.assert_allclose(np.diff(stacked['sim_time']), .05, atol=1e-12)
        for camera in ('operator', 'wrist'):
            assert runners[1].model.camera(camera).id >= 0
        assert frames == list(range(0,500,50)) and DATASET_HZ == 20
        assert runners[0].haptics is None
        for r in runners:
            h = r.backend.pipeline.controller.helpers
            assert (h.POSITION_KP,h.TARGET_ROTATION_KP,h.ORIENTATION_KP)==(8,8,4)
            assert (h.MAX_TARGET_LINEAR_SPEED,h.MAX_TARGET_LINEAR_ACCEL)==(.09,2.)
            assert (h.MAX_TARGET_ANGULAR_SPEED,h.MAX_TARGET_ANGULAR_ACCEL)==(2.,4.)
    finally:
        for r in runners:
            r.close()


@pytest.mark.parametrize('enabled', [False, True])
def test_metadata_is_outside_learning_contract(enabled):
    assert haptic_metadata(enabled) == dict(haptics_enabled=enabled,
        haptic_mode=HAPTIC_MODE if enabled else 'OFF', haptic_force_N=FORCE_N if enabled else 0.)


@pytest.mark.parametrize('leader,source,flag,expected', [
    ('omy',None,'1','none'), ('geomagic','synthetic_udp','1','none'),
    ('geomagic','physical_hdapi','0','off'), ('geomagic','physical_hdapi',None,'on'),
    ('geomagic','physical_hdapi','1','on'), ('geomagic','physical_hdapi','true','off')])
def test_profile_starts_exactly_one_selected_owner(leader, source, flag, expected):
    env={'MIMIC_UI_TASK':'mimicgen_square','MIMIC_UI_LEADER':leader}
    if source: env['MIMIC_UI_GEOMAGIC_SOURCE']=source
    if flag: env['MIMIC_GEOMAGIC_HAPTICS']=flag
    with patch.dict(os.environ,env,clear=True), \
         patch.object(GeomagicDeviceSupervisor,'start') as off, \
         patch.object(GeomagicHapticSupervisor,'start') as on:
        r=WorkRuntime()
        assert off.call_count==int(expected=='off')
        assert on.call_count==int(expected=='on')
        assert r._haptics_enabled == (expected=='on')
        r.close()


def test_conflicting_owner_is_not_started_or_killed(tmp_path):
    with OwnerLock(tmp_path/'lock'):
        supervisor=GeomagicHapticSupervisor(lock_factory=lambda:OwnerLock(tmp_path/'lock'),
            preflight_fn=lambda **kw:pytest.fail('preflight after conflict'),
            owner_factory=lambda *a,**kw:pytest.fail('spawn after conflict'))
        supervisor.start()
        supervisor._thread.join(2)
        assert supervisor.starts==0 and 'lock' in supervisor.error
        supervisor.stop()


def test_spawned_owner_shared_transport_sequence_and_explicit_zero(tmp_path):
    trace=tmp_path/'fake-owner.json'
    owner=HapticOwner(tmp_path,5599,command=[sys.executable,str(ROOT/'geomagic/tests/fake_teleop_geo_owner.py'),
                                          '--trace',str(trace)],sequence_clock=True)
    try:
        owner.start();owner.wait_ready(3)
        senders=[]
        for _ in range(2):
            sender=ForceSender(socket.socket(fileno=os.dup(owner.sender.fileno())),owner.token)
            assert sender.send((0,0,1))
            sender.close()
            senders.append(sender)
        owner.close()

        packets=json.loads(trace.read_text())['packets']
        seq=[x[1] for x in packets]
        assert all(b>a for a,b in zip(seq,seq[1:]))
        assert packets[-1][3:7]==[0.,0.,0.,0]
        assert owner.sent_zero_before_close and owner.process.poll() is not None
    finally:
        owner.close()


def test_unified_supervisor_lifecycle_across_spawned_work_workers(tmp_path):
    import multiprocessing as mp
    trace=tmp_path/'trace.json'
    def factory(runtime, port, **kw):
        return HapticOwner(runtime,port,command=[sys.executable,str(ROOT/'geomagic/tests/fake_teleop_geo_owner.py'),
                                               '--trace',str(trace)],**kw)
    with patch.dict(os.environ,{},clear=True):
        s=GeomagicHapticSupervisor(owner_factory=factory, preflight_fn=lambda **kw:tmp_path,
                                  lock_factory=lambda:OwnerLock(tmp_path/'lock'))
    try:
        s.start()
        deadline=time.monotonic()+3
        while not s.ready()[0] and time.monotonic()<deadline:
            time.sleep(.01)
        assert s.ready()[0], s.snapshot()
        pid=s.pid
        for _ in range(2):
            cfg=s.worker_transport()
            p=mp.get_context('spawn').Process(target=send_from_work,
                args=(cfg['_haptic_socket'],cfg['_haptic_token']))
            p.start();p.join(5)
            assert p.exitcode==0
            assert s.pid==pid and s.starts==1
        s.stop()
        data=json.loads(trace.read_text())
        assert data['force_at_close']==0
        seq=[v[1] for v in data['packets']]
        assert all(b>a for a,b in zip(seq,seq[1:]))
        assert data['packets'][-1][3:7]==[0.,0.,0.,0]
        assert s.pid is None
    finally:
        s.stop()


def test_full_command_buffer_never_waits():
    a,b=socket.socketpair(socket.AF_UNIX,socket.SOCK_DGRAM)
    sender=ForceSender(a,42)
    try:
        a.setsockopt(socket.SOL_SOCKET,socket.SO_SNDBUF,4096)
        start=time.monotonic()
        results=[sender.send((1,0,0)) for _ in range(1000)]
        assert False in results and time.monotonic()-start<.5
    finally:
        sender.close();b.close()


def test_unified_physical_session_is_verified_but_foreign_session_is_rejected():
    state=replace(sample(0,time=time.monotonic()),session_id='teleop-geo-42',
                  source_clock_domain='server.monotonic.hdapi')
    client=SimpleNamespace(poll=lambda:None,latest_state=lambda:state,
                           status=lambda:SimpleNamespace(fresh=True,packets_accepted=1,age_ms=.1))
    b=SimpleNamespace(client=client)
    assert _physical_geomagic_preflight(b,dict(force_output_enabled=True,session_id=state.session_id))['passed']
    with pytest.raises(RuntimeError):
        _physical_geomagic_preflight(b,dict(force_output_enabled=True,session_id='teleop-geo-99'),timeout_s=0)


@pytest.mark.parametrize('extra,enabled,capture', [({},True,False),
    ({'MIMIC_GEOMAGIC_HAPTICS':'0'},False,True),
    ({'MIMIC_GEOMAGIC_HAPTICS':'1'},True,False),
    ({'MIMIC_GEOMAGIC_HAPTICS':'1','GEOMAGIC_SYNC_AUTO':'1'},True,True)])
def test_manual_launcher_flags_without_launching_any_runtime(tmp_path,extra,enabled,capture):
    launcher=tmp_path/'ui_geo_sq';launcher.symlink_to(MIMIC_ROOT/'scripts/ui/run_ui.sh')
    env={k:v for k,v in os.environ.items() if not k.startswith(('MIMIC_GEOMAGIC_HAPTICS','GEOMAGIC_SYNC'))}
    env.update(extra)
    r=subprocess.run([str(launcher),'--canonical','--print-defaults'],env=env,capture_output=True,text=True,check=True)
    cfg=json.loads(r.stdout)
    assert cfg['haptics_enabled'] is enabled
    assert bool(cfg['sync_capture_root']) is capture
