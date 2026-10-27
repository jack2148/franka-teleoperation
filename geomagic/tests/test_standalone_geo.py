"""Standalone contract, physical contact signs, watchdog and fake process tests.

No test initializes HDAPI, opens Touch, displays a viewer, or writes a dataset.
"""
import contextlib
from dataclasses import replace
import io
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
import pytest

from geomagic.standalone.control import StandaloneControl, KEY_MODES, config_summary
from geomagic.standalone.haptics import (
    ContactSelector, ContactResult, FORCE_N, FORCE_PACKET, FORCE_MAGIC, fixed_device_force, force_packet,
)
from geomagic.standalone.device import HapticOwner, OwnerLock, other_owners, preflight
from geomagic.standalone.viewer import ViewerBridge, GlfwRenderer, _viewer_main
from geomagic.standalone.app import main, run_session
from geomagic.teleop.synthetic import sample
from geomagic.teleop.validate_gravity_timing import OfflineClient

ROOT = Path(__file__).resolve().parents[2]


def test_current_config_and_safe_config_only_entry():
    cfg = config_summary()
    assert cfg['position_scale'] == [.7, .7, .7]
    assert cfg['rotation_scale'] == .4
    assert KEY_MODES == {49: 'position', 50: 'rotation_z', 51: 'rotation', 52: '6dof'}
    assert cfg['haptics_default'] and not cfg['recording'] and cfg['collection_plan'] is None
    with patch('geomagic.standalone.app.preflight', side_effect=AssertionError('must not preflight')), contextlib.redirect_stdout(io.StringIO()) as output:
        assert main(['--config']) == 0
    assert json.loads(output.getvalue())['force_n'] == 1


@pytest.mark.parametrize('world,device', [
    ((1,0,0),(0,0,1)), ((-1,0,0),(0,0,-1)),
    ((0,1,0),(1,0,0)), ((0,-1,0),(-1,0,0)),
    ((0,0,1),(0,1,0)), ((0,0,-1),(0,-1,0)),
])
def test_virtual_work_force_axes(world, device):
    force = fixed_device_force(np.asarray(world)*123, contact=True, deadman=True)
    np.testing.assert_array_equal(force, device)
    assert np.linalg.norm(force) == 1
    displacement = np.array([.2,-.3,.4])
    world_displacement = np.array([displacement[2],displacement[0],displacement[1]])
    assert np.dot(force, displacement) == pytest.approx(np.dot(world,world_displacement))


@pytest.mark.parametrize('gates', [
    {'contact': False}, {'deadman': False}, {'fresh': False},
    {'state_valid': False}, {'enabled': False}, {'clutched': True},
])
def test_force_is_zero_at_every_host_safety_gate(gates):
    params = {'contact': True, 'deadman': True, **gates}
    np.testing.assert_array_equal(fixed_device_force((1,2,3), **params), 0)


@pytest.mark.parametrize('direction', [
    (0,0,0), (1e-16,0,0), (math.nan,0,0), (math.inf,0,0),
    (1e308,1e308,1e308), (1,2), ((1,2,3),),
])
def test_invalid_or_zero_direction_fails_closed(direction):
    np.testing.assert_array_equal(fixed_device_force(direction, contact=True, deadman=True), 0)


def test_private_force_packet_abi_cap_and_explicit_zero():
    fields = FORCE_PACKET.unpack(force_packet(42,7,1.5,(2,0,0),enabled=True))
    assert FORCE_PACKET.size == 64
    assert fields[:4] == (FORCE_MAGIC,42,7,1.5)
    assert fields[4:] == (1.,0.,0.,1,0)
    fields = FORCE_PACKET.unpack(force_packet(42,8,1.5,(math.nan,0,0),enabled=True))
    assert fields[4:] == (0.,0.,0.,0,0)


def contact_model(direction=1):
    # Native MuJoCo plane is +Z; rotate to +/-X. The dynamic hand sphere
    # penetrates 1 cm from the allowed side: reaction must point away.
    return mujoco.MjModel.from_xml_string(f'''
      <mujoco><option gravity="0 0 0" timestep="0.001"/>
      <worldbody>
        <geom name="obstacle" type="plane" size="2 2 .1" quat=".7071067812 0 {direction*.7071067812} 0"/>
        <body name="base" pos="{direction*.04} 0 0"><freejoint/>
          <body name="fr3_link0"><body name="fr3_link5"><body name="fr3_link7">
            <body name="fr3_hand"><geom type="sphere" size=".05" mass="1"/>
              <body name="fr3_left_finger"/><body name="fr3_right_finger"/>
            </body></body></body></body>
        </body><body name="square_nut" pos="{direction*2} 0 0"><site size=".001"/></body>
        <body name="square_peg" pos="{direction*3} 0 0"><geom type="sphere" size=".001"/></body>
      </worldbody></mujoco>''')


@pytest.mark.parametrize('side', [-1,1])
def test_actual_mujoco_plane_reaction_has_restoring_sign(side):
    model = contact_model(side)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    result = ContactSelector(model).collect(data)
    assert result.counts['ee_environment'] > 0 and result.renderable
    assert json.loads(json.dumps({'renderable_contact':result.renderable}))['renderable_contact'] is True
    force = fixed_device_force(result.direction_world,contact=True,deadman=True)
    np.testing.assert_allclose(force,(0,0,side),atol=1e-12)
    assert result.selected[0]['normal_force_n'] > 0


def test_contact_classes_include_peg_link5_and_self_separately():
    model = contact_model()
    selector = ContactSelector(model)
    hand = model.body('fr3_hand').id
    finger = model.body('fr3_left_finger').id
    nut = model.body('square_nut').id
    peg = model.body('square_peg').id
    link5 = model.body('fr3_link5').id
    assert selector.classify(hand,nut) == selector.classify(nut,finger) == 'ee_object'
    assert selector.classify(hand,0) == selector.classify(peg,finger) == 'ee_environment'
    assert selector.classify(link5,peg) == 'arm_environment'
    assert selector.classify(link5,finger) == 'self'
    assert selector.classify(nut,peg) == 'other'


@pytest.mark.parametrize('robot_first', [False,True])
def test_geom_order_does_not_invert_force_on_robot(robot_first):
    model = contact_model()
    selector = ContactSelector(model)
    hand_geom = next(i for i,b in enumerate(model.geom_bodyid) if b==model.body('fr3_hand').id)
    obstacle = model.geom('obstacle').id
    pair = (hand_geom,obstacle) if robot_first else (obstacle,hand_geom)
    # Normal always points geom0 -> geom1; reaction on hand is +X.
    contact = SimpleNamespace(geom=pair,frame=np.array([-1 if robot_first else 1,0,0,0,0,0,0,0,0]))
    data = SimpleNamespace(ncon=1,contact=[contact])
    def wrench(model,data,index,out): out[:]=[25,0,0,0,0,0]
    with patch('geomagic.standalone.haptics.mujoco.mj_contactForce',wrench): result=selector.collect(data)
    assert result.renderable
    np.testing.assert_array_equal(fixed_device_force(result.direction_world,contact=True,deadman=True),(0,0,1))


def test_self_and_arm_contacts_do_not_render_or_hide_diagnostics():
    model = contact_model()
    selector = ContactSelector(model)
    # Classification is independent of force extraction. Native fixture has
    # only hand/plane contact; classify the body pairs for nonrendering cases.
    assert selector.classify(model.body('fr3_link5').id,model.body('fr3_hand').id)=='self'
    assert selector.classify(model.body('fr3_link5').id,0)=='arm_environment'
    with patch.object(selector,'classify',return_value='self'):
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        result=selector.collect(data)
    assert result.counts['self']>0
    assert not result.renderable
    np.testing.assert_array_equal(result.direction_world,0)


def test_cancelled_or_invalid_contact_normals_produce_zero():
    model=contact_model();selector=ContactSelector(model)
    hand=next(i for i,b in enumerate(model.geom_bodyid) if b==model.body('fr3_hand').id)
    plane=model.geom('obstacle').id
    contacts=[SimpleNamespace(geom=(plane,hand),frame=np.r_[normal,np.zeros(6)]) for normal in ((1,0,0),(-1,0,0))]
    data=SimpleNamespace(ncon=2,contact=contacts)
    def wrench(model,data,index,out):out[:]=[10,0,0,0,0,0]
    with patch('geomagic.standalone.haptics.mujoco.mj_contactForce',wrench):
        result=selector.collect(data)
        assert not result.renderable
        np.testing.assert_array_equal(result.direction_world,0)
        contacts[1].frame[0]=math.nan
        result=selector.collect(data)
        assert result.invalid and not result.renderable
        np.testing.assert_array_equal(result.direction_world,0)


@pytest.mark.parametrize('mode', ['position','rotation_z','rotation','6dof'])
def test_standalone_reuses_exact_backend_multirate_and_no_recording(mode):
    client=OfflineClient();clock=[0.]
    control=StandaloneControl(client=client,monotonic=lambda:clock[0])
    try:
        control.backend.pipeline.mapper.activation_samples=2
        control.key(next(key for key, value in KEY_MODES.items() if value == mode))
        assert control.backend.pipeline.mapper.mode == mode
        for tick in range(151):
            clock[0]=tick*.001
            if tick%2==0:client.state=sample(tick,time=clock[0],
                position=(.01,.02,.03) if mode=='6dof' and tick>=4 else (0,0,0),
                rotvec=(.1,-.2,.3) if mode=='6dof' and tick>=4 else (0,0,0))
            action,force=control.step()
            assert action.shape==(8,)
            assert np.linalg.norm(force)<=1
        assert control.ticks==151
        assert control.data.time==pytest.approx(.151,abs=1e-12)
        assert control.backend.control_updates==74
        assert control.backend.last_control_dt==pytest.approx(.002,abs=1e-12)
        assert not hasattr(control,'recorder')
        control.key(32);assert control.backend.gripper_command==255
        control.key(32);assert control.backend.gripper_command==0
        control.key(72);assert not control.haptics_enabled
    finally: control.close()


def test_viewer_snapshot_cannot_block_control_when_copy_lock_busy():
    client=OfflineClient();control=StandaloneControl(client=client)
    viewer=ViewerBridge(control.model,control.xml_path)
    try:
        viewer.lock.acquire()
        started=time.monotonic()
        assert viewer.publish(control.data) is False
        assert time.monotonic()-started < .02
        viewer.lock.release()
        assert viewer.publish(control.data)
        np.testing.assert_array_equal(np.frombuffer(viewer.values)[:control.model.nq],control.data.qpos)
    finally:
        viewer.close();control.close()


def test_haptics_drop_on_20ms_state_or_source_loss_before_unchanged_teleop_stale_gate():
    client=OfflineClient();clock=[0.]
    control=StandaloneControl(client=client,monotonic=lambda:clock[0])
    contact=ContactResult(np.array([1.,0.,0.]),{'ee_environment':1},
        ({'normal_force_n':10,'body_pair':('fr3_hand','table')},))
    try:
        control.backend.pipeline.mapper.activation_samples=2
        with patch.object(control.selector,'collect',return_value=contact):
            for i in range(6):
                clock[0]=i*.001
                if i%2==0:client.state=sample(i,time=clock[0])
                _,force=control.step()
            assert np.linalg.norm(force)==1
            clock[0]=.030
            _,force=control.step()
            assert control.backend.input_fresh  # Production 50 ms gate preserved.
            np.testing.assert_array_equal(force,0)
            # A freshly received but delayed local HDAPI packet also fails
            # haptic freshness instead of making an old pose fresh again.
            clock[0]=.031
            client.state=replace(sample(8,time=clock[0]),source_timestamp=.001,
                source_clock_domain='server.monotonic.hdapi')
            _,force=control.step()
            assert control.backend.input_fresh
            np.testing.assert_array_equal(force,0)
    finally:control.close()


def test_owner_lock_rejects_second_launcher_without_killing_anything(tmp_path):
    with OwnerLock(tmp_path/'lock'):
        with pytest.raises(RuntimeError,match='owner lock'):
            with OwnerLock(tmp_path/'lock'): pass


def test_named_or_generic_openhaptics_owner_is_detected(tmp_path):
    named=tmp_path/'12';named.mkdir();(named/'cmdline').write_bytes(b'/fake/hdapi_state_stream\0')
    generic=tmp_path/'13';generic.mkdir();(generic/'cmdline').write_bytes(b'/usr/bin/python\0app.py\0')
    (generic/'maps').write_text('/fake/libHD.so.3.4.0')
    owners=other_owners(tmp_path)
    assert {p for p,_ in owners}=={12,13}


def fake_preflight_files(tmp_path):
    proc=tmp_path/'proc';proc.mkdir()
    usb=tmp_path/'usb';usb.mkdir();device=usb/'touch';device.mkdir()
    (device/'idVendor').write_text('2988');(device/'idProduct').write_text('0302')
    runtime=tmp_path/'runtime'
    for item in ('include/HD/hd.h','lib/libHD.so','lib/libHDU.a'):
        path=runtime/item;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fake')
    (runtime/'3DSystems').mkdir()
    binary=tmp_path/'binary';binary.write_text('fake');binary.chmod(0o700)
    return dict(runtime=runtime,sys_root=usb,proc_root=proc,tty=Path('/dev/null'),binary=binary)


def test_preflight_checks_runtime_device_libraries_and_udp_without_starting(tmp_path):
    files=fake_preflight_files(tmp_path)
    with patch('geomagic.standalone.device.subprocess.run',return_value=SimpleNamespace(returncode=0,stdout='libHD => local',stderr='')):
        assert preflight(state_port=0,**files)==files['runtime']
        (files['runtime']/'lib/libHD.so').unlink()
        with pytest.raises(RuntimeError,match='runtime missing'):preflight(state_port=0,**files)


def test_preflight_reports_existing_owner_before_any_launch(tmp_path):
    files=fake_preflight_files(tmp_path)
    process=files['proc_root']/'123';process.mkdir();(process/'cmdline').write_bytes(b'lib/teleop_geo_hdapi\0')
    with patch('geomagic.standalone.device.subprocess.Popen',side_effect=AssertionError('must not launch')):
        with pytest.raises(RuntimeError,match='possible owner'):preflight(state_port=0,**files)


def test_preflight_reports_occupied_state_port(tmp_path):
    files=fake_preflight_files(tmp_path)
    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as occupied:
        occupied.bind(('127.0.0.1',0))
        with patch('geomagic.standalone.device.subprocess.run',return_value=SimpleNamespace(returncode=0,stdout='',stderr='')):
            with pytest.raises(RuntimeError,match='occupied'):preflight(state_port=occupied.getsockname()[1],**files)


@pytest.mark.parametrize('exceptional',[False,True])
def test_owned_fake_process_gets_explicit_zero_before_clean_or_exceptional_close(tmp_path,exceptional):
    trace=tmp_path/'trace.json'
    owner=HapticOwner(tmp_path,command=[sys.executable,str(ROOT/'geomagic/tests/fake_teleop_geo_owner.py'),'--trace',str(trace)])
    try:
        with owner:
            owner.wait_ready(timeout=3)
            owner.send((1,0,0))
            if exceptional:raise RuntimeError('simulated MuJoCo exception')
    except RuntimeError:
        assert exceptional
    record=json.loads(trace.read_text())
    assert owner.sent_zero_before_close
    assert record['packets'][-1][3:]==[0.,0.,0.,0,0]
    assert record['force_at_close']==0
    assert record['events']==['zero_force','disable_force','close_device']
    assert owner.process.poll()==0


@pytest.mark.parametrize('exceptional',[False,True])
def test_session_stops_device_before_viewer_and_control_on_exit(exceptional):
    events=[]
    class Owner:
        process=SimpleNamespace(poll=lambda:None)
        def start(self):events.append('start_device')
        def wait_ready(self,**kwargs):pass
        def close(self):events.append('zero_and_close_device')
    class Viewer:
        ready=threading.Event()
        def start(self,data):self.ready.set()
        def running(self):
            if exceptional:raise RuntimeError('viewer error')
            return False
        def close(self):events.append('close_viewer')
    control=SimpleNamespace(data=SimpleNamespace(time=0),ticks=0,
        backend=SimpleNamespace(control_updates=0),close=lambda:events.append('close_control'))
    if exceptional:
        with pytest.raises(RuntimeError,match='viewer error'):run_session(control,Owner(),Viewer())
    else:run_session(control,Owner(),Viewer())
    assert events[-3:]==['zero_and_close_device','close_viewer','close_control']


def test_native_force_watchdog_cap_ramp_and_zero_before_close(tmp_path):
    binary=tmp_path/'native_force_tests'
    compiled=subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-pedantic',
        str(ROOT/'geomagic/tests/teleop_geo_force_test.cpp'),'-o',str(binary)],capture_output=True,text=True)
    assert compiled.returncode==0,compiled.stderr
    result=subprocess.run([str(binary)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert '23 passed, 0 failed, 0 skipped' in result.stdout


@pytest.mark.parametrize('reason',['viewer_close','ctrl_c','exception'])
def test_real_lifecycle_with_fake_hdapi_force_turns_off_at_session_exit(tmp_path,reason):
    trace=tmp_path/'trace.json'
    owner=HapticOwner(tmp_path,command=[sys.executable,str(ROOT/'geomagic/tests/fake_teleop_geo_owner.py'),'--trace',str(trace)])
    class Control:
        ticks=0
        data=SimpleNamespace(time=0)
        backend=SimpleNamespace(control_updates=0)
        haptics_enabled=True
        def step(self):
            if reason=='exception' and self.ticks==2:raise RuntimeError('physics failed')
            self.ticks+=1;self.data.time+=.001
            return np.zeros(8),np.array([1.,0.,0.])
        def close(self):pass
    control=Control()
    class Viewer:
        ready=threading.Event()
        def start(self,data):self.ready.set()
        def running(self):return control.ticks<3 if reason=='viewer_close' else True
        def drain_keys(self):return []
        def publish(self,data):pass
        def close(self):pass
    stop=lambda:reason=='ctrl_c' and control.ticks>=3
    if reason=='exception':
        with pytest.raises(RuntimeError,match='physics failed'):run_session(control,owner,Viewer(),stop_requested=stop)
    else:run_session(control,owner,Viewer(),stop_requested=stop)
    record=json.loads(trace.read_text())
    assert any(packet[3:6]==[1.,0.,0.] for packet in record['packets'])
    assert record['packets'][-1][3:]==[0.,0.,0.,0,0]
    assert record['force_at_close']==0 and owner.process.poll()==0


def test_failed_glx_viewer_cannot_start_hdapi():
    events=[]
    class Viewer:
        ready=threading.Event()
        def start(self,data): events.append('start_viewer')
        def running(self): return False
        def raise_if_failed(self): raise RuntimeError('MuJoCo viewer exited (code 1); GLXBadDrawable')
        def close(self): events.append('close_viewer')
    owner=SimpleNamespace(start=lambda:events.append('start_device'),
        close=lambda:events.append('zero_and_close_device'))
    control=SimpleNamespace(data=None,close=lambda:events.append('close_control'))
    with pytest.raises(RuntimeError,match='GLXBadDrawable'):
        run_session(control,owner,Viewer())
    assert 'start_device' not in events
    assert events[-3:]==['zero_and_close_device','close_viewer','close_control']


def test_device_start_waits_for_successful_first_viewer_frame():
    events=[]
    class Viewer:
        ready=threading.Event()
        def start(self,data): events.append('render_and_swap');self.ready.set()
        def running(self): return True
        def close(self): events.append('close_viewer')
    class Owner:
        def start(self): events.append('start_device')
        def wait_ready(self,**kwargs): events.append('device_ready')
        def close(self): events.append('zero_and_close_device')
    control=SimpleNamespace(data=SimpleNamespace(time=0),ticks=0,
        backend=SimpleNamespace(control_updates=0),close=lambda:events.append('close_control'))
    run_session(control,Owner(),Viewer(),stop_requested=lambda:'device_ready' in events)
    assert events[:3]==['render_and_swap','start_device','device_ready']
    assert events[-3:]==['zero_and_close_device','close_viewer','close_control']


def test_minimized_or_closed_drawable_is_not_swapped():
    renderer=GlfwRenderer.__new__(GlfwRenderer)
    calls=[]
    renderer.window=object()
    renderer.glfw=SimpleNamespace(poll_events=lambda:calls.append('poll'),
        window_should_close=lambda w:False,get_framebuffer_size=lambda w:(0,0),
        swap_buffers=lambda w:calls.append('swap'))
    assert not renderer.draw(None)
    assert calls==['poll']
    renderer.glfw.window_should_close=lambda w:True
    assert not renderer.draw(None)
    assert 'swap' not in calls


def test_gl_resources_are_freed_before_window_context_destruction():
    renderer=GlfwRenderer.__new__(GlfwRenderer)
    events=[]
    renderer.window=object()
    renderer.context=SimpleNamespace(free=lambda:events.append('free_gl'))
    renderer.glfw=SimpleNamespace(make_context_current=lambda w:events.append('current'),
        destroy_window=lambda w:events.append('destroy_window'),terminate=lambda:events.append('terminate'))
    renderer.close()
    assert events==['current','free_gl','destroy_window','terminate']


def test_hdapi_exit_description_preserves_device_error_codes():
    owner=HapticOwner.__new__(HapticOwner)
    owner.thread=None
    owner.process=SimpleNamespace(returncode=9,poll=lambda:9)
    owner.last_error={'description':'Communication error','code':123,'internal_code':456}
    assert owner.exit_description()=='code 9; Communication error, HD 123, internal 456'


def test_viewer_child_reports_python_render_error_and_requests_force_off():
    model=contact_model()
    values=np.r_[np.zeros(model.nq+model.nv+model.nu),0.]
    closed,ready=threading.Event(),threading.Event()
    import queue
    errors=queue.Queue(maxsize=1)
    with patch('geomagic.standalone.viewer.mujoco.MjModel.from_xml_path',return_value=model), \
         patch('geomagic.standalone.viewer.GlfwRenderer',side_effect=RuntimeError('GL context failure')):
        with pytest.raises(RuntimeError,match='GL context failure'):
            _viewer_main('unused.xml',values,threading.Lock(),closed,ready,queue.Queue(),errors)
    assert closed.is_set() and not ready.is_set()
    assert errors.get_nowait()=='RuntimeError: GL context failure'




def test_force_peer_close_reports_hdapi_cause_instead_of_connection_refused(tmp_path):
    owner=HapticOwner(tmp_path)
    try:
        owner.receiver.close()
        assert owner.send((0,0,0)) is False
        assert isinstance(owner.transport_error,OSError)
        owner.process=SimpleNamespace(poll=lambda:9,returncode=9,stdout=None)
        owner.last_error={'description':'Illegal frame begin','code':1024,'internal_code':0}
        message=owner.send_failure_description()
        assert 'HDAPI owner stopped' in message and 'HD 1024' in message
        assert 'Connection refused' not in message
    finally:owner.close()


def test_session_force_channel_failure_keeps_hdapi_diagnosis_and_cleans_up():
    events=[]
    owner=SimpleNamespace(process=SimpleNamespace(poll=lambda:None),
        start=lambda:None,wait_ready=lambda **kwargs:None,
        send=lambda *args,**kwargs:False,
        send_failure_description=lambda:'HDAPI owner stopped (code 9; HD 1024)',
        close=lambda:events.append('zero_and_close_device'))
    class Viewer:
        ready=threading.Event()
        def start(self,data):self.ready.set()
        def running(self):return True
        def drain_keys(self):return []
        def close(self):events.append('close_viewer')
    control=SimpleNamespace(data=SimpleNamespace(time=0),ticks=0,
        backend=SimpleNamespace(control_updates=0),haptics_enabled=True,
        step=lambda:(None,np.zeros(3)),close=lambda:events.append('close_control'))
    with pytest.raises(RuntimeError,match='HDAPI owner stopped .*HD 1024'):
        run_session(control,owner,Viewer())
    assert events==['zero_and_close_device','close_viewer','close_control']
