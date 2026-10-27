"""Collection rotation modes: anchor-local projection, finite pose, safety edges."""
from dataclasses import replace
import itertools
import math
import numpy as np
import pytest
from geomagic.teleop.pose_mapper import GeomagicPoseMapper, POSITION_SCALE, ROTATION_SCALE, ROTATION_CONTROL_SIGN
from geomagic.teleop.synthetic import sample
from geomagic.vendor.geomagic_core.rotations import rotation_vector_to_matrix as Exp, matrix_to_rotation_vector as Log, matrix_to_quaternion_xyzw

P=np.array((.4,.2,.3));R=Exp((.5,-.7,.4));D=Exp((-.3,.2,.4))
MODES=('position','rotation_z','rotation')

def device_vector(local):
    # Inverse of [z,-x,y], after expressing intended local rotation in world.
    world=R@np.asarray(local)
    return np.array((-world[1],world[2],world[0]))

def state(i,local=(0.,0.,0.),**kwargs):
    return sample(i,rotvec=Log(Exp(device_vector(local))@D),**kwargs)

def update(m,s,**kw):return m.update(s,now=s.timestamp,actual_position=P,actual_rotation=R,**kw)
def activated(mode):
    m=GeomagicPoseMapper(P,R,mode=mode,activation_samples=2)
    update(m,state(0));t=update(m,state(1));assert t.active and t.reanchored
    return m

def valid(r):
    assert np.isfinite(r).all()
    np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-12)
    assert np.linalg.det(r)==pytest.approx(1,abs=1e-12)
    assert np.linalg.norm(matrix_to_quaternion_xyzw(r))==pytest.approx(1,abs=1e-12)

@pytest.mark.parametrize('mode',('rotation_z','rotation'))
@pytest.mark.parametrize('local',[(.2,0,0),(0,.2,0),(0,0,.2),(.2,0,.3),(.2,-.1,.3)])
def test_tool_axes_project_only_after_spatial_mapping(mode,local):
    m=activated(mode);t=update(m,state(2,local,position=(2,-3,4)))
    kept=np.asarray((0,0,local[2]) if mode=='rotation_z' else local)*.4
    np.testing.assert_allclose(t.rotation,R@Exp(kept),atol=2e-13)
    np.testing.assert_array_equal(t.position,P);valid(t.rotation)
    if mode=='rotation_z':np.testing.assert_allclose(t.rotation[:,2],R[:,2],atol=2e-13)

@pytest.mark.parametrize('mode',('rotation_z','rotation'))
@pytest.mark.parametrize('degrees',(10,20,-10,-20))
def test_collection_scale_before_conditioning(mode,degrees):
    assert ROTATION_SCALE==.4 and POSITION_SCALE==(.7,.7,.7)
    assert ROTATION_CONTROL_SIGN==(1.,-1.,1.)
    m=activated(mode);t=update(m,state(2,(0,0,math.radians(degrees))))
    local=Log(R.T@t.rotation)
    np.testing.assert_allclose(local,(0,0,math.radians(degrees)*.4),atol=1e-13)

@pytest.mark.parametrize('mode',('rotation_z','rotation'))
@pytest.mark.parametrize('new_packets',(False,True))
def test_identical_device_pose_1000_updates_has_no_raw_accumulation(mode,new_packets):
    m=activated(mode);s=state(2,(.2,-.1,math.radians(20)));fixed=update(m,s)
    for i in range(3,1003):
        incoming=replace(s,sequence=i,timestamp=i*.005) if new_packets else s
        t=update(m,incoming)
        np.testing.assert_array_equal(t.rotation,fixed.rotation)
        np.testing.assert_array_equal(t.position,P)

@pytest.mark.parametrize('old,new',list(itertools.permutations(MODES,2)))
@pytest.mark.parametrize('tracking_error_deg',(60,90))
def test_all_mode_transitions_discard_old_target_and_anchor_actual(old,new,tracking_error_deg):
    m=activated(old)
    # Deliberately unreachable previous command must not become the new anchor.
    m._position=P+(.2,-.1,.3);m._rotation=Exp((math.radians(tracking_error_deg),0,0))@R
    assert m.set_mode(new,actual_position=P,actual_rotation=R)
    first=m._snapshot()
    np.testing.assert_array_equal(first.position,P);np.testing.assert_array_equal(first.rotation,R)
    for i in (2,3):t=update(m,state(i,(.3,-.2,.4)))
    assert t.reanchored
    np.testing.assert_array_equal(t.position,P);np.testing.assert_array_equal(t.rotation,R)

@pytest.mark.parametrize('mode',('rotation_z','rotation'))
@pytest.mark.parametrize('edge',('grey','white','stale','session'))
def test_rotation_safety_edges_reanchor_actual_in_both_modes(mode,edge):
    m=activated(mode);update(m,state(2,(.3,-.2,.4)))
    if edge=='grey':t=update(m,state(3,grey=False));assert not t.active
    elif edge=='white':t=update(m,state(3,white=True));assert t.clutched
    elif edge=='stale':
        s=state(2);t=m.update(s,now=s.timestamp+.06,actual_position=P,actual_rotation=R);assert not t.fresh
    else:t=update(m,state(3,session='restart'));assert not t.active
    np.testing.assert_array_equal(t.position,P);np.testing.assert_array_equal(t.rotation,R)
    # A safety event resets branch state and discards the earlier intent.
    for i in (20,21,22):t=update(m,state(i,session='restart' if edge=='session' else 'synthetic-session'))
    assert t.active
    np.testing.assert_array_equal(t.position,P);np.testing.assert_allclose(t.rotation,R,atol=1e-13)
