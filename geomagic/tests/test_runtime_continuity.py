"""Continuous anchor-relative pose across pi; no hardware or rendering."""
import math
import numpy as np
import pytest
from geomagic.teleop.pose_mapper import GeomagicPoseMapper, POSITION_SCALE, ROTATION_SCALE
from geomagic.teleop.synthetic import sample
from geomagic.vendor.geomagic_core.rotations import matrix_to_rotation_vector, rotation_vector_to_matrix
from geomagic.teleop.runtime_rotation import validate_rotation_matrix

P=np.array((.4,.2,.3)); A=rotation_vector_to_matrix((.4,-.2,.3))

@pytest.mark.parametrize('axis',[(1,0,0),(0,1,0),(0,0,1),(1,-2,3)])
@pytest.mark.parametrize('sign',[1,-1])
@pytest.mark.parametrize('flip_quaternions',[False,True])
@pytest.mark.parametrize('device_anchor',[(0.,0.,0.),(.4,-.2,.3)])
@pytest.mark.parametrize('mode',('rotation','rotation_z','6dof'))
def test_cross_pi_and_reverse_is_continuous_finite_pose(axis,sign,flip_quaternions,device_anchor,mode):
    from dataclasses import replace
    m=GeomagicPoseMapper(P,A,mode=mode,activation_samples=2)
    axis=np.array(axis,dtype=float);axis/=np.linalg.norm(axis)
    poses=[];degrees=[0,0,170,175,179,180,181,185,190,185,181,180,179,175,170]
    for i,deg in enumerate(degrees):
        device_R=rotation_vector_to_matrix(sign*math.radians(deg)*axis)@rotation_vector_to_matrix(device_anchor)
        s=sample(i,rotvec=matrix_to_rotation_vector(device_R))
        if flip_quaternions and i%2:s=replace(s,orientation_xyzw=tuple(-np.array(s.orientation_xyzw)))
        target=m.update(s,now=s.timestamp,actual_position=P,actual_rotation=A)
        mapped=np.array((axis[2],-axis[0],axis[1]))*sign*math.radians(deg)*ROTATION_SCALE
        expected=(A@rotation_vector_to_matrix((0.,0.,float(A[:,2]@mapped))) if mode=='rotation_z'
                  else rotation_vector_to_matrix(mapped)@A)
        np.testing.assert_allclose(target.rotation,expected,atol=1e-12)
        validate_rotation_matrix(target.rotation);poses.append(target.rotation)
    # Requested 170->190 passage: <=2 degree robot step for <=5 degree input.
    jumps=[np.linalg.norm(matrix_to_rotation_vector(b@a.T)) for a,b in zip(poses[2:],poses[3:])]
    assert math.degrees(max(jumps))<=5*ROTATION_SCALE+1e-10
    fixed=poses[-1]
    for i in range(15,1015):
        device_R=rotation_vector_to_matrix(sign*math.radians(170)*axis)@rotation_vector_to_matrix(device_anchor)
        s=sample(i,rotvec=matrix_to_rotation_vector(device_R))
        target=m.update(s,now=s.timestamp,actual_position=P,actual_rotation=A)
        np.testing.assert_array_equal(target.rotation,fixed)
    assert POSITION_SCALE==(.7,.7,.7) and ROTATION_SCALE==.4

@pytest.mark.parametrize('invalid',[np.eye(2),np.eye(3)*2,np.diag((-1,1,1)),np.full((3,3),float('nan'))])
def test_optimized_validation_keeps_error_checks(invalid):
    with pytest.raises(ValueError):validate_rotation_matrix(invalid)
