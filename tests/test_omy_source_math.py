"""Execute the original bridge's pure functions without starting ROS or hardware.

Only AST nodes from the shipped bridge are executed, not reimplemented formulas.
This does not certify the ROS launcher or missing OMY scene assets.
"""
from pathlib import Path
import ast
import numpy as np

SOURCE=Path(__file__).resolve().parents[1]/'launch/FR3_omy_bridge.py'
def bridge_math():
    tree=ast.parse(SOURCE.read_text())
    functions={'transform_from_pose','inverse_transform','skew','rotvec_to_matrix','matrix_to_rotvec','limit_norm','rotation_distance_deg','make_desired_target','condition_target','dls_pseudoinverse','exact_nullspace_projector'}
    nodes=[n for n in tree.body if (isinstance(n,ast.Assign) and 70<=n.lineno<136) or (isinstance(n,ast.FunctionDef) and n.name in functions)]
    namespace={'np':np};exec(compile(ast.Module(nodes,type_ignores=[]),str(SOURCE),'exec'),namespace);return namespace

def test_omy_clutch_anchor_is_stationary():
    b=bridge_math();p=np.array([.2,.3,.4]);r=b['rotvec_to_matrix'](np.array([.1,.2,-.3]));f=np.array([.5,0,.6])
    actual=b['make_desired_target'](p,r,p,r,f,r)
    np.testing.assert_allclose(actual[0],f,atol=1e-12);np.testing.assert_allclose(actual[1],r,atol=1e-12)

def test_omy_mapped_rotation_stays_proper_and_scaled():
    b=bridge_math();identity=np.eye(3);zero=np.zeros(3)
    for axis in np.eye(3):
        target=b['make_desired_target'](zero,identity,zero,b['rotvec_to_matrix'](.2*axis),zero,identity)[1]
        np.testing.assert_allclose(target.T@target,identity,atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(b['matrix_to_rotvec'](target)),.2*b['ORIENTATION_SCALE'],atol=1e-12)
        assert np.linalg.det(target)>0

def test_omy_dls_singular_input_is_finite_and_exact_nullspace_has_no_task_leakage():
    b=bridge_math();j=np.random.default_rng(2).normal(size=(6,7));j[-1]=j[0]
    pinv=b['dls_pseudoinverse'](j,b['IK_DAMPING']);projector=b['exact_nullspace_projector'](j)
    assert pinv.shape==(7,6) and np.isfinite(pinv).all()
    np.testing.assert_allclose(j@projector,0,atol=1e-12)
    np.testing.assert_allclose(projector@projector,projector,atol=1e-12)

def test_omy_pose_inverse_roundtrip():
    b=bridge_math();t=b['transform_from_pose'](np.array([.4,.2,-.5]),b['rotvec_to_matrix'](np.array([.1,.2,.3])))
    np.testing.assert_allclose(t@b['inverse_transform'](t),np.eye(4),atol=1e-12)
