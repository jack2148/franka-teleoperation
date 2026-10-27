"""Deterministic hardware-free sequence harness; optional single-model FR3 test.

python -B -m geomagic.teleop.synthetic [--fr3]
Only --fr3 creates one caller-owned shared Square model/data. No physics steps,
viewer, sockets, recording, or hardware. Tests use the same sequence builders.
"""
import argparse
import json
import math
import numpy as np

from geomagic.teleop.pose_mapper import GeomagicPoseMapper, POSITION_SCALE, ROTATION_SCALE
from geomagic.vendor.geomagic_core.state import GeomagicState
from geomagic.vendor.geomagic_core.rotations import rotation_vector_to_matrix


def sample(sequence, *, time=None, position=(0., 0., 0.), rotvec=(0., 0., 0.),
           grey=True, white=False, session="synthetic-session", valid=True):
    """Source timestamp deliberately differs from receive clock."""
    vector = np.asarray(rotvec, dtype=float)
    angle = float(np.linalg.norm(vector))
    xyz = vector * (math.sin(angle / 2) / angle) if angle else np.zeros(3)
    return GeomagicState(
        position_xyz=tuple(position), orientation_xyzw=(*xyz, math.cos(angle / 2)),
        buttons=(grey, white), timestamp=sequence * .005 if time is None else time,
        valid=valid, sequence=sequence, session_id=session,
        source_timestamp=1700000000. + sequence, source_clock_domain="device-test",
    )


def axis_cases():
    for axis, name in enumerate("XYZ"):
        for sign in (1, -1):
            delta = np.zeros(3)
            delta[axis] = sign * .001
            expected = np.zeros(3)
            expected[(axis + 1) % 3] = sign * .001 * POSITION_SCALE[(axis + 1) % 3]
            yield f"{'+' if sign > 0 else '-'}device_{name}", delta, expected


def run(*, fr3=False):
    pipeline = None
    initial_position, initial_rotation = np.array((.4, .2, .3)), np.eye(3)
    if fr3:
        import mujoco
        from geomagic.teleop.pipeline import GeomagicTargetPipeline
        from geomagic.teleop.robot_helpers import MIMIC_ROOT
        model = mujoco.MjModel.from_xml_path(str(MIMIC_ROOT / "environments/mimicgen_square/scene.xml"))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        mujoco.mj_forward(model, data)
        pipeline = GeomagicTargetPipeline(model, data, activation_samples=2)
        initial_position, initial_rotation = pipeline.controller.actual_pose()

    def drive(mode, final):
        if pipeline:
            pipeline.reset()
            pipeline.set_mode(mode)
            for state in (sample(0), sample(1), final):
                command = pipeline.step(state, now=state.timestamp, dt=.001, gripper_command=128.)
            assert command.joint_position.shape == (7,)
            assert np.isfinite(command.joint_position).all()
            return pipeline.last_target, command.joint_position.tolist()
        mapper = GeomagicPoseMapper(initial_position, initial_rotation,
                                    mode=mode, activation_samples=2)
        for state in (sample(0), sample(1), final):
            target = mapper.update(state, now=state.timestamp,
                                   actual_position=initial_position, actual_rotation=initial_rotation)
        return target, None

    results = []
    for name, delta, expected in axis_cases():
        target, qtarget = drive("position", sample(2, position=delta))
        actual = target.position - initial_position
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)
        np.testing.assert_array_equal(target.rotation, initial_rotation)
        results.append(dict(case=name, cartesian_delta_m=actual.tolist(), absolute_qtarget=qtarget))
    for axis, name in enumerate("XYZ"):
        for sign in (1, -1):
            vector = np.eye(3)[axis] * sign * .005
            target, qtarget = drive("rotation", sample(2, rotvec=vector))
            # Current mapped spatial vector = [z, -x, y].
            world_vector = ROTATION_SCALE * np.array((vector[2], -vector[0], vector[1]))
            expected = rotation_vector_to_matrix(world_vector) @ initial_rotation
            np.testing.assert_allclose(target.rotation, expected, atol=1e-12, rtol=0)
            np.testing.assert_array_equal(target.position, initial_position)
            results.append(dict(case=f"{'+' if sign > 0 else '-'}rotation_{name}",
                                target_rotation=target.rotation.tolist(), absolute_qtarget=qtarget))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fr3", action="store_true")
    print(json.dumps(run(fr3=parser.parse_args().fr3), indent=2))
