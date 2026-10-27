"""Fixed-pose, tick-count and physical-time invariants; hardware-free."""
import math
import numpy as np
import pytest

from geomagic.teleop.pose_mapper import GeomagicPoseMapper, ROTATION_SCALE
from geomagic.teleop.synthetic import sample
from geomagic.vendor.geomagic_core.rotations import (
    matrix_to_rotation_vector, rotation_vector_to_matrix,
)

P = np.array((.4, .2, .3))
R = rotation_vector_to_matrix((.3, -.2, .4))


@pytest.mark.parametrize("hz", (250, 500, 1000))
@pytest.mark.parametrize("axis", ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.), (1., -2., 3.)))
@pytest.mark.parametrize("degrees", (5., 30.))
def test_fixed_pose_1000_new_packets_and_five_seconds(hz, axis, degrees):
    m = GeomagicPoseMapper(P, R, mode="rotation", activation_samples=2)
    vector = np.asarray(axis) / np.linalg.norm(axis) * math.radians(degrees)
    first = None
    expected = rotation_vector_to_matrix(ROTATION_SCALE * np.array((vector[2], -vector[0], vector[1]))) @ R
    for tick in range(max(1002, 5 * hz + 3)):
        target = m.update(sample(tick, time=tick/hz, rotvec=vector if tick >= 2 else (0.,)*3),
                          now=tick/hz, actual_position=P, actual_rotation=R)
        if tick >= 2:
            if first is None:
                first = target.rotation.copy()
            np.testing.assert_allclose(target.rotation, expected, atol=2e-14, rtol=0)
            np.testing.assert_array_equal(target.rotation, first)
            assert math.degrees(np.linalg.norm(matrix_to_rotation_vector(target.rotation @ R.T))) \
                == pytest.approx(degrees * ROTATION_SCALE, abs=1e-10)


def test_same_ramp_pose_at_common_times_across_rates():
    values = []
    for hz in (250, 500, 1000):
        m = GeomagicPoseMapper(P, R, mode="rotation", activation_samples=2)
        selected = []
        for tick in range(hz * 2 + 1):
            t = tick / hz
            angle = math.radians(30) * min(1., max(0., (t - .1) / .5))
            target = m.update(sample(tick, time=t, rotvec=(angle, 0., 0.)), now=t,
                              actual_position=P, actual_rotation=R)
            if tick % (hz // 50) == 0:
                selected.append(target.rotation.copy())
        values.append(np.array(selected))
    for rotations in values[1:]:
        np.testing.assert_allclose(rotations, values[0], atol=2e-14, rtol=0)


@pytest.mark.parametrize("hz", (250, 500, 1000))
def test_grey_held_without_any_device_motion(hz):
    m = GeomagicPoseMapper(P, R, mode="rotation", activation_samples=2)
    fixed = None
    for tick in range(1002):
        target = m.update(sample(tick, time=tick/hz), now=tick/hz,
                          actual_position=P, actual_rotation=R)
        np.testing.assert_allclose(target.rotation, R, atol=2e-14, rtol=0)
        if tick >= 2:
            if fixed is None: fixed = target.rotation.copy()
            np.testing.assert_array_equal(target.rotation, fixed)
