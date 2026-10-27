"""Pure mapping regression, stream timing and discontinuity measurements."""
import ast
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from geomagic.teleop.pose_mapper import (
    GeomagicPoseMapper, M_POSITION, POSITION_SCALE, ROTATION_SCALE, ROTATION_CONTROL_SIGN,
)
from geomagic.teleop.synthetic import axis_cases, run, sample
from geomagic.vendor.geomagic_core.frames import FrameMap
from geomagic.vendor.geomagic_core.pose_mapping import AnchoredPoseMapper
from geomagic.vendor.geomagic_core.controller import PositionTeleopController
from geomagic.vendor.geomagic_core.rotations import (
    matrix_to_rotation_vector, matrix_to_quaternion_xyzw,
    rotation_vector_to_matrix, quaternion_xyzw_to_matrix,
)

P = np.array((.4, .2, .3))  # deliberately outside the old standalone box
R = rotation_vector_to_matrix((.3, -.2, .4))
M = np.array(((0., 0., 1.), (1., 0., 0.), (0., 1., 0.)))


def mapper(mode="position", activation_samples=2):
    return GeomagicPoseMapper(P, R, mode=mode, activation_samples=activation_samples)


def tick(m, s, now=None, **kwargs):
    # Actual pose intentionally lags command; reanchor must discard command.
    return m.update(s, now=s.timestamp if now is None else now,
                    actual_position=P, actual_rotation=R, **kwargs)


def activate(m, **kwargs):
    first = tick(m, sample(0, **kwargs))
    assert not first.active
    target = tick(m, sample(1, **kwargs))
    assert target.active and target.reanchored
    np.testing.assert_array_equal(target.position, P)
    np.testing.assert_array_equal(target.rotation, R)
    return target


@pytest.mark.parametrize("name,delta,expected", list(axis_cases()))
def test_frozen_translation_axes(name, delta, expected):
    m = mapper()
    activate(m, position=(.1, .2, .3))
    t = tick(m, sample(2, position=np.array((.1, .2, .3)) + delta,
                       rotvec=(.2, .3, .4)))
    np.testing.assert_allclose(t.position - P, expected, atol=1e-14, rtol=0)
    np.testing.assert_array_equal(t.rotation, R)


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("sign", (1, -1))
def test_100_mm_translation_maps_to_70_mm_with_frozen_axes(axis, sign):
    np.testing.assert_array_equal(POSITION_SCALE, (.7, .7, .7))
    np.testing.assert_array_equal(M_POSITION, M)
    m = mapper()
    activate(m, position=(.1, .2, .3))
    delta = np.eye(3)[axis] * sign * .100
    target = tick(m, sample(2, position=np.array((.1, .2, .3)) + delta))
    expected = np.eye(3)[(axis + 1) % 3] * sign * .070
    np.testing.assert_allclose(target.position - P, expected, atol=1e-14, rtol=0)
    np.testing.assert_array_equal(target.rotation, R)


def test_position_matches_validated_old_controller_except_gain():
    # Compare mapping within one active interval. Actual-state transitions
    # intentionally differ from the vendored command-retention controller.
    reference = PositionTeleopController(
        AnchoredPoseMapper(FrameMap("geomagic_base", "fr3_world", M), (.3,) * 3, .05),
        P, deadman_button_index=0, clutch_button_index=1,
        activation_min_consecutive_samples=2,
    )
    m = mapper()
    for seq, position in enumerate(((0., 0., 0.), (0., 0., 0.),
                                    (.1, -.2, .3), (-.2, .3, -.1), (.5, .1, -.2))):
        state = sample(seq, position=position, rotvec=(.4, -.3, .2))
        old, new = reference.update(state, P, state.timestamp, R), tick(m, state)
        np.testing.assert_allclose(new.position - P, (old.target_xyz - P) * (POSITION_SCALE[0] / .3),
                                   atol=1e-14, rtol=0)
        np.testing.assert_array_equal(new.rotation, R)
        assert (new.active, new.clutched, new.reanchored) == (
            old.teleop_active, old.clutch_pressed, old.reanchored)


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("sign", (1, -1))
def test_frozen_rotation_full_basis_and_left_multiplication(axis, sign):
    # Verify the validated spatial-delta formula with nonidentity anchors.
    pure = AnchoredPoseMapper(FrameMap("geomagic_base", "fr3_world", M), (.3,) * 3,
                              .05, orientation_enabled=True,
                              rotation_control_sign=(1., -1., 1.))
    anchor = sample(0, rotvec=(.2, -.1, .3))
    A = quaternion_xyzw_to_matrix(anchor.orientation_xyzw)
    delta = np.eye(3)[axis] * sign * .005
    current_R = rotation_vector_to_matrix(delta) @ A
    current = sample(1, rotvec=matrix_to_rotation_vector(current_R))
    pure.activate(anchor, P, now=anchor.timestamp, slave_actual_rotation=R)
    result = pure.update(current, P, False, now=current.timestamp, slave_actual_rotation=R)
    expected_vector = np.array((delta[2], -delta[0], delta[1]))
    np.testing.assert_allclose(result.mapped_rotation_vector, expected_vector, atol=1e-13)
    np.testing.assert_allclose(result.target_rotation,
                               rotation_vector_to_matrix(expected_vector) @ R, atol=1e-13)


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("sign", (1, -1))
def test_rotation_mode_retains_each_physical_axis(axis, sign):
    m = mapper("rotation")
    activate(m)
    vector = np.eye(3)[axis] * sign * .25
    t = tick(m, sample(2, position=(.1, .2, .3), rotvec=vector))
    world = ROTATION_SCALE * np.array((vector[2], -vector[0], vector[1]))
    np.testing.assert_allclose(t.rotation, rotation_vector_to_matrix(world) @ R, atol=1e-13)
    np.testing.assert_array_equal(t.position, P)
    np.testing.assert_allclose(matrix_to_rotation_vector(t.rotation @ R.T), world, atol=1e-13)
    assert np.linalg.norm(world) == pytest.approx(.25 * ROTATION_SCALE)
    assert_rotation_valid(t.rotation)


def assert_rotation_valid(rotation):
    assert np.isfinite(rotation).all()
    assert np.linalg.det(rotation) == pytest.approx(1., abs=1e-12)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12, rtol=0)
    quaternion = matrix_to_quaternion_xyzw(rotation)
    assert np.isfinite(quaternion).all()
    assert np.linalg.norm(quaternion) == pytest.approx(1., abs=1e-12)
    np.testing.assert_allclose(quaternion_xyzw_to_matrix(quaternion), rotation, atol=1e-12)


def test_three_rotation_inputs_span_three_output_axes():
    spatial, tool_local = [], []
    for axis in range(3):
        m = mapper("rotation")
        activate(m)
        t = tick(m, sample(2, rotvec=np.eye(3)[axis] * .25))
        spatial.append(matrix_to_rotation_vector(t.rotation @ R.T))
        tool_local.append(matrix_to_rotation_vector(R.T @ t.rotation))
    assert np.linalg.matrix_rank(spatial, tol=1e-10) == 3
    assert np.linalg.matrix_rank(tool_local, tol=1e-10) == 3
    np.testing.assert_allclose(spatial, ROTATION_SCALE * np.array(((0., -.25, 0.), (0., 0., .25), (.25, 0., 0.))),
                               atol=1e-13)


@pytest.mark.parametrize("device_anchor_vector", [(0., 0., 0.), (.4, -.3, .2)])
@pytest.mark.parametrize("components", [(.3, -.2, 0.), (0., -.2, .4), (.3, -.2, .4)])
def test_mixed_rotations_preserve_composition_and_multi_axis_content(device_anchor_vector, components):
    m = mapper("rotation")
    activate(m, rotvec=device_anchor_vector)
    A = rotation_vector_to_matrix(device_anchor_vector)
    # Ordered, noncommuting physical rotations in the device-base frame.
    delta = (rotation_vector_to_matrix((components[0], 0., 0.))
             @ rotation_vector_to_matrix((0., components[1], 0.))
             @ rotation_vector_to_matrix((0., 0., components[2])))
    t = tick(m, sample(2, rotvec=matrix_to_rotation_vector(delta @ A)))
    mapped_vector = np.array((1., -1., 1.)) * matrix_to_rotation_vector(M @ delta @ M.T) * ROTATION_SCALE
    expected = rotation_vector_to_matrix(mapped_vector) @ R
    np.testing.assert_allclose(t.rotation, expected, atol=1e-12)
    np.testing.assert_allclose(matrix_to_rotation_vector(t.rotation @ R.T), mapped_vector,
                               atol=1e-12)
    assert np.count_nonzero(np.abs(mapped_vector) > .01) >= 2
    assert np.linalg.norm(mapped_vector) == pytest.approx(ROTATION_SCALE * np.linalg.norm(matrix_to_rotation_vector(delta)))
    np.testing.assert_array_equal(t.position, P)
    assert_rotation_valid(t.rotation)
    np.testing.assert_array_equal(ROTATION_CONTROL_SIGN, (1., -1., 1.))


def test_rotation_zero_relative_input_holds_nonidentity_anchor():
    m = mapper("rotation")
    anchor = activate(m, rotvec=(.4, -.3, .2))
    target = tick(m, sample(2, position=(1., 2., 3.), rotvec=(.4, -.3, .2)))
    np.testing.assert_allclose(target.rotation, anchor.rotation, atol=1e-14, rtol=0)
    np.testing.assert_array_equal(target.position, anchor.position)
    assert_rotation_valid(target.rotation)


def test_neutral_and_no_movement_at_anchor():
    m = mapper()
    anchor = activate(m, position=(1., 2., 3.))
    t = tick(m, sample(2, position=(1., 2., 3.)))
    np.testing.assert_array_equal(t.position, anchor.position)
    np.testing.assert_array_equal(t.rotation, anchor.rotation)


@pytest.mark.parametrize("mode", ("position", "rotation"))
def test_duplicate_control_ticks_never_spend_mapper_step(mode):
    m = mapper(mode)
    activate(m)
    s = sample(2, position=(.2, .2, .2), rotvec=(.2, .2, .2))
    target = tick(m, s)
    count = m.mapper_updates
    for i in range(1, 41):  # 1 kHz ticks; same accepted source sample
        held = tick(m, s, now=s.timestamp + i * .001)
        assert held.fresh and not held.mapper_updated
        np.testing.assert_array_equal(held.position, target.position)
        np.testing.assert_array_equal(held.rotation, target.rotation)
    assert m.mapper_updates == count
    stale = tick(m, s, now=s.timestamp + .051)
    assert not stale.fresh and not stale.active
    np.testing.assert_array_equal(stale.position, P)
    np.testing.assert_array_equal(stale.rotation, R)


def test_default_activation_counts_unique_samples_not_ticks():
    m = mapper(activation_samples=50)
    for seq in range(49):
        s = sample(seq)
        assert not tick(m, s).active
        for j in range(4):
            assert not tick(m, s, now=s.timestamp + j * .001).active
    t = tick(m, sample(49))
    assert t.active and t.reanchored
    assert m.mapper_updates == 50


@pytest.mark.parametrize("first_motion", [
    ((.001, .002, .001), (.003, .004, .002)),   # small, as before
    ((.20, -.30, .25), (.9, -.6, .7)),          # far beyond the removed caps
], ids=("small", "large"))
@pytest.mark.parametrize("mode", ("position", "rotation"))
def test_reanchor_actual_state_all_events(mode, first_motion):
    m = mapper(mode)
    activate(m)
    previous = tick(m, sample(2, position=first_motion[0], rotvec=first_motion[1]))
    assert np.linalg.norm(previous.position - P) > 0 or np.linalg.norm(previous.rotation - R) > 0
    measurements = []

    def held(label, s, *, now=None, reanchored=False, **kwargs):
        nonlocal previous
        t = tick(m, s, now=now, **kwargs)
        dp = float(np.linalg.norm(t.position - P))
        dr = float(np.linalg.norm(matrix_to_rotation_vector(t.rotation @ R.T)))
        measurements.append((label, dp, dr))
        np.testing.assert_allclose(t.position, P, atol=1e-14, rtol=0)
        np.testing.assert_allclose(t.rotation, R, atol=1e-14, rtol=0)
        if reanchored:
            assert t.reanchored
        previous = t
        return t

    held("deadman release", sample(3, position=(1., 2., 3.), rotvec=(1., 0., 0.), grey=False))
    held("deadman reengage", sample(4, position=(1., 2., 3.), rotvec=(1., 0., 0.)), reanchored=True)
    held("clutch engage", sample(5, white=True))
    held("clutch reposition", sample(6, position=(-2., 1., 3.), rotvec=(0., 1., 0.), white=True))
    held("clutch release", sample(7, position=(-2., 1., 3.), rotvec=(0., 1., 0.)), reanchored=True)
    assert not held("stale", sample(7), now=.100).fresh
    held("reconnect gate", sample(8, time=.105))
    held("reconnect anchor", sample(9, time=.110), reanchored=True)
    held("session gate", sample(0, time=.115, session="restart"))
    held("session anchor", sample(1, time=.120, session="restart"), reanchored=True)
    # Detect a gap even when control caller never observed a stale tick.
    held("missed stale gate", sample(2, time=.200, session="restart"))
    held("missed stale anchor", sample(3, time=.205, session="restart"), reanchored=True)
    # A latched restart event can arrive after a drain ending on the same ID.
    held("latched event gate", sample(4, time=.210, session="restart"), session_changed=True)
    held("latched event anchor", sample(5, time=.215, session="restart"), reanchored=True)
    m.set_mode("rotation" if mode == "position" else "position")
    assert not held("mode held sample", sample(5, time=.215, session="restart")).active
    held("mode gate", sample(6, time=.220, session="restart"))
    held("mode anchor", sample(7, time=.225, session="restart"), reanchored=True)
    # Return to the original mode, checking both entering and leaving Rotation.
    m.set_mode(mode)
    held("return mode gate", sample(8, time=.230, session="restart"))
    held("return mode anchor", sample(9, time=.235, session="restart"), reanchored=True)
    for label, _, angle in measurements:
        print(f"TRANSITION mode={mode} event={label!r} actual_error_deg={np.rad2deg(angle):.17g}")
    print(f"REANCHOR {mode}: events={len(measurements)}, "
          f"max_position_m={max(x[1] for x in measurements):.17g}, "
          f"max_rotation_rad={max(x[2] for x in measurements):.17g}")


@pytest.mark.parametrize("mode", ("position", "rotation"))
def test_clutch_already_held_at_activation(mode):
    m = mapper(mode)
    anchor = activate(m, white=True)
    released = tick(m, sample(2, white=False, position=(1., 2., 3.), rotvec=(.5, .4, .3)))
    assert released.reanchored
    np.testing.assert_array_equal(released.position, anchor.position)
    np.testing.assert_array_equal(released.rotation, anchor.rotation)


@pytest.mark.parametrize("event", ("deadman", "clutch", "stale", "session", "mode"))
def test_full_3d_rotation_resumes_relative_to_each_new_anchor(event):
    m = mapper("rotation")
    activate(m, rotvec=(.2, -.1, .3))
    held = tick(m, sample(2, rotvec=(.6, -.3, .4)))
    anchor_vector = np.array((-.4, .5, -.2))
    anchor_options = dict(rotvec=anchor_vector, position=(1., 2., 3.))
    if event == "deadman":
        tick(m, sample(3, grey=False, **anchor_options))
        anchor_sample = sample(4, **anchor_options)
    elif event == "clutch":
        tick(m, sample(3, white=True, **anchor_options))
        anchor_sample = sample(4, **anchor_options)
    elif event == "stale":
        assert not tick(m, sample(2), now=.080).fresh
        tick(m, sample(3, time=.085, **anchor_options))
        anchor_sample = sample(4, time=.090, **anchor_options)
    elif event == "session":
        tick(m, sample(0, time=.015, session="new-hdapi", **anchor_options), session_changed=True)
        anchor_sample = sample(1, time=.020, session="new-hdapi", **anchor_options)
    else:
        m.set_mode("position")
        tick(m, sample(3, **anchor_options))
        tick(m, sample(4, **anchor_options))
        m.set_mode("rotation")
        tick(m, sample(5, **anchor_options))
        anchor_sample = sample(6, **anchor_options)
    anchored = tick(m, anchor_sample)
    assert anchored.reanchored
    np.testing.assert_array_equal(anchored.rotation, R)
    np.testing.assert_array_equal(anchored.position, P)
    delta = rotation_vector_to_matrix((.2, -.3, .4))
    moved_device = delta @ rotation_vector_to_matrix(anchor_vector)
    moved = tick(m, sample(anchor_sample.sequence + 1, time=anchor_sample.timestamp + .005,
                           session=anchor_sample.session_id, position=(-1., -2., -3.),
                           rotvec=matrix_to_rotation_vector(moved_device)))
    expected_delta = rotation_vector_to_matrix(ROTATION_SCALE * np.array((.4, -.2, -.3)))
    np.testing.assert_allclose(moved.rotation, expected_delta @ R, atol=1e-12)
    np.testing.assert_array_equal(moved.position, P)
    assert_rotation_valid(moved.rotation)


@pytest.mark.parametrize("change", [
    {"valid": False}, {"source_valid": False}, {"units": "mm"}, {"frame_id": "other"},
    {"position_xyz": (float("nan"), 0., 0.)}, {"orientation_xyzw": (0., 0., 0., 0.)},
    {"orientation_xyzw": None}, {"timestamp": float("nan")},
])
def test_invalid_state_holds_and_requires_new_samples(change):
    m = mapper()
    activate(m)
    target = tick(m, sample(2, position=(.001, 0., 0.)))
    t = tick(m, replace(sample(3), **change), now=.015)
    assert not t.fresh and not t.active
    np.testing.assert_array_equal(t.position, P)
    assert not tick(m, sample(4)).active
    t = tick(m, sample(5))
    assert t.reanchored
    np.testing.assert_array_equal(t.position, P)


def test_repeat_invalid_flag_and_missing_sample_immediately_hold():
    m = mapper()
    activate(m)
    s = sample(2, position=(.001, 0., 0.))
    target = tick(m, s)
    t = tick(m, replace(s, valid=False))
    assert not t.active and not t.fresh
    np.testing.assert_array_equal(t.position, P)
    t = tick(m, None, now=.011)
    assert not t.active and not t.fresh


def test_old_sequence_cannot_refresh_timestamp_or_spend_limit():
    m = mapper()
    activate(m)
    s = sample(2, position=(.1, 0., 0.))
    target = tick(m, s)
    old = sample(1, time=.020, position=(-1., 0., 0.))
    t = tick(m, old)
    assert not t.mapper_updated
    np.testing.assert_array_equal(t.position, target.position)
    assert not tick(m, replace(s, timestamp=.1), now=.1).fresh


REMOVED_LIMIT_ATTRIBUTES = ("max_step", "max_excursion", "max_orientation_step",
                            "max_orientation_excursion")


def test_mapping_limits_disabled_and_no_workspace_box():
    m = mapper()
    activate(m)
    pure = m._controller.mapper
    for name in REMOVED_LIMIT_ATTRIBUTES:
        assert not hasattr(pure, name)
    assert np.isneginf(pure.workspace_min).all() and np.isposinf(pure.workspace_max).all()
    np.testing.assert_array_equal(pure.scale, (.7, .7, .7))


@pytest.mark.parametrize("device_delta", [
    (.01, 0., 0.),            # 7 mm mapped in ONE sample (> old 1 mm/sample)
    (0., 0., .40),            # 280 mm mapped (> old +/-75 mm excursion)
    (-.50, .35, -.30),        # all axes, mixed signs, beyond both old caps
])
def test_position_step_and_excursion_not_clipped(device_delta):
    m = mapper()
    activate(m, position=(.1, .2, .3))
    t = tick(m, sample(2, position=np.array((.1, .2, .3)) + device_delta))
    expected = np.array(POSITION_SCALE) * (M @ np.asarray(device_delta))   # scale x frozen frame map only
    np.testing.assert_allclose(t.position - P, expected, atol=1e-14, rtol=0)
    np.testing.assert_array_equal(t.rotation, R)       # position mode holds orientation
    # Frozen axis/sign: device +X -> FR3 +Y, +Z -> +X, +Y -> +Z.
    for axis in range(3):
        if device_delta[axis]:
            k = int(np.argmax(np.abs(M[:, axis])))
            assert np.sign(t.position[k] - P[k]) == np.sign(device_delta[axis])


@pytest.mark.parametrize("degrees", (10., 60., 120.))  # > old 1 deg/sample; 60/120 > old 45 deg
def test_pure_rotation_step_and_excursion_not_clipped(degrees):
    pure = AnchoredPoseMapper(FrameMap("geomagic_base", "fr3_world", M), (.3,) * 3,
                              .05, orientation_enabled=True,
                              rotation_control_sign=(1., -1., 1.))
    anchor = sample(0, rotvec=(.2, -.1, .3))
    A = quaternion_xyzw_to_matrix(anchor.orientation_xyzw)
    delta = np.array((.3, -.5, .8)) / np.linalg.norm((.3, -.5, .8)) * np.deg2rad(degrees)
    current = sample(1, rotvec=matrix_to_rotation_vector(rotation_vector_to_matrix(delta) @ A))
    pure.activate(anchor, P, now=anchor.timestamp, slave_actual_rotation=R)
    result = pure.update(current, P, False, now=current.timestamp, slave_actual_rotation=R)
    expected_vector = np.array((delta[2], -delta[0], delta[1]))   # frozen map + sign
    np.testing.assert_allclose(result.mapped_rotation_vector, expected_vector, atol=1e-12)
    assert np.rad2deg(np.linalg.norm(result.mapped_rotation_vector)) == pytest.approx(degrees)
    np.testing.assert_allclose(result.target_rotation,
                               rotation_vector_to_matrix(expected_vector) @ R, atol=1e-12)
    assert not (result.step_limited or result.orientation_step_limited
                or result.orientation_excursion_limited)


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("degrees", (10., 60., 120., -90.))
def test_rotation_mode_full_3d_not_clipped(axis, degrees):
    m = mapper("rotation")
    activate(m)                      # device anchor orientation = identity
    local_vector = np.eye(3)[axis] * np.deg2rad(degrees)
    world = R @ local_vector
    device_vector = np.array((-world[1], world[2], world[0]))  # inverse current map
    t = tick(m, sample(2, rotvec=device_vector))
    np.testing.assert_allclose(t.rotation, R @ rotation_vector_to_matrix(local_vector * ROTATION_SCALE),
                               atol=1e-12)
    local = matrix_to_rotation_vector(R.T @ t.rotation)
    np.testing.assert_allclose(local, local_vector * ROTATION_SCALE, atol=1e-12)
    assert np.rad2deg(local[axis]) == pytest.approx(degrees * ROTATION_SCALE)
    assert_rotation_valid(t.rotation)
    np.testing.assert_array_equal(t.position, P)


@pytest.mark.parametrize("mode", ("full_pose", "6d", "", None))
def test_full_6d_mode_is_not_exposed(mode):
    with pytest.raises(ValueError):
        mapper(mode)


def test_reset_uses_caller_pose_and_restarts_activation():
    m = mapper()
    activate(m)
    m.reset(P + 1., np.eye(3))
    t = m.update(sample(0), now=0., actual_position=P + 1., actual_rotation=np.eye(3))
    assert not t.active
    np.testing.assert_array_equal(t.position, P + 1.)
    np.testing.assert_array_equal(t.rotation, np.eye(3))


def test_pure_synthetic_harness():
    assert len(run()) == 12


def test_vendored_ast_matches_frozen_source_with_only_declared_changes():
    root = Path(__file__).resolve().parents[1] / "vendor/geomagic_core"
    provenance = json.loads((root / "PROVENANCE.json").read_text())
    for name, digest in provenance["source_sha256"].items():
        source = (root / provenance["source_root"]).resolve() / name
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
        expected = ast.parse(source.read_text().replace(
            "from src.device.geomagic_state import GeomagicState",
            "from .state import GeomagicState"))
        if name == "frames.py":
            for node in expected.body:
                if isinstance(node, ast.ClassDef):
                    node.body = [x for x in node.body if not (
                        isinstance(x, ast.FunctionDef) and x.name == "map_force_to_device")]
        vendored = ast.parse((root / name).read_text())
        if name == "pose_mapping.py":
            # Declared change: GEOMAGIC_MAPPING_LIMITS = DISABLED. Only the
            # mapper constructor/update and the rotations import may differ;
            # everything else must stay AST-identical to the frozen source.
            identifiers = {n.id for n in ast.walk(vendored) if isinstance(n, ast.Name)} | \
                {n.attr for n in ast.walk(vendored) if isinstance(n, ast.Attribute)} | \
                {n.arg for n in ast.walk(vendored) if isinstance(n, ast.arg)} | \
                {a.name for n in ast.walk(vendored) if isinstance(n, ast.ImportFrom) for a in n.names}
            removed = set(REMOVED_LIMIT_ATTRIBUTES) | {
                "limited_rotation", "max_delta_per_update_m", "max_workspace_excursion_m",
                "max_orientation_delta_per_update_rad", "max_orientation_excursion_rad"}
            assert identifiers.isdisjoint(removed)
            imports = lambda tree: {a.name for n in tree.body if isinstance(n, ast.ImportFrom)
                                    and n.module == "rotations" for a in n.names}
            assert imports(vendored) == imports(expected) - {"limited_rotation"}
            _strip_declared_limit_changes(expected)
            _strip_declared_limit_changes(vendored)
        assert ast.dump(vendored) == ast.dump(expected)


def _strip_declared_limit_changes(tree):
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and node.module == "rotations")]
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "AnchoredPoseMapper":
            node.body = [x for x in node.body if not (
                isinstance(x, ast.FunctionDef) and x.name in {"__init__", "update"})]
