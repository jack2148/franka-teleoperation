"""Single caller-owned Square model; no stepping, rendering, or devices.

Uses unittest because the existing MuJoCo interpreter does not contain pytest.
"""
from collections import deque
from dataclasses import replace
import sys
import unittest
from unittest.mock import patch
import numpy as np

try:
    import mujoco
except ImportError:
    mujoco = None

from geomagic.bridge.protocol import GeomagicStatePacket, encode_packet
from geomagic.bridge.udp_state_client import GeomagicUDPStateClient
from geomagic.teleop.fr3_target_controller import FR3TargetController, JOINT_NAMES, EE_SITE
from geomagic.teleop.pipeline import GeomagicTargetPipeline
from geomagic.teleop.pose_mapper import CartesianTarget, POSITION_SCALE, ROTATION_SCALE
from geomagic.teleop.diagnostics import TrackingDiagnostics
from geomagic.teleop.robot_helpers import MIMIC_ROOT, BRIDGE_PATH, robot_helpers, command_adapter
from geomagic.teleop.synthetic import axis_cases, sample
from geomagic.vendor.geomagic_core.rotations import (
    matrix_to_rotation_vector, rotation_vector_to_matrix,
)


class DatagramSocket:
    """Only transport is faked; production packet codec and client are exercised."""
    def __init__(self):
        self.pending = deque()

    def setblocking(self, value):
        assert value is False

    def bind(self, address):
        self.address = address

    def getsockname(self):
        return self.address

    def recvfrom(self, size):
        if not self.pending:
            raise BlockingIOError
        return self.pending.popleft()[:size], ("127.0.0.1", 1234)

    def close(self):
        pass


@unittest.skipIf(mujoco is None, "MuJoCo unavailable in this interpreter; use existing isr_lerobot Python")
class FR3ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(str(
            MIMIC_ROOT / "environments/mimicgen_square/scene.xml"))
        cls.data = mujoco.MjData(cls.model)

    def setUp(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        mujoco.mj_forward(self.model, self.data)
        self.controller = FR3TargetController(self.model, self.data)
        self.p, self.r = self.controller.actual_pose()
        self.initial_q = self.controller.hold_q_target.copy()

    def target(self, **changes):
        value = CartesianTarget(self.p.copy(), self.r.copy(), "position",
                                True, True, False, False, "test", True)
        return replace(value, **changes)

    def assert_q(self, q, previous):
        self.assertEqual(q.shape, (7,))
        self.assertTrue(np.isfinite(q).all())
        self.assertTrue(np.all(q >= self.controller.joint_lower))
        self.assertTrue(np.all(q <= self.controller.joint_upper))
        self.assertLessEqual(np.max(np.abs(q - previous)), .8 * .001 + 1e-14)

    def test_existing_site_joint_order_and_caller_ownership(self):
        ctrl, qpos, qvel, time = (self.data.ctrl.copy(), self.data.qpos.copy(),
                                 self.data.qvel.copy(), self.data.time)
        with patch.object(mujoco, "MjModel", side_effect=AssertionError("second model")), \
             patch.object(mujoco, "MjData", side_effect=AssertionError("second data")):
            controller = FR3TargetController(self.model, self.data)
            q = controller.compute(self.target(position=self.p + (.001, 0., 0.)), .001)
        self.assertIs(controller.model, self.model)
        self.assertIs(controller.data, self.data)
        self.assertEqual(self.model.site(controller.ee_site_id).name, EE_SITE)
        self.assertEqual(EE_SITE, "fr3_gripper_center_site")
        for index, name in enumerate(JOINT_NAMES):
            joint = self.model.joint(name).id
            self.assertEqual(controller.qpos_indices[index], self.model.jnt_qposadr[joint])
            self.assertEqual(controller.dof_indices[index], self.model.jnt_dofadr[joint])
        self.assert_q(q, self.initial_q)
        np.testing.assert_array_equal(self.data.ctrl, ctrl)
        np.testing.assert_array_equal(self.data.qpos, qpos)
        np.testing.assert_array_equal(self.data.qvel, qvel)
        self.assertEqual(self.data.time, time)

    def test_reuses_original_functions_without_runner_imports(self):
        helpers = robot_helpers()
        for name in ("condition_target", "compute_joint_target", "dls_pseudoinverse",
                     "exact_nullspace_projector", "read_site_pose"):
            self.assertEqual(getattr(helpers, name).__code__.co_filename, str(BRIDGE_PATH))
        self.assertFalse(hasattr(helpers, "make_desired_target"))
        self.assertFalse(hasattr(helpers, "main"))
        for name in ("rclpy", "rospy", "zmq", "mujoco.viewer"):
            self.assertNotIn(name, sys.modules)

    def test_conditioning_and_dls_output_matches_existing_helper(self):
        c = self.controller
        c.compute(self.target(), .001)  # activation anchors actual before IK
        desired = self.target(position=self.p + (.001, -.0005, .0003))
        # Nonzero posture reference exercises real null-space math as well.
        c.nullspace_posture_reference = self.initial_q + np.linspace(-.01, .01, 7)
        conditioned = c.helpers.condition_target(self.p, self.r, desired.position, self.r,
                                                 np.zeros(3), np.zeros(3), .001)
        expected, diagnostics = c.helpers.compute_joint_target(
            self.model, self.data, c.ee_site_id, c.dof_indices, c.qpos_indices,
            c.joint_lower, c.joint_upper, conditioned[0], conditioned[1],
            self.initial_q, np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv)), .001,
            teleop_mode="full_pose", enable_nullspace_posture=True,
            q_posture_reference=c.nullspace_posture_reference)
        actual = c.compute(desired, .001)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(c.target_position, conditioned[0])
        self.assertTrue(c.diagnostics["nullspace_enabled"])
        self.assertGreater(c.diagnostics["qdot_null_norm"], 0.)
        self.assertEqual(c.diagnostics["teleop_mode"], "full_pose")
        self.assertEqual(c.diagnostics["nullspace_gain"], diagnostics["nullspace_gain"])
        self.assert_q(actual, self.initial_q)

    def test_each_synthetic_translation_yields_bounded_7d_absolute_targets(self):
        for name, delta, expected in axis_cases():
            with self.subTest(name=name):
                pipeline = GeomagicTargetPipeline(self.model, self.data, activation_samples=2)
                previous = self.initial_q
                for state in (sample(0), sample(1), sample(2, position=delta)):
                    command = pipeline.step(state, now=state.timestamp, dt=.001, gripper_command=128)
                    self.assert_q(command.joint_position, previous)
                    previous = command.joint_position
                np.testing.assert_allclose(pipeline.last_target.position - self.p, expected, atol=1e-14)
                self.assertGreater(np.max(np.abs(previous - self.initial_q)), 0.)
                # It is an absolute joint configuration, not a delta near zero.
                self.assertGreater(np.linalg.norm(previous), 1.)
                self.assertEqual(command.gripper_command, 128)

    def test_each_synthetic_rotation_yields_bounded_7d_targets(self):
        for axis in range(3):
            for sign in (1, -1):
                with self.subTest(axis=axis, sign=sign):
                    pipeline = GeomagicTargetPipeline(self.model, self.data, mode="rotation",
                                                      activation_samples=2)
                    vector = np.eye(3)[axis] * sign * .25
                    previous = self.initial_q
                    for state in (sample(0), sample(1), sample(2, rotvec=vector)):
                        command = pipeline.step(state, now=state.timestamp, dt=.001, gripper_command=64)
                        self.assert_q(command.joint_position, previous)
                        previous = command.joint_position
                    np.testing.assert_array_equal(pipeline.last_target.position, self.p)
                    world = ROTATION_SCALE * np.array((vector[2], -vector[0], vector[1]))
                    np.testing.assert_allclose(pipeline.last_target.rotation,
                                               rotation_vector_to_matrix(world) @ self.r, atol=1e-12)
                    self.assertGreater(np.max(np.abs(previous - self.initial_q)), 1e-10)

    def test_mixed_orientation_uses_one_existing_dls_path_without_physics(self):
        for components in ((.3, -.2, 0.), (0., -.2, .4), (.3, -.2, .4)):
            with self.subTest(components=components):
                pipeline = GeomagicTargetPipeline(self.model, self.data, mode="rotation",
                                                  activation_samples=2)
                device_anchor = rotation_vector_to_matrix((.4, -.3, .2))
                delta = (rotation_vector_to_matrix((components[0], 0., 0.))
                         @ rotation_vector_to_matrix((0., components[1], 0.))
                         @ rotation_vector_to_matrix((0., 0., components[2])))
                anchor_vector = matrix_to_rotation_vector(device_anchor)
                for s in (sample(0, rotvec=anchor_vector), sample(1, rotvec=anchor_vector)):
                    pipeline.step(s, now=s.timestamp, dt=.001, gripper_command=128)
                state = sample(2, rotvec=matrix_to_rotation_vector(delta @ device_anchor))
                helpers = pipeline.controller.helpers
                qpos, qvel, ctrl = self.data.qpos.copy(), self.data.qvel.copy(), self.data.ctrl.copy()
                with patch.object(helpers, "compute_joint_target", wraps=helpers.compute_joint_target) as dls, \
                     patch.object(mujoco, "mj_step", side_effect=AssertionError("physics step")):
                    command = pipeline.step(state, now=state.timestamp, dt=.001, gripper_command=128)
                self.assertEqual(dls.call_count, 1)
                self.assertEqual(dls.call_args.kwargs["teleop_mode"], "full_pose")
                self.assertEqual(helpers.compute_joint_target.__code__.co_filename, str(BRIDGE_PATH))
                self.assert_q(command.joint_position, self.initial_q)
                self.assertGreater(np.max(np.abs(command.joint_position - self.initial_q)), 1e-10)
                np.testing.assert_array_equal(pipeline.last_target.position, self.p)
                for key, value in pipeline.controller.diagnostics.items():
                    if key != "teleop_mode":
                        self.assertTrue(np.isfinite(value).all(), key)
                np.testing.assert_array_equal(self.data.qpos, qpos)
                np.testing.assert_array_equal(self.data.qvel, qvel)
                np.testing.assert_array_equal(self.data.ctrl, ctrl)
                self.assertEqual(self.data.time, 0.)

    def test_unclipped_large_mapping_is_still_bounded_by_fr3_conditioning(self):
        """Mapping limits are disabled; the robot side must still rate-limit."""
        helpers = robot_helpers()
        cases = (("position", dict(position=(0., 0., .40)), POSITION_SCALE[0] * .40),  # 280 mm mapped
                 ("rotation", dict(rotvec=(0., 0., 0.)), None))
        for mode, motion, mapped_norm in cases:
            with self.subTest(mode=mode):
                pipeline = GeomagicTargetPipeline(self.model, self.data, mode=mode,
                                                  activation_samples=2)
                if mode == "rotation":
                    world = self.r @ np.array((0., 0., np.deg2rad(90.)))   # 90 deg yaw
                    motion = dict(rotvec=(-world[1], world[2], world[0]))
                states = (sample(0), sample(1), sample(2, **motion))
                previous = self.initial_q
                for state in states:
                    command = pipeline.step(state, now=state.timestamp, dt=.001, gripper_command=0)
                    self.assert_q(command.joint_position, previous)
                    previous = command.joint_position
                target = pipeline.last_target
                if mode == "position":
                    # Mapped target is the full scaled displacement (no 1 mm / 75 mm cap).
                    self.assertAlmostEqual(float(np.linalg.norm(target.position - self.p)),
                                           mapped_norm, places=12)
                else:
                    local = np.linalg.norm(matrix_to_rotation_vector(self.r.T @ target.rotation))
                    self.assertAlmostEqual(float(np.rad2deg(local)), 90. * ROTATION_SCALE, places=9)
                position, rotation = (pipeline.controller.target_position.copy(),
                                      pipeline.controller.target_rotation.copy())
                for i in range(1, 201):
                    command = pipeline.step(states[-1], now=states[-1].timestamp + i * .0002,
                                            dt=.001, gripper_command=0)
                    self.assert_q(command.joint_position, previous)
                    previous = command.joint_position
                    c = pipeline.controller
                    # Existing condition_target speed bounds remain authoritative.
                    self.assertLessEqual(np.linalg.norm(c.target_position - position),
                                         helpers.MAX_TARGET_LINEAR_SPEED * .001 + 1e-12)
                    self.assertLessEqual(
                        np.linalg.norm(matrix_to_rotation_vector(c.target_rotation @ rotation.T)),
                        helpers.MAX_TARGET_ANGULAR_SPEED * .001 + 1e-12)
                    position, rotation = c.target_position.copy(), c.target_rotation.copy()
                # Still converging after 200 ms: the conditioned target did not jump.
                if mode == "position":
                    self.assertLess(np.linalg.norm(position - self.p), mapped_norm)

    def test_zero_error_held_target_produces_stable_command(self):
        target = self.target()
        for _ in range(20):
            q = self.controller.compute(target, .001)
            np.testing.assert_allclose(q, self.initial_q, atol=1e-14, rtol=0)

    def test_stale_deadman_and_clutch_hold_actual_once(self):
        for gates in ({"fresh": False}, {"active": False}, {"clutched": True}):
            with self.subTest(gates=gates):
                c = self.controller
                c.reset()
                c.compute(self.target(), .001)
                c.hold_q_target = self.initial_q + .2
                c.target_position = self.p + .15
                c.target_rotation = rotation_vector_to_matrix((0., 0., np.deg2rad(80))) @ self.r
                target = self.target(position=self.p + 1., **gates)
                np.testing.assert_array_equal(c.compute(target, .001), self.initial_q)
                np.testing.assert_array_equal(c.target_position, self.p)
                np.testing.assert_array_equal(c.target_rotation, self.r)
                # Inactive hold must not chase new actual sag on subsequent ticks.
                self.data.qpos[c.qpos_indices] += .01
                mujoco.mj_forward(self.model, self.data)
                for _ in range(20):
                    np.testing.assert_array_equal(c.compute(target, .001), self.initial_q)
                np.testing.assert_array_equal(c.linear_velocity, np.zeros(3))
                np.testing.assert_array_equal(c.angular_velocity, np.zeros(3))
                self.data.qpos[c.qpos_indices] = self.initial_q
                mujoco.mj_forward(self.model, self.data)

    def test_control_ticks_continue_conditioning_without_mapper_updates(self):
        pipeline = GeomagicTargetPipeline(self.model, self.data, activation_samples=2)
        for state in (sample(0), sample(1), sample(2, position=(.002, 0., 0.))):
            pipeline.step(state, now=state.timestamp, dt=.001, gripper_command=100)
        mapped = pipeline.last_target.position.copy()
        conditioned = pipeline.controller.target_position.copy()
        count = pipeline.mapper.mapper_updates
        previous = pipeline.controller.hold_q_target.copy()
        for i in range(1, 11):
            command = pipeline.step(state, now=state.timestamp + i * .001,
                                    dt=.001, gripper_command=100)
            self.assert_q(command.joint_position, previous)
            previous = command.joint_position
        self.assertEqual(pipeline.mapper.mapper_updates, count)
        np.testing.assert_array_equal(pipeline.last_target.position, mapped)
        self.assertGreater(np.linalg.norm(pipeline.controller.target_position - conditioned), 0.)
        # Caller did not advance physics: this checks rate semantics, not tracking.
        self.assertEqual(self.data.time, 0.)
        before = pipeline.controller.hold_q_target.copy()
        command = pipeline.step(state, now=state.timestamp + .051, dt=.001, gripper_command=100)
        np.testing.assert_array_equal(command.joint_position, self.data.qpos[pipeline.controller.qpos_indices])

    def test_mode_switch_synchronizes_both_components_to_actual(self):
        c = self.controller
        for mode in ("position", "rotation", "position"):
            c.target_position = self.p + .15
            c.target_rotation = rotation_vector_to_matrix((0., 0., 1.4)) @ self.r
            c.hold_q_target = self.initial_q + .2
            c.compute(self.target(mode=mode), .001)
            np.testing.assert_array_equal(c.target_position, self.p)
            np.testing.assert_array_equal(c.target_rotation, self.r)
            np.testing.assert_array_equal(c.hold_q_target, self.initial_q)
            np.testing.assert_array_equal(c.linear_velocity, np.zeros(3))
            np.testing.assert_array_equal(c.angular_velocity, np.zeros(3))

    def test_joint_limits_clip_existing_integration_reference(self):
        c = self.controller
        for edge in (c.joint_lower, c.joint_upper):
            for mode in ("position", "rotation"):
                with self.subTest(edge=edge.tolist(), mode=mode):
                    # Test joint-limit clipping without injecting an unrelated
                    # multi-radian follower backlog into the new hard leash.
                    self.data.qpos[c.qpos_indices] = edge
                    mujoco.mj_forward(self.model, self.data)
                    c.reset()
                    c.compute(self.target(mode=mode), .001)
                    c.hold_q_target = edge.copy()
                    q = c.compute(self.target(mode=mode, position=self.p + (.01, -.01, .01),
                                               rotation=rotation_vector_to_matrix((.3, -.4, .2)) @ self.r),
                                  .001)
                    self.assert_q(q, edge)

    def test_real_teleop_command_and_existing_8d_adapter(self):
        c = self.controller
        q = c.compute(self.target(position=self.p + (.001, 0., 0.)), .001)
        command = c.teleop_command(123.)
        adapter_module = command_adapter()
        self.assertIsInstance(command, adapter_module.TeleopCommand)
        action = adapter_module.OmyFr3MimicAdapter(self.model).convert(command)
        self.assertEqual(action.shape, (8,))
        np.testing.assert_array_equal(action[:7], q)
        self.assertEqual(action[7], 123.)
        command.joint_position[:] = 0.
        np.testing.assert_array_equal(c.hold_q_target, q)  # no mutable alias

    def test_invalid_dt_pose_state_and_gripper_rejected(self):
        c = self.controller
        for dt in (0., -1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                c.compute(self.target(), dt)
        for changes in ({"position": np.array((float("nan"), 0., 0.))},
                        {"rotation": np.zeros((3, 3))}, {"mode": "full_pose"}):
            with self.assertRaises(ValueError):
                c.compute(self.target(**changes), .001)
        for gripper in (-1., 256., float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                c.teleop_command(gripper)
        self.data.qpos[c.qpos_indices[0]] = float("nan")
        with self.assertRaises(ValueError):
            c.compute(self.target(), .001)

    def test_reset_from_authoritative_data(self):
        pipeline = GeomagicTargetPipeline(self.model, self.data, activation_samples=2)
        for state in (sample(0), sample(1), sample(2, position=(.002, 0., 0.))):
            pipeline.step(state, now=state.timestamp, dt=.001, gripper_command=0)
        self.data.qpos[pipeline.controller.qpos_indices[0]] += .01
        mujoco.mj_forward(self.model, self.data)
        pipeline.reset()
        np.testing.assert_array_equal(pipeline.controller.hold_q_target,
                                      self.data.qpos[pipeline.controller.qpos_indices])
        np.testing.assert_array_equal(pipeline.controller.nullspace_posture_reference,
                                      pipeline.controller.hold_q_target)
        self.assertIsNone(pipeline.last_target)
        state = sample(0)
        command = pipeline.step(state, now=state.timestamp, dt=.001, gripper_command=255)
        self.assertFalse(pipeline.last_target.active)
        np.testing.assert_array_equal(command.joint_position, pipeline.controller.hold_q_target)

    def test_production_transport_to_controller_session_and_stale(self):
        sock, clock = DatagramSocket(), [0.]
        with patch("geomagic.bridge.udp_state_client.socket.socket", return_value=sock):
            client = GeomagicUDPStateClient(monotonic=lambda: clock[0])
        pipeline = GeomagicTargetPipeline(self.model, self.data, activation_samples=2)

        def step(seq=None, *, session="one", position=(0., 0., 0.), grey=True, white=False):
            if seq is not None:
                sock.pending.append(encode_packet(GeomagicStatePacket(
                    1, session, seq, 90000. + seq, "synthetic-source",
                    position, (0., 0., 0., 1.), grey, white, True)))
            command = pipeline.step_from_client(client, now=clock[0], dt=.001, gripper_command=111)
            self.assertFalse(client.status().session_changed)
            clock[0] += .005
            return command

        try:
            step(0)
            self.assertFalse(pipeline.last_target.active)
            step(1)
            self.assertTrue(pipeline.last_target.reanchored)
            command = step(2, position=(.001, 0., 0.))
            held_position = pipeline.last_target.position.copy()
            count = pipeline.mapper.mapper_updates
            step()
            self.assertEqual(pipeline.mapper.mapper_updates, count)
            clock[0] += .060
            before = pipeline.controller.hold_q_target.copy()
            command = step()
            self.assertFalse(pipeline.last_target.fresh)
            np.testing.assert_array_equal(command.joint_position, self.data.qpos[pipeline.controller.qpos_indices])
            step(0, session="two", position=(1., 1., 1.))
            step(1, session="two", position=(1., 1., 1.))
            self.assertTrue(pipeline.last_target.reanchored)
            np.testing.assert_array_equal(pipeline.last_target.position, self.p)
            command = step(2, session="two", grey=False)
            self.assertFalse(pipeline.last_target.active)
            self.assertEqual(command.gripper_command, 111.)  # buttons do not drive gripper
        finally:
            client.close()

    def test_large_tracking_error_all_pipeline_transitions(self):
        """All command owners discard 150 mm/80 deg intent; actual moves too."""
        for mode in ("position", "rotation"):
            for event in ("deadman", "clutch", "stale", "mode", "session", "missed_stale"):
                with self.subTest(mode=mode, event=event):
                    self.setUp()
                    pipe = GeomagicTargetPipeline(self.model, self.data, mode=mode,
                                                 activation_samples=2)
                    c = pipe.controller
                    def step(state, **kwargs):
                        return pipe.step(state, now=kwargs.pop("now", state.timestamp),
                                         dt=.001, gripper_command=0, **kwargs)
                    step(sample(0)); step(sample(1))
                    # Partial tracking already happened: actual differs from
                    # reset/home, while every command owner is still far ahead.
                    self.data.qpos[c.qpos_indices] += .02
                    mujoco.mj_forward(self.model, self.data)
                    p, r = c.actual_pose()
                    actual_q = self.data.qpos[c.qpos_indices].copy()
                    old_p = p + np.array((.15, 0., 0.))
                    old_r = rotation_vector_to_matrix((0., 0., np.deg2rad(80))) @ r
                    c.target_position, c.target_rotation = old_p.copy(), old_r.copy()
                    c.hold_q_target = actual_q + .2
                    c.linear_velocity[:] = .05
                    c.angular_velocity[:] = 1.
                    pipe.mapper._position, pipe.mapper._rotation = old_p.copy(), old_r.copy()
                    vendor = pipe.mapper._controller
                    vendor.target = old_p.copy()
                    if mode == "rotation":
                        vendor.target_rotation = old_r.copy()
                    posture_reference = c.nullspace_posture_reference.copy()

                    def check():
                        t = pipe.last_target
                        np.testing.assert_allclose(t.position, p, atol=1e-14, rtol=0)
                        np.testing.assert_allclose(t.rotation, r, atol=1e-14, rtol=0)
                        np.testing.assert_allclose(c.target_position, p, atol=1e-14, rtol=0)
                        np.testing.assert_allclose(c.target_rotation, r, atol=1e-14, rtol=0)
                        np.testing.assert_array_equal(c.hold_q_target, actual_q)
                        np.testing.assert_array_equal(c.linear_velocity, np.zeros(3))
                        np.testing.assert_array_equal(c.angular_velocity, np.zeros(3))
                        np.testing.assert_array_equal(c.nullspace_posture_reference, posture_reference)
                    if event in ("deadman", "clutch"):
                        step(sample(2, grey=event != "deadman", white=event == "clutch")); check()
                        # Actual drift while held: capture again at RE-engagement.
                        held = c.hold_q_target.copy()
                        self.data.qpos[c.qpos_indices] += .01
                        mujoco.mj_forward(self.model, self.data)
                        step(sample(3, grey=event != "deadman", white=event == "clutch"))
                        np.testing.assert_array_equal(c.hold_q_target, held)
                        p, r = c.actual_pose()
                        actual_q = self.data.qpos[c.qpos_indices].copy()
                        step(sample(4, position=(1., 2., 3.), rotvec=(.5, -.2, .3))); check()
                        self.assertTrue(pipe.last_target.reanchored)
                    elif event == "stale":
                        step(sample(1), now=.080); check()
                        step(sample(2, time=.085)); check()
                        step(sample(3, time=.090)); check()
                    elif event == "missed_stale":
                        step(sample(2, time=.080)); check()
                        step(sample(3, time=.085)); check()
                    elif event == "session":
                        step(sample(0, time=.015, session="restart"), session_changed=True); check()
                        step(sample(1, time=.020, session="restart")); check()
                    else:
                        pipe.set_mode("rotation" if mode == "position" else "position"); check()
                        step(sample(2)); check()
                        step(sample(3)); check()

    def test_backlog_large_rotation_partial_tracking_then_five_degrees(self):
        pipe = GeomagicTargetPipeline(self.model, self.data, mode="rotation", activation_samples=2)
        def step(state):
            return pipe.step(state, now=state.timestamp, dt=.001, gripper_command=0)
        step(sample(0)); step(sample(1))
        step(sample(2, rotvec=(0., 0., np.deg2rad(90))))
        old_rotation = pipe.last_target.rotation.copy()
        # Only partial following (actual changed by ~10 deg); no physics/device.
        self.data.qpos[pipe.controller.qpos_indices[0]] += np.deg2rad(10)
        mujoco.mj_forward(self.model, self.data)
        p, r = pipe.controller.actual_pose()
        old_error = np.degrees(np.linalg.norm(matrix_to_rotation_vector(old_rotation @ r.T)))
        self.assertGreater(old_error, 60. * ROTATION_SCALE)
        pipe.controller.hold_q_target += .2
        pipe.controller.target_rotation = old_rotation.copy()
        step(sample(3, grey=False)); step(sample(4, rotvec=(.4, -.2, .3)))
        np.testing.assert_allclose(pipe.last_target.rotation, r, atol=1e-14)
        np.testing.assert_array_equal(pipe.last_target.position, p)
        np.testing.assert_array_equal(pipe.controller.hold_q_target,
                                      self.data.qpos[pipe.controller.qpos_indices])
        anchor = rotation_vector_to_matrix((.4, -.2, .3))
        delta = rotation_vector_to_matrix((0., 0., np.deg2rad(5)))
        step(sample(5, rotvec=matrix_to_rotation_vector(delta @ anchor)))
        expected = rotation_vector_to_matrix((np.deg2rad(5) * ROTATION_SCALE, 0., 0.)) @ r
        np.testing.assert_allclose(pipe.last_target.rotation, expected, atol=1e-12)
        angle = np.degrees(np.linalg.norm(matrix_to_rotation_vector(pipe.last_target.rotation @ r.T)))
        self.assertAlmostEqual(angle, 5. * ROTATION_SCALE, places=10)
        print(f"BACKLOG old_actual_error_deg={old_error:.9f} new_input_deg=5 new_target_deg={angle:.9f}")

    def test_opt_in_diagnostics_do_not_change_commands_and_count_held_input(self):
        tracker = TrackingDiagnostics(capacity=16)
        instrumented = GeomagicTargetPipeline(self.model, self.data, mode="rotation",
                                             activation_samples=2, tracking=tracker)
        normal = GeomagicTargetPipeline(self.model, self.data, mode="rotation", activation_samples=2)
        states = (sample(0), sample(1), sample(2, rotvec=(0., 0., 2.8)))
        for tick in range(12):
            state = states[min(tick, 2)]
            tracker.begin(.001)
            a = instrumented.step(state, now=state.timestamp, dt=.001, gripper_command=0)
            b = normal.step(state, now=state.timestamp, dt=.001, gripper_command=0)
            np.testing.assert_array_equal(a.joint_position, b.joint_position)
            tracker.capture(instrumented, controller_updated=bool(tracker.dls_calls),
                            control_dt=.001 if tracker.dls_calls else None)
            tracker.finish(.0001)
        report = tracker.summary()
        self.assertEqual(report["counts"]["active.dls_calls"], 10)
        self.assertEqual(report["counts"]["active.held_input.dls_calls"], 9)
        self.assertEqual(report["metrics"]["active.dls_calls_per_tick"]["max"], 1.)
        self.assertGreater(report["metrics"]["active.angular_speed_limited"]["mean"], .8)
        self.assertEqual(report["metrics"]["active.translation_speed_limited"]["max"], 0.)
        self.assertEqual(report["metrics"]["active.raw_conditioned_position_error_mm"]["max"], 0.)
        self.assertAlmostEqual(report["metrics"]["active.raw_orientation_error_deg"]["max"],
                               np.degrees(2.8 * ROTATION_SCALE), places=10)
        for name in ("state_update", "mapping", "conditioning", "dls", "env_step", "other"):
            self.assertIn(f"active.{name}_ms", report["metrics"])

    def test_rotation_subspace_holds_position_with_both_mapper_and_conditioner(self):
        pipe = GeomagicTargetPipeline(self.model, self.data, mode="rotation", activation_samples=2)
        for seq in range(60):
            state = sample(seq, time=seq * .001, position=(seq * .02, -.1, .3),
                           rotvec=np.array((.02, -.01, .015)) * seq)
            pipe.step(state, now=state.timestamp, dt=.001, gripper_command=0)
            np.testing.assert_array_equal(pipe.last_target.position, self.p)
            np.testing.assert_array_equal(pipe.controller.target_position, self.p)


if __name__ == "__main__":
    unittest.main()
