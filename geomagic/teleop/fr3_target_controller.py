"""FR3 command generation on caller-owned model/data; no physics or I/O."""

import math
import numpy as np

from geomagic.teleop.pose_mapper import CartesianTarget, MODES
from geomagic.teleop.robot_helpers import (
    GEOMAGIC_ORIENTATION_CONDITIONING_GAIN, command_adapter, robot_helpers,
)
from geomagic.teleop.runtime_rotation import validate_rotation_matrix
from geomagic.teleop.runtime_profile import get_profiler
from geomagic.teleop.anti_windup import JointCommandAntiWindup


JOINT_NAMES = tuple(f"fr3_joint{i}" for i in range(1, 8))
EE_SITE = "fr3_gripper_center_site"


class FR3TargetController:
    """Reuse existing conditioning, full-pose DLS, limits, and nullspace.

    compute() returns seven ABSOLUTE joint targets. It never writes data.ctrl,
    advances physics, creates a model/data, or builds an OMY FK model.
    """

    def __init__(self, model, data, *, tracking=None,
                 orientation_conditioning_gain=GEOMAGIC_ORIENTATION_CONDITIONING_GAIN,
                 anti_windup_config=None):
        self.model, self.data = model, data
        self.tracking = tracking
        self.sync = None
        self.helpers = robot_helpers(orientation_conditioning_gain)
        self.anti_windup = JointCommandAntiWindup(anti_windup_config)
        self.ee_site_id = model.site(EE_SITE).id
        joints = np.array([model.joint(name).id for name in JOINT_NAMES])
        self.qpos_indices = model.jnt_qposadr[joints].astype(int)
        self.dof_indices = model.jnt_dofadr[joints].astype(int)
        self.joint_lower = model.jnt_range[joints, 0].copy()
        self.joint_upper = model.jnt_range[joints, 1].copy()
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))
        self.reset()

    def actual_pose(self):
        return self.helpers.read_site_pose(self.data, self.ee_site_id)

    def reset(self):
        self.target_position, self.target_rotation = self.actual_pose()
        self.hold_q_target = self.data.qpos[self.qpos_indices].copy()
        if not np.isfinite(self.hold_q_target).all():
            raise ValueError("FR3 reset joint state is non-finite")
        self.nullspace_posture_reference = self.hold_q_target.copy()
        self.linear_velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.mode = None
        self.diagnostics = {}
        self._enabled = False
        self.anti_windup.reset()

    def synchronize_actual(self):
        """Reset command/rate state without retuning the null-space reference."""
        position, rotation = self.actual_pose()
        actual_q = self.data.qpos[self.qpos_indices].copy()
        if not np.isfinite(actual_q).all():
            raise ValueError("FR3 actual joint state is non-finite")
        self.target_position, self.target_rotation = position, rotation
        # Actual normally lies inside the joint ranges. Preserve the existing
        # legal-command bounds if physics has overshot a range slightly.
        self.hold_q_target = np.clip(actual_q, self.joint_lower, self.joint_upper)
        self.linear_velocity.fill(0)
        self.angular_velocity.fill(0)
        self.diagnostics = {}
        self.anti_windup.synchronize()

    def compute(self, target: CartesianTarget, dt: float):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        position = np.asarray(target.position, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("Cartesian position must be a finite 3-vector")
        rotation = validate_rotation_matrix(target.rotation)
        if target.mode not in MODES:
            raise ValueError("unsupported operator mode")
        if not np.isfinite(self.data.qpos[self.qpos_indices]).all():
            raise ValueError("FR3 joint state is non-finite")
        enabled = bool(target.fresh and target.active and not target.clutched)
        mode_changed = self.mode is not None and self.mode != target.mode
        synchronize = (target.synchronized or target.reanchored
                       or enabled != self._enabled or mode_changed)
        self.mode, self._enabled = target.mode, enabled
        if synchronize:
            self.synchronize_actual()
            # The edge itself commands actual q; no IK/null-space motion is
            # spent before a subsequent relative input/control tick.
            return self.hold_q_target.copy()
        if not enabled:
            # Keep one actual-state snapshot, not a continuously moving hold.
            return self.hold_q_target.copy()
        if target.mode == "position":
            rotation = self.target_rotation
            self.angular_velocity.fill(0)
        elif target.mode in ("rotation_z", "rotation"):
            position = self.target_position
            self.linear_velocity.fill(0)
        tracking = self.tracking
        if tracking:
            h = self.helpers
            error_p = np.linalg.norm(position - self.target_position)
            error_r = np.linalg.norm(h.matrix_to_rotvec(rotation @ self.target_rotation.T))
            tracking.flags["translation_speed_limited"] = bool(
                min(h.POSITION_KP * error_p, np.sqrt(2 * h.MAX_TARGET_LINEAR_ACCEL * error_p))
                > h.MAX_TARGET_LINEAR_SPEED)
            tracking.flags["angular_speed_limited"] = bool(
                min(h.TARGET_ROTATION_KP * error_r, np.sqrt(2 * h.MAX_TARGET_ANGULAR_ACCEL * error_r))
                > h.MAX_TARGET_ANGULAR_SPEED)
            before = tracking.clock()
        if self.sync:
            self.sync.stage('T3_begin')
        conditioned = self.helpers.condition_target(
            self.target_position, self.target_rotation, position, rotation,
            self.linear_velocity, self.angular_velocity, dt,
        )
        if self.sync:
            self.sync.stage('T3')
        if get_profiler():
            get_profiler().stage('T3')
        if tracking:
            tracking.stages["conditioning"] += tracking.clock() - before
            before = tracking.clock()
        if self.sync:
            self.sync.stage('T4_begin')
        qtarget, diagnostics = self.helpers.compute_joint_target(
            self.model, self.data, self.ee_site_id, self.dof_indices, self.qpos_indices,
            self.joint_lower, self.joint_upper, conditioned[0], conditioned[1],
            self.hold_q_target, self.jacp, self.jacr, dt,
            teleop_mode="full_pose",
            enable_nullspace_posture=self.helpers.ENABLE_NULLSPACE_POSTURE,
            q_posture_reference=self.nullspace_posture_reference,
        )
        qtarget = self.anti_windup.apply(
            self.hold_q_target, self.data.qpos[self.qpos_indices], qtarget,
            self.joint_lower, self.joint_upper)
        if self.sync:
            self.sync.stage('T4')
        if tracking:
            tracking.stages["dls"] += tracking.clock() - before
            tracking.dls_calls += 1
        if qtarget.shape != (7,) or not np.isfinite(qtarget).all():
            raise ValueError("FR3 helper produced an invalid 7D target")
        self.target_position, self.target_rotation = conditioned[0], conditioned[1]
        self.linear_velocity, self.angular_velocity = conditioned[5], conditioned[6]
        self.hold_q_target = qtarget.copy()
        self.diagnostics = diagnostics
        return qtarget.copy()

    def teleop_command(self, gripper_command):
        """Use the real existing command class; gripper is explicitly external."""
        if (isinstance(gripper_command, bool) or not math.isfinite(gripper_command)
                or not 0 <= gripper_command <= 255):
            raise ValueError("gripper command must be finite and in native [0, 255]")
        return command_adapter().TeleopCommand(self.hold_q_target.copy(), float(gripper_command))
