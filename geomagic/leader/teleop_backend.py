"""Phase-2A adapter for the existing runner; caller owns model and physics."""

from dataclasses import asdict, dataclass
import math
import time

import mujoco
import numpy as np

from geomagic.bridge.udp_state_client import GeomagicUDPStateClient
from geomagic.teleop.pipeline import GeomagicTargetPipeline
from geomagic.teleop.diagnostics import TrackingDiagnostics
from geomagic.teleop.gravity_support import PositionServoGravitySupport
from geomagic.teleop.input_deadband import DeadbandConfig
from geomagic.teleop.runtime_profile import get_profiler
from geomagic.teleop.sync_capture import get_sync_capture


@dataclass(frozen=True)
class GeomagicTeleopState:
    mode: str
    active: bool
    input_fresh: bool
    clutch_count: int
    command_position: np.ndarray
    target_position: np.ndarray
    actual_position: np.ndarray


class GeomagicTeleopBackend:
    """Runtime contract with independent native [0, 255] gripper commands.

    Grey is deadman; white is clutch. Each Phase-2A reanchor event counts once
    toward clutch_count (activation/reengagement, clutch release, or rebase).
    """

    def __init__(self, model, data, *, bind_address=("127.0.0.1", 5558),
                 teleop_mode="position", freeze_fr3_motion=False,
                 activation_samples=50, monotonic=time.monotonic, deadband_config=None):
        self.model, self.data = model, data
        self.input_backend = "geomagic"
        self.bind_address = bind_address
        self._clock = monotonic
        self.profile = get_profiler()
        self.sync = get_sync_capture()
        self.tracking = TrackingDiagnostics.from_environment()
        if self.profile and self.tracking:
            self.profile.instrument(self.tracking, {'capture':'ui.diagnostics_stats', 'publish':'ui.diagnostics_publish'})
        self.pipeline = GeomagicTargetPipeline(
            model, data, mode=self._mode_value(teleop_mode),
            activation_samples=activation_samples,
            tracking=self.tracking,
            deadband_config=(DeadbandConfig.from_environment() if deadband_config is None
                             else deadband_config),
        )
        controller = self.pipeline.controller
        if self.sync:
            from geomagic.teleop.pose_mapper import POSITION_SCALE, ROTATION_SCALE
            self.sync.settings = dict(position_scale=list(POSITION_SCALE), rotation_scale=ROTATION_SCALE,
                                      deadband=asdict(self.pipeline.deadband.config),
                                      geomagic_orientation_conditioning_gain=controller.helpers.TARGET_ROTATION_KP)
        controller.sync = self.sync
        self.pipeline.sync = self.sync
        if self.profile:
            self.profile.instrument(self.pipeline, {'update_mapping':'rotation.mapping_total'})
            self.profile.instrument(self.pipeline.mapper, {'update':'rotation.fresh_sample'})
            self.profile.instrument(self.pipeline.deadband, {'update':'rotation.deadband'})
        self.fr3_qpos_indices = controller.qpos_indices.copy()
        self.fr3_dof_indices = controller.dof_indices.copy()
        self.fr3_actuator_indices = np.array([
            model.actuator(f"fr3_joint{i}").id for i in range(1, 8)
        ])
        self.fr3_gripper_actuator_id = model.actuator("fr3_gripper").id
        # Opt-in response tuning; the frozen Square/default path stays unchanged.
        self._gripper_velocity_bias = float(model.actuator_biasprm[self.fr3_gripper_actuator_id, 2])
        self._gripper_close_damping_scale = 1.0
        self.freeze_fr3_motion = bool(freeze_fr3_motion)
        self.frozen_gripper_command = float(data.ctrl[self.fr3_gripper_actuator_id])
        self.gripper_command = 0.0
        self.mode_change_events = []
        self.client = None
        self.gravity_support = PositionServoGravitySupport(
            model, data, self.fr3_actuator_indices, self.fr3_dof_indices)
        self.control_period = 2 * float(model.opt.timestep)
        if self.profile:
            self.profile.instrument(self.gravity_support, {'capture':'controller.gravity_capture', 'command':'controller.gravity_command'})
        self.reset_controller_state()

    @staticmethod
    def _mode_value(mode):
        value = str(getattr(mode, "value", mode)).lower()
        return "rotation" if value == "rotation_3d" else value

    def initialize(self):
        if self.client is None:
            self.client = GeomagicUDPStateClient(
                self.bind_address, monotonic=self._clock, max_packets_per_poll=256,
            )
            if self.profile:
                self.profile.instrument(self.client, {'poll':'input.udp_receive_decode', 'latest_state':'input.fresh_state'})

    def reset_controller_state(self):
        if self._gripper_close_damping_scale != 1.0:
            self.model.actuator_biasprm[self.fr3_gripper_actuator_id, 2] = self._gripper_velocity_bias
        self.pipeline.reset()
        self._supported_actual_hold()
        self._last_control_sim_time = float(self.data.time)
        self._episode_start_sim_time = float(self.data.time)
        self.last_control_dt = None
        self.control_updates = 0
        self.physics_command_ticks = 0
        self.input_fresh = False
        self.teleop_active = False

        self.clutch_count = 0
        self._latest_state = None

    def _supported_actual_hold(self):
        self.gravity_support.capture()
        logical = self.pipeline.controller.hold_q_target
        self.hold_q_target = (logical.copy() if self.freeze_fr3_motion
                              else self.gravity_support.command(logical))

    def update_command(self, dt):
        if self.client is None:
            raise RuntimeError("initialize() must be called before teleoperation")
        if self.tracking:
            self.tracking.begin(dt)
            before = self.tracking.clock()
        self.client.poll()
        changed = self.client.status().session_changed
        self._latest_state = self.client.latest_state()
        if self.sync:
            self.sync.input(self._latest_state, float(self.data.time),
                            accepted_ns=getattr(self.client, 'accepted_ns', None), mode=self.pipeline.mapper.mode)
        if self.profile:
            self.profile.sample(self._latest_state)
        if self.tracking:
            self.tracking.stages["state_update"] += self.tracking.clock() - before
        target = self.pipeline.update_mapping(
            self._latest_state, now=self._clock(), session_changed=changed)
        if changed:
            self.client.acknowledge_session_change()
        if self.profile:
            self.profile.stage('T2')
        controller = self.pipeline.controller
        enabled = bool(target.fresh and target.active and not target.clutched)
        if self.profile:
            self.profile.phase = target.mode if enabled else 'idle'
            if not enabled:
                self.profile.last_qtarget_input_time = None
        edge = (target.synchronized or target.reanchored
                or enabled != controller._enabled or controller.mode != target.mode)
        elapsed = float(self.data.time) - self._last_control_sim_time
        if elapsed < -1e-12:
            raise ValueError("simulation clock moved backwards; reset controller state")
        updated = False
        if edge:
            # Synchronization does no IK. The nominal dt only validates this
            # call; rate integration starts at this actual-state edge.
            controller.compute(target, float(self.model.opt.timestep))
            if not self.freeze_fr3_motion:
                self._supported_actual_hold()
            self._last_control_sim_time = float(self.data.time)
        elif enabled and elapsed >= self.control_period - 1e-12:
            controller.compute(target, elapsed)
            self._last_control_sim_time = float(self.data.time)
            self.last_control_dt = elapsed
            self.control_updates += 1
            if self.profile:
                self.profile.stage('T4')
                self.profile.arm_response(self.data.qpos[self.fr3_qpos_indices])
            updated = True
            if not self.freeze_fr3_motion:
                self.hold_q_target = self.gravity_support.command(controller.hold_q_target)
        self.physics_command_ticks += 1
        self.input_fresh = target.fresh
        self.teleop_active = target.active and not target.clutched
        if target.reanchored and not target.clutched:
            self.clutch_count += 1
        if self.tracking:
            self.tracking.capture(self.pipeline, controller_updated=updated,
                                  control_dt=elapsed if updated else None,
                                  servo_q_target=self.hold_q_target)

    def finish_diagnostic_tick(self, env_step_s):
        if self.tracking:
            self.tracking.finish(env_step_s)

    def apply_controller(self, model, data, dt):
        """Write computed targets; no IK, conditioning, or physics here.

        The runner then passes the same command through adapter/env.control()
        and advances the shared physics once.
        """
        if model is not self.model or data is not self.data:
            raise ValueError("Geomagic backend requires its caller-owned model/data")
        data.ctrl[self.fr3_actuator_indices] = self.hold_q_target
        data.ctrl[self.fr3_gripper_actuator_id] = (
            self.frozen_gripper_command if self.freeze_fr3_motion
            else self.gripper_command
        )
        if self._gripper_close_damping_scale != 1.0:
            actuator = self.fr3_gripper_actuator_id
            # Native command/position gains still define exactly the same target.
            position_force = (model.actuator_gainprm[actuator, 0] * data.ctrl[actuator]
                              + model.actuator_biasprm[actuator, 1] * data.actuator_length[actuator])
            closing = not self.freeze_fr3_motion and position_force < -1e-9
            model.actuator_biasprm[actuator, 2] = self._gripper_velocity_bias * (
                self._gripper_close_damping_scale if closing else 1.0)

    def set_gripper_close_damping_scale(self, scale=1.0):
        """Tune closing only on this caller-owned model; 1 preserves the default.

        Opening, position gain, native target semantics and force limits stay
        unchanged. There is no command interpolation or change to action data.
        """
        if isinstance(scale, (bool, np.bool_)):
            raise ValueError("gripper close damping scale must be finite and in (0, 1]")
        scale = float(scale)
        if not math.isfinite(scale) or not 0 < scale <= 1:
            raise ValueError("gripper close damping scale must be finite and in (0, 1]")
        self._gripper_close_damping_scale = scale
        self.model.actuator_biasprm[self.fr3_gripper_actuator_id, 2] = self._gripper_velocity_bias

    def set_teleop_mode(self, mode):
        new_mode = self._mode_value(mode)
        old_mode = self.pipeline.mapper.mode
        if not self.pipeline.set_mode(new_mode):
            return False
        if not self.freeze_fr3_motion:
            self._supported_actual_hold()
        self._last_control_sim_time = float(self.data.time)
        self.teleop_active = False
        self.mode_change_events.append({
            "timestamp": float(self.data.time), "wall_time_unix_s": time.time(),
            "old_mode": old_mode, "new_mode": new_mode,
            "clutch_count": int(self.clutch_count),
        })
        return True

    def set_gripper_command(self, value):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("gripper command must be finite and in native [0, 255]")
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 255:
            raise ValueError("gripper command must be finite and in native [0, 255]")
        self.gripper_command = value

    def get_state(self):
        controller = self.pipeline.controller
        target = self.pipeline.last_target
        actual, _ = controller.actual_pose()
        return GeomagicTeleopState(
            mode=self.pipeline.mapper.mode, active=self.teleop_active,
            input_fresh=self.input_fresh, clutch_count=self.clutch_count,
            command_position=(controller.target_position if target is None
                              else target.position).copy(),
            target_position=controller.target_position.copy(),
            actual_position=actual.copy(),
        )

    @staticmethod
    def _pose(position, rotation):
        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quaternion, rotation.reshape(9))
        return np.concatenate((quaternion, position))

    def diagnostic_state(self, *, detailed=True):
        controller = self.pipeline.controller
        target = self.pipeline.last_target
        sample = self._latest_state
        if not detailed:
            return {
                'teleop_mode': np.asarray(self.pipeline.mapper.mode),
                'input_fresh': np.asarray(self.input_fresh),
                'teleop_active': np.asarray(self.teleop_active),
                'clutch_count': np.asarray(self.clutch_count, dtype=np.int64),
                'gripper_command': np.asarray(self.gripper_command),
            }
        actual_position, actual_rotation = controller.actual_pose()
        raw = self.pipeline.last_raw_target
        deadband = self.pipeline.deadband
        result = {
            "teleop_mode": np.asarray(self.pipeline.mapper.mode),
            "input_fresh": np.asarray(self.input_fresh, dtype=np.bool_),
            "teleop_active": np.asarray(self.teleop_active, dtype=np.bool_),
            "clutch_count": np.asarray(self.clutch_count, dtype=np.int64),
            "gripper_command": np.asarray(self.gripper_command, dtype=np.float64),
            "command_ee_pose": self._pose(
                controller.target_position if target is None else target.position,
                controller.target_rotation if target is None else target.rotation),
            "target_ee_pose": self._pose(controller.target_position, controller.target_rotation),
            "actual_ee_pose": self._pose(actual_position, actual_rotation),
            "raw_mapped_ee_pose": self._pose(
                controller.target_position if raw is None else raw.position,
                controller.target_rotation if raw is None else raw.rotation),
            "deadband_translation_active": np.asarray(deadband.translation.active),
            "deadband_rotation_active": np.asarray(deadband.rotation.active),
            "deadband_translation_activation_count": np.asarray(deadband.translation.activations),
            "deadband_rotation_activation_count": np.asarray(deadband.rotation.activations),
            "deadband_translation_suppressed_samples": np.asarray(deadband.translation.suppressed),
            "deadband_rotation_suppressed_samples": np.asarray(deadband.rotation.suppressed),
            # Before any packet, finite placeholders are marked unavailable.
            "geomagic_state_available": np.asarray(sample is not None),
            "geomagic_position_m": np.asarray(
                sample.position_xyz if sample is not None else (0., 0., 0.)),
            "geomagic_orientation_xyzw": np.asarray(
                sample.orientation_xyzw if sample is not None else (0., 0., 0., 1.)),
            "grey_button": np.asarray(sample is not None and sample.grey_button),
            "white_button": np.asarray(sample is not None and sample.white_button),
            "clutched": np.asarray(target is not None and target.clutched),
        }
        if self.tracking:
            result.update({key: np.asarray(value) for key, value
                           in self.tracking.rotation_command.values.items()})
        return result

    def input_metrics(self):
        if self.client is None:
            return {"received_messages": 0, "rejected_messages": 0,
                    "last_message_age_s": None}
        status = self.client.status()  # Never poll or consume session events.
        result = asdict(status)
        result.update(
            received_messages=status.packets_received,
            rejected_messages=(status.invalid_packets + status.duplicate_packets
                               + status.reordered_packets + status.unsupported_version_packets),
            last_message_age_s=None if status.age_ms is None else status.age_ms / 1000,
        )
        result["control_schedule"] = {
            "geomagic_orientation_conditioning_gain": self.pipeline.controller.helpers.TARGET_ROTATION_KP,
            "physics_dt_s": float(self.model.opt.timestep),
            "cartesian_control_period_sim_s": self.control_period,
            "cartesian_control_hz_sim": 1. / self.control_period,
            "control_updates": self.control_updates,
            "command_ticks": self.physics_command_ticks,
            "simulation_elapsed_s": float(self.data.time) - self._episode_start_sim_time,
            "last_control_dt_sim_s": self.last_control_dt,
            "gravity_support": "captured_actual_gravity_over_kp_servo_offset",
            "gravity_support_clipped": self.gravity_support.clipped,
        }
        result["input_deadband"] = self.pipeline.deadband.status()
        if self.tracking:
            result["tracking_diagnostics"] = self.tracking.publish()
        return result

    def shutdown(self):
        if self.profile:
            self.profile.write()
        if self.tracking:
            self.tracking.publish(force=True)
        if self.client is not None:
            self.client.close()
            self.client = None
        self.input_fresh = False
        self.teleop_active = False
        if self.sync:
            self.sync.write()
