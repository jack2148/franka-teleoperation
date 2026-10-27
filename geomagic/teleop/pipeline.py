"""Small synthetic/network-to-command harness; caller owns all simulation."""

from dataclasses import replace
import math

from geomagic.teleop.fr3_target_controller import FR3TargetController
from geomagic.teleop.pose_mapper import GeomagicPoseMapper
from geomagic.teleop.input_deadband import PoseInputDeadband


class GeomagicTargetPipeline:
    def __init__(self, model, data, *, mode="position", activation_samples=50, tracking=None,
                 deadband_config=None):
        self.tracking = tracking
        self.controller = FR3TargetController(model, data, tracking=tracking)
        self.mapper = GeomagicPoseMapper(*self.controller.actual_pose(), mode=mode,
                                        activation_samples=activation_samples)
        self.last_target = None
        self.last_raw_target = None
        self.deadband = PoseInputDeadband(deadband_config)
        self._mapped_sample = None
        self.sync = None

    def update_mapping(self, sample, *, now, session_changed=False):
        """Map new accepted state; retain per-physics receive-age safety checks."""
        before = self.tracking.clock() if self.tracking else None
        age = None if sample is None else now - sample.timestamp
        age_fresh = bool(age is not None and math.isfinite(age)
                         and -.01 <= age <= self.mapper.stale_timeout_s)
        previous = self.last_raw_target
        # latest_state() returns a dataclass copy with current freshness. Check
        # contents too: identical accepted state can bypass SO(3) validation;
        # changed validity or any other field takes the full validated path.
        same_sample = (sample is self._mapped_sample or sample == self._mapped_sample)
        reuse = (same_sample and previous is not None
                 and not session_changed and not self.mapper._pending_sync
                 and math.isfinite(now)
                 and (not previous.fresh or age_fresh == previous.fresh)
                 and not (not previous.fresh and age_fresh and sample is not None))
        if reuse:
            target = replace(previous, mapper_updated=False, synchronized=False,
                             reanchored=False, reason="held_sample")
        else:
            position, rotation = self.controller.actual_pose()
            target = self.mapper.update(sample, now=now, actual_position=position,
                                        actual_rotation=rotation, session_changed=session_changed)
            self._mapped_sample = sample
        if self.tracking:
            self.tracking.stages["mapping"] += self.tracking.clock() - before
        self.last_raw_target = target
        if self.sync and target.mapper_updated:
            self.sync.stage('T2')
        # Keep the deadband algorithm/config unchanged. It already filters
        # rotation poses using the legacy rotation channel name. Projection is
        # complete before this dispatch, so discarded tool X/Y cannot open it.
        deadband_target = replace(target, mode="rotation") if target.mode == "rotation_z" else target
        filtered = self.deadband.update(deadband_target, now=sample.timestamp if sample is not None else now)
        target = replace(filtered, mode="rotation_z") if target.mode == "rotation_z" else filtered
        self.last_target = target
        return target

    def step(self, sample, *, now, dt, gripper_command, session_changed=False):
        target = self.update_mapping(sample, now=now, session_changed=session_changed)
        self.controller.compute(target, dt)
        return self.controller.teleop_command(gripper_command)

    def step_from_client(self, client, *, now, dt, gripper_command):
        """Client is caller-owned; consume its event once after mapper handles it."""
        client.poll()
        changed = client.status().session_changed
        result = self.step(client.latest_state(), now=now, dt=dt,
                           gripper_command=gripper_command, session_changed=changed)
        if changed:
            self._mapped_sample = None
            client.acknowledge_session_change()
        return result

    def set_mode(self, mode):
        position, rotation = self.controller.actual_pose()
        changed = self.mapper.set_mode(mode, actual_position=position, actual_rotation=rotation)
        if changed:
            self.controller.synchronize_actual()
            self.controller.mode = mode
            self.controller._enabled = False
            self.last_target = self.mapper._snapshot(reason="mode_change", synchronized=True)
            self.last_raw_target = self.last_target
            self.deadband.reset(position, rotation)
        return changed

    def reset(self):
        self.controller.reset()
        self.mapper.reset(*self.controller.actual_pose())
        self.last_target = None
        self.last_raw_target = None
        self.deadband.reset(*self.controller.actual_pose())
        self._mapped_sample = None
