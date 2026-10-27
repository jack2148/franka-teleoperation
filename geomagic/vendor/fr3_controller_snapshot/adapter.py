"""Thin adapter from the existing FR3 teleop output to Mimic's FR3 action."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class TeleopCommand:
    """Command already produced by the migrated teleoperation controller."""

    joint_position: np.ndarray
    gripper_command: float


class OmyFr3MimicAdapter:
    """Map 7D FR3 q targets plus native gripper command to an 8D action."""

    ARM_ACTUATORS = tuple(f"fr3_joint{i}" for i in range(1, 8))
    GRIPPER_ACTUATOR = "fr3_gripper"

    def __init__(self, model: mujoco.MjModel):
        names = self.ARM_ACTUATORS + (self.GRIPPER_ACTUATOR,)
        self._actuator_ids = np.array([model.actuator(name).id for name in names], dtype=int)
        if model.nu != len(names) or not np.array_equal(self._actuator_ids, np.arange(model.nu)):
            raise ValueError("FR3 Mimic action layout must be the named 8-actuator order")
        self._ctrl_min = model.actuator_ctrlrange[self._actuator_ids, 0].copy()
        self._ctrl_max = model.actuator_ctrlrange[self._actuator_ids, 1].copy()

    def convert(self, command: TeleopCommand) -> np.ndarray:
        joint_position = np.asarray(command.joint_position, dtype=float)
        if joint_position.shape != (7,):
            raise ValueError(f"joint_position must have shape (7,), got {joint_position.shape}")
        action = np.concatenate([joint_position, [float(command.gripper_command)]])
        if not np.all(np.isfinite(action)):
            raise ValueError("teleop command contains a non-finite value")
        return np.clip(action, self._ctrl_min, self._ctrl_max)
