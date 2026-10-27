"""Gravity-equilibrium position references; no change to MuJoCo dynamics.

The offset is captured from actual kinematics on each synchronization edge,
then held constant. It never becomes the Cartesian IK integration baseline.
Saved actions contain the final supported servo reference and replay unchanged.
"""

import mujoco
import numpy as np


class PositionServoGravitySupport:
    def __init__(self, model, data, actuator_indices, dof_indices):
        self.model, self.data = model, data
        self.actuators = np.asarray(actuator_indices)
        self.dofs = np.asarray(dof_indices)
        a = self.actuators
        self.kp = model.actuator_gainprm[a, 0].copy()
        joints = model.actuator_trnid[a, 0]
        expected_gear = np.tile([1., 0., 0., 0., 0., 0.], (len(a), 1))
        if not (
            np.all(model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT)
            and np.all(model.actuator_dyntype[a] == mujoco.mjtDyn.mjDYN_NONE)
            and np.all(model.actuator_gaintype[a] == mujoco.mjtGain.mjGAIN_FIXED)
            and np.all(model.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE)
            and np.array_equal(model.actuator_gear[a], expected_gear)
            and np.all(np.isfinite(self.kp)) and np.all(self.kp > 0)
            and np.all(model.actuator_biasprm[a, 0] == 0)
            and np.array_equal(model.actuator_biasprm[a, 1], -self.kp)
            and np.all(model.actuator_biasprm[a, 2] <= 0)
            and np.array_equal(model.jnt_dofadr[joints], self.dofs)
            and np.all(model.actuator_ctrllimited[a])
            and not np.any(model.body_gravcomp)
        ):
            raise ValueError("FR3 gravity support requires unit-gear position servos without existing gravcomp")
        self.lower, self.upper = model.actuator_ctrlrange[a].T.copy()
        self.root = model.body("fr3_link0").id
        self.jacobian = np.zeros((3, model.nv))
        self.offset = np.zeros(len(a))
        self.clipped = False

    def capture(self):
        # COM Jacobian excludes velocity-dependent Coriolis terms in qfrc_bias.
        # Reads caller-owned cached kinematics; no forward/step or model copy.
        mujoco.mj_jacSubtreeCom(self.model, self.data, self.jacobian, self.root)
        gravity = (-self.model.body_subtreemass[self.root]
                   * (self.jacobian[:, self.dofs].T @ self.model.opt.gravity))
        self.offset[:] = gravity / self.kp
        if not np.isfinite(self.offset).all():
            raise ValueError("non-finite FR3 gravity support")

    def command(self, logical_q):
        reference = np.asarray(logical_q) + self.offset
        result = np.clip(reference, self.lower, self.upper)
        self.clipped = bool(np.any(reference != result))
        return result
