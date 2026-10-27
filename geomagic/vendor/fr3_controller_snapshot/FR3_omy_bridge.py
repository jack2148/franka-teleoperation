#!/usr/bin/python3
"""OMY-L100 -> MuJoCo FR3 6D Cartesian teleoperation.

This version separates:
- OMY-to-FR3 target mapping
- target rate limiting
- 6D damped least-squares velocity IK
- MuJoCo stepping
- low-rate viewer/logging

It targets a 1000 Hz simulation/control step, but Python is not hard real-time.
For a real FR3, use the robot real-time interface and its measured period.
"""

import csv
import os
import resource
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import UInt8
except ImportError:  # MuJoCo-only conda interpreter; use the IPC input below.
    rclpy = None
    Node = object
    JointState = None
    UInt8 = None

import zmq

from runtime.ipc.protocol import ProtocolError, decode_message


ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Paths and runtime options
# ---------------------------------------------------------------------------

OMY_MODEL_PATH = ROOT / "robotis_mujoco_menagerie" / "robotis_omy" / "scene.xml"
FR3_MODEL_PATH = ROOT / "mujoco_menagerie" / "franka_fr3" / "scene.xml"

ENABLE_LOGGING = True
ENABLE_VIEWER = True
ENABLE_MAPPING_DEBUG_LOGS = False
ENABLE_OMY_INTERNAL_DEBUG_LOGS = False
ENABLE_NULLSPACE_DEBUG_LOGS = False

CONTROL_HZ = 1000.0
CONTROL_DT = 1.0 / CONTROL_HZ
LOG_HZ = 100.0
VIEWER_HZ = 30.0
PRINT_HZ = 1.0
ROS_TIMEOUT_S = 0.20


# ---------------------------------------------------------------------------
# Teleoperation parameters
# ---------------------------------------------------------------------------

OMY_ROS_JOINTS = [f"joint{i}" for i in range(1, 7)]
OMY_MUJOCO_JOINTS = [f"Joint{i}" for i in range(1, 7)]
OMY_BASE_BODY_NAME = "base_unit"
OMY_EE_SITE_NAME = "omy_ee_site"
FR3_BASE_BODY_NAME = "base"
FR3_EE_SITE_NAME = "fr3_gripper_center_site"
TRIGGER_JOINT = "rh_r1_joint"

TRIGGER_ON_THRESHOLD = -0.73
TRIGGER_OFF_THRESHOLD = -0.70

# Position mapping candidates for manual one-axis calibration.
POSITION_SAME_AXIS_SIGNS = np.array([1.0, 1.0, 1.0])
POSITION_AXIS_MAP_CANDIDATES = {
    "identity": np.eye(3),
    "current_90": np.array([
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]),
    "same_axis": np.diag(POSITION_SAME_AXIS_SIGNS),
}

# Manual selection only. Keep the current behavior as the initial baseline.
POSITION_MAP_CANDIDATE = "current_90"
POSITION_AXIS_MAP = POSITION_AXIS_MAP_CANDIDATES[POSITION_MAP_CANDIDATE].copy()

# Orientation mapping remains independent from position mapping. This is the
# exact value of the previous ROTATION_VECTOR_MAP.
ORIENTATION_AXIS_MAP = np.array([
    [0.0, -1.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
])

TELEOP_MODE = "full_pose"
SUPPORTED_TELEOP_MODES = {
    "position_only",
    "orientation_only",
    "full_pose",
}

# Conservative initial values.
POSITION_SCALE = 0.4
ORIENTATION_SCALE = 0.3

# Cartesian target rate limits.
MAX_TARGET_LINEAR_SPEED = 0.09       # m/s
MAX_TARGET_LINEAR_ACCEL = 2.0        # m/s^2
POSITION_SNAP_ERROR = 1e-6           # m
POSITION_SNAP_SPEED = 1e-3           # m/s
MAX_TARGET_ANGULAR_SPEED = 2.0      # rad/s
MAX_TARGET_ANGULAR_ACCEL = 4.0      # rad/s^2
TARGET_ROTATION_KP = 4.0             # 1/s, target conditioning
ROTATION_SNAP_ERROR = 1e-4           # rad

# Task-space feedback used to generate a desired Cartesian twist.
POSITION_KP = 8.0                    # 1/s
ORIENTATION_KP = 4.0                 # 1/s

# DLS velocity IK.
IK_DAMPING = 0.05
ROTATION_LENGTH_SCALE = 0.10         # m/rad task metric
MAX_JOINT_SPEED = 0.80               # rad/s, simulation initial value
ENABLE_NULLSPACE_POSTURE = True
NULLSPACE_GAIN = 0.1

# ---------------------------------------------------------------------------
# ROS latest-state buffer
# ---------------------------------------------------------------------------

class OmyPose(Node):
    """Store the latest valid OMY leader state."""

    def __init__(self, joint_positions, state_lock):
        super().__init__("fr3_omy_bridge")
        self.joint_positions = joint_positions
        self.state_lock = state_lock
        self.trigger_position = 0.0
        self.gripper_command = 0.0
        self.last_message_time = 0.0
        self.has_joint_state = False

        self.subscription = self.create_subscription(
            JointState,
            "/leader/joint_states",
            self.joint_state_callback,
            10,
        )
        self.gripper_subscription = self.create_subscription(
            UInt8,
            "/fr3/gripper_command",
            self.gripper_command_callback,
            10,
        )
        self.gripper_publisher = self.create_publisher(
            UInt8,
            "/fr3/gripper_command",
            10,
        )

    def gripper_command_callback(self, message):
        with self.state_lock:
            self.gripper_command = float(message.data)
        state = "open" if message.data else "closed"
        print(f"FR3 gripper command: {state} (ctrl={message.data})", flush=True)

    def joint_state_callback(self, message):
        received = dict(zip(message.name, message.position))
        required = OMY_ROS_JOINTS + [TRIGGER_JOINT]

        if not all(name in received for name in required):
            return

        with self.state_lock:
            for index, name in enumerate(OMY_ROS_JOINTS):
                self.joint_positions[index] = received[name]

            self.trigger_position = received[TRIGGER_JOINT]
            self.last_message_time = time.perf_counter()
        self.has_joint_state = True


class ZmqOmyPose:
    """OMY state input for the MuJoCo Python environment.

    ROS 2 Humble is built for system Python 3.10 while MuJoCo is installed in
    the Python 3.12 environment.  The UI starts a system-Python ROS publisher
    and this process consumes the versioned localhost IPC stream.
    """

    def __init__(self, joint_positions, state_lock, endpoint):
        self.joint_positions = joint_positions
        self.state_lock = state_lock
        self.trigger_position = 0.0
        self.gripper_command = 0.0
        self.last_message_time = 0.0
        self.has_joint_state = False
        self.endpoint = endpoint
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, name="omy-zmq-input", daemon=True)
        self._thread.start()

    def _run(self):
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self.endpoint)
        socket.setsockopt(zmq.SUBSCRIBE, b"")
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        last_sequence = None
        try:
            while not self._stop.is_set():
                events = dict(poller.poll(100))
                if socket not in events:
                    continue
                try:
                    message = decode_message(socket.recv(zmq.NOBLOCK), last_sequence=last_sequence)
                except (zmq.Again, ProtocolError):
                    continue
                last_sequence = message.sequence
                with self.state_lock:
                    self.joint_positions[:] = message.position
                    self.trigger_position = message.trigger_position
                    self.gripper_command = message.gripper_command
                    self.last_message_time = time.perf_counter()
                    self.has_joint_state = True
        finally:
            socket.close(linger=0)
            context.term()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def publish_gripper(self, command):
        # Gripper feedback is intentionally one-way until a ROS publisher is
        # added for the operator hardware.  The viewer toggle still updates
        # the local simulation command.
        with self.state_lock:
            self.gripper_command = float(command)

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def keyframe_id(model, name):
    result = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    if result < 0:
        raise ValueError(f"keyframe not found: {name}")
    return result


def site_id(model, name):
    result = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if result < 0:
        raise ValueError(f"site not found: {name}")
    return result


def body_id(model, name):
    result = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if result < 0:
        raise ValueError(f"body not found: {name}")
    return result


def read_site_pose(data, site):
    position = data.site_xpos[site].copy()
    rotation = data.site_xmat[site].reshape(3, 3).copy()
    return position, rotation


def transform_from_pose(position, rotation):
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform


def inverse_transform(transform):
    inverse = np.eye(4)
    rotation = transform[:3, :3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ transform[:3, 3]
    return inverse


def read_body_transform(data, body):
    return transform_from_pose(
        data.xpos[body],
        data.xmat[body].reshape(3, 3),
    )


def read_site_transform(data, site):
    return transform_from_pose(
        data.site_xpos[site],
        data.site_xmat[site].reshape(3, 3),
    )


def read_base_ee_transform(data, base_body, ee_site):
    t_world_base = read_body_transform(data, base_body)
    t_world_ee = read_site_transform(data, ee_site)
    return inverse_transform(t_world_base) @ t_world_ee


def rotation_diagnostics(rotation):
    return (
        float(np.linalg.det(rotation)),
        float(np.linalg.norm(rotation.T @ rotation - np.eye(3))),
    )


def print_frame_inspection(
    label,
    data,
    base_body_name,
    base_body,
    ee_site_name,
    ee_site,
):
    t_world_base = read_body_transform(data, base_body)
    t_base_ee = read_base_ee_transform(data, base_body, ee_site)
    base_rotation = t_world_base[:3, :3]
    ee_rotation = t_base_ee[:3, :3]
    base_det, base_orth_error = rotation_diagnostics(base_rotation)
    ee_det, ee_orth_error = rotation_diagnostics(ee_rotation)
    identity_aligned = np.allclose(base_rotation, np.eye(3), atol=1e-9)

    print(f"{label} frame inspection:")
    print(f"  base body: {base_body_name}")
    print(f"  EE site: {ee_site_name}")
    print(f"  world base position: {t_world_base[:3, 3]}")
    print(f"  world base rotation:\n{base_rotation}")
    print(f"  base-relative EE position: {t_base_ee[:3, 3]}")
    print(f"  base-relative EE rotation:\n{ee_rotation}")
    print(f"  base rotation det: {base_det:.9f}")
    print(f"  base rotation orthogonality error: {base_orth_error:.3e}")
    print(f"  EE rotation det: {ee_det:.9f}")
    print(f"  EE rotation orthogonality error: {ee_orth_error:.3e}")
    print(f"  base identity-aligned with world: {identity_aligned}")


def skew(vector):
    x, y, z = vector
    return np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ])


def rotvec_to_matrix(rotvec):
    """SO(3) exponential map."""
    angle = float(np.linalg.norm(rotvec))

    if angle < 1e-9:
        return np.eye(3) + skew(rotvec)

    axis = rotvec / angle
    axis_skew = skew(axis)
    return (
        np.eye(3)
        + np.sin(angle) * axis_skew
        + (1.0 - np.cos(angle)) * (axis_skew @ axis_skew)
    )


def matrix_to_rotvec(rotation):
    """SO(3) logarithm map."""
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    angle = float(np.arccos(cosine))

    if angle < 1e-7:
        return 0.5 * np.array([
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ])

    if np.pi - angle < 1e-5:
        axis = np.sqrt(np.maximum(np.diag(rotation) + 1.0, 0.0) * 0.5)
        major = int(np.argmax(axis))
        if axis[major] < 1e-7:
            return np.zeros(3)

        for index in range(3):
            if index != major:
                axis[index] = (
                    rotation[major, index] + rotation[index, major]
                ) / (4.0 * axis[major])

        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-9:
            return np.zeros(3)
        return angle * axis / axis_norm

    vee = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ])
    return angle * vee / (2.0 * np.sin(angle))


def limit_norm(vector, max_norm):
    norm = float(np.linalg.norm(vector))
    if norm <= max_norm or norm < 1e-12:
        return vector
    return vector * (max_norm / norm)


def rotation_distance_deg(reference_rotation, current_rotation):
    """Return the magnitude of the relative rotation in degrees."""
    relative_rotation = current_rotation @ reference_rotation.T
    return float(
        np.rad2deg(np.linalg.norm(matrix_to_rotvec(relative_rotation)))
    )


# ---------------------------------------------------------------------------
# Target generation and conditioning
# ---------------------------------------------------------------------------

def make_desired_target(
    omy_anchor_position,
    omy_anchor_rotation,
    omy_current_position,
    omy_current_rotation,
    fr3_anchor_position,
    fr3_anchor_rotation,
):
    """Map the OMY clutch-relative pose to an unconstrained FR3 target."""
    delta_position_omy = omy_current_position - omy_anchor_position
    desired_position = (
        fr3_anchor_position
        + POSITION_SCALE * (POSITION_AXIS_MAP @ delta_position_omy)
    )

    omy_relative_rotation = omy_anchor_rotation.T @ omy_current_rotation
    omy_relative_rotvec = matrix_to_rotvec(omy_relative_rotation)
    mapped_rotvec = (
        ORIENTATION_SCALE
        * (ORIENTATION_AXIS_MAP @ omy_relative_rotvec)
    )
    desired_rotation = fr3_anchor_rotation @ rotvec_to_matrix(mapped_rotvec)

    return desired_position, desired_rotation


def condition_target(
    current_target_position,
    current_target_rotation,
    desired_position,
    desired_rotation,
    previous_linear_velocity,
    previous_angular_velocity,
    dt,
):
    """Rate- and acceleration-limit the Cartesian target."""
    position_error = desired_position - current_target_position
    error_norm = float(np.linalg.norm(position_error))
    if error_norm > 1e-12:
        position_direction = position_error / error_norm
        proportional_speed = POSITION_KP * error_norm
        linear_stopping_speed = np.sqrt(
            max(
                0.0,
                2.0 * MAX_TARGET_LINEAR_ACCEL * error_norm,
            )
        )
        desired_linear_speed = min(
            proportional_speed,
            MAX_TARGET_LINEAR_SPEED,
            linear_stopping_speed,
        )
        desired_linear_velocity = (
            position_direction * desired_linear_speed
        )
    else:
        linear_stopping_speed = 0.0
        desired_linear_speed = 0.0
        desired_linear_velocity = np.zeros(3)

    delta_linear_velocity = (
        desired_linear_velocity - previous_linear_velocity
    )
    limited_delta_linear_velocity = limit_norm(
        delta_linear_velocity,
        MAX_TARGET_LINEAR_ACCEL * dt,
    )
    linear_velocity = previous_linear_velocity + limited_delta_linear_velocity
    position_step = linear_velocity * dt
    next_position = current_target_position + position_step
    snapped_to_command = False

    new_error_norm = float(np.linalg.norm(desired_position - next_position))
    can_snap = (
        new_error_norm < POSITION_SNAP_ERROR
        and np.linalg.norm(previous_linear_velocity) < POSITION_SNAP_SPEED
        and np.linalg.norm(linear_velocity) < POSITION_SNAP_SPEED
        and np.linalg.norm(previous_linear_velocity)
        <= MAX_TARGET_LINEAR_ACCEL * dt
    )
    if can_snap:
        limited_delta_linear_velocity = -previous_linear_velocity
        linear_velocity = np.zeros(3)
        next_position = desired_position.copy()
        snapped_to_command = True

    commanded_linear_acceleration = (
        limited_delta_linear_velocity / max(dt, 1e-9)
    )

    target_rotation_error = desired_rotation @ current_target_rotation.T
    rotation_error = matrix_to_rotvec(target_rotation_error)
    error_angle = float(np.linalg.norm(rotation_error))

    if error_angle < ROTATION_SNAP_ERROR:
        next_rotation = desired_rotation.copy()
        limited_delta_angular_velocity = -previous_angular_velocity
        angular_velocity = np.zeros(3)
    else:
        direction = rotation_error / error_angle
        angular_stopping_speed = np.sqrt(
            2.0 * MAX_TARGET_ANGULAR_ACCEL * error_angle
        )
        desired_speed = min(
            TARGET_ROTATION_KP * error_angle,
            MAX_TARGET_ANGULAR_SPEED,
            angular_stopping_speed,
        )
        desired_angular_velocity = direction * desired_speed
        delta_angular_velocity = (
            desired_angular_velocity - previous_angular_velocity
        )
        limited_delta_angular_velocity = limit_norm(
            delta_angular_velocity,
            MAX_TARGET_ANGULAR_ACCEL * dt,
        )
        angular_velocity = (
            previous_angular_velocity + limited_delta_angular_velocity
        )
        rotation_step = angular_velocity * dt
        next_rotation = (
            rotvec_to_matrix(rotation_step)
            @ current_target_rotation
        )

    linear_speed = float(np.linalg.norm(position_step) / max(dt, 1e-9))
    angular_speed = float(np.linalg.norm(angular_velocity))
    commanded_angular_acceleration = (
        limited_delta_angular_velocity / max(dt, 1e-9)
    )
    return (
        next_position,
        next_rotation,
        linear_speed,
        angular_speed,
        commanded_linear_acceleration,
        linear_velocity,
        angular_velocity,
        linear_stopping_speed,
        desired_linear_speed,
        snapped_to_command,
        commanded_angular_acceleration,
    )


# ---------------------------------------------------------------------------
# 6D DLS velocity IK
# ---------------------------------------------------------------------------

def dls_pseudoinverse(jacobian, damping):
    task_dim = jacobian.shape[0]
    regularized = (
        jacobian @ jacobian.T
        + damping**2 * np.eye(task_dim)
    )
    return jacobian.T @ np.linalg.solve(
        regularized,
        np.eye(task_dim),
    )


def exact_nullspace_projector(jacobian):
    """Return an SVD-based projector onto null(J)."""
    _, singular_values, vh = np.linalg.svd(
        jacobian,
        full_matrices=True,
    )
    if singular_values.size == 0:
        return np.eye(jacobian.shape[1])

    tolerance = (
        max(jacobian.shape)
        * np.finfo(jacobian.dtype).eps
        * singular_values[0]
    )
    rank = int(np.sum(singular_values > tolerance))
    null_basis = vh[rank:].T
    if null_basis.shape[1] == 0:
        return np.zeros((jacobian.shape[1], jacobian.shape[1]))
    return null_basis @ null_basis.T


def compute_joint_target(
    model,
    data,
    ee_site_id,
    dof_indices,
    qpos_indices,
    joint_lower,
    joint_upper,
    target_position,
    target_rotation,
    command_reference,
    jacp,
    jacr,
    dt,
    teleop_mode="full_pose",
    enable_nullspace_posture=False,
    q_posture_reference=None,
):
    """Compute a mode-specific velocity-DLS joint target."""
    current_position, current_rotation = read_site_pose(data, ee_site_id)

    position_error = target_position - current_position
    rotation_error_matrix = target_rotation @ current_rotation.T
    rotation_error = matrix_to_rotvec(rotation_error_matrix)

    desired_linear_velocity = POSITION_KP * position_error
    desired_angular_velocity = ORIENTATION_KP * rotation_error

    jacp.fill(0.0)
    jacr.fill(0.0)
    mujoco.mj_jacSite(
        model,
        data,
        jacp,
        jacr,
        ee_site_id,
    )

    j_pos = jacp[:, dof_indices]
    j_rot = jacr[:, dof_indices]
    if j_pos.shape != (3, 7) or j_rot.shape != (3, 7):
        raise ValueError(
            f"expected 3x7 position/rotation Jacobians, got "
            f"{j_pos.shape} and {j_rot.shape}"
        )

    if teleop_mode == "position_only":
        task_jacobian = j_pos
        task_velocity = desired_linear_velocity
    elif teleop_mode == "orientation_only":
        task_jacobian = ROTATION_LENGTH_SCALE * j_rot
        task_velocity = ROTATION_LENGTH_SCALE * desired_angular_velocity
    elif teleop_mode == "full_pose":
        task_jacobian = np.vstack((
            j_pos,
            ROTATION_LENGTH_SCALE * j_rot,
        ))
        task_velocity = np.concatenate((
            desired_linear_velocity,
            ROTATION_LENGTH_SCALE * desired_angular_velocity,
        ))
    else:
        raise ValueError(f"Unsupported teleop mode: {teleop_mode}")

    expected_task_rows = 3 if teleop_mode != "full_pose" else 6
    if task_jacobian.shape != (expected_task_rows, 7):
        raise ValueError(
            f"unexpected task Jacobian shape: {task_jacobian.shape}"
        )
    if task_velocity.shape != (expected_task_rows,):
        raise ValueError(
            f"unexpected task velocity shape: {task_velocity.shape}"
        )

    q_current = data.qpos[qpos_indices].copy()
    if q_current.shape != (7,):
        raise ValueError(
            f"expected 7 controlled FR3 joints, got {q_current.shape}"
        )
    if task_jacobian.shape[1] != q_current.shape[0]:
        raise ValueError(
            "Jacobian column count does not match controlled joint count"
        )

    finite_values = {
        "J_task": task_jacobian,
        "task_twist": task_velocity,
    }
    nonfinite_names = [
        name for name, value in finite_values.items()
        if not np.all(np.isfinite(value))
    ]
    task_jacobian_pinv = dls_pseudoinverse(task_jacobian, IK_DAMPING)
    qdot_task = task_jacobian_pinv @ task_velocity
    finite_values.update({
        "J_dls": task_jacobian_pinv,
        "qdot_task": qdot_task,
    })
    nonfinite_names.extend([
        name for name, value in finite_values.items()
        if name not in nonfinite_names and not np.all(np.isfinite(value))
    ])
    if nonfinite_names:
        print(
            "WARNING: non-finite 6D DLS values: "
            + ", ".join(nonfinite_names)
        )
        qdot_task = np.zeros_like(q_current)

    nullspace_enabled = bool(
        enable_nullspace_posture
        and q_posture_reference is not None
    )
    if nullspace_enabled:
        if q_posture_reference.shape != q_current.shape:
            raise ValueError(
                "q_posture_reference shape does not match q_current"
            )
        null_projector = exact_nullspace_projector(task_jacobian)
        qdot_posture = NULLSPACE_GAIN * (
            q_posture_reference - q_current
        )
        qdot_null = null_projector @ qdot_posture
    else:
        qdot_null = np.zeros_like(qdot_task)

    qdot_total = qdot_task + qdot_null
    finite_values.update({
        "qdot_null": qdot_null,
        "qdot_total": qdot_total,
    })
    nonfinite_names.extend([
        name for name, value in finite_values.items()
        if name not in nonfinite_names and not np.all(np.isfinite(value))
    ])
    if nonfinite_names:
        print(
            "WARNING: non-finite null-space IK values: "
            + ", ".join(nonfinite_names)
        )
        qdot_task = np.zeros_like(q_current)
        qdot_null = np.zeros_like(q_current)
        qdot_total = np.zeros_like(q_current)

    qdot_command = np.clip(
        qdot_total,
        -MAX_JOINT_SPEED,
        MAX_JOINT_SPEED,
    )
    if not np.all(np.isfinite(qdot_command)):
        print("WARNING: non-finite clipped joint velocity; commanding zero")
        qdot_command = np.zeros_like(q_current)
    saturated = bool(np.any(np.abs(qdot_total) > MAX_JOINT_SPEED))

    # Use the held actuator command as the integration reference. Using the
    # sagging actual q here would make gravity-induced drift continue when
    # teleoperation becomes active.
    q_target = np.clip(
        command_reference + qdot_command * dt,
        joint_lower,
        joint_upper,
    )

    try:
        condition_number = float(np.linalg.cond(task_jacobian))
    except np.linalg.LinAlgError:
        condition_number = float("inf")

    task_residual = task_jacobian @ qdot_task - task_velocity

    diagnostics = {
        "teleop_mode": teleop_mode,
        "position_error_m": float(np.linalg.norm(position_error)),
        "position_error_norm": float(np.linalg.norm(position_error)),
        "orientation_error_deg": float(
            np.degrees(np.linalg.norm(rotation_error))
        ),
        "orientation_error_norm_rad": float(np.linalg.norm(rotation_error)),
        "linear_task_velocity_norm": float(
            np.linalg.norm(desired_linear_velocity)
        ),
        "angular_task_velocity_norm": float(
            np.linalg.norm(desired_angular_velocity)
        ),
        "weighted_task_twist_norm": float(np.linalg.norm(task_velocity)),
        "raw_max_qdot": float(np.max(np.abs(qdot_total))),
        "cmd_max_qdot": float(np.max(np.abs(qdot_command))),
        "qdot_saturated": saturated,
        "jacobian_condition": condition_number,
        "nullspace_enabled": nullspace_enabled,
        "nullspace_gain": NULLSPACE_GAIN if nullspace_enabled else 0.0,
        "qdot_task_norm": float(np.linalg.norm(qdot_task)),
        "qdot_null_norm": float(np.linalg.norm(qdot_null)),
        "qdot_total_norm": float(np.linalg.norm(qdot_total)),
        "task_residual_norm": float(np.linalg.norm(task_residual)),
        "posture_reference_distance": float(
            np.linalg.norm(q_posture_reference - q_current)
        ) if q_posture_reference is not None else 0.0,
        "null_task_leak": float(np.linalg.norm(task_jacobian @ qdot_null)),
        "joint_speed_saturated": saturated,
        "joint_positions": q_current.copy(),
        "qdot_task": qdot_task.copy(),
        "qdot_null": qdot_null.copy(),
        "qdot_total": qdot_total.copy(),
        "max_q_command_error": float(
            np.max(np.abs(q_target - q_current))
        ),
    }
    return q_target, diagnostics


# ---------------------------------------------------------------------------
# Viewer and logging
# ---------------------------------------------------------------------------

def draw_target_marker(viewer, position, rotation):
    if viewer is None:
        return

    scene = viewer.user_scn
    scene.ngeom = 4

    mujoco.mjv_initGeom(
        scene.geoms[0],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([0.025, 0.0, 0.0]),
        position,
        np.eye(3).reshape(-1),
        np.array([1.0, 0.0, 0.0, 1.0]),
    )

    colors = (
        np.array([1.0, 0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0, 1.0]),
        np.array([0.0, 0.0, 1.0, 1.0]),
    )
    length = 0.045
    radius = 0.004

    for geom_index, axis_index in enumerate(range(3), start=1):
        axis = rotation[:, axis_index]
        other_a = rotation[:, (axis_index + 1) % 3]
        other_b = rotation[:, (axis_index + 2) % 3]
        geom_rotation = np.column_stack((other_a, other_b, axis))
        geom_position = position + 0.5 * length * axis

        mujoco.mjv_initGeom(
            scene.geoms[geom_index],
            mujoco.mjtGeom.mjGEOM_BOX,
            np.array([radius, radius, 0.5 * length]),
            geom_position,
            geom_rotation.reshape(-1),
            colors[axis_index],
        )


def write_log(log_path, rows):
    if not rows:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with log_path.open("w", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if TELEOP_MODE not in SUPPORTED_TELEOP_MODES:
        raise ValueError(
            f"unsupported TELEOP_MODE: {TELEOP_MODE!r}; "
            f"choose one of {sorted(SUPPORTED_TELEOP_MODES)}"
        )

    input_mode = os.environ.get("OMY_INPUT", "ros").lower()
    if input_mode not in {"ros", "zmq"}:
        raise ValueError("OMY_INPUT must be ros or zmq")
    if input_mode == "ros" and rclpy is None:
        raise RuntimeError("ROS input requires an interpreter with rclpy; use OMY_INPUT=zmq")
    if input_mode == "ros":
        rclpy.init()

    # OMY model: FK only.
    omy_model = mujoco.MjModel.from_xml_path(str(OMY_MODEL_PATH))
    omy_data = mujoco.MjData(omy_model)
    mujoco.mj_resetDataKeyframe(
        omy_model,
        omy_data,
        keyframe_id(omy_model, "home"),
    )
    mujoco.mj_forward(omy_model, omy_data)
    omy_base_body_id = body_id(omy_model, OMY_BASE_BODY_NAME)
    omy_ee_site_id = site_id(omy_model, OMY_EE_SITE_NAME)
    omy_joint_ids = np.array(
        [omy_model.joint(name).id for name in OMY_MUJOCO_JOINTS],
        dtype=int,
    )
    omy_dof_indices = np.array(
        [omy_model.jnt_dofadr[index] for index in omy_joint_ids],
        dtype=int,
    )

    # FR3 model: IK + actuator dynamics.
    fr3_model = mujoco.MjModel.from_xml_path(str(FR3_MODEL_PATH))
    fr3_model.opt.timestep = CONTROL_DT
    fr3_data = mujoco.MjData(fr3_model)
    mujoco.mj_resetDataKeyframe(
        fr3_model,
        fr3_data,
        keyframe_id(fr3_model, "home"),
    )
    mujoco.mj_forward(fr3_model, fr3_data)
    fr3_base_body_id = body_id(fr3_model, FR3_BASE_BODY_NAME)
    fr3_ee_site_id = site_id(fr3_model, FR3_EE_SITE_NAME)

    print_frame_inspection(
        "OMY",
        omy_data,
        OMY_BASE_BODY_NAME,
        omy_base_body_id,
        OMY_EE_SITE_NAME,
        omy_ee_site_id,
    )
    print_frame_inspection(
        "FR3",
        fr3_data,
        FR3_BASE_BODY_NAME,
        fr3_base_body_id,
        FR3_EE_SITE_NAME,
        fr3_ee_site_id,
    )

    fr3_joint_ids = np.array(
        [fr3_model.joint(f"fr3_joint{i}").id for i in range(1, 8)],
        dtype=int,
    )
    fr3_qpos_indices = np.array(
        [fr3_model.jnt_qposadr[index] for index in fr3_joint_ids],
        dtype=int,
    )
    fr3_dof_indices = np.array(
        [fr3_model.jnt_dofadr[index] for index in fr3_joint_ids],
        dtype=int,
    )
    fr3_joint_lower = fr3_model.jnt_range[fr3_joint_ids, 0].copy()
    fr3_joint_upper = fr3_model.jnt_range[fr3_joint_ids, 1].copy()
    fr3_actuator_indices = np.array(
        [fr3_model.actuator(f"fr3_joint{i}").id for i in range(1, 8)],
        dtype=int,
    )
    fr3_gripper_actuator_id = fr3_model.actuator("fr3_gripper").id

    q_home = fr3_data.qpos[fr3_qpos_indices].copy()
    fr3_data.ctrl[fr3_actuator_indices] = q_home
    fr3_data.ctrl[fr3_gripper_actuator_id] = 0.0
    mujoco.mj_forward(fr3_model, fr3_data)

    fr3_target_position, fr3_target_rotation = read_site_pose(
        fr3_data,
        fr3_ee_site_id,
    )
    fr3_command_position = fr3_target_position.copy()
    fr3_command_rotation = fr3_target_rotation.copy()
    fr3_initial_command_rotation = fr3_command_rotation.copy()
    hold_q_target = q_home.copy()

    jacp = np.zeros((3, fr3_model.nv))
    jacr = np.zeros((3, fr3_model.nv))

    omy_qpos_addresses = [
        omy_model.jnt_qposadr[index] for index in omy_joint_ids
    ]

    joint_positions = omy_data.qpos[omy_qpos_addresses].copy()
    state_lock = threading.Lock()
    if input_mode == "ros":
        ros_node = OmyPose(joint_positions, state_lock)
        ros_thread = threading.Thread(
            target=rclpy.spin,
            args=(ros_node,),
            daemon=True,
        )
        ros_thread.start()
    else:
        ros_node = ZmqOmyPose(
            joint_positions,
            state_lock,
            os.environ.get("OMY_ZMQ_ENDPOINT", "tcp://127.0.0.1:5557"),
        )
        ros_thread = None
        ros_node.start()

    teleop_active = False
    clutch_id = 0
    target_linear_velocity = np.zeros(3)
    target_angular_velocity = np.zeros(3)
    target_stopping_speed = 0.0
    target_desired_linear_speed = 0.0
    target_snapped_to_command = False
    target_angular_acceleration = np.zeros(3)
    nullspace_posture_reference = q_home.copy()
    omy_anchor_position = None
    omy_anchor_rotation = None
    omy_anchor_base_ee_transform = None
    fr3_anchor_position = fr3_command_position.copy()
    fr3_anchor_rotation = fr3_command_rotation.copy()

    next_tick = time.perf_counter()
    last_viewer_sync = next_tick
    last_print = next_tick
    last_rate_report = next_tick
    cycles_since_report = 0
    deadline_misses = 0
    rows = []
    cpu_wall_reference = time.perf_counter()
    cpu_time_reference = (
        resource.getrusage(resource.RUSAGE_SELF).ru_utime
        + resource.getrusage(resource.RUSAGE_SELF).ru_stime
    )
    process_cpu_percent = 0.0

    run_id = time.strftime("%Y%m%d_%H%M%S")
    log_path = ROOT / "logs" / f"refactored_teleop_{run_id}.csv"
    print(f"CSV log will be written on shutdown: {log_path}", flush=True)

    def viewer_key_callback(keycode):
        if keycode != ord(" "):
            return
        with state_lock:
            ros_node.gripper_command = (
                0.0 if ros_node.gripper_command >= 127.5 else 255.0
            )
            command = int(ros_node.gripper_command)
        if input_mode == "ros":
            ros_node.gripper_publisher.publish(UInt8(data=command))
        else:
            ros_node.publish_gripper(command)
        state = "closed" if command == 0 else "open"
        print(
            f"FR3 gripper Space toggle: {state} (ctrl={command})",
            flush=True,
        )

    viewer_context = (
        mujoco.viewer.launch_passive(
            fr3_model,
            fr3_data,
            key_callback=viewer_key_callback,
        )
        if ENABLE_VIEWER
        else None
    )

    try:
        if viewer_context is None:
            viewer = None
            keep_running = lambda: rclpy.ok() if input_mode == "ros" else not ros_node._stop.is_set()
        else:
            viewer = viewer_context.__enter__()
            keep_running = lambda: viewer.is_running() and (rclpy.ok() if input_mode == "ros" else not ros_node._stop.is_set())

        print(f"Configured control rate: {CONTROL_HZ:.0f} Hz")
        print(f"MuJoCo timestep: {fr3_model.opt.timestep:.6f} s")
        print(f"Position axis map candidate: {POSITION_MAP_CANDIDATE}")
        print(f"Teleoperation mode: {TELEOP_MODE}")
        print(f"MuJoCo viewer: {'ON' if ENABLE_VIEWER else 'OFF'}")
        print("Waiting for /leader/joint_states...")

        step_index = 0

        while keep_running():
            cycle_start = time.perf_counter()

            with state_lock:
                omy_target = joint_positions.copy()
                trigger_position = ros_node.trigger_position
                gripper_command = ros_node.gripper_command
                last_message_time = ros_node.last_message_time
                has_joint_state = ros_node.has_joint_state

            ros_fresh = (
                has_joint_state
                and cycle_start - last_message_time <= ROS_TIMEOUT_S
            )

            omy_base_ee_position = np.zeros(3)
            omy_base_ee_rotation_vector = np.zeros(3)
            omy_current_position = np.zeros(3)
            omy_current_rotation = np.eye(3)
            omy_base_ee_transform = np.eye(4)
            omy_teleop_delta = np.zeros(3)
            omy_clutch_base_position_delta = np.zeros(3)
            omy_clutch_spatial_rotation_vector = np.zeros(3)
            omy_raw_clutch_position_delta = np.zeros(3)
            omy_delta_rotvec = np.zeros(3)
            fr3_command_position_delta = np.zeros(3)
            fr3_mapped_rotation_delta = np.zeros(3)
            fr3_target_position_delta = np.zeros(3)
            fr3_actual_position_delta = np.zeros(3)
            target_follow_active = False

            if ros_fresh:
                for address, position in zip(omy_qpos_addresses, omy_target):
                    omy_data.qpos[address] = position
                mujoco.mj_forward(omy_model, omy_data)

                omy_current_position, omy_current_rotation = read_site_pose(
                    omy_data,
                    omy_ee_site_id,
                )
                omy_base_ee_transform = read_base_ee_transform(
                    omy_data,
                    omy_base_body_id,
                    omy_ee_site_id,
                )
                omy_base_ee_position = omy_base_ee_transform[:3, 3].copy()
                omy_base_ee_rotation_vector = matrix_to_rotvec(
                    omy_base_ee_transform[:3, :3]
                )

                if (
                    not teleop_active
                    and trigger_position <= TRIGGER_ON_THRESHOLD
                ):
                    clutch_id += 1
                    omy_anchor_position = omy_current_position.copy()
                    omy_anchor_rotation = omy_current_rotation.copy()
                    omy_anchor_base_ee_transform = omy_base_ee_transform.copy()
                    fr3_anchor_position = fr3_command_position.copy()
                    fr3_anchor_rotation = fr3_command_rotation.copy()
                    teleop_active = True
                    target_linear_velocity = np.zeros(3)
                    target_angular_velocity = np.zeros(3)
                    command_anchor_angle_deg = rotation_distance_deg(
                        fr3_initial_command_rotation,
                        fr3_anchor_rotation,
                    )
                    print(
                        f"Teleoperation ON: clutch={clutch_id} | "
                        f"command anchor angle={command_anchor_angle_deg:.2f} deg"
                    )

                elif (
                    teleop_active
                    and trigger_position >= TRIGGER_OFF_THRESHOLD
                ):
                    teleop_active = False
                    held_command_angle_deg = rotation_distance_deg(
                        fr3_initial_command_rotation,
                        fr3_command_rotation,
                    )
                    print(
                        f"Teleoperation OFF: clutch={clutch_id} | "
                        f"held command angle={held_command_angle_deg:.2f} deg"
                    )

                if (
                    teleop_active
                    and omy_anchor_base_ee_transform is not None
                    and (
                        ENABLE_MAPPING_DEBUG_LOGS
                        or ENABLE_OMY_INTERNAL_DEBUG_LOGS
                    )
                ):
                    if ENABLE_MAPPING_DEBUG_LOGS:
                        omy_raw_clutch_position_delta = (
                            omy_current_position - omy_anchor_position
                        )
                    if ENABLE_OMY_INTERNAL_DEBUG_LOGS:
                        omy_clutch_base_position_delta = (
                            omy_base_ee_transform[:3, 3]
                            - omy_anchor_base_ee_transform[:3, 3]
                        )
                        omy_clutch_spatial_rotation_vector = matrix_to_rotvec(
                            omy_base_ee_transform[:3, :3]
                            @ omy_anchor_base_ee_transform[:3, :3].T
                        )

                if teleop_active:
                    omy_teleop_delta = (
                        omy_current_position - omy_anchor_position
                    )
                    (
                        desired_command_position,
                        desired_command_rotation,
                    ) = make_desired_target(
                        omy_anchor_position,
                        omy_anchor_rotation,
                        omy_current_position,
                        omy_current_rotation,
                        fr3_anchor_position,
                        fr3_anchor_rotation,
                    )
                    if TELEOP_MODE == "position_only":
                        candidate_position = desired_command_position
                        candidate_rotation = fr3_anchor_rotation.copy()
                    elif TELEOP_MODE == "orientation_only":
                        candidate_position = fr3_anchor_position.copy()
                        candidate_rotation = desired_command_rotation
                    else:
                        candidate_position = desired_command_position
                        candidate_rotation = desired_command_rotation

                    fr3_command_position = candidate_position
                    fr3_command_rotation = candidate_rotation

                (
                    fr3_target_position,
                    fr3_target_rotation,
                    target_linear_speed,
                    target_angular_speed,
                    target_linear_acceleration,
                    target_linear_velocity,
                    target_angular_velocity,
                    target_stopping_speed,
                    target_desired_linear_speed,
                    target_snapped_to_command,
                    target_angular_acceleration,
                ) = condition_target(
                    fr3_target_position,
                    fr3_target_rotation,
                    fr3_command_position,
                    fr3_command_rotation,
                    target_linear_velocity,
                    target_angular_velocity,
                    CONTROL_DT,
                )

                command_position_gap = np.linalg.norm(
                    fr3_command_position - fr3_target_position
                )
                command_rotation_gap = np.linalg.norm(
                    matrix_to_rotvec(
                        fr3_command_rotation @ fr3_target_rotation.T
                    )
                )
                command_reached = (
                    command_position_gap < 1e-6
                    and command_rotation_gap < 1e-4
                )
                target_follow_active = teleop_active or not command_reached
                if not teleop_active and command_reached:
                    target_linear_velocity = np.zeros(3)
                    target_angular_velocity = np.zeros(3)
            else:
                teleop_active = False
                target_linear_speed = 0.0
                target_angular_speed = 0.0
                target_linear_acceleration = np.zeros(3)
                target_linear_velocity = np.zeros(3)
                target_angular_velocity = np.zeros(3)
                target_stopping_speed = 0.0
                target_desired_linear_speed = 0.0
                target_snapped_to_command = False
                target_angular_acceleration = np.zeros(3)
                fr3_command_position = fr3_target_position.copy()
                fr3_command_rotation = fr3_target_rotation.copy()
                target_follow_active = False

            command_target_rotation_gap_deg = rotation_distance_deg(
                fr3_target_rotation,
                fr3_command_rotation,
            )
            if ENABLE_MAPPING_DEBUG_LOGS and omy_anchor_rotation is not None:
                fr3_mapped_rotation_delta = matrix_to_rotvec(
                    fr3_anchor_rotation.T @ fr3_command_rotation
                )

            if target_follow_active:
                q_target, diagnostics = compute_joint_target(
                    fr3_model,
                    fr3_data,
                    fr3_ee_site_id,
                    fr3_dof_indices,
                    fr3_qpos_indices,
                    fr3_joint_lower,
                    fr3_joint_upper,
                    fr3_target_position,
                    fr3_target_rotation,
                    hold_q_target,
                    jacp,
                    jacr,
                    CONTROL_DT,
                    teleop_mode=TELEOP_MODE,
                    enable_nullspace_posture=ENABLE_NULLSPACE_POSTURE,
                    q_posture_reference=nullspace_posture_reference,
                )
                hold_q_target = q_target.copy()
            else:
                # Do not recompute the command from sagging q_current while
                # gravity acts. Once the conditioned target has converged,
                # hold the last actuator command until teleop resumes.
                q_target = hold_q_target.copy()
                diagnostics = {
                    "position_error_m": 0.0,
                    "teleop_mode": TELEOP_MODE,
                    "position_error_norm": 0.0,
                    "orientation_error_deg": 0.0,
                    "orientation_error_norm_rad": 0.0,
                    "linear_task_velocity_norm": 0.0,
                    "angular_task_velocity_norm": 0.0,
                    "weighted_task_twist_norm": 0.0,
                    "raw_max_qdot": 0.0,
                    "cmd_max_qdot": 0.0,
                    "qdot_saturated": False,
                    "jacobian_condition": 0.0,
                    "nullspace_enabled": False,
                    "nullspace_gain": NULLSPACE_GAIN,
                    "qdot_task_norm": 0.0,
                    "qdot_null_norm": 0.0,
                    "qdot_total_norm": 0.0,
                    "posture_reference_distance": float(
                        np.linalg.norm(
                            q_home - fr3_data.qpos[fr3_qpos_indices]
                        )
                    ),
                    "null_task_leak": 0.0,
                    "task_residual_norm": 0.0,
                    "joint_speed_saturated": False,
                    "joint_positions": fr3_data.qpos[
                        fr3_qpos_indices
                    ].copy(),
                    "qdot_task": np.zeros(7),
                    "qdot_null": np.zeros(7),
                    "qdot_total": np.zeros(7),
                    "max_q_command_error": float(
                        np.max(np.abs(q_target - fr3_data.qpos[fr3_qpos_indices]))
                    ),
                }

            fr3_data.ctrl[fr3_actuator_indices] = q_target
            fr3_data.ctrl[fr3_gripper_actuator_id] = gripper_command
            mujoco.mj_step(fr3_model, fr3_data)
            fr3_actual_position, fr3_actual_rotation = read_site_pose(
                fr3_data,
                fr3_ee_site_id,
            )
            now = time.perf_counter()
            # Tracking diagnostics are measured after mj_step and do not alter
            # the command.
            target_actual_error = fr3_target_position - fr3_actual_position
            command_actual_error = fr3_command_position - fr3_actual_position
            command_target_gap = fr3_command_position - fr3_target_position
            omy_state_age = (
                max(0.0, now - last_message_time)
                if has_joint_state
                else float("inf")
            )
            fr3_command_session_rotvec = matrix_to_rotvec(
                fr3_command_rotation @ fr3_initial_command_rotation.T
            )
            fr3_target_session_rotvec = matrix_to_rotvec(
                fr3_target_rotation @ fr3_initial_command_rotation.T
            )
            fr3_actual_session_rotvec = matrix_to_rotvec(
                fr3_actual_rotation @ fr3_initial_command_rotation.T
            )

            if ENABLE_MAPPING_DEBUG_LOGS and omy_anchor_position is not None:
                fr3_command_position_delta = (
                    fr3_command_position - fr3_anchor_position
                )
                fr3_target_position_delta = (
                    fr3_target_position - fr3_anchor_position
                )
                fr3_actual_position_delta = (
                    fr3_actual_position - fr3_anchor_position
                )

            step_index += 1
            cycles_since_report += 1

            if (
                ENABLE_VIEWER
                and viewer is not None
                and now - last_viewer_sync >= 1.0 / VIEWER_HZ
            ):
                draw_target_marker(
                    viewer,
                    fr3_target_position,
                    fr3_target_rotation,
                )
                viewer.sync()
                last_viewer_sync = now

            if (
                ENABLE_LOGGING
                and step_index % max(1, int(CONTROL_HZ / LOG_HZ)) == 0
            ):
                cpu_wall_now = time.perf_counter()
                cpu_usage = resource.getrusage(resource.RUSAGE_SELF)
                cpu_time_now = cpu_usage.ru_utime + cpu_usage.ru_stime
                cpu_wall_delta = cpu_wall_now - cpu_wall_reference
                if cpu_wall_delta > 1e-9:
                    process_cpu_percent = (
                        100.0
                        * (cpu_time_now - cpu_time_reference)
                        / cpu_wall_delta
                    )
                cpu_wall_reference = cpu_wall_now
                cpu_time_reference = cpu_time_now

                log_row = {
                    "wall_time": now,
                    "sim_time": float(fr3_data.time),
                    "process_cpu_percent": process_cpu_percent,
                    "teleop_active": int(teleop_active),
                    "ros_fresh": int(ros_fresh),
                    "clutch_id": clutch_id,
                    "fr3_command_position_x": fr3_command_position[0],
                    "fr3_command_position_y": fr3_command_position[1],
                    "fr3_command_position_z": fr3_command_position[2],
                    "fr3_target_position_x": fr3_target_position[0],
                    "fr3_target_position_y": fr3_target_position[1],
                    "fr3_target_position_z": fr3_target_position[2],
                    "fr3_actual_position_x": fr3_actual_position[0],
                    "fr3_actual_position_y": fr3_actual_position[1],
                    "fr3_actual_position_z": fr3_actual_position[2],
                    "fr3_target_error_x_m": target_actual_error[0],
                    "fr3_target_error_y_m": target_actual_error[1],
                    "fr3_target_error_z_m": target_actual_error[2],
                    "fr3_target_error_norm_m": float(
                        np.linalg.norm(target_actual_error)
                    ),
                    "fr3_command_error_x_m": command_actual_error[0],
                    "fr3_command_error_y_m": command_actual_error[1],
                    "fr3_command_error_z_m": command_actual_error[2],
                    "fr3_command_error_norm_m": float(
                        np.linalg.norm(command_actual_error)
                    ),
                    "fr3_command_target_gap_norm_m": float(
                        np.linalg.norm(command_target_gap)
                    ),
                    "omy_state_age_s": omy_state_age,
                    "fr3_command_rotvec_x": fr3_command_session_rotvec[0],
                    "fr3_command_rotvec_y": fr3_command_session_rotvec[1],
                    "fr3_command_rotvec_z": fr3_command_session_rotvec[2],
                    "fr3_target_rotvec_x": fr3_target_session_rotvec[0],
                    "fr3_target_rotvec_y": fr3_target_session_rotvec[1],
                    "fr3_target_rotvec_z": fr3_target_session_rotvec[2],
                    "fr3_actual_rotvec_x": fr3_actual_session_rotvec[0],
                    "fr3_actual_rotvec_y": fr3_actual_session_rotvec[1],
                    "fr3_actual_rotvec_z": fr3_actual_session_rotvec[2],
                    "command_target_rotation_gap_deg": (
                        command_target_rotation_gap_deg
                    ),
                    "teleop_mode": diagnostics["teleop_mode"],
                    "position_error_norm": diagnostics["position_error_norm"],
                    "orientation_error_norm_rad": diagnostics[
                        "orientation_error_norm_rad"
                    ],
                    "linear_task_velocity_norm": diagnostics[
                        "linear_task_velocity_norm"
                    ],
                    "angular_task_velocity_norm": diagnostics[
                        "angular_task_velocity_norm"
                    ],
                    "weighted_task_twist_norm": diagnostics[
                        "weighted_task_twist_norm"
                    ],
                    "target_linear_speed_mps": target_linear_speed,
                    "target_linear_acceleration_norm_mps2": float(
                        np.linalg.norm(target_linear_acceleration)
                    ),
                    "control_dt": CONTROL_DT,
                    "target_angular_speed_radps": target_angular_speed,
                    "target_angular_acceleration_norm_radps2": float(
                        np.linalg.norm(target_angular_acceleration)
                    ),
                    "raw_max_qdot_radps": diagnostics["raw_max_qdot"],
                    "cmd_max_qdot_radps": diagnostics["cmd_max_qdot"],
                    "qdot_task_norm": diagnostics["qdot_task_norm"],
                    "task_residual_norm": diagnostics[
                        "task_residual_norm"
                    ],
                    "joint_speed_saturated": int(
                        diagnostics["joint_speed_saturated"]
                    ),
                    "omy_joint_1": float(omy_data.qpos[omy_qpos_addresses[0]]),
                    "omy_joint_2": float(omy_data.qpos[omy_qpos_addresses[1]]),
                    "omy_joint_3": float(omy_data.qpos[omy_qpos_addresses[2]]),
                    "omy_joint_4": float(omy_data.qpos[omy_qpos_addresses[3]]),
                    "omy_joint_5": float(omy_data.qpos[omy_qpos_addresses[4]]),
                    "omy_joint_6": float(omy_data.qpos[omy_qpos_addresses[5]]),
                    "omy_ee_position_x": float(omy_current_position[0]),
                    "omy_ee_position_y": float(omy_current_position[1]),
                    "omy_ee_position_z": float(omy_current_position[2]),
                    "omy_teleop_delta_x": float(omy_teleop_delta[0]),
                    "omy_teleop_delta_y": float(omy_teleop_delta[1]),
                    "omy_teleop_delta_z": float(omy_teleop_delta[2]),
                    "jacobian_condition": diagnostics["jacobian_condition"],
                    "deadline_misses": deadline_misses,
                    "fr3_joint_1": diagnostics["joint_positions"][0],
                    "fr3_joint_2": diagnostics["joint_positions"][1],
                    "fr3_joint_3": diagnostics["joint_positions"][2],
                    "fr3_joint_4": diagnostics["joint_positions"][3],
                    "fr3_joint_5": diagnostics["joint_positions"][4],
                    "fr3_joint_6": diagnostics["joint_positions"][5],
                    "fr3_joint_7": diagnostics["joint_positions"][6],
                }

                if ENABLE_MAPPING_DEBUG_LOGS:
                    log_row.update({
                        "omy_delta_position_x_m": (
                            omy_raw_clutch_position_delta[0]
                        ),
                        "omy_delta_position_y_m": (
                            omy_raw_clutch_position_delta[1]
                        ),
                        "omy_delta_position_z_m": (
                            omy_raw_clutch_position_delta[2]
                        ),
                        "omy_delta_rotvec_x_rad": omy_delta_rotvec[0],
                        "omy_delta_rotvec_y_rad": omy_delta_rotvec[1],
                        "omy_delta_rotvec_z_rad": omy_delta_rotvec[2],
                        "mapped_position_delta_x_m": (
                            fr3_command_position_delta[0]
                        ),
                        "mapped_position_delta_y_m": (
                            fr3_command_position_delta[1]
                        ),
                        "mapped_position_delta_z_m": (
                            fr3_command_position_delta[2]
                        ),
                        "mapped_rotation_delta_x_rad": (
                            fr3_mapped_rotation_delta[0]
                        ),
                        "mapped_rotation_delta_y_rad": (
                            fr3_mapped_rotation_delta[1]
                        ),
                        "mapped_rotation_delta_z_rad": (
                            fr3_mapped_rotation_delta[2]
                        ),
                    })

                if ENABLE_OMY_INTERNAL_DEBUG_LOGS:
                    log_row.update({
                        "omy_base_ee_position_x": omy_base_ee_position[0],
                        "omy_base_ee_position_y": omy_base_ee_position[1],
                        "omy_base_ee_position_z": omy_base_ee_position[2],
                        "omy_base_ee_rotvec_x": (
                            omy_base_ee_rotation_vector[0]
                        ),
                        "omy_base_ee_rotvec_y": (
                            omy_base_ee_rotation_vector[1]
                        ),
                        "omy_base_ee_rotvec_z": (
                            omy_base_ee_rotation_vector[2]
                        ),
                        "omy_clutch_base_delta_x": (
                            omy_clutch_base_position_delta[0]
                        ),
                        "omy_clutch_base_delta_y": (
                            omy_clutch_base_position_delta[1]
                        ),
                        "omy_clutch_base_delta_z": (
                            omy_clutch_base_position_delta[2]
                        ),
                        "omy_clutch_spatial_rotvec_x": (
                            omy_clutch_spatial_rotation_vector[0]
                        ),
                        "omy_clutch_spatial_rotvec_y": (
                            omy_clutch_spatial_rotation_vector[1]
                        ),
                        "omy_clutch_spatial_rotvec_z": (
                            omy_clutch_spatial_rotation_vector[2]
                        ),
                    })

                rows.append(log_row)

            if now - last_print >= 1.0 / PRINT_HZ:
                if ENABLE_MAPPING_DEBUG_LOGS:
                    print(
                        f"OMY dp={omy_raw_clutch_position_delta} | "
                        f"mapped dp={fr3_command_position_delta} | "
                        f"OMY dr={omy_delta_rotvec} | "
                        f"mapped dr={fr3_mapped_rotation_delta}"
                    )
                print(
                    f"pos={1000.0 * diagnostics['position_error_m']:.2f} mm | "
                    f"rot={diagnostics['orientation_error_deg']:.2f} deg | "
                    f"mode={TELEOP_MODE} | "
                    f"qdot={diagnostics['cmd_max_qdot']:.3f} rad/s | "
                    f"cond={diagnostics['jacobian_condition']:.1f} | "
                    f"task_res={diagnostics['task_residual_norm']:.4f} | "
                    f"sat={diagnostics['joint_speed_saturated']} | "
                    f"track_err={1000.0 * np.linalg.norm(target_actual_error):.2f} mm | "
                    f"state_age={1000.0 * omy_state_age:.1f} ms"
                )
                last_print = now

            if now - last_rate_report >= 1.0:
                measured_hz = cycles_since_report / (now - last_rate_report)
                print(
                    f"measured loop={measured_hz:.1f} Hz | "
                    f"deadline misses={deadline_misses} | "
                    f"process CPU={process_cpu_percent:.1f}%"
                )
                cycles_since_report = 0
                deadline_misses = 0
                last_rate_report = now

            next_tick += CONTROL_DT
            remaining = next_tick - time.perf_counter()

            if remaining > 0.0:
                time.sleep(remaining)
            else:
                deadline_misses += 1
                if remaining < -5.0 * CONTROL_DT:
                    next_tick = time.perf_counter()

    finally:
        if viewer_context is not None:
            viewer_context.__exit__(None, None, None)

        if ENABLE_LOGGING:
            write_log(log_path, rows)
            print(f"Log written: {log_path}")

        if input_mode == "ros":
            ros_node.destroy_node()
            rclpy.shutdown()
            ros_thread.join(timeout=1.0)
        else:
            ros_node.stop()


if __name__ == "__main__":
    main()
