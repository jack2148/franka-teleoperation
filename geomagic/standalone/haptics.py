"""External contact normals and the isolated 1 N test force wire contract."""
from dataclasses import dataclass
import math
import struct
import time

import mujoco
import numpy as np

from geomagic.teleop.pose_mapper import M_POSITION
from .haptic_release import RetreatRelease

FORCE_N = 1.0
HAPTIC_MODE = 'CONTACT-GATED / FIXED MAGNITUDE'
WATCHDOG_S = 0.020
SLEW_N_PER_S = 3.0
RESTORING_FORCE_SIGN = 1.0  # source is already the environment force on robot
FORCE_PACKET = struct.Struct('<8sQQd3dII')
FORCE_MAGIC = b'TGFORCE1'
MIN_NORMAL_FORCE_N = 1e-6
EE_BODIES = frozenset(('fr3_link7', 'fr3_hand', 'fr3_left_finger', 'fr3_right_finger'))
CONTACT_CLASSES = ('ee_object', 'ee_environment', 'arm_environment', 'self', 'other')


def fixed_device_force(world_direction, *, contact, deadman, fresh=True,
                       state_valid=True, enabled=True, clutched=False, diagnostics=None):
    """Normalize a restoring world direction; no rotation signs or orientation."""
    zero = np.zeros(3)
    if diagnostics is not None:
        diagnostics['H1_ns'] = time.monotonic_ns()
    if not (contact and deadman and fresh and state_valid and enabled) or clutched:
        return zero
    direction = np.asarray(world_direction, dtype=float)
    if direction.shape != (3,) or not np.isfinite(direction).all():
        return zero
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm <= 1e-12:
        return zero
    world = RESTORING_FORCE_SIGN * direction / norm
    if diagnostics is not None:
        diagnostics['H1_ns'] = time.monotonic_ns()
    device = np.asarray(M_POSITION).T @ world
    return device * (FORCE_N / np.linalg.norm(device))


def force_packet(token, sequence, timestamp, force, *, enabled):
    vector = np.asarray(force, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all() or not math.isfinite(timestamp):
        vector = np.zeros(3)
        enabled = False
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm):
        vector = np.zeros(3)
        enabled = False
    elif norm > FORCE_N:
        vector *= FORCE_N / norm
    return FORCE_PACKET.pack(FORCE_MAGIC, token, sequence, timestamp,
                             *vector, int(bool(enabled and np.linalg.norm(vector) > 0)), 0)


@dataclass(frozen=True)
class ContactResult:
    direction_world: np.ndarray
    counts: dict
    selected: tuple
    invalid: bool = False
    diagnostic_pairs: tuple = ()
    contact_ids: tuple = ()  # True external EE geom pairs, including zero-force contacts.

    @property
    def renderable(self):
        total = sum(contact['normal_force_n'] for contact in self.selected)
        return bool(bool(self.selected) and not self.invalid and
                    np.linalg.norm(self.direction_world) > max(MIN_NORMAL_FORCE_N, total * 1e-6))


class ContactSelector:
    """Render hand/fingers external contacts, classify arm and self separately.

    MuJoCo contact frame row 0 points geom[0] -> geom[1]. A positive
    normal force acts +normal on geom[1], -normal on geom[0]. Friction
    is deliberately excluded from this initial normal-only test.
    Opposing normals that cancel produce zero; no arbitrary fallback axis.
    """
    def __init__(self, model, object_body='square_nut'):
        self.model = model
        robot_root = model.body('base').id
        self.robot = set()
        for body in range(model.nbody):
            current = body
            while current > 0 and current != robot_root:
                current = int(model.body_parentid[current])
            if current == robot_root:
                self.robot.add(body)
        self.ee = {model.body(name).id for name in EE_BODIES}
        self.object = model.body(object_body).id
        self.scratch = np.zeros(6)

    def classify(self, body0, body1):
        robot0, robot1 = body0 in self.robot, body1 in self.robot
        if robot0 and robot1:
            return 'self'
        if robot0 == robot1:
            return 'other'
        robot, external = (body0, body1) if robot0 else (body1, body0)
        if robot in self.ee:
            return 'ee_object' if external == self.object else 'ee_environment'
        return 'arm_environment'

    def collect(self, data):
        counts = dict.fromkeys(CONTACT_CLASSES, 0)
        direction = np.zeros(3)
        selected = []
        diagnostic_pairs = []
        contact_ids = set()
        invalid = False
        for index in range(data.ncon):
            contact = data.contact[index]
            geom0, geom1 = int(contact.geom[0]), int(contact.geom[1])
            if geom0 < 0 or geom1 < 0:  # unsupported flex contacts fail closed
                invalid = True
                continue
            body0, body1 = int(self.model.geom_bodyid[geom0]), int(self.model.geom_bodyid[geom1])
            category = self.classify(body0, body1)
            counts[category] += 1
            if category not in ('ee_object', 'ee_environment'):
                if category in ('self', 'arm_environment'):
                    diagnostic_pairs.append({'class': category,
                        'body_pair': (self.model.body(body0).name, self.model.body(body1).name)})
                continue
            contact_ids.add(tuple(sorted((geom0, geom1))))
            mujoco.mj_contactForce(self.model, data, index, self.scratch)
            magnitude = float(self.scratch[0])
            normal = np.asarray(contact.frame[:3], dtype=float)
            if not math.isfinite(magnitude) or not np.isfinite(normal).all():
                invalid = True
                continue
            if magnitude <= MIN_NORMAL_FORCE_N or np.linalg.norm(normal) <= 1e-12:
                continue
            normal = normal / np.linalg.norm(normal)
            restoring = normal if body1 in self.robot else -normal
            direction += magnitude * restoring
            selected.append({
                'class': category,
                'body_pair': (self.model.body(body0).name, self.model.body(body1).name),
                'normal_on_robot_world': restoring.tolist(), 'normal_force_n': magnitude,
            })
        if invalid or not np.isfinite(direction).all():
            direction[:] = 0
        return ContactResult(direction, counts, tuple(selected), invalid,
                             tuple(diagnostic_pairs), tuple(sorted(contact_ids)))


def contact_force(backend, data, contacts, *, enabled, now, release, diagnostics=None):
    """Shared original host gates followed by the release-only retreat latch."""
    finite = all(np.isfinite(v).all() for v in (data.qpos, data.qvel, data.ctrl))
    state = backend._latest_state
    age = now - state.timestamp if state is not None else float('inf')
    fresh = bool(np.isfinite(age) and 0 <= age < WATCHDOG_S)
    if state is not None and state.source_clock_domain == 'server.monotonic.hdapi':
        source_age = now - state.source_timestamp if state.source_timestamp is not None else float('inf')
        fresh &= bool(np.isfinite(source_age) and 0 <= source_age < WATCHDOG_S)
    fresh = backend.input_fresh and fresh
    force = fixed_device_force(contacts.direction_world,
        contact=contacts.renderable, deadman=backend.teleop_active,
        fresh=fresh, state_valid=finite and state is not None and state.valid,
        enabled=enabled, clutched=state is not None and state.white_button,
        diagnostics=diagnostics)
    force = release.apply(force, contacts, state)
    if diagnostics is not None:
        diagnostics.update(contact=contacts.renderable,
            world_force=(np.asarray(M_POSITION) @ force).tolist(), device_force=force.tolist(),
            grey=state is not None and state.grey_button,
            white=state is not None and state.white_button, fresh=fresh,
            **backend.sync.preview_context(), **release.diagnostics())
    if not finite:
        raise RuntimeError('non-finite MuJoCo state; force disabled')
    return force
