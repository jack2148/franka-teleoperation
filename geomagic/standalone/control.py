"""Minimal scene orchestration around the unchanged production backend."""
import sys
import time
from pathlib import Path

import numpy as np

from geomagic.teleop.robot_helpers import MIMIC_ROOT
if not (MIMIC_ROOT / "runtime/live_runner.py").is_file():
    raise ImportError("Square standalone requires franka-imitation-learning; set FRANKA_LEARNING_ROOT to its checkout")
if str(MIMIC_ROOT) not in sys.path:
    sys.path.insert(0, str(MIMIC_ROOT))

from runtime.live_runner import _create_environment, CONTROL_DT
from mimic.envs.adapter import OmyFr3MimicAdapter, TeleopCommand
from geomagic.leader.teleop_backend import GeomagicTeleopBackend
from geomagic.teleop.pose_mapper import POSITION_SCALE, ROTATION_SCALE, MODE_LABELS, M_POSITION
from .haptics import ContactSelector, RetreatRelease, contact_force

KEY_MODES = {49: 'position', 50: 'rotation_z', 51: 'rotation', 52: '6dof'}


class StandaloneControl:
    """One authoritative model/data; no UI, recorder, plan or seed allocator.

    The fixed seed only initializes the existing scene. It is not a
    collection seed and is never accepted, advanced, allocated or stored.
    Tests may inject an in-memory accepted-state source without a socket.
    """
    def __init__(self, *, state_port=5568, client=None, monotonic=None):
        self.env, self.xml_path, _ = _create_environment('mimicgen_square', (480, 640), 42)
        self.model, self.data = self.env.model, self.env.data
        if not np.isclose(self.model.opt.timestep, CONTROL_DT, atol=1e-12):
            raise RuntimeError('scene must use the production 1 ms physics step')
        self.adapter = OmyFr3MimicAdapter(self.model)
        options = {'bind_address': ('127.0.0.1', state_port)}
        if monotonic is not None:
            options['monotonic'] = monotonic
        self.backend = GeomagicTeleopBackend(self.model, self.data, **options)
        if self.backend.sync:
            self.backend.sync.settings['recording_enabled'] = False
        self.clock = monotonic or time.monotonic
        if client is None:
            self.backend.initialize()
        else:
            self.backend.client = client
        self.selector = ContactSelector(self.model)
        self.haptic_release = RetreatRelease()
        self.haptics_enabled = True
        self.ticks = 0
        self.last_contacts = None

    def key(self, code):
        if code in KEY_MODES:
            self.backend.set_teleop_mode(KEY_MODES[code])
        elif code == 32:
            self.backend.set_gripper_command(0 if self.backend.gripper_command > 0 else 255)
        elif code in (72, 104):
            self.haptics_enabled = not self.haptics_enabled

    def step(self):
        b = self.backend
        b.update_command(CONTROL_DT)
        b.apply_controller(self.model, self.data, CONTROL_DT)
        action = self.adapter.convert(TeleopCommand(
            joint_position=b.hold_q_target, gripper_command=b.gripper_command))
        self.env.control(action)
        self.env.step_physics()
        self.ticks += 1
        if b.sync:
            b.sync.tick(b)
        self.last_contacts = self.selector.collect(self.data)
        timing = None
        if b.sync:
            timing = {'H0_ns': time.monotonic_ns()}
        force = contact_force(b, self.data, self.last_contacts,
            enabled=self.haptics_enabled, now=self.clock(), release=self.haptic_release,
            diagnostics=timing)
        if b.sync:
            self.sync_force = timing
        return action, force

    def close(self):
        self.backend.shutdown()
        self.env.close()


def config_summary():
    return {'position_scale': list(POSITION_SCALE), 'rotation_scale': ROTATION_SCALE,
            'keys': {str(key - 48): MODE_LABELS[mode] for key, mode in KEY_MODES.items()},
            'task': 'mimicgen_square', 'recording': False, 'collection_plan': None,
            'input_backend': 'geomagic', 'haptics_default': True}
