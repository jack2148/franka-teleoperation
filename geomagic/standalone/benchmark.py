"""Hardware-free standalone control benchmark. No launcher/HDAPI/viewer/recorder."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from .control import StandaloneControl, config_summary
from .haptics import force_packet
from geomagic.teleop.synthetic import sample
from geomagic.teleop.validate_gravity_timing import OfflineClient


def benchmark(seconds=5):
    phases = []
    for mode in ('idle', 'position', 'rotation_z', 'rotation', '6dof'):
        client = OfflineClient()
        control = StandaloneControl(client=client)
        try:
            if mode != 'idle': control.backend.set_teleop_mode(mode)
            tick = 0
            sequence = 0
            def step():
                nonlocal tick, sequence
                if tick % 2 == 0:
                    now = time.monotonic()
                    sim = float(control.data.time)
                    rot = ((0, .15*np.sin(sim), 0) if mode == 'rotation_z' else
                           (.15*np.sin(sim), 0, 0) if mode in ('rotation', '6dof') else (0,0,0))
                    position = (.01*np.sin(sim), .01*np.sin(.7*sim), .01*np.sin(.5*sim)) if mode in ('position', '6dof') else (0,0,0)
                    client.state = sample(sequence, time=now, grey=mode != 'idle', position=position, rotvec=rot)
                    sequence += 1
                _, force = control.step()
                force_packet(42, tick + 1, time.monotonic(), force, enabled=True)
                tick += 1
            deadline = time.monotonic()
            for _ in range(250):
                step()
                deadline += .001
                if deadline > time.monotonic(): time.sleep(deadline - time.monotonic())
            started = time.monotonic()
            start_sim = float(control.data.time)
            start_steps, start_dls = control.ticks, control.backend.control_updates
            deadline = started
            while time.monotonic() - started < seconds:
                step()
                deadline += .001
                now = time.monotonic()
                if now - deadline > .010: deadline = now
                if deadline > now: time.sleep(deadline - now)
            elapsed = time.monotonic() - started
            result = {'mode': mode, 'wall_seconds': elapsed,
                'physics_hz': (control.ticks - start_steps) / elapsed,
                'dls_hz': (control.backend.control_updates - start_dls) / elapsed,
                'rtf': (float(control.data.time) - start_sim) / elapsed,
                'physics_dt_s': control.model.opt.timestep,
                'dls_dt_s': control.backend.last_control_dt,
                'dataset_samples': 0, 'physical_device': False, 'viewer': False,
                'input_transport': 'in-memory accepted 500 Hz samples',
                'force_transport': 'wire serialization only; no HDAPI process'}
            phases.append(result)
            print(json.dumps(result), flush=True)
        finally: control.close()
    return {'config': config_summary(), 'phases': phases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0: parser.error('seconds must be positive')
    result = benchmark(args.seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')

if __name__ == '__main__': main()
