"""Explicit manual physical entry point. Never imported by the production UI."""
import argparse
import json
import os
import signal
import time

import numpy as np

from .control import StandaloneControl, config_summary
from .device import HapticOwner, OwnerLock, preflight
from .haptics import FORCE_N, WATCHDOG_S
from .viewer import ViewerBridge


def _viewer_running(viewer):
    running = viewer.running()
    if not running and hasattr(viewer, 'raise_if_failed'):
        viewer.raise_if_failed()
    return running


def _owner_exited(owner, context):
    code = owner.process.poll()
    detail = owner.exit_description() if hasattr(owner, 'exit_description') else f'code {code}'
    return RuntimeError(f'HDAPI owner exited {context} ({detail}); force and teleop stopped')


def run_session(control, owner, viewer, *, stop_requested=lambda: False):
    """Force-off cleanup is unconditional on viewer close, signal or exception."""
    try:
        if stop_requested(): return
        viewer.start(control.data)
        wait_until = time.monotonic() + 15
        while not viewer.ready.wait(.05):
            if stop_requested() or not _viewer_running(viewer): return
            if time.monotonic() > wait_until: raise RuntimeError('MuJoCo viewer startup timed out')
        if stop_requested() or not _viewer_running(viewer): return
        # A GLX/context failure must be discovered before enabling device output.
        owner.start()
        owner.wait_ready(stop_requested=lambda: stop_requested() or not _viewer_running(viewer))
        started = time.monotonic()
        start_sim = float(control.data.time)
        deadline = started
        next_display = started
        next_status = started + 1
        previous_time = started
        previous_ticks = control.ticks
        previous_dls = control.backend.control_updates
        while not stop_requested() and _viewer_running(viewer):
            if owner.process.poll() is not None: raise _owner_exited(owner, 'during teleop')
            for key in viewer.drain_keys(): control.key(key)
            _, force = control.step()
            # Recheck viewer close and signal after the physics tick before
            # issuing any nonzero command; exit always sends explicit zero.
            if stop_requested() or not _viewer_running(viewer): break
            send_options = ({'diagnostics': control.sync_force, 'sim_time': float(control.data.time)}
                            if getattr(control.backend, 'sync', None) else {})
            if not owner.send(force, enabled=control.haptics_enabled, **send_options):
                detail = owner.send_failure_description() if hasattr(owner, 'send_failure_description') else 'force command send failed'
                raise RuntimeError(detail + '; force and teleop stopped')
            now = time.monotonic()
            if now >= next_display:
                viewer.publish(control.data)
                next_display = now + 1 / 60
            if now >= next_status:
                elapsed = now - previous_time
                contacts = control.last_contacts
                external_count = sum(contacts.counts[name] for name in ('ee_object','ee_environment','arm_environment'))
                diagnostics = {
                    'contact': 'external' if external_count else 'self' if contacts.counts['self'] else 'none',
                    'contact_classes': contacts.counts,
                    'renderable_contact': contacts.renderable,
                    'selected_pairs': [item['body_pair'] for item in contacts.selected],
                    'selected_normals_world': [item['normal_on_robot_world'] for item in contacts.selected],
                    'nonrendered_robot_pairs': contacts.diagnostic_pairs,
                    'raw_selected_normal_force_n': sum(item['normal_force_n'] for item in contacts.selected),
                    'force_target_n': float(np.linalg.norm(force)),
                    'force_direction_device': force.tolist(),
                    'rendered_force_n': owner.status.get('rendered_force_n'),
                    'rendered_force_device': owner.status.get('force_device'),
                    'servo_hz': owner.status.get('servo_hz'), 'state_hz': owner.status.get('state_hz'),
                    'physics_hz': (control.ticks - previous_ticks) / elapsed,
                    'dls_hz': (control.backend.control_updates - previous_dls) / elapsed,
                    'rtf': (float(control.data.time) - start_sim) / (now - started),
                    'watchdog': owner.status.get('watchdog', 'WAITING'),
                    'mode': control.backend.pipeline.mapper.mode, 'haptics_enabled': control.haptics_enabled,
                    **control.haptic_release.diagnostics(),
                }
                print(json.dumps(diagnostics, allow_nan=False), flush=True)
                previous_ticks, previous_dls, previous_time = control.ticks, control.backend.control_updates, now
                next_status = now + 1
            deadline += .001
            if now - deadline > .010: deadline = now  # No large catch-up burst.
            delay = deadline - time.monotonic()
            if delay > 0: time.sleep(delay)
    finally:
        # Stop the device before waiting for visualization/control cleanup.
        try:
            owner.close()
        finally:
            try: viewer.close()
            finally: control.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description='Standalone physical Touch + MuJoCo FR3, contact-gated 1 N haptic test')
    parser.add_argument('--state-port', type=int, default=5568)
    parser.add_argument('--config', action='store_true', help='print configuration only; no device/viewer/preflight')
    args = parser.parse_args(argv)
    if args.config:
        print(json.dumps({**config_summary(), 'force_n': FORCE_N, 'watchdog_s': WATCHDOG_S, 'state_port': args.state_port}, indent=2))
        return 0
    if not 1 <= args.state_port <= 65535: parser.error('state port must be 1..65535')
    stopped = False
    def request_stop(signum, frame):
        nonlocal stopped
        stopped = True
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, request_stop)
    control = owner = viewer = None
    try:
        with OwnerLock():
            runtime = preflight(state_port=args.state_port)
            if stopped: raise KeyboardInterrupt
            if not os.environ.get('DISPLAY'):
                raise RuntimeError('DISPLAY is required for the local MuJoCo viewer')
            print('teleop_geo\nPosition scale: 0.7\nRotation scale: 0.4\n'
                  'Modes: 1 Position / 2 Rotation Z / 3 Rotation 3D / 4 6DoF\n'
                  'Space: gripper open/close; Grey: deadman; White: clutch; H: haptics ON/OFF\n'
                  'Haptic: CONTACT-GATED / FIXED MAGNITUDE = 1.0 N / MAX FORCE = 1.0 N\n'
                  'State rate target: 500 Hz / Haptic servo: ~1 kHz / Physics: 1 kHz\n'
                  'Watchdog: 20 ms / onset vector slew: 3 N/s / no-contact release: immediate\n'
                  'Close viewer or Ctrl+C to exit.', flush=True)
            control = StandaloneControl(state_port=args.state_port)
            owner = HapticOwner(runtime, args.state_port)
            viewer = ViewerBridge(control.model, control.xml_path)
            run_session(control, owner, viewer, stop_requested=lambda: stopped)
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f'teleop_geo: {error}', flush=True)
        return 1
    finally:
        # Also cover partial construction. Methods are idempotent for devices
        # and viewers; avoid duplicate backend shutdown after run_session.
        if owner is not None and not owner.closed: owner.close()
        if viewer is not None and not viewer.closed.is_set(): viewer.close()
        if control is not None and control.backend.client is not None: control.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
