"""UI orchestration around teleop_geo's shared contact law and wire sender.

No device initialization. The UI parent owns HDAPI; its WORK child receives
only the existing nonblocking command socket, never an HDAPI device handle.
"""
import time

import numpy as np

from .haptics import ContactSelector, RetreatRelease, contact_force, FORCE_N, HAPTIC_MODE


def haptic_metadata(enabled):
    return dict(haptics_enabled=bool(enabled),
                haptic_mode=HAPTIC_MODE if enabled else 'OFF',
                haptic_force_N=FORCE_N if enabled else 0.0)


def haptic_status(enabled, *, contacts=None, force=None, release=None):
    counts = contacts.counts if contacts is not None else {}
    external = sum(counts.get(k, 0) for k in ('ee_object', 'ee_environment', 'arm_environment'))
    status = dict(available=bool(enabled), enabled=bool(enabled),
                  message=f'ON — {FORCE_N:.1f} N contact feedback' if enabled else 'OFF',
                  contact='external' if external else 'self' if counts.get('self') else 'none',
                  renderable_contact=contacts.renderable if contacts is not None else False,
                  force_target_n=float(np.linalg.norm(force)) if force is not None else 0.0,
                  watchdog='STALE' if enabled else 'OFF', **haptic_metadata(enabled))
    if release is not None:
        status.update(release.diagnostics())
    return status


class UIContactHaptics:
    def __init__(self, backend, data, sender, *, clock=time.monotonic):
        self.backend, self.data, self.sender, self.clock = backend, data, sender, clock
        self.selector = ContactSelector(backend.model)
        self.haptic_release = RetreatRelease()
        self.last_contacts = None
        self.force = np.zeros(3)
        self.closed = False
        self._snapshot = haptic_status(True, release=self.haptic_release)
        self._snapshot_at = 0.0

    def step(self):
        if self.closed:
            return
        self.last_contacts = self.selector.collect(self.data)
        timing = {'H0_ns': time.monotonic_ns()} if self.backend.sync else None
        self.force = contact_force(self.backend, self.data, self.last_contacts,
                                   enabled=True, now=self.clock(), release=self.haptic_release,
                                   diagnostics=timing)
        if not self.sender.send(self.force, enabled=True, diagnostics=timing,
                                sim_time=float(self.data.time)):
            raise RuntimeError('haptic command channel stopped; force and teleop stopped')

    def snapshot(self):
        now = self.clock()
        if now - self._snapshot_at >= .1 or self.closed:
            self._snapshot = haptic_status(True, contacts=self.last_contacts, force=self.force,
                                           release=self.haptic_release)
            self._snapshot_at = now
        return dict(self._snapshot)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.force[:] = 0
        self.haptic_release.haptic_force_N = 0.0
        self.sender.close()  # Explicit zero precedes renderer/recorder cleanup.
