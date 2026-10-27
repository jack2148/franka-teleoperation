"""Stateful release-only gate; never changes robot targets or physics contacts."""
import math

import numpy as np

from .haptic_config import RetreatConfig


class RetreatRelease:
    """Latch retreat until separation, or a previously unseen external geom pair.

    Velocity uses unconditioned device positions and physical acquisition time.
    Repeated 1 kHz host ticks holding the same 500 Hz sample cannot confirm
    retreat. Invalid samples, clock/session changes and gaps break confirmation.
    A temporary cancellation of net force does not end a physical episode.
    """
    def __init__(self, config=None, *, max_sample_gap_s=0.020):
        self.config = config if config is not None else RetreatConfig()
        self.max_sample_gap_s = max_sample_gap_s
        self.contact_active = False
        self.contact_episode_id = 0
        self.retreat_suppressed = False
        self.haptic_force_N = 0.0
        self._seen_contacts = set()
        self._clear_motion()

    def _clear_motion(self):
        self._sample = None
        self._velocity = None
        self._candidate_since = None
        self.retreat_score_mps = None
        self.retreat_detected = False

    @staticmethod
    def _contact_ids(contacts):
        if contacts.contact_ids:
            return set(contacts.contact_ids)
        # Compatibility with offline fixtures / callers providing body pairs.
        return {(item.get('class'), tuple(sorted(item['body_pair'])))
                for item in contacts.selected if 'body_pair' in item}

    def _motion(self, state):
        position = np.asarray(state.position_xyz, dtype=float)
        physical = state.source_clock_domain == 'server.monotonic.hdapi'
        stamp = state.source_timestamp if physical else state.timestamp
        if (position.shape != (3,) or not np.isfinite(position).all()
                or stamp is None or not math.isfinite(stamp)
                or state.frame_id != 'geomagic_base' or state.units != 'm'):
            self._clear_motion()
            return None
        current = (state.session_id, state.source_clock_domain, state.sequence,
                   stamp, position.copy())
        previous = self._sample
        if previous is not None and current[:3] == previous[:3]:
            return None  # Never count a held sample as new motion evidence.
        self._sample = current
        if previous is None:
            return None
        dt = stamp - previous[3]
        if (current[:2] != previous[:2] or state.sequence <= previous[2]
                or not 0 < dt <= self.max_sample_gap_s):
            self._clear_motion()
            self._sample = current
            return None
        self._velocity = (position - previous[4]) / dt
        if not np.isfinite(self._velocity).all():
            self._clear_motion()
            return None
        return previous[3], stamp

    def apply(self, force, contacts, state):
        """Only replace an existing nonzero cue with zero; preserve all gates."""
        active = bool(contacts.contact_ids or contacts.selected or
                      contacts.counts.get('ee_object', 0) or
                      contacts.counts.get('ee_environment', 0))
        identities = self._contact_ids(contacts)
        new_contact = self.retreat_suppressed and bool(identities - self._seen_contacts)
        if active and (not self.contact_active or new_contact):
            self.contact_episode_id += 1
            self.retreat_suppressed = False
            self._clear_motion()
            if not self.contact_active:
                self._seen_contacts.clear()
        if not active:
            self.retreat_suppressed = False
            self._seen_contacts.clear()
            self._clear_motion()
        self.contact_active = active
        self._seen_contacts.update(identities)

        magnitude = float(np.linalg.norm(force))
        if magnitude > 0 and state is not None:
            interval = self._motion(state)
            if self._velocity is not None:
                self.retreat_score_mps = float(np.dot(self._velocity, force / magnitude))
                if self.retreat_score_mps <= self.config.speed_threshold_mps:
                    self._candidate_since = None
                    self.retreat_detected = False
                elif interval is not None:
                    if self._candidate_since is None:
                        self._candidate_since = interval[0]
                    self.retreat_detected = bool(
                        interval[1] - self._candidate_since >= self.config.confirm_ms / 1000 - 1e-12)
                    if self.retreat_detected:
                        self.retreat_suppressed = True
        else:
            # Existing deadman/clutch/freshness/zero gates interrupt motion
            # evidence, but cannot re-arm a suppressed physical contact.
            self._clear_motion()
        result = np.zeros(3) if self.retreat_suppressed else force
        self.haptic_force_N = float(np.linalg.norm(result))
        return result

    def diagnostics(self):
        return dict(contact_active=self.contact_active, haptic_force_N=self.haptic_force_N,
                    retreat_score_mps=self.retreat_score_mps,
                    retreat_detected=self.retreat_detected,
                    retreat_suppressed=self.retreat_suppressed,
                    contact_episode_id=self.contact_episode_id)
