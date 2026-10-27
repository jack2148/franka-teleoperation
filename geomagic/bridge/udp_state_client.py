"""Nonblocking latest-state UDP receiver; no threads or device dependencies."""

from dataclasses import dataclass, replace
import math
import socket
import time
from typing import Callable, Optional, Tuple

from geomagic.bridge.protocol import (
    MAX_PACKET_BYTES, ProtocolError, UnsupportedVersionError, decode_packet,
)
from geomagic.vendor.geomagic_core.state import GeomagicState
from geomagic.teleop.sync_capture import get_sync_capture


DEFAULT_STALE_TIMEOUT_S = 0.050


@dataclass(frozen=True)
class TransportStatus:
    packets_received: int
    packets_accepted: int
    invalid_packets: int
    duplicate_packets: int
    reordered_packets: int
    unsupported_version_packets: int
    session_changes: int
    poll_budget_hits: int
    last_sequence: Optional[int]
    current_session_id: Optional[str]
    session_changed: bool
    fresh: bool
    stale: bool
    age_ms: Optional[float]
    time_since_last_valid_packet_ms: Optional[float]


class GeomagicUDPStateClient:
    """Bind once and poll a single trusted producer, retaining one state.

    Import/constructor never starts a thread. The default bind is localhost
    with an ephemeral port; address reports the selected address/port. A remote
    deployment must explicitly choose a suitable bind address.

    poll drains to EWOULDBLOCK by default. An optional packet budget bounds work
    under continuous traffic; remaining datagrams wait in the bounded OS socket
    buffer until the next poll. This is not a hard real-time scheduler.
    """

    def __init__(
        self,
        bind_address: Tuple[str, int] = ("127.0.0.1", 0),
        *,
        stale_timeout_s: float = DEFAULT_STALE_TIMEOUT_S,
        monotonic: Callable[[], float] = time.monotonic,
        max_packets_per_poll: Optional[int] = None,
    ) -> None:
        self.sync = get_sync_capture()
        self.accepted_ns = None
        if (isinstance(stale_timeout_s, bool) or not isinstance(stale_timeout_s, (int, float))
                or not math.isfinite(stale_timeout_s) or stale_timeout_s <= 0):
            raise ValueError("stale_timeout_s must be finite and positive")
        if max_packets_per_poll is not None and (
            type(max_packets_per_poll) is not int or max_packets_per_poll <= 0
        ):
            raise ValueError("max_packets_per_poll must be a positive integer or None")
        self.stale_timeout_s = float(stale_timeout_s)
        self._clock = monotonic
        self._budget = max_packets_per_poll
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._socket.setblocking(False)
            self._socket.bind(bind_address)
        except BaseException:
            self._socket.close()
            raise
        self.address = self._socket.getsockname()
        self._closed = False
        self._latest: Optional[GeomagicState] = None
        self._last_valid_receive: Optional[float] = None
        self._last_sequence: Optional[int] = None
        self._session: Optional[str] = None
        self._session_changed = False
        self._counts = dict.fromkeys((
            "packets_received", "packets_accepted", "invalid_packets",
            "duplicate_packets", "reordered_packets", "unsupported_version_packets",
            "session_changes", "poll_budget_hits",
        ), 0)

    def poll(self) -> Optional[GeomagicState]:
        """Drain pending datagrams; return latest accepted state, else None.

        A structurally valid packet with valid=False is accepted to invalidate
        the latest state immediately, but does not refresh valid-pose freshness.
        No rejected packet changes the sequence, session, state, or receive time.
        """
        if self._closed:
            raise RuntimeError("receiver is closed")
        accepted = False
        processed = 0
        while self._budget is None or processed < self._budget:
            try:
                payload, _sender = self._socket.recvfrom(MAX_PACKET_BYTES + 1)
            except BlockingIOError:
                break
            received_at = self._clock()
            processed += 1
            self._counts["packets_received"] += 1
            try:
                packet = decode_packet(payload)
            except UnsupportedVersionError:
                self._counts["unsupported_version_packets"] += 1
                continue
            except ProtocolError:
                self._counts["invalid_packets"] += 1
                continue
            if packet.session_id == self._session:
                if packet.sequence == self._last_sequence:
                    self._counts["duplicate_packets"] += 1
                    continue
                if packet.sequence < self._last_sequence:
                    self._counts["reordered_packets"] += 1
                    continue
                if packet.source_clock_domain != self._latest.source_clock_domain:
                    self._counts["invalid_packets"] += 1
                    continue
            else:
                if self._session is not None:
                    self._counts["session_changes"] += 1
                self._session_changed = True
                self._last_valid_receive = None
            self._session = packet.session_id
            self._last_sequence = packet.sequence
            if packet.valid:
                self._last_valid_receive = received_at
            self._latest = GeomagicState(
                position_xyz=packet.position_m,
                orientation_xyzw=packet.orientation_xyzw,
                buttons=(packet.grey_button, packet.white_button),
                timestamp=received_at,
                valid=packet.valid,
                sequence=packet.sequence,
                session_id=packet.session_id,
                source_timestamp=packet.source_timestamp,
                source_clock_domain=packet.source_clock_domain,
                source_valid=packet.valid,
            )
            self._counts["packets_accepted"] += 1
            if self.sync:
                self.accepted_ns = time.monotonic_ns()
            accepted = True
        if self._budget is not None and processed == self._budget:
            self._counts["poll_budget_hits"] += 1
        return self.latest_state() if accepted else None

    def latest_state(self) -> Optional[GeomagicState]:
        """Return a freshness-adjusted snapshot without reading the socket."""
        if self._latest is None:
            return None
        return replace(self._latest, valid=self._latest.source_valid and self.status().fresh)

    def status(self) -> TransportStatus:
        """Read counters and freshness; does not poll or acknowledge events.

        stale means no valid-pose receive time or age > timeout. A fresh-age
        source invalidation has fresh=False but need not yet have stale=True.
        Counter categories are disjoint; packets_accepted includes valid=False.
        """
        age = (None if self._last_valid_receive is None
               else max(0.0, self._clock() - self._last_valid_receive))
        stale = age is None or age > self.stale_timeout_s
        fresh = bool(not self._closed and not stale and self._latest is not None
                     and self._latest.source_valid)
        age_ms = None if age is None else age * 1000.0
        return TransportStatus(
            **self._counts,
            last_sequence=self._last_sequence,
            current_session_id=self._session,
            session_changed=self._session_changed,
            fresh=fresh,
            stale=stale,
            age_ms=age_ms,
            time_since_last_valid_packet_ms=age_ms,
        )

    def acknowledge_session_change(self) -> None:
        """Clear the latched event only after downstream handles reanchoring.

        First attachment also latches the event. session_changes counts only
        subsequent transitions. The network layer itself never moves a robot.
        """
        self._session_changed = False

    def close(self) -> None:
        if not self._closed:
            self._socket.close()
            self._closed = True

    def __enter__(self) -> "GeomagicUDPStateClient":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
