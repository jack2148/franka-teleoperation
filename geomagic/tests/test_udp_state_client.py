"""Deterministic fake-clock transport tests and a real localhost UDP check."""

from collections import deque
from dataclasses import replace
import select
import socket

import pytest

from geomagic.bridge.protocol import MAX_PACKET_BYTES, encode_packet
from geomagic.bridge.udp_state_client import GeomagicUDPStateClient
from geomagic.tests.test_protocol import example_packet, wire


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


class DatagramSocket:
    def __init__(self):
        self.pending = deque()
        self.blocking = True
        self.closed = False

    def setblocking(self, blocking):
        self.blocking = blocking

    def bind(self, address):
        self.address = address

    def getsockname(self):
        return self.address

    def recvfrom(self, length):
        assert self.blocking is False, "poll must never use a blocking socket"
        if not self.pending:
            raise BlockingIOError
        return self.pending.popleft()[:length], ("127.0.0.1", 43210)

    def close(self):
        self.closed = True


@pytest.fixture
def transport(monkeypatch):
    clock = Clock()
    sock = DatagramSocket()
    monkeypatch.setattr("geomagic.bridge.udp_state_client.socket.socket", lambda *args: sock)
    client = GeomagicUDPStateClient(monotonic=clock)
    yield client, sock, clock
    client.close()


def send(sock, sequence=1, session_id="boot-a", **changes):
    packet = replace(example_packet(), sequence=sequence, session_id=session_id, **changes)
    sock.pending.append(encode_packet(packet))


def test_valid_packet_retains_pose_buttons_and_distinct_timestamps(transport):
    client, sock, clock = transport
    clock.now = 5.0
    send(sock, source_timestamp=9e12)
    state = client.poll()
    assert state.position_xyz == (0.012, -0.034, 0.056)
    assert state.orientation_xyzw == (0.0, 0.0, 0.6, 0.8)
    assert state.buttons == (True, False)
    assert state.grey_button is True and state.white_button is False
    assert state.timestamp == state.server_receive_monotonic_time == 5.0
    assert state.source_timestamp == 9e12
    assert state.source_clock_domain == "touch-pc.monotonic"
    status = client.status()
    assert state.valid and status.fresh and not status.stale
    assert status.packets_received == status.packets_accepted == 1
    assert status.last_sequence == 1 and status.current_session_id == "boot-a"


def test_empty_poll_is_nonblocking_and_status_does_not_poll(transport):
    client, sock, _ = transport
    assert not sock.blocking
    assert client.poll() is None
    assert client.latest_state() is None
    assert not client.status().fresh and client.status().age_ms is None
    send(sock)
    assert client.status().packets_received == 0
    assert client.latest_state() is None
    assert len(sock.pending) == 1


def test_duplicate_and_reordered_packets_do_not_refresh_freshness(transport):
    client, sock, clock = transport
    send(sock, sequence=10)
    original = client.poll()
    clock.now = 0.04
    send(sock, sequence=10, position_m=(9, 9, 9))
    send(sock, sequence=9, position_m=(8, 8, 8))
    assert client.poll() is None
    clock.now = 0.051
    status = client.status()
    assert status.stale and not status.fresh
    assert status.age_ms == pytest.approx(51)
    assert status.time_since_last_valid_packet_ms == status.age_ms
    assert status.duplicate_packets == status.reordered_packets == 1
    assert status.packets_accepted == 1 and status.last_sequence == 10
    state = client.latest_state()
    assert not state.valid and state.source_valid
    assert state.timestamp == original.timestamp
    assert state.source_timestamp == original.source_timestamp
    assert state.position_xyz == original.position_xyz
    assert original.valid  # previously returned immutable snapshot is untouched


def test_default_50ms_timeout_and_new_packet_recovery(transport):
    client, sock, clock = transport
    send(sock)
    client.poll()
    clock.now = 0.05
    assert client.status().fresh  # stale iff strictly greater than timeout
    clock.now = 0.050001
    assert client.status().stale and not client.latest_state().valid
    send(sock, sequence=2, source_timestamp=0.1)
    assert client.poll().valid
    assert client.status().fresh and client.status().age_ms == 0
    assert client.latest_state().source_timestamp == 0.1


def test_new_session_allows_restart_and_latches_change_across_drain(transport):
    client, sock, _ = transport
    send(sock, sequence=100)
    client.poll()
    assert client.status().session_changed  # initial attach needs anchor too
    assert client.status().session_changes == 0
    client.acknowledge_session_change()
    send(sock, sequence=0, session_id="boot-b")
    send(sock, sequence=1, session_id="boot-b")
    assert client.poll().sequence == 1
    assert client.status().session_changed
    assert client.status().session_changes == 1
    assert client.status().current_session_id == "boot-b"
    assert client.poll() is None
    assert client.status().session_changed  # reads do not consume the event
    client.acknowledge_session_change()
    assert not client.status().session_changed


def test_newest_sequence_wins_with_reordered_packets_in_same_drain(transport):
    client, sock, _ = transport
    send(sock, sequence=1)
    send(sock, sequence=3, position_m=(3, 2, 1))
    send(sock, sequence=2, position_m=(2, 2, 2))
    send(sock, sequence=3)
    state = client.poll()
    assert state.sequence == 3 and state.position_xyz == (3, 2, 1)
    assert not sock.pending
    assert client.status().packets_accepted == 2
    assert client.status().reordered_packets == 1
    assert client.status().duplicate_packets == 1


def test_malformed_oversized_and_unsupported_datagrams_do_not_crash(transport):
    client, sock, _ = transport
    sock.pending.extend([
        b"garbage", b"\xff", b"x" * (MAX_PACKET_BYTES + 100),
        wire(protocol_version=2), wire(position_m=[float("nan"), 0, 0]),
    ])
    send(sock, sequence=7)
    assert client.poll().sequence == 7
    status = client.status()
    assert status.packets_received == 6
    assert status.invalid_packets == 4
    assert status.unsupported_version_packets == 1
    assert status.packets_accepted == 1


def test_malformed_packet_cannot_change_session_or_freshness(transport):
    client, sock, clock = transport
    send(sock)
    client.poll()
    client.acknowledge_session_change()
    clock.now = 0.06
    sock.pending.append(wire(session_id="other", sequence=0, valid="yes"))
    assert client.poll() is None
    assert client.status().current_session_id == "boot-a"
    assert client.status().stale and not client.status().session_changed


def test_source_invalid_flag_disables_state_without_refreshing_valid_time(transport):
    client, sock, clock = transport
    send(sock)
    client.poll()
    clock.now = 0.04
    send(sock, sequence=2, valid=False, grey_button=False, white_button=True)
    state = client.poll()
    assert not state.valid and not state.source_valid
    assert state.buttons == (False, True)
    assert state.timestamp == 0.04
    assert not client.status().fresh and not client.status().stale
    assert client.status().age_ms == pytest.approx(40)
    clock.now = 0.06
    assert client.status().stale
    send(sock, sequence=3)
    assert client.poll().valid and client.status().fresh


def test_invalid_first_sample_of_new_session_cannot_inherit_previous_freshness(transport):
    client, sock, _ = transport
    send(sock)
    client.poll()
    send(sock, sequence=0, session_id="boot-b", valid=False)
    assert not client.poll().valid
    assert client.status().age_ms is None and not client.status().fresh


def test_clock_domain_change_requires_new_session(transport):
    client, sock, _ = transport
    send(sock)
    client.poll()
    send(sock, sequence=2, source_clock_domain="touch-pc.unix")
    assert client.poll() is None
    assert client.status().invalid_packets == 1 and client.status().last_sequence == 1
    send(sock, sequence=0, session_id="boot-b", source_clock_domain="touch-pc.unix")
    assert client.poll().source_clock_domain == "touch-pc.unix"


def test_configurable_timeout_and_optional_poll_budget(monkeypatch):
    clock, sock = Clock(), DatagramSocket()
    monkeypatch.setattr("geomagic.bridge.udp_state_client.socket.socket", lambda *args: sock)
    with GeomagicUDPStateClient(monotonic=clock, stale_timeout_s=0.1,
                               max_packets_per_poll=2) as client:
        for sequence in range(3):
            send(sock, sequence=sequence)
        assert client.poll().sequence == 1
        assert client.status().poll_budget_hits == 1
        assert client.poll().sequence == 2
        assert not sock.pending
        clock.now = 0.075
        assert client.status().fresh
        clock.now = 0.101
        assert client.status().stale
    assert sock.closed


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True, "0.05"])
def test_invalid_timeouts_rejected(timeout):
    with pytest.raises(ValueError):
        GeomagicUDPStateClient(stale_timeout_s=timeout)


@pytest.mark.parametrize("budget", [0, -1, 1.5, True])
def test_invalid_budgets_rejected(budget):
    with pytest.raises(ValueError):
        GeomagicUDPStateClient(max_packets_per_poll=budget)


def test_close_is_idempotent_and_invalidates_cached_state(transport):
    client, sock, _ = transport
    send(sock)
    client.poll()
    client.close()
    client.close()
    assert sock.closed and not client.status().fresh
    assert not client.latest_state().valid
    with pytest.raises(RuntimeError, match="closed"):
        client.poll()


def test_real_localhost_udp_latest_value_and_empty_nonblocking_poll():
    clock = Clock()
    with GeomagicUDPStateClient(monotonic=clock) as client:
        assert client._socket.gettimeout() == 0.0
        assert client.poll() is None
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            for sequence in range(3):
                sender.sendto(encode_packet(replace(example_packet(), sequence=sequence)),
                              client.address)
            # Readiness wait is in the test harness only, never in the receiver.
            assert select.select([client._socket], [], [], 1.0)[0]
            state = client.poll()
        assert state.sequence == 2
        assert client.status().packets_received == 3
        assert client.poll() is None
        assert client.latest_state().sequence == 2
