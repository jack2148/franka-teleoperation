"""Standard-library-only synthetic UDP sender; also packaged as one .pyz file."""
import argparse
import ipaddress
import json
import math
import socket
import time
import uuid

from geomagic.bridge.protocol import GeomagicStatePacket, PROTOCOL_VERSION, encode_packet
from geomagic.bridge.timing_stats import TimingStats, positive_number


def packet(sequence, elapsed, timestamp, session_id):
    # <= 5 mm on each device axis; <= 0.05 rad about device Y.
    angle = .05 * math.sin(2 * math.pi * .2 * elapsed)
    return GeomagicStatePacket(
        PROTOCOL_VERSION, session_id, sequence, timestamp, "synthetic-sender.monotonic",
        tuple(.005 * math.sin(2 * math.pi * frequency * elapsed)
              for frequency in (.17, .23, .31)),
        (0., math.sin(angle / 2), 0., math.cos(angle / 2)), True, False, True,
    )


def send_stream(send, *, rate_hz=250., duration=60., session_id=None,
                inject_duplicate=False, inject_reorder=False, inject_at=100,
                probe_after_stop=False, clock=time.monotonic, sleep=time.sleep):
    """Injected callable accepts encoded bytes; no socket/model ownership here.

    Emitted original sequences are contiguous. Missed scheduler slots are skipped
    without burst catch-up and reported separately, not disguised as network loss.
    Optional post-stop rejection probes occur 75/100 ms after the final original.
    """
    rate_hz, duration = positive_number(rate_hz), positive_number(duration)
    if rate_hz > 1000 or duration > 3600:
        raise ValueError("experiment limits: rate <= 1000 Hz, duration <= 3600 s")
    if type(inject_at) is not int or inject_at < 1:
        raise ValueError("inject_at must be an integer >= 1")
    if probe_after_stop and not (inject_duplicate or inject_reorder):
        raise ValueError("--probe-after-stop requires an injection flag")
    session_id = session_id or str(uuid.uuid4())
    # Validate the actual wire contract before any traffic is sent.
    encode_packet(packet(0, 0., clock(), session_id))
    start = clock()
    deadline, end, period = start, start + duration, 1. / rate_hz
    slot = 0
    count = duplicates = reorders = skipped = 0
    previous = latest = None
    first_sent = last_sent = None
    lateness = TimingStats()
    interrupted = False

    def wait_until(target):
        while True:
            remaining = target - clock()
            if remaining <= 0:
                return
            sleep(remaining)

    try:
        while deadline < end:
            wait_until(deadline)
            now = clock()
            if now >= end:
                break
            lateness.add(max(0., now - deadline) * 1000.)
            payload = encode_packet(packet(count, now - start, now, session_id))
            send(payload)
            first_sent = now if first_sent is None else first_sent
            last_sent = now
            previous, latest = latest, payload
            if not probe_after_stop and count == inject_at:
                if inject_duplicate:
                    send(latest)
                    duplicates += 1
                if inject_reorder and previous is not None:
                    send(previous)
                    reorders += 1
            count += 1
            next_slot = slot + 1
            after = clock()
            slot = max(next_slot, math.ceil((after - start) * rate_hz))
            skipped += slot - next_slot
            deadline = start + slot * period
        wait_until(end)
        if probe_after_stop and latest is not None:
            # No new valid sequence follows these: rejection cannot refresh age.
            if inject_duplicate:
                wait_until(max(end, last_sent + .075))
                send(latest)
                duplicates += 1
            if inject_reorder and previous is not None:
                wait_until(max(end, last_sent + .100))
                send(previous)
                reorders += 1
    except KeyboardInterrupt:
        interrupted = True
    finish = clock()
    span = None if first_sent is None else last_sent - first_sent
    return dict(
        kind="synthetic_sender", session_id=session_id,
        source_clock_domain="synthetic-sender.monotonic", requested_rate_hz=rate_hz,
        requested_duration_s=duration, actual_duration_s=finish - start,
        original_send_count=count, total_datagrams_sent=count + duplicates + reorders,
        duplicate_datagrams_sent=duplicates, reordered_datagrams_sent=reorders,
        original_send_span_s=span,
        effective_original_rate_hz=(count - 1) / span if span and count > 1 else None,
        scheduler_skipped_slots=skipped, scheduler_lateness_ms=lateness.report(),
        probe_after_stop=probe_after_stop, interrupted=interrupted,
        note="send success means local UDP submission, not remote delivery; clocks are not compared",
    )


def ipv4_host(text):
    address = ipaddress.IPv4Address(text)
    if address.is_unspecified or address.is_multicast or text == "255.255.255.255":
        raise ValueError("use one explicit unicast IPv4 address")
    return str(address)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, type=ipv4_host)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--rate-hz", type=positive_number, default=250.)
    parser.add_argument("--duration", type=positive_number, default=60.)
    parser.add_argument("--session-id", default=None, help="unique per run; default UUID")
    parser.add_argument("--inject-duplicate", action="store_true")
    parser.add_argument("--inject-reorder", action="store_true")
    parser.add_argument("--inject-at", type=int, default=100)
    parser.add_argument("--probe-after-stop", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            report = send_stream(
                lambda payload: sock.sendto(payload, (args.host, args.port)),
                rate_hz=args.rate_hz, duration=args.duration, session_id=args.session_id,
                inject_duplicate=args.inject_duplicate, inject_reorder=args.inject_reorder,
                inject_at=args.inject_at, probe_after_stop=args.probe_after_stop)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Sender stopped: {error}\nNo network configuration was changed.\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 130 if report["interrupted"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
