#!/usr/bin/env python3
"""Record the localhost Geomagic state stream to CSV until Ctrl+C.

Uses the unchanged Phase-1 GeomagicUDPStateClient (one datagram per poll, so
every accepted packet becomes one row). No auto-stop, no prompts, no analysis;
run analyze_capture.py on the CSV afterwards.

The protocol carries no HDAPI sample counter. ``source_timestamp`` is the
HDAPI callback time of the published sample (unique per device sample) and is
the source-sample identifier; ``sequence`` is the per-packet counter.
"""

import argparse
import csv
import os
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from geomagic.bridge.udp_state_client import GeomagicUDPStateClient  # noqa: E402

COLUMNS = (
    "receive_time_monotonic", "source_timestamp", "sequence",
    "position_x_m", "position_y_m", "position_z_m",
    "quat_x", "quat_y", "quat_z", "quat_w",
    "grey_button", "white_button", "valid", "session_id",
)
DEFAULT_DIR = Path(__file__).resolve().parent / "captures"


def _terminate(_signum, _frame):
    raise KeyboardInterrupt


def default_output():
    return DEFAULT_DIR / time.strftime("geomagic_raw_%Y%m%d_%H%M%S.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5558)
    parser.add_argument("--output", type=Path, default=None,
                        help="CSV path (default: geomagic/device/captures/geomagic_raw_<time>.csv)")
    parser.add_argument("--flush-interval", type=float, default=0.5)
    args = parser.parse_args()

    output = (args.output or default_output()).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # SIGTERM takes the same clean path as Ctrl+C.
    signal.signal(signal.SIGTERM, _terminate)

    client = GeomagicUDPStateClient(("127.0.0.1", args.port), max_packets_per_poll=1)
    rows = 0
    first = last = None
    with output.open("x", newline="", encoding="utf-8") as stream:  # never overwrite
        writer = csv.writer(stream)
        writer.writerow(COLUMNS)
        print(f"output_csv={output}", flush=True)
        next_flush = time.monotonic() + args.flush_interval
        try:
            while True:
                state = client.poll()
                if state is None:
                    time.sleep(0.0005)
                else:
                    p, q = state.position_xyz, state.orientation_xyzw
                    writer.writerow((
                        repr(state.timestamp), repr(state.source_timestamp), state.sequence,
                        repr(p[0]), repr(p[1]), repr(p[2]),
                        repr(q[0]), repr(q[1]), repr(q[2]), repr(q[3]),
                        int(state.grey_button), int(state.white_button),
                        int(state.source_valid), state.session_id,
                    ))
                    rows += 1
                    first = state.timestamp if first is None else first
                    last = state.timestamp
                now = time.monotonic()
                if now >= next_flush:
                    stream.flush()
                    next_flush = now + args.flush_interval
        except KeyboardInterrupt:
            pass
        finally:
            # A second Ctrl+C must not interrupt the final flush.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            stream.flush()
            os.fsync(stream.fileno())
            status = client.status()
            client.close()

    duration = (last - first) if rows > 1 else 0.0
    rate = (rows - 1) / duration if duration > 0 else 0.0
    print(f"output_csv={output}")
    print(f"samples_recorded={rows}")
    print(f"duration_s={duration:.3f}")
    print(f"accepted_rate_hz={rate:.2f}")
    print(f"invalid={status.invalid_packets} duplicate={status.duplicate_packets} "
          f"reordered={status.reordered_packets}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
