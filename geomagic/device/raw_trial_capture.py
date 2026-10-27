#!/usr/bin/env python3
"""Operator-guided RAW device-frame check over the existing Phase-1 receiver.

Receives the streamer's localhost packets with the unchanged
GeomagicUDPStateClient and segments physical trials by button holds:

- each GREY-held interval is one translation trial, expected in this order:
  right, left, forward, backward, up, down;
- each WHITE-held interval is one small-rotation trial.

Buttons are only used to mark trial boundaries here. Nothing is mapped to the
FR3 and nothing is written to disk; one JSON summary goes to stdout.
"""

import argparse
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from geomagic.bridge.udp_state_client import GeomagicUDPStateClient  # noqa: E402

EXPECTED = (  # validated RAW convention (docs/RAW_DEVICE_FRAME_OBSERVATION.md)
    ("right", 0, +1), ("left", 0, -1), ("forward", 2, +1),
    ("backward", 2, -1), ("up", 1, +1), ("down", 1, -1),
)
AXES = "XYZ"


def quat_angle_deg(a, b):
    """Rotation angle between two unit xyzw quaternions."""
    dot = abs(sum(x * y for x, y in zip(a, b)))
    return math.degrees(2.0 * math.acos(min(1.0, dot)))


def segments(samples, index):
    """(start, end) sample indices of intervals where buttons[index] is held."""
    result, start = [], None
    for i, sample in enumerate(samples):
        held = sample[3][index]
        if held and start is None:
            start = i
        elif not held and start is not None:
            result.append((start, i - 1))
            start = None
    return result


def edges(samples, index):
    rising = falling = 0
    for previous, current in zip(samples, samples[1:]):
        rising += (not previous[3][index]) and current[3][index]
        falling += previous[3][index] and (not current[3][index])
    return rising, falling


def analyse(samples, min_displacement_mm):
    translations = []
    for n, (start, end) in enumerate(segments(samples, 0)):
        p0, p1 = samples[start][1], samples[end][1]
        delta_mm = [(b - a) * 1000.0 for a, b in zip(p0, p1)]
        norm = math.sqrt(sum(d * d for d in delta_mm))
        axis = max(range(3), key=lambda k: abs(delta_mm[k]))
        sign = 1 if delta_mm[axis] >= 0 else -1
        row = {
            "trial": n + 1,
            "duration_s": round(samples[end][0] - samples[start][0], 3),
            "delta_mm": [round(d, 2) for d in delta_mm],
            "dominant_axis": AXES[axis], "sign": "+" if sign > 0 else "-",
            "displacement_mm": round(norm, 2),
            "dominance": round(abs(delta_mm[axis]) / norm, 3) if norm else None,
        }
        if n < len(EXPECTED):
            label, expected_axis, expected_sign = EXPECTED[n]
            row.update(motion=label, expected=f"{'+' if expected_sign > 0 else '-'}{AXES[expected_axis]}")
            row["result"] = ("PASS" if axis == expected_axis and sign == expected_sign
                             and abs(delta_mm[axis]) >= min_displacement_mm else "FAIL")
        else:
            row.update(motion="extra", expected=None, result="IGNORED")
        translations.append(row)

    rotations = []
    for n, (start, end) in enumerate(segments(samples, 1)):
        q_start = samples[start][2]
        peak = max(quat_angle_deg(q_start, samples[i][2]) for i in range(start, end + 1))
        moved_mm = math.dist(samples[start][1], samples[end][1]) * 1000.0
        rotations.append({"trial": n + 1, "duration_s": round(samples[end][0] - samples[start][0], 3),
                          "net_angle_deg": round(quat_angle_deg(q_start, samples[end][2]), 2),
                          "peak_angle_deg": round(peak, 2), "tip_translation_mm": round(moved_mm, 1)})

    norms = [math.sqrt(sum(c * c for c in s[2])) for s in samples]
    steps = [quat_angle_deg(a[2], b[2]) for a, b in zip(samples, samples[1:])]
    q0 = samples[0][2]
    position_mm = [[s[1][k] * 1000.0 for s in samples] for k in range(3)]
    grey_r, grey_f = edges(samples, 0)
    white_r, white_f = edges(samples, 1)
    both = sum(1 for s in samples if s[3][0] and s[3][1])
    return {
        "translations": translations,
        "translation_pass": sum(r["result"] == "PASS" for r in translations[:6]),
        "rotations": rotations,
        "orientation": {
            "all_finite": all(math.isfinite(c) for s in samples for c in s[2]),
            "norm_min": round(min(norms), 9), "norm_max": round(max(norms), 9),
            "max_step_between_samples_deg": round(max(steps), 4) if steps else None,
            "p99_step_deg": round(sorted(steps)[int(0.99 * (len(steps) - 1))], 4) if steps else None,
            "max_angle_from_first_deg": round(max(quat_angle_deg(q0, s[2]) for s in samples), 2),
        },
        "position_range_mm": {AXES[k]: [round(min(v), 1), round(max(v), 1)]
                              for k, v in enumerate(position_mm)},
        "buttons": {"grey_rising": grey_r, "grey_falling": grey_f,
                    "white_rising": white_r, "white_falling": white_f,
                    "samples_both_pressed": both},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5558)
    parser.add_argument("--max-duration", type=float, default=240.0)
    parser.add_argument("--rotation-trials", type=int, default=3)
    parser.add_argument("--min-displacement-mm", type=float, default=20.0)
    parser.add_argument("--settle", type=float, default=2.0,
                        help="seconds after the last required trial before stopping")
    args = parser.parse_args()

    client = GeomagicUDPStateClient(("127.0.0.1", args.port), max_packets_per_poll=1)
    samples = []  # (receive_time, position_m, orientation_xyzw, (grey, white))
    started = time.monotonic()
    done_at = None
    try:
        while time.monotonic() - started < args.max_duration:
            state = client.poll()
            if state is None:
                time.sleep(0.0005)
                continue
            samples.append((state.timestamp, state.position_xyz, state.orientation_xyzw,
                            (state.grey_button, state.white_button)))
            if done_at is None and len(samples) % 250 == 0:
                grey = len(segments(samples, 0))
                white = len(segments(samples, 1))
                if grey >= len(EXPECTED) and white >= args.rotation_trials \
                        and not any(samples[-1][3]):
                    done_at = time.monotonic()
            if done_at is not None and time.monotonic() - done_at >= args.settle:
                break
    except KeyboardInterrupt:
        pass
    status = client.status()
    client.close()
    if len(samples) < 2:
        print(json.dumps({"error": "too few samples", "received": len(samples)}))
        return 2
    result = analyse(samples, args.min_displacement_mm)
    span = samples[-1][0] - samples[0][0]
    result["receiver"] = {
        "accepted_samples": len(samples), "span_s": round(span, 3),
        "accepted_hz": round((len(samples) - 1) / span, 2) if span > 0 else None,
        "packets_received": status.packets_received, "packets_accepted": status.packets_accepted,
        "invalid": status.invalid_packets, "duplicate": status.duplicate_packets,
        "reordered": status.reordered_packets, "session_changes": status.session_changes,
        "session_id": status.current_session_id,
    }
    result["completed_protocol"] = done_at is not None
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
