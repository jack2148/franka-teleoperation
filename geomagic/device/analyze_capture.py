#!/usr/bin/env python3
"""Offline analysis of a csv_capture.py recording (RAW Geomagic device frame).

Every contiguous grey-held run is a translation interval and every white-held
run a rotation interval; all intervals are numbered and none are dropped.
Physical labels are only assigned when given explicitly:

    --translation-labels right,left,forward,backward,up,down
    [--label-intervals 1,2,3,4,5,6]   # which grey intervals the labels name

Without --label-intervals, labels are applied in order only when their count
equals the grey interval count; otherwise no comparison is made.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import sys

AXES = "XYZ"
# Validated RAW convention (docs/RAW_DEVICE_FRAME_OBSERVATION.md).
EXPECTED = {
    "right": (0, +1), "left": (0, -1), "forward": (2, +1),
    "backward": (2, -1), "up": (1, +1), "down": (1, -1),
}


def load(path):
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for record in csv.DictReader(stream):
            rows.append({
                "t": float(record["receive_time_monotonic"]),
                "src": float(record["source_timestamp"]),
                "seq": int(record["sequence"]),
                "p": tuple(float(record[f"position_{a}_m"]) for a in "xyz"),
                "q": tuple(float(record[f"quat_{c}"]) for c in "xyzw"),
                "grey": record["grey_button"] == "1",
                "white": record["white_button"] == "1",
                "valid": record["valid"] == "1",
                "session": record.get("session_id", ""),
            })
    return rows


def angle_deg(a, b):
    """Rotation angle between two quaternions (sign-invariant, normalized)."""
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    dot = abs(sum(x * y for x, y in zip(a, b))) / (na * nb)
    return math.degrees(2.0 * math.acos(min(1.0, dot)))


def percentile(sorted_values, fraction):
    if not sorted_values:
        return None
    position = fraction * (len(sorted_values) - 1)
    low, high = math.floor(position), math.ceil(position)
    weight = position - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


def intervals(rows, key):
    """[(start, end, closed)] index ranges where rows[i][key] is held."""
    result, start = [], None
    for i, row in enumerate(rows):
        if row[key] and start is None:
            start = i
        elif not row[key] and start is not None:
            result.append((start, i - 1, True))
            start = None
    if start is not None:
        result.append((start, len(rows) - 1, False))  # still held at end of file
    return result


def edges(rows, key):
    press = sum(1 for a, b in zip(rows, rows[1:]) if not a[key] and b[key])
    release = sum(1 for a, b in zip(rows, rows[1:]) if a[key] and not b[key])
    return press, release


def translation(rows, number, start, end, closed):
    p0, p1 = rows[start]["p"], rows[end]["p"]
    delta = [(b - a) * 1000.0 for a, b in zip(p0, p1)]
    norm = math.sqrt(sum(d * d for d in delta))
    axis = max(range(3), key=lambda k: abs(delta[k]))
    return {
        "interval": number, "closed": closed,
        "duration_s": rows[end]["t"] - rows[start]["t"],
        "start_mm": [v * 1000.0 for v in p0], "end_mm": [v * 1000.0 for v in p1],
        "delta_mm": delta, "dominant_axis": AXES[axis],
        "sign": "+" if delta[axis] >= 0 else "-",
        "displacement_mm": norm,
        "dominance": abs(delta[axis]) / norm if norm > 0 else None,
    }


def rotation(rows, number, start, end, closed):
    segment = rows[start:end + 1]
    q0 = segment[0]["q"]
    norms = [math.sqrt(sum(c * c for c in r["q"])) for r in segment]
    steps = [angle_deg(a["q"], b["q"]) for a, b in zip(segment, segment[1:])]
    finite = all(math.isfinite(c) for r in segment for c in r["q"])
    return {
        "interval": number, "closed": closed,
        "duration_s": segment[-1]["t"] - segment[0]["t"],
        "net_angle_deg": angle_deg(q0, segment[-1]["q"]),
        "peak_angle_deg": max(angle_deg(q0, r["q"]) for r in segment),
        "norm_min": min(norms), "norm_max": max(norms),
        "max_step_deg": max(steps) if steps else 0.0,
        "finite": finite,
    }


def sequence_report(rows):
    gaps, missing, max_gap, non_increasing, sessions = 0, 0, 0, 0, set()
    for a, b in zip(rows, rows[1:]):
        sessions.add(a["session"])
        if a["session"] != b["session"]:
            continue  # sequence restarts with a new streamer session
        step = b["seq"] - a["seq"]
        if step <= 0:
            non_increasing += 1
        elif step > 1:
            gaps += 1
            missing += step - 1
            max_gap = max(max_gap, step - 1)
    if rows:
        sessions.add(rows[-1]["session"])
    return {"gaps": gaps, "missing_packets": missing, "max_missing_run": max_gap,
            "non_increasing": non_increasing, "sessions": len(sessions)}


def timing_report(rows):
    def stats(values):
        ordered = sorted(values)
        return {"median_ms": percentile(ordered, .5), "p95_ms": percentile(ordered, .95),
                "p99_ms": percentile(ordered, .99), "max_ms": ordered[-1] if ordered else None}
    receive = [(b["t"] - a["t"]) * 1000.0 for a, b in zip(rows, rows[1:])]
    source = [(b["src"] - a["src"]) * 1000.0 for a, b in zip(rows, rows[1:])
              if a["session"] == b["session"]]
    span = rows[-1]["t"] - rows[0]["t"] if len(rows) > 1 else 0.0
    return {"samples": len(rows), "duration_s": span,
            "rate_hz": (len(rows) - 1) / span if span > 0 else None,
            "invalid_rows": sum(not r["valid"] for r in rows),
            "receive_interarrival": stats(receive), "source_interval": stats(source),
            "receive_gaps_over_50ms": sum(1 for v in receive if v > 50.0)}


def label_comparison(translations, labels, label_intervals, min_displacement_mm):
    if label_intervals is None:
        if len(labels) != len(translations):
            return {"performed": False,
                    "reason": f"{len(labels)} labels but {len(translations)} grey intervals; "
                              "pass --label-intervals to choose intervals explicitly"}
        label_intervals = [t["interval"] for t in translations]
    if len(label_intervals) != len(labels):
        return {"performed": False, "reason": "--label-intervals and labels differ in length"}
    by_number = {t["interval"]: t for t in translations}
    results = []
    for label, number in zip(labels, label_intervals):
        if label not in EXPECTED:
            raise SystemExit(f"unknown label {label!r}; use {', '.join(EXPECTED)}")
        if number not in by_number:
            raise SystemExit(f"grey interval {number} does not exist")
        t = by_number[number]
        axis, sign = EXPECTED[label]
        component = t["delta_mm"][axis]
        ok = (t["dominant_axis"] == AXES[axis] and (component >= 0) == (sign > 0)
              and abs(component) >= min_displacement_mm)
        results.append({"label": label, "interval": number,
                        "expected": f"{'+' if sign > 0 else '-'}{AXES[axis]}",
                        "observed": f"{t['sign']}{t['dominant_axis']}",
                        "signed_displacement_mm": component,
                        "result": "PASS" if ok else "FAIL"})
    return {"performed": True, "results": results,
            "passed": sum(r["result"] == "PASS" for r in results), "total": len(results)}


def fmt(values, digits=1):
    return "[" + ", ".join(f"{v:+.{digits}f}" for v in values) + "]"


def print_report(path, report):
    t = report["timing"]
    print(f"capture: {path}")
    print(f"samples={t['samples']} duration={t['duration_s']:.2f}s rate={t['rate_hz'] or 0:.1f}Hz "
          f"invalid_rows={t['invalid_rows']} gaps>50ms={t['receive_gaps_over_50ms']}")
    ri, si = t["receive_interarrival"], t["source_interval"]
    print(f"receive inter-arrival ms: median={ri['median_ms']:.3f} p95={ri['p95_ms']:.3f} "
          f"p99={ri['p99_ms']:.3f} max={ri['max_ms']:.3f}")
    if si["median_ms"] is not None:
        print(f"source sample interval ms: median={si['median_ms']:.3f} p95={si['p95_ms']:.3f} "
              f"p99={si['p99_ms']:.3f} max={si['max_ms']:.3f}")
    s = report["sequence"]
    print(f"sequence: gaps={s['gaps']} missing={s['missing_packets']} "
          f"max_missing_run={s['max_missing_run']} non_increasing={s['non_increasing']} "
          f"sessions={s['sessions']}")
    b = report["buttons"]
    print(f"buttons: grey press={b['grey_press']} release={b['grey_release']} "
          f"intervals={b['grey_intervals']} | white press={b['white_press']} "
          f"release={b['white_release']} intervals={b['white_intervals']} | "
          f"both_held_samples={b['both_held_samples']} | "
          f"held_at_start grey={b['grey_held_at_start']} white={b['white_held_at_start']}")

    print("\nGREY intervals (translation, RAW mm)")
    print(" #  dur_s  start_mm                  end_mm                    delta_mm"
          "                  axis  |disp|  dominance")
    for r in report["translations"]:
        flag = "" if r["closed"] else "  (open at end)"
        dominance = f"{r['dominance']:.3f}" if r["dominance"] is not None else "  -  "
        print(f"{r['interval']:>2} {r['duration_s']:6.2f}  {fmt(r['start_mm']):24}  "
              f"{fmt(r['end_mm']):24}  {fmt(r['delta_mm']):24}  {r['sign']}{r['dominant_axis']}"
              f"   {r['displacement_mm']:6.1f}  {dominance}{flag}")

    print("\nWHITE intervals (rotation)")
    print(" #  dur_s  net_deg  peak_deg  norm_min     norm_max     max_step_deg  finite")
    for r in report["rotations"]:
        flag = "" if r["closed"] else "  (open at end)"
        print(f"{r['interval']:>2} {r['duration_s']:6.2f}  {r['net_angle_deg']:7.2f}  "
              f"{r['peak_angle_deg']:8.2f}  {r['norm_min']:.9f}  {r['norm_max']:.9f}  "
              f"{r['max_step_deg']:12.4f}  {r['finite']}{flag}")

    labels = report.get("labels")
    if labels is not None:
        print("\nLABEL comparison (expected RAW convention)")
        if not labels["performed"]:
            print(f"  not performed: {labels['reason']}")
        else:
            for r in labels["results"]:
                print(f"  {r['label']:>8} -> interval {r['interval']:>2}: expected {r['expected']} "
                      f"observed {r['observed']} signed {r['signed_displacement_mm']:+.1f} mm  "
                      f"{r['result']}")
            print(f"  {labels['passed']}/{labels['total']} PASS")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--translation-labels", default=None,
                        help="comma list, e.g. right,left,forward,backward,up,down")
    parser.add_argument("--label-intervals", default=None,
                        help="comma list of grey interval numbers the labels refer to")
    parser.add_argument("--min-displacement-mm", type=float, default=10.0,
                        help="minimum |signed displacement| on the expected axis for PASS")
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = parser.parse_args(argv)

    rows = load(args.csv)
    if len(rows) < 2:
        raise SystemExit(f"{args.csv}: fewer than 2 samples")
    grey, white = intervals(rows, "grey"), intervals(rows, "white")
    translations = [translation(rows, n + 1, *span) for n, span in enumerate(grey)]
    rotations = [rotation(rows, n + 1, *span) for n, span in enumerate(white)]
    grey_press, grey_release = edges(rows, "grey")
    white_press, white_release = edges(rows, "white")
    report = {
        "timing": timing_report(rows), "sequence": sequence_report(rows),
        "buttons": {"grey_press": grey_press, "grey_release": grey_release,
                    "grey_intervals": len(grey), "white_press": white_press,
                    "white_release": white_release, "white_intervals": len(white),
                    "both_held_samples": sum(r["grey"] and r["white"] for r in rows),
                    "grey_held_at_start": rows[0]["grey"], "white_held_at_start": rows[0]["white"]},
        "translations": translations, "rotations": rotations,
    }
    if args.translation_labels:
        labels = [x.strip().lower() for x in args.translation_labels.split(",") if x.strip()]
        chosen = ([int(x) for x in args.label_intervals.split(",")]
                  if args.label_intervals else None)
        report["labels"] = label_comparison(translations, labels, chosen,
                                            args.min_displacement_mm)
    if args.json:
        json.dump(report, sys.stdout, indent=2)
        print()
    else:
        print_report(args.csv, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
