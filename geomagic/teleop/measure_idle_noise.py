"""Offline physical CSV analysis; never opens a device or socket.

Select a known stationary interval explicitly with --start-s/--duration-s.
Without --stationary-confirmed this is a candidate measurement, not a final
tuning recommendation. Source timestamps, not network-arrival jitter, set dt.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from geomagic.vendor.geomagic_core.rotations import quaternion_xyzw_to_matrix


def rotation_angles(a, b):
    relative = np.matmul(a, np.swapaxes(b, -1, -2))
    vee = np.stack((relative[..., 2, 1] - relative[..., 1, 2],
                    relative[..., 0, 2] - relative[..., 2, 0],
                    relative[..., 1, 0] - relative[..., 0, 1]), axis=-1)
    sine = .5 * np.linalg.norm(vee, axis=-1)
    cosine = .5 * (np.trace(relative, axis1=-2, axis2=-1) - 1.)
    return np.arctan2(sine, np.clip(cosine, -1., 1.))


def percentiles(values):
    values = np.asarray(values)
    if not len(values):
        return None
    return dict(zip(("median", "p95", "p99", "p99_9", "max"),
                    map(float, np.percentile(values, (50, 95, 99, 99.9, 100)))))


def measure(path, *, start_s=0., duration_s=None, stationary_confirmed=False):
    path = Path(path)
    rows = list(csv.DictReader(path.open()))
    if not rows:
        raise ValueError("CSV is empty")
    source_start = float(rows[0]["source_timestamp"])
    rows = [r for r in rows if float(r["source_timestamp"]) - source_start >= start_s
            and (duration_s is None
                 or float(r["source_timestamp"]) - source_start <= start_s + duration_s)]
    if len(rows) < 3:
        raise ValueError("selected interval requires at least three samples")
    if len({r["session_id"] for r in rows}) != 1:
        raise ValueError("select one uninterrupted session")
    if any(r["valid"] != "1" for r in rows):
        raise ValueError("selected interval contains invalid samples")
    t = np.array([float(r["source_timestamp"]) for r in rows])
    p = np.array([[float(r["position_" + k + "_m"]) for k in "xyz"] for r in rows])
    q = np.array([[float(r["quat_" + k]) for k in "xyzw"] for r in rows])
    if not all(np.isfinite(x).all() for x in (t, p, q)):
        raise ValueError("selected input must be finite")
    dt = np.diff(t)
    if np.any(dt <= 0) or np.max(dt) > .050:
        raise ValueError("selected source timestamps must progress with no stale gaps")
    rotation = np.array([quaternion_xyzw_to_matrix(x) for x in q])
    v = np.linalg.norm(np.diff(p, axis=0), axis=1) / dt
    omega = rotation_angles(rotation[1:], rotation[:-1]) / dt
    windows = {}
    for seconds in (.010, .020, .050, .100, .500):
        indices = np.searchsorted(t, t + seconds)
        i = np.flatnonzero(indices < len(t))
        j = indices[i]
        windows[str(round(seconds * 1000))] = {
            "pairs": int(len(i)),
            "actual_lag_s": percentiles(t[j] - t[i]),
            "translation_displacement_mm": percentiles(1000 * np.linalg.norm(p[j] - p[i], axis=1)),
            "rotation_displacement_deg": percentiles(np.rad2deg(rotation_angles(rotation[j], rotation[i]))),
        }
    return {
        "source": str(path.resolve()), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "selection_start_s": start_s, "samples": len(t), "duration_s": float(t[-1] - t[0]),
        "stationary_confirmed": stationary_confirmed,
        "tuning_status": "MEASUREMENT_ONLY" if stationary_confirmed else "TUNING_PENDING",
        "limitations": ("Stationarity is an operator declaration, not inferred from buttons. "
                        "Holder measurements do not characterize a free-hand resting grip. "
                        "Zero observed translation jitter cannot establish a translation threshold."),
        "translation_velocity_mm_s": percentiles(1000 * v),
        "rotation_velocity_deg_s": percentiles(np.rad2deg(omega)),
        "source_dt_s": percentiles(dt),
        "position_range_mm": (1000 * np.ptp(p, axis=0)).tolist(),
        "orientation_from_start_deg": percentiles(np.rad2deg(rotation_angles(rotation, rotation[0]))),
        "windows_ms": windows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--start-s", type=float, default=0.)
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--stationary-confirmed", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.start_s < 0 or (args.duration_s is not None and args.duration_s <= 0):
        parser.error("start must be nonnegative and duration positive")
    result = measure(args.csv, start_s=args.start_s, duration_s=args.duration_s,
                     stationary_confirmed=args.stationary_confirmed)
    value = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(value)
    print(value, end="")


if __name__ == "__main__":
    main()
