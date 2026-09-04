#!/usr/bin/env python3
"""Estimate circular coverage and scan low-error contiguous windows."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def nearest(truth, t):
    key = min(truth, key=lambda x: abs(x - t))
    return truth[key] if abs(key - t) <= 2.0 else None


def circle_fit(xy):
    x, y = xy[:, 0], xy[:, 1]
    A = np.column_stack((2 * x, 2 * y, np.ones_like(x)))
    b = x * x + y * y
    cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    radius = math.sqrt(max(c + cx * cx + cy * cy, 0.0))
    return np.asarray([cx, cy]), radius


def angular_path(xy, center):
    angle = np.unwrap(np.arctan2(xy[:, 1] - center[1], xy[:, 0] - center[0]))
    return angle


def window_metrics(rows, truth, length):
    candidates = []
    for start in range(len(rows) - length + 1):
        window = rows[start:start + length]
        errors = []
        valid = 0
        for row in window:
            if row.get("valid", "False").lower() != "true":
                continue
            target = nearest(truth, float(row["time_s"]))
            if target is None:
                continue
            est = np.asarray([float(row[f"y_{k}"]) for k in ("E", "N", "U")])
            errors.append(float(np.linalg.norm(est - target)))
            valid += 1
        if errors:
            candidates.append({
                "start_index": start,
                "stop_index_exclusive": start + length,
                "start_time_s": float(window[0]["time_s"]),
                "end_time_s": float(window[-1]["time_s"]),
                "frames": len(window),
                "valid_frames": valid,
                "valid_fraction": valid / len(window),
                "rmse_m": math.sqrt(sum(e * e for e in errors) / len(errors)),
                "median_error_m": statistics.median(errors),
                "p90_error_m": sorted(errors)[min(len(errors) - 1, int(0.90 * len(errors)))],
            })
    return sorted(candidates, key=lambda x: (x["rmse_m"], -x["valid_fraction"], x["start_index"]))


def hms(t):
    t = int(round(t)) % 86400
    return f"{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--gps", type=Path, required=True)
    parser.add_argument("--segment", default="danyuan_panxuan_3")
    parser.add_argument("--lengths", default="10,20,30,40,60,90,120,150,180,210,240")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    observation_rows = [row for row in rows(args.observations) if row.get("segment") == args.segment]
    observation_rows.sort(key=lambda row: float(row["time_s"]))
    gps_rows = rows(args.gps)
    truth = {float(row["time_s"]): np.asarray([float(row[k]) for k in ("px", "py", "pz")]) for row in gps_rows}
    # Fit the circle on the GPS interval actually covered by the evaluation
    # observations. GPS remains an offline geometric audit, never a runtime input.
    if observation_rows:
        obs_start = float(observation_rows[0]["time_s"])
        obs_end = float(observation_rows[-1]["time_s"])
        truth_items = [(t, value) for t, value in sorted(truth.items()) if obs_start <= t <= obs_end]
    else:
        truth_items = sorted(truth.items())
    if len(truth_items) < 10:
        truth_items = sorted(truth.items())
    truth_fit = np.asarray([value for _, value in truth_items])
    truth_xy = truth_fit[:, :2]
    center, radius = circle_fit(truth_xy)
    angle = angular_path(truth_xy, center)
    sweep_deg = float(np.degrees(angle[-1] - angle[0]))
    fit_duration_s = float(truth_items[-1][0] - truth_items[0][0])
    arc_period_s = float(fit_duration_s * 360.0 / max(abs(sweep_deg), 1e-9))
    lengths = [int(value) for value in args.lengths.split(",")]
    scan = {str(length): window_metrics(observation_rows, truth, length)[:10] for length in lengths if length <= len(observation_rows)}
    payload = {
        "segment": args.segment,
        "gps_time_start": hms(truth_items[0][0]), "gps_time_end": hms(truth_items[-1][0]),
        "gps_time_start_s": float(truth_items[0][0]), "gps_time_end_s": float(truth_items[-1][0]),
        "gps_frames": len(truth_items), "circle_fit_center_EN_m": center.tolist(), "circle_fit_radius_m": radius,
        "unwrapped_sweep_deg": sweep_deg, "estimated_full_circle_period_s": arc_period_s,
        "near_full_circle_threshold_deg": 300.0,
        "minimum_observed_duration_for_300deg_s": float(fit_duration_s * 300.0 / max(abs(sweep_deg), 1e-9) if abs(sweep_deg) >= 300.0 else float("nan")),
        "window_selection_uses_gps": "offline error scan only; not a method selection input",
        "lowest_error_windows": scan,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
