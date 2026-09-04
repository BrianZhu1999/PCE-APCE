#!/usr/bin/env python3
"""Select single-source windows from acoustic quality, then score GPS offline.

The selection objective never reads GPS.  GPS is loaded only after candidate
windows are ranked, so the output distinguishes an acoustic-quality selection
from a truth-selected inspection window.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def read_rows(path: Path, segment: str) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return [row for row in csv.DictReader(stream) if row.get("segment") == segment]


def truth_rows(path: Path) -> dict[float, tuple[float, float, float]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return {float(row["time_s"]): tuple(float(row[k]) for k in ("px", "py", "pz")) for row in csv.DictReader(stream)}


def nearest_truth(truth: dict[float, tuple[float, float, float]], time_s: float):
    if not truth:
        return None
    key = min(truth, key=lambda value: abs(value - time_s))
    return truth[key] if abs(key - time_s) <= 2.0 else None


def acoustic_score(rows: list[dict[str, str]]) -> tuple[float, dict]:
    valid = [row for row in rows if row.get("valid", "False").lower() == "true"]
    if not rows:
        return float("inf"), {"frames": 0}
    valid_fraction = len(valid) / len(rows)
    rms = [float(row["reprojection_rms_deg"]) for row in valid]
    inliers = [float(row["inlier_nodes"]) for row in valid]
    jumps = []
    for left, right in zip(valid, valid[1:]):
        if float(right["time_s"]) - float(left["time_s"]) <= 1.1:
            p0 = [float(left[f"y_{k}"]) for k in ("E", "N", "U")]
            p1 = [float(right[f"y_{k}"]) for k in ("E", "N", "U")]
            jumps.append(math.sqrt(sum((a - b) ** 2 for a, b in zip(p0, p1))))
    # Lower is better; invalid frames and large acoustic jumps are penalized.
    score = (1.0 - valid_fraction) * 100.0 + statistics.median(rms or [99.0])
    score += max(0.0, 6.0 - statistics.median(inliers or [0.0])) * 2.0
    score += max(0.0, statistics.median(jumps or [0.0]) - 80.0) * 0.02
    return score, {
        "frames": len(rows), "valid_frames": len(valid), "valid_fraction": valid_fraction,
        "median_reprojection_rms_deg": statistics.median(rms) if rms else None,
        "p90_reprojection_rms_deg": sorted(rms)[min(len(rms) - 1, int(0.90 * len(rms)))] if rms else None,
        "median_inlier_nodes": statistics.median(inliers) if inliers else None,
        "median_consecutive_jump_m": statistics.median(jumps) if jumps else None,
        "max_consecutive_jump_m": max(jumps) if jumps else None,
    }


def gps_metrics(rows: list[dict[str, str]], truth: dict[float, tuple[float, float, float]]) -> dict:
    errors = []
    for row in rows:
        if row.get("valid", "False").lower() != "true":
            continue
        target = nearest_truth(truth, float(row["time_s"]))
        if target is None:
            continue
        estimate = [float(row[f"y_{k}"]) for k in ("E", "N", "U")]
        errors.append(math.sqrt(sum((a - b) ** 2 for a, b in zip(estimate, target))))
    return {
        "gps_scored_frames": len(errors),
        "gps_rmse_m": math.sqrt(sum(x * x for x in errors) / len(errors)) if errors else None,
        "gps_median_error_m": statistics.median(errors) if errors else None,
        "gps_p90_error_m": sorted(errors)[min(len(errors) - 1, int(0.90 * len(errors)))] if errors else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--gps", type=Path, required=True)
    parser.add_argument("--segment", default="danyuan_panxuan_3")
    parser.add_argument("--lengths", default="5,10,20")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = read_rows(args.observations, args.segment)
    rows.sort(key=lambda row: float(row["time_s"]))
    truth = truth_rows(args.gps)
    candidates = []
    for length in (int(value) for value in args.lengths.split(",")):
        if len(rows) < length:
            continue
        for start in range(len(rows) - length + 1):
            window = rows[start:start + length]
            score, quality = acoustic_score(window)
            candidates.append({
                "window_seconds_nominal": length,
                "start_index": start,
                "stop_index_exclusive": start + length,
                "start_time_s": float(window[0]["time_s"]),
                "end_time_s": float(window[-1]["time_s"]),
                "acoustic_selection_score": score,
                "acoustic_quality": quality,
                "gps_offline_audit": gps_metrics(window, truth),
            })
    candidates.sort(key=lambda item: (item["window_seconds_nominal"], item["acoustic_selection_score"], item["start_index"]))
    best = {}
    for length in sorted({item["window_seconds_nominal"] for item in candidates}):
        best[length] = next(item for item in candidates if item["window_seconds_nominal"] == length)
    payload = {
        "claim_status": "acoustic_quality_window_audit",
        "selection_uses_gps": False,
        "gps_role": "offline scoring only after acoustic ranking",
        "source_observations": str(args.observations),
        "source_gps": str(args.gps),
        "segment": args.segment,
        "best_by_length": best,
        "top_candidates_by_length": {
            str(length): [item for item in candidates if item["window_seconds_nominal"] == length][:10]
            for length in sorted(best)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
