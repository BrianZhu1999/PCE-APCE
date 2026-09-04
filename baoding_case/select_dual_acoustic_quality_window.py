#!/usr/bin/env python3
"""Select a contiguous dual-source operating window without GPS errors.

Selection uses only post-calibration association cost and triangulation geometry:
both targets must be valid, retain sufficient ray inliers and have bounded
condition numbers. GPS coordinates are deliberately excluded from selection.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def median(values: list[float]) -> float:
    return statistics.median(values) if values else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--association-gate", type=Path, required=True)
    parser.add_argument("--target1-triangulation", type=Path, required=True)
    parser.add_argument("--target2-triangulation", type=Path, required=True)
    parser.add_argument("--window", type=int, default=25)
    parser.add_argument("--min-inliers", type=int, default=5)
    parser.add_argument("--max-condition", type=float, default=25.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gate = json.loads(args.association_gate.read_text(encoding="utf-8"))
    diagnostics = gate["global_association_audit"]["frame_diagnostics"]
    target1 = {int(row["frame_index"]): row for row in read_csv(args.target1_triangulation)}
    target2 = {int(row["frame_index"]): row for row in read_csv(args.target2_triangulation)}
    candidates = []
    for diag in diagnostics:
        frame = int(diag["frame_index"])
        one, two = target1.get(frame), target2.get(frame)
        if not one or not two or bool(diag["calibration_frame"]):
            continue
        try:
            valid = one["valid"].lower() == "true" and two["valid"].lower() == "true"
            inliers = min(int(one["inlier_nodes"]), int(two["inlier_nodes"]))
            condition = max(float(one["condition_number"]), float(two["condition_number"]))
            cost = float(diag["total_cost"])
        except (TypeError, ValueError):
            continue
        candidates.append({"frame_index": frame, "time_s": float(diag["time_s"]), "valid": valid, "min_inliers": inliers, "max_condition": condition, "association_cost": cost})

    by_frame = {row["frame_index"]: row for row in candidates}
    available = sorted(by_frame)
    windows = []
    for start in available:
        rows = [by_frame.get(start + offset) for offset in range(args.window)]
        if any(row is None for row in rows):
            continue
        if not all(row["valid"] and row["min_inliers"] >= args.min_inliers and row["max_condition"] <= args.max_condition for row in rows):
            continue
        score = (median([row["association_cost"] for row in rows]), median([row["max_condition"] for row in rows]), -median([row["min_inliers"] for row in rows]))
        windows.append({"start_frame": start, "end_frame": start + args.window - 1, "time_start_s": rows[0]["time_s"], "time_end_s": rows[-1]["time_s"], "quality_score": score, "median_association_cost": score[0], "median_max_condition": score[1], "median_min_inliers": -score[2], "frames": rows})
    if not windows:
        raise RuntimeError("no continuous acoustic-quality window satisfies the predeclared geometry thresholds")
    chosen = min(windows, key=lambda row: tuple(row["quality_score"]))
    result = {"claim_status": "acoustic_quality_selected_window", "selection_rule": {"uses_gps_error": False, "uses_gps_runtime": False, "excludes_calibration_frames": True, "window_frames": args.window, "min_inliers": args.min_inliers, "max_condition": args.max_condition, "ranking": "lexicographic minimum of median association cost, median maximum condition number, and negative median minimum inliers"}, "source": {"association_gate": str(args.association_gate), "target1_triangulation": str(args.target1_triangulation), "target2_triangulation": str(args.target2_triangulation)}, "selected": chosen, "eligible_window_count": len(windows), "warning": "GPS truth is not used to select this window. Any GPS errors reported later are evaluation-only and do not establish full-record performance."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
