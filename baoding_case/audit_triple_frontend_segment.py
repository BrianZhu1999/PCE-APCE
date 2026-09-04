#!/usr/bin/env python3
"""Find stable cross-node windows in provisional three-peak MUSIC output.

This does not assign peaks to helicopters and cannot produce a PCE/APCE gate.
It only identifies candidate windows where the provisional rank-wise peaks are
consistent across the nine nodes, which is the input to a later association
audit.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


NODES = (1, 2, 3, 5, 6, 7, 8, 11, 13)


def circ(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def frame_cost(rows: list[list[dict[str, str]]], i: int) -> float:
    cost = 0.0
    for rank in (1, 2, 3):
        az = [float(node[i][f"azimuth_{rank}_deg"]) for node in rows]
        ze = [float(node[i][f"zenith_{rank}_deg"]) for node in rows]
        az_center = math.degrees(math.atan2(sum(math.sin(math.radians(x)) for x in az), sum(math.cos(math.radians(x)) for x in az))) % 360.0
        cost += statistics.median(circ(x, az_center) for x in az)
        cost += statistics.median(abs(x - statistics.median(ze)) for x in ze)
        cost += 0.05 * statistics.median(float(node[i][f"azimuth_strength_{rank}"]) for node in rows) ** -1
    return cost


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=25)
    args = parser.parse_args()
    rows = [read(args.input_root / f"node{node}" / f"triple_doa_node_{node}_132614.csv") for node in NODES]
    count = min(len(item) for item in rows)
    rows = [item[:count] for item in rows]
    frame_scores = [frame_cost(rows, i) for i in range(count)]
    candidates = []
    for start in range(count - args.window + 1):
        values = frame_scores[start:start + args.window]
        candidates.append({"start_index": start, "stop_index_exclusive": start + args.window,
                           "score": statistics.mean(values),
                           "start_time_s": float(rows[0][start]["time_s"]),
                           "end_time_s": float(rows[0][start + args.window - 1]["time_s"])})
    candidates.sort(key=lambda item: (item["score"], item["start_index"]))
    result = {
        "claim_status": "frontend_inspection_only",
        "target_association_available": False,
        "gps_truth_available_for_three_targets": False,
        "input_root": str(args.input_root), "nodes": NODES, "common_frames": count,
        "selection_rule": "minimum mean rank-wise cross-node azimuth/zenith dispersion plus inverse peak strength",
        "window_frames": args.window, "top_candidates": candidates[:20],
        "warning": "rank-wise peak agreement is not target identity; no triangulation, tracking, PCE/APCE, or superiority gate is inferred",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
