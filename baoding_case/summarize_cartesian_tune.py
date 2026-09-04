#!/usr/bin/env python3
"""Summarize calibration-only Cartesian filter sweeps without changing runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--method", default="apce")
    parser.add_argument("--seed", type=int, default=2026082501)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.root.glob(f"*/runs/{args.method}_seed_{args.seed}.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = [row for row in payload["records"] if row.get("position_error_m") is not None]
        if not records:
            continue
        errors = [float(row["position_error_m"]) for row in records]
        coverage = [float(row["coverage_90"]) for row in records]
        crps = [float(row["crps_position_m"]) for row in records]
        rows.append({
            "candidate": path.parents[1].name,
            "method": payload["method"],
            "q_min_accel_mps2": payload["q_min_accel_mps2"],
            "q_max_accel_mps2": payload["q_max_accel_mps2"],
            "observation_covariance_scale": payload["observation_covariance_scale"],
            "frames": len(records),
            "rmse_m": math.sqrt(sum(value * value for value in errors) / len(errors)),
            "mean_crps_m": sum(crps) / len(crps),
            "coverage_90": sum(coverage) / len(coverage),
            "coverage_gap_to_0p90": abs(sum(coverage) / len(coverage) - 0.90),
        })
    rows.sort(key=lambda row: (row["rmse_m"], row["coverage_gap_to_0p90"], row["mean_crps_m"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["candidate"])
        writer.writeheader(); writer.writerows(rows)
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
