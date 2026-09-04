#!/usr/bin/env python3
"""Tune one shared turn rate on the frozen dual-source calibration segment."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import run_cartesian_pce_apce as runner


TURN_RATES = (-0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, choices=(1, 2), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, default=2026082600)
    args = parser.parse_args()
    rows = []
    frontend = args.root / f"target{args.target}" / "frontend"
    for turn_rate in TURN_RATES:
        tag = f"turn_{turn_rate:+.2f}".replace("+", "p").replace("-", "m").replace(".", "p")
        payload = runner.run_track(
            frontend=frontend,
            output=args.output / f"target{args.target}" / tag / "runs",
            method="apce",
            seed=args.seed,
            device_name=args.device,
            segment="dual_calibration",
            ensemble_size=48,
            q_min=2.0,
            q_max=12.0,
            position_init_std=50.0,
            velocity_init_std=10.0,
            observation_covariance_scale=1.0,
            turn_rate_radps=turn_rate,
        )
        errors = [float(row["position_error_m"]) for row in payload["records"]]
        rows.append({
            "target": args.target,
            "turn_rate_radps": turn_rate,
            "frames": len(errors),
            "rmse_m": math.sqrt(statistics.mean(value * value for value in errors)),
            "median_error_m": statistics.median(errors),
            "p90_error_m": sorted(errors)[min(len(errors) - 1, int(0.90 * len(errors)))],
            "q_min_accel_mps2": 2.0,
            "q_max_accel_mps2": 12.0,
            "observation_covariance_scale": 1.0,
            "seed": args.seed,
        })
    destination = args.output / f"target{args.target}" / "turn_rate_probe.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
