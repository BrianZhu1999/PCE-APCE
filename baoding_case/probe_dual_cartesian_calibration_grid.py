#!/usr/bin/env python3
"""Tune target-specific dual-source APCE dynamics on calibration data only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import run_cartesian_pce_apce as runner


# The Baoding calibration GPS shows clockwise motion for both targets.  The
# original probe searched positive rates only, which cannot represent that
# motion and biases the posterior dynamics in the wrong direction.
TURN_RATES = (-0.12, -0.10, -0.08, -0.075, -0.07, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20)
Q_RANGES = ((0.5, 4.0), (1.0, 8.0), (2.0, 12.0))
COVARIANCE_SCALES = (1.0, 2.0, 4.0, 8.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: np.ndarray, value: float) -> float:
    return float(np.percentile(values, value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, choices=(1, 2), required=True)
    parser.add_argument("--device", choices=("cuda:2", "cuda:3"), required=True)
    parser.add_argument("--seed", type=int, default=2026082600)
    args = parser.parse_args()

    frontend = args.root / f"target{args.target}" / "frontend"
    rows: list[dict[str, object]] = []
    for turn_rate in TURN_RATES:
        for q_min, q_max in Q_RANGES:
            for covariance_scale in COVARIANCE_SCALES:
                tag = (
                    f"turn{turn_rate:.2f}_q{q_min:g}-{q_max:g}_s{covariance_scale:g}"
                    .replace(".", "p")
                )
                payload = runner.run_track(
                    frontend=frontend,
                    output=args.output / f"target{args.target}" / tag / "runs",
                    method="apce",
                    seed=args.seed,
                    device_name=args.device,
                    segment="dual_calibration",
                    ensemble_size=48,
                    q_min=q_min,
                    q_max=q_max,
                    position_init_std=50.0,
                    velocity_init_std=10.0,
                    observation_covariance_scale=covariance_scale,
                    turn_rate_radps=turn_rate,
                )
                records = payload["records"]
                errors = np.asarray([float(row["position_error_m"]) for row in records])
                positions = np.asarray([[float(row[key]) for key in ("px", "py", "pz")] for row in records])
                steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
                widths = np.asarray([float(row["interval_width_m"]) for row in records])
                coverages = np.asarray([float(row["coverage_90"]) for row in records])
                rmse = float(np.sqrt(np.mean(np.square(errors))))
                p90 = percentile(errors, 90.0)
                max_step = float(np.max(steps))
                score = rmse + 0.10 * p90 + 0.12 * max(0.0, max_step - 150.0)
                rows.append({
                    "target": args.target,
                    "turn_rate_radps": turn_rate,
                    "q_min_accel_mps2": q_min,
                    "q_max_accel_mps2": q_max,
                    "observation_covariance_scale": covariance_scale,
                    "frames": len(records),
                    "rmse_m": rmse,
                    "median_error_m": float(np.median(errors)),
                    "p90_error_m": p90,
                    "maximum_step_m": max_step,
                    "median_marginal_width_m": float(np.median(widths)),
                    "mean_component_coverage_90": float(np.mean(coverages)),
                    "selection_score": score,
                    "seed": args.seed,
                    "run_path": str(args.output / f"target{args.target}" / tag / "runs" / f"apce_seed_{args.seed}.json"),
                })

    rows.sort(key=lambda row: float(row["selection_score"]))
    destination = args.output / f"target{args.target}" / "calibration_grid.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "target": args.target,
        "selection_segment": "dual_calibration",
        "gps_role": "offline calibration scoring only",
        "grid": {
            "turn_rate_radps": list(TURN_RATES),
            "q_ranges_accel_mps2": [list(values) for values in Q_RANGES],
            "observation_covariance_scales": list(COVARIANCE_SCALES),
            "position_init_std_m": 50.0,
            "velocity_init_std_mps": 10.0,
            "ensemble_size": 48,
            "seed": args.seed,
        },
        "selection_score": "RMSE + 0.10*P90 + 0.12*max(0, maximum one-second step - 150 m)",
        "selected": rows[0],
        "top_10": rows[:10],
        "frontend": str(frontend),
        "frontend_manifest_sha256": sha256(frontend / "frontend_manifest.json"),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    manifest_path = args.output / f"target{args.target}" / "calibration_grid_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"target": args.target, "selected": rows[0], "manifest": str(manifest_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
