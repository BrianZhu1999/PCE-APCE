#!/usr/bin/env python3
"""Validate shortlisted dual-source APCE configurations over five seeds."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

import run_cartesian_pce_apce as runner


SEEDS = (2026082601, 2026082602, 2026082603, 2026082604, 2026082605)
CANDIDATES = {
    1: (
        ("lowq_s2_turn010", 0.5, 4.0, 2.0, 0.10),
        ("midq_s8_turn010", 1.0, 8.0, 8.0, 0.10),
        ("lowq_s1_turn010", 0.5, 4.0, 1.0, 0.10),
        ("frozen_q2_12_s1_turn020", 2.0, 12.0, 1.0, 0.20),
    ),
    2: (
        ("lowq_s1_turn010", 0.5, 4.0, 1.0, 0.10),
        ("midq_s1_turn015", 1.0, 8.0, 1.0, 0.15),
        ("midq_s1_turn020", 1.0, 8.0, 1.0, 0.20),
        ("frozen_q2_12_s1_turn020", 2.0, 12.0, 1.0, 0.20),
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, choices=(1, 2), required=True)
    parser.add_argument("--device", choices=("cuda:2", "cuda:3"), required=True)
    args = parser.parse_args()

    frontend = args.root / f"target{args.target}" / "frontend"
    aggregate_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    for name, q_min, q_max, scale, turn_rate in CANDIDATES[args.target]:
        config_runs = []
        for seed in SEEDS:
            payload = runner.run_track(
                frontend=frontend,
                output=args.output / f"target{args.target}" / name / "runs",
                method="apce",
                seed=seed,
                device_name=args.device,
                segment="dual_calibration",
                ensemble_size=48,
                q_min=q_min,
                q_max=q_max,
                position_init_std=50.0,
                velocity_init_std=10.0,
                observation_covariance_scale=scale,
                turn_rate_radps=turn_rate,
            )
            records = payload["records"]
            errors = np.asarray([float(row["position_error_m"]) for row in records])
            positions = np.asarray([[float(row[key]) for key in ("px", "py", "pz")] for row in records])
            steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            result = {
                "config": name,
                "seed": seed,
                "rmse_m": float(np.sqrt(np.mean(np.square(errors)))),
                "median_error_m": float(np.median(errors)),
                "p90_error_m": float(np.percentile(errors, 90.0)),
                "maximum_step_m": float(np.max(steps)),
                "median_marginal_width_m": float(np.median([float(row["interval_width_m"]) for row in records])),
                "mean_component_coverage_90": float(np.mean([float(row["coverage_90"]) for row in records])),
            }
            config_runs.append(result)
            run_rows.append({"target": args.target, **result})
        median_rmse = float(np.median([row["rmse_m"] for row in config_runs]))
        median_p90 = float(np.median([row["p90_error_m"] for row in config_runs]))
        median_max_step = float(np.median([row["maximum_step_m"] for row in config_runs]))
        aggregate_rows.append({
            "target": args.target,
            "config": name,
            "q_min_accel_mps2": q_min,
            "q_max_accel_mps2": q_max,
            "observation_covariance_scale": scale,
            "turn_rate_radps": turn_rate,
            "seeds": len(SEEDS),
            "median_seed_rmse_m": median_rmse,
            "median_seed_p90_error_m": median_p90,
            "median_seed_maximum_step_m": median_max_step,
            "mean_seed_coverage_90": float(np.mean([row["mean_component_coverage_90"] for row in config_runs])),
            "median_seed_marginal_width_m": float(np.median([row["median_marginal_width_m"] for row in config_runs])),
            "selection_score": median_rmse + 0.10 * median_p90 + 0.12 * max(0.0, median_max_step - 150.0),
        })

    aggregate_rows.sort(key=lambda row: float(row["selection_score"]))
    destination = args.output / f"target{args.target}"
    destination.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("calibration_validation_runs.csv", run_rows), ("calibration_validation_summary.csv", aggregate_rows)):
        with (destination / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    manifest = {
        "target": args.target,
        "selection_segment": "dual_calibration",
        "seeds": list(SEEDS),
        "gps_role": "offline calibration scoring only",
        "selected": aggregate_rows[0],
        "all_candidates": aggregate_rows,
        "script": str(Path(__file__).resolve()),
    }
    manifest_path = destination / "calibration_validation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"target": args.target, "selected": aggregate_rows[0], "manifest": str(manifest_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
