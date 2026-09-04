#!/usr/bin/env python3
"""Run the frozen dual-source matrix with acoustic quality and robust innovation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_cartesian_pce_apce as runner


METHODS = ("pce", "apce")
SEEDS = (2026082601, 2026082602, 2026082603, 2026082604, 2026082605)
INNOVATION_CHI2_THRESHOLD = 16.26623619623813  # chi-square, 3 df, 99.9%


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, choices=(1, 2), required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    frontend = args.root / f"target{args.target}" / "frontend"
    destination = args.output / f"target{args.target}"
    frontend_manifest_path = frontend / "frontend_manifest.json"
    frontend_manifest = json.loads(frontend_manifest_path.read_text(encoding="utf-8"))
    completed = []
    for method in METHODS:
        for seed in SEEDS:
            payload = runner.run_track(
                frontend=frontend,
                output=destination / "runs",
                method=method,
                seed=seed,
                device_name=args.device,
                segment="dual_evaluation",
                ensemble_size=48,
                q_min=2.0,
                q_max=12.0,
                position_init_std=50.0,
                velocity_init_std=10.0,
                observation_covariance_scale=1.0,
                turn_rate_radps=0.20,
                innovation_chi2_threshold=INNOVATION_CHI2_THRESHOLD,
            )
            completed.append({
                "target": args.target,
                "method": method,
                "seed": seed,
                "status": payload["status"],
                "records": len(payload["records"]),
            })
    runner.aggregate(destination / "runs", frontend, "dual_evaluation")
    manifest = {
        "target": args.target,
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "segment": "dual_evaluation",
        "configuration": {
            "ensemble_size": 48,
            "q_min_accel_mps2": 2.0,
            "q_max_accel_mps2": 12.0,
            "position_init_std_m": 50.0,
            "velocity_init_std_mps": 10.0,
            "observation_covariance_scale": 1.0,
            "turn_rate_radps": 0.20,
            "innovation_chi2_threshold_3df_999": INNOVATION_CHI2_THRESHOLD,
        },
        "gps_role": "offline evaluation only; no GPS in quality scaling, innovation clipping, state update or branch evidence",
        "frontend_reliability_profile": frontend_manifest.get("profile", "legacy_a6_quality_gated"),
        "frontend_manifest": str(frontend_manifest_path),
        "frontend_manifest_sha256": runner.sha256(frontend_manifest_path),
        "completed": completed,
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "matrix_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
