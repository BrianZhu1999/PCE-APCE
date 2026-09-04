#!/usr/bin/env python3
"""Tune and validate PCE/APCE on a frozen dual-source candidate segment.

The candidate segment is already selected by a separate audit. Parameters are
chosen on three numerical seeds and evaluated on two held-out seeds. GPS is
used to score configurations, never as a filter observation or state update.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
from pathlib import Path


TARGETS = (1, 2)
METHODS = ("pce", "apce")
TUNE_SEEDS = (2026081900, 2026081901, 2026081902)
VALIDATION_SEEDS = (2026081903, 2026081904)


def score(payload: dict) -> float:
    errors = [float(row["position_error_m"]) for row in payload["records"] if row.get("position_error_m") is not None]
    return math.sqrt(sum(value * value for value in errors) / len(errors))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    code_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(code_dir))
    import run_baoding
    from run_shuangyuan_pce_apce import install_real_data_stability_layer

    stability = install_real_data_stability_layer(run_baoding)
    base_cfg = json.loads(args.config.read_text(encoding="utf-8"))
    grid = list(itertools.product((20.0, 50.0), (5.0, 10.0), (4.0, 8.0)))
    trials = []
    for trial_id, (position_std, velocity_std, observation_std) in enumerate(grid):
        cfg = dict(base_cfg)
        cfg.update({"position_init_std_m": position_std, "velocity_init_std_mps": velocity_std,
                    "observation_angle_std_deg": observation_std, "ensemble_size": 24})
        values = []
        for target in TARGETS:
            frontend = args.segment_root / f"target{target}" / "frontend"
            for method in METHODS:
                for seed in TUNE_SEEDS:
                    payload = run_baoding.run_track(cfg, frontend, args.output_root / "tuning" / f"trial{trial_id}" / f"target{target}", method, seed, args.device)
                    values.append(score(payload))
        trials.append({"trial_id": trial_id, "position_init_std_m": position_std,
                       "velocity_init_std_mps": velocity_std, "observation_angle_std_deg": observation_std,
                       "mean_rmse_m": statistics.mean(values), "median_rmse_m": statistics.median(values),
                       "max_rmse_m": max(values), "component_rmse_m": values})
    best = min(trials, key=lambda row: (row["mean_rmse_m"] + 0.20 * row["max_rmse_m"], row["trial_id"]))
    selected_cfg = dict(base_cfg)
    selected_cfg.update({"position_init_std_m": best["position_init_std_m"],
                         "velocity_init_std_mps": best["velocity_init_std_mps"],
                         "observation_angle_std_deg": best["observation_angle_std_deg"], "ensemble_size": 24})
    validation = []
    for target in TARGETS:
        source_frontend = args.segment_root / f"target{target}" / "frontend"
        destination = args.output_root / f"target{target}"
        destination.mkdir(parents=True, exist_ok=True)
        frontend = destination / "frontend"
        if not frontend.exists():
            import shutil
            shutil.copytree(source_frontend, frontend)
        for method in METHODS:
            for seed in VALIDATION_SEEDS:
                payload = run_baoding.run_track(selected_cfg, frontend, destination / "runs", method, seed, args.device)
                validation.append({"target": target, "method": method, "seed": seed, "rmse_m": score(payload)})
    manifest = {
        "claim_status": "inspection_selected_segment_heldout_seed_validation",
        "gps_role": "parameter scoring and final evaluation only; not an observation or state update",
        "candidate_segment_root": str(args.segment_root),
        "tuning_seeds": TUNE_SEEDS, "validation_seeds": VALIDATION_SEEDS,
        "selection_objective": "mean RMSE + 0.20 * maximum RMSE across targets, methods, and tuning seeds",
        "parameter_grid": trials, "selected": best, "selected_config": selected_cfg,
        "validation": validation, "validation_mean_rmse_m": statistics.mean(row["rmse_m"] for row in validation),
        "stability_layer": stability,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "tuning_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected": best, "validation": validation,
                      "validation_mean_rmse_m": manifest["validation_mean_rmse_m"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
