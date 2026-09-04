#!/usr/bin/env python3
"""Audit the frozen A6 frontend and fixed-window PCE/APCE comparison."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a6-root", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--a2-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.a6_root / "a6_manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((args.a6_root / "a6_metrics.json").read_text(encoding="utf-8"))
    selection = json.loads((args.matrix_root / "figure2_25s_selection" / "dual_long_window_selection.json").read_text(encoding="utf-8"))
    baseline = json.loads(args.a2_selection.read_text(encoding="utf-8"))
    selected = manifest["selected_calibration_parameters"]
    trials = manifest["calibration_trials"]
    best = min(trials, key=lambda row: (
        row["worst_calibration_rmse_m"] + 0.2 * row["mean_calibration_rmse_m"],
        row["bias_alpha"], row["quality_beta"], row["prior_sigma_m"],
    ))
    current = selection["selected"]
    previous = baseline["selected"]

    checks: dict[str, bool] = {
        "claim_boundary_present": "not a Zhang DBN-LA-NM reproduction" in manifest["claim_boundary"],
        "gps_runtime_excluded": "no GPS is read by held-out state updates" in manifest["gps_role"],
        "calibration_interval_frozen": manifest["calibration_interval_s"] == [46540, 46561],
        "evaluation_interval_frozen": manifest["evaluation_interval_s"] == [46561, 46741],
        "selection_recomputed": all(abs(float(selected[key]) - float(best[key])) < 1e-12 for key in ("bias_alpha", "quality_beta", "prior_sigma_m")),
        "selected_parameters_expected": selected["bias_alpha"] == 1.0 and selected["quality_beta"] == 0.2 and selected["prior_sigma_m"] == 80.0,
        "a6_outputs_complete": all(len(rows(args.a6_root / "a6" / f"target{target}_state.csv")) == 201 for target in (1, 2)),
        "evaluation_outputs_complete": all(metrics["variants"]["a6"][str(target)]["evaluation"]["valid_frames"] == 180 for target in (1, 2)),
        "state_covariances_psd": all(bool(manifest["covariance"]["psd"][str(target)]["finite_psd"]) and float(manifest["covariance"]["psd"][str(target)]["minimum_eigenvalue"]) > 0.0 for target in (1, 2)),
        "fixed_window": current["start_time_s"] == 46673 and current["end_time_s"] == 46697 and current["length_frames"] == 25,
        "window_admitted": all(bool(current[key]) for key in ("admitted_identity", "admitted_jump", "admitted_error", "admitted_uncertainty", "admitted_covariance")),
        "five_seed_matrix_complete": all(len([row for row in rows(args.matrix_root / f"target{target}" / "runs" / "method_summary.csv") if row["method"] == "apce"]) == 5 for target in (1, 2)),
        "t1_rmse_improved": float(current["target1_rmse_m"]) < float(previous["target1_rmse_m"]),
        "t2_rmse_improved": float(current["target2_rmse_m"]) < float(previous["target2_rmse_m"]),
        "worst_p90_improved": float(current["worst_target_p90_m"]) < float(previous["worst_target_p90_m"]),
    }
    payload = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "selected_calibration_parameters": selected,
        "fixed_window_comparison": {
            "a2": {"target1_rmse_m": previous["target1_rmse_m"], "target2_rmse_m": previous["target2_rmse_m"], "worst_target_p90_m": previous["worst_target_p90_m"]},
            "a6": {"target1_rmse_m": current["target1_rmse_m"], "target2_rmse_m": current["target2_rmse_m"], "worst_target_p90_m": current["worst_target_p90_m"]},
        },
        "claim_limit": "Fixed-window point tracking improves; full-record T2 RMSE and uncertainty calibration do not both improve.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
