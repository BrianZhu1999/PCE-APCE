#!/usr/bin/env python3
"""Decide whether a Baoding smoke supports a larger PCE/APCE run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--raw-admission", type=Path, required=True)
    args = parser.parse_args()
    with (args.result / "method_summary.csv").open(encoding="utf-8") as stream:
        rows = {row["method"]: row for row in csv.DictReader(stream)}
    required = {"denkf", "aug_enkf", "bma", "pce", "apce"}
    missing = sorted(required - set(rows))
    baselines = [rows[name] for name in ("denkf", "aug_enkf", "bma") if name in rows]
    proposed = [rows[name] for name in ("pce", "apce") if name in rows]
    best_baseline_rmse = min(float(row["position_rmse_m_mean"]) for row in baselines)
    best_proposed_rmse = min(float(row["position_rmse_m_mean"]) for row in proposed)
    best_baseline_crps = min(float(row["crps_position_m_mean"]) for row in baselines)
    best_proposed_crps = min(float(row["crps_position_m_mean"]) for row in proposed)
    best_baseline_coverage_error = min(abs(float(row["coverage_90_mean"]) - 0.90) for row in baselines)
    best_proposed_coverage_error = min(abs(float(row["coverage_90_mean"]) - 0.90) for row in proposed)
    gate = {
        "task": "2017 Baoding corrected direct-WAV MUSIC near-field PCE/APCE smoke gate",
        "matrix_complete": not missing and all(int(rows[name]["runs"]) == 5 for name in required),
        "missing_methods": missing,
        "seeds_role": "numerical sensitivity on one held-out physical trajectory; not independent experiments",
        "best_baseline_position_rmse_m": best_baseline_rmse,
        "best_pce_apce_position_rmse_m": best_proposed_rmse,
        "best_baseline_crps_m": best_baseline_crps,
        "best_pce_apce_crps_m": best_proposed_crps,
        "best_baseline_coverage_error": best_baseline_coverage_error,
        "best_pce_apce_coverage_error": best_proposed_coverage_error,
        "method_summary": rows,
    }
    gate["performance_gate"] = bool(
        gate["matrix_complete"]
        and best_proposed_rmse < best_baseline_rmse
        and best_proposed_crps < best_baseline_crps
        and best_proposed_coverage_error <= best_baseline_coverage_error
    )
    gate["formal_admission"] = False
    if gate["performance_gate"]:
        gate["archive_decision"] = "candidate_improvement; require independent-trajectory replication before formal claim"
        gate["interpretation"] = (
            "The guarded smoke is a positive candidate signal: a PCE/APCE method beats the best baseline on the predefined "
            "point-error, CRPS and coverage-error criteria. The five seeds remain numerical sensitivity runs on one held-out "
            "physical trajectory, so this does not independently authorize a formal performance claim."
        )
    else:
        gate["archive_decision"] = "archive_as_negative_robustness; do_not_add_to_main_manuscript"
        gate["interpretation"] = (
            "The corrected raw-WAV frontend supports near-field 3-D tracking, but PCE/APCE do not beat the best baseline "
            "on point error and CRPS. APCE improves coverage relative to the baselines only by widening intervals; this is "
            "not sufficient evidence for a paper performance claim."
        )
    gate["provenance"] = {
        "result": str(args.result),
        "method_summary_sha256": sha256(args.result / "method_summary.csv"),
        "run_summary_sha256": sha256(args.result / "run_summary.csv"),
        "tracking_manifest_sha256": sha256(args.result / "tracking_manifest.json"),
        "raw_admission": str(args.raw_admission),
        "raw_admission_sha256": sha256(args.raw_admission / "raw_frontend_gate.json"),
        "runner_sha256": sha256(Path(__file__)),
    }
    (args.result / "method_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
