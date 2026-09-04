#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_record(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def entropy_fraction(weights: np.ndarray) -> float:
    if len(weights) <= 1:
        return 0.0
    safe = np.maximum(np.asarray(weights, dtype=float), 1e-300)
    return float(-np.sum(safe * np.log(safe)) / np.log(len(safe)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--strong-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for seed_dir in sorted(args.smoke_root.glob("seed_*")):
        seed = int(seed_dir.name.split("_")[-1])
        for method in ("DEnKF", "BMA", "PCE", "APCE"):
            record = load_record(seed_dir / f"{method.lower()}.json")
            rows.append({
                "baseline": "linear_DEnKF_or_ensemble",
                "seed": seed,
                "method": method,
                "analysis_nrmse": float(record["interior_analysis_nrmse"]),
                "coverage_90": float(record["interior_coverage_90"]),
                "forecast_4ms_nrmse": float(record["interior_forecast_4ms_nrmse"]),
                "final_weight_entropy_fraction": entropy_fraction(np.asarray(record["final_weights"])),
            })
        strong = load_record(args.strong_root / seed_dir.name / "denkf.json")
        rows.append({
            "baseline": "cv_selected_fixed_RBF",
            "seed": seed,
            "method": "DEnKF-fixed-RBF",
            "analysis_nrmse": float(strong["interior_analysis_nrmse"]),
            "coverage_90": float(strong["interior_coverage_90"]),
            "forecast_4ms_nrmse": float(strong["interior_forecast_4ms_nrmse"]),
            "final_weight_entropy_fraction": 0.0,
        })

    summary = []
    for method in ("DEnKF", "DEnKF-fixed-RBF", "BMA", "PCE", "APCE"):
        selected = [row for row in rows if row["method"] == method]
        summary.append({
            "method": method,
            "seeds": len(selected),
            "analysis_nrmse_mean": float(np.mean([row["analysis_nrmse"] for row in selected])),
            "analysis_nrmse_std": float(np.std([row["analysis_nrmse"] for row in selected], ddof=1)),
            "coverage_90_mean": float(np.mean([row["coverage_90"] for row in selected])),
            "coverage_90_std": float(np.std([row["coverage_90"] for row in selected], ddof=1)),
            "forecast_4ms_nrmse_mean": float(np.mean([row["forecast_4ms_nrmse"] for row in selected])),
        })

    strong = next(row for row in summary if row["method"] == "DEnKF-fixed-RBF")
    comparisons = {}
    for method in ("BMA", "PCE", "APCE"):
        item = next(row for row in summary if row["method"] == method)
        comparisons[method] = {
            "analysis_nrmse_change_vs_cv_selected_fixed_RBF": item["analysis_nrmse_mean"] - strong["analysis_nrmse_mean"],
            "coverage_change_vs_cv_selected_fixed_RBF": item["coverage_90_mean"] - strong["coverage_90_mean"],
            "forecast_4ms_change_vs_cv_selected_fixed_RBF": item["forecast_4ms_nrmse_mean"] - strong["forecast_4ms_nrmse_mean"],
        }
    report = {
        "case": "meshir_s1_reconstruction_192_stage_d_rbf_v4_shared_noise",
        "claim_boundary": "PCE/APCE improve uncertainty calibration and remain competitive in analysis reconstruction; no claim of universal point-error superiority over a CV-selected fixed RBF model.",
        "shared_candidate_noise": True,
        "candidate_selection": "four-fold pseudo-holdout within 128 observed boundary points; 1314 formal held-out boundary points excluded from selection",
        "manuscript_modified": False,
        "summary": summary,
        "comparisons_vs_cv_selected_fixed_RBF": comparisons,
        "forecast_interpretation": "64 ms boundary observations stop; 4 ms forecast remains a propagation diagnostic and is not a passed prediction claim.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "final_run_source_data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "final_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (args.output / "FINAL_AUDIT_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
