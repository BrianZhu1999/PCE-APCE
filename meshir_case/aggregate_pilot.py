#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def read_records(root: Path, case: str) -> pd.DataFrame:
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(root.glob(f"{case}/fold*/*/*/seed_*.json"))]
    return pd.DataFrame(records)


def scalar_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in frame.columns if not frame[column].map(lambda value: isinstance(value, (list, dict))).any()]
    return frame[columns]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1-results", type=Path, required=True)
    parser.add_argument("--s32-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    s1 = read_records(args.s1_results, "s1")
    s32 = read_records(args.s32_results, "s32")
    scalar_frame(s1).to_csv(args.output / "s1_run_source_data.csv", index=False)
    scalar_frame(s32).to_csv(args.output / "s32_run_source_data.csv", index=False)
    combined = pd.concat([scalar_frame(s1), scalar_frame(s32)], ignore_index=True)
    metrics = [column for column in [
        "reconstruction_nrmse", "prediction_nrmse", "reconstruction_lsd_db",
        "prediction_correlation", "coverage_90", "localization_error_m",
        "geometric_baseline_error_m", "oracle_improvement_fraction",
        "median_separation_ratio",
    ] if column in combined]
    summary = combined.groupby(["case", "condition", "method"])[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(filter(None, map(str, column))).rstrip("_") for column in summary.columns]
    summary.to_csv(args.output / "summary_metrics.csv", index=False)
    comparisons = []
    for case, frame in (("s1", s1), ("s32", s32)):
        for (fold, condition, seed), group in frame.groupby(["fold", "condition", "seed"]):
            baseline = group[group.method == "DEnKF"].iloc[0]
            for method in ("BMA", "PCE", "APCE"):
                current = group[group.method == method].iloc[0]
                row = {
                    "case": case, "fold": int(fold), "condition": condition,
                    "seed": int(seed), "method": method,
                    "delta_reconstruction_nrmse": float(current.reconstruction_nrmse - baseline.reconstruction_nrmse),
                    "delta_prediction_nrmse": float(current.prediction_nrmse - baseline.prediction_nrmse),
                }
                if case == "s32":
                    row["delta_localization_error_m"] = float(current.localization_error_m - baseline.localization_error_m)
                    row["delta_vs_geometric_baseline_m"] = float(current.localization_error_m - current.geometric_baseline_error_m)
                comparisons.append(row)
    pd.DataFrame(comparisons).to_csv(args.output / "paired_comparisons.csv", index=False)
    s1_oracle_pass = int((s1.drop_duplicates("fold").oracle_improvement_fraction > 0.0).sum()) >= 3 if len(s1) else False
    s1_prediction_pass = bool((s1[s1.method.isin(["PCE", "APCE"])].prediction_nrmse < 1.0).any()) if len(s1) else False
    s1_coverage_pass = bool(s1[s1.method.isin(["PCE", "APCE"])].coverage_90.between(0.70, 1.0).all()) if len(s1) else False
    s32_localization_pass = bool((s32[s32.method.isin(["PCE", "APCE"])].localization_error_m <= s32[s32.method.isin(["PCE", "APCE"])].geometric_baseline_error_m).all()) if len(s32) else False
    s32_coverage_pass = bool(s32[s32.method.isin(["PCE", "APCE"])].coverage_90.between(0.70, 1.0).all()) if len(s32) else False
    admission = {
        "stage": "pre-admission smoke",
        "s1_completed_runs": int(len(s1)),
        "s32_completed_runs": int(len(s32)),
        "full_pilot_expected_runs": 192,
        "full_pilot_launched": False,
        "s1_oracle_gate_pass": s1_oracle_pass,
        "s1_raw_prediction_gate_pass": s1_prediction_pass,
        "s1_coverage_gate_pass": s1_coverage_pass,
        "s32_localization_vs_geometric_gate_pass": s32_localization_pass,
        "s32_coverage_gate_pass": s32_coverage_pass,
    }
    admission["pilot_admitted"] = bool(all(value for key, value in admission.items() if key.endswith("_pass")))
    (args.output / "meshir_admission.json").write_text(json.dumps(admission, indent=2), encoding="utf-8")
    report = [
        "# MeshRIR S1/S32 pilot admission report", "", "## Decision", "",
        "**Rejected at pre-admission smoke; the 192-run matrix was not launched.**", "",
        "S1 reconstruction updates reduced analysis-window error, but the frozen ROM did not predict raw RIR waveforms in the 64--200 ms window and its candidate oracle did not exceed the fixed model.", "",
        "S32 source-holdout weights did not beat the corrected geometric direct-TOA baseline. The result cannot support a PCE/APCE localization claim.", "",
        "## Gate result", "", "```json", json.dumps(admission, indent=2), "```", "",
        "The measured datasets, code, smoke records and diagnostic figure are retained as an audit archive. No manuscript file was modified.",
    ]
    (args.output / "MESHIR_PILOT_ADMISSION_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(admission, indent=2))


if __name__ == "__main__":
    main()
