#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METHODS = ("DEnKF", "BMA", "PCE", "APCE")
CONDITIONS = ("standard", "blackout")
METRICS = (
    "heldout_nrmse", "heldout_erms", "normalized_crps", "coverage_90",
    "normalized_interval_width_90", "spectral_nrmse_6_8p5hz",
    "blackout_heldout_nrmse", "oracle_candidate_nrmse",
    "mean_candidate_separation_ratio", "minimum_candidate_separation_ratio",
    "final_effective_candidate_count",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = []
    for path in sorted(args.run_root.glob("level*/*/*/seed_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["run_json"] = str(path)
        rows.append(record)
    expected = len(config["levels"]["validation"]) * len(CONDITIONS) * len(METHODS) * len(config["assimilation"]["pilot_seeds"])
    keys = [(row["level"], row["condition"], row["method"], row["seed"]) for row in rows]
    duplicates = len(keys) - len(set(keys))
    missing = []
    for level in config["levels"]["validation"]:
        for condition in CONDITIONS:
            for method in METHODS:
                for seed in config["assimilation"]["pilot_seeds"]:
                    if (level, condition, method, seed) not in set(keys):
                        missing.append([level, condition, method, seed])
    if len(rows) != expected or duplicates or missing:
        raise RuntimeError(f"pilot matrix incomplete: rows={len(rows)}/{expected}, duplicates={duplicates}, missing={missing[:5]}")
    summary = []
    for level in config["levels"]["validation"]:
        for condition in CONDITIONS:
            for method in METHODS:
                subset = [row for row in rows if row["level"] == level and row["condition"] == condition and row["method"] == method]
                item = {"level": level, "condition": condition, "method": method, "n_seeds": len(subset)}
                for metric in METRICS:
                    values = np.asarray([float(row[metric]) for row in subset], dtype=float)
                    finite = values[np.isfinite(values)]
                    item[f"mean_{metric}"] = float(np.mean(finite)) if finite.size else float("nan")
                    item[f"sd_{metric}"] = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
                summary.append(item)
    paired = []
    index = {(row["level"], row["condition"], row["method"], row["seed"]): row for row in rows}
    for level in config["levels"]["validation"]:
        for condition in CONDITIONS:
            for comparator in ("DEnKF", "BMA", "PCE"):
                for metric in ("heldout_nrmse", "normalized_crps", "coverage_90"):
                    differences = []
                    for seed in config["assimilation"]["pilot_seeds"]:
                        apce = float(index[(level, condition, "APCE", seed)][metric])
                        other = float(index[(level, condition, comparator, seed)][metric])
                        difference = other - apce if metric != "coverage_90" else abs(other - 0.90) - abs(apce - 0.90)
                        differences.append(difference)
                    paired.append({
                        "level": level, "condition": condition,
                        "comparison": f"APCE_vs_{comparator}", "metric": metric,
                        "positive_favours_APCE": True,
                        "mean_improvement": float(np.mean(differences)),
                        "wins": int(np.sum(np.asarray(differences) > 0.0)),
                        "n": len(differences),
                    })
    standard = {(row["level"], row["method"]): row for row in summary if row["condition"] == "standard"}
    blackout = {(row["level"], row["method"]): row for row in summary if row["condition"] == "blackout"}
    oracle_improved = []
    method_improved = []
    nrmse_admitted = []
    direction_consistent = []
    for level in config["levels"]["validation"]:
        fixed = standard[(level, "DEnKF")]["mean_heldout_nrmse"]
        oracle = standard[(level, "PCE")]["mean_oracle_candidate_nrmse"]
        oracle_improved.append((fixed - oracle) / max(fixed, 1e-12) >= float(config["gates"]["minimum_oracle_improvement_fraction"]))
        best_standard = min(standard[(level, method)]["mean_heldout_nrmse"] for method in ("PCE", "APCE"))
        best_standard_crps = min(standard[(level, method)]["mean_normalized_crps"] for method in ("PCE", "APCE"))
        best_blackout = min(blackout[(level, method)]["mean_heldout_nrmse"] for method in ("PCE", "APCE"))
        method_improved.append(best_standard < fixed or best_standard_crps < standard[(level, "DEnKF")]["mean_normalized_crps"])
        nrmse_admitted.append(best_standard <= float(config["gates"]["maximum_heldout_nrmse"]))
        direction_consistent.append((best_standard - fixed) * (best_blackout - blackout[(level, "DEnKF")]["mean_heldout_nrmse"]) >= 0.0)
    pce_apce_rows = [row for row in summary if row["method"] in ("PCE", "APCE")]
    separation_pass = all(row["mean_mean_candidate_separation_ratio"] > float(config["gates"]["minimum_candidate_separation_ratio"]) for row in pce_apce_rows)
    coverage_pass = all(
        float(config["gates"]["minimum_coverage_90"]) <= row["mean_coverage_90"] <= float(config["gates"]["maximum_coverage_90"])
        for row in pce_apce_rows
    )
    no_weight_updates_in_blackout = all(not bool(row["weight_update_inside_blackout"]) for row in rows if row["condition"] == "blackout")
    gate = {
        "matrix_complete": True,
        "run_count": len(rows),
        "oracle_improved_levels": int(sum(oracle_improved)),
        "method_improved_levels": int(sum(method_improved)),
        "nrmse_admitted_levels": int(sum(nrmse_admitted)),
        "standard_blackout_direction_consistent_levels": int(sum(direction_consistent)),
        "candidate_separation_pass": separation_pass,
        "coverage_pass": coverage_pass,
        "blackout_weight_freeze_pass": no_weight_updates_in_blackout,
    }
    gate["pilot_admitted"] = bool(
        gate["oracle_improved_levels"] >= int(config["gates"]["minimum_levels_oracle_improved"])
        and gate["method_improved_levels"] >= int(config["gates"]["minimum_levels_method_improved"])
        and gate["nrmse_admitted_levels"] >= 2
        and gate["standard_blackout_direction_consistent_levels"] >= 2
        and separation_pass and coverage_pass and no_weight_updates_in_blackout
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "f16_gvt_run_source_data.csv", rows)
    write_csv(args.output_root / "f16_gvt_summary.csv", summary)
    write_csv(args.output_root / "f16_gvt_paired_comparisons.csv", paired)
    (args.output_root / "f16_gvt_admission.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
