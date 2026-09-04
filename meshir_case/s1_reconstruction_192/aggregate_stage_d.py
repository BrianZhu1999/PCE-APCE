#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METHODS = ("DEnKF", "BMA", "PCE", "APCE")
METRICS = (
    "interior_analysis_nrmse",
    "interior_analysis_correlation",
    "interior_coverage_90",
    "interior_forecast_1ms_nrmse",
    "interior_forecast_2ms_nrmse",
    "interior_forecast_4ms_nrmse",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    errors: list[str] = []
    rows: list[dict[str, object]] = []
    for seed_dir in sorted(path for path in args.root.glob("seed_*") if path.is_dir()):
        seed = int(seed_dir.name.split("_")[-1])
        records = {}
        for method in METHODS:
            path = seed_dir / f"{method.lower()}.json"
            if not path.is_file():
                errors.append(f"missing {path}")
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            records[method] = record
            weights = np.asarray(record["final_weights"], dtype=float)
            if not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
                errors.append(f"invalid weights seed={seed} method={method}")
            for metric in METRICS:
                if not np.isfinite(float(record[metric])):
                    errors.append(f"nonfinite {metric} seed={seed} method={method}")
            rows.append({
                "seed": seed,
                "method": method,
                **{metric: float(record[metric]) for metric in METRICS},
                "median_separation_ratio": float(record["median_separation_ratio"]),
                "weight_sum": float(weights.sum()),
                "paired_noise_digest": record["paired_noise_digest"],
                "final_weights": ";".join(f"{value:.8f}" for value in weights),
            })
        digests = {record["paired_noise_digest"] for record in records.values()}
        if len(digests) != 1:
            errors.append(f"unpaired noise seed={seed}: {sorted(digests)}")

    if not rows:
        raise ValueError("no completed rows")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "run_source_data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        item: dict[str, object] = {"method": method, "seeds": len(selected)}
        for metric in METRICS:
            values = np.asarray([row[metric] for row in selected], dtype=float)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(item)
    with (args.output / "summary_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    paired = []
    for method in ("BMA", "PCE", "APCE"):
        for seed in sorted({int(row["seed"]) for row in rows}):
            baseline = next(row for row in rows if row["seed"] == seed and row["method"] == "DEnKF")
            target = next(row for row in rows if row["seed"] == seed and row["method"] == method)
            paired.append({
                "seed": seed,
                "method": method,
                "analysis_nrmse_change_vs_denkf": float(target["interior_analysis_nrmse"] - baseline["interior_analysis_nrmse"]),
                "coverage_change_vs_denkf": float(target["interior_coverage_90"] - baseline["interior_coverage_90"]),
                "forecast_4ms_nrmse_change_vs_denkf": float(target["interior_forecast_4ms_nrmse"] - baseline["interior_forecast_4ms_nrmse"]),
            })
    with (args.output / "paired_comparisons.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)

    report = {
        "case": "meshir_s1_reconstruction_192_stage_d_rbf_v3",
        "errors": errors,
        "passed": not errors,
        "completed_runs": len(rows),
        "summary": summary,
        "mean_paired_changes_vs_denkf": {
            method: {
                key: float(np.mean([row[key] for row in paired if row["method"] == method]))
                for key in (
                    "analysis_nrmse_change_vs_denkf",
                    "coverage_change_vs_denkf",
                    "forecast_4ms_nrmse_change_vs_denkf",
                )
            }
            for method in ("BMA", "PCE", "APCE")
        },
        "manuscript_modified": False,
    }
    (args.output / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
