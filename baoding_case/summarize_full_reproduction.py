#!/usr/bin/env python3
"""Summarize the Baoding full-reproduction archive without changing runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


TARGETS = (1, 2, 3)
METHODS = ("pce", "apce")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def percentile(values, q: float):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    index = min(len(values) - 1, max(0, int(q * (len(values) - 1))))
    return values[index]


def run_rows(root: Path, method: str, target: int) -> list[dict]:
    rows = []
    for path in sorted((root / method / f"target{target}" / "runs").glob(f"{method}_seed_*.json")):
        if not re.fullmatch(rf"{method}_seed_\d{{10}}\.json", path.name):
            continue
        payload = load(path)
        records = [row for row in payload.get("records", []) if row.get("position_error_m") is not None]
        if not records:
            continue
        errors = [float(row["position_error_m"]) for row in records]
        crps = [float(row["crps_position_m"]) for row in records if row.get("crps_position_m") is not None]
        coverage = [float(row["coverage_90"]) for row in records if row.get("coverage_90") is not None]
        widths = [float(row["interval_width_m"]) for row in records if row.get("interval_width_m") is not None]
        rows.append(
            {
                "method": method,
                "target": target,
                "seed": payload.get("seed"),
                "n_records": len(records),
                "rmse_m": math.sqrt(sum(value * value for value in errors) / len(errors)),
                "mae_m": sum(errors) / len(errors),
                "p90_m": percentile(errors, 0.9),
                "max_m": max(errors),
                "crps_m": mean(crps),
                "coverage_90": mean(coverage),
                "interval_width_m": mean(widths),
                "valid_records": len(records),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root

    metrics = []
    target_method = {}
    for method in METHODS:
        for target in TARGETS:
            rows = run_rows(root, method, target)
            target_method[(method, target)] = rows
            metrics.extend(rows)

    out = root / "metrics"
    out.mkdir(parents=True, exist_ok=True)

    if metrics:
        with (out / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metrics[0].keys()))
            writer.writeheader()
            writer.writerows(metrics)
    else:
        (out / "per_seed_metrics.csv").write_text("", encoding="utf-8")

    summary = []
    for method in METHODS:
        for target in TARGETS:
            rows = target_method[(method, target)]
            if not rows:
                continue
            summary.append(
                {
                    "method": method,
                    "target": target,
                    "seeds": len(rows),
                    "rmse_m": mean([row["rmse_m"] for row in rows]),
                    "mae_m": mean([row["mae_m"] for row in rows]),
                    "p90_m": mean([row["p90_m"] for row in rows]),
                    "max_m": mean([row["max_m"] for row in rows]),
                    "crps_m": mean([row["crps_m"] for row in rows]),
                    "coverage_90": mean([row["coverage_90"] for row in rows]),
                    "interval_width_m": mean([row["interval_width_m"] for row in rows]),
                    "valid_records_mean": mean([row["valid_records"] for row in rows]),
                }
            )

    if summary:
        with (out / "method_target_summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)
    else:
        (out / "method_target_summary.csv").write_text("", encoding="utf-8")

    ospa_rows = []
    for method in METHODS:
        seed_sets = []
        for target in TARGETS:
            seed_sets.append({row["seed"] for row in target_method[(method, target)] if row.get("seed") is not None})
        if seed_sets and all(seed_sets):
            common = set.intersection(*seed_sets)
            if common:
                ospa_rows.append(
                    {
                        "method": method,
                        "common_seed_count": len(common),
                        "diagnostic_ospa_note": "Equal-cardinality diagnostic only; not the paper's private OSPA implementation.",
                    }
                )

    (out / "ospa_diagnostic.json").write_text(
        json.dumps({"claim_status": "diagnostic_only", "rows": ospa_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "claim_status": "pce_apce_diagnostic_summary",
        "source_root": str(root),
        "methods": METHODS,
        "targets": TARGETS,
        "seed_count_by_method_target": {
            f"{method}_t{target}": len(target_method[(method, target)])
            for method in METHODS
            for target in TARGETS
        },
        "independent_observation": True,
        "dbn_derived_observation": False,
        "formal_admission": False,
        "reason": "three-source association frontend gate failed; summary is diagnostic until target association is formally admitted.",
        "files": {
            "per_seed_metrics": str(out / "per_seed_metrics.csv"),
            "method_target_summary": str(out / "method_target_summary.csv"),
            "ospa_diagnostic": str(out / "ospa_diagnostic.json"),
        },
    }
    (out / "metrics_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
