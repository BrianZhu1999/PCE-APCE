from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


CASES = ("wave", "spring", "heat")
METHODS = ("denkf", "letkf", "iensf", "pce", "apce", "oracle_alpha")
METRICS = ("nrmse", "rmse", "crps", "coverage_90", "interval_width_90", "runtime_seconds", "alpha_absolute_error")


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, resamples: int = 10000) -> tuple[float, float]:
    if values.size < 2:
        value = float(values.mean())
        return value, value
    draws = values[rng.integers(0, values.size, size=(resamples, values.size))].mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def paired_permutation_p(first: np.ndarray, second: np.ndarray, rng: np.random.Generator, draws: int = 20000) -> float:
    delta = first - second
    observed = abs(float(delta.mean()))
    if delta.size == 0:
        return 1.0
    signs = rng.choice(np.array([-1.0, 1.0]), size=(draws, delta.size))
    null = np.abs((signs * delta[None, :]).mean(axis=1))
    return float((1.0 + np.count_nonzero(null >= observed)) / (draws + 1.0))


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(np.asarray(p_values))
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    n = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (n - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def read_rows(result_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(result_root.glob("figure2_*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if row.get("status") != "completed" or not row.get("valid", False):
            continue
        if row.get("case") not in CASES or row.get("method") not in METHODS:
            continue
        row["_path"] = str(path)
        rows.append(row)
    return rows


def metric_value(row: dict[str, Any], metric: str) -> float:
    if metric == "alpha_absolute_error" and metric not in row:
        if row.get("alpha_estimate") is not None and row.get("alpha_true") is not None:
            return abs(float(row["alpha_estimate"]) - float(row["alpha_true"]))
        if row.get("fixed_alpha") is not None and row.get("alpha_true") is not None:
            return abs(float(row["fixed_alpha"]) - float(row["alpha_true"]))
        return float("nan")
    return float(row[metric])


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Figure 2 formal 50-seed outputs.")
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026080707)
    args = parser.parse_args()

    rows = read_rows(args.result_root)
    if not rows:
        raise SystemExit("No completed valid Figure 2 outputs found.")
    rng = np.random.default_rng(args.seed)
    summary: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["case"], row["method"]), []).append(row)

    for case in CASES:
        for method in METHODS:
            subset = sorted(grouped.get((case, method), []), key=lambda item: int(item["seed"]))
            if not subset:
                continue
            item: dict[str, Any] = {
                "case": case,
                "method": method,
                "label": subset[0].get("label", method),
                "n": len(subset),
                "seeds": ",".join(str(int(row["seed"])) for row in subset),
            }
            for metric in METRICS:
                values = np.asarray([metric_value(row, metric) for row in subset], dtype=float)
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                item[f"{metric}_mean"] = float(values.mean())
                item[f"{metric}_sd"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
                lo, hi = bootstrap_ci(values, rng, args.bootstrap_resamples)
                item[f"{metric}_ci95_low"] = lo
                item[f"{metric}_ci95_high"] = hi
            summary.append(item)

    paired: list[dict[str, Any]] = []
    for case in CASES:
        by_method = {
            method: {int(row["seed"]): row for row in grouped.get((case, method), [])}
            for method in METHODS
        }
        common = sorted(set.intersection(*(set(values) for values in by_method.values() if values)))
        for baseline in ("denkf", "letkf", "iensf"):
            if not common or not by_method["apce"] or not by_method[baseline]:
                continue
            p_values: list[float] = []
            entries: list[dict[str, Any]] = []
            for metric in ("nrmse", "crps", "coverage_90", "interval_width_90", "runtime_seconds"):
                first = np.asarray([float(by_method["apce"][seed][metric]) for seed in common])
                second = np.asarray([float(by_method[baseline][seed][metric]) for seed in common])
                delta = first - second
                lo, hi = bootstrap_ci(delta, rng, args.bootstrap_resamples)
                p = paired_permutation_p(first, second, rng)
                p_values.append(p)
                entries.append({
                    "case": case,
                    "reference": "apce",
                    "baseline": baseline,
                    "metric": metric,
                    "n": len(common),
                    "mean_difference_apce_minus_baseline": float(delta.mean()),
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "p_raw": p,
                    "paired_seeds": ",".join(str(seed) for seed in common),
                })
            adjusted = holm_adjust(p_values)
            for entry, p_adj in zip(entries, adjusted, strict=True):
                entry["p_holm"] = p_adj
                paired.append(entry)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "figure2_formal_aggregate.json").write_text(
        json.dumps(
            {
                "protocol": "figure2-formal-50paired-seeds-20260807-v1",
                "result_root": str(args.result_root),
                "total_completed_valid": len(rows),
                "summary": summary,
                "paired_comparisons": paired,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with (args.output / "figure2_method_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = list(summary[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    with (args.output / "figure2_paired_comparisons.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = list(paired[0].keys()) if paired else ["case", "reference", "baseline", "metric"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(paired)
    with (args.output / "figure2_run_source_data.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = sorted({key for row in rows for key in row if not key.startswith("_")})
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"completed_valid": len(rows), "summary_rows": len(summary), "paired_rows": len(paired)}))


if __name__ == "__main__":
    main()
