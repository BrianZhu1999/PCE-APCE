from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


CASES = ("wave", "spring", "heat")
OLD_METHODS = ("denkf", "letkf", "iensf")
NEW_METHOD_MAP = {
    "aug_enkf": "aug_enkf",
    "bma_static": "bma_static",
    "pce_refined_v2": "pce",
    "apce_refined_v2": "apce",
}
METHODS = ("denkf", "letkf", "iensf", "aug_enkf", "bma_static", "pce", "apce")
LABELS = {
    "denkf": "DEnKF",
    "letkf": "LETKF",
    "iensf": "IEnSF",
    "aug_enkf": "Aug-EnKF",
    "bma_static": "BMA",
    "pce": "PCE",
    "apce": "APCE",
}
METRICS = (
    "nrmse",
    "rmse",
    "crps",
    "coverage_90",
    "interval_width_90",
    "runtime_seconds",
    "forward_member_steps",
    "alpha_estimate",
    "alpha_absolute_error",
)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_json_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("workers/gpu*/*/*/seed_*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append(row)
    return rows


def numeric(value: Any) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return output if math.isfinite(output) else float("nan")


def valid_row(row: dict[str, Any]) -> bool:
    return str(row.get("valid", "")).lower() in {"true", "1", "yes"} and str(row.get("status", "completed")) == "completed"


def normalize_old_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        case = str(row.get("case", ""))
        method = str(row.get("method", ""))
        if case not in CASES or method not in OLD_METHODS:
            continue
        if not valid_row(row):
            continue
        copied = dict(row)
        copied["method"] = method
        copied["label"] = LABELS[method]
        copied["seed"] = int(float(copied["seed"]))
        copied["source_layer"] = "formal_20260807_classic_baselines"
        copied["method_role"] = "classic_training_free_baseline"
        copied["alpha_estimate"] = ""
        copied["alpha_absolute_error"] = ""
        output.append(copied)
    return output


def normalize_new_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        case = str(row.get("case", ""))
        raw_method = str(row.get("method", ""))
        if case not in CASES or raw_method not in NEW_METHOD_MAP:
            continue
        if not valid_row(row):
            continue
        method = NEW_METHOD_MAP[raw_method]
        copied = dict(row)
        copied["raw_method"] = raw_method
        copied["method"] = method
        copied["label"] = LABELS[method]
        copied["seed"] = int(float(copied["seed"]))
        copied["source_layer"] = "formal_20260810_v2_local_mixture"
        copied["method_role"] = "reviewer_risk_baseline" if method in {"aug_enkf", "bma_static"} else "main_method_final_implementation"
        output.append(copied)
    return output


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, resamples: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        value = float(values[0])
        return value, value
    draws = values[rng.integers(0, values.size, size=(resamples, values.size))].mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def paired_permutation_p(first: np.ndarray, second: np.ndarray, rng: np.random.Generator, draws: int = 20000) -> float:
    delta = first - second
    delta = delta[np.isfinite(delta)]
    if delta.size == 0:
        return float("nan")
    observed = abs(float(delta.mean()))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(draws, delta.size))
    null = np.abs((signs * delta[None, :]).mean(axis=1))
    return float((1.0 + np.count_nonzero(null >= observed)) / (draws + 1.0))


def holm_adjust(p_values: list[float]) -> list[float]:
    finite = [(i, p) for i, p in enumerate(p_values) if math.isfinite(p)]
    adjusted = [float("nan")] * len(p_values)
    if not finite:
        return adjusted
    order = sorted(finite, key=lambda item: item[1])
    running = 0.0
    n = len(order)
    for rank, (index, p) in enumerate(order):
        value = min(1.0, (n - rank) * p)
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]], resamples: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["case"]), str(row["method"])), []).append(row)
    summary: list[dict[str, Any]] = []
    for case in CASES:
        for method in METHODS:
            subset = sorted(grouped.get((case, method), []), key=lambda item: int(item["seed"]))
            if not subset:
                continue
            item: dict[str, Any] = {
                "case": case,
                "method": method,
                "label": LABELS[method],
                "n": len(subset),
                "seeds": ",".join(str(int(row["seed"])) for row in subset),
                "source_layer": subset[0].get("source_layer", ""),
                "method_role": subset[0].get("method_role", ""),
            }
            for metric in METRICS:
                values = np.asarray([numeric(row.get(metric, "")) for row in subset], dtype=float)
                values = values[np.isfinite(values)]
                if values.size == 0:
                    item[f"{metric}_mean"] = ""
                    item[f"{metric}_sd"] = ""
                    item[f"{metric}_ci95_low"] = ""
                    item[f"{metric}_ci95_high"] = ""
                    continue
                lo, hi = bootstrap_ci(values, rng, resamples)
                item[f"{metric}_mean"] = float(values.mean())
                item[f"{metric}_sd"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
                item[f"{metric}_ci95_low"] = lo
                item[f"{metric}_ci95_high"] = hi
            summary.append(item)
    return summary


def paired(rows: list[dict[str, Any]], resamples: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed + 17)
    grouped: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["case"]), str(row["method"])), {})[int(row["seed"])] = row
    output: list[dict[str, Any]] = []
    for case in CASES:
        apce = grouped.get((case, "apce"), {})
        if not apce:
            continue
        for baseline in [method for method in METHODS if method != "apce"]:
            comp = grouped.get((case, baseline), {})
            seeds = sorted(set(apce) & set(comp))
            entries: list[dict[str, Any]] = []
            p_values: list[float] = []
            for metric in ("nrmse", "rmse", "crps", "coverage_90", "interval_width_90", "runtime_seconds", "alpha_absolute_error"):
                pairs = [
                    (numeric(apce[s].get(metric, "")), numeric(comp[s].get(metric, "")))
                    for s in seeds
                ]
                pairs = [(a, b) for a, b in pairs if math.isfinite(a) and math.isfinite(b)]
                if not pairs:
                    continue
                first = np.asarray([a for a, _ in pairs], dtype=float)
                second = np.asarray([b for _, b in pairs], dtype=float)
                delta = first - second
                lo, hi = bootstrap_ci(delta, rng, resamples)
                p = paired_permutation_p(first, second, rng)
                p_values.append(p)
                entries.append(
                    {
                        "case": case,
                        "reference": "apce",
                        "baseline": baseline,
                        "metric": metric,
                        "n": len(pairs),
                        "mean_difference_apce_minus_baseline": float(delta.mean()),
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "p_raw": p,
                        "paired_seeds": ",".join(str(s) for s in seeds),
                    }
                )
            for entry, p_adj in zip(entries, holm_adjust(p_values), strict=True):
                entry["p_holm"] = p_adj
                output.append(entry)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Figure 2 final-method source data from old formal baselines and V2 formal methods.")
    parser.add_argument("--old-runs", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026081017)
    args = parser.parse_args()

    rows = normalize_old_rows(read_csv_rows(args.old_runs))
    rows.extend(normalize_new_rows(read_json_rows(args.new_root)))
    if not rows:
        raise SystemExit("No valid rows found.")
    summary = summarize(rows, args.bootstrap_resamples, args.seed)
    paired_rows = paired(rows, args.bootstrap_resamples, args.seed)

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "figure2_v2_main_run_source_data.csv", rows)
    write_csv(args.output / "figure2_v2_main_method_summary.csv", summary)
    write_csv(args.output / "figure2_v2_main_paired_comparisons.csv", paired_rows)
    (args.output / "figure2_v2_main_aggregate.json").write_text(
        json.dumps(
            {
                "cases": CASES,
                "methods": METHODS,
                "method_mapping": NEW_METHOD_MAP,
                "old_runs": str(args.old_runs),
                "new_root": str(args.new_root),
                "rows": len(rows),
                "summary_rows": len(summary),
                "paired_rows": len(paired_rows),
                "note": "Main-figure PCE/APCE labels use the refined V2 local-mixture implementation; the label is intentionally kept as PCE/APCE in the manuscript.",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(rows), "summary_rows": len(summary), "paired_rows": len(paired_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
