from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


CASES = ("wave", "spring", "heat")
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
    "coverage_error_90",
    "interval_width_90",
    "alpha_absolute_error",
    "runtime_seconds",
    "forward_member_steps",
    "peak_gpu_memory_mb",
)
LOWER_IS_BETTER = {
    "nrmse",
    "rmse",
    "crps",
    "coverage_error_90",
    "interval_width_90",
    "alpha_absolute_error",
    "runtime_seconds",
    "forward_member_steps",
    "peak_gpu_memory_mb",
}


def numeric(value: Any) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return output if math.isfinite(output) else float("nan")


def valid_row(row: dict[str, Any]) -> bool:
    return (
        str(row.get("status", "completed")) == "completed"
        and str(row.get("valid", "")).lower() in {"true", "1", "yes"}
        and str(row.get("case", "")) in CASES
        and str(row.get("method", "")) in METHODS
    )


def read_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("seed_*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        row["_path"] = str(path)
        if valid_row(row):
            if row.get("alpha_absolute_error") is None:
                row["alpha_absolute_error"] = ""
            row["label"] = LABELS.get(str(row.get("method")), str(row.get("method")))
            row["coverage_error_90"] = abs(numeric(row.get("coverage_90")) - 0.90)
            rows.append(row)
    return rows


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
    finite = np.isfinite(first) & np.isfinite(second)
    first = first[finite]
    second = second[finite]
    if first.size == 0:
        return float("nan")
    delta = first - second
    observed = abs(float(delta.mean()))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(draws, delta.size))
    null = np.abs((signs * delta[None, :]).mean(axis=1))
    return float((1.0 + np.count_nonzero(null >= observed)) / (draws + 1.0))


def holm_adjust(p_values: list[float]) -> list[float]:
    finite = [(index, p) for index, p in enumerate(p_values) if math.isfinite(p)]
    adjusted = [float("nan")] * len(p_values)
    if not finite:
        return adjusted
    ordered = sorted(finite, key=lambda item: item[1])
    running = 0.0
    n = len(ordered)
    for rank, (index, p) in enumerate(ordered):
        value = min(1.0, (n - rank) * p)
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row if not key.startswith("_")})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
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
                "protocol": subset[0].get("protocol", ""),
                "source_hash": subset[0].get("source_hash", ""),
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
    rng = np.random.default_rng(seed + 41)
    grouped: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["case"]), str(row["method"])), {})[int(row["seed"])] = row

    output: list[dict[str, Any]] = []
    p_values_for_holm: list[float] = []
    p_value_indices: list[int] = []
    for case in CASES:
        for reference in ("pce", "apce"):
            ref = grouped.get((case, reference), {})
            if not ref:
                continue
            for competitor in METHODS:
                if competitor == reference:
                    continue
                comp = grouped.get((case, competitor), {})
                seeds = sorted(set(ref) & set(comp))
                if not seeds:
                    continue
                for metric in METRICS:
                    first = np.asarray([numeric(ref[s].get(metric, "")) for s in seeds], dtype=float)
                    second = np.asarray([numeric(comp[s].get(metric, "")) for s in seeds], dtype=float)
                    finite = np.isfinite(first) & np.isfinite(second)
                    if not finite.any():
                        continue
                    first = first[finite]
                    second = second[finite]
                    delta = first - second
                    lo, hi = bootstrap_ci(delta, rng, resamples)
                    p = paired_permutation_p(first, second, rng)
                    ref_better = bool(delta.mean() < 0.0) if metric in LOWER_IS_BETTER else bool(delta.mean() > 0.0)
                    output.append(
                        {
                            "case": case,
                            "reference": reference,
                            "competitor": competitor,
                            "metric": metric,
                            "n": int(first.size),
                            "reference_mean": float(first.mean()),
                            "competitor_mean": float(second.mean()),
                            "mean_difference_reference_minus_competitor": float(delta.mean()),
                            "relative_change_reference_vs_competitor_percent": float(100.0 * delta.mean() / max(abs(second.mean()), 1.0e-30)),
                            "ci95_low": lo,
                            "ci95_high": hi,
                            "p_raw": p,
                            "reference_better": ref_better,
                            "paired_seeds": ",".join(str(seed_value) for seed_value in seeds),
                        }
                    )
                    p_values_for_holm.append(p)
                    p_value_indices.append(len(output) - 1)
    for index, p_adj in zip(p_value_indices, holm_adjust(p_values_for_holm), strict=True):
        output[index]["p_holm"] = p_adj
    return output


def audit(rows: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    expected = {(case, method, seed) for case in CASES for method in METHODS for seed in range(2026080700, 2026080750)}
    present = {(str(row["case"]), str(row["method"]), int(row["seed"])) for row in rows}
    failed = []
    for path in sorted(root.rglob("seed_*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if row.get("status") == "failed" or str(row.get("valid", "")).lower() not in {"true", "1", "yes"}:
            failed.append(
                {
                    "path": str(path),
                    "case": row.get("case"),
                    "method": row.get("method"),
                    "seed": row.get("seed"),
                    "status": row.get("status"),
                    "valid": row.get("valid"),
                    "error_type": row.get("error_type"),
                    "error": row.get("error"),
                }
            )
    return {
        "expected_rows": len(expected),
        "valid_rows": len(rows),
        "missing": [
            {"case": case, "method": method, "seed": seed}
            for case, method, seed in sorted(expected - present)
        ],
        "failed_or_invalid": failed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate corrected-score Figure 2 formal outputs.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026081117)
    args = parser.parse_args()

    rows = read_rows(args.root)
    if not rows:
        raise SystemExit("No completed valid corrected Figure 2 rows found.")
    summary = summarize(rows, args.bootstrap_resamples, args.seed)
    paired_rows = paired(rows, args.bootstrap_resamples, args.seed)
    audit_payload = audit(rows, args.root)

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "figure2_corrected_formal_run_source_data.csv", rows)
    write_csv(args.output / "figure2_corrected_formal_method_summary.csv", summary)
    write_csv(args.output / "figure2_corrected_formal_paired_comparisons.csv", paired_rows)
    (args.output / "figure2_corrected_formal_aggregate.json").write_text(
        json.dumps(
            {
                "protocol": "figure2-corrected-dimension-score-50paired-seeds-20260811",
                "root": str(args.root),
                "audit": audit_payload,
                "summary": summary,
                "paired_comparisons": paired_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "summary_rows": len(summary),
                "paired_rows": len(paired_rows),
                "missing": len(audit_payload["missing"]),
                "failed_or_invalid": len(audit_payload["failed_or_invalid"]),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
