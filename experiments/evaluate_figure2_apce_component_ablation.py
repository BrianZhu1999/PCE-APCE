from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


CASES = ("wave", "spring", "heat")
METHODS = (
    "pce_baseline",
    "apce_full",
    "apce_no_dim",
    "apce_fixed_temp",
    "apce_no_forgetting",
    "apce_no_entropy",
)
COMPONENT_ABLATIONS = (
    "apce_no_dim",
    "apce_fixed_temp",
    "apce_no_forgetting",
    "apce_no_entropy",
)
DISPLAY_METRICS = (
    "nrmse",
    "crps",
    "coverage_90",
    "coverage_error_90",
    "interval_width_90",
    "alpha_absolute_error",
    "mechanism_mean_normalized_entropy",
    "early_collapse_rate_proxy",
)
PRIMARY_METRICS = ("nrmse", "crps", "alpha_absolute_error")
ROUNDING_TOLERANCE = 1.0e-12
LABELS = {
    "pce_baseline": r"\PCE{}",
    "apce_full": r"\APCE{} (full)",
    "apce_no_dim": r"\APCE{}--no dim.",
    "apce_fixed_temp": r"\APCE{}--fixed temp.",
    "apce_no_forgetting": r"\APCE{}--no forgetting",
    "apce_no_entropy": r"\APCE{}--no entropy floor",
}


def number(row: dict[str, Any], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    samples = values[rng.integers(0, values.size, size=(draws, values.size))].mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def paired_permutation_p(delta: np.ndarray, rng: np.random.Generator, draws: int) -> float:
    delta = delta[np.isfinite(delta)]
    if delta.size == 0:
        return float("nan")
    observed = abs(float(delta.mean()))
    signs = rng.choice(np.array((-1.0, 1.0)), size=(draws, delta.size))
    null = np.abs((signs * delta[None, :]).mean(axis=1))
    return float((1 + np.count_nonzero(null >= observed)) / (draws + 1))


def holm_adjust(p_values: list[float]) -> list[float]:
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [float("nan")] * len(p_values)
    running = 0.0
    for rank, (index, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p_value))
        adjusted[index] = running
    return adjusted


def read_rows(
    path: Path, seed_base: int, seed_count: int
) -> tuple[dict[tuple[str, str, int], dict[str, Any]], dict[str, Any]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    expected = {
        (case, method, seed)
        for case in CASES
        for method in METHODS
        for seed in range(seed_base, seed_base + seed_count)
    }
    indexed: dict[tuple[str, str, int], dict[str, Any]] = {}
    nonfinite: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row.get("case")), str(row.get("method")), int(row.get("seed", -1)))
        if key in indexed:
            duplicates.append({"case": key[0], "method": key[1], "seed": key[2]})
            continue
        indexed[key] = row
        for metric in DISPLAY_METRICS:
            if not math.isfinite(number(row, metric)):
                nonfinite.append({"case": key[0], "method": key[1], "seed": key[2], "metric": metric})
    observed = set(indexed)
    assets_by_case_seed: dict[tuple[str, int], set[str]] = {}
    for (case, _method, seed), row in indexed.items():
        assets_by_case_seed.setdefault((case, seed), set()).add(str(row.get("common_asset_sha256", "")))
    asset_mismatches = [
        {"case": case, "seed": seed, "asset_hashes": sorted(asset_hashes)}
        for (case, seed), asset_hashes in sorted(assets_by_case_seed.items())
        if len(asset_hashes) != 1
    ]
    audit = {
        "expected_rows": len(expected),
        "observed_rows": len(rows),
        "unique_rows": len(indexed),
        "missing": [
            {"case": case, "method": method, "seed": seed}
            for case, method, seed in sorted(expected - observed)
        ],
        "unexpected": [
            {"case": case, "method": method, "seed": seed}
            for case, method, seed in sorted(observed - expected)
        ],
        "duplicates": duplicates,
        "nonfinite": nonfinite,
        "common_asset_groups": len(assets_by_case_seed),
        "common_asset_mismatches_by_case_seed": asset_mismatches,
        "source_hash": sorted({row.get("source_hash", "") for row in indexed.values()}),
    }
    if audit["missing"] or audit["unexpected"] or duplicates or nonfinite or asset_mismatches:
        raise RuntimeError(f"Input audit failed: {json.dumps(audit, ensure_ascii=False)}")
    return indexed, audit


def summarize(
    indexed: dict[tuple[str, str, int], dict[str, Any]],
    seed_base: int,
    seed_count: int,
    rng: np.random.Generator,
    draws: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for case in CASES:
        for method in METHODS:
            rows = [indexed[(case, method, seed)] for seed in range(seed_base, seed_base + seed_count)]
            item: dict[str, Any] = {"case": case, "method": method, "label": LABELS[method], "n": len(rows)}
            for metric in DISPLAY_METRICS:
                values = np.asarray([number(row, metric) for row in rows], dtype=float)
                item[f"{metric}_mean"] = float(values.mean())
                item[f"{metric}_ci95"] = bootstrap_ci(values, rng, draws)
            output.append(item)
    return output


def paired_primary(
    indexed: dict[tuple[str, str, int], dict[str, Any]],
    seed_base: int,
    seed_count: int,
    rng: np.random.Generator,
    bootstrap_draws: int,
    permutation_draws: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for metric in PRIMARY_METRICS:
        metric_rows: list[dict[str, Any]] = []
        for case in CASES:
            for competitor in COMPONENT_ABLATIONS:
                full = np.asarray(
                    [number(indexed[(case, "apce_full", seed)], metric) for seed in range(seed_base, seed_base + seed_count)],
                    dtype=float,
                )
                ablated = np.asarray(
                    [number(indexed[(case, competitor, seed)], metric) for seed in range(seed_base, seed_base + seed_count)],
                    dtype=float,
                )
                delta = full - ablated
                # The Spring score has one observed component, so this ablation is
                # algebraically identical to full APCE. Do not test roundoff noise.
                delta[np.abs(delta) < ROUNDING_TOLERANCE] = 0.0
                ci_low, ci_high = bootstrap_ci(delta, rng, bootstrap_draws)
                metric_rows.append(
                    {
                        "case": case,
                        "reference": "apce_full",
                        "competitor": competitor,
                        "metric": metric,
                        "n": int(delta.size),
                        "full_mean": float(full.mean()),
                        "ablation_mean": float(ablated.mean()),
                        "mean_full_minus_ablation": float(delta.mean()),
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "p_raw": paired_permutation_p(delta, rng, permutation_draws),
                    }
                )
        adjusted = holm_adjust([row["p_raw"] for row in metric_rows])
        for row, p_holm in zip(metric_rows, adjusted, strict=True):
            row["p_holm_within_metric"] = p_holm
        output.extend(metric_rows)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rendered_value(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def write_table(path: Path, summary: list[dict[str, Any]], provenance: dict[str, Any]) -> None:
    by_key = {(row["case"], row["method"]): row for row in summary}
    lines = [
        "% Generated by evaluate_figure2_apce_component_ablation.py.",
        f"% Remote formal result root: {provenance['remote_result_root']}",
        f"% Aggregate CSV SHA-256: {provenance['input_sha256']}",
        f"% Evaluator SHA-256: {provenance['evaluator_sha256']}",
        r"{\scriptsize",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\begin{longtable}{llrrrrrrr}",
        r"\caption{\textbf{APCE component ablation under the Figure~2 protocol.} Values are means across 50 paired seeds. All APCE rows use the same shadow-evidence source, candidate grid, coarse-to-local protocol, analysis filter, random assets and forward-member budget; the indicated component alone is removed or fixed. Lower nRMSE, CRPS, coverage error, interval width and alpha error are better. \(\bar H\) is mean normalized path-weight entropy.}\label{tab:apce-component-ablation}\\",
        r"\toprule",
        r"System & Variant & nRMSE & CRPS $\times10^3$ & Cov. & Cov. err. & Width $\times10^3$ & $|\hat\alpha-\alpha^\star|$ & $\bar H$ \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"System & Variant & nRMSE & CRPS $\times10^3$ & Cov. & Cov. err. & Width $\times10^3$ & $|\hat\alpha-\alpha^\star|$ & $\bar H$ \\",
        r"\midrule",
        r"\endhead",
    ]
    for case_index, case in enumerate(CASES):
        for method in METHODS:
            row = by_key[(case, method)]
            values = [
                rendered_value(row["nrmse_mean"], 5),
                rendered_value(1000 * row["crps_mean"], 3),
                rendered_value(row["coverage_90_mean"], 3),
                rendered_value(row["coverage_error_90_mean"], 3),
                rendered_value(1000 * row["interval_width_90_mean"], 3),
                rendered_value(row["alpha_absolute_error_mean"], 4),
                rendered_value(row["mechanism_mean_normalized_entropy_mean"], 3),
            ]
            lines.append(
                f"{case.title() if method == METHODS[0] else ''} & {LABELS[method]} & "
                + " & ".join(values)
                + r" \\"
            )
        if case_index < len(CASES) - 1:
            lines.append(r"\addlinespace[2pt]")
    lines.extend([r"\bottomrule", r"\end{longtable}", r"}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the formal Figure 2 APCE component-ablation matrix.")
    parser.add_argument("--root", type=Path, required=True, help="Formal remote result directory.")
    parser.add_argument("--output", type=Path, required=True, help="Directory for derived audit files.")
    parser.add_argument("--table-output", type=Path, required=True, help="Generated Supplementary LaTeX table.")
    parser.add_argument("--seed-base", type=int, default=2026080700)
    parser.add_argument("--seed-count", type=int, default=50)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--permutation-draws", type=int, default=200000)
    parser.add_argument("--rng-seed", type=int, default=2026082501)
    args = parser.parse_args()

    input_path = args.root / "aggregate" / "figure2_corrected_formal_run_source_data.csv"
    indexed, audit = read_rows(input_path, args.seed_base, args.seed_count)
    rng = np.random.default_rng(args.rng_seed)
    summary = summarize(indexed, args.seed_base, args.seed_count, rng, args.bootstrap_draws)
    paired = paired_primary(
        indexed,
        args.seed_base,
        args.seed_count,
        rng,
        args.bootstrap_draws,
        args.permutation_draws,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "protocol": "figure2-apce-component-ablation-20260825",
        "remote_result_root": str(args.root),
        "input_path": str(input_path),
        "input_sha256": sha256(input_path),
        "evaluator_path": str(Path(__file__).resolve()),
        "evaluator_sha256": sha256(Path(__file__).resolve()),
        "bootstrap_draws": args.bootstrap_draws,
        "permutation_draws": args.permutation_draws,
        "paired_rounding_tolerance": ROUNDING_TOLERANCE,
        "holm_families": {
            metric: "12 full-APCE versus component-ablation comparisons across 3 systems x 4 ablations"
            for metric in PRIMARY_METRICS
        },
        "audit": audit,
    }
    write_csv(args.output / "component_ablation_summary.csv", summary)
    write_csv(args.output / "component_ablation_primary_paired_holm.csv", paired)
    (args.output / "component_ablation_statistics.json").write_text(
        json.dumps({"provenance": provenance, "summary": summary, "paired_primary": paired}, indent=2),
        encoding="utf-8",
    )
    args.table_output.parent.mkdir(parents=True, exist_ok=True)
    write_table(args.table_output, summary, provenance)
    print(json.dumps({"audit": audit, "summary_rows": len(summary), "paired_rows": len(paired), "table": str(args.table_output)}, indent=2))


if __name__ == "__main__":
    main()
