from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.aggregate_formal_results import aggregate_results, write_outputs
from hilda_da.metrics import paired_bootstrap_ci, paired_effect_size


LOWER_IS_BETTER = {"state_rmse", "observation_rmse", "crps", "energy_score"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate the paired HILDA advantage diagnostic")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260805)
    return parser.parse_args()


def _seed(base: int, *parts: str) -> int:
    digest = hashlib.sha256("\x00".join(parts).encode("utf-8")).digest()
    return (base + int.from_bytes(digest[:8], "big")) % (2**63 - 1)


def _effect(first: torch.Tensor, second: torch.Tensor) -> tuple[float | None, str]:
    if first.numel() < 2:
        return None, "insufficient_pairs"
    value = float(paired_effect_size(first, second))
    if math.isinf(value):
        return None, "positive_infinity" if value > 0 else "negative_infinity"
    return value, "finite"


def _condition_ids(aggregate: dict[str, Any]) -> dict[tuple[str, str], str]:
    labels: dict[tuple[str, str], str] = {}
    per_system: dict[str, list[dict[str, Any]]] = {}
    for condition in aggregate["conditions"]:
        per_system.setdefault(condition["system"], []).append(condition)
    for system, conditions in per_system.items():
        if len(conditions) != 2:
            raise ValueError(f"{system} must have exactly two conditions")
        for condition in conditions:
            parameters = condition["parameters"]
            true_alpha = float(parameters["alpha_true"])
            fixed_alpha = float(parameters["fixed_model_alpha"])
            label = "matched" if true_alpha == fixed_alpha else "misspecified"
            key = (system, label)
            if key in labels:
                raise ValueError(f"{system} has duplicate {label} conditions")
            labels[key] = condition["condition_id"]
        if (system, "matched") not in labels or (system, "misspecified") not in labels:
            raise ValueError(f"{system} needs one matched and one misspecified condition")
    return labels


def analyze_advantage(
    aggregate: dict[str, Any], *, bootstrap_resamples: int, bootstrap_seed: int
) -> dict[str, Any]:
    labels = _condition_ids(aggregate)
    values: dict[tuple[str, str, str, int, str], float] = {}
    units: dict[str, str] = {}
    condition_by_id = {
        (system, condition_id): label
        for (system, label), condition_id in labels.items()
    }
    for row in aggregate["run_metrics"]:
        label = condition_by_id[(row["system"], row["condition_id"])]
        values[(row["system"], label, row["method"], row["seed"], row["metric"])] = row["value"]
        units[row["metric"]] = row["unit"]

    systems = sorted({row["system"] for row in aggregate["run_metrics"]})
    methods = sorted({row["method"] for row in aggregate["run_metrics"]})
    metrics = sorted({row["metric"] for row in aggregate["run_metrics"]})
    deltas: list[dict[str, Any]] = []
    delta_samples: dict[tuple[str, str, str], dict[int, float]] = {}
    for system in systems:
        for method in methods:
            for metric in metrics:
                matched = {
                    seed: value
                    for (candidate_system, label, candidate_method, seed, candidate_metric), value in values.items()
                    if (candidate_system, label, candidate_method, candidate_metric)
                    == (system, "matched", method, metric)
                }
                misspecified = {
                    seed: value
                    for (candidate_system, label, candidate_method, seed, candidate_metric), value in values.items()
                    if (candidate_system, label, candidate_method, candidate_metric)
                    == (system, "misspecified", method, metric)
                }
                paired_seeds = sorted(set(matched) & set(misspecified))
                if not paired_seeds:
                    continue
                first = torch.tensor([misspecified[seed] for seed in paired_seeds], dtype=torch.float64)
                second = torch.tensor([matched[seed] for seed in paired_seeds], dtype=torch.float64)
                ci = paired_bootstrap_ci(
                    first,
                    second,
                    resamples=bootstrap_resamples,
                    seed=_seed(bootstrap_seed, system, method, metric, "delta"),
                )
                effect, effect_status = _effect(first, second)
                samples = {seed: misspecified[seed] - matched[seed] for seed in paired_seeds}
                delta_samples[(system, method, metric)] = samples
                deltas.append({
                    "system": system,
                    "method": method,
                    "metric": metric,
                    "unit": units[metric],
                    "contrast": "misspecified - matched",
                    "paired_count": len(paired_seeds),
                    "paired_seeds": paired_seeds,
                    "mean_delta": float(ci.estimate),
                    "bootstrap_95_ci_lower": float(ci.lower),
                    "bootstrap_95_ci_upper": float(ci.upper),
                    "cohen_dz": effect,
                    "cohen_dz_status": effect_status,
                })

    direct_comparisons: list[dict[str, Any]] = []
    for system in systems:
        for label in ("matched", "misspecified"):
            for method in methods:
                if method == "hilda":
                    continue
                for metric in metrics:
                    hilda = {
                        seed: value
                        for (candidate_system, candidate_label, candidate_method, seed, candidate_metric), value in values.items()
                        if (candidate_system, candidate_label, candidate_method, candidate_metric)
                        == (system, label, "hilda", metric)
                    }
                    baseline = {
                        seed: value
                        for (candidate_system, candidate_label, candidate_method, seed, candidate_metric), value in values.items()
                        if (candidate_system, candidate_label, candidate_method, candidate_metric)
                        == (system, label, method, metric)
                    }
                    paired_seeds = sorted(set(hilda) & set(baseline))
                    if not paired_seeds:
                        continue
                    first = torch.tensor([hilda[seed] for seed in paired_seeds], dtype=torch.float64)
                    second = torch.tensor([baseline[seed] for seed in paired_seeds], dtype=torch.float64)
                    ci = paired_bootstrap_ci(
                        first,
                        second,
                        resamples=bootstrap_resamples,
                        seed=_seed(bootstrap_seed, system, label, method, metric, "direct"),
                    )
                    effect, effect_status = _effect(first, second)
                    mean_baseline = float(second.mean())
                    relative = None if mean_baseline == 0.0 else float(ci.estimate) / abs(mean_baseline)
                    direct_comparisons.append({
                        "system": system,
                        "condition": label,
                        "baseline": method,
                        "metric": metric,
                        "unit": units[metric],
                        "contrast": "HILDA - baseline",
                        "paired_count": len(paired_seeds),
                        "paired_seeds": paired_seeds,
                        "mean_hilda": float(first.mean()),
                        "mean_baseline": mean_baseline,
                        "mean_difference": float(ci.estimate),
                        "relative_difference": relative,
                        "bootstrap_95_ci_lower": float(ci.lower),
                        "bootstrap_95_ci_upper": float(ci.upper),
                        "cohen_dz": effect,
                        "cohen_dz_status": effect_status,
                        "direction_favors_hilda": metric in LOWER_IS_BETTER and float(ci.estimate) < 0.0,
                        "ci_excludes_zero_in_hilda_direction": metric in LOWER_IS_BETTER and float(ci.upper) < 0.0,
                    })

    interactions: list[dict[str, Any]] = []
    for system in systems:
        for method in methods:
            if method == "hilda":
                continue
            for metric in metrics:
                hilda = delta_samples.get((system, "hilda", metric), {})
                baseline = delta_samples.get((system, method, metric), {})
                paired_seeds = sorted(set(hilda) & set(baseline))
                if not paired_seeds:
                    continue
                first = torch.tensor([hilda[seed] for seed in paired_seeds], dtype=torch.float64)
                second = torch.tensor([baseline[seed] for seed in paired_seeds], dtype=torch.float64)
                ci = paired_bootstrap_ci(
                    first,
                    second,
                    resamples=bootstrap_resamples,
                    seed=_seed(bootstrap_seed, system, method, metric, "interaction"),
                )
                effect, effect_status = _effect(first, second)
                interactions.append({
                    "system": system,
                    "baseline": method,
                    "metric": metric,
                    "unit": units[metric],
                    "contrast": "HILDA degradation - baseline degradation",
                    "paired_count": len(paired_seeds),
                    "paired_seeds": paired_seeds,
                    "mean_interaction": float(ci.estimate),
                    "bootstrap_95_ci_lower": float(ci.lower),
                    "bootstrap_95_ci_upper": float(ci.upper),
                    "cohen_dz": effect,
                    "cohen_dz_status": effect_status,
                    "direction_favors_hilda": metric in LOWER_IS_BETTER and float(ci.estimate) < 0.0,
                    "ci_excludes_zero_in_hilda_direction": metric in LOWER_IS_BETTER and float(ci.upper) < 0.0,
                })

    robustness_rows = [row for row in interactions if row["metric"] == "state_rmse"]
    mismatch_rows = [
        row for row in direct_comparisons
        if row["condition"] == "misspecified" and row["metric"] == "state_rmse"
    ]
    matched_rows = [
        row for row in direct_comparisons
        if row["condition"] == "matched" and row["metric"] == "state_rmse"
    ]
    completed = aggregate["job_counts"]["completed"]
    manifest_count = aggregate["job_counts"]["manifest"]
    absolute_wins = sum(row["direction_favors_hilda"] for row in mismatch_rows)
    if mismatch_rows and absolute_wins == len(mismatch_rows):
        status = "consistent_absolute_hilda_advantage"
    elif absolute_wins == 0:
        status = "no_absolute_hilda_advantage"
    else:
        status = "mixed_absolute_performance"
    verdict = {
        "status": status,
        "execution_complete": completed == manifest_count,
        "completed_jobs": completed,
        "manifest_jobs": manifest_count,
        "failed_jobs": aggregate["job_counts"].get("failed", 0),
        "state_rmse_absolute_baselines": len(mismatch_rows),
        "state_rmse_absolute_wins_under_misspecification": absolute_wins,
        "state_rmse_absolute_ci_confirmed_wins": sum(
            row["ci_excludes_zero_in_hilda_direction"] for row in mismatch_rows
        ),
        "state_rmse_matched_condition_wins": sum(
            row["direction_favors_hilda"] for row in matched_rows
        ),
        "state_rmse_robustness_baselines": len(robustness_rows),
        "state_rmse_robustness_directional_wins": sum(
            row["direction_favors_hilda"] for row in robustness_rows
        ),
        "state_rmse_robustness_ci_confirmed_wins": sum(
            row["ci_excludes_zero_in_hilda_direction"] for row in robustness_rows
        ),
        "interpretation_limit": "Five paired seeds are a directional diagnostic, not the frozen formal claim.",
    }
    return {
        "schema_version": 1,
        "verdict": verdict,
        "condition_deltas": deltas,
        "direct_comparisons": direct_comparisons,
        "interactions": interactions,
    }


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()})
    return output.getvalue()


def write_advantage_outputs(output_directory: Path, result: dict[str, Any]) -> list[Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "quick_advantage.json": json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        "condition_deltas.csv": _csv(result["condition_deltas"]),
        "direct_comparisons.csv": _csv(result["direct_comparisons"]),
        "hilda_interactions.csv": _csv(result["interactions"]),
    }
    written = []
    for name, content in paths.items():
        path = output_directory / name
        _atomic_write(path, content)
        written.append(path)
    return written


def main() -> None:
    args = parse_args()
    aggregate = aggregate_results(
        args.manifest,
        results_root=args.results_root,
        reference_method="hilda",
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    formal_paths = write_outputs(args.output_directory, aggregate)
    result = analyze_advantage(
        aggregate,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    advantage_paths = write_advantage_outputs(args.output_directory, result)
    print(json.dumps({
        "output_directory": str(args.output_directory),
        "verdict": result["verdict"],
        "files": [str(path) for path in (*formal_paths, *advantage_paths)],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
