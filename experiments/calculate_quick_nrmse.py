from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from hilda_da.metrics import paired_bootstrap_ci, paired_effect_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute auditable nRMSE variants from saved trajectories")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--condition", choices=("matched", "misspecified"), required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--wave-nx", type=int, default=128)
    return parser.parse_args()


def _finite_ratio(numerator: torch.Tensor, denominator: torch.Tensor, label: str) -> float:
    denominator_value = float(denominator)
    if not math.isfinite(denominator_value) or denominator_value <= 0.0:
        raise ValueError(f"{label} denominator must be positive and finite")
    value = float(numerator / denominator)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def trajectory_nrmse(path: Path, *, wave_nx: int) -> dict[str, float]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    truth = payload["truth"].to(torch.float64)
    estimate = payload["estimate"].to(torch.float64)
    if truth.ndim != 2 or estimate.ndim != 2:
        raise ValueError("truth and estimate must be time-by-state tensors")
    if truth.shape[0] == estimate.shape[0] + 1:
        truth = truth[1:]
    if truth.shape != estimate.shape or truth.numel() == 0:
        raise ValueError("truth and estimate trajectories must align and be non-empty")
    if truth.shape[1] != 2 * wave_nx:
        raise ValueError(f"expected Wave1D state dimension {2 * wave_nx}, got {truth.shape[1]}")

    displacement_error = estimate[:, :wave_nx] - truth[:, :wave_nx]
    displacement_truth = truth[:, :wave_nx]
    step_rmse = displacement_error.square().mean(dim=1).sqrt()
    step_truth_rms = displacement_truth.square().mean(dim=1).sqrt()
    displacement_rmse = displacement_error.square().mean().sqrt()
    displacement_truth_rms = displacement_truth.square().mean().sqrt()
    return {
        "primary_mean_step_rmse": float(step_rmse.mean()),
        "primary_global_rmse": float(displacement_rmse),
        "primary_mean_truth_rms": float(step_truth_rms.mean()),
        "primary_global_truth_rms": float(displacement_truth_rms),
        "primary_ratio_of_means_nrmse": _finite_ratio(
            step_rmse.mean(), step_truth_rms.mean(), "primary ratio-of-means nRMSE"
        ),
        "primary_mean_step_nrmse": float((step_rmse / step_truth_rms).mean()),
        "primary_global_relative_l2": _finite_ratio(
            displacement_rmse,
            displacement_truth_rms,
            "primary global relative L2",
        ),
    }


def _paired_primary_comparisons(
    rows: list[dict[str, Any]], condition: str
) -> list[dict[str, Any]]:
    metric = "primary_mean_step_rmse"
    by_method = {
        method: {int(row["seed"]): float(row[metric]) for row in rows if row["method"] == method}
        for method in sorted({row["method"] for row in rows})
    }
    hilda = by_method.get("hilda", {})
    comparisons = []
    for baseline, samples in by_method.items():
        if baseline == "hilda":
            continue
        seeds = sorted(set(hilda) & set(samples))
        if not seeds:
            continue
        first = torch.tensor([hilda[seed] for seed in seeds], dtype=torch.float64)
        second = torch.tensor([samples[seed] for seed in seeds], dtype=torch.float64)
        ci = paired_bootstrap_ci(first, second, resamples=10_000, seed=20260805 + len(comparisons))
        effect = float(paired_effect_size(first, second)) if len(seeds) > 1 else None
        mean_baseline = float(second.mean())
        comparisons.append({
            "condition": condition,
            "baseline": baseline,
            "metric": metric,
            "paired_count": len(seeds),
            "paired_seeds": seeds,
            "mean_hilda": float(first.mean()),
            "mean_baseline": mean_baseline,
            "mean_difference": float(ci.estimate),
            "relative_difference": float(ci.estimate) / abs(mean_baseline),
            "bootstrap_95_ci_lower": float(ci.lower),
            "bootstrap_95_ci_upper": float(ci.upper),
            "cohen_dz": effect,
            "direction_favors_hilda": float(ci.estimate) < 0.0,
        })
    return comparisons


def collect_results(results_root: Path, condition: str, wave_nx: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    pattern = f"quick_{condition}_wave_*_s*"
    for run_directory in sorted(results_root.glob(pattern)):
        provenance_path = run_directory / "provenance.json"
        trajectory_path = run_directory / "trajectories.pt"
        if not provenance_path.is_file() or not trajectory_path.is_file():
            skipped.append({"run_id": run_directory.name, "reason": "missing artifacts"})
            continue
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("completed") is not True or provenance.get("failed") is True:
            skipped.append({
                "run_id": run_directory.name,
                "reason": provenance.get("failure_message") or "run not completed",
            })
            continue
        configuration = provenance["configuration"]
        values = trajectory_nrmse(trajectory_path, wave_nx=wave_nx)
        rows.append({
            "run_id": run_directory.name,
            "method": configuration["method"],
            "seed": int(configuration["seed"]),
            **values,
        })

    summaries: list[dict[str, Any]] = []
    for method in sorted({row["method"] for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        metric_names = [key for key in method_rows[0] if key not in {"run_id", "method", "seed"}]
        for metric in metric_names:
            samples = [float(row[metric]) for row in method_rows]
            mean = math.fsum(samples) / len(samples)
            standard_deviation = None
            if len(samples) > 1:
                standard_deviation = math.sqrt(
                    math.fsum((value - mean) ** 2 for value in samples) / (len(samples) - 1)
                )
            summaries.append({
                "condition": condition,
                "method": method,
                "metric": metric,
                "runs": len(samples),
                "mean": mean,
                "sample_standard_deviation": standard_deviation,
            })
    return {
        "condition": condition,
        "run_metrics": rows,
        "method_summaries": summaries,
        "paired_primary_comparisons": _paired_primary_comparisons(rows, condition),
        "skipped": skipped,
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
    writer.writerows(rows)
    return output.getvalue()


def main() -> None:
    args = parse_args()
    result = collect_results(args.results_root, args.condition, args.wave_nx)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    _atomic_write(
        args.output_directory / f"nrmse_{args.condition}.json",
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
    )
    _atomic_write(
        args.output_directory / f"nrmse_{args.condition}.csv",
        _csv(result["method_summaries"]),
    )
    _atomic_write(
        args.output_directory / f"primary_comparisons_{args.condition}.csv",
        _csv(result["paired_primary_comparisons"]),
    )
    print(json.dumps({
        "condition": args.condition,
        "completed_runs": len(result["run_metrics"]),
        "skipped_runs": len(result["skipped"]),
        "output_directory": str(args.output_directory),
    }))


if __name__ == "__main__":
    main()
