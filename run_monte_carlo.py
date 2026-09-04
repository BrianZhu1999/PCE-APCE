from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from run_hybrid_wave import make_config, run


def percentile_interval(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    tail = (1.0 - level) / 2.0
    return float(np.quantile(values, tail)), float(np.quantile(values, 1.0 - tail))


def bootstrap_mean_interval(
    values: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int = 5000,
    level: float = 0.95,
) -> tuple[float, float]:
    indices = rng.integers(0, values.size, size=(n_bootstrap, values.size))
    means = values[indices].mean(axis=1)
    return percentile_interval(means, level)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo statistics for nested alpha-path EnSF")
    parser.add_argument("--mode", choices=["quick", "balanced", "large"], default="quick")
    parser.add_argument("--n-seeds", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=20260803)
    parser.add_argument("--filter", choices=["lr", "direct"], default="lr")
    parser.add_argument("--output", default="results_statistics_50seeds")
    args = parser.parse_args()

    if args.n_seeds < 30:
        raise ValueError("For the requested statistical experiment, --n-seeds must be at least 30.")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, float | int | bool | str]] = []

    base_cfg = replace(make_config(args.mode), filter_variant=args.filter)
    for index in range(args.n_seeds):
        seed = args.base_seed + index
        cfg = replace(base_cfg, seed=seed)
        result = run(cfg, output_dir=None, make_figures=False, save_data=False)
        metrics = result["metrics"]
        record = {"seed": seed, **metrics}
        records.append(record)
        print(
            f"[{index + 1:02d}/{args.n_seeds}] seed={seed} "
            f"mean_RMSE={metrics['mean_rmse_hybrid']:.6g}, "
            f"reduction={metrics['relative_rmse_reduction_percent']:.2f}%, "
            f"alpha_hit={metrics['alpha_top1_correct']}"
        )

    fieldnames = list(records[0].keys())
    with (output_dir / "monte_carlo_runs.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    hybrid = np.array([float(r["mean_rmse_hybrid"]) for r in records])
    baseline = np.array([float(r["mean_rmse_baseline"]) for r in records])
    final_hybrid = np.array([float(r["final_rmse_hybrid"]) for r in records])
    final_baseline = np.array([float(r["final_rmse_baseline"]) for r in records])
    reduction = 100.0 * (1.0 - hybrid / np.maximum(baseline, 1.0e-15))
    paired_difference = baseline - hybrid
    coverage = np.array([float(r["coverage_90_final"]) for r in records])
    alpha_hit = np.array([bool(r["alpha_top1_correct"]) for r in records], dtype=float)
    alpha_best = np.array([float(r["alpha_best_final"]) for r in records])

    rng = np.random.default_rng(args.base_seed + 100000)
    diff_ci = bootstrap_mean_interval(paired_difference, rng)
    reduction_ci = bootstrap_mean_interval(reduction, rng)
    coverage_ci = bootstrap_mean_interval(coverage, rng)
    hit_ci = bootstrap_mean_interval(alpha_hit, rng)

    summary = {
        "n_seeds": args.n_seeds,
        "base_seed": args.base_seed,
        "mode": args.mode,
        "filter_variant": args.filter,
        "mean_rmse_hybrid_mean": float(hybrid.mean()),
        "mean_rmse_hybrid_std": float(hybrid.std(ddof=1)),
        "mean_rmse_baseline_mean": float(baseline.mean()),
        "mean_rmse_baseline_std": float(baseline.std(ddof=1)),
        "paired_rmse_improvement_mean": float(paired_difference.mean()),
        "paired_rmse_improvement_95ci": list(diff_ci),
        "relative_reduction_percent_mean": float(reduction.mean()),
        "relative_reduction_percent_std": float(reduction.std(ddof=1)),
        "relative_reduction_percent_95ci": list(reduction_ci),
        "win_rate_percent": float(100.0 * np.mean(hybrid < baseline)),
        "final_rmse_hybrid_mean": float(final_hybrid.mean()),
        "final_rmse_baseline_mean": float(final_baseline.mean()),
        "coverage_90_mean": float(coverage.mean()),
        "coverage_90_95ci": list(coverage_ci),
        "alpha_top1_accuracy": float(alpha_hit.mean()),
        "alpha_top1_accuracy_95ci": list(hit_ci),
        "alpha_best_mean": float(alpha_best.mean()),
    }

    with (output_dir / "monte_carlo_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.boxplot([hybrid, baseline], tick_labels=["Hybrid EnSF-LR", "Deterministic baseline"], showmeans=True)
    ax.set_ylabel("time-mean RMSE")
    ax.set_title(f"Paired Monte Carlo RMSE over {args.n_seeds} seeds")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "13_mc_rmse_boxplot.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.hist(reduction, bins=min(15, max(8, args.n_seeds // 4)), edgecolor="black", alpha=0.8)
    ax.axvline(reduction.mean(), linestyle="--", linewidth=1.5, label=f"mean={reduction.mean():.2f}%")
    ax.axvline(0.0, linewidth=1.0)
    ax.set_xlabel("relative RMSE reduction (%)")
    ax.set_ylabel("count")
    ax.set_title("Distribution of paired RMSE reduction")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "14_mc_reduction_histogram.png", dpi=180)
    plt.close(fig)

    unique_alpha, counts = np.unique(alpha_best, return_counts=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.bar([f"{a:.2f}" for a in unique_alpha], counts)
    ax.axhline(0.0, linewidth=0.5)
    ax.set_xlabel("final top-1 alpha path")
    ax.set_ylabel("number of runs")
    ax.set_title("Alpha-path recovery across random seeds")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "15_mc_alpha_recovery.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.scatter(np.arange(1, args.n_seeds + 1), coverage, s=28)
    ax.axhline(0.90, linestyle="--", linewidth=1.3, label="nominal 90%")
    ax.axhline(coverage.mean(), linestyle=":", linewidth=1.3, label=f"mean={coverage.mean():.3f}")
    ax.set_xlabel("run index")
    ax.set_ylabel("final spatial coverage")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Final nested-band coverage across random seeds")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "16_mc_coverage.png", dpi=180)
    plt.close(fig)

    print("\nMonte Carlo summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"\nOutput: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
