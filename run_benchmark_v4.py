from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import run_benchmark_v3 as v3
from run_hybrid_wave import ensf_update_lr, propagate_batch


AblationName = Literal[
    "A0_old",
    "A1_paired_init",
    "A2_paired_sampler",
    "A3_shadow",
    "A4_gaussian",
    "A5_shrinkage",
    "A6_pce",
    "A7_apce",
]

ABLATION_LABELS: dict[str, str] = {
    "A0_old": "A0 old evidence",
    "A1_paired_init": "A1 + paired initial ensemble",
    "A2_paired_sampler": "A2 + paired sampler noise",
    "A3_shadow": "A3 + shadow forecast bank",
    "A4_gaussian": "A4 + one-step Gaussian evidence",
    "A5_shrinkage": "A5 + cumulative shrinkage evidence",
    "A6_pce": "A6 + continuous refinement (full PCE)",
    "A7_apce": "A7 adaptive calibrated PCE",
}


@dataclass(frozen=True)
class V4EvidenceConfig:
    paired_initial: bool = True
    paired_sampler: bool = True
    shadow_bank: bool = True
    gaussian_evidence: bool = True
    shrinkage: float = 0.35
    temperature: float = 0.50
    forgetting: float = 1.0
    sensitivity_floor: float = 1.0
    adaptive_temperature: bool = False
    min_temperature: float = 0.15
    analysis_blend_start: float = 0.55
    analysis_blend_max: float = 0.0
    entropy_floor_start: float = 0.0
    entropy_floor_mid: float = 0.0
    entropy_floor_end: float = 0.0
    entropy_turn: float = 0.70
    weight_floor: float = 1.0e-8
    continuous_estimator: str = "quadratic"


ABLATION_CONFIGS: dict[str, V4EvidenceConfig] = {
    "A0_old": V4EvidenceConfig(
        paired_initial=False,
        paired_sampler=False,
        shadow_bank=False,
        gaussian_evidence=False,
        shrinkage=0.0,
        temperature=0.55,
    ),
    "A1_paired_init": V4EvidenceConfig(
        paired_initial=True,
        paired_sampler=False,
        shadow_bank=False,
        gaussian_evidence=False,
        shrinkage=0.0,
        temperature=0.55,
    ),
    "A2_paired_sampler": V4EvidenceConfig(
        paired_initial=True,
        paired_sampler=True,
        shadow_bank=False,
        gaussian_evidence=False,
        shrinkage=0.0,
        temperature=0.55,
    ),
    "A3_shadow": V4EvidenceConfig(
        paired_initial=True,
        paired_sampler=True,
        shadow_bank=True,
        gaussian_evidence=False,
        shrinkage=0.0,
        temperature=0.55,
    ),
    "A4_gaussian": V4EvidenceConfig(
        paired_initial=True,
        paired_sampler=True,
        shadow_bank=True,
        gaussian_evidence=True,
        shrinkage=0.0,
        temperature=0.50,
        forgetting=0.0,
        continuous_estimator="theta_mean",
    ),
    "A5_shrinkage": V4EvidenceConfig(
        paired_initial=True,
        paired_sampler=True,
        shadow_bank=True,
        gaussian_evidence=True,
        shrinkage=0.35,
        temperature=0.50,
        forgetting=1.0,
        continuous_estimator="theta_mean",
    ),
    "A6_pce": V4EvidenceConfig(
        paired_initial=True,
        paired_sampler=True,
        shadow_bank=True,
        gaussian_evidence=True,
        shrinkage=0.35,
        temperature=0.50,
        forgetting=1.0,
        sensitivity_floor=1.0,
        adaptive_temperature=False,
        analysis_blend_max=0.0,
        entropy_floor_start=0.0,
        entropy_floor_mid=0.0,
        entropy_floor_end=0.0,
        continuous_estimator="quadratic",
    ),
    "A7_apce": V4EvidenceConfig(
        paired_initial=True,
        paired_sampler=True,
        shadow_bank=True,
        gaussian_evidence=True,
        shrinkage=0.35,
        temperature=0.45,
        forgetting=0.975,
        sensitivity_floor=0.35,
        adaptive_temperature=True,
        min_temperature=0.08,
        analysis_blend_start=0.55,
        analysis_blend_max=0.12,
        entropy_floor_start=0.38,
        entropy_floor_mid=0.30,
        entropy_floor_end=0.22,
        entropy_turn=0.70,
        continuous_estimator="hybrid",
    ),
}


def entropy(weights: np.ndarray) -> float:
    w = np.maximum(weights, 1.0e-300)
    return float(-np.sum(w * np.log(w)))


def entropy_project(weights: np.ndarray, target: float) -> np.ndarray:
    """Minimum uniform mixing required to reach a target Shannon entropy."""
    if target <= 0.0 or entropy(weights) >= target:
        return weights
    uniform = np.ones_like(weights) / weights.size
    low, high = 0.0, 1.0
    for _ in range(50):
        middle = 0.5 * (low + high)
        mixed = (1.0 - middle) * weights + middle * uniform
        if entropy(mixed) < target:
            low = middle
        else:
            high = middle
    mixed = (1.0 - high) * weights + high * uniform
    return mixed / mixed.sum()


def entropy_target(progress: float, config: V4EvidenceConfig) -> float:
    if config.entropy_floor_start <= 0.0:
        return 0.0
    turn = float(np.clip(config.entropy_turn, 1.0e-6, 1.0 - 1.0e-6))
    if progress <= turn:
        ratio = progress / turn
        return config.entropy_floor_start + ratio * (
            config.entropy_floor_mid - config.entropy_floor_start
        )
    ratio = (progress - turn) / (1.0 - turn)
    return config.entropy_floor_mid + ratio * (
        config.entropy_floor_end - config.entropy_floor_mid
    )


def alpha_continuous_estimate(
    alpha_grid: np.ndarray,
    theta_grid: np.ndarray,
    weights: np.ndarray,
    scenario_config: Any,
    estimator: str,
) -> float:
    log_weights = np.log(np.maximum(weights, 1.0e-300))
    quadratic = v3.continuous_alpha_estimate(alpha_grid, log_weights)
    theta_mean = float(np.sum(weights * theta_grid))
    posterior_mean = v3.theta_to_alpha(theta_mean, scenario_config)
    if estimator == "quadratic":
        return quadratic
    if estimator == "theta_mean":
        return posterior_mean
    concentration = float(np.max(weights))
    blend = float(np.clip((concentration - 0.25) / 0.35, 0.0, 1.0))
    return float(blend * quadratic + (1.0 - blend) * posterior_mean)


def evidence_vector(
    branch_observations: list[np.ndarray],
    observation: np.ndarray,
    obs_noise: float,
    config: V4EvidenceConfig,
) -> np.ndarray:
    if not config.gaussian_evidence:
        return np.asarray(
            [
                v3.original_compatibility_log(item, observation, obs_noise)
                for item in branch_observations
            ],
            dtype=float,
        )
    dimension_weights = v3.alpha_sensitivity_weights(
        branch_observations,
        config.sensitivity_floor,
    )
    return np.asarray(
        [
            v3.gaussian_log_evidence(
                item,
                observation,
                obs_noise,
                config.shrinkage,
                dimension_weights,
            )
            for item in branch_observations
        ],
        dtype=float,
    )


def run_ablation(
    scenario: v3.Scenario,
    ablation: AblationName,
    return_details: bool = False,
) -> dict[str, Any]:
    config = ABLATION_CONFIGS[ablation]
    cfg = scenario.cfg
    rng = np.random.default_rng(cfg.seed + v3.stable_offset(ablation))
    n_steps = scenario.times.size - 1
    nx = cfg.nx
    truth = scenario.truth_states
    observation_indices = scenario.observation_indices

    branches = (
        scenario.branch_initial.copy()
        if config.paired_initial
        else scenario.branch_initial_independent.copy()
    )
    evidence_branches = branches.copy() if config.shadow_bank else None
    log_weights = np.zeros(cfg.n_alpha, dtype=float)
    weights = v3.softmax(log_weights)
    rmse = np.zeros(n_steps + 1, dtype=float)
    entropy_history: list[float] = []
    alpha_history: list[np.ndarray] = []

    def combined_estimate() -> np.ndarray:
        return np.sum(weights[:, None] * branches.mean(axis=1), axis=0)

    rmse[0] = v3.evaluate_estimate(combined_estimate(), truth[0], nx)

    for step in range(1, n_steps + 1):
        current_time = scenario.times[step - 1]
        for branch_index, theta in enumerate(scenario.theta_grid):
            branches[branch_index] = propagate_batch(
                branches[branch_index],
                float(theta),
                current_time,
                cfg,
                rng,
                stochastic=True,
                noise_draw=scenario.forecast_noise[step - 1],
            )
            if evidence_branches is not None:
                evidence_branches[branch_index] = propagate_batch(
                    evidence_branches[branch_index],
                    float(theta),
                    current_time,
                    cfg,
                    rng,
                    stochastic=True,
                    noise_draw=scenario.forecast_noise[step - 1],
                )

        if step in scenario.observations:
            observation = scenario.observations[step]
            shadow_source = evidence_branches if evidence_branches is not None else branches
            shadow_observations = [
                shadow_source[q][:, observation_indices].copy()
                for q in range(cfg.n_alpha)
            ]
            log_likelihood_shadow = evidence_vector(
                shadow_observations,
                observation,
                cfg.obs_noise,
                config,
            )

            progress = step / max(n_steps, 1)
            blend = 0.0
            if config.analysis_blend_max > 0.0 and progress > config.analysis_blend_start:
                blend = config.analysis_blend_max * min(
                    1.0,
                    (progress - config.analysis_blend_start)
                    / max(1.0 - config.analysis_blend_start, 1.0e-12),
                )
            if blend > 0.0:
                analysis_observations = [
                    branches[q][:, observation_indices].copy()
                    for q in range(cfg.n_alpha)
                ]
                log_likelihood_analysis = evidence_vector(
                    analysis_observations,
                    observation,
                    cfg.obs_noise,
                    config,
                )
                log_likelihood = (
                    (1.0 - blend) * log_likelihood_shadow
                    + blend * log_likelihood_analysis
                )
            else:
                log_likelihood = log_likelihood_shadow

            centered = log_likelihood - np.mean(log_likelihood)
            current_temperature = config.temperature
            if config.adaptive_temperature:
                entropy_ratio = entropy(weights) / max(math.log(cfg.n_alpha), 1.0e-12)
                current_temperature = float(
                    np.clip(
                        config.temperature * entropy_ratio**0.75,
                        config.min_temperature,
                        config.temperature,
                    )
                )
            log_weights = (
                config.forgetting * log_weights
                + current_temperature * centered
            )
            weights = v3.softmax(log_weights)
            weights = np.maximum(weights, config.weight_floor)
            weights /= weights.sum()
            weights = entropy_project(weights, entropy_target(progress, config))
            log_weights = np.log(np.maximum(weights, 1.0e-300))

            paired_seed = cfg.seed + 10_000_000 + step
            for branch_index in range(cfg.n_alpha):
                branch_rng = (
                    np.random.default_rng(paired_seed)
                    if config.paired_sampler
                    else rng
                )
                branches[branch_index] = ensf_update_lr(
                    branches[branch_index],
                    observation,
                    observation_indices,
                    cfg,
                    branch_rng,
                )

            entropy_history.append(entropy(weights))
            alpha_history.append(weights.copy())

        rmse[step] = v3.evaluate_estimate(combined_estimate(), truth[step], nx)

    alpha_best = float(scenario.alpha_grid[int(np.argmax(weights))])
    alpha_continuous = alpha_continuous_estimate(
        scenario.alpha_grid,
        scenario.theta_grid,
        weights,
        cfg,
        config.continuous_estimator,
    )
    result = {
        "ablation": ablation,
        "label": ABLATION_LABELS[ablation],
        "mean_rmse": float(np.mean(rmse)),
        "final_rmse": float(rmse[-1]),
        "peak_rmse": float(np.max(rmse)),
        "alpha_best": alpha_best,
        "alpha_continuous": alpha_continuous,
        "alpha_top1_correct": bool(abs(alpha_best - cfg.alpha_true) < 1.0e-12),
        "alpha_abs_error": float(abs(alpha_continuous - cfg.alpha_true)),
        "alpha_entropy": entropy(weights),
        "rmse": rmse,
        "alpha_history": np.asarray(alpha_history),
        "entropy_history": np.asarray(entropy_history),
        "final_weights": weights,
    }
    if return_details:
        result.update(
            {
                "final_branches": branches.copy(),
                "final_truth": truth[-1].copy(),
                "observation_indices": observation_indices.copy(),
            }
        )
    return result


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(values.size, dtype=float)
    index = 0
    while index < values.size:
        end = index + 1
        while end < values.size and values[order[end]] == values[order[index]]:
            end += 1
        mean_rank = 0.5 * ((index + 1) + end)
        ranks[order[index:end]] = mean_rank
        index = end
    return ranks


def signed_rank_permutation_pvalue(
    differences: np.ndarray,
    seed: int,
    n_permutations: int = 100_000,
) -> float:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.abs(differences) > 1.0e-15]
    if differences.size == 0:
        return 1.0
    ranks = average_ranks(np.abs(differences))
    observed = abs(float(np.sum(np.sign(differences) * ranks)))
    rng = np.random.default_rng(seed)
    batch_size = 10_000
    exceed = 0
    completed = 0
    while completed < n_permutations:
        current = min(batch_size, n_permutations - completed)
        signs = rng.choice((-1.0, 1.0), size=(current, differences.size))
        statistics = np.abs(signs @ ranks)
        exceed += int(np.sum(statistics >= observed - 1.0e-12))
        completed += current
    return float((exceed + 1) / (n_permutations + 1))


def exact_mcnemar_pvalue(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    first_only = int(np.sum(first & ~second))
    second_only = int(np.sum(~first & second))
    discordant = first_only + second_only
    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(first_only, second_only)
        tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "first_correct_second_wrong": first_only,
        "first_wrong_second_correct": second_only,
        "discordant": discordant,
        "exact_two_sided_p": float(p_value),
    }


def bootstrap_mean_ci(
    values: np.ndarray,
    seed: int,
    n_bootstrap: int = 10_000,
) -> list[float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(n_bootstrap, values.size))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def paired_statistics(
    proposed: np.ndarray,
    reference: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    difference = reference - proposed
    dz = float(difference.mean() / max(difference.std(ddof=1), 1.0e-15))
    relative = 100.0 * difference / np.maximum(reference, 1.0e-15)
    return {
        "mean_absolute_improvement": float(difference.mean()),
        "absolute_improvement_95ci": bootstrap_mean_ci(difference, seed),
        "mean_relative_improvement_percent": float(relative.mean()),
        "relative_improvement_95ci_percent": bootstrap_mean_ci(relative, seed + 1),
        "win_rate_percent": float(100.0 * np.mean(proposed < reference)),
        "signed_rank_permutation_p": signed_rank_permutation_pvalue(
            difference,
            seed + 2,
        ),
        "cohen_dz": dz,
    }


def run_v4(
    mode: str,
    n_seeds: int,
    base_seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    trajectories: dict[str, list[np.ndarray]] = {
        name: [] for name in ABLATION_CONFIGS
    }

    for seed_index in range(n_seeds):
        cfg = replace(
            v3.make_config(mode),
            seed=base_seed + seed_index,
            filter_variant="lr",
        )
        scenario = v3.generate_scenario(cfg)
        for ablation in ABLATION_CONFIGS:
            result = run_ablation(scenario, ablation)  # type: ignore[arg-type]
            trajectories[ablation].append(result.pop("rmse"))
            result.pop("alpha_history", None)
            result.pop("entropy_history", None)
            result.pop("final_weights", None)
            records.append({"seed": cfg.seed, **result})
        proposed = next(
            item for item in records
            if item["seed"] == cfg.seed and item["ablation"] == "A7_apce"
        )
        print(
            f"[{seed_index + 1:02d}/{n_seeds}] seed={cfg.seed} "
            f"APCE_RMSE={proposed['mean_rmse']:.6g}, "
            f"alpha_hit={proposed['alpha_top1_correct']}"
        )

    fieldnames = sorted({key for record in records for key in record})
    with (output_dir / "v4_ablation_runs.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    summaries: dict[str, Any] = {}
    for ablation in ABLATION_CONFIGS:
        subset = [item for item in records if item["ablation"] == ablation]
        mean_rmse = np.asarray([item["mean_rmse"] for item in subset], dtype=float)
        final_rmse = np.asarray([item["final_rmse"] for item in subset], dtype=float)
        alpha_correct = np.asarray(
            [item["alpha_top1_correct"] for item in subset], dtype=bool
        )
        alpha_error = np.asarray([item["alpha_abs_error"] for item in subset], dtype=float)
        alpha_entropy = np.asarray([item["alpha_entropy"] for item in subset], dtype=float)
        summaries[ablation] = {
            "label": ABLATION_LABELS[ablation],
            "config": asdict(ABLATION_CONFIGS[ablation]),
            "mean_rmse_mean": float(mean_rmse.mean()),
            "mean_rmse_std": float(mean_rmse.std(ddof=1)),
            "mean_rmse_95ci": bootstrap_mean_ci(
                mean_rmse, base_seed + v3.stable_offset(ablation)
            ),
            "final_rmse_mean": float(final_rmse.mean()),
            "final_rmse_std": float(final_rmse.std(ddof=1)),
            "alpha_top1_accuracy_percent": float(100.0 * alpha_correct.mean()),
            "alpha_continuous_mae": float(alpha_error.mean()),
            "alpha_entropy_mean": float(alpha_entropy.mean()),
        }

    proposed_records = [item for item in records if item["ablation"] == "A7_apce"]
    proposed_rmse = np.asarray([item["mean_rmse"] for item in proposed_records], dtype=float)
    proposed_final = np.asarray([item["final_rmse"] for item in proposed_records], dtype=float)
    proposed_correct = np.asarray(
        [item["alpha_top1_correct"] for item in proposed_records], dtype=bool
    )
    comparisons: dict[str, Any] = {}
    for ablation in ABLATION_CONFIGS:
        if ablation == "A7_apce":
            continue
        reference_records = [item for item in records if item["ablation"] == ablation]
        reference_rmse = np.asarray(
            [item["mean_rmse"] for item in reference_records], dtype=float
        )
        reference_final = np.asarray(
            [item["final_rmse"] for item in reference_records], dtype=float
        )
        reference_correct = np.asarray(
            [item["alpha_top1_correct"] for item in reference_records], dtype=bool
        )
        comparisons[ablation] = {
            "time_mean_rmse": paired_statistics(
                proposed_rmse,
                reference_rmse,
                base_seed + 10_000 + v3.stable_offset(ablation),
            ),
            "final_rmse": paired_statistics(
                proposed_final,
                reference_final,
                base_seed + 20_000 + v3.stable_offset(ablation),
            ),
            "alpha_top1_mcnemar": exact_mcnemar_pvalue(
                proposed_correct,
                reference_correct,
            ),
        }

    payload = {
        "mode": mode,
        "n_seeds": n_seeds,
        "base_seed": base_seed,
        "summaries": summaries,
        "apce_paired_comparisons": comparisons,
        "interpretation_constraints": [
            "Ablations isolate components incrementally; A7 is not assumed superior before testing.",
            "Top-1 accuracy must be interpreted with paired McNemar statistics.",
            "Continuous alpha MAE and state RMSE are separate objectives.",
            "Time-mean and terminal RMSE are reported separately because their rankings may differ.",
        ],
    }
    with (output_dir / "v4_summary.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    save_trajectories(trajectories, output_dir)
    make_figures(records, trajectories, output_dir, n_seeds)
    write_latex_assets(payload, output_dir)
    return payload


def save_trajectories(
    trajectories: dict[str, list[np.ndarray]],
    output_dir: Path,
) -> None:
    arrays = {
        key: np.stack(value, axis=0)
        for key, value in trajectories.items()
    }
    np.savez_compressed(output_dir / "v4_trajectories.npz", **arrays)


def make_figures(
    records: list[dict[str, Any]],
    trajectories: dict[str, list[np.ndarray]],
    output_dir: Path,
    n_seeds: int,
) -> None:
    order = list(ABLATION_CONFIGS)
    values = [
        [item["mean_rmse"] for item in records if item["ablation"] == name]
        for name in order
    ]
    labels = [ABLATION_LABELS[name] for name in order]

    figure, axis = plt.subplots(figsize=(14, 6.5))
    try:
        axis.boxplot(values, tick_labels=labels, showmeans=True)
    except TypeError:
        axis.boxplot(values, labels=labels, showmeans=True)
    axis.set_ylabel("time-mean RMSE")
    axis.set_title(f"V4 component ablation over {n_seeds} paired seeds")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "26_v4_ablation_rmse_boxplot.png", dpi=180)
    plt.close(figure)

    means = np.asarray([np.mean(value) for value in values], dtype=float)
    incremental = np.zeros_like(means)
    incremental[1:] = 100.0 * (means[:-1] - means[1:]) / np.maximum(means[:-1], 1.0e-15)
    figure, axis = plt.subplots(figsize=(12, 5.8))
    axis.bar(labels[1:], incremental[1:])
    axis.axhline(0.0, linewidth=1.0)
    axis.set_ylabel("incremental RMSE change (%)")
    axis.set_title("Incremental contribution of each V4 component")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "27_v4_component_waterfall.png", dpi=180)
    plt.close(figure)

    accuracies = []
    maes = []
    entropies = []
    for name in order:
        subset = [item for item in records if item["ablation"] == name]
        accuracies.append(100.0 * np.mean([item["alpha_top1_correct"] for item in subset]))
        maes.append(np.mean([item["alpha_abs_error"] for item in subset]))
        entropies.append(np.mean([item["alpha_entropy"] for item in subset]))

    figure, axis = plt.subplots(figsize=(12, 5.8))
    axis.bar(labels, accuracies)
    axis.axhline(100.0 / 7.0, linestyle="--", linewidth=1.1, label="random 7-path guess")
    axis.set_ylim(0.0, 100.0)
    axis.set_ylabel("alpha Top-1 accuracy (%)")
    axis.set_title("V4 alpha-path identification ablation")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "28_v4_alpha_top1.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5.8))
    axis.scatter(entropies, means)
    for label, x_value, y_value in zip(labels, entropies, means):
        axis.annotate(label.split()[0], (x_value, y_value), xytext=(4, 3), textcoords="offset points")
    axis.set_xlabel("mean final alpha entropy")
    axis.set_ylabel("time-mean RMSE")
    axis.set_title("Weight concentration versus state error")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "29_v4_entropy_rmse_tradeoff.png", dpi=180)
    plt.close(figure)

    selected = ["A0_old", "A3_shadow", "A6_pce", "A7_apce"]
    figure, axis = plt.subplots(figsize=(11, 6))
    for name in selected:
        curve = np.mean(np.stack(trajectories[name], axis=0), axis=0)
        axis.plot(curve, label=ABLATION_LABELS[name])
    axis.set_xlabel("time-step index")
    axis.set_ylabel("mean RMSE")
    axis.set_title(f"V4 mean RMSE trajectories over {n_seeds} seeds")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "30_v4_mean_rmse_trajectories.png", dpi=180)
    plt.close(figure)

    final_means = np.asarray(
        [
            np.mean([item["final_rmse"] for item in records if item["ablation"] == name])
            for name in order
        ],
        dtype=float,
    )
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.scatter(means, final_means)
    for label, x_value, y_value in zip(labels, means, final_means):
        axis.annotate(label.split()[0], (x_value, y_value), xytext=(4, 3), textcoords="offset points")
    axis.set_xlabel("time-mean RMSE")
    axis.set_ylabel("terminal RMSE")
    axis.set_title("Time-mean versus terminal accuracy")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "31_v4_terminal_tradeoff.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 5.8))
    axis.bar(labels, maes)
    axis.set_ylabel("continuous alpha MAE")
    axis.set_title("V4 continuous alpha estimation error")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "32_v4_alpha_mae.png", dpi=180)
    plt.close(figure)


def write_latex_assets(payload: dict[str, Any], output_dir: Path) -> None:
    summaries = payload["summaries"]
    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Variant & Mean RMSE & Final RMSE & $\alpha$ Top-1 & $\alpha$ MAE & Entropy \\",
        r"\midrule",
    ]
    for name in ABLATION_CONFIGS:
        item = summaries[name]
        lines.append(
            f"{ABLATION_LABELS[name]} & "
            f"{item['mean_rmse_mean']:.4e} & "
            f"{item['final_rmse_mean']:.4e} & "
            f"{item['alpha_top1_accuracy_percent']:.1f}\\% & "
            f"{item['alpha_continuous_mae']:.4f} & "
            f"{item['alpha_entropy_mean']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output_dir / "v4_ablation_table.tex").write_text("\n".join(lines), encoding="utf-8")

    proposed = summaries["A7_apce"]
    pce = summaries["A6_pce"]
    old = summaries["A0_old"]
    macros = {
        "VFourSeeds": str(payload["n_seeds"]),
        "APCEMeanRMSE": f"{proposed['mean_rmse_mean']:.4e}",
        "APCEFinalRMSE": f"{proposed['final_rmse_mean']:.4e}",
        "APCEAlphaAccuracy": f"{proposed['alpha_top1_accuracy_percent']:.1f}\\%",
        "APCEAlphaMAE": f"{proposed['alpha_continuous_mae']:.4f}",
        "APCEEntropy": f"{proposed['alpha_entropy_mean']:.3f}",
        "PCEMeanRMSEVFour": f"{pce['mean_rmse_mean']:.4e}",
        "OldMeanRMSEVFour": f"{old['mean_rmse_mean']:.4e}",
    }
    with (output_dir / "v4_results_macros.tex").open("w", encoding="utf-8") as file:
        for name, value in macros.items():
            file.write(f"\\newcommand{{\\{name}}}{{{value}}}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V4 PCE component ablation, calibration and paired statistics"
    )
    parser.add_argument("--mode", choices=["quick", "balanced", "large"], default="quick")
    parser.add_argument("--n-seeds", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=20260803)
    parser.add_argument("--output", default="results_benchmark_v4_50seeds")
    parser.add_argument("--allow-small-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_seeds < 30 and not args.allow_small_test:
        raise ValueError("Formal V4 benchmark requires at least 30 seeds.")
    payload = run_v4(
        args.mode,
        args.n_seeds,
        args.base_seed,
        Path(args.output),
    )
    print("\nV4 benchmark completed.")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nOutput: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
