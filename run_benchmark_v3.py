from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_hybrid_wave import (
    Config,
    ensf_update_direct,
    ensf_update_lr,
    initial_state,
    liu_normal_inverse,
    make_config,
    propagate_batch,
    smooth_noise,
)
from experiments.wave_scenario_assets import WaveScenarioAssets

MethodName = Literal[
    "deterministic",
    "enkf",
    "ensf_direct",
    "ensf_lr",
    "alpha_only",
    "oracle_alpha",
    "joint_param_enkf",
    "alpha_ensf_lr",
    "alpha_ensf_lr_pce",
]

METHOD_LABELS: dict[str, str] = {
    "deterministic": "Deterministic reference",
    "enkf": "EnKF",
    "ensf_direct": "Original-style EnSF",
    "ensf_lr": "EnSF-LR (no alpha)",
    "alpha_only": "Alpha-only model averaging",
    "oracle_alpha": "Oracle-alpha EnSF-LR",
    "joint_param_enkf": "Joint state-parameter EnKF",
    "alpha_ensf_lr": "Alpha-EnSF-LR (old evidence)",
    "alpha_ensf_lr_pce": "Alpha-EnSF-LR + PCE",
}

BASELINE_METHODS = [
    "enkf",
    "ensf_direct",
    "ensf_lr",
    "alpha_only",
    "oracle_alpha",
    "joint_param_enkf",
]

ALL_METHODS = [
    "deterministic",
    *BASELINE_METHODS,
    "alpha_ensf_lr",
    "alpha_ensf_lr_pce",
]


@dataclass(frozen=True)
class AlphaEvidenceConfig:
    window: int = 1
    shrinkage: float = 0.35
    forgetting: float = 1.0
    temperature: float = 0.50
    sensitivity_floor: float = 1.0
    weight_floor: float = 1.0e-6
    entropy_mix: float = 0.0


@dataclass
class Scenario:
    cfg: Config
    x: np.ndarray
    times: np.ndarray
    alpha_grid: np.ndarray
    theta_grid: np.ndarray
    theta_true: float
    truth_states: np.ndarray
    observations: dict[int, np.ndarray]
    observation_indices: np.ndarray
    ensemble_initial: np.ndarray
    branch_initial: np.ndarray
    branch_initial_independent: np.ndarray
    truth_noise: np.ndarray
    forecast_noise: np.ndarray


def stable_offset(name: str) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def softmax(log_weights: np.ndarray) -> np.ndarray:
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)
    return weights / np.sum(weights)


def alpha_to_theta(alpha: float, cfg: Config) -> float:
    return float(cfg.epistemic_scale * liu_normal_inverse(alpha))


def theta_to_alpha(theta: float, cfg: Config) -> float:
    z = theta / max(cfg.epistemic_scale, 1.0e-12)
    return float(1.0 / (1.0 + math.exp(-math.pi * z / math.sqrt(3.0))))


def generate_scenario(cfg: Config) -> Scenario:
    rng = np.random.default_rng(cfg.seed)
    x = np.linspace(0.0, cfg.length, cfg.nx)
    n_steps = int(round(cfg.t_end / cfg.dt))
    times = np.arange(n_steps + 1) * cfg.dt
    alpha_grid = np.linspace(cfg.alpha_min, cfg.alpha_max, cfg.n_alpha)
    theta_grid = cfg.epistemic_scale * liu_normal_inverse(alpha_grid)
    theta_true = alpha_to_theta(cfg.alpha_true, cfg)
    observation_indices = np.linspace(4, cfg.nx - 5, cfg.n_sensors, dtype=int)

    x0 = initial_state(x)
    truth = x0[None, :].copy()
    truth_states = np.zeros((n_steps + 1, 2 * cfg.nx), dtype=float)
    truth_states[0] = truth[0]

    truth_noise = np.stack(
        [smooth_noise(rng.normal(size=(1, cfg.nx))) for _ in range(n_steps)],
        axis=0,
    )
    observation_noise = {
        step: cfg.obs_noise * rng.normal(size=cfg.n_sensors)
        for step in range(cfg.obs_interval, n_steps + 1, cfg.obs_interval)
    }

    for step in range(1, n_steps + 1):
        truth = propagate_batch(
            truth,
            theta_true,
            times[step - 1],
            cfg,
            rng,
            stochastic=True,
            noise_draw=truth_noise[step - 1],
        )
        truth_states[step] = truth[0]

    observations = {
        step: truth_states[step, observation_indices] + noise
        for step, noise in observation_noise.items()
    }

    # Paired ensemble anomalies are deliberately shared by alpha branches.
    # This removes branch-to-branch Monte Carlo confounding and improves
    # epistemic parameter discrimination without using the truth.
    du = 0.012 * smooth_noise(rng.normal(size=(cfg.ensemble_size, cfg.nx)))
    dv = 0.025 * smooth_noise(rng.normal(size=(cfg.ensemble_size, cfg.nx)))
    ensemble_initial = x0[None, :] + np.concatenate([du, dv], axis=1)
    ensemble_initial[:, [0, cfg.nx - 1, cfg.nx, 2 * cfg.nx - 1]] = 0.0
    branch_initial = np.repeat(ensemble_initial[None, :, :], cfg.n_alpha, axis=0)
    branch_initial_independent = np.empty_like(branch_initial)
    for q in range(cfg.n_alpha):
        du_q = 0.012 * smooth_noise(rng.normal(size=(cfg.ensemble_size, cfg.nx)))
        dv_q = 0.025 * smooth_noise(rng.normal(size=(cfg.ensemble_size, cfg.nx)))
        branch_initial_independent[q] = x0[None, :] + np.concatenate([du_q, dv_q], axis=1)
        branch_initial_independent[q, :, [0, cfg.nx - 1, cfg.nx, 2 * cfg.nx - 1]] = 0.0

    forecast_noise = np.stack(
        [smooth_noise(rng.normal(size=(cfg.ensemble_size, cfg.nx))) for _ in range(n_steps)],
        axis=0,
    )

    return Scenario(
        cfg=cfg,
        x=x,
        times=times,
        alpha_grid=alpha_grid,
        theta_grid=theta_grid,
        theta_true=theta_true,
        truth_states=truth_states,
        observations=observations,
        observation_indices=observation_indices,
        ensemble_initial=ensemble_initial,
        branch_initial=branch_initial,
        branch_initial_independent=branch_initial_independent,
        truth_noise=truth_noise,
        forecast_noise=forecast_noise,
    )


def scenario_from_assets(assets: WaveScenarioAssets, cfg: Config | None = None) -> Scenario:
    """Rebuild the legacy runner view from frozen common Wave inputs.

    This adapter deliberately does not regenerate truth, observations, initial
    ensembles, or forecast noise.  It is the entry point for Gate A fairness
    checks; method implementations can keep their native update components
    while consuming one immutable scenario.
    """
    base_cfg = cfg or replace(make_config("quick"), seed=assets.seed, nx=assets.nx, ensemble_size=assets.ensemble_size)
    if base_cfg.nx != assets.nx or base_cfg.ensemble_size != assets.ensemble_size:
        raise ValueError("cfg dimensions disagree with frozen WaveScenarioAssets")
    x = np.linspace(0.0, base_cfg.length, assets.nx)
    alpha_grid = np.linspace(base_cfg.alpha_min, base_cfg.alpha_max, base_cfg.n_alpha)
    theta_grid = base_cfg.epistemic_scale * liu_normal_inverse(alpha_grid)
    observations = {
        int(step): assets.observations[step].copy()
        for step in np.flatnonzero(assets.observation_mask)
    }
    branch_initial = np.repeat(assets.initial_ensemble[None, :, :], base_cfg.n_alpha, axis=0)
    return Scenario(
        cfg=base_cfg,
        x=x,
        times=assets.times.copy(),
        alpha_grid=alpha_grid,
        theta_grid=theta_grid,
        theta_true=alpha_to_theta(assets.alpha_true, base_cfg),
        truth_states=assets.truth_states.copy(),
        observations=observations,
        observation_indices=assets.observation_indices.copy(),
        ensemble_initial=assets.initial_ensemble.copy(),
        branch_initial=branch_initial,
        # Gate A uses one common initial ensemble for every alpha branch.
        # Legacy independent branches remain available only in regenerated
        # scenarios and are intentionally excluded from the frozen protocol.
        branch_initial_independent=branch_initial.copy(),
        truth_noise=assets.truth_noise.copy(),
        forecast_noise=assets.forecast_noise.copy(),
    )


def gaussian_log_evidence(
    ensemble_observation: np.ndarray,
    observation: np.ndarray,
    obs_noise: float,
    shrinkage: float,
    dimension_weights: np.ndarray | None = None,
) -> float:
    mean = ensemble_observation.mean(axis=0)
    anomalies = ensemble_observation - mean
    denom = max(ensemble_observation.shape[0] - 1, 1)
    covariance = anomalies.T @ anomalies / denom
    diagonal = np.diag(np.diag(covariance))
    covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal
    covariance += (obs_noise**2 + 1.0e-7) * np.eye(observation.size)

    residual = observation - mean
    if dimension_weights is not None:
        weights = np.asarray(dimension_weights, dtype=float)
        if weights.shape != residual.shape:
            raise ValueError("dimension_weights must match the observation dimension")
        weights = np.maximum(weights, 1.0e-8)
        weights = observation.size * weights / np.sum(weights)
        variances = np.maximum(np.diag(covariance), 1.0e-12)
        marginal_terms = residual * residual / variances + np.log(variances) + math.log(
            2.0 * math.pi
        )
        return float(-0.5 * np.sum(weights * marginal_terms))

    sign, log_det = np.linalg.slogdet(covariance)
    if sign <= 0:
        covariance += 1.0e-6 * np.eye(observation.size)
        sign, log_det = np.linalg.slogdet(covariance)
    mahalanobis = float(residual @ np.linalg.solve(covariance, residual))
    return float(-0.5 * (mahalanobis + log_det + observation.size * math.log(2.0 * math.pi)))


def original_compatibility_log(
    ensemble_observation: np.ndarray,
    observation: np.ndarray,
    obs_noise: float,
) -> float:
    mean = ensemble_observation.mean(axis=0)
    covariance = np.cov(ensemble_observation, rowvar=False)
    covariance = np.atleast_2d(covariance) + (obs_noise**2 + 1.0e-7) * np.eye(observation.size)
    residual = observation - mean
    mahalanobis = float(residual @ np.linalg.solve(covariance, residual))
    return float(-0.5 * mahalanobis / max(observation.size, 1))


def alpha_sensitivity_weights(branch_observations: list[np.ndarray], floor: float) -> np.ndarray:
    means = np.stack([item.mean(axis=0) for item in branch_observations], axis=0)
    between = np.var(means, axis=0, ddof=1)
    normalized = between / max(float(np.max(between)), 1.0e-12)
    return floor + (1.0 - floor) * normalized


def continuous_alpha_estimate(alpha_grid: np.ndarray, log_weights: np.ndarray) -> float:
    q = int(np.argmax(log_weights))
    if q == 0 or q == alpha_grid.size - 1:
        return float(alpha_grid[q])
    x = alpha_grid[q - 1 : q + 2]
    y = log_weights[q - 1 : q + 2]
    coefficients = np.polyfit(x, y, deg=2)
    a, b, _ = coefficients
    if a >= -1.0e-12:
        return float(alpha_grid[q])
    estimate = -b / (2.0 * a)
    return float(np.clip(estimate, x[0], x[-1]))


def enkf_update(
    prior: np.ndarray,
    observation: np.ndarray,
    obs_indices: np.ndarray,
    obs_noise: float,
    rng: np.random.Generator,
) -> np.ndarray:
    predicted = prior[:, obs_indices]
    x_anomaly = prior - prior.mean(axis=0)
    y_anomaly = predicted - predicted.mean(axis=0)
    denom = max(prior.shape[0] - 1, 1)
    covariance_xy = x_anomaly.T @ y_anomaly / denom
    covariance_yy = y_anomaly.T @ y_anomaly / denom
    covariance_yy += (obs_noise**2 + 1.0e-7) * np.eye(observation.size)
    gain = covariance_xy @ np.linalg.pinv(covariance_yy)
    perturbed = observation[None, :] + obs_noise * rng.normal(size=predicted.shape)
    return prior + (perturbed - predicted) @ gain.T


def joint_parameter_enkf_update(
    states: np.ndarray,
    theta: np.ndarray,
    observation: np.ndarray,
    obs_indices: np.ndarray,
    obs_noise: float,
    rng: np.random.Generator,
    theta_bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    augmented = np.column_stack([states, theta])
    predicted = states[:, obs_indices]
    a_anomaly = augmented - augmented.mean(axis=0)
    y_anomaly = predicted - predicted.mean(axis=0)
    denom = max(states.shape[0] - 1, 1)
    covariance_ay = a_anomaly.T @ y_anomaly / denom
    covariance_yy = y_anomaly.T @ y_anomaly / denom
    covariance_yy += (obs_noise**2 + 1.0e-7) * np.eye(observation.size)
    gain = covariance_ay @ np.linalg.pinv(covariance_yy)
    perturbed = observation[None, :] + obs_noise * rng.normal(size=predicted.shape)
    updated = augmented + (perturbed - predicted) @ gain.T
    updated_theta = np.clip(updated[:, -1], theta_bounds[0], theta_bounds[1])
    return updated[:, :-1], updated_theta


def propagate_memberwise_theta(
    states: np.ndarray,
    theta: np.ndarray,
    t: float,
    cfg: Config,
    rng: np.random.Generator,
    noise_draw: np.ndarray,
) -> np.ndarray:
    updated = np.empty_like(states)
    for member in range(states.shape[0]):
        updated[member : member + 1] = propagate_batch(
            states[member : member + 1],
            float(theta[member]),
            t,
            cfg,
            rng,
            stochastic=True,
            noise_draw=noise_draw[member : member + 1],
        )
    return updated


def method_has_alpha(method: str) -> bool:
    return method in {"alpha_only", "joint_param_enkf", "alpha_ensf_lr", "alpha_ensf_lr_pce"}


def evaluate_estimate(estimate: np.ndarray, truth: np.ndarray, nx: int) -> float:
    return float(np.sqrt(np.mean((estimate[:nx] - truth[:nx]) ** 2)))


def run_method(
    scenario: Scenario,
    method: MethodName,
    evidence_cfg: AlphaEvidenceConfig | None = None,
) -> dict[str, Any]:
    cfg = scenario.cfg
    evidence_cfg = evidence_cfg or AlphaEvidenceConfig()
    rng = np.random.default_rng(cfg.seed + stable_offset(method))
    n_steps = scenario.times.size - 1
    nx = cfg.nx
    truth = scenario.truth_states
    obs_idx = scenario.observation_indices

    rmse = np.zeros(n_steps + 1)
    alpha_history: list[np.ndarray] = []
    alpha_times: list[float] = []
    continuous_alpha_history: list[float] = []

    if method == "deterministic":
        state = initial_state(scenario.x)[None, :]
        estimate = state[0]
        rmse[0] = evaluate_estimate(estimate, truth[0], nx)
        for step in range(1, n_steps + 1):
            state = propagate_batch(
                state, 0.0, scenario.times[step - 1], cfg, rng, stochastic=False
            )
            rmse[step] = evaluate_estimate(state[0], truth[step], nx)
        return finalize_metrics(method, rmse, cfg, None, None, None)

    if method in {"enkf", "ensf_direct", "ensf_lr", "oracle_alpha"}:
        ensemble = scenario.ensemble_initial.copy()
        theta = scenario.theta_true if method == "oracle_alpha" else 0.0
        rmse[0] = evaluate_estimate(ensemble.mean(axis=0), truth[0], nx)
        for step in range(1, n_steps + 1):
            ensemble = propagate_batch(
                ensemble,
                theta,
                scenario.times[step - 1],
                cfg,
                rng,
                stochastic=True,
                noise_draw=scenario.forecast_noise[step - 1],
            )
            if step in scenario.observations:
                observation = scenario.observations[step]
                if method == "enkf":
                    ensemble = enkf_update(ensemble, observation, obs_idx, cfg.obs_noise, rng)
                elif method == "ensf_direct":
                    ensemble = ensf_update_direct(ensemble, observation, obs_idx, cfg, rng)
                else:
                    ensemble = ensf_update_lr(ensemble, observation, obs_idx, cfg, rng)
            rmse[step] = evaluate_estimate(ensemble.mean(axis=0), truth[step], nx)
        return finalize_metrics(method, rmse, cfg, None, None, None)

    if method == "joint_param_enkf":
        ensemble = scenario.ensemble_initial.copy()
        theta_min = float(np.min(scenario.theta_grid))
        theta_max = float(np.max(scenario.theta_grid))
        theta_ensemble = np.linspace(theta_min, theta_max, cfg.ensemble_size)
        rng.shuffle(theta_ensemble)
        rmse[0] = evaluate_estimate(ensemble.mean(axis=0), truth[0], nx)
        for step in range(1, n_steps + 1):
            # Small random walk avoids parameter collapse while remaining bounded.
            theta_ensemble = np.clip(
                theta_ensemble + 0.002 * rng.normal(size=theta_ensemble.size),
                theta_min,
                theta_max,
            )
            ensemble = propagate_memberwise_theta(
                ensemble,
                theta_ensemble,
                scenario.times[step - 1],
                cfg,
                rng,
                scenario.forecast_noise[step - 1],
            )
            if step in scenario.observations:
                ensemble, theta_ensemble = joint_parameter_enkf_update(
                    ensemble,
                    theta_ensemble,
                    scenario.observations[step],
                    obs_idx,
                    cfg.obs_noise,
                    rng,
                    (theta_min, theta_max),
                )
                alpha_estimate = theta_to_alpha(float(theta_ensemble.mean()), cfg)
                continuous_alpha_history.append(alpha_estimate)
                alpha_times.append(float(scenario.times[step]))
            rmse[step] = evaluate_estimate(ensemble.mean(axis=0), truth[step], nx)
        alpha_estimate = theta_to_alpha(float(theta_ensemble.mean()), cfg)
        alpha_best = float(scenario.alpha_grid[np.argmin(np.abs(scenario.alpha_grid - alpha_estimate))])
        return finalize_metrics(method, rmse, cfg, alpha_best, alpha_estimate, None)

    # Multi-branch methods.
    branches = (
        scenario.branch_initial_independent.copy()
        if method == "alpha_ensf_lr"
        else scenario.branch_initial.copy()
    )
    # PCE uses a shadow forecast bank for parameter evidence. These branches
    # are never nudged by the observations, so parameter evidence remains
    # identifiable instead of being erased by state assimilation.
    evidence_branches = (
        scenario.branch_initial.copy()
        if method == "alpha_ensf_lr_pce"
        else None
    )
    log_weights = np.zeros(cfg.n_alpha, dtype=float)
    weights = softmax(log_weights)
    history_ensemble: list[list[np.ndarray]] = [[] for _ in range(cfg.n_alpha)]
    history_observation: list[np.ndarray] = []

    def combined_estimate() -> np.ndarray:
        means = branches.mean(axis=1)
        return np.sum(weights[:, None] * means, axis=0)

    rmse[0] = evaluate_estimate(combined_estimate(), truth[0], nx)

    for step in range(1, n_steps + 1):
        for q, theta in enumerate(scenario.theta_grid):
            branches[q] = propagate_batch(
                branches[q],
                float(theta),
                scenario.times[step - 1],
                cfg,
                rng,
                stochastic=True,
                noise_draw=scenario.forecast_noise[step - 1],
            )
            if evidence_branches is not None:
                evidence_branches[q] = propagate_batch(
                    evidence_branches[q],
                    float(theta),
                    scenario.times[step - 1],
                    cfg,
                    rng,
                    stochastic=True,
                    noise_draw=scenario.forecast_noise[step - 1],
                )

        if step in scenario.observations:
            observation = scenario.observations[step]
            evidence_source = evidence_branches if evidence_branches is not None else branches
            branch_observations = [
                evidence_source[q][:, obs_idx].copy() for q in range(cfg.n_alpha)
            ]

            if method == "alpha_ensf_lr_pce":
                for q in range(cfg.n_alpha):
                    history_ensemble[q].append(branch_observations[q])
                    history_ensemble[q] = history_ensemble[q][-evidence_cfg.window :]
                history_observation.append(observation.copy())
                history_observation = history_observation[-evidence_cfg.window :]

                # Sensors and time points where alpha branches separate receive
                # more evidence weight. This is data-independent with respect to
                # the unknown truth and uses only predictive branch spread.
                sensitivity_now = alpha_sensitivity_weights(
                    branch_observations, evidence_cfg.sensitivity_floor
                )
                sensitivity_window = np.tile(sensitivity_now, len(history_observation))
                log_likelihood = np.empty(cfg.n_alpha)
                stacked_observation = np.concatenate(history_observation)
                for q in range(cfg.n_alpha):
                    stacked_ensemble = np.concatenate(history_ensemble[q], axis=1)
                    log_likelihood[q] = gaussian_log_evidence(
                        stacked_ensemble,
                        stacked_observation,
                        cfg.obs_noise,
                        evidence_cfg.shrinkage,
                        sensitivity_window,
                    )
                centered = log_likelihood - np.mean(log_likelihood)
                log_weights = (
                    evidence_cfg.forgetting * log_weights
                    + evidence_cfg.temperature * centered
                )
                weights = softmax(log_weights)
                weights = np.maximum(weights, evidence_cfg.weight_floor)
                weights /= weights.sum()
                # Mild entropy mixing prevents early irreversible elimination.
                weights = (
                    (1.0 - evidence_cfg.entropy_mix) * weights
                    + evidence_cfg.entropy_mix / cfg.n_alpha
                )
                weights /= weights.sum()
                log_weights = np.log(weights)
            else:
                log_scores = np.array(
                    [
                        original_compatibility_log(item, observation, cfg.obs_noise)
                        for item in branch_observations
                    ]
                )
                log_weights += cfg.credibility_rate * log_scores
                log_weights -= np.max(log_weights)
                weights = softmax(log_weights)

            if method != "alpha_only":
                # PCE also pairs the reverse-diffusion random numbers across
                # alpha branches. This preserves fair branch comparison after
                # each analysis update and removes artificial path ranking
                # caused by branch-specific sampler noise.
                paired_analysis_seed = cfg.seed + 10_000_000 + step
                for q in range(cfg.n_alpha):
                    branch_rng = (
                        np.random.default_rng(paired_analysis_seed)
                        if method == "alpha_ensf_lr_pce"
                        else rng
                    )
                    branches[q] = ensf_update_lr(
                        branches[q], observation, obs_idx, cfg, branch_rng
                    )

            alpha_history.append(weights.copy())
            alpha_times.append(float(scenario.times[step]))
            continuous_alpha_history.append(
                continuous_alpha_estimate(scenario.alpha_grid, np.log(np.maximum(weights, 1.0e-300)))
            )

        rmse[step] = evaluate_estimate(combined_estimate(), truth[step], nx)

    alpha_best = float(scenario.alpha_grid[int(np.argmax(weights))])
    alpha_continuous = continuous_alpha_estimate(
        scenario.alpha_grid, np.log(np.maximum(weights, 1.0e-300))
    )
    entropy = float(-np.sum(weights * np.log(np.maximum(weights, 1.0e-300))))
    return finalize_metrics(
        method,
        rmse,
        cfg,
        alpha_best,
        alpha_continuous,
        entropy,
        alpha_history=np.asarray(alpha_history),
        alpha_times=np.asarray(alpha_times),
        final_weights=weights,
    )


def finalize_metrics(
    method: str,
    rmse: np.ndarray,
    cfg: Config,
    alpha_best: float | None,
    alpha_continuous: float | None,
    alpha_entropy: float | None,
    alpha_history: np.ndarray | None = None,
    alpha_times: np.ndarray | None = None,
    final_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "method": method,
        "label": METHOD_LABELS[method],
        "mean_rmse": float(np.mean(rmse)),
        "final_rmse": float(rmse[-1]),
        "peak_rmse": float(np.max(rmse)),
    }
    if alpha_best is not None:
        metrics.update(
            {
                "alpha_best": float(alpha_best),
                "alpha_continuous": float(alpha_continuous),
                "alpha_top1_correct": bool(abs(alpha_best - cfg.alpha_true) < 1.0e-12),
                "alpha_abs_error": float(abs(alpha_continuous - cfg.alpha_true)),
                "alpha_entropy": None if alpha_entropy is None else float(alpha_entropy),
            }
        )
    return {
        "metrics": metrics,
        "rmse": rmse,
        "alpha_history": alpha_history,
        "alpha_times": alpha_times,
        "final_weights": final_weights,
    }


def bootstrap_mean_ci(values: np.ndarray, seed: int, n_bootstrap: int = 5000) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(n_bootstrap, values.size))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def run_benchmark(
    mode: str,
    n_seeds: int,
    base_seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    trajectories: dict[str, list[np.ndarray]] = {method: [] for method in ALL_METHODS}

    for seed_index in range(n_seeds):
        cfg = replace(make_config(mode), seed=base_seed + seed_index, filter_variant="lr")
        scenario = generate_scenario(cfg)
        seed_results: dict[str, dict[str, Any]] = {}
        for method in ALL_METHODS:
            result = run_method(scenario, method)  # type: ignore[arg-type]
            seed_results[method] = result
            trajectories[method].append(result["rmse"])

        deterministic_rmse = seed_results["deterministic"]["metrics"]["mean_rmse"]
        for method in ALL_METHODS:
            metrics = seed_results[method]["metrics"]
            record = {
                "seed": cfg.seed,
                **metrics,
                "reduction_vs_deterministic_percent": float(
                    100.0 * (1.0 - metrics["mean_rmse"] / max(deterministic_rmse, 1.0e-15))
                ),
            }
            records.append(record)

        improved = seed_results["alpha_ensf_lr_pce"]["metrics"]
        old = seed_results["alpha_ensf_lr"]["metrics"]
        print(
            f"[{seed_index + 1:02d}/{n_seeds}] seed={cfg.seed} "
            f"PCE_RMSE={improved['mean_rmse']:.6g}, "
            f"PCE_alpha_hit={improved.get('alpha_top1_correct')}, "
            f"old_alpha_hit={old.get('alpha_top1_correct')}"
        )

    fieldnames = sorted({key for record in records for key in record.keys()})
    with (output_dir / "benchmark_runs.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    summaries: dict[str, Any] = {}
    for method in ALL_METHODS:
        method_records = [record for record in records if record["method"] == method]
        rmse = np.array([float(record["mean_rmse"]) for record in method_records])
        final_rmse = np.array([float(record["final_rmse"]) for record in method_records])
        reduction = np.array(
            [float(record["reduction_vs_deterministic_percent"]) for record in method_records]
        )
        summary: dict[str, Any] = {
            "label": METHOD_LABELS[method],
            "mean_rmse_mean": float(rmse.mean()),
            "mean_rmse_std": float(rmse.std(ddof=1)),
            "mean_rmse_95ci": bootstrap_mean_ci(rmse, base_seed + stable_offset(method)),
            "final_rmse_mean": float(final_rmse.mean()),
            "reduction_vs_deterministic_mean_percent": float(reduction.mean()),
            "win_rate_vs_deterministic_percent": float(100.0 * np.mean(reduction > 0.0)),
        }
        if method_has_alpha(method):
            alpha_records = [record for record in method_records if record.get("alpha_best") not in (None, "")]
            top1 = np.array([float(bool(record["alpha_top1_correct"])) for record in alpha_records])
            error = np.array([float(record["alpha_abs_error"]) for record in alpha_records])
            summary.update(
                {
                    "alpha_top1_accuracy_percent": float(100.0 * top1.mean()),
                    "alpha_top1_95ci_percent": [100.0 * value for value in bootstrap_mean_ci(top1, base_seed + 77 + stable_offset(method))],
                    "alpha_continuous_mae": float(error.mean()),
                }
            )
        summaries[method] = summary

    proposed_rmse = np.array(
        [record["mean_rmse"] for record in records if record["method"] == "alpha_ensf_lr_pce"],
        dtype=float,
    )
    paired_vs: dict[str, Any] = {}
    for method in BASELINE_METHODS + ["alpha_ensf_lr"]:
        baseline_rmse = np.array(
            [record["mean_rmse"] for record in records if record["method"] == method],
            dtype=float,
        )
        improvement = 100.0 * (1.0 - proposed_rmse / np.maximum(baseline_rmse, 1.0e-15))
        paired_vs[method] = {
            "mean_improvement_percent": float(improvement.mean()),
            "improvement_95ci_percent": bootstrap_mean_ci(
                improvement, base_seed + 999 + stable_offset(method)
            ),
            "win_rate_percent": float(100.0 * np.mean(proposed_rmse < baseline_rmse)),
        }

    summary_payload = {
        "mode": mode,
        "n_seeds": n_seeds,
        "base_seed": base_seed,
        "methods": summaries,
        "proposed_pce_paired_comparison": paired_vs,
        "alpha_method": {
            "name": "PCE: paired cumulative predictive evidence",
            "features": [
                "paired branch ensembles and paired sampler noise",
                "shadow forecast bank separated from state assimilation",
                "proper Gaussian predictive log evidence",
                "covariance shrinkage for small ensembles",
                "sequential cumulative evidence with temperature scaling",
                "continuous quadratic alpha refinement",
            ],
        },
    }
    with (output_dir / "benchmark_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary_payload, file, ensure_ascii=False, indent=2)

    make_benchmark_figures(records, summaries, trajectories, output_dir, n_seeds)
    write_latex_table(summaries, output_dir)
    return summary_payload


def make_benchmark_figures(
    records: list[dict[str, Any]],
    summaries: dict[str, Any],
    trajectories: dict[str, list[np.ndarray]],
    output_dir: Path,
    n_seeds: int,
) -> None:
    methods_for_box = [*BASELINE_METHODS, "alpha_ensf_lr", "alpha_ensf_lr_pce"]
    values = [
        [float(record["mean_rmse"]) for record in records if record["method"] == method]
        for method in methods_for_box
    ]
    labels = [METHOD_LABELS[method] for method in methods_for_box]

    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    try:
        ax.boxplot(values, tick_labels=labels, showmeans=True)
    except TypeError:
        ax.boxplot(values, labels=labels, showmeans=True)
    ax.set_ylabel("time-mean RMSE")
    ax.set_title(f"Six baselines and proposed methods over {n_seeds} paired seeds")
    ax.tick_params(axis="x", rotation=28)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "22_six_baselines_rmse_boxplot.png", dpi=180)
    plt.close(fig)

    alpha_methods = ["alpha_only", "joint_param_enkf", "alpha_ensf_lr", "alpha_ensf_lr_pce"]
    accuracy = [summaries[method]["alpha_top1_accuracy_percent"] for method in alpha_methods]
    mae = [summaries[method]["alpha_continuous_mae"] for method in alpha_methods]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.bar([METHOD_LABELS[m] for m in alpha_methods], accuracy)
    ax.axhline(100.0 / 7.0, linestyle="--", linewidth=1.2, label="7-path random guess")
    ax.set_ylim(0.0, 100.0)
    ax.set_ylabel("alpha Top-1 accuracy (%)")
    ax.set_title("Alpha-path identification accuracy")
    ax.tick_params(axis="x", rotation=22)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "23_alpha_identification_accuracy.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.bar([METHOD_LABELS[m] for m in alpha_methods], mae)
    ax.set_ylabel("continuous alpha MAE")
    ax.set_title("Continuous alpha estimation error")
    ax.tick_params(axis="x", rotation=22)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "24_alpha_continuous_mae.png", dpi=180)
    plt.close(fig)

    selected = ["enkf", "ensf_lr", "alpha_only", "oracle_alpha", "alpha_ensf_lr", "alpha_ensf_lr_pce"]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for method in selected:
        mean_curve = np.mean(np.stack(trajectories[method], axis=0), axis=0)
        ax.plot(mean_curve, label=METHOD_LABELS[method])
    ax.set_xlabel("time-step index")
    ax.set_ylabel("mean RMSE")
    ax.set_title(f"Mean RMSE trajectories over {n_seeds} seeds")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "25_mean_rmse_trajectories.png", dpi=180)
    plt.close(fig)


def write_latex_table(summaries: dict[str, Any], output_dir: Path) -> None:
    order = [*BASELINE_METHODS, "alpha_ensf_lr", "alpha_ensf_lr_pce"]
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & Mean RMSE & Reduction vs. deterministic & $\alpha$ Top-1 & $\alpha$ MAE \\",
        r"\midrule",
    ]
    for method in order:
        item = summaries[method]
        top1 = item.get("alpha_top1_accuracy_percent")
        mae = item.get("alpha_continuous_mae")
        top1_text = "--" if top1 is None else f"{top1:.1f}\\%"
        mae_text = "--" if mae is None else f"{mae:.4f}"
        lines.append(
            f"{METHOD_LABELS[method]} & {item['mean_rmse_mean']:.4e} & "
            f"{item['reduction_vs_deterministic_mean_percent']:.2f}\\% & "
            f"{top1_text} & {mae_text} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output_dir / "benchmark_table.tex").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Six-baseline benchmark and PCE alpha-path identification"
    )
    parser.add_argument("--mode", choices=["quick", "balanced", "large"], default="quick")
    parser.add_argument("--n-seeds", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=20260803)
    parser.add_argument("--output", default="results_benchmark_v3_50seeds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_seeds < 30:
        raise ValueError("The formal benchmark requires at least 30 seeds.")
    summary = run_benchmark(
        args.mode,
        args.n_seeds,
        args.base_seed,
        Path(args.output),
    )
    print("\nBenchmark completed.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nOutput: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
