from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from . import wave_protocol as wave_core
from . import adaptive_evidence
from . import comparison_methods
from . import wave_evaluation
from pce_assimilation.ensemble_filters import denkf_analysis
from pce_assimilation.observations import SparseObservation
from pce_assimilation.systems.one_dimensional import Heat1D
from . import spring_heat as sh


CASES = ("wave", "spring", "heat")
METHODS = ("aug_enkf", "bma_static", "pce", "apce")
LABELS = {
    "aug_enkf": "Aug-EnKF",
    "bma_static": "BMA",
    "pce": "PCE",
    "apce": "APCE",
}


class AugmentedSparseObservation:
    def __init__(self, indices: torch.Tensor) -> None:
        self.indices = indices

    def __call__(self, augmented_ensemble: torch.Tensor) -> torch.Tensor:
        return augmented_ensemble.index_select(-1, self.indices)


def softmax_np(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    output = np.exp(shifted)
    return output / np.sum(output)


def finite_metrics(row: dict[str, Any]) -> bool:
    keys = ("nrmse", "rmse", "crps", "coverage_90", "interval_width_90")
    return all(key in row and math.isfinite(float(row[key])) for key in keys)


def metric_prefix(row: dict[str, Any], prefix: str, metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        row[f"{prefix}_{key}"] = float(value)


def weighted_alpha_mean(alpha_grid: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(alpha_grid * weights / np.sum(weights)))


def adaptive_local_alpha_grid(
    alpha_grid: np.ndarray,
    weights: np.ndarray,
    log_weights: np.ndarray,
    *,
    points: int = 11,
) -> np.ndarray:
    alpha = np.asarray(alpha_grid, dtype=float)
    w = np.asarray(weights, dtype=float)
    score = np.asarray(log_weights, dtype=float)
    if alpha.ndim != 1 or alpha.size == 0:
        raise ValueError("alpha_grid must be one-dimensional and non-empty")
    if alpha.size == 1:
        return alpha.copy()
    if points < 5:
        points = 5
    if points % 2 == 0:
        points += 1

    step = float(np.median(np.diff(alpha)))
    concentration = float(np.max(w)) if w.size else 0.0
    coarse_mean = weighted_alpha_mean(alpha, w if w.size else np.full(alpha.shape, 1.0 / alpha.size))
    coarse_peak = refined_alpha_from_scores(alpha, score)
    coarse_map = float(alpha[int(np.argmax(w))]) if w.size else float(alpha[0])
    center = float(np.clip(0.45 * coarse_mean + 0.35 * coarse_peak + 0.20 * coarse_map, alpha[0], alpha[-1]))
    radius = step * (1.05 + 0.35 * (1.0 - concentration))
    lower = max(float(alpha[0]), center - radius)
    upper = min(float(alpha[-1]), center + radius)

    if w.size:
        topk = np.sort(alpha[np.argsort(w)[-min(3, alpha.size):]])
        lower = min(lower, max(float(alpha[0]), float(topk[0] - 0.25 * step)))
        upper = max(upper, min(float(alpha[-1]), float(topk[-1] + 0.25 * step)))

    if upper - lower < step:
        lower = max(float(alpha[0]), center - step)
        upper = min(float(alpha[-1]), center + step)

    local = np.linspace(lower, upper, points, dtype=float)
    local = np.unique(np.concatenate([local, [center]]))
    local.sort()
    return np.clip(local, float(alpha[0]), float(alpha[-1]))


def refined_alpha_from_scores(alpha_grid: np.ndarray, scores: np.ndarray) -> float:
    """Local continuous refinement from discrete evidence scores.

    The estimate stays inside the candidate span. A local concave quadratic is
    used around the best grid node; a local softmax-weighted mean handles
    non-concave fits and boundary cases.
    """

    alpha = np.asarray(alpha_grid, dtype=float)
    score = np.asarray(scores, dtype=float)
    if alpha.ndim != 1 or score.shape != alpha.shape:
        raise ValueError("alpha_grid and scores must be one-dimensional arrays of equal length")
    best = int(np.argmax(score))
    if alpha.size < 3:
        return weighted_alpha_mean(alpha, softmax_np(score))

    if best == 0:
        indices = np.array([0, 1, 2])
    elif best == alpha.size - 1:
        indices = np.array([alpha.size - 3, alpha.size - 2, alpha.size - 1])
    else:
        indices = np.array([best - 1, best, best + 1])

    x = alpha[indices]
    y = score[indices]
    try:
        a, b, _ = np.polyfit(x, y, deg=2)
        if a < -1.0e-12:
            vertex = -b / (2.0 * a)
            if x[0] <= vertex <= x[-1]:
                return float(np.clip(vertex, alpha[0], alpha[-1]))
    except np.linalg.LinAlgError:
        pass

    local_weights = softmax_np(y)
    return float(np.clip(weighted_alpha_mean(x, local_weights), alpha[0], alpha[-1]))


def make_local_wave_scenario(
    assets: comparison_methods.WaveScenarioAssets,
    local_alpha_grid: np.ndarray,
) -> wave_core.Scenario:
    base_cfg = wave_core.make_config()
    base_scenario = wave_core.scenario_from_assets(assets, base_cfg)
    local_alpha = np.asarray(local_alpha_grid, dtype=float)
    local_cfg = replace(base_scenario.cfg, n_alpha=int(local_alpha.size))
    theta_grid = np.asarray([wave_core.alpha_to_theta(float(alpha), local_cfg) for alpha in local_alpha], dtype=float)
    branch_initial = np.repeat(base_scenario.ensemble_initial[None, :, :], local_cfg.n_alpha, axis=0)
    return wave_core.Scenario(
        cfg=local_cfg,
        x=base_scenario.x,
        times=base_scenario.times,
        alpha_grid=local_alpha,
        theta_grid=theta_grid,
        theta_true=wave_core.alpha_to_theta(assets.alpha_true, local_cfg),
        truth_states=base_scenario.truth_states,
        observations=base_scenario.observations,
        observation_indices=base_scenario.observation_indices,
        ensemble_initial=base_scenario.ensemble_initial.copy(),
        branch_initial=branch_initial,
        branch_initial_independent=branch_initial.copy(),
        truth_noise=base_scenario.truth_noise,
        forecast_noise=base_scenario.forecast_noise,
    )


def tile_alpha_members(
    alpha_grid: torch.Tensor,
    ensemble_size: int,
    generator: torch.Generator,
    *,
    jitter: float,
) -> torch.Tensor:
    repeats = int(math.ceil(ensemble_size / int(alpha_grid.numel())))
    tiled = alpha_grid.repeat(repeats)[:ensemble_size].clone()
    if jitter > 0.0:
        tiled = tiled + jitter * torch.randn(
            tiled.shape,
            dtype=tiled.dtype,
            device=tiled.device,
            generator=generator,
        )
    return tiled.clamp(float(alpha_grid[0]), float(alpha_grid[-1]))


def random_walk_alpha(
    alpha: torch.Tensor,
    generator: torch.Generator,
    *,
    lower: float,
    upper: float,
    std: float,
) -> torch.Tensor:
    if std <= 0.0:
        return alpha
    return (
        alpha
        + std
        * torch.randn(
            alpha.shape,
            dtype=alpha.dtype,
            device=alpha.device,
            generator=generator,
        )
    ).clamp(lower, upper)


def spring_heat_primary(ensemble: torch.Tensor, scenario: sh.Scenario) -> torch.Tensor:
    return ensemble.index_select(-1, scenario.primary_indices)


def propagate_spring_heat_memberwise(
    scenario: sh.Scenario,
    state_ensemble: torch.Tensor,
    alpha_members: torch.Tensor,
    step: int,
) -> torch.Tensor:
    config = scenario.config
    system = sh.make_system(config)
    if isinstance(system, Heat1D):
        system.grid = system.grid.to(state_ensemble.device)
    pieces = []
    for member in range(state_ensemble.shape[0]):
        pieces.append(
            sh.step_with_noise(
                system,
                state_ensemble[member : member + 1],
                step * config.dt,
                config.dt,
                float(alpha_members[member]),
                scenario.forecast_noise[step, member : member + 1],
            )
        )
    return torch.cat(pieces, dim=0)


def run_spring_heat_aug_enkf(
    scenario: sh.Scenario,
    device: torch.device,
) -> dict[str, Any]:
    config = scenario.config
    ensemble = scenario.initial_ensemble.clone()
    generator = torch.Generator(device=device).manual_seed(config.seed + 741_001)
    alpha = tile_alpha_members(
        scenario.alpha_grid,
        config.ensemble_size,
        generator,
        jitter=0.012,
    )
    operator = AugmentedSparseObservation(scenario.observation_indices)
    covariance = config.obs_noise**2 * torch.eye(
        scenario.observation_indices.numel(), dtype=ensemble.dtype, device=device
    )
    weights = torch.full((config.ensemble_size,), 1.0 / config.ensemble_size, dtype=ensemble.dtype, device=device)
    metrics = sh.TrajectoryMetrics()
    started = time.perf_counter()
    analyses = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(config.steps + 1):
        metrics.add(spring_heat_primary(ensemble, scenario), spring_heat_primary(scenario.truth[step], scenario), weights)
        if step == config.steps:
            break
        alpha = random_walk_alpha(
            alpha,
            generator,
            lower=float(scenario.alpha_grid[0]),
            upper=float(scenario.alpha_grid[-1]),
            std=0.004,
        )
        ensemble = propagate_spring_heat_memberwise(scenario, ensemble, alpha, step)
        if step + 1 not in scenario.observations:
            continue
        analyses += 1
        augmented = torch.cat([ensemble, alpha[:, None]], dim=-1)
        updated = denkf_analysis(augmented, scenario.observations[step + 1], operator, covariance)
        ensemble = updated[:, :-1]
        alpha = updated[:, -1].clamp(float(scenario.alpha_grid[0]), float(scenario.alpha_grid[-1]))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = metrics.finalize()
    alpha_estimate = float(alpha.mean())
    result.update(
        runtime_seconds=float(time.perf_counter() - started),
        forward_member_steps=int(config.steps * config.ensemble_size),
        peak_gpu_memory_mb=(
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else 0.0
        ),
        alpha_estimate=alpha_estimate,
        alpha_absolute_error=abs(alpha_estimate - config.alpha_true),
        alpha_spread=float(alpha.std(unbiased=True)),
        analyses=analyses,
    )
    return result


def run_spring_heat_bma_static(
    scenario: sh.Scenario,
    device: torch.device,
) -> dict[str, Any]:
    config = scenario.config
    system = sh.make_system(config)
    if isinstance(system, Heat1D):
        system.grid = system.grid.to(device)
    path_count = int(scenario.alpha_grid.numel())
    branches = scenario.initial_ensemble.unsqueeze(0).repeat(path_count, 1, 1)
    log_weights = torch.zeros(path_count, dtype=branches.dtype, device=device)
    weights = torch.softmax(log_weights, dim=0)
    operator = SparseObservation(scenario.observation_indices)
    covariance = config.obs_noise**2 * torch.eye(
        scenario.observation_indices.numel(), dtype=branches.dtype, device=device
    )
    metrics = sh.TrajectoryMetrics()
    started = time.perf_counter()
    analyses = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(config.steps + 1):
        flat = branches.reshape(-1, branches.shape[-1])
        flat_weights = weights.unsqueeze(1).expand(-1, config.ensemble_size).reshape(-1) / config.ensemble_size
        metrics.add(spring_heat_primary(flat, scenario), spring_heat_primary(scenario.truth[step], scenario), flat_weights)
        if step == config.steps:
            break
        for path_index, alpha in enumerate(scenario.alpha_grid):
            branches[path_index] = sh.step_with_noise(
                system,
                branches[path_index],
                step * config.dt,
                config.dt,
                float(alpha),
                scenario.forecast_noise[step],
            )
        if step + 1 not in scenario.observations:
            continue
        analyses += 1
        observation = scenario.observations[step + 1]
        branch_observations = torch.stack([operator(branch) for branch in branches])
        evidence = torch.stack(
            [
                sh.evidence_score(
                    branch_observations[path_index],
                    observation,
                    config.obs_noise,
                    config.evidence_shrinkage,
                    None,
                )
                for path_index in range(path_count)
            ]
        )
        log_weights = log_weights + (evidence - evidence.mean())
        weights = torch.softmax(log_weights, dim=0)
        for path_index in range(path_count):
            branches[path_index] = denkf_analysis(branches[path_index], observation, operator, covariance)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    alpha_estimate = float((scenario.alpha_grid * weights).sum())
    result = metrics.finalize()
    result.update(
        runtime_seconds=float(time.perf_counter() - started),
        forward_member_steps=int(config.steps * path_count * config.ensemble_size),
        peak_gpu_memory_mb=(
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else 0.0
        ),
        alpha_estimate=alpha_estimate,
        alpha_absolute_error=abs(alpha_estimate - config.alpha_true),
        alpha_final_entropy=float(sh.entropy(weights)),
        alpha_map=float(scenario.alpha_grid[int(torch.argmax(weights))]),
        analyses=analyses,
    )
    return result


def run_spring_heat_pce_pass(
    scenario: sh.Scenario,
    method: str,
    device: torch.device,
    *,
    alpha_grid_override: torch.Tensor | np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    config = scenario.config
    system = sh.make_system(config)
    if isinstance(system, Heat1D):
        system.grid = system.grid.to(device)
    alpha_grid = (
        torch.as_tensor(alpha_grid_override, dtype=scenario.alpha_grid.dtype, device=device)
        if alpha_grid_override is not None
        else scenario.alpha_grid
    )
    path_count = int(alpha_grid.numel())
    branches = scenario.initial_ensemble.unsqueeze(0).repeat(path_count, 1, 1)
    shadow = branches.clone()
    log_weights = torch.zeros(path_count, dtype=branches.dtype, device=device)
    weights = torch.softmax(log_weights, dim=0)
    operator = SparseObservation(scenario.observation_indices)
    covariance = config.obs_noise**2 * torch.eye(
        scenario.observation_indices.numel(), dtype=branches.dtype, device=device
    )
    metrics = sh.TrajectoryMetrics()
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(config.steps + 1):
        flat = branches.reshape(-1, branches.shape[-1])
        flat_weights = weights.unsqueeze(1).expand(-1, config.ensemble_size).reshape(-1) / config.ensemble_size
        metrics.add(spring_heat_primary(flat, scenario), spring_heat_primary(scenario.truth[step], scenario), flat_weights)
        if step == config.steps:
            break
        for path_index, alpha in enumerate(alpha_grid):
            branches[path_index] = sh.step_with_noise(
                system,
                branches[path_index],
                step * config.dt,
                config.dt,
                float(alpha),
                scenario.forecast_noise[step],
            )
            shadow[path_index] = sh.step_with_noise(
                system,
                shadow[path_index],
                step * config.dt,
                config.dt,
                float(alpha),
                scenario.forecast_noise[step],
            )
        if step + 1 not in scenario.observations:
            continue
        observation = scenario.observations[step + 1]
        shadow_observations = torch.stack([operator(branch) for branch in shadow])
        dimension_weights = None
        if method == "apce":
            between = shadow_observations.mean(dim=1).var(dim=0, unbiased=True)
            dimension_weights = 0.35 + 0.65 * between / between.max().clamp_min(1.0e-12)
        evidence = torch.stack(
            [
                sh.evidence_score(
                    shadow_observations[path_index],
                    observation,
                    config.obs_noise,
                    config.evidence_shrinkage,
                    dimension_weights,
                )
                for path_index in range(path_count)
            ]
        )
        centered = evidence - evidence.mean()
        if method == "pce":
            log_weights = log_weights + config.pce_temperature * centered
        elif method == "apce":
            entropy_ratio = float(sh.entropy(weights) / math.log(path_count))
            temperature = float(
                np.clip(
                    config.apce_temperature * entropy_ratio**0.75,
                    config.apce_min_temperature,
                    config.apce_temperature,
                )
            )
            log_weights = config.apce_forgetting * log_weights + temperature * centered
        else:
            raise ValueError(method)
        weights = torch.softmax(log_weights, dim=0)
        if method == "apce":
            progress = (step + 1) / max(config.steps, 1)
            target_entropy = config.apce_entropy_floor + 0.20 * (1.0 - progress)
            weights = sh.entropy_project(weights, target_entropy)
            log_weights = weights.clamp_min(1.0e-300).log()
        for path_index in range(path_count):
            branches[path_index] = denkf_analysis(branches[path_index], observation, operator, covariance)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = metrics.finalize()
    result.update(
        runtime_seconds=float(time.perf_counter() - started),
        forward_member_steps=int(2 * config.steps * path_count * config.ensemble_size),
        peak_gpu_memory_mb=(
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else 0.0
        ),
    )
    return (
        result,
        alpha_grid.detach().cpu().numpy(),
        weights.detach().cpu().numpy(),
        log_weights.detach().cpu().numpy(),
    )


def run_spring_heat_pce(
    scenario: sh.Scenario,
    method: str,
    device: torch.device,
) -> dict[str, Any]:
    source_method = method
    coarse_pass, alpha_grid, weights, log_weights = run_spring_heat_pce_pass(
        scenario, source_method, device
    )
    local_alpha_grid = adaptive_local_alpha_grid(alpha_grid, weights, log_weights, points=11)
    local_pass, local_grid, local_weights, local_log_weights = run_spring_heat_pce_pass(
        scenario,
        source_method,
        device,
        alpha_grid_override=local_alpha_grid,
    )
    alpha_estimate = weighted_alpha_mean(local_grid, local_weights)
    alpha_map = float(alpha_grid[int(np.argmax(weights))])
    alpha_mean = weighted_alpha_mean(alpha_grid, weights)
    row = dict(local_pass)
    metric_prefix(row, "coarse_pass", coarse_pass)
    row.update(
        alpha_estimate=float(alpha_estimate),
        alpha_absolute_error=abs(float(alpha_estimate) - scenario.config.alpha_true),
        coarse_alpha_map=alpha_map,
        coarse_alpha_map_error=abs(alpha_map - scenario.config.alpha_true),
        coarse_alpha_mean=alpha_mean,
        coarse_alpha_mean_error=abs(alpha_mean - scenario.config.alpha_true),
        local_alpha_grid_min=float(np.min(local_grid)),
        local_alpha_grid_max=float(np.max(local_grid)),
        local_alpha_grid_points=int(local_grid.size),
        coarse_pass_forward_member_steps=int(coarse_pass["forward_member_steps"]),
        local_pass_forward_member_steps=int(local_pass["forward_member_steps"]),
        forward_member_steps=int(coarse_pass["forward_member_steps"] + local_pass["forward_member_steps"]),
        runtime_seconds=float(coarse_pass["runtime_seconds"] + local_pass["runtime_seconds"]),
        alpha_final_entropy=float(
            -np.sum(np.maximum(local_weights, 1.0e-300) * np.log(np.maximum(local_weights, 1.0e-300)))
        ),
        alpha_final_map=float(local_grid[int(np.argmax(local_weights))]),
        alpha_final_quadratic=float(refined_alpha_from_scores(local_grid, local_log_weights)),
    )
    return row


def wave_memberwise_propagate(
    states: torch.Tensor,
    alpha_members: torch.Tensor,
    step: int,
    assets: comparison_methods.WaveScenarioAssets,
    cfg: wave_core.Config,
) -> torch.Tensor:
    state_np = states.detach().cpu().numpy()
    alpha_np = alpha_members.detach().cpu().numpy()
    output = np.empty_like(state_np)
    for member in range(state_np.shape[0]):
        output[member : member + 1] = wave_core.propagate_batch(
            state_np[member : member + 1],
            wave_core.alpha_to_theta(float(alpha_np[member]), cfg),
            float(assets.times[step]),
            cfg,
            np.random.default_rng(assets.seed),
            stochastic=True,
            noise_draw=assets.forecast_noise[step, member : member + 1],
        )
    return torch.as_tensor(output, dtype=states.dtype, device=states.device)


def run_wave_aug_enkf(
    assets: comparison_methods.WaveScenarioAssets,
    device: torch.device,
) -> dict[str, Any]:
    cfg = wave_evaluation.configuration(assets)
    dtype = torch.float64
    ensemble = torch.as_tensor(assets.initial_ensemble, dtype=dtype, device=device)
    truth = torch.as_tensor(assets.truth_states, dtype=dtype, device=device)
    indices = torch.as_tensor(assets.observation_indices, dtype=torch.int64, device=device)
    alpha_grid = torch.as_tensor(np.linspace(cfg.alpha_min, cfg.alpha_max, cfg.n_alpha), dtype=dtype, device=device)
    generator = torch.Generator(device=device).manual_seed(assets.seed + 741_101)
    alpha = tile_alpha_members(alpha_grid, assets.ensemble_size, generator, jitter=0.012)
    operator = AugmentedSparseObservation(indices)
    covariance = cfg.obs_noise**2 * torch.eye(indices.numel(), dtype=dtype, device=device)
    weights = torch.full((assets.ensemble_size,), 1.0 / assets.ensemble_size, dtype=dtype, device=device)
    metrics = wave_evaluation.MetricAccumulator(assets.nx)
    started = time.perf_counter()
    analyses = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(assets.n_steps + 1):
        metrics.add(ensemble, truth[step], weights)
        if step == assets.n_steps:
            break
        alpha = random_walk_alpha(alpha, generator, lower=float(alpha_grid[0]), upper=float(alpha_grid[-1]), std=0.004)
        ensemble = wave_memberwise_propagate(ensemble, alpha, step, assets, cfg)
        if not assets.observation_mask[step + 1]:
            continue
        analyses += 1
        observation = torch.as_tensor(assets.observations[step + 1], dtype=dtype, device=device)
        augmented = torch.cat([ensemble, alpha[:, None]], dim=-1)
        updated = denkf_analysis(augmented, observation, operator, covariance)
        ensemble = updated[:, :-1]
        alpha = updated[:, -1].clamp(float(alpha_grid[0]), float(alpha_grid[-1]))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = metrics.finalize()
    result.update(nrmse=result.pop("displacement_nrmse"), rmse=result.pop("displacement_rmse"))
    alpha_estimate = float(alpha.mean())
    result.update(
        runtime_seconds=float(time.perf_counter() - started),
        forward_member_steps=int(assets.n_steps * assets.ensemble_size),
        peak_gpu_memory_mb=(
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else 0.0
        ),
        alpha_estimate=alpha_estimate,
        alpha_absolute_error=abs(alpha_estimate - assets.alpha_true),
        alpha_spread=float(alpha.std(unbiased=True)),
        analyses=analyses,
    )
    return result


def run_wave_bma_static(
    assets: comparison_methods.WaveScenarioAssets,
    device: torch.device,
) -> dict[str, Any]:
    cfg = wave_evaluation.configuration(assets)
    scenario = wave_core.scenario_from_assets(assets, cfg)
    branches = scenario.branch_initial.copy()
    log_weights = np.zeros(cfg.n_alpha, dtype=float)
    weights = softmax_np(log_weights)
    truth = torch.as_tensor(assets.truth_states, dtype=torch.float64, device=device)
    indices = torch.as_tensor(assets.observation_indices, dtype=torch.int64, device=device)
    operator = SparseObservation(indices)
    covariance = cfg.obs_noise**2 * torch.eye(indices.numel(), dtype=torch.float64, device=device)
    metrics = wave_evaluation.MetricAccumulator(assets.nx)
    evidence_config = adaptive_evidence.EvidenceConfig(
        gaussian_evidence=True,
        shrinkage=0.35,
        sensitivity_floor=1.0,
    )
    started = time.perf_counter()
    analyses = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(assets.n_steps + 1):
        flat = torch.as_tensor(branches.reshape(-1, branches.shape[-1]), dtype=torch.float64, device=device)
        member_weights = torch.as_tensor(
            np.repeat(weights / cfg.ensemble_size, cfg.ensemble_size),
            dtype=torch.float64,
            device=device,
        )
        metrics.add(flat, truth[step], member_weights)
        if step == assets.n_steps:
            break
        for q, theta in enumerate(scenario.theta_grid):
            branches[q] = wave_core.propagate_batch(
                branches[q],
                float(theta),
                float(assets.times[step]),
                cfg,
                np.random.default_rng(assets.seed),
                stochastic=True,
                noise_draw=assets.forecast_noise[step],
            )
        if not assets.observation_mask[step + 1]:
            continue
        analyses += 1
        observation = assets.observations[step + 1]
        branch_observations = [branches[q][:, scenario.observation_indices].copy() for q in range(cfg.n_alpha)]
        evidence = adaptive_evidence.evidence_vector(branch_observations, observation, cfg.obs_noise, evidence_config)
        log_weights = log_weights + (evidence - np.mean(evidence))
        weights = softmax_np(log_weights)
        observation_t = torch.as_tensor(observation, dtype=torch.float64, device=device)
        for q in range(cfg.n_alpha):
            branch_t = torch.as_tensor(branches[q], dtype=torch.float64, device=device)
            branches[q] = (
                denkf_analysis(branch_t, observation_t, operator, covariance)
                .detach()
                .cpu()
                .numpy()
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = metrics.finalize()
    result.update(nrmse=result.pop("displacement_nrmse"), rmse=result.pop("displacement_rmse"))
    alpha_estimate = weighted_alpha_mean(scenario.alpha_grid, weights)
    result.update(
        runtime_seconds=float(time.perf_counter() - started),
        forward_member_steps=int(assets.n_steps * cfg.n_alpha * cfg.ensemble_size),
        peak_gpu_memory_mb=(
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else 0.0
        ),
        alpha_estimate=alpha_estimate,
        alpha_absolute_error=abs(alpha_estimate - assets.alpha_true),
        alpha_final_entropy=float(-np.sum(np.maximum(weights, 1.0e-300) * np.log(np.maximum(weights, 1.0e-300)))),
        alpha_map=float(scenario.alpha_grid[int(np.argmax(weights))]),
        analyses=analyses,
    )
    return result


def run_wave_pce_pass(
    assets: comparison_methods.WaveScenarioAssets,
    method: str,
    device: torch.device,
    *,
    alpha_grid_override: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    cfg = wave_evaluation.configuration(assets)
    scenario = (
        wave_core.scenario_from_assets(assets, cfg)
        if alpha_grid_override is None
        else make_local_wave_scenario(assets, np.asarray(alpha_grid_override, dtype=float))
    )
    cfg = scenario.cfg
    config = adaptive_evidence.EVIDENCE_CONFIGS[method]
    branches = scenario.branch_initial.copy()
    shadow = branches.copy()
    weights = np.full(cfg.n_alpha, 1.0 / cfg.n_alpha)
    log_weights = np.zeros(cfg.n_alpha)
    truth = torch.as_tensor(assets.truth_states, dtype=torch.float64, device=device)
    metrics = wave_evaluation.MetricAccumulator(assets.nx)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(assets.n_steps + 1):
        flat = torch.as_tensor(branches.reshape(-1, branches.shape[-1]), dtype=torch.float64, device=device)
        member_weights = torch.as_tensor(
            np.repeat(weights / cfg.ensemble_size, cfg.ensemble_size),
            dtype=torch.float64,
            device=device,
        )
        metrics.add(flat, truth[step], member_weights)
        if step == assets.n_steps:
            break
        for q, theta in enumerate(scenario.theta_grid):
            branches[q] = wave_core.propagate_batch(
                branches[q],
                float(theta),
                float(assets.times[step]),
                cfg,
                np.random.default_rng(assets.seed),
                stochastic=True,
                noise_draw=assets.forecast_noise[step],
            )
            shadow[q] = wave_core.propagate_batch(
                shadow[q],
                float(theta),
                float(assets.times[step]),
                cfg,
                np.random.default_rng(assets.seed),
                stochastic=True,
                noise_draw=assets.forecast_noise[step],
            )
        if not assets.observation_mask[step + 1]:
            continue
        log_weights, weights = wave_evaluation.update_adaptive_weights(
            branches,
            shadow,
            assets.observations[step + 1],
            cfg,
            scenario,
            config,
            log_weights,
            step + 1,
        )
        paired_seed = cfg.seed + 10_000_000 + step + 1
        for q in range(cfg.n_alpha):
            branches[q] = wave_evaluation.ensf_update_lr(
                branches[q],
                assets.observations[step + 1],
                scenario.observation_indices,
                cfg,
                np.random.default_rng(paired_seed),
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = metrics.finalize()
    result.update(
        nrmse=result.pop("displacement_nrmse"),
        rmse=result.pop("displacement_rmse"),
        runtime_seconds=float(time.perf_counter() - started),
        forward_member_steps=int(2 * assets.n_steps * cfg.n_alpha * cfg.ensemble_size),
        peak_gpu_memory_mb=(
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else 0.0
        ),
    )
    return result, scenario.alpha_grid.copy(), weights.copy(), np.log(np.maximum(weights, 1.0e-300))


def run_wave_pce(
    assets: comparison_methods.WaveScenarioAssets,
    method: str,
    device: torch.device,
) -> dict[str, Any]:
    source_method = method
    coarse_pass, alpha_grid, weights, log_weights = run_wave_pce_pass(
        assets, source_method, device
    )
    local_alpha_grid = adaptive_local_alpha_grid(alpha_grid, weights, log_weights, points=11)
    local_pass, local_grid, local_weights, local_log_weights = run_wave_pce_pass(
        assets,
        source_method,
        device,
        alpha_grid_override=local_alpha_grid,
    )
    alpha_estimate = weighted_alpha_mean(local_grid, local_weights)
    alpha_map = float(alpha_grid[int(np.argmax(weights))])
    alpha_mean = weighted_alpha_mean(alpha_grid, weights)
    row = dict(local_pass)
    metric_prefix(row, "coarse_pass", coarse_pass)
    row.update(
        alpha_estimate=float(alpha_estimate),
        alpha_absolute_error=abs(float(alpha_estimate) - assets.alpha_true),
        coarse_alpha_map=alpha_map,
        coarse_alpha_map_error=abs(alpha_map - assets.alpha_true),
        coarse_alpha_mean=alpha_mean,
        coarse_alpha_mean_error=abs(alpha_mean - assets.alpha_true),
        local_alpha_grid_min=float(np.min(local_grid)),
        local_alpha_grid_max=float(np.max(local_grid)),
        local_alpha_grid_points=int(local_grid.size),
        coarse_pass_forward_member_steps=int(coarse_pass["forward_member_steps"]),
        local_pass_forward_member_steps=int(local_pass["forward_member_steps"]),
        forward_member_steps=int(coarse_pass["forward_member_steps"] + local_pass["forward_member_steps"]),
        runtime_seconds=float(coarse_pass["runtime_seconds"] + local_pass["runtime_seconds"]),
        alpha_final_entropy=float(
            -np.sum(np.maximum(local_weights, 1.0e-300) * np.log(np.maximum(local_weights, 1.0e-300)))
        ),
        alpha_final_map=float(local_grid[int(np.argmax(local_weights))]),
        alpha_final_quadratic=float(refined_alpha_from_scores(local_grid, local_log_weights)),
    )
    return row


def run_case_method_seed(
    case: str,
    method: str,
    seed: int,
    device: torch.device,
    *,
    protocol: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    if case == "wave":
        assets = comparison_methods.make_wave_assets(seed)
        if method == "aug_enkf":
            result = run_wave_aug_enkf(assets, device)
        elif method == "bma_static":
            result = run_wave_bma_static(assets, device)
        elif method in {"pce", "apce"}:
            result = run_wave_pce(assets, method, device)
        else:
            raise ValueError(method)
        state_dim = int(assets.truth_states.shape[1])
        primary_dim = int(assets.nx)
        observation_count = int(assets.observation_indices.size)
        observation_indices = ",".join(str(int(v)) for v in assets.observation_indices.tolist())
        steps = int(assets.n_steps)
        dt = float(assets.times[1] - assets.times[0])
        obs_interval = int(np.flatnonzero(assets.observation_mask)[0])
        ensemble_size = int(assets.ensemble_size)
        obs_noise = float(wave_evaluation.configuration(assets).obs_noise)
        alpha_true = float(assets.alpha_true)
    else:
        scenario = sh.generate_scenario(sh.config_for_case(case, seed), device)  # type: ignore[arg-type]
        if method == "aug_enkf":
            result = run_spring_heat_aug_enkf(scenario, device)
        elif method == "bma_static":
            result = run_spring_heat_bma_static(scenario, device)
        elif method in {"pce", "apce"}:
            result = run_spring_heat_pce(scenario, method, device)
        else:
            raise ValueError(method)
        state_dim = int(scenario.truth.shape[1])
        primary_dim = int(scenario.primary_indices.numel())
        observation_count = int(scenario.observation_indices.numel())
        observation_indices = ",".join(str(int(v)) for v in scenario.observation_indices.detach().cpu().tolist())
        steps = int(scenario.config.steps)
        dt = float(scenario.config.dt)
        obs_interval = int(scenario.config.obs_interval)
        ensemble_size = int(scenario.config.ensemble_size)
        obs_noise = float(scenario.config.obs_noise)
        alpha_true = float(scenario.config.alpha_true)

    result.update(
        case=case,
        method=method,
        label=LABELS[method],
        seed=int(seed),
        state_dim=state_dim,
        primary_dim=primary_dim,
        observation_count=observation_count,
        observation_indices=observation_indices,
        steps=steps,
        dt=dt,
        observation_interval=obs_interval,
        ensemble_size=ensemble_size,
        observation_noise=obs_noise,
        alpha_true=alpha_true,
        valid=finite_metrics(result),
        protocol=(
            protocol
            if protocol is not None
            else "classical-systems-published-protocol"
        ),
        elapsed_seconds_wall=float(time.perf_counter() - started),
        status="completed",
    )
    return result
