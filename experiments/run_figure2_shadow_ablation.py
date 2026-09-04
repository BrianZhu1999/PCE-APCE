from __future__ import annotations

import argparse
import csv
import hashlib
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

import run_benchmark_v3 as wave_v3
import run_benchmark_v4 as wave_v4
from experiments import run_modern_baseline_admission as modern
from experiments import run_wave_repair_validation as wave_repair
from hilda_da.baselines import denkf_analysis
from hilda_da.observations import SparseObservation
from hilda_da.systems.one_dimensional import Heat1D
from paper_experiments import run_spring_heat_gate as sh


CASES = ("wave", "spring", "heat")
NEW_METHODS = ("pce_baseline", "apce_full", "apce_no_dim", "apce_fixed_temp", "apce_no_forgetting", "apce_no_entropy")
LABELS = {
    "pce_baseline": "PCE",
    "apce_full": "APCE-full",
    "apce_no_dim": "APCE-no-dimension-weighting",
    "apce_fixed_temp": "APCE-fixed-temperature",
    "apce_no_forgetting": "APCE-no-forgetting",
    "apce_no_entropy": "APCE-no-entropy-floor",
}


def ablation_family_source(method: str) -> tuple[str, str]:
    if method not in NEW_METHODS:
        raise ValueError(f"unsupported component ablation method: {method}")
    return ("pce" if method == "pce_baseline" else "apce", "shadow")


def component_options(method: str) -> dict[str, bool]:
    if method == "pce_baseline":
        return {"dimension_weighting": False, "adaptive_temperature": False, "forgetting": False, "entropy_floor": False}
    return {
        "dimension_weighting": method != "apce_no_dim",
        "adaptive_temperature": method != "apce_fixed_temp",
        "forgetting": method != "apce_no_forgetting",
        "entropy_floor": method != "apce_no_entropy",
    }


def pairwise_observation_separation(observations: torch.Tensor | list[np.ndarray]) -> float:
    """Mean pairwise distance between candidate observation means."""
    if isinstance(observations, torch.Tensor):
        means = observations.mean(dim=1)
        distances = torch.pdist(means)
        return float(distances.mean().detach().cpu()) if distances.numel() else 0.0
    means = np.asarray([np.asarray(item).mean(axis=0) for item in observations], dtype=float)
    if means.shape[0] < 2:
        return 0.0
    return float(np.mean([np.linalg.norm(means[i] - means[j]) for i in range(means.shape[0]) for j in range(i + 1, means.shape[0])]))


def mechanism_summary(trace: list[dict[str, float]], weights: np.ndarray, *, steps: int) -> dict[str, Any]:
    erasure = np.asarray([item["erasure_ratio"] for item in trace], dtype=float)
    early = np.asarray([item["max_weight"] for item in trace if item["progress"] <= 0.5], dtype=float)
    entropy_values = np.asarray([item["normalized_entropy"] for item in trace], dtype=float)
    return {
        "mechanism_observation_count": int(len(trace)),
        "mechanism_mean_erasure_ratio": float(np.nanmean(erasure)) if erasure.size else float("nan"),
        "mechanism_median_erasure_ratio": float(np.nanmedian(erasure)) if erasure.size else float("nan"),
        "mechanism_erasure_ratio_p90": float(np.nanpercentile(erasure, 90)) if erasure.size else float("nan"),
        "mechanism_mean_normalized_entropy": float(np.nanmean(entropy_values)) if entropy_values.size else float("nan"),
        "mechanism_min_normalized_entropy": float(np.nanmin(entropy_values)) if entropy_values.size else float("nan"),
        "early_collapse": bool(np.any(early > 0.95)),
        "early_collapse_rate_proxy": float(np.mean(early > 0.95)) if early.size else 0.0,
        "final_max_weight": float(np.max(weights)) if np.asarray(weights).size else float("nan"),
        "mechanism_trace_arrays": {
            key: [float(item[key]) for item in trace]
            for key in ("step", "progress", "shadow_separation", "analysis_separation", "erasure_ratio", "score_range", "normalized_entropy", "max_weight")
        },
    }


class AugmentedSparseObservation:
    def __init__(self, indices: torch.Tensor) -> None:
        self.indices = indices

    def __call__(self, augmented_ensemble: torch.Tensor) -> torch.Tensor:
        return augmented_ensemble.index_select(-1, self.indices)


def source_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for folder in ("hilda_da", "experiments", "paper_experiments"):
        for path in sorted((root / folder).glob("*.py")):
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def softmax_np(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    output = np.exp(shifted)
    return output / np.sum(output)


def finite_metrics(row: dict[str, Any]) -> bool:
    keys = ("nrmse", "rmse", "crps", "coverage_90", "interval_width_90")
    return all(key in row and math.isfinite(float(row[key])) for key in keys)


def metric_prefix(row: dict[str, Any], prefix: str, metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
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

    The routine deliberately stays inside the original candidate span.  It first
    tries a local concave quadratic maximum around the best grid node, including
    one-sided three-point fits at boundaries.  If the fit is not concave or the
    vertex is outside the local bracket, it falls back to a local softmax-weighted
    mean.  This gives a continuous diagnostic without pretending that the coarse
    grid contains enough information for unconstrained exact parameter recovery.
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
    assets: modern.WaveScenarioAssets,
    local_alpha_grid: np.ndarray,
) -> wave_v3.Scenario:
    base_cfg = wave_v3.make_config("quick")
    base_scenario = wave_v3.scenario_from_assets(assets, base_cfg)
    local_alpha = np.asarray(local_alpha_grid, dtype=float)
    local_cfg = replace(base_scenario.cfg, n_alpha=int(local_alpha.size))
    theta_grid = np.asarray([wave_v3.alpha_to_theta(float(alpha), local_cfg) for alpha in local_alpha], dtype=float)
    branch_initial = np.repeat(base_scenario.ensemble_initial[None, :, :], local_cfg.n_alpha, axis=0)
    return wave_v3.Scenario(
        cfg=local_cfg,
        x=base_scenario.x,
        times=base_scenario.times,
        alpha_grid=local_alpha,
        theta_grid=theta_grid,
        theta_true=wave_v3.alpha_to_theta(assets.alpha_true, local_cfg),
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


def run_spring_heat_fixed_alpha_filter(
    scenario: sh.Scenario,
    alpha: float,
    device: torch.device,
) -> dict[str, Any]:
    config = scenario.config
    system = sh.make_system(config)
    if isinstance(system, Heat1D):
        system.grid = system.grid.to(device)
    ensemble = scenario.initial_ensemble.clone()
    operator = SparseObservation(scenario.observation_indices)
    covariance = config.obs_noise**2 * torch.eye(
        scenario.observation_indices.numel(), dtype=ensemble.dtype, device=device
    )
    weights = torch.full((config.ensemble_size,), 1.0 / config.ensemble_size, dtype=ensemble.dtype, device=device)
    metrics = sh.TrajectoryMetrics()
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(config.steps + 1):
        metrics.add(spring_heat_primary(ensemble, scenario), spring_heat_primary(scenario.truth[step], scenario), weights)
        if step == config.steps:
            break
        ensemble = sh.step_with_noise(
            system,
            ensemble,
            step * config.dt,
            config.dt,
            float(alpha),
            scenario.forecast_noise[step],
        )
        if step + 1 in scenario.observations:
            ensemble = denkf_analysis(ensemble, scenario.observations[step + 1], operator, covariance)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = metrics.finalize()
    result.update(
        runtime_seconds=float(time.perf_counter() - started),
        forward_member_steps=int(config.steps * config.ensemble_size),
        peak_gpu_memory_mb=(
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else 0.0
        ),
    )
    return result


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
    family, evidence_source = ablation_family_source(method)
    options = component_options(method)
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
    mechanism_trace: list[dict[str, float]] = []
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
        analysis_observations = torch.stack([operator(branch) for branch in branches])
        shadow_separation = pairwise_observation_separation(shadow_observations)
        analysis_separation = pairwise_observation_separation(analysis_observations)
        analysis_shadow_ratio = analysis_separation / max(shadow_separation, 1.0e-12)
        score_observations = shadow_observations if evidence_source == "shadow" else analysis_observations
        dimension_weights = None
        if family == "apce" and options["dimension_weighting"]:
            between = score_observations.mean(dim=1).var(dim=0, unbiased=True)
            dimension_weights = 0.35 + 0.65 * between / between.max().clamp_min(1.0e-12)
        evidence = torch.stack(
            [
                sh.evidence_score(
                    score_observations[path_index],
                    observation,
                    config.obs_noise,
                    config.evidence_shrinkage,
                    dimension_weights,
                )
                for path_index in range(path_count)
            ]
        )
        centered = evidence - evidence.mean()
        if family == "pce":
            log_weights = log_weights + config.pce_temperature * centered
        elif family == "apce":
            entropy_ratio = float(sh.entropy(weights) / math.log(path_count))
            temperature = config.apce_temperature
            if options["adaptive_temperature"]:
                temperature = float(np.clip(config.apce_temperature * entropy_ratio**0.75, config.apce_min_temperature, config.apce_temperature))
            forgetting = config.apce_forgetting if options["forgetting"] else 1.0
            log_weights = forgetting * log_weights + temperature * centered
        else:
            raise ValueError(method)
        weights = torch.softmax(log_weights, dim=0)
        if family == "apce" and options["entropy_floor"]:
            progress = (step + 1) / max(config.steps, 1)
            target_entropy = config.apce_entropy_floor + 0.20 * (1.0 - progress)
            weights = sh.entropy_project(weights, target_entropy)
            log_weights = weights.clamp_min(1.0e-300).log()
        normalized_entropy = float(sh.entropy(weights) / math.log(path_count))
        trace_item = {
            "step": float(step + 1),
            "progress": float((step + 1) / max(config.steps, 1)),
            "shadow_separation": float(shadow_separation),
            "analysis_separation": float(analysis_separation),
            "analysis_shadow_ratio": float(analysis_shadow_ratio),
            "erasure_ratio": float("nan"),
            "score_range": float((evidence.max() - evidence.min()).detach().cpu()),
            "normalized_entropy": normalized_entropy,
            "max_weight": float(weights.max().detach().cpu()),
        }
        for path_index in range(path_count):
            branches[path_index] = denkf_analysis(branches[path_index], observation, operator, covariance)
        post_observations = torch.stack([operator(branch) for branch in branches])
        post_separation = pairwise_observation_separation(post_observations)
        trace_item["post_analysis_separation"] = float(post_separation)
        trace_item["erasure_ratio"] = float(post_separation / max(analysis_separation, 1.0e-12))
        mechanism_trace.append(trace_item)
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
    result.update(mechanism_summary(mechanism_trace, weights.detach().cpu().numpy(), steps=config.steps))
    mechanism_arrays = result.pop("mechanism_trace_arrays")
    result["mechanism_trace"] = mechanism_arrays
    result.update(evidence_source=evidence_source, evidence_family=family)
    return (
        result,
        alpha_grid.detach().cpu().numpy(),
        weights.detach().cpu().numpy(),
        log_weights.detach().cpu().numpy(),
    )


def run_spring_heat_refined(
    scenario: sh.Scenario,
    method: str,
    device: torch.device,
) -> dict[str, Any]:
    source_method = "pce" if method == "pce_refined" else "apce"
    first_pass, alpha_grid, weights, log_weights = run_spring_heat_pce_pass(scenario, source_method, device)
    refined_alpha = refined_alpha_from_scores(alpha_grid, log_weights)
    refined = run_spring_heat_fixed_alpha_filter(scenario, refined_alpha, device)
    alpha_map = float(alpha_grid[int(np.argmax(weights))])
    alpha_mean = weighted_alpha_mean(alpha_grid, weights)
    row = dict(refined)
    metric_prefix(row, "first_pass", first_pass)
    row.update(
        alpha_estimate=float(refined_alpha),
        alpha_absolute_error=abs(float(refined_alpha) - scenario.config.alpha_true),
        coarse_alpha_map=alpha_map,
        coarse_alpha_map_error=abs(alpha_map - scenario.config.alpha_true),
        coarse_alpha_mean=alpha_mean,
        coarse_alpha_mean_error=abs(alpha_mean - scenario.config.alpha_true),
        refinement_source_method=source_method,
        first_pass_forward_member_steps=int(first_pass["forward_member_steps"]),
        rerun_forward_member_steps=int(refined["forward_member_steps"]),
        forward_member_steps=int(first_pass["forward_member_steps"] + refined["forward_member_steps"]),
        runtime_seconds=float(first_pass["runtime_seconds"] + refined["runtime_seconds"]),
        alpha_final_entropy=float(-np.sum(np.maximum(weights, 1.0e-300) * np.log(np.maximum(weights, 1.0e-300)))),
    )
    return row


def run_spring_heat_refined_v2(
    scenario: sh.Scenario,
    method: str,
    device: torch.device,
) -> dict[str, Any]:
    coarse_pass, alpha_grid, weights, log_weights = run_spring_heat_pce_pass(
        scenario, method, device
    )
    coarse_trace = coarse_pass.pop("mechanism_trace")
    local_alpha_grid = adaptive_local_alpha_grid(alpha_grid, weights, log_weights, points=11)
    local_pass, local_grid, local_weights, local_log_weights = run_spring_heat_pce_pass(
        scenario,
        method,
        device,
        alpha_grid_override=local_alpha_grid,
    )
    local_trace = local_pass.pop("mechanism_trace")
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
        refinement_source_method=method,
        coarse_pass_forward_member_steps=int(coarse_pass["forward_member_steps"]),
        local_pass_forward_member_steps=int(local_pass["forward_member_steps"]),
        forward_member_steps=int(coarse_pass["forward_member_steps"] + local_pass["forward_member_steps"]),
        runtime_seconds=float(coarse_pass["runtime_seconds"] + local_pass["runtime_seconds"]),
        alpha_final_entropy=float(
            -np.sum(np.maximum(local_weights, 1.0e-300) * np.log(np.maximum(local_weights, 1.0e-300)))
        ),
        alpha_final_map=float(local_grid[int(np.argmax(local_weights))]),
        alpha_final_quadratic=float(refined_alpha_from_scores(local_grid, local_log_weights)),
        _mechanism_trace_arrays={
            **{f"coarse_{key}": values for key, values in coarse_trace.items()},
            **{f"local_{key}": values for key, values in local_trace.items()},
        },
    )
    return row


def wave_memberwise_propagate(
    states: torch.Tensor,
    alpha_members: torch.Tensor,
    step: int,
    assets: modern.WaveScenarioAssets,
    cfg: wave_v3.Config,
) -> torch.Tensor:
    state_np = states.detach().cpu().numpy()
    alpha_np = alpha_members.detach().cpu().numpy()
    output = np.empty_like(state_np)
    for member in range(state_np.shape[0]):
        output[member : member + 1] = wave_v3.propagate_batch(
            state_np[member : member + 1],
            wave_v3.alpha_to_theta(float(alpha_np[member]), cfg),
            float(assets.times[step]),
            cfg,
            np.random.default_rng(assets.seed),
            stochastic=True,
            noise_draw=assets.forecast_noise[step, member : member + 1],
        )
    return torch.as_tensor(output, dtype=states.dtype, device=states.device)


def run_wave_fixed_alpha_filter(
    assets: modern.WaveScenarioAssets,
    alpha: float,
    device: torch.device,
) -> dict[str, Any]:
    cfg = wave_repair.configuration(assets)
    dtype = torch.float64
    ensemble = torch.as_tensor(assets.initial_ensemble, dtype=dtype, device=device)
    truth = torch.as_tensor(assets.truth_states, dtype=dtype, device=device)
    indices = torch.as_tensor(assets.observation_indices, dtype=torch.int64, device=device)
    operator = SparseObservation(indices)
    covariance = cfg.obs_noise**2 * torch.eye(indices.numel(), dtype=dtype, device=device)
    weights = torch.full((assets.ensemble_size,), 1.0 / assets.ensemble_size, dtype=dtype, device=device)
    metrics = wave_repair.MetricAccumulator(assets.nx)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(assets.n_steps + 1):
        metrics.add(ensemble, truth[step], weights)
        if step == assets.n_steps:
            break
        ensemble = wave_repair.propagate_numpy(
            ensemble,
            wave_v3.alpha_to_theta(float(alpha), cfg),
            step,
            assets,
            cfg,
        )
        if assets.observation_mask[step + 1]:
            observation = torch.as_tensor(assets.observations[step + 1], dtype=dtype, device=device)
            ensemble = denkf_analysis(ensemble, observation, operator, covariance)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = metrics.finalize()
    result.update(
        nrmse=result.pop("displacement_nrmse"),
        rmse=result.pop("displacement_rmse"),
        runtime_seconds=float(time.perf_counter() - started),
        forward_member_steps=int(assets.n_steps * assets.ensemble_size),
        peak_gpu_memory_mb=(
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else 0.0
        ),
    )
    return result


def run_wave_aug_enkf(
    assets: modern.WaveScenarioAssets,
    device: torch.device,
) -> dict[str, Any]:
    cfg = wave_repair.configuration(assets)
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
    metrics = wave_repair.MetricAccumulator(assets.nx)
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
    assets: modern.WaveScenarioAssets,
    device: torch.device,
) -> dict[str, Any]:
    cfg = wave_repair.configuration(assets)
    scenario = wave_v3.scenario_from_assets(assets, cfg)
    branches = scenario.branch_initial.copy()
    log_weights = np.zeros(cfg.n_alpha, dtype=float)
    weights = softmax_np(log_weights)
    truth = torch.as_tensor(assets.truth_states, dtype=torch.float64, device=device)
    indices = torch.as_tensor(assets.observation_indices, dtype=torch.int64, device=device)
    operator = SparseObservation(indices)
    covariance = cfg.obs_noise**2 * torch.eye(indices.numel(), dtype=torch.float64, device=device)
    metrics = wave_repair.MetricAccumulator(assets.nx)
    evidence_config = wave_v4.V4EvidenceConfig(
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
            branches[q] = wave_v3.propagate_batch(
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
        evidence = wave_v4.evidence_vector(branch_observations, observation, cfg.obs_noise, evidence_config)
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
    assets: modern.WaveScenarioAssets,
    method: str,
    device: torch.device,
    *,
    alpha_grid_override: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    raise RuntimeError("run_wave_pce_pass is replaced by run_wave_shadow_ablation_pass")


def update_wave_shadow_ablation_weights(
    branches: np.ndarray,
    shadow: np.ndarray,
    observation: np.ndarray,
    cfg: wave_v3.Config,
    scenario: wave_v3.Scenario,
    config: wave_v4.V4EvidenceConfig,
    log_weights: np.ndarray,
    step: int,
    family: str,
    evidence_source: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    shadow_observations = [shadow[q][:, scenario.observation_indices].copy() for q in range(cfg.n_alpha)]
    analysis_observations = [branches[q][:, scenario.observation_indices].copy() for q in range(cfg.n_alpha)]
    shadow_score = wave_v4.evidence_vector(shadow_observations, observation, cfg.obs_noise, config)
    analysis_score = wave_v4.evidence_vector(analysis_observations, observation, cfg.obs_noise, config)
    progress = step / max(scenario.times.size - 1, 1)
    blend = 0.0
    if evidence_source == "shadow" and config.analysis_blend_max > 0.0 and progress > config.analysis_blend_start:
        blend = config.analysis_blend_max * min(1.0, (progress - config.analysis_blend_start) / max(1.0 - config.analysis_blend_start, 1.0e-12))
        score = (1.0 - blend) * shadow_score + blend * analysis_score
    elif evidence_source == "shadow":
        score = shadow_score
    else:
        score = analysis_score

    centered = score - np.mean(score)
    weights_before = wave_v3.softmax(log_weights)
    temperature = config.temperature
    if config.adaptive_temperature:
        entropy_ratio = wave_v4.entropy(weights_before) / max(math.log(cfg.n_alpha), 1.0e-12)
        temperature = float(np.clip(config.temperature * entropy_ratio**0.75, config.min_temperature, config.temperature))
    log_weights = config.forgetting * log_weights + temperature * centered
    weights = wave_v3.softmax(log_weights)
    weights = np.maximum(weights, config.weight_floor)
    weights /= weights.sum()
    weights = wave_v4.entropy_project(weights, wave_v4.entropy_target(progress, config))
    log_weights = np.log(np.maximum(weights, 1.0e-300))

    shadow_means = np.asarray([item.mean(axis=0) for item in shadow_observations])
    analysis_means = np.asarray([item.mean(axis=0) for item in analysis_observations])
    shadow_sep = float(np.mean([np.linalg.norm(shadow_means[i] - shadow_means[j]) for i in range(cfg.n_alpha) for j in range(i + 1, cfg.n_alpha)]))
    analysis_sep = float(np.mean([np.linalg.norm(analysis_means[i] - analysis_means[j]) for i in range(cfg.n_alpha) for j in range(i + 1, cfg.n_alpha)]))
    trace = {
        "step": float(step),
        "progress": float(progress),
        "shadow_separation": shadow_sep,
        "analysis_separation": analysis_sep,
        "analysis_shadow_ratio": analysis_sep / max(shadow_sep, 1.0e-12),
        "erasure_ratio": float("nan"),
        "score_range": float(np.max(score) - np.min(score)),
        "normalized_entropy": float(wave_v4.entropy(weights) / max(math.log(cfg.n_alpha), 1.0e-12)),
        "max_weight": float(np.max(weights)),
        "analysis_blend": float(blend),
    }
    return log_weights, weights, trace


def run_wave_shadow_ablation_pass(
    assets: modern.WaveScenarioAssets,
    method: str,
    device: torch.device,
    *,
    alpha_grid_override: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    family, evidence_source = ablation_family_source(method)
    cfg = wave_repair.configuration(assets)
    scenario = (
        wave_v3.scenario_from_assets(assets, cfg)
        if alpha_grid_override is None
        else make_local_wave_scenario(assets, np.asarray(alpha_grid_override, dtype=float))
    )
    cfg = scenario.cfg
    ablation = "A6_pce" if family == "pce" else "A7_apce"
    config = wave_v4.ABLATION_CONFIGS[ablation]
    options = component_options(method)
    if family == "apce":
        config = replace(
            config,
            sensitivity_floor=0.35 if options["dimension_weighting"] else 1.0,
            adaptive_temperature=options["adaptive_temperature"],
            forgetting=0.975 if options["forgetting"] else 1.0,
            entropy_floor_start=0.38 if options["entropy_floor"] else 0.0,
            entropy_floor_mid=0.30 if options["entropy_floor"] else 0.0,
            entropy_floor_end=0.22 if options["entropy_floor"] else 0.0,
        )
    branches = scenario.branch_initial.copy()
    shadow = branches.copy()
    weights = np.full(cfg.n_alpha, 1.0 / cfg.n_alpha)
    log_weights = np.zeros(cfg.n_alpha)
    truth = torch.as_tensor(assets.truth_states, dtype=torch.float64, device=device)
    metrics = wave_repair.MetricAccumulator(assets.nx)
    mechanism_trace: list[dict[str, float]] = []
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
            branches[q] = wave_v3.propagate_batch(
                branches[q],
                float(theta),
                float(assets.times[step]),
                cfg,
                np.random.default_rng(assets.seed),
                stochastic=True,
                noise_draw=assets.forecast_noise[step],
            )
            shadow[q] = wave_v3.propagate_batch(
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
        log_weights, weights, trace = update_wave_shadow_ablation_weights(
            branches,
            shadow,
            assets.observations[step + 1],
            cfg,
            scenario,
            config,
            log_weights,
            step + 1,
            family,
            evidence_source,
        )
        paired_seed = cfg.seed + 10_000_000 + step + 1
        for q in range(cfg.n_alpha):
            branches[q] = wave_repair.ensf_update_lr(
                branches[q],
                assets.observations[step + 1],
                scenario.observation_indices,
                cfg,
                np.random.default_rng(paired_seed),
            )
        post_observations = [branches[q][:, scenario.observation_indices].copy() for q in range(cfg.n_alpha)]
        post_sep = pairwise_observation_separation(post_observations)
        trace["post_analysis_separation"] = float(post_sep)
        trace["erasure_ratio"] = float(post_sep / max(trace["analysis_separation"], 1.0e-12))
        mechanism_trace.append(trace)
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
    result.update(mechanism_summary(mechanism_trace, weights, steps=assets.n_steps))
    mechanism_arrays = result.pop("mechanism_trace_arrays")
    result["mechanism_trace"] = mechanism_arrays
    result.update(evidence_source=evidence_source, evidence_family=family)
    return result, scenario.alpha_grid.copy(), weights.copy(), np.log(np.maximum(weights, 1.0e-300))


def run_wave_refined(
    assets: modern.WaveScenarioAssets,
    method: str,
    device: torch.device,
) -> dict[str, Any]:
    source_method = "pce" if method == "pce_refined" else "apce"
    first_pass, alpha_grid, weights, log_weights = run_wave_pce_pass(assets, source_method, device)
    refined_alpha = refined_alpha_from_scores(alpha_grid, log_weights)
    refined = run_wave_fixed_alpha_filter(assets, refined_alpha, device)
    alpha_map = float(alpha_grid[int(np.argmax(weights))])
    alpha_mean = weighted_alpha_mean(alpha_grid, weights)
    row = dict(refined)
    metric_prefix(row, "first_pass", first_pass)
    row.update(
        alpha_estimate=float(refined_alpha),
        alpha_absolute_error=abs(float(refined_alpha) - assets.alpha_true),
        coarse_alpha_map=alpha_map,
        coarse_alpha_map_error=abs(alpha_map - assets.alpha_true),
        coarse_alpha_mean=alpha_mean,
        coarse_alpha_mean_error=abs(alpha_mean - assets.alpha_true),
        refinement_source_method=source_method,
        first_pass_forward_member_steps=int(first_pass["forward_member_steps"]),
        rerun_forward_member_steps=int(refined["forward_member_steps"]),
        forward_member_steps=int(first_pass["forward_member_steps"] + refined["forward_member_steps"]),
        runtime_seconds=float(first_pass["runtime_seconds"] + refined["runtime_seconds"]),
        alpha_final_entropy=float(-np.sum(np.maximum(weights, 1.0e-300) * np.log(np.maximum(weights, 1.0e-300)))),
    )
    return row


def run_wave_refined_v2(
    assets: modern.WaveScenarioAssets,
    method: str,
    device: torch.device,
) -> dict[str, Any]:
    coarse_pass, alpha_grid, weights, log_weights = run_wave_shadow_ablation_pass(
        assets, method, device
    )
    coarse_trace = coarse_pass.pop("mechanism_trace")
    local_alpha_grid = adaptive_local_alpha_grid(alpha_grid, weights, log_weights, points=11)
    local_pass, local_grid, local_weights, local_log_weights = run_wave_shadow_ablation_pass(
        assets,
        method,
        device,
        alpha_grid_override=local_alpha_grid,
    )
    local_trace = local_pass.pop("mechanism_trace")
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
        refinement_source_method=method,
        coarse_pass_forward_member_steps=int(coarse_pass["forward_member_steps"]),
        local_pass_forward_member_steps=int(local_pass["forward_member_steps"]),
        forward_member_steps=int(coarse_pass["forward_member_steps"] + local_pass["forward_member_steps"]),
        runtime_seconds=float(coarse_pass["runtime_seconds"] + local_pass["runtime_seconds"]),
        alpha_final_entropy=float(
            -np.sum(np.maximum(local_weights, 1.0e-300) * np.log(np.maximum(local_weights, 1.0e-300)))
        ),
        alpha_final_map=float(local_grid[int(np.argmax(local_weights))]),
        alpha_final_quadratic=float(refined_alpha_from_scores(local_grid, local_log_weights)),
        _mechanism_trace_arrays={
            **{f"coarse_{key}": values for key, values in coarse_trace.items()},
            **{f"local_{key}": values for key, values in local_trace.items()},
        },
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
        assets = modern.make_wave_assets(seed)
        if method not in NEW_METHODS:
            raise ValueError(method)
        result = run_wave_refined_v2(assets, method, device)
        state_dim = int(assets.truth_states.shape[1])
        primary_dim = int(assets.nx)
        observation_count = int(assets.observation_indices.size)
        observation_indices = ",".join(str(int(v)) for v in assets.observation_indices.tolist())
        steps = int(assets.n_steps)
        dt = float(assets.times[1] - assets.times[0])
        obs_interval = int(np.flatnonzero(assets.observation_mask)[0])
        ensemble_size = int(assets.ensemble_size)
        obs_noise = float(wave_repair.configuration(assets).obs_noise)
        alpha_true = float(assets.alpha_true)
    else:
        scenario = sh.generate_scenario(sh.config_for_case(case, seed), device)  # type: ignore[arg-type]
        if method not in NEW_METHODS:
            raise ValueError(method)
        result = run_spring_heat_refined_v2(scenario, method, device)
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
            else (
                "figure2-apce-component-ablation-5paired-seeds-20260825"
            )
        ),
        source_hash=source_hash(PROJECT_ROOT),
        torch_version=torch.__version__,
        cuda_available=bool(torch.cuda.is_available()),
        device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        elapsed_seconds_wall=float(time.perf_counter() - started),
        status="completed",
    )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_existing_rows(path: Path, cases: set[str], seeds: set[int]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                seed = int(float(row.get("seed", "")))
            except ValueError:
                continue
            # Formal Figure 2 jobs use date-prefixed seeds such as 2026080700,
            # whereas this reviewer gate uses the paired-seed coordinates 0--4.
            # Match both representations without altering the original seed field.
            paired_seed = seed if seed in seeds else seed % 100
            if row.get("case") in cases and paired_seed in seeds:
                copied = dict(row)
                copied["protocol_source"] = "existing_figure2_formal_20260807"
                copied["paired_seed"] = paired_seed
                rows.append(copied)
    return rows


def numeric(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "nrmse",
        "rmse",
        "crps",
        "coverage_90",
        "interval_width_90",
        "alpha_absolute_error",
        "runtime_seconds",
        "forward_member_steps",
    )
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if "case" not in row or "method" not in row:
            continue
        groups.setdefault((str(row["case"]), str(row["method"])), []).append(row)
    summary: list[dict[str, Any]] = []
    for (case, method), group in sorted(groups.items()):
        item: dict[str, Any] = {
            "case": case,
            "method": method,
            "label": group[0].get("label", method),
            "runs": len(group),
            "valid_runs": sum(str(row.get("valid", "True")).lower() in {"true", "1", "yes"} for row in group),
        }
        for metric in metrics:
            values = [numeric(row.get(metric)) for row in group]
            clean = [value for value in values if value is not None]
            if clean:
                item[f"{metric}_mean"] = float(np.mean(clean))
                item[f"{metric}_std"] = float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0
        summary.append(item)
    return summary


def write_report(path: Path, summary: list[dict[str, Any]], cases: Iterable[str]) -> None:
    lines = [
        "# Figure 2 reviewer-gate smoke report",
        "",
        "Purpose: admit reviewer-risk baselines for joint state--parameter estimation and Bayesian multiple-model filtering, and test whether PCE/APCE evidence supports continuous off-grid alpha refinement.",
        "",
        "Cases: " + ", ".join(cases) + ".",
        "",
        "| case | method | runs | nRMSE | CRPS | 90% coverage | interval width | alpha error | runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            "| {case} | {label} | {runs} | {nrmse:.4g} | {crps:.4g} | {coverage:.4g} | {width:.4g} | {alpha:.4g} | {runtime:.4g} |".format(
                case=item["case"],
                label=item.get("label", item["method"]),
                runs=item["runs"],
                nrmse=item.get("nrmse_mean", float("nan")),
                crps=item.get("crps_mean", float("nan")),
                coverage=item.get("coverage_90_mean", float("nan")),
                width=item.get("interval_width_90_mean", float("nan")),
                alpha=item.get("alpha_absolute_error_mean", float("nan")),
                runtime=item.get("runtime_seconds_mean", float("nan")),
            )
        )
    lines.extend(
        [
            "",
            "Interpretation rule:",
            "",
            "- If Aug-state EnKF or Bayesian model averaging beats APCE/PCE on a case, the main claim must explicitly narrow or the method section must justify the remaining advantage mechanism.",
            "- If refined PCE/APCE improves alpha error but not state/probabilistic metrics, continuous refinement should be described as a diagnostic extension rather than a core performance enhancer.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_ints(values: list[str]) -> list[int]:
    output: list[int] = []
    for value in values:
        if ":" in value:
            start, end = value.split(":", 1)
            output.extend(range(int(start), int(end)))
        else:
            output.append(int(value))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Reviewer-risk Figure 2 smoke: Aug-EnKF, BMA, and continuous alpha refinement.")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--methods", nargs="+", choices=NEW_METHODS, default=list(NEW_METHODS))
    parser.add_argument("--seeds", nargs="+", default=["0:5"], help="Seeds or Python-style ranges, e.g. 0:5.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--existing-source-data",
        type=Path,
        default=PROJECT_ROOT / "ncs_chinese_submission" / "source_data" / "figure2_run_source_data_20260807.csv",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    seeds = parse_ints(args.seeds)
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for case in args.cases:
        for seed in seeds:
            for method in args.methods:
                print(f"RUN case={case} seed={seed} method={method}", flush=True)
                try:
                    row = run_case_method_seed(case, method, seed, device)
                    rows.append(row)
                    print("RESULT", json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
                except Exception as exc:
                    failure = {
                        "case": case,
                        "seed": seed,
                        "method": method,
                        "label": LABELS[method],
                        "status": "failed",
                        "valid": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    failures.append(failure)
                    rows.append(failure)
                    print("FAILED", json.dumps(failure, ensure_ascii=False, sort_keys=True), flush=True)

    new_runs = args.output / "new_method_runs.csv"
    write_csv(new_runs, rows)
    (args.output / "new_method_runs.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    existing = read_existing_rows(args.existing_source_data, set(args.cases), set(seeds))
    combined = existing + rows
    if combined:
        write_csv(args.output / "combined_with_existing_figure2.csv", combined)
    summary = summarize(combined if combined else rows)
    write_csv(args.output / "summary.csv", summary)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(args.output / "REVIEWER_GATE_SMOKE_REPORT.md", summary, args.cases)
    if failures:
        (args.output / "failures.json").write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
    print("WROTE", str(args.output.resolve()), flush=True)


if __name__ == "__main__":
    main()
