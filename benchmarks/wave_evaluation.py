from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from . import wave_protocol
from . import adaptive_evidence
from .wave_assets import WaveScenarioAssets
from pce_assimilation.evidence import AlphaEvidenceTracker
from pce_assimilation.refinement import (
    apce_calibration_parameters,
    local_alpha_grid,
    numpy_regrid_paths,
    refined_alpha_map,
)
from pce_assimilation.ensemble_filters import denkf_analysis, letkf_analysis
from pce_assimilation.config import AlphaConfig, AssimilationConfig
from pce_assimilation.assimilation import PCEFilter
from pce_assimilation.metrics import (
    weighted_central_interval_coverage_width,
    weighted_ensemble_crps,
)
from pce_assimilation.observation_flow import analytic_posterior_mixture
from pce_assimilation.observations import SparseObservation
from pce_assimilation.comparison_filters import ensf_lr_ridge_analysis
from .wave_model import ensf_update_lr


def configuration(assets: WaveScenarioAssets) -> wave_protocol.Config:
    observation_steps = np.flatnonzero(assets.observation_mask)
    interval = int(observation_steps[0]) if observation_steps.size else assets.n_steps + 1
    return dataclasses.replace(
        wave_protocol.make_config(),
        seed=assets.seed,
        nx=assets.nx,
        ensemble_size=assets.ensemble_size,
        alpha_true=assets.alpha_true,
        t_end=float(assets.times[-1]),
        dt=float(assets.times[1] - assets.times[0]),
        n_sensors=int(assets.observation_indices.size),
        obs_interval=interval,
    )


class MetricAccumulator:
    def __init__(self, nx: int) -> None:
        self.nx = nx
        self.squared_error = 0.0
        self.truth_square = 0.0
        self.crps: list[float] = []
        self.coverage: list[float] = []
        self.width: list[float] = []
        self.points = 0

    def add(
        self,
        ensemble: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        displacement = ensemble[:, : self.nx]
        truth = target[: self.nx]
        weights = weights / weights.sum()
        estimate = (weights.unsqueeze(1) * displacement).sum(0)
        self.squared_error += float((estimate - truth).square().sum())
        self.truth_square += float(truth.square().sum())
        self.points += truth.numel()
        self.crps.append(float(weighted_ensemble_crps(displacement, truth, weights)))
        coverage, width = weighted_central_interval_coverage_width(
            displacement, truth, weights, level=0.90
        )
        self.coverage.append(float(coverage))
        self.width.append(float(width))

    def finalize(self) -> dict[str, float]:
        return {
            "displacement_nrmse": math.sqrt(self.squared_error / max(self.truth_square, 1.0e-30)),
            "displacement_rmse": math.sqrt(self.squared_error / max(self.points, 1)),
            "crps": float(np.mean(self.crps)),
            "coverage_90": float(np.mean(self.coverage)),
            "interval_width_90": float(np.mean(self.width)),
        }


def flat_path_distribution(
    branches: torch.Tensor,
    path_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    members = branches.shape[1]
    ensemble = branches.reshape(-1, branches.shape[-1])
    weights = path_weights.unsqueeze(1).expand(-1, members).reshape(-1) / members
    return ensemble, weights.to(ensemble)


def propagate_numpy(
    ensemble: torch.Tensor,
    theta: float,
    step: int,
    assets: WaveScenarioAssets,
    cfg: wave_protocol.Config,
) -> torch.Tensor:
    result = wave_protocol.propagate_batch(
        ensemble.detach().cpu().numpy(),
        theta,
        float(assets.times[step]),
        cfg,
        np.random.default_rng(assets.seed),
        stochastic=True,
        noise_draw=assets.forecast_noise[step],
    )
    return torch.as_tensor(result, dtype=ensemble.dtype, device=ensemble.device)


def run_single_path(
    assets: WaveScenarioAssets,
    method: str,
    device: torch.device,
) -> dict[str, Any]:
    cfg = configuration(assets)
    dtype = torch.float64
    ensemble = torch.as_tensor(assets.initial_ensemble, dtype=dtype, device=device)
    truth = torch.as_tensor(assets.truth_states, dtype=dtype, device=device)
    indices = torch.as_tensor(assets.observation_indices, dtype=torch.int64, device=device)
    operator = SparseObservation(indices)
    covariance = cfg.obs_noise**2 * torch.eye(indices.numel(), dtype=dtype, device=device)
    generator = torch.Generator(device=device).manual_seed(assets.seed + wave_protocol.stable_offset(method))
    metrics = MetricAccumulator(assets.nx)
    weights = torch.full((assets.ensemble_size,), 1.0 / assets.ensemble_size, dtype=dtype, device=device)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(assets.n_steps + 1):
        metrics.add(ensemble, truth[step], weights)
        if step == assets.n_steps:
            break
        ensemble = propagate_numpy(
            ensemble, wave_protocol.alpha_to_theta(0.50, cfg), step, assets, cfg
        )
        if not assets.observation_mask[step + 1]:
            continue
        observation = torch.as_tensor(
            assets.observations[step + 1], dtype=dtype, device=device
        )
        if method == "denkf":
            ensemble = denkf_analysis(ensemble, observation, operator, covariance)
        elif method == "letkf":
            ensemble = letkf_analysis(ensemble, observation, operator, covariance)
        elif method == "ensf_lr_ridge":
            ensemble = ensf_lr_ridge_analysis(
                ensemble, observation, operator, covariance, generator=generator
            )
        else:
            raise ValueError(method)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = metrics.finalize()
    result.update(
        runtime_seconds=time.perf_counter() - started,
        forward_member_steps=assets.n_steps * assets.ensemble_size,
        peak_gpu_memory_mb=(
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda" else 0.0
        ),
        alpha_estimate=None,
        alpha_absolute_error=None,
    )
    return result


def trace_single_path(
    assets: WaveScenarioAssets,
    method: str,
    device: torch.device,
) -> np.ndarray:
    """Return the posterior mean state trajectory for a fixed-alpha filter."""

    cfg = configuration(assets)
    dtype = torch.float64
    ensemble = torch.as_tensor(assets.initial_ensemble, dtype=dtype, device=device)
    indices = torch.as_tensor(assets.observation_indices, dtype=torch.int64, device=device)
    operator = SparseObservation(indices)
    covariance = cfg.obs_noise**2 * torch.eye(indices.numel(), dtype=dtype, device=device)
    generator = torch.Generator(device=device).manual_seed(assets.seed + wave_protocol.stable_offset(method))
    means: list[np.ndarray] = []
    for step in range(assets.n_steps + 1):
        means.append(ensemble.mean(0).detach().cpu().numpy())
        if step == assets.n_steps:
            break
        ensemble = propagate_numpy(
            ensemble, wave_protocol.alpha_to_theta(0.50, cfg), step, assets, cfg
        )
        if not assets.observation_mask[step + 1]:
            continue
        observation = torch.as_tensor(
            assets.observations[step + 1], dtype=dtype, device=device
        )
        if method == "denkf":
            ensemble = denkf_analysis(ensemble, observation, operator, covariance)
        elif method == "letkf":
            ensemble = letkf_analysis(ensemble, observation, operator, covariance)
        elif method == "ensf_lr_ridge":
            ensemble = ensf_lr_ridge_analysis(
                ensemble, observation, operator, covariance, generator=generator
            )
        else:
            raise ValueError(method)
    return np.stack(means)


def update_adaptive_weights(
    branches: np.ndarray,
    shadow: np.ndarray,
    observation: np.ndarray,
    cfg: wave_protocol.Config,
    scenario: wave_protocol.Scenario,
    config: adaptive_evidence.EvidenceConfig,
    log_weights: np.ndarray,
    step: int,
) -> tuple[np.ndarray, np.ndarray]:
    shadow_observations = [
        shadow[q][:, scenario.observation_indices].copy() for q in range(cfg.n_alpha)
    ]
    log_likelihood_shadow = adaptive_evidence.evidence_vector(
        shadow_observations, observation, cfg.obs_noise, config
    )
    progress = step / max(scenario.times.size - 1, 1)
    blend = 0.0
    if config.analysis_blend_max > 0.0 and progress > config.analysis_blend_start:
        blend = config.analysis_blend_max * min(
            1.0,
            (progress - config.analysis_blend_start)
            / max(1.0 - config.analysis_blend_start, 1.0e-12),
        )
    if blend > 0.0:
        analysis_observations = [
            branches[q][:, scenario.observation_indices].copy()
            for q in range(cfg.n_alpha)
        ]
        analysis_evidence = adaptive_evidence.evidence_vector(
            analysis_observations, observation, cfg.obs_noise, config
        )
        log_likelihood = (1.0 - blend) * log_likelihood_shadow + blend * analysis_evidence
    else:
        log_likelihood = log_likelihood_shadow
    centered = log_likelihood - np.mean(log_likelihood)
    weights = wave_protocol.softmax(log_weights)
    temperature = config.temperature
    if config.adaptive_temperature:
        entropy_ratio = adaptive_evidence.entropy(weights) / max(math.log(cfg.n_alpha), 1.0e-12)
        temperature = float(
            np.clip(config.temperature * entropy_ratio**0.75, config.min_temperature, config.temperature)
        )
    log_weights = config.forgetting * log_weights + temperature * centered
    weights = wave_protocol.softmax(log_weights)
    weights = np.maximum(weights, config.weight_floor)
    weights /= weights.sum()
    weights = adaptive_evidence.entropy_project(weights, adaptive_evidence.entropy_target(progress, config))
    return np.log(np.maximum(weights, 1.0e-300)), weights


def update_refined_weights(
    branches: np.ndarray,
    shadow: np.ndarray,
    observation: np.ndarray,
    cfg: wave_protocol.Config,
    scenario: wave_protocol.Scenario,
    config: adaptive_evidence.EvidenceConfig,
    alpha_log_scores: np.ndarray,
    step: int,
    method: str,
) -> tuple[np.ndarray, np.ndarray, Any]:
    shadow_observations = [
        shadow[q][:, scenario.observation_indices].copy() for q in range(cfg.n_alpha)
    ]
    log_likelihood_shadow = adaptive_evidence.evidence_vector(
        shadow_observations, observation, cfg.obs_noise, config
    )
    progress = step / max(scenario.times.size - 1, 1)
    blend = 0.0
    if config.analysis_blend_max > 0.0 and progress > config.analysis_blend_start:
        blend = config.analysis_blend_max * min(
            1.0,
            (progress - config.analysis_blend_start)
            / max(1.0 - config.analysis_blend_start, 1.0e-12),
        )
    if blend > 0.0:
        analysis_observations = [
            branches[q][:, scenario.observation_indices].copy()
            for q in range(cfg.n_alpha)
        ]
        analysis_evidence = adaptive_evidence.evidence_vector(
            analysis_observations, observation, cfg.obs_noise, config
        )
        log_likelihood = (1.0 - blend) * log_likelihood_shadow + blend * analysis_evidence
    else:
        log_likelihood = log_likelihood_shadow
    centered = log_likelihood - np.mean(log_likelihood)
    if method == "pce":
        alpha_log_scores = alpha_log_scores + config.temperature * centered
        alpha_weights = wave_protocol.softmax(alpha_log_scores)
        state_weights = alpha_weights
        calibration = None
    elif method == "apce":
        calibration = apce_calibration_parameters(
            centered,
            pce_temperature=float(adaptive_evidence.PCE_CONFIG.temperature),
            apce_temperature=config.temperature,
            apce_min_temperature=config.min_temperature,
            apce_forgetting=config.forgetting,
            apce_entropy_floor=max(
                config.entropy_floor_start,
                config.entropy_floor_mid,
                config.entropy_floor_end,
                0.0,
            ),
            progress=progress,
        )
        alpha_log_scores = calibration.forgetting * alpha_log_scores + calibration.temperature * centered
        alpha_weights = wave_protocol.softmax(alpha_log_scores)
        state_weights = adaptive_evidence.entropy_project(alpha_weights, calibration.entropy_floor)
    else:
        raise ValueError(method)
    return alpha_log_scores, state_weights, calibration


def run_pce_family(
    assets: WaveScenarioAssets,
    method: str,
    device: torch.device,
) -> dict[str, Any]:
    cfg = configuration(assets)
    scenario = wave_protocol.scenario_from_assets(assets, cfg)
    config = adaptive_evidence.EVIDENCE_CONFIGS[method]
    alpha_grid = scenario.alpha_grid.copy()
    global_bounds = (float(scenario.alpha_grid[0]), float(scenario.alpha_grid[-1]))
    theta_grid = np.asarray([wave_protocol.alpha_to_theta(float(alpha), cfg) for alpha in alpha_grid], dtype=float)
    branches = scenario.branch_initial.copy()
    shadow = branches.copy()
    state_weights = np.full(cfg.n_alpha, 1.0 / cfg.n_alpha)
    alpha_log_scores = np.zeros(cfg.n_alpha)
    truth = torch.as_tensor(assets.truth_states, dtype=torch.float64, device=device)
    metrics = MetricAccumulator(assets.nx)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    regrid_count = 0
    for step in range(assets.n_steps + 1):
        flat = torch.as_tensor(
            branches.reshape(-1, branches.shape[-1]), dtype=torch.float64, device=device
        )
        member_weights = torch.as_tensor(
            np.repeat(state_weights / cfg.ensemble_size, cfg.ensemble_size),
            dtype=torch.float64,
            device=device,
        )
        metrics.add(flat, truth[step], member_weights)
        if step == assets.n_steps:
            break
        for q, theta in enumerate(theta_grid):
            branches[q] = wave_protocol.propagate_batch(
                branches[q], float(theta), float(assets.times[step]), cfg,
                np.random.default_rng(assets.seed), stochastic=True,
                noise_draw=assets.forecast_noise[step],
            )
            shadow[q] = wave_protocol.propagate_batch(
                shadow[q], float(theta), float(assets.times[step]), cfg,
                np.random.default_rng(assets.seed), stochastic=True,
                noise_draw=assets.forecast_noise[step],
            )
        if not assets.observation_mask[step + 1]:
            continue
        observation = assets.observations[step + 1]
        alpha_log_scores, state_weights, calibration = update_refined_weights(
            branches,
            shadow,
            observation,
            cfg,
            scenario,
            config,
            alpha_log_scores,
            step + 1,
            method,
        )
        refined_grid = local_alpha_grid(
            alpha_grid,
            alpha_log_scores,
            points=cfg.n_alpha,
            bounds=global_bounds,
        )
        if not np.allclose(refined_grid, alpha_grid):
            branches = numpy_regrid_paths(alpha_grid, branches, refined_grid)
            shadow = numpy_regrid_paths(alpha_grid, shadow, refined_grid)
            alpha_log_scores = numpy_regrid_paths(alpha_grid, alpha_log_scores, refined_grid)
            alpha_weights = wave_protocol.softmax(alpha_log_scores)
            if method == "apce":
                state_weights = numpy_regrid_paths(alpha_grid, state_weights, refined_grid)
                state_weights = np.maximum(state_weights, 1.0e-12)
                state_weights /= state_weights.sum()
                if calibration is not None:
                    state_weights = adaptive_evidence.entropy_project(state_weights, calibration.entropy_floor)
            else:
                state_weights = alpha_weights
            alpha_grid = refined_grid
            theta_grid = np.asarray([wave_protocol.alpha_to_theta(float(alpha), cfg) for alpha in alpha_grid], dtype=float)
            regrid_count += 1
        paired_seed = cfg.seed + 10_000_000 + step + 1
        for q in range(cfg.n_alpha):
            branches[q] = ensf_update_lr(
                branches[q], observation, scenario.observation_indices, cfg,
                np.random.default_rng(paired_seed),
            )
    alpha_estimate = refined_alpha_map(alpha_grid, alpha_log_scores)
    alpha_weights = wave_protocol.softmax(alpha_log_scores)
    result = metrics.finalize()
    result.update(
        runtime_seconds=time.perf_counter() - started,
        forward_member_steps=2 * assets.n_steps * cfg.n_alpha * cfg.ensemble_size,
        peak_gpu_memory_mb=(
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda" else 0.0
        ),
        alpha_estimate=alpha_estimate,
        alpha_absolute_error=abs(alpha_estimate - assets.alpha_true),
        alpha_map=float(alpha_grid[int(np.argmax(alpha_log_scores))]),
        alpha_final_entropy=float(adaptive_evidence.entropy(state_weights)),
        alpha_evidence_entropy=float(adaptive_evidence.entropy(alpha_weights)),
        alpha_regrid_count=int(regrid_count),
        alpha_grid_min=float(np.min(alpha_grid)),
        alpha_grid_max=float(np.max(alpha_grid)),
        alpha_grid_points=int(alpha_grid.size),
    )
    return result


def trace_pce_family(
    assets: WaveScenarioAssets,
    method: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the posterior mean trajectory and final alpha weights."""

    cfg = configuration(assets)
    scenario = wave_protocol.scenario_from_assets(assets, cfg)
    config = adaptive_evidence.EVIDENCE_CONFIGS[method]
    alpha_grid = scenario.alpha_grid.copy()
    global_bounds = (float(scenario.alpha_grid[0]), float(scenario.alpha_grid[-1]))
    theta_grid = np.asarray([wave_protocol.alpha_to_theta(float(alpha), cfg) for alpha in alpha_grid], dtype=float)
    branches = scenario.branch_initial.copy()
    shadow = branches.copy()
    state_weights = np.full(cfg.n_alpha, 1.0 / cfg.n_alpha)
    alpha_log_scores = np.zeros(cfg.n_alpha)
    means: list[np.ndarray] = []
    for step in range(assets.n_steps + 1):
        means.append(np.sum(state_weights[:, None] * branches.mean(axis=1), axis=0))
        if step == assets.n_steps:
            break
        for q, theta in enumerate(theta_grid):
            branches[q] = wave_protocol.propagate_batch(
                branches[q], float(theta), float(assets.times[step]), cfg,
                np.random.default_rng(assets.seed), stochastic=True,
                noise_draw=assets.forecast_noise[step],
            )
            shadow[q] = wave_protocol.propagate_batch(
                shadow[q], float(theta), float(assets.times[step]), cfg,
                np.random.default_rng(assets.seed), stochastic=True,
                noise_draw=assets.forecast_noise[step],
            )
        if not assets.observation_mask[step + 1]:
            continue
        observation = assets.observations[step + 1]
        alpha_log_scores, state_weights, calibration = update_refined_weights(
            branches,
            shadow,
            observation,
            cfg,
            scenario,
            config,
            alpha_log_scores,
            step + 1,
            method,
        )
        refined_grid = local_alpha_grid(
            alpha_grid,
            alpha_log_scores,
            points=cfg.n_alpha,
            bounds=global_bounds,
        )
        if not np.allclose(refined_grid, alpha_grid):
            branches = numpy_regrid_paths(alpha_grid, branches, refined_grid)
            shadow = numpy_regrid_paths(alpha_grid, shadow, refined_grid)
            alpha_log_scores = numpy_regrid_paths(alpha_grid, alpha_log_scores, refined_grid)
            alpha_weights = wave_protocol.softmax(alpha_log_scores)
            if method == "apce":
                state_weights = numpy_regrid_paths(alpha_grid, state_weights, refined_grid)
                state_weights = np.maximum(state_weights, 1.0e-12)
                state_weights /= state_weights.sum()
                if calibration is not None:
                    state_weights = adaptive_evidence.entropy_project(state_weights, calibration.entropy_floor)
            else:
                state_weights = alpha_weights
            alpha_grid = refined_grid
            theta_grid = np.asarray([wave_protocol.alpha_to_theta(float(alpha), cfg) for alpha in alpha_grid], dtype=float)
        paired_seed = cfg.seed + 10_000_000 + step + 1
        for q in range(cfg.n_alpha):
            branches[q] = ensf_update_lr(
                branches[q], observation, scenario.observation_indices, cfg,
                np.random.default_rng(paired_seed),
            )
    return np.stack(means), state_weights.copy()
