from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pce_assimilation.evidence import liu_quantile
from pce_assimilation.refinement import (
    apce_calibration_parameters,
    torch_local_alpha_grid,
    torch_refined_alpha_map,
    torch_regrid_paths,
)
from pce_assimilation.ensemble_filters import denkf_analysis, letkf_analysis
from pce_assimilation.metrics import weighted_central_interval_coverage_width, weighted_ensemble_crps
from pce_assimilation.observations import SparseObservation
from pce_assimilation.systems.one_dimensional import Heat1D, HeatConfig, SpringConfig, SpringOscillator


CaseName = Literal["spring", "heat"]
MethodName = Literal[
    "misspecified_forecast",
    "denkf",
    "letkf",
    "pce",
    "apce",
]

METHODS: tuple[MethodName, ...] = (
    "misspecified_forecast",
    "denkf",
    "letkf",
    "pce",
    "apce",
)
METHOD_LABELS = {
    "misspecified_forecast": "Misspecified forecast",
    "denkf": "DEnKF",
    "letkf": "LETKF",
    "pce": "PCE",
    "apce": "APCE",
}


@dataclass(frozen=True)
class CaseConfig:
    name: CaseName
    seed: int
    steps: int
    dt: float
    obs_interval: int
    ensemble_size: int
    obs_noise: float
    alpha_true: float = 0.12
    fixed_alpha: float = 0.50
    alpha_grid: tuple[float, ...] = (0.08, 0.22, 0.36, 0.50, 0.64, 0.78, 0.92)
    pce_temperature: float = 0.60
    apce_temperature: float = 0.54
    apce_min_temperature: float = 0.08
    apce_forgetting: float = 0.975
    apce_entropy_floor: float = 0.36
    evidence_shrinkage: float = 0.18


@dataclass
class Scenario:
    config: CaseConfig
    truth: torch.Tensor
    observations: dict[int, torch.Tensor]
    observation_indices: torch.Tensor
    initial_ensemble: torch.Tensor
    forecast_noise: torch.Tensor
    alpha_grid: torch.Tensor
    primary_indices: torch.Tensor


class TrajectoryMetrics:
    def __init__(self) -> None:
        self.squared_error = 0.0
        self.truth_square = 0.0
        self.crps: list[float] = []
        self.coverage: list[float] = []
        self.width: list[float] = []
        self.points = 0

    def add(
        self,
        ensemble_primary: torch.Tensor,
        truth_primary: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        weights = weights / weights.sum()
        estimate = (weights.unsqueeze(-1) * ensemble_primary).sum(dim=0)
        self.squared_error += float((estimate - truth_primary).square().sum())
        self.truth_square += float(truth_primary.square().sum())
        self.points += int(truth_primary.numel())
        self.crps.append(float(weighted_ensemble_crps(ensemble_primary, truth_primary, weights)))
        coverage, width = weighted_central_interval_coverage_width(
            ensemble_primary, truth_primary, weights, level=0.90
        )
        self.coverage.append(float(coverage))
        self.width.append(float(width))

    def finalize(self) -> dict[str, float]:
        return {
            "nrmse": math.sqrt(self.squared_error / max(self.truth_square, 1.0e-30)),
            "rmse": math.sqrt(self.squared_error / max(self.points, 1)),
            "crps": float(np.mean(self.crps)),
            "coverage_90": float(np.mean(self.coverage)),
            "interval_width_90": float(np.mean(self.width)),
        }


def config_for_case(name: CaseName, seed: int) -> CaseConfig:
    if name == "spring":
        return CaseConfig(
            name=name,
            seed=seed,
            steps=260,
            dt=0.010,
            obs_interval=5,
            ensemble_size=28,
            obs_noise=0.035,
        )
    if name == "heat":
        return CaseConfig(
            name=name,
            seed=seed,
            steps=260,
            dt=0.00075,
            obs_interval=10,
            ensemble_size=24,
            obs_noise=0.025,
        )
    raise ValueError(name)


def make_system(config: CaseConfig) -> SpringOscillator | Heat1D:
    if config.name == "spring":
        return SpringOscillator(
            SpringConfig(
                damping=0.10,
                frequency=1.0,
                cubic_stiffness=0.06,
                forcing_amplitude=0.72,
                forcing_frequency=0.75,
                stochastic_scale=0.10,
                epistemic_scale=0.42,
            )
        )
    return Heat1D(
        HeatConfig(
            nx=64,
            diffusivity=0.060,
            reaction=0.08,
            stochastic_scale=0.010,
            epistemic_scale=0.18,
        )
    )


def initial_state(system: SpringOscillator | Heat1D, config: CaseConfig, device: torch.device) -> torch.Tensor:
    dtype = torch.float64
    if config.name == "spring":
        return torch.tensor([0.62, -0.08], dtype=dtype, device=device)
    heat = system
    assert isinstance(heat, Heat1D)
    grid = heat.grid.to(dtype=dtype, device=device)
    state = (
        0.72 * torch.sin(math.pi * grid)
        + 0.12 * torch.sin(2.0 * math.pi * grid)
        + 0.10 * torch.exp(-((grid - 0.33) / 0.09).square())
    )
    state[0] = 0.0
    state[-1] = 0.0
    return state


def smooth_heat_noise(noise: torch.Tensor) -> torch.Tensor:
    smoothed = noise
    for _ in range(3):
        smoothed = 0.25 * torch.roll(smoothed, 1, dims=-1) + 0.5 * smoothed + 0.25 * torch.roll(smoothed, -1, dims=-1)
    smoothed[..., 0] = 0.0
    smoothed[..., -1] = 0.0
    return smoothed


def step_with_noise(
    system: SpringOscillator | Heat1D,
    state: torch.Tensor,
    time_value: float,
    dt: float,
    alpha: float,
    noise: torch.Tensor,
) -> torch.Tensor:
    quantile = liu_quantile(torch.tensor(alpha, dtype=state.dtype, device=state.device))
    k1 = system.drift(state, time_value, quantile)
    k2 = system.drift(state + 0.5 * dt * k1, time_value + 0.5 * dt, quantile)
    k3 = system.drift(state + 0.5 * dt * k2, time_value + 0.5 * dt, quantile)
    k4 = system.drift(state + dt * k3, time_value + dt, quantile)
    deterministic = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    stochastic = math.sqrt(dt) * system.diffusion(state, time_value) * noise
    return system.project(deterministic + stochastic)


def observation_indices(config: CaseConfig, state_dim: int, device: torch.device) -> torch.Tensor:
    if config.name == "spring":
        return torch.tensor([0], dtype=torch.int64, device=device)
    values = np.linspace(4, state_dim - 5, 8, dtype=np.int64)
    return torch.as_tensor(values, dtype=torch.int64, device=device)


def primary_indices(config: CaseConfig, state_dim: int, device: torch.device) -> torch.Tensor:
    if config.name == "spring":
        return torch.tensor([0], dtype=torch.int64, device=device)
    return torch.arange(state_dim, dtype=torch.int64, device=device)


def generate_scenario(config: CaseConfig, device: torch.device) -> Scenario:
    dtype = torch.float64
    system = make_system(config)
    if isinstance(system, Heat1D):
        system.grid = system.grid.to(device)
    generator = torch.Generator(device=device).manual_seed(config.seed)
    state0 = initial_state(system, config, device)
    state_dim = int(state0.numel())
    obs_idx = observation_indices(config, state_dim, device)
    primary_idx = primary_indices(config, state_dim, device)
    truth_noise = torch.randn((config.steps, state_dim), dtype=dtype, device=device, generator=generator)
    forecast_noise = torch.randn(
        (config.steps, config.ensemble_size, state_dim),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    initial_noise = torch.randn((config.ensemble_size, state_dim), dtype=dtype, device=device, generator=generator)
    obs_noise = torch.randn(
        (config.steps // config.obs_interval + 1, obs_idx.numel()),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    if config.name == "heat":
        truth_noise = smooth_heat_noise(truth_noise)
        forecast_noise = smooth_heat_noise(forecast_noise)
        initial_noise = smooth_heat_noise(initial_noise)
    initial_scale = 0.060 if config.name == "spring" else 0.035
    initial_ensemble = system.project(state0.unsqueeze(0) + initial_scale * initial_noise)
    truth = torch.empty((config.steps + 1, state_dim), dtype=dtype, device=device)
    truth[0] = state0
    for step in range(config.steps):
        truth[step + 1] = step_with_noise(
            system,
            truth[step],
            step * config.dt,
            config.dt,
            config.alpha_true,
            truth_noise[step],
        )
    observations: dict[int, torch.Tensor] = {}
    noise_row = 0
    for step in range(config.obs_interval, config.steps + 1, config.obs_interval):
        observations[step] = truth[step, obs_idx] + config.obs_noise * obs_noise[noise_row]
        noise_row += 1
    return Scenario(
        config=config,
        truth=truth,
        observations=observations,
        observation_indices=obs_idx,
        initial_ensemble=initial_ensemble,
        forecast_noise=forecast_noise,
        alpha_grid=torch.tensor(config.alpha_grid, dtype=dtype, device=device),
        primary_indices=primary_idx,
    )


def evidence_score(
    ensemble_observation: torch.Tensor,
    observation: torch.Tensor,
    obs_noise: float,
    shrinkage: float,
    dimension_weights: torch.Tensor | None,
) -> torch.Tensor:
    mean = ensemble_observation.mean(dim=0)
    anomalies = ensemble_observation - mean
    covariance = anomalies.mT @ anomalies / max(ensemble_observation.shape[0] - 1, 1)
    covariance = (1.0 - shrinkage) * covariance + shrinkage * torch.diag(torch.diagonal(covariance))
    covariance = covariance + (obs_noise**2 + 1.0e-8) * torch.eye(
        observation.numel(), dtype=observation.dtype, device=observation.device
    )
    residual = observation - mean
    if dimension_weights is not None:
        weights = dimension_weights.to(dtype=observation.dtype, device=observation.device)
        if tuple(weights.shape) != tuple(residual.shape):
            raise ValueError("dimension_weights must match the observation dimension")
        weights = weights.clamp_min(1.0e-8)
        weights = observation.numel() * weights / weights.sum().clamp_min(1.0e-12)
        variances = torch.diagonal(covariance).clamp_min(1.0e-12)
        marginal_terms = residual.square() / variances + variances.log() + math.log(2.0 * math.pi)
        return -0.5 * torch.sum(weights * marginal_terms)
    factor = torch.linalg.cholesky(covariance)
    solve = torch.cholesky_solve(residual[:, None], factor).squeeze(-1)
    log_det = 2.0 * torch.log(torch.diagonal(factor)).sum()
    return -0.5 * (residual @ solve + log_det + observation.numel() * math.log(2.0 * math.pi))


def entropy(weights: torch.Tensor) -> torch.Tensor:
    safe = weights.clamp_min(1.0e-300)
    return -(safe * safe.log()).sum()


def entropy_project(weights: torch.Tensor, target_entropy: float) -> torch.Tensor:
    if float(entropy(weights)) >= target_entropy:
        return weights
    uniform = torch.full_like(weights, 1.0 / weights.numel())
    low, high = 0.0, 1.0
    for _ in range(45):
        middle = 0.5 * (low + high)
        mixed = (1.0 - middle) * weights + middle * uniform
        if float(entropy(mixed)) < target_entropy:
            low = middle
        else:
            high = middle
    output = (1.0 - high) * weights + high * uniform
    return output / output.sum()


def continuous_alpha(alpha_grid: torch.Tensor, log_scores: torch.Tensor) -> float:
    return torch_refined_alpha_map(alpha_grid, log_scores)


def primary(ensemble: torch.Tensor, scenario: Scenario) -> torch.Tensor:
    return ensemble.index_select(-1, scenario.primary_indices)


def run_fixed_method(
    scenario: Scenario,
    method: MethodName,
    device: torch.device,
    *,
    record_trace: bool = False,
) -> dict[str, Any]:
    config = scenario.config
    system = make_system(config)
    if isinstance(system, Heat1D):
        system.grid = system.grid.to(device)
    ensemble = scenario.initial_ensemble.clone()
    operator = SparseObservation(scenario.observation_indices)
    covariance = config.obs_noise**2 * torch.eye(
        scenario.observation_indices.numel(),
        dtype=ensemble.dtype,
        device=device,
    )
    metrics = TrajectoryMetrics()
    weights = torch.full((config.ensemble_size,), 1.0 / config.ensemble_size, dtype=ensemble.dtype, device=device)
    alpha = config.fixed_alpha
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    trace_mean_states: list[torch.Tensor] = []
    for step in range(config.steps + 1):
        if record_trace:
            trace_mean_states.append(ensemble.mean(dim=0).detach().cpu())
        metrics.add(primary(ensemble, scenario), primary(scenario.truth[step], scenario), weights)
        if step == config.steps:
            break
        ensemble = step_with_noise(system, ensemble, step * config.dt, config.dt, alpha, scenario.forecast_noise[step])
        if method == "misspecified_forecast" or step + 1 not in scenario.observations:
            continue
        observation = scenario.observations[step + 1]
        if method == "denkf":
            ensemble = denkf_analysis(ensemble, observation, operator, covariance)
        elif method == "letkf":
            ensemble = letkf_analysis(ensemble, observation, operator, covariance)
        else:
            raise ValueError(method)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = metrics.finalize()
    result.update(
        runtime_seconds=float(time.perf_counter() - start),
        forward_member_steps=config.steps * config.ensemble_size,
        peak_gpu_memory_mb=(
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda" else 0.0
        ),
        alpha_estimate=float(alpha),
        alpha_absolute_error=abs(float(alpha) - config.alpha_true),
    )
    if record_trace:
        result["mean_states"] = torch.stack(trace_mean_states).numpy()
    return result


def run_pce_method(
    scenario: Scenario,
    method: MethodName,
    device: torch.device,
    *,
    record_trace: bool = False,
) -> dict[str, Any]:
    config = scenario.config
    system = make_system(config)
    if isinstance(system, Heat1D):
        system.grid = system.grid.to(device)
    alpha_grid = scenario.alpha_grid.clone()
    global_bounds = (float(scenario.alpha_grid[0]), float(scenario.alpha_grid[-1]))
    path_count = int(alpha_grid.numel())
    branches = scenario.initial_ensemble.unsqueeze(0).repeat(path_count, 1, 1)
    shadow = branches.clone()
    alpha_log_scores = torch.zeros(path_count, dtype=branches.dtype, device=device)
    alpha_weights = torch.softmax(alpha_log_scores, dim=0)
    state_weights = alpha_weights.clone()
    operator = SparseObservation(scenario.observation_indices)
    covariance = config.obs_noise**2 * torch.eye(
        scenario.observation_indices.numel(),
        dtype=branches.dtype,
        device=device,
    )
    metrics = TrajectoryMetrics()
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    trace_mean_states: list[torch.Tensor] = []
    trace_alpha_weights: list[torch.Tensor] = []
    trace_alpha_grid: list[torch.Tensor] = []
    trace_alpha_estimate: list[float] = []
    regrid_count = 0
    for step in range(config.steps + 1):
        flat = branches.reshape(-1, branches.shape[-1])
        flat_weights = state_weights.unsqueeze(1).expand(-1, config.ensemble_size).reshape(-1) / config.ensemble_size
        if record_trace:
            branch_means = branches.mean(dim=1)
            trace_mean_states.append((state_weights.unsqueeze(-1) * branch_means).sum(dim=0).detach().cpu())
            trace_alpha_weights.append(state_weights.detach().cpu())
            trace_alpha_grid.append(alpha_grid.detach().cpu())
            trace_alpha_estimate.append(continuous_alpha(alpha_grid, alpha_log_scores))
        metrics.add(primary(flat, scenario), primary(scenario.truth[step], scenario), flat_weights)
        if step == config.steps:
            break
        for path_index, alpha in enumerate(alpha_grid):
            branches[path_index] = step_with_noise(
                system,
                branches[path_index],
                step * config.dt,
                config.dt,
                float(alpha),
                scenario.forecast_noise[step],
            )
            shadow[path_index] = step_with_noise(
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
                evidence_score(
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
            alpha_log_scores = alpha_log_scores + config.pce_temperature * centered
            alpha_weights = torch.softmax(alpha_log_scores, dim=0)
            state_weights = alpha_weights
        elif method == "apce":
            calibration = apce_calibration_parameters(
                centered,
                pce_temperature=config.pce_temperature,
                apce_temperature=config.apce_temperature,
                apce_min_temperature=config.apce_min_temperature,
                apce_forgetting=config.apce_forgetting,
                apce_entropy_floor=config.apce_entropy_floor,
                progress=(step + 1) / max(config.steps, 1),
            )
            alpha_log_scores = calibration.forgetting * alpha_log_scores + calibration.temperature * centered
            alpha_weights = torch.softmax(alpha_log_scores, dim=0)
            state_weights = entropy_project(alpha_weights, calibration.entropy_floor)
        else:
            raise ValueError(method)
        refined_grid = torch_local_alpha_grid(
            alpha_grid,
            alpha_log_scores,
            points=path_count,
            bounds=global_bounds,
        )
        if not torch.allclose(refined_grid, alpha_grid):
            branches = torch_regrid_paths(alpha_grid, branches, refined_grid)
            shadow = torch_regrid_paths(alpha_grid, shadow, refined_grid)
            alpha_log_scores = torch_regrid_paths(alpha_grid, alpha_log_scores, refined_grid)
            alpha_weights = torch.softmax(alpha_log_scores, dim=0)
            if method == "apce":
                calibration = apce_calibration_parameters(
                    centered,
                    pce_temperature=config.pce_temperature,
                    apce_temperature=config.apce_temperature,
                    apce_min_temperature=config.apce_min_temperature,
                    apce_forgetting=config.apce_forgetting,
                    apce_entropy_floor=config.apce_entropy_floor,
                    progress=(step + 1) / max(config.steps, 1),
                )
                state_weights = entropy_project(alpha_weights, calibration.entropy_floor)
            else:
                state_weights = alpha_weights
            alpha_grid = refined_grid
            regrid_count += 1
        for path_index in range(path_count):
            branches[path_index] = denkf_analysis(branches[path_index], observation, operator, covariance)
            branches[path_index] = system.project(branches[path_index])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    alpha_estimate = continuous_alpha(alpha_grid, alpha_log_scores)
    result = metrics.finalize()
    result.update(
        runtime_seconds=float(time.perf_counter() - start),
        forward_member_steps=2 * config.steps * path_count * config.ensemble_size,
        peak_gpu_memory_mb=(
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda" else 0.0
        ),
        alpha_estimate=alpha_estimate,
        alpha_absolute_error=abs(alpha_estimate - config.alpha_true),
        alpha_map=float(alpha_grid[int(torch.argmax(alpha_log_scores))]),
        alpha_final_entropy=float(entropy(state_weights)),
        alpha_evidence_entropy=float(entropy(alpha_weights)),
        alpha_regrid_count=int(regrid_count),
        alpha_grid_min=float(alpha_grid.min().detach().cpu()),
        alpha_grid_max=float(alpha_grid.max().detach().cpu()),
        alpha_grid_points=int(alpha_grid.numel()),
    )
    if record_trace:
        result["mean_states"] = torch.stack(trace_mean_states).numpy()
        result["alpha_weight_history"] = torch.stack(trace_alpha_weights).numpy()
        result["alpha_grid_history"] = torch.stack(trace_alpha_grid).numpy()
        result["alpha_estimate_history"] = np.asarray(trace_alpha_estimate, dtype=float)
    return result


def run_method(
    scenario: Scenario,
    method: MethodName,
    device: torch.device,
    *,
    record_trace: bool = False,
) -> dict[str, Any]:
    if method in {"pce", "apce"}:
        return run_pce_method(scenario, method, device, record_trace=record_trace)
    return run_fixed_method(scenario, method, device, record_trace=record_trace)


def bootstrap_ci(values: np.ndarray, seed: int, n_bootstrap: int = 10_000) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, values.size, size=(n_bootstrap, values.size))].mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def summarize(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for case in ("spring", "heat"):
        for method in METHODS:
            subset = [row for row in records if row["case"] == case and row["method"] == method]
            item: dict[str, Any] = {
                "case": case,
                "method": method,
                "label": METHOD_LABELS[method],
                "n_seeds": len(subset),
            }
            for key in (
                "nrmse",
                "rmse",
                "crps",
                "coverage_90",
                "interval_width_90",
                "alpha_absolute_error",
                "runtime_seconds",
                "forward_member_steps",
                "peak_gpu_memory_mb",
            ):
                values = np.asarray([float(row[key]) for row in subset], dtype=float)
                item[key] = float(values.mean())
                item[f"{key}_ci95"] = bootstrap_ci(values, seed + len(summary))
            summary.append(item)
    return summary
