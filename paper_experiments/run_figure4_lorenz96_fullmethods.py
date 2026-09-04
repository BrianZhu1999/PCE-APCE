from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hilda_da.alpha import liu_quantile
from hilda_da.baselines import denkf_analysis, letkf_analysis
from hilda_da.math_utils import stable_cholesky
from hilda_da.metrics import (
    weighted_central_interval_coverage_width,
    weighted_ensemble_crps,
)
from hilda_da.observations import SparseObservation


MethodName = Literal[
    "misspecified_forecast",
    "denkf",
    "letkf",
    "aug_enkf",
    "bma_static",
    "pce",
    "apce",
    "oracle_alpha",
]

METHODS: tuple[MethodName, ...] = (
    "misspecified_forecast",
    "denkf",
    "letkf",
    "aug_enkf",
    "bma_static",
    "pce",
    "apce",
    "oracle_alpha",
)

METHOD_LABELS = {
    "misspecified_forecast": "Misspecified forecast",
    "denkf": "DEnKF",
    "letkf": "LETKF",
    "aug_enkf": "Aug-EnKF",
    "bma_static": "BMA",
    "pce": "PCE",
    "apce": "APCE",
    "oracle_alpha": "Oracle-alpha",
}

DEFAULT_OUTPUT = PROJECT_ROOT / "results_figure4_lorenz96_fullmethods_5seeds_20260815"


@dataclass(frozen=True)
class Lorenz96RunConfig:
    seed: int
    steps: int = 300
    dt: float = 0.010
    obs_interval: int = 5
    obs_stride: int = 2
    ensemble_size: int = 32
    obs_noise: float = 0.45
    state_dim: int = 40
    alpha_true: float = 0.12
    fixed_alpha: float = 0.50
    alpha_grid: tuple[float, ...] = (0.08, 0.22, 0.36, 0.50, 0.64, 0.78, 0.92)
    pce_temperature: float = 0.36
    apce_temperature: float = 0.50
    apce_min_temperature: float = 0.14
    apce_forgetting: float = 0.985
    apce_entropy_floor: float = 0.70
    evidence_shrinkage: float = 0.22
    aug_alpha_jitter: float = 0.012
    aug_alpha_random_walk_std: float = 0.004
    max_valid_amplitude_ratio: float = 100.0


@dataclass(frozen=True)
class Lorenz96PhysicsConfig:
    dim: int = 40
    forcing_base: float = 8.0
    epistemic_scale: float = 1.55
    stochastic_scale: float = 0.035


@dataclass(frozen=True)
class Scenario:
    config: Lorenz96RunConfig
    times: torch.Tensor
    coordinates: torch.Tensor
    truth: torch.Tensor
    observations: dict[int, torch.Tensor]
    observation_indices: torch.Tensor
    initial_ensemble: torch.Tensor
    forecast_noise: torch.Tensor
    alpha_grid: torch.Tensor


class Lorenz96System:
    def __init__(self, config: Lorenz96PhysicsConfig | None = None) -> None:
        self.config = config or Lorenz96PhysicsConfig()
        self.state_dim = self.config.dim

    def drift(self, state: torch.Tensor, alpha_quantile: torch.Tensor) -> torch.Tensor:
        forcing = self.config.forcing_base + self.config.epistemic_scale * alpha_quantile
        while forcing.ndim < state.ndim:
            forcing = forcing.unsqueeze(-1)
        return (
            (torch.roll(state, shifts=-1, dims=-1) - torch.roll(state, shifts=2, dims=-1))
            * torch.roll(state, shifts=1, dims=-1)
            - state
            + forcing
        )

    def diffusion(self, state: torch.Tensor) -> torch.Tensor:
        return torch.full_like(state, self.config.stochastic_scale)

    def project(self, state: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(state, nan=0.0, posinf=40.0, neginf=-40.0).clamp(-40.0, 40.0)


class TrajectoryMetrics:
    def __init__(self) -> None:
        self.squared_error = 0.0
        self.truth_square = 0.0
        self.points = 0
        self.crps: list[float] = []
        self.coverage: list[float] = []
        self.width: list[float] = []

    def add(
        self,
        ensemble: torch.Tensor,
        truth: torch.Tensor,
        weights: torch.Tensor,
        *,
        point_estimate: torch.Tensor | None = None,
    ) -> None:
        weights = weights / weights.sum().clamp_min(1.0e-300)
        estimate = point_estimate
        if estimate is None:
            estimate = (weights.unsqueeze(-1) * ensemble).sum(dim=0)
        self.squared_error += float((estimate - truth).square().sum())
        self.truth_square += float(truth.square().sum())
        self.points += int(truth.numel())
        self.crps.append(float(weighted_ensemble_crps(ensemble, truth, weights)))
        coverage, width = weighted_central_interval_coverage_width(ensemble, truth, weights, level=0.90)
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def torch_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(seed)
    return generator


def smooth_periodic_noise(noise: torch.Tensor, passes: int = 1) -> torch.Tensor:
    smoothed = noise
    for _ in range(passes):
        smoothed = (
            0.20 * torch.roll(smoothed, 2, dims=-1)
            + 0.25 * torch.roll(smoothed, 1, dims=-1)
            + 0.10 * smoothed
            + 0.25 * torch.roll(smoothed, -1, dims=-1)
            + 0.20 * torch.roll(smoothed, -2, dims=-1)
        )
    scale = smoothed.std(dim=-1, keepdim=True).clamp_min(1.0e-12)
    return smoothed / scale


def rk4_step(system: Lorenz96System, state: torch.Tensor, dt: float, alpha_quantile: torch.Tensor) -> torch.Tensor:
    k1 = system.drift(state, alpha_quantile)
    k2 = system.drift(state + 0.5 * dt * k1, alpha_quantile)
    k3 = system.drift(state + 0.5 * dt * k2, alpha_quantile)
    k4 = system.drift(state + dt * k3, alpha_quantile)
    return system.project(state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0)


def step_with_noise(
    system: Lorenz96System,
    state: torch.Tensor,
    dt: float,
    alpha: float | torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    alpha_tensor = torch.as_tensor(alpha, dtype=state.dtype, device=state.device)
    quantile = liu_quantile(alpha_tensor)
    deterministic = rk4_step(system, state, dt, quantile)
    return system.project(deterministic + math.sqrt(dt) * system.diffusion(state) * noise)


def spinup_initial(
    system: Lorenz96System,
    config: Lorenz96RunConfig,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    state = 8.0 + 0.25 * torch.randn(config.state_dim, dtype=torch.float64, device=device, generator=generator)
    zero = torch.zeros_like(state)
    for _ in range(900):
        state = step_with_noise(system, state, config.dt, config.alpha_true, zero)
    return state.detach()


def observation_indices(config: Lorenz96RunConfig, device: torch.device) -> torch.Tensor:
    return torch.arange(0, config.state_dim, config.obs_stride, dtype=torch.int64, device=device)


def generate_scenario(config: Lorenz96RunConfig, device: torch.device) -> Scenario:
    system = Lorenz96System(Lorenz96PhysicsConfig(dim=config.state_dim))
    generator = torch_generator(device, config.seed)
    state0 = spinup_initial(system, config, generator, device)
    obs_idx = observation_indices(config, device)
    truth_noise = torch.randn((config.steps, config.state_dim), dtype=torch.float64, device=device, generator=generator)
    forecast_noise = torch.randn(
        (config.steps, config.ensemble_size, config.state_dim),
        dtype=torch.float64,
        device=device,
        generator=generator,
    )
    initial_noise = torch.randn(
        (config.ensemble_size, config.state_dim),
        dtype=torch.float64,
        device=device,
        generator=generator,
    )
    obs_noise = torch.randn(
        (config.steps // config.obs_interval + 1, obs_idx.numel()),
        dtype=torch.float64,
        device=device,
        generator=generator,
    )
    truth_noise = smooth_periodic_noise(truth_noise, passes=1)
    forecast_noise = smooth_periodic_noise(forecast_noise, passes=1)
    initial_noise = smooth_periodic_noise(initial_noise, passes=1)

    initial_ensemble = system.project(state0.unsqueeze(0) + 0.42 * initial_noise)
    truth = torch.empty((config.steps + 1, config.state_dim), dtype=torch.float64, device=device)
    truth[0] = state0
    for step in range(config.steps):
        truth[step + 1] = step_with_noise(
            system,
            truth[step],
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
        times=torch.arange(config.steps + 1, dtype=torch.float64, device=device) * config.dt,
        coordinates=torch.arange(config.state_dim, dtype=torch.float64, device=device),
        truth=truth,
        observations=observations,
        observation_indices=obs_idx,
        initial_ensemble=initial_ensemble,
        forecast_noise=forecast_noise,
        alpha_grid=torch.tensor(config.alpha_grid, dtype=torch.float64, device=device),
    )


def entropy(weights: torch.Tensor) -> torch.Tensor:
    safe = weights.clamp_min(1.0e-300)
    return -(safe * safe.log()).sum()


def entropy_project(weights: torch.Tensor, target_entropy: float) -> torch.Tensor:
    if float(entropy(weights)) >= target_entropy:
        return weights / weights.sum().clamp_min(1.0e-300)
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
    return output / output.sum().clamp_min(1.0e-300)


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
        weights = dimension_weights.to(dtype=observation.dtype, device=observation.device).clamp_min(1.0e-8)
        if tuple(weights.shape) != tuple(residual.shape):
            raise ValueError("dimension_weights must match the observation dimension")
        weights = observation.numel() * weights / weights.sum().clamp_min(1.0e-12)
        variances = torch.diagonal(covariance).clamp_min(1.0e-12)
        marginal_terms = residual.square() / variances + variances.log() + math.log(2.0 * math.pi)
        return -0.5 * torch.sum(weights * marginal_terms)
    factor = stable_cholesky(covariance)
    solve = torch.cholesky_solve(residual[:, None], factor).squeeze(-1)
    log_det = 2.0 * torch.log(torch.diagonal(factor)).sum()
    return -0.5 * (residual @ solve + log_det + observation.numel() * math.log(2.0 * math.pi))


def tile_alpha_members(
    alpha_grid: torch.Tensor,
    ensemble_size: int,
    generator: torch.Generator,
    *,
    jitter: float,
) -> torch.Tensor:
    repeats = math.ceil(ensemble_size / int(alpha_grid.numel()))
    alpha = alpha_grid.repeat(repeats)[:ensemble_size].clone()
    if jitter > 0.0:
        scale = jitter * float(alpha_grid[-1] - alpha_grid[0])
        alpha = alpha + scale * torch.randn(alpha.shape, dtype=alpha.dtype, device=alpha.device, generator=generator)
    return alpha.clamp(float(alpha_grid[0]), float(alpha_grid[-1]))


def random_walk_alpha(
    alpha: torch.Tensor,
    generator: torch.Generator,
    lower: float,
    upper: float,
    *,
    std: float,
) -> torch.Tensor:
    updated = alpha + std * torch.randn(alpha.shape, dtype=alpha.dtype, device=alpha.device, generator=generator)
    return updated.clamp(lower, upper)


def run_fixed_method(
    scenario: Scenario,
    method: MethodName,
    device: torch.device,
    *,
    record_trace: bool,
) -> dict[str, Any]:
    config = scenario.config
    system = Lorenz96System(Lorenz96PhysicsConfig(dim=config.state_dim))
    ensemble = scenario.initial_ensemble.clone()
    alpha = config.alpha_true if method == "oracle_alpha" else config.fixed_alpha
    operator = SparseObservation(scenario.observation_indices)
    covariance = config.obs_noise**2 * torch.eye(
        scenario.observation_indices.numel(), dtype=ensemble.dtype, device=device
    )
    weights = torch.full((config.ensemble_size,), 1.0 / config.ensemble_size, dtype=ensemble.dtype, device=device)
    metrics = TrajectoryMetrics()
    trace_mean_states: list[torch.Tensor] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(config.steps + 1):
        mean_state = ensemble.mean(dim=0)
        if record_trace:
            trace_mean_states.append(mean_state.detach().cpu())
        metrics.add(ensemble, scenario.truth[step], weights, point_estimate=mean_state)
        if step == config.steps:
            break
        ensemble = step_with_noise(system, ensemble, config.dt, alpha, scenario.forecast_noise[step])
        if method == "misspecified_forecast" or step + 1 not in scenario.observations:
            continue
        observation = scenario.observations[step + 1]
        if method in {"denkf", "oracle_alpha"}:
            ensemble = denkf_analysis(ensemble, observation, operator, covariance)
        elif method == "letkf":
            ensemble = letkf_analysis(ensemble, observation, operator, covariance)
        else:
            raise ValueError(method)
        ensemble = system.project(ensemble)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = metrics.finalize()
    result.update(
        runtime_seconds=float(time.perf_counter() - started),
        forward_member_steps=config.steps * config.ensemble_size,
        peak_gpu_memory_mb=float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
        alpha_estimate=float(alpha),
        alpha_map=float(alpha),
        alpha_absolute_error=abs(float(alpha) - config.alpha_true),
        max_abs_state=float(torch.max(torch.abs(ensemble)).detach().cpu()),
    )
    if record_trace:
        result["mean_states"] = torch.stack(trace_mean_states).numpy()
    return result


def run_aug_enkf_method(
    scenario: Scenario,
    device: torch.device,
    *,
    record_trace: bool,
) -> dict[str, Any]:
    config = scenario.config
    system = Lorenz96System(Lorenz96PhysicsConfig(dim=config.state_dim))
    ensemble = scenario.initial_ensemble.clone()
    generator = torch_generator(device, config.seed * 1000 + 741_001)
    alpha = tile_alpha_members(scenario.alpha_grid, config.ensemble_size, generator, jitter=config.aug_alpha_jitter)
    operator = SparseObservation(scenario.observation_indices)
    covariance = config.obs_noise**2 * torch.eye(
        scenario.observation_indices.numel(), dtype=ensemble.dtype, device=device
    )
    weights = torch.full((config.ensemble_size,), 1.0 / config.ensemble_size, dtype=ensemble.dtype, device=device)
    metrics = TrajectoryMetrics()
    trace_mean_states: list[torch.Tensor] = []
    trace_alpha_mean: list[float] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(config.steps + 1):
        mean_state = ensemble.mean(dim=0)
        if record_trace:
            trace_mean_states.append(mean_state.detach().cpu())
            trace_alpha_mean.append(float(alpha.mean().detach().cpu()))
        metrics.add(ensemble, scenario.truth[step], weights, point_estimate=mean_state)
        if step == config.steps:
            break
        alpha = random_walk_alpha(
            alpha,
            generator,
            lower=float(scenario.alpha_grid[0]),
            upper=float(scenario.alpha_grid[-1]),
            std=config.aug_alpha_random_walk_std,
        )
        ensemble = step_with_noise(system, ensemble, config.dt, alpha, scenario.forecast_noise[step])
        if step + 1 not in scenario.observations:
            continue
        augmented = torch.cat([ensemble, alpha[:, None]], dim=-1)
        updated = denkf_analysis(augmented, scenario.observations[step + 1], operator, covariance)
        ensemble = system.project(updated[:, :-1])
        alpha = updated[:, -1].clamp(float(scenario.alpha_grid[0]), float(scenario.alpha_grid[-1]))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    alpha_estimate = float(alpha.mean().detach().cpu())
    result = metrics.finalize()
    result.update(
        runtime_seconds=float(time.perf_counter() - started),
        forward_member_steps=config.steps * config.ensemble_size,
        peak_gpu_memory_mb=float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
        alpha_estimate=alpha_estimate,
        alpha_map=float(alpha.median().detach().cpu()),
        alpha_absolute_error=abs(alpha_estimate - config.alpha_true),
        alpha_spread=float(alpha.std(unbiased=True).detach().cpu()) if alpha.numel() > 1 else 0.0,
        max_abs_state=float(torch.max(torch.abs(ensemble)).detach().cpu()),
    )
    if record_trace:
        result["mean_states"] = torch.stack(trace_mean_states).numpy()
        result["alpha_mean_history"] = np.asarray(trace_alpha_mean, dtype=float)
    return result


def run_bma_static_method(
    scenario: Scenario,
    device: torch.device,
    *,
    record_trace: bool,
) -> dict[str, Any]:
    config = scenario.config
    system = Lorenz96System(Lorenz96PhysicsConfig(dim=config.state_dim))
    path_count = int(scenario.alpha_grid.numel())
    branches = scenario.initial_ensemble.unsqueeze(0).repeat(path_count, 1, 1)
    log_weights = torch.zeros(path_count, dtype=branches.dtype, device=device)
    path_weights = torch.softmax(log_weights, dim=0)
    operator = SparseObservation(scenario.observation_indices)
    covariance = config.obs_noise**2 * torch.eye(
        scenario.observation_indices.numel(), dtype=branches.dtype, device=device
    )
    metrics = TrajectoryMetrics()
    trace_mean_states: list[torch.Tensor] = []
    trace_weights: list[torch.Tensor] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(config.steps + 1):
        flat = branches.reshape(-1, config.state_dim)
        flat_weights = path_weights.unsqueeze(1).expand(-1, config.ensemble_size).reshape(-1) / config.ensemble_size
        branch_means = branches.mean(dim=1)
        estimate = (path_weights.unsqueeze(-1) * branch_means).sum(dim=0)
        if record_trace:
            trace_mean_states.append(estimate.detach().cpu())
            trace_weights.append(path_weights.detach().cpu())
        metrics.add(flat, scenario.truth[step], flat_weights, point_estimate=estimate)
        if step == config.steps:
            break
        for path_index, alpha in enumerate(scenario.alpha_grid):
            branches[path_index] = step_with_noise(
                system,
                branches[path_index],
                config.dt,
                float(alpha),
                scenario.forecast_noise[step],
            )
        if step + 1 not in scenario.observations:
            continue
        observation = scenario.observations[step + 1]
        branch_observations = torch.stack([operator(branch) for branch in branches])
        evidence = torch.stack(
            [
                evidence_score(
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
        path_weights = torch.softmax(log_weights, dim=0)
        for path_index in range(path_count):
            branches[path_index] = system.project(
                denkf_analysis(branches[path_index], observation, operator, covariance)
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    alpha_estimate = float((scenario.alpha_grid * path_weights).sum().detach().cpu())
    result = metrics.finalize()
    result.update(
        runtime_seconds=float(time.perf_counter() - started),
        forward_member_steps=config.steps * path_count * config.ensemble_size,
        peak_gpu_memory_mb=float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
        alpha_estimate=alpha_estimate,
        alpha_map=float(scenario.alpha_grid[int(torch.argmax(path_weights))].detach().cpu()),
        alpha_absolute_error=abs(alpha_estimate - config.alpha_true),
        alpha_final_entropy=float(entropy(path_weights).detach().cpu()),
        max_abs_state=float(torch.max(torch.abs(branches)).detach().cpu()),
    )
    if record_trace:
        result["mean_states"] = torch.stack(trace_mean_states).numpy()
        result["alpha_weight_history"] = torch.stack(trace_weights).numpy()
    return result


def run_pce_apce_method(
    scenario: Scenario,
    method: Literal["pce", "apce"],
    device: torch.device,
    *,
    record_trace: bool,
) -> dict[str, Any]:
    config = scenario.config
    system = Lorenz96System(Lorenz96PhysicsConfig(dim=config.state_dim))
    path_count = int(scenario.alpha_grid.numel())
    branches = scenario.initial_ensemble.unsqueeze(0).repeat(path_count, 1, 1)
    shadow = branches.clone()
    log_weights = torch.zeros(path_count, dtype=branches.dtype, device=device)
    weights = torch.softmax(log_weights, dim=0)
    operator = SparseObservation(scenario.observation_indices)
    covariance = config.obs_noise**2 * torch.eye(
        scenario.observation_indices.numel(), dtype=branches.dtype, device=device
    )
    metrics = TrajectoryMetrics()
    trace_mean_states: list[torch.Tensor] = []
    trace_weights: list[torch.Tensor] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(config.steps + 1):
        flat = branches.reshape(-1, config.state_dim)
        flat_weights = weights.unsqueeze(1).expand(-1, config.ensemble_size).reshape(-1) / config.ensemble_size
        branch_means = branches.mean(dim=1)
        estimate = (weights.unsqueeze(-1) * branch_means).sum(dim=0)
        if record_trace:
            trace_mean_states.append(estimate.detach().cpu())
            trace_weights.append(weights.detach().cpu())
        metrics.add(flat, scenario.truth[step], flat_weights, point_estimate=estimate)
        if step == config.steps:
            break
        for path_index, alpha in enumerate(scenario.alpha_grid):
            branches[path_index] = step_with_noise(
                system,
                branches[path_index],
                config.dt,
                float(alpha),
                scenario.forecast_noise[step],
            )
            shadow[path_index] = step_with_noise(
                system,
                shadow[path_index],
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
            dimension_weights = 0.30 + 0.70 * between / between.max().clamp_min(1.0e-12)
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
            log_weights = log_weights + config.pce_temperature * centered
        else:
            entropy_ratio = float(entropy(weights) / math.log(path_count))
            temperature = float(
                np.clip(
                    config.apce_temperature * entropy_ratio**0.75,
                    config.apce_min_temperature,
                    config.apce_temperature,
                )
            )
            log_weights = config.apce_forgetting * log_weights + temperature * centered
        weights = torch.softmax(log_weights, dim=0)
        if method == "apce":
            progress = (step + 1) / max(config.steps, 1)
            target_entropy = config.apce_entropy_floor + 0.18 * (1.0 - progress)
            weights = entropy_project(weights, target_entropy)
            log_weights = weights.clamp_min(1.0e-300).log()
        for path_index in range(path_count):
            branches[path_index] = system.project(
                denkf_analysis(branches[path_index], observation, operator, covariance)
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    alpha_estimate = float((scenario.alpha_grid * weights).sum().detach().cpu())
    result = metrics.finalize()
    result.update(
        runtime_seconds=float(time.perf_counter() - started),
        forward_member_steps=2 * config.steps * path_count * config.ensemble_size,
        peak_gpu_memory_mb=float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
        alpha_estimate=alpha_estimate,
        alpha_map=float(scenario.alpha_grid[int(torch.argmax(log_weights))].detach().cpu()),
        alpha_absolute_error=abs(alpha_estimate - config.alpha_true),
        alpha_final_entropy=float(entropy(weights).detach().cpu()),
        max_abs_state=float(torch.max(torch.abs(branches)).detach().cpu()),
    )
    if record_trace:
        result["mean_states"] = torch.stack(trace_mean_states).numpy()
        result["alpha_weight_history"] = torch.stack(trace_weights).numpy()
    return result


def run_method(scenario: Scenario, method: MethodName, device: torch.device, *, record_trace: bool) -> dict[str, Any]:
    if method in {"misspecified_forecast", "denkf", "letkf", "oracle_alpha"}:
        return run_fixed_method(scenario, method, device, record_trace=record_trace)
    if method == "aug_enkf":
        return run_aug_enkf_method(scenario, device, record_trace=record_trace)
    if method == "bma_static":
        return run_bma_static_method(scenario, device, record_trace=record_trace)
    if method in {"pce", "apce"}:
        return run_pce_apce_method(scenario, method, device, record_trace=record_trace)
    raise ValueError(method)


def classify_numerical_status(row: dict[str, Any], truth: torch.Tensor, config: Lorenz96RunConfig) -> str:
    keys = ("nrmse", "rmse", "crps", "coverage_90", "interval_width_90", "max_abs_state")
    if not all(math.isfinite(float(row[key])) for key in keys):
        return "nonfinite"
    truth_scale = float(torch.max(torch.abs(truth)).detach().cpu())
    if truth_scale > 0 and float(row["max_abs_state"]) > config.max_valid_amplitude_ratio * truth_scale:
        return "diverged"
    return "valid"


def save_trace(output: Path, method: MethodName, seed: int, scenario: Scenario, result: dict[str, Any]) -> str:
    trace_dir = output / "traces" / method
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"lorenz96_{method}_seed_{seed}.npz"
    observations = np.full(
        (scenario.config.steps + 1, scenario.observation_indices.numel()),
        np.nan,
        dtype=float,
    )
    for step, value in scenario.observations.items():
        observations[step] = value.detach().cpu().numpy()
    payload: dict[str, Any] = {
        "times": scenario.times.detach().cpu().numpy(),
        "coordinates": scenario.coordinates.detach().cpu().numpy(),
        "truth": scenario.truth.detach().cpu().numpy(),
        "observations": observations,
        "observation_indices": scenario.observation_indices.detach().cpu().numpy(),
        "alpha_grid": scenario.alpha_grid.detach().cpu().numpy(),
    }
    for key in ("mean_states", "alpha_mean_history", "alpha_weight_history"):
        if key in result:
            payload[key] = result[key]
    np.savez_compressed(path, **payload)
    return str(path)


def run_one(seed: int, method: MethodName, device: torch.device, output: Path, record_trace: bool) -> dict[str, Any]:
    config = Lorenz96RunConfig(seed=seed)
    scenario = generate_scenario(config, device)
    started = time.perf_counter()
    result = run_method(scenario, method, device, record_trace=record_trace)
    status = classify_numerical_status(result, scenario.truth, config)
    trace_path = save_trace(output, method, seed, scenario, result) if record_trace else ""
    row: dict[str, Any] = {
        "case": "lorenz96",
        "method": method,
        "label": METHOD_LABELS[method],
        "seed": seed,
        "status": "completed",
        "numerical_status": status,
        "state_dim": config.state_dim,
        "observation_count": int(scenario.observation_indices.numel()),
        "observed_variables": ",".join(str(int(x)) for x in scenario.observation_indices.detach().cpu().tolist()),
        "dt": config.dt,
        "steps": config.steps,
        "assimilation_interval": config.obs_interval,
        "observation_noise": config.obs_noise,
        "ensemble_size": config.ensemble_size,
        "alpha_true": config.alpha_true,
        "fixed_alpha": config.fixed_alpha,
        "alpha_grid": ",".join(f"{x:.4g}" for x in config.alpha_grid),
        "elapsed_seconds_wall": float(time.perf_counter() - started),
        "trace_npz": trace_path,
    }
    for key, value in result.items():
        if isinstance(value, (float, int, str)):
            row[key] = value
    return row


def bootstrap_ci(values: np.ndarray, seed: int, n_bootstrap: int = 10_000) -> list[float]:
    if values.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, values.size, size=(n_bootstrap, values.size))].mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def summarize(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for method in METHODS:
        subset = [row for row in records if row["method"] == method]
        item: dict[str, Any] = {
            "case": "lorenz96",
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
            item[key] = float(values.mean()) if values.size else float("nan")
            item[f"{key}_ci95"] = bootstrap_ci(values, seed + len(summary) * 97 + len(key))
        summary.append(item)
    return summary


def paired_decisions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method_seed = {(row["method"], row["seed"]): row for row in records}
    baselines = ("denkf", "letkf", "aug_enkf", "bma_static")
    decisions: list[dict[str, Any]] = []
    for method in ("pce", "apce"):
        rows = []
        for seed in sorted({int(row["seed"]) for row in records}):
            if (method, seed) not in by_method_seed:
                continue
            row = by_method_seed[(method, seed)]
            baseline_rows = [by_method_seed[(baseline, seed)] for baseline in baselines if (baseline, seed) in by_method_seed]
            if not baseline_rows:
                continue
            rows.append(
                {
                    "seed": seed,
                    "nrmse_wins_all_strong_baselines": all(float(row["nrmse"]) < float(base["nrmse"]) for base in baseline_rows),
                    "crps_wins_all_strong_baselines": all(float(row["crps"]) < float(base["crps"]) for base in baseline_rows),
                    "best_baseline_nrmse": min(float(base["nrmse"]) for base in baseline_rows),
                    "best_baseline_crps": min(float(base["crps"]) for base in baseline_rows),
                    "method_nrmse": float(row["nrmse"]),
                    "method_crps": float(row["crps"]),
                }
            )
        decisions.append(
            {
                "case": "lorenz96",
                "method": method,
                "paired_seed_count": len(rows),
                "nrmse_win_all_strong_baselines_count": int(sum(item["nrmse_wins_all_strong_baselines"] for item in rows)),
                "crps_win_all_strong_baselines_count": int(sum(item["crps_wins_all_strong_baselines"] for item in rows)),
                "mean_nrmse_gain_vs_best_baseline": float(
                    np.mean([item["best_baseline_nrmse"] - item["method_nrmse"] for item in rows])
                )
                if rows else float("nan"),
                "mean_crps_gain_vs_best_baseline": float(
                    np.mean([item["best_baseline_crps"] - item["method_crps"] for item in rows])
                )
                if rows else float("nan"),
            }
        )
    return decisions


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(output: Path, summary: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> None:
    lines = [
        "# Lorenz-96 Figure 4 full-method 5-seed report",
        "",
        "This run is Lorenz-96 only. KS is not part of this output.",
        "",
        "| Method | nRMSE | RMSE | CRPS | 90% coverage | Width | alpha error | Runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {label} | {nrmse:.2%} | {rmse:.4g} | {crps:.4g} | {coverage_90:.2%} | {interval_width_90:.4g} | {alpha_absolute_error:.4g} | {runtime_seconds:.2f} |".format(
                **row
            )
        )
    lines += ["", "## Paired decisions", ""]
    for item in decisions:
        lines.append(
            "- {method}: nRMSE win-all-strong-baselines {nrmse_win_all_strong_baselines_count}/{paired_seed_count}; "
            "CRPS win-all-strong-baselines {crps_win_all_strong_baselines_count}/{paired_seed_count}; "
            "mean nRMSE gain vs best strong baseline = {mean_nrmse_gain_vs_best_baseline:.4g}; "
            "mean CRPS gain vs best strong baseline = {mean_crps_gain_vs_best_baseline:.4g}.".format(**item)
        )
    (output / "LORENZ96_FULLMETHODS_5SEED_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_suite(n_seeds: int, base_seed: int, output: Path, device: torch.device, record_trace: bool) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    total = n_seeds * len(METHODS)
    counter = 0
    for seed_offset in range(n_seeds):
        seed = base_seed + seed_offset
        for method in METHODS:
            counter += 1
            row = run_one(seed, method, device, output, record_trace)
            records.append(row)
            print(
                f"[{counter}/{total}] case=lorenz96 seed={seed} method={method} "
                f"nrmse={100.0 * float(row['nrmse']):.4f}% crps={float(row['crps']):.4e} "
                f"alpha_err={float(row['alpha_absolute_error']):.4g}",
                flush=True,
            )
    summary = summarize(records, base_seed)
    decisions = paired_decisions(records)
    write_csv(output / "run_metrics.csv", records)
    write_csv(output / "summary.csv", summary)
    write_report(output, summary, decisions)
    payload = {
        "case": "lorenz96",
        "n_seeds": n_seeds,
        "base_seed": base_seed,
        "device": str(device),
        "methods": list(METHODS),
        "record_trace": bool(record_trace),
        "config": asdict(Lorenz96RunConfig(seed=base_seed)),
        "source_files": {
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": file_sha256(Path(__file__).resolve()),
        },
        "summary": summary,
        "decisions": decisions,
    }
    (output / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Figure 4 Lorenz-96-only full-method 5-seed runner.")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=2026080600)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-record-trace", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    result = run_suite(
        n_seeds=args.n_seeds,
        base_seed=args.base_seed,
        output=args.output,
        device=device,
        record_trace=not args.no_record_trace,
    )
    print(json.dumps({"decisions": result["decisions"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
