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
from hilda_da.metrics import weighted_central_interval_coverage_width, weighted_ensemble_crps
from hilda_da.observations import SparseObservation


MethodName = Literal["denkf", "letkf", "aug_enkf", "bma_static", "pce", "apce"]

METHODS: tuple[MethodName, ...] = (
    "denkf",
    "letkf",
    "aug_enkf",
    "bma_static",
    "pce",
    "apce",
)

METHOD_LABELS: dict[str, str] = {
    "denkf": "DEnKF",
    "letkf": "LETKF",
    "aug_enkf": "Aug-EnKF",
    "bma_static": "BMA",
    "pce": "PCE",
    "apce": "APCE",
}

DEFAULT_OUTPUT = PROJECT_ROOT / "results_figure4_lorenz96_obs20_rewrite_5seeds_20260815"


@dataclass(frozen=True)
class Lorenz96Config:
    seed: int
    state_dim: int = 40
    steps: int = 300
    dt: float = 0.01
    obs_interval: int = 5
    obs_stride: int = 2
    ensemble_size: int = 32
    obs_noise: float = 0.45
    alpha_true: float = 0.12
    fixed_alpha: float = 0.50
    alpha_min: float = 0.08
    alpha_max: float = 0.92
    alpha_grid: tuple[float, ...] = (0.08, 0.22, 0.36, 0.50, 0.64, 0.78, 0.92)
    forcing_base: float = 8.0
    forcing_scale: float = 1.55
    stochastic_scale: float = 0.035
    initial_spread: float = 0.42
    spinup_steps: int = 900
    max_valid_amplitude_ratio: float = 100.0
    evidence_shrinkage: float = 0.22
    pce_temperature: float = 0.36
    apce_temperature: float = 0.50
    apce_min_temperature: float = 0.14
    apce_forgetting: float = 0.985
    apce_entropy_floor: float = 0.70
    apce_dimension_floor: float = 0.30
    apce_dimension_gain: float = 0.70
    branch_member_alpha_jitter: float = 0.045
    aug_alpha_jitter: float = 0.035
    aug_alpha_random_walk_std: float = 0.004
    branch_augmented_alpha_analysis_strength: float = 0.35
    global_augmented_alpha_analysis_strength: float = 0.10
    global_state_analysis_strength: float = 0.12
    local_grid_points: int = 11
    local_grid_radius: float = 0.18
    local_grid_min_spacing: float = 0.012


@dataclass(frozen=True)
class Scenario:
    config: Lorenz96Config
    times: torch.Tensor
    truth: torch.Tensor
    observations: dict[int, torch.Tensor]
    observation_indices: torch.Tensor
    initial_ensemble: torch.Tensor
    forecast_noise: torch.Tensor
    alpha_grid: torch.Tensor


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(seed)
    return generator


def randn(shape: tuple[int, ...], device: torch.device, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(shape, dtype=torch.float64, device=device, generator=generator)


def smooth_periodic_noise(noise: torch.Tensor, passes: int = 1) -> torch.Tensor:
    out = noise
    for _ in range(passes):
        out = (
            0.20 * torch.roll(out, 2, dims=-1)
            + 0.25 * torch.roll(out, 1, dims=-1)
            + 0.10 * out
            + 0.25 * torch.roll(out, -1, dims=-1)
            + 0.20 * torch.roll(out, -2, dims=-1)
        )
    return out / out.std(dim=-1, keepdim=True).clamp_min(1.0e-12)


class Lorenz96:
    def __init__(self, config: Lorenz96Config) -> None:
        self.config = config

    def forcing(self, alpha: torch.Tensor | float, like: torch.Tensor) -> torch.Tensor:
        alpha_tensor = torch.as_tensor(alpha, dtype=like.dtype, device=like.device)
        forcing = self.config.forcing_base + self.config.forcing_scale * liu_quantile(alpha_tensor)
        while forcing.ndim < like.ndim:
            forcing = forcing.unsqueeze(-1)
        return forcing

    def drift(self, state: torch.Tensor, alpha: torch.Tensor | float) -> torch.Tensor:
        return (
            (torch.roll(state, shifts=-1, dims=-1) - torch.roll(state, shifts=2, dims=-1))
            * torch.roll(state, shifts=1, dims=-1)
            - state
            + self.forcing(alpha, state)
        )

    def project(self, state: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(state, nan=0.0, posinf=40.0, neginf=-40.0).clamp(-40.0, 40.0)

    def step(self, state: torch.Tensor, alpha: torch.Tensor | float, noise: torch.Tensor) -> torch.Tensor:
        dt = self.config.dt
        k1 = self.drift(state, alpha)
        k2 = self.drift(state + 0.5 * dt * k1, alpha)
        k3 = self.drift(state + 0.5 * dt * k2, alpha)
        k4 = self.drift(state + dt * k3, alpha)
        deterministic = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        stochastic = math.sqrt(dt) * self.config.stochastic_scale * noise
        return self.project(deterministic + stochastic)


def observation_indices(config: Lorenz96Config, device: torch.device) -> torch.Tensor:
    return torch.arange(0, config.state_dim, config.obs_stride, dtype=torch.int64, device=device)


def generate_scenario(
    seed: int,
    device: torch.device,
    *,
    obs_stride: int,
    obs_interval: int,
) -> Scenario:
    if obs_stride < 1 or obs_stride > 40:
        raise ValueError("--obs-stride must lie in [1, 40]")
    if obs_interval < 1 or obs_interval > 300:
        raise ValueError("--obs-interval must lie in [1, 300]")
    config = Lorenz96Config(seed=seed, obs_stride=obs_stride, obs_interval=obs_interval)
    system = Lorenz96(config)
    generator = make_generator(device, seed)
    state = 8.0 + 0.25 * randn((config.state_dim,), device, generator)
    zero = torch.zeros_like(state)
    for _ in range(config.spinup_steps):
        state = system.step(state, config.alpha_true, zero)
    obs_idx = observation_indices(config, device)
    truth_noise = smooth_periodic_noise(randn((config.steps, config.state_dim), device, generator), passes=1)
    forecast_noise = smooth_periodic_noise(
        randn((config.steps, config.ensemble_size, config.state_dim), device, generator),
        passes=1,
    )
    initial_noise = smooth_periodic_noise(randn((config.ensemble_size, config.state_dim), device, generator), passes=1)
    obs_noise_rows = config.steps // config.obs_interval
    obs_noise = randn((obs_noise_rows, int(obs_idx.numel())), device, generator)
    initial_ensemble = system.project(state.unsqueeze(0) + config.initial_spread * initial_noise)
    truth = torch.empty((config.steps + 1, config.state_dim), dtype=torch.float64, device=device)
    truth[0] = state
    for step in range(config.steps):
        truth[step + 1] = system.step(truth[step], config.alpha_true, truth_noise[step])
    observations: dict[int, torch.Tensor] = {}
    for row, step in enumerate(range(config.obs_interval, config.steps + 1, config.obs_interval)):
        observations[step] = truth[step, obs_idx] + config.obs_noise * obs_noise[row]
    return Scenario(
        config=config,
        times=torch.arange(config.steps + 1, dtype=torch.float64, device=device) * config.dt,
        truth=truth,
        observations=observations,
        observation_indices=obs_idx,
        initial_ensemble=initial_ensemble,
        forecast_noise=forecast_noise,
        alpha_grid=torch.tensor(config.alpha_grid, dtype=torch.float64, device=device),
    )


class RunningMetrics:
    def __init__(self) -> None:
        self.sse = 0.0
        self.truth_energy = 0.0
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
        estimate = point_estimate if point_estimate is not None else (weights[:, None] * ensemble).sum(dim=0)
        self.sse += float((estimate - truth).square().sum().detach().cpu())
        self.truth_energy += float(truth.square().sum().detach().cpu())
        self.points += int(truth.numel())
        self.crps.append(float(weighted_ensemble_crps(ensemble, truth, weights).detach().cpu()))
        coverage, width = weighted_central_interval_coverage_width(ensemble, truth, weights, level=0.90)
        self.coverage.append(float(coverage.detach().cpu()))
        self.width.append(float(width.detach().cpu()))

    def finalize(self) -> dict[str, float]:
        return {
            "nrmse": math.sqrt(self.sse / max(self.truth_energy, 1.0e-30)),
            "rmse": math.sqrt(self.sse / max(self.points, 1)),
            "crps": float(np.mean(self.crps)),
            "coverage_90": float(np.mean(self.coverage)),
            "interval_width_90": float(np.mean(self.width)),
        }


def observation_covariance(scenario: Scenario) -> torch.Tensor:
    size = int(scenario.observation_indices.numel())
    return scenario.config.obs_noise**2 * torch.eye(size, dtype=torch.float64, device=scenario.truth.device)


def uniform_weights(size: int, device: torch.device) -> torch.Tensor:
    return torch.full((size,), 1.0 / size, dtype=torch.float64, device=device)


def entropy(weights: torch.Tensor) -> torch.Tensor:
    safe = weights.clamp_min(1.0e-300)
    return -(safe * safe.log()).sum()


def entropy_project(weights: torch.Tensor, target_entropy: float) -> torch.Tensor:
    weights = weights / weights.sum().clamp_min(1.0e-300)
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
    out = (1.0 - high) * weights + high * uniform
    return out / out.sum().clamp_min(1.0e-300)


def evidence_score(
    ensemble_observation: torch.Tensor,
    observation: torch.Tensor,
    obs_noise: float,
    shrinkage: float,
    dimension_weights: torch.Tensor | None,
) -> torch.Tensor:
    mean = ensemble_observation.mean(dim=0)
    residual = observation - mean
    anomalies = ensemble_observation - mean
    covariance = anomalies.mT @ anomalies / max(ensemble_observation.shape[0] - 1, 1)
    covariance = (1.0 - shrinkage) * covariance + shrinkage * torch.diag(torch.diagonal(covariance))
    covariance = covariance + (obs_noise**2 + 1.0e-8) * torch.eye(
        observation.numel(),
        dtype=observation.dtype,
        device=observation.device,
    )
    if dimension_weights is not None:
        weights = dimension_weights.to(dtype=observation.dtype, device=observation.device).clamp_min(1.0e-8)
        weights = observation.numel() * weights / weights.sum().clamp_min(1.0e-12)
        variance = torch.diagonal(covariance).clamp_min(1.0e-12)
        marginal = residual.square() / variance + variance.log() + math.log(2.0 * math.pi)
        return -0.5 * (weights * marginal).sum()
    factor = stable_cholesky(covariance)
    solved = torch.cholesky_solve(residual[:, None], factor).squeeze(-1)
    log_det = 2.0 * torch.log(torch.diagonal(factor)).sum()
    return -0.5 * (residual @ solved + log_det + observation.numel() * math.log(2.0 * math.pi))


def weighted_denkf_analysis(
    state_ensemble: torch.Tensor,
    weights: torch.Tensor,
    observation: torch.Tensor,
    observation_operator: SparseObservation,
    observation_covariance: torch.Tensor,
) -> torch.Tensor:
    weights = weights.to(dtype=state_ensemble.dtype, device=state_ensemble.device)
    weights = weights.clamp_min(1.0e-300)
    weights = weights / weights.sum().clamp_min(1.0e-300)
    predicted = observation_operator(state_ensemble)
    x_mean = (weights[:, None] * state_ensemble).sum(dim=0)
    z_mean = (weights[:, None] * predicted).sum(dim=0)
    x_anom = state_ensemble - x_mean
    z_anom = predicted - z_mean
    denom = (1.0 - weights.square().sum()).clamp_min(torch.finfo(state_ensemble.dtype).eps)
    weighted_z = weights[:, None] * z_anom
    cross_covariance = x_anom.mT @ weighted_z / denom
    innovation_covariance = z_anom.mT @ weighted_z / denom + observation_covariance
    factor = stable_cholesky(innovation_covariance)
    gain = torch.cholesky_solve(cross_covariance.mT, factor).mT
    updated_mean = x_mean + gain @ (observation - z_mean)
    updated_anomalies = x_anom - 0.5 * (z_anom @ gain.mT)
    return updated_mean.unsqueeze(0) + updated_anomalies


def augmented_alpha_denkf(
    state_ensemble: torch.Tensor,
    alpha_ensemble: torch.Tensor,
    observation: torch.Tensor,
    observation_operator: SparseObservation,
    observation_covariance: torch.Tensor,
    config: Lorenz96Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    augmented = torch.cat([state_ensemble, alpha_ensemble[:, None]], dim=-1)
    updated = denkf_analysis(augmented, observation, observation_operator, observation_covariance)
    return updated[:, :-1], updated[:, -1].clamp(config.alpha_min, config.alpha_max)


def weighted_augmented_alpha_denkf(
    state_ensemble: torch.Tensor,
    alpha_ensemble: torch.Tensor,
    weights: torch.Tensor,
    observation: torch.Tensor,
    observation_operator: SparseObservation,
    observation_covariance: torch.Tensor,
    config: Lorenz96Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    augmented = torch.cat([state_ensemble, alpha_ensemble[:, None]], dim=-1)
    updated = weighted_denkf_analysis(augmented, weights, observation, observation_operator, observation_covariance)
    return updated[:, :-1], updated[:, -1].clamp(config.alpha_min, config.alpha_max)


def alpha_cloud(
    center: torch.Tensor,
    grid: torch.Tensor,
    ensemble_size: int,
    jitter_fraction: float,
    generator: torch.Generator,
) -> torch.Tensor:
    span = float(grid[-1] - grid[0])
    jitter = jitter_fraction * span
    values = center + jitter * torch.randn((ensemble_size,), dtype=grid.dtype, device=grid.device, generator=generator)
    return values.clamp(float(grid[0]), float(grid[-1]))


def tile_alpha_members(grid: torch.Tensor, ensemble_size: int, jitter_fraction: float, generator: torch.Generator) -> torch.Tensor:
    repeats = math.ceil(ensemble_size / int(grid.numel()))
    values = grid.repeat(repeats)[:ensemble_size].clone()
    span = float(grid[-1] - grid[0])
    values = values + jitter_fraction * span * torch.randn(
        values.shape,
        dtype=values.dtype,
        device=values.device,
        generator=generator,
    )
    return values.clamp(float(grid[0]), float(grid[-1]))


def interpolate_paths(old_alpha: torch.Tensor, values: torch.Tensor, new_alpha: torch.Tensor) -> torch.Tensor:
    out = []
    for alpha in new_alpha:
        if bool(alpha <= old_alpha[0]):
            out.append(values[0])
            continue
        if bool(alpha >= old_alpha[-1]):
            out.append(values[-1])
            continue
        right = int(torch.searchsorted(old_alpha, alpha).detach().cpu())
        left = right - 1
        fraction = ((alpha - old_alpha[left]) / (old_alpha[right] - old_alpha[left])).to(values.dtype)
        out.append((1.0 - fraction) * values[left] + fraction * values[right])
    return torch.stack(out, dim=0)


def local_refined_grid(alpha_grid: torch.Tensor, log_scores: torch.Tensor, config: Lorenz96Config) -> torch.Tensor:
    if float((log_scores.max() - log_scores.min()).detach().cpu()) < 1.0e-9:
        return alpha_grid
    center = alpha_grid[int(torch.argmax(log_scores))]
    radius = config.local_grid_radius
    left = max(config.alpha_min, float(center) - radius)
    right = min(config.alpha_max, float(center) + radius)
    target_width = min(2.0 * radius, config.alpha_max - config.alpha_min)
    if right - left < target_width:
        if left <= config.alpha_min:
            right = min(config.alpha_max, config.alpha_min + target_width)
        elif right >= config.alpha_max:
            left = max(config.alpha_min, config.alpha_max - target_width)
    new_grid = torch.linspace(left, right, config.local_grid_points, dtype=alpha_grid.dtype, device=alpha_grid.device)
    nearest = int(torch.argmin((new_grid - center).abs()))
    new_grid[nearest] = center
    keep = [new_grid[0]]
    for item in new_grid[1:]:
        if float(item - keep[-1]) >= config.local_grid_min_spacing:
            keep.append(item)
    return torch.stack(keep)


def run_fixed_filter(scenario: Scenario, method: Literal["denkf", "letkf"], *, record_trace: bool) -> dict[str, Any]:
    config = scenario.config
    device = scenario.truth.device
    system = Lorenz96(config)
    ensemble = scenario.initial_ensemble.clone()
    operator = SparseObservation(scenario.observation_indices)
    covariance = observation_covariance(scenario)
    weights = uniform_weights(config.ensemble_size, device)
    metrics = RunningMetrics()
    trace_mean: list[torch.Tensor] = []
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(config.steps + 1):
        mean_state = ensemble.mean(dim=0)
        if record_trace:
            trace_mean.append(mean_state.detach().cpu())
        metrics.add(ensemble, scenario.truth[step], weights, point_estimate=mean_state)
        if step == config.steps:
            break
        ensemble = system.step(ensemble, config.fixed_alpha, scenario.forecast_noise[step])
        if step + 1 in scenario.observations:
            if method == "denkf":
                ensemble = denkf_analysis(ensemble, scenario.observations[step + 1], operator, covariance)
            else:
                ensemble = letkf_analysis(ensemble, scenario.observations[step + 1], operator, covariance)
            ensemble = system.project(ensemble)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = metrics.finalize()
    result.update(
        runtime_seconds=float(time.perf_counter() - start),
        forward_member_steps=int(config.steps * config.ensemble_size),
        peak_gpu_memory_mb=float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
        alpha_estimate=float(config.fixed_alpha),
        alpha_map=float(config.fixed_alpha),
        alpha_absolute_error=abs(config.fixed_alpha - config.alpha_true),
        max_abs_state=float(torch.max(torch.abs(ensemble)).detach().cpu()),
    )
    if record_trace:
        result["mean_states"] = torch.stack(trace_mean).numpy()
    return result


def run_aug_enkf(scenario: Scenario, *, record_trace: bool) -> dict[str, Any]:
    config = scenario.config
    device = scenario.truth.device
    system = Lorenz96(config)
    ensemble = scenario.initial_ensemble.clone()
    generator = make_generator(device, config.seed * 1000 + 714_001)
    alpha = tile_alpha_members(scenario.alpha_grid, config.ensemble_size, config.aug_alpha_jitter, generator)
    operator = SparseObservation(scenario.observation_indices)
    covariance = observation_covariance(scenario)
    weights = uniform_weights(config.ensemble_size, device)
    metrics = RunningMetrics()
    trace_mean: list[torch.Tensor] = []
    trace_alpha: list[float] = []
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(config.steps + 1):
        mean_state = ensemble.mean(dim=0)
        if record_trace:
            trace_mean.append(mean_state.detach().cpu())
            trace_alpha.append(float(alpha.mean().detach().cpu()))
        metrics.add(ensemble, scenario.truth[step], weights, point_estimate=mean_state)
        if step == config.steps:
            break
        alpha = (
            alpha
            + config.aug_alpha_random_walk_std
            * torch.randn(alpha.shape, dtype=alpha.dtype, device=device, generator=generator)
        ).clamp(config.alpha_min, config.alpha_max)
        ensemble = system.step(ensemble, alpha, scenario.forecast_noise[step])
        if step + 1 in scenario.observations:
            augmented = torch.cat([ensemble, alpha[:, None]], dim=-1)
            updated = denkf_analysis(augmented, scenario.observations[step + 1], operator, covariance)
            ensemble = system.project(updated[:, :-1])
            alpha = updated[:, -1].clamp(config.alpha_min, config.alpha_max)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    alpha_est = float(alpha.mean().detach().cpu())
    result = metrics.finalize()
    result.update(
        runtime_seconds=float(time.perf_counter() - start),
        forward_member_steps=int(config.steps * config.ensemble_size),
        peak_gpu_memory_mb=float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
        alpha_estimate=alpha_est,
        alpha_map=float(alpha.median().detach().cpu()),
        alpha_absolute_error=abs(alpha_est - config.alpha_true),
        alpha_spread=float(alpha.std(unbiased=True).detach().cpu()),
        max_abs_state=float(torch.max(torch.abs(ensemble)).detach().cpu()),
    )
    if record_trace:
        result["mean_states"] = torch.stack(trace_mean).numpy()
        result["alpha_mean_history"] = np.asarray(trace_alpha, dtype=float)
    return result


def run_bma(scenario: Scenario, *, record_trace: bool) -> dict[str, Any]:
    config = scenario.config
    device = scenario.truth.device
    system = Lorenz96(config)
    path_count = int(scenario.alpha_grid.numel())
    branches = scenario.initial_ensemble.unsqueeze(0).repeat(path_count, 1, 1)
    log_weights = torch.zeros(path_count, dtype=torch.float64, device=device)
    path_weights = torch.softmax(log_weights, dim=0)
    operator = SparseObservation(scenario.observation_indices)
    covariance = observation_covariance(scenario)
    metrics = RunningMetrics()
    trace_mean: list[torch.Tensor] = []
    trace_weights: list[torch.Tensor] = []
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(config.steps + 1):
        flat = branches.reshape(-1, config.state_dim)
        flat_weights = path_weights[:, None].expand(-1, config.ensemble_size).reshape(-1) / config.ensemble_size
        estimate = (path_weights[:, None] * branches.mean(dim=1)).sum(dim=0)
        if record_trace:
            trace_mean.append(estimate.detach().cpu())
            trace_weights.append(path_weights.detach().cpu())
        metrics.add(flat, scenario.truth[step], flat_weights, point_estimate=estimate)
        if step == config.steps:
            break
        for idx, alpha in enumerate(scenario.alpha_grid):
            branches[idx] = system.step(branches[idx], float(alpha), scenario.forecast_noise[step])
        if step + 1 in scenario.observations:
            observation = scenario.observations[step + 1]
            branch_observations = torch.stack([operator(branch) for branch in branches])
            evidence = torch.stack(
                [
                    evidence_score(branch_observations[idx], observation, config.obs_noise, config.evidence_shrinkage, None)
                    for idx in range(path_count)
                ]
            )
            log_weights = log_weights + evidence - evidence.mean()
            path_weights = torch.softmax(log_weights, dim=0)
            for idx in range(path_count):
                branches[idx] = system.project(denkf_analysis(branches[idx], observation, operator, covariance))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    alpha_est = float((scenario.alpha_grid * path_weights).sum().detach().cpu())
    result = metrics.finalize()
    result.update(
        runtime_seconds=float(time.perf_counter() - start),
        forward_member_steps=int(config.steps * path_count * config.ensemble_size),
        peak_gpu_memory_mb=float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
        alpha_estimate=alpha_est,
        alpha_map=float(scenario.alpha_grid[int(torch.argmax(path_weights))].detach().cpu()),
        alpha_absolute_error=abs(alpha_est - config.alpha_true),
        alpha_final_entropy=float(entropy(path_weights).detach().cpu()),
        max_abs_state=float(torch.max(torch.abs(branches)).detach().cpu()),
    )
    if record_trace:
        result["mean_states"] = torch.stack(trace_mean).numpy()
        result["alpha_weight_history"] = torch.stack(trace_weights).numpy()
    return result


def pce_state_weights(method: Literal["pce", "apce"], alpha_weights: torch.Tensor, config: Lorenz96Config) -> torch.Tensor:
    if method == "apce":
        return entropy_project(alpha_weights, config.apce_entropy_floor)
    return alpha_weights / alpha_weights.sum().clamp_min(1.0e-300)


def maybe_regrid(
    alpha_grid: torch.Tensor,
    branches: torch.Tensor,
    shadow: torch.Tensor,
    alpha_members: torch.Tensor,
    log_scores: torch.Tensor,
    config: Lorenz96Config,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    new_grid = local_refined_grid(alpha_grid, log_scores, config)
    if new_grid.shape == alpha_grid.shape and bool(torch.allclose(new_grid, alpha_grid)):
        return alpha_grid, branches, shadow, alpha_members, log_scores, False
    return (
        new_grid,
        interpolate_paths(alpha_grid, branches, new_grid),
        interpolate_paths(alpha_grid, shadow, new_grid),
        interpolate_paths(alpha_grid, alpha_members, new_grid),
        interpolate_paths(alpha_grid, log_scores, new_grid),
        True,
    )


def run_pce_apce(scenario: Scenario, method: Literal["pce", "apce"], *, record_trace: bool) -> dict[str, Any]:
    config = scenario.config
    device = scenario.truth.device
    system = Lorenz96(config)
    generator = make_generator(device, config.seed * 1000 + (733_001 if method == "pce" else 733_501))
    alpha_grid = scenario.alpha_grid.clone()
    path_count = int(alpha_grid.numel())
    branches = scenario.initial_ensemble.unsqueeze(0).repeat(path_count, 1, 1)
    shadow = branches.clone()
    alpha_members = torch.stack(
        [
            alpha_cloud(alpha, alpha_grid, config.ensemble_size, config.branch_member_alpha_jitter, generator)
            for alpha in alpha_grid
        ]
    )
    log_scores = torch.zeros(path_count, dtype=torch.float64, device=device)
    alpha_weights = torch.softmax(log_scores, dim=0)
    state_weights = pce_state_weights(method, alpha_weights, config)
    operator = SparseObservation(scenario.observation_indices)
    covariance = observation_covariance(scenario)
    metrics = RunningMetrics()
    trace_mean: list[torch.Tensor] = []
    trace_weights: list[torch.Tensor] = []
    trace_alpha_grid: list[torch.Tensor] = []
    regrid_count = 0
    forward_member_steps = 0
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(config.steps + 1):
        path_count = int(alpha_grid.numel())
        flat = branches.reshape(-1, config.state_dim)
        flat_weights = state_weights[:, None].expand(-1, config.ensemble_size).reshape(-1) / config.ensemble_size
        estimate = (state_weights[:, None] * branches.mean(dim=1)).sum(dim=0)
        if record_trace:
            trace_mean.append(estimate.detach().cpu())
            trace_weights.append(state_weights.detach().cpu())
            trace_alpha_grid.append(alpha_grid.detach().cpu())
        metrics.add(flat, scenario.truth[step], flat_weights, point_estimate=estimate)
        if step == config.steps:
            break
        for idx in range(path_count):
            member_alpha = alpha_members[idx]
            branches[idx] = system.step(branches[idx], member_alpha, scenario.forecast_noise[step])
            shadow[idx] = system.step(shadow[idx], member_alpha, scenario.forecast_noise[step])
        forward_member_steps += 2 * path_count * config.ensemble_size
        if step + 1 not in scenario.observations:
            continue
        observation = scenario.observations[step + 1]
        shadow_observations = torch.stack([operator(path_shadow) for path_shadow in shadow])
        dimension_weights = None
        if method == "apce":
            between = shadow_observations.mean(dim=1).var(dim=0, unbiased=True)
            dimension_weights = config.apce_dimension_floor + config.apce_dimension_gain * between / between.max().clamp_min(1.0e-12)
        evidence = torch.stack(
            [
                evidence_score(
                    shadow_observations[idx],
                    observation,
                    config.obs_noise,
                    config.evidence_shrinkage,
                    dimension_weights,
                )
                for idx in range(path_count)
            ]
        )
        centered = evidence - evidence.mean()
        if method == "pce":
            log_scores = log_scores + config.pce_temperature * centered
        else:
            entropy_ratio = float(entropy(alpha_weights) / math.log(path_count))
            temperature = float(
                np.clip(
                    config.apce_temperature * entropy_ratio**0.75,
                    config.apce_min_temperature,
                    config.apce_temperature,
                )
            )
            log_scores = config.apce_forgetting * log_scores + temperature * centered
        alpha_weights = torch.softmax(log_scores, dim=0)
        state_weights = pce_state_weights(method, alpha_weights, config)
        if method == "apce":
            log_scores = state_weights.clamp_min(1.0e-300).log()
            alpha_weights = torch.softmax(log_scores, dim=0)
        alpha_grid, branches, shadow, alpha_members, log_scores, changed = maybe_regrid(
            alpha_grid,
            branches,
            shadow,
            alpha_members,
            log_scores,
            config,
        )
        if changed:
            regrid_count += 1
            alpha_weights = torch.softmax(log_scores, dim=0)
            state_weights = pce_state_weights(method, alpha_weights, config)
        path_count = int(alpha_grid.numel())
        local_analysis = torch.empty_like(branches)
        for idx in range(path_count):
            local_analysis[idx] = denkf_analysis(branches[idx], observation, operator, covariance)
        branches = local_analysis
        if config.global_state_analysis_strength > 1.0e-12:
            flat_forecast = branches.reshape(-1, config.state_dim)
            flat_analysis_weights = state_weights[:, None].expand(-1, config.ensemble_size).reshape(-1) / config.ensemble_size
            global_analysis = weighted_denkf_analysis(flat_forecast, flat_analysis_weights, observation, operator, covariance)
            branches = (
                (1.0 - config.global_state_analysis_strength) * branches
                + config.global_state_analysis_strength * global_analysis.reshape_as(branches)
            )
        if config.branch_augmented_alpha_analysis_strength > 1.0e-12:
            joint_branches = torch.empty_like(branches)
            joint_alpha = torch.empty_like(alpha_members)
            for idx in range(path_count):
                state_update, alpha_update = augmented_alpha_denkf(
                    branches[idx],
                    alpha_members[idx],
                    observation,
                    operator,
                    covariance,
                    config,
                )
                joint_branches[idx] = state_update
                joint_alpha[idx] = alpha_update
            strength = config.branch_augmented_alpha_analysis_strength
            branches = (1.0 - strength) * branches + strength * joint_branches
            alpha_members = (1.0 - strength) * alpha_members + strength * joint_alpha
        if config.global_augmented_alpha_analysis_strength > 1.0e-12:
            flat_branches = branches.reshape(-1, config.state_dim)
            flat_alpha = alpha_members.reshape(-1)
            flat_weights = state_weights[:, None].expand(-1, config.ensemble_size).reshape(-1) / config.ensemble_size
            state_update, alpha_update = weighted_augmented_alpha_denkf(
                flat_branches,
                flat_alpha,
                flat_weights,
                observation,
                operator,
                covariance,
                config,
            )
            strength = config.global_augmented_alpha_analysis_strength
            branches = (1.0 - strength) * branches + strength * state_update.reshape_as(branches)
            alpha_members = (1.0 - strength) * alpha_members + strength * alpha_update.reshape_as(alpha_members)
        branches = system.project(branches)
        alpha_members = alpha_members.clamp(config.alpha_min, config.alpha_max)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    alpha_branch_means = alpha_members.mean(dim=1)
    alpha_est = float((state_weights * alpha_branch_means).sum().detach().cpu())
    result = metrics.finalize()
    result.update(
        runtime_seconds=float(time.perf_counter() - start),
        forward_member_steps=int(forward_member_steps),
        peak_gpu_memory_mb=float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
        alpha_estimate=alpha_est,
        alpha_map=float(alpha_branch_means[int(torch.argmax(state_weights))].detach().cpu()),
        alpha_absolute_error=abs(alpha_est - config.alpha_true),
        alpha_final_entropy=float(entropy(state_weights).detach().cpu()),
        alpha_evidence_entropy=float(entropy(alpha_weights).detach().cpu()),
        alpha_regrid_count=int(regrid_count),
        alpha_grid_points=int(alpha_grid.numel()),
        max_abs_state=float(torch.max(torch.abs(branches)).detach().cpu()),
    )
    if record_trace:
        result["mean_states"] = torch.stack(trace_mean).numpy()
        result["alpha_weight_history"] = pad_history(trace_weights)
        result["alpha_grid_history"] = pad_history(trace_alpha_grid)
    return result


def pad_history(history: list[torch.Tensor]) -> np.ndarray:
    if not history:
        return np.empty((0, 0), dtype=float)
    width = max(int(item.numel()) for item in history)
    out = np.full((len(history), width), np.nan, dtype=float)
    for idx, item in enumerate(history):
        values = item.detach().cpu().numpy().reshape(-1)
        out[idx, : values.size] = values
    return out


def run_method(scenario: Scenario, method: MethodName, *, record_trace: bool) -> dict[str, Any]:
    if method in {"denkf", "letkf"}:
        return run_fixed_filter(scenario, method, record_trace=record_trace)
    if method == "aug_enkf":
        return run_aug_enkf(scenario, record_trace=record_trace)
    if method == "bma_static":
        return run_bma(scenario, record_trace=record_trace)
    if method in {"pce", "apce"}:
        return run_pce_apce(scenario, method, record_trace=record_trace)
    raise ValueError(method)


def numerical_status(result: dict[str, Any], scenario: Scenario) -> str:
    keys = ("nrmse", "rmse", "crps", "coverage_90", "interval_width_90", "max_abs_state")
    if not all(math.isfinite(float(result[key])) for key in keys):
        return "nonfinite"
    truth_scale = float(torch.max(torch.abs(scenario.truth)).detach().cpu())
    if truth_scale > 0.0 and float(result["max_abs_state"]) > scenario.config.max_valid_amplitude_ratio * truth_scale:
        return "diverged"
    return "valid"


def save_trace(output: Path, method: MethodName, seed: int, scenario: Scenario, result: dict[str, Any]) -> str:
    trace_dir = output / "traces" / method
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"lorenz96_{method}_seed_{seed}.npz"
    observations = np.full((scenario.config.steps + 1, scenario.observation_indices.numel()), np.nan, dtype=float)
    for step, obs in scenario.observations.items():
        observations[step] = obs.detach().cpu().numpy()
    payload: dict[str, Any] = {
        "truth": scenario.truth.detach().cpu().numpy(),
        "observations": observations,
        "observation_indices": scenario.observation_indices.detach().cpu().numpy(),
        "times": scenario.times.detach().cpu().numpy(),
        "alpha_true": np.asarray(scenario.config.alpha_true),
        "alpha_grid": scenario.alpha_grid.detach().cpu().numpy(),
    }
    for key in ("mean_states", "alpha_mean_history", "alpha_weight_history", "alpha_grid_history"):
        if key in result:
            payload[key] = result[key]
    np.savez_compressed(path, **payload)
    return str(path)


def run_one(scenario: Scenario, method: MethodName, output: Path, record_trace: bool) -> dict[str, Any]:
    started = time.perf_counter()
    result = run_method(scenario, method, record_trace=record_trace)
    trace_path = save_trace(output, method, scenario.config.seed, scenario, result) if record_trace else ""
    row: dict[str, Any] = {
        "case": "lorenz96",
        "method": method,
        "label": METHOD_LABELS[method],
        "seed": scenario.config.seed,
        "status": "completed",
        "numerical_status": numerical_status(result, scenario),
        "state_dim": scenario.config.state_dim,
        "observed_state_count": int(scenario.observation_indices.numel()),
        "observed_state_indices": ",".join(str(int(x)) for x in scenario.observation_indices.detach().cpu().tolist()),
        "dt": scenario.config.dt,
        "steps": scenario.config.steps,
        "obs_interval": scenario.config.obs_interval,
        "obs_stride": scenario.config.obs_stride,
        "obs_noise": scenario.config.obs_noise,
        "ensemble_size": scenario.config.ensemble_size,
        "alpha_true": scenario.config.alpha_true,
        "fixed_alpha": scenario.config.fixed_alpha,
        "alpha_grid_initial": ",".join(f"{x:.4g}" for x in scenario.config.alpha_grid),
        "run_total_wall_seconds": float(time.perf_counter() - started),
        "trace_npz": trace_path,
    }
    for key, value in result.items():
        if isinstance(value, (float, int, str)):
            row[key] = value
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_ci(values: np.ndarray, seed: int, n_bootstrap: int = 5000) -> list[float]:
    if values.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, values.size, size=(n_bootstrap, values.size))].mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def summarize(
    records: list[dict[str, Any]],
    seed: int,
    methods: tuple[MethodName, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in methods:
        subset = [row for row in records if row["method"] == method]
        out: dict[str, Any] = {
            "case": "lorenz96",
            "method": method,
            "label": METHOD_LABELS[method],
            "n_seeds": len(subset),
            "valid_count": sum(row.get("numerical_status") == "valid" for row in subset),
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
            out[key] = float(np.mean(values)) if values.size else float("nan")
            out[f"{key}_ci95"] = bootstrap_ci(values, seed + 193 * (len(rows) + 1) + len(key))
        rows.append(out)
    return rows


def paired_gain_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["method"], int(row["seed"])): row for row in records}
    seeds = sorted({int(row["seed"]) for row in records})
    rows: list[dict[str, Any]] = []
    for method in ("pce", "apce"):
        for baseline in ("aug_enkf", "bma_static", "denkf", "letkf"):
            paired = [
                (by_key[(method, seed)], by_key[(baseline, seed)])
                for seed in seeds
                if (method, seed) in by_key and (baseline, seed) in by_key
            ]
            if not paired:
                continue
            rows.append(
                {
                    "case": "lorenz96",
                    "method": method,
                    "baseline": baseline,
                    "paired_seed_count": len(paired),
                    "nrmse_win_count": sum(float(row["nrmse"]) < float(base["nrmse"]) for row, base in paired),
                    "crps_win_count": sum(float(row["crps"]) < float(base["crps"]) for row, base in paired),
                    "alpha_mae_win_count": sum(
                        float(row["alpha_absolute_error"]) < float(base["alpha_absolute_error"]) for row, base in paired
                    ),
                    "mean_nrmse_gain": float(np.mean([float(base["nrmse"]) - float(row["nrmse"]) for row, base in paired])),
                    "mean_crps_gain": float(np.mean([float(base["crps"]) - float(row["crps"]) for row, base in paired])),
                    "mean_alpha_mae_gain": float(
                        np.mean([float(base["alpha_absolute_error"]) - float(row["alpha_absolute_error"]) for row, base in paired])
                    ),
                }
            )
    return rows


def write_report(
    output: Path,
    summary: list[dict[str, Any]],
    gains: list[dict[str, Any]],
    config: Lorenz96Config,
) -> None:
    observed = ",".join(str(index) for index in range(0, config.state_dim, config.obs_stride))
    lines = [
        "# Figure 4 Lorenz-96 rewrite full-method smoke",
        "",
        (
            f"Protocol: Lorenz-96 only; D={config.state_dim}; observed states are {observed}; "
            f"{config.steps} steps; observations every {config.obs_interval} steps."
        ),
        "",
        "| Method | valid | nRMSE | CRPS | alpha MAE | coverage | width | runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {label} | {valid_count}/{n_seeds} | {nrmse:.4f} | {crps:.4f} | {alpha_absolute_error:.4f} | {coverage_90:.3f} | {interval_width_90:.3f} | {runtime_seconds:.2f} |".format(
                **row
            )
        )
    lines += ["", "## Paired gains", ""]
    for row in gains:
        lines.append(
            "- {method} vs {baseline}: nRMSE wins {nrmse_win_count}/{paired_seed_count}, "
            "CRPS wins {crps_win_count}/{paired_seed_count}, alpha-MAE wins {alpha_mae_win_count}/{paired_seed_count}; "
            "mean nRMSE gain={mean_nrmse_gain:.4f}, CRPS gain={mean_crps_gain:.4f}, alpha-MAE gain={mean_alpha_mae_gain:.4f}.".format(
                **row
            )
        )
    (output / "LORENZ96_REWRITE_FULLMETHODS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_methods(text: str) -> tuple[MethodName, ...]:
    methods = tuple(item.strip() for item in text.split(",") if item.strip())
    unknown = [item for item in methods if item not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Available: {METHODS}")
    return methods  # type: ignore[return-value]


def run_suite(
    *,
    n_seeds: int,
    base_seed: int,
    output: Path,
    device: torch.device,
    methods: tuple[MethodName, ...],
    record_trace: bool,
    obs_stride: int,
    obs_interval: int,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    total = n_seeds * len(methods)
    counter = 0
    for seed_offset in range(n_seeds):
        seed = base_seed + seed_offset
        scenario = generate_scenario(
            seed,
            device,
            obs_stride=obs_stride,
            obs_interval=obs_interval,
        )
        for method in methods:
            counter += 1
            row = run_one(scenario, method, output, record_trace)
            records.append(row)
            print(
                f"[{counter}/{total}] lorenz96 seed={seed} method={method} "
                f"nrmse={float(row['nrmse']):.5f} crps={float(row['crps']):.5f} "
                f"alpha_mae={float(row['alpha_absolute_error']):.5f} status={row['numerical_status']}",
                flush=True,
            )
    summary = summarize(records, base_seed, methods)
    gains = paired_gain_rows(records)
    write_csv(output / "run_metrics.csv", records)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "paired_gains.csv", gains)
    write_report(
        output,
        summary,
        gains,
        Lorenz96Config(
            seed=base_seed,
            obs_stride=obs_stride,
            obs_interval=obs_interval,
        ),
    )
    manifest = {
        "protocol": "figure4-lorenz96-only-rewrite-fullmethods",
        "case": "lorenz96",
        "n_seeds": n_seeds,
        "base_seed": base_seed,
        "methods": list(methods),
        "device": str(device),
        "record_trace": bool(record_trace),
        "config": asdict(
            Lorenz96Config(
                seed=base_seed,
                obs_stride=obs_stride,
                obs_interval=obs_interval,
            )
        ),
        "source_files": {
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": file_sha256(Path(__file__).resolve()),
        },
        "summary": summary,
        "paired_gains": gains,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean Lorenz-96-only full-method Figure 4 smoke runner.")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=2026080600)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--obs-stride", type=int, default=2)
    parser.add_argument("--obs-interval", type=int, default=5)
    parser.add_argument("--no-record-trace", action="store_true")
    args = parser.parse_args()
    manifest = run_suite(
        n_seeds=args.n_seeds,
        base_seed=args.base_seed,
        output=args.output,
        device=torch.device(args.device),
        methods=parse_methods(args.methods),
        record_trace=not args.no_record_trace,
        obs_stride=args.obs_stride,
        obs_interval=args.obs_interval,
    )
    print(json.dumps({"output": str(args.output), "summary": manifest["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
