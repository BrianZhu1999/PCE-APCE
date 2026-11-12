from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pce_assimilation.evidence import liu_quantile
from pce_assimilation.ensemble_filters import denkf_analysis
from pce_assimilation.math_utils import stable_cholesky
from pce_assimilation.metrics import weighted_central_interval_coverage_width, weighted_ensemble_crps
from pce_assimilation.observations import SparseObservation


MethodName = Literal["aug_enkf", "bma_static", "pce", "apce"]
METHODS: tuple[MethodName, ...] = ("aug_enkf", "bma_static", "pce", "apce")
METHOD_LABELS = {
    "aug_enkf": "Aug-EnKF",
    "bma_static": "BMA",
    "pce": "PCE",
    "apce": "APCE",
}

@dataclass(frozen=True)
class L96ScalingConfig:
    seed: int
    state_dim: int = 1024
    observed_points: int = 128
    obs_interval: int = 8
    steps: int = 300
    dt: float = 0.01
    spinup_steps: int = 900
    ensemble_size: int = 64
    obs_noise: float = 0.45
    alpha_true: float = 0.12
    alpha_min: float = 0.08
    alpha_max: float = 0.92
    fixed_alpha: float = 0.50
    forcing_base: float = 8.0
    forcing_scale: float = 1.55
    stochastic_scale: float = 0.035
    initial_spread: float = 0.42
    coarse_alpha_grid: tuple[float, ...] = (0.08, 0.22, 0.36, 0.50, 0.64, 0.78, 0.92)
    bma_alpha_grid_size: int = 21
    pce_temperature: float = 0.20
    apce_temperature: float = 0.50
    apce_min_temperature: float = 0.14
    apce_forgetting: float = 0.985
    apce_entropy_floor: float = 0.45
    apce_recycle_entropy_projected_scores: bool = True
    evidence_shrinkage: float = 0.22
    apce_dimension_floor: float = 0.30
    apce_dimension_gain: float = 0.70
    branch_member_alpha_jitter: float = 0.030
    aug_alpha_jitter: float = 0.035
    aug_alpha_random_walk_std: float = 0.004
    branch_augmented_alpha_analysis_strength: float = 0.25
    global_augmented_alpha_analysis_strength: float = 0.05
    global_state_analysis_strength: float = 0.12
    local_grid_points: int = 11
    local_grid_radius: float = 0.18
    local_grid_min_spacing: float = 0.012
    dynamic_regrid_from_alpha_members: bool = False
    localization_scale: float = 32.0
    probabilistic_metric_stride: int = 5
    max_valid_amplitude_ratio: float = 100.0


@dataclass(frozen=True)
class SharedAssets:
    config: L96ScalingConfig
    truth: torch.Tensor
    initial_ensemble: torch.Tensor
    forecast_noise: torch.Tensor
    observation_noise: torch.Tensor
    observation_indices: torch.Tensor
    asset_path: Path


@dataclass(frozen=True)
class Scenario:
    config: L96ScalingConfig
    truth: torch.Tensor
    observations: dict[int, torch.Tensor]
    initial_ensemble: torch.Tensor
    forecast_noise: torch.Tensor
    observation_indices: torch.Tensor
    localization: torch.Tensor
    augmented_localization: torch.Tensor
    asset_path: Path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(seed)
    return generator


def randn(shape: tuple[int, ...], device: torch.device, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(shape, dtype=torch.float64, device=device, generator=generator)


def smooth_periodic_noise(noise: torch.Tensor, passes: int = 1) -> torch.Tensor:
    output = noise
    for _ in range(passes):
        output = (
            0.20 * torch.roll(output, 2, dims=-1)
            + 0.25 * torch.roll(output, 1, dims=-1)
            + 0.10 * output
            + 0.25 * torch.roll(output, -1, dims=-1)
            + 0.20 * torch.roll(output, -2, dims=-1)
        )
    return output / output.std(dim=-1, keepdim=True).clamp_min(1.0e-12)


class Lorenz96:
    def __init__(self, config: L96ScalingConfig) -> None:
        self.config = config

    def drift(self, state: torch.Tensor, alpha: torch.Tensor | float) -> torch.Tensor:
        alpha_tensor = torch.as_tensor(alpha, dtype=state.dtype, device=state.device)
        forcing = self.config.forcing_base + self.config.forcing_scale * liu_quantile(alpha_tensor)
        while forcing.ndim < state.ndim:
            forcing = forcing.unsqueeze(-1)
        return (
            (torch.roll(state, shifts=-1, dims=-1) - torch.roll(state, shifts=2, dims=-1))
            * torch.roll(state, shifts=1, dims=-1)
            - state
            + forcing
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
        return self.project(deterministic + math.sqrt(dt) * self.config.stochastic_scale * noise)


def observation_indices(config: L96ScalingConfig, device: torch.device) -> torch.Tensor:
    if config.state_dim % config.observed_points != 0:
        raise ValueError("observed_points must evenly divide state_dim")
    stride = config.state_dim // config.observed_points
    return torch.arange(0, config.state_dim, stride, dtype=torch.int64, device=device)


def gaspari_cohn(distance_ratio: torch.Tensor) -> torch.Tensor:
    distance = distance_ratio.abs()
    output = torch.zeros_like(distance)
    inner = distance <= 1.0
    middle = (distance > 1.0) & (distance <= 2.0)
    first = distance[inner]
    output[inner] = 1.0 - (5.0 / 3.0) * first.square() + 0.625 * first.pow(3) + 0.5 * first.pow(4) - 0.25 * first.pow(5)
    second = distance[middle]
    output[middle] = (
        4.0
        - 5.0 * second
        + (5.0 / 3.0) * second.square()
        + 0.625 * second.pow(3)
        - 0.5 * second.pow(4)
        + (1.0 / 12.0) * second.pow(5)
        - 2.0 / (3.0 * second)
    )
    return output.clamp(0.0, 1.0)


def cyclic_localization(config: L96ScalingConfig, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    state_indices = torch.arange(config.state_dim, dtype=torch.float64, device=indices.device)[:, None]
    observed = indices.to(dtype=torch.float64)[None, :]
    difference = (state_indices - observed).abs()
    distance = torch.minimum(difference, config.state_dim - difference)
    physical = gaspari_cohn(distance / config.localization_scale)
    augmented = torch.cat((physical, torch.ones((1, physical.shape[1]), dtype=physical.dtype, device=physical.device)), dim=0)
    return physical, augmented


def asset_path(asset_root: Path, seed: int) -> Path:
    return asset_root / f"lorenz96_1024_shared_seed_{seed}.npz"


def create_shared_assets(config: L96ScalingConfig, asset_root: Path, device: torch.device) -> Path:
    asset_root.mkdir(parents=True, exist_ok=True)
    path = asset_path(asset_root, config.seed)
    if path.exists():
        return path
    system = Lorenz96(config)
    generator = make_generator(device, config.seed)
    state = 8.0 + 0.25 * randn((config.state_dim,), device, generator)
    zeros = torch.zeros_like(state)
    for _ in range(config.spinup_steps):
        state = system.step(state, config.alpha_true, zeros)
    idx = observation_indices(config, device)
    truth_noise = smooth_periodic_noise(randn((config.steps, config.state_dim), device, generator), passes=1)
    forecast_noise = smooth_periodic_noise(
        randn((config.steps, config.ensemble_size, config.state_dim), device, generator),
        passes=1,
    )
    initial_noise = smooth_periodic_noise(randn((config.ensemble_size, config.state_dim), device, generator), passes=1)
    observation_noise = randn((config.steps, config.observed_points), device, generator)
    truth = torch.empty((config.steps + 1, config.state_dim), dtype=torch.float64, device=device)
    truth[0] = state
    for step in range(config.steps):
        truth[step + 1] = system.step(truth[step], config.alpha_true, truth_noise[step])
    initial_ensemble = system.project(state.unsqueeze(0) + config.initial_spread * initial_noise)
    tmp_path = path.with_name(f"{path.stem}.tmp_{os.getpid()}.npz")
    np.savez_compressed(
        tmp_path,
        truth=truth.detach().cpu().numpy(),
        initial_ensemble=initial_ensemble.detach().cpu().numpy(),
        forecast_noise=forecast_noise.detach().cpu().numpy(),
        observation_noise=observation_noise.detach().cpu().numpy(),
        observation_indices=idx.detach().cpu().numpy(),
        config_json=np.asarray(json.dumps(asdict(config), ensure_ascii=False)),
    )
    tmp_path.replace(path)
    return path


def load_shared_assets(config: L96ScalingConfig, asset_root: Path, device: torch.device) -> SharedAssets:
    path = create_shared_assets(config, asset_root, device)
    with np.load(path, allow_pickle=False) as data:
        stored_config = json.loads(str(data["config_json"].item()))
        immutable_fields = ("state_dim", "observed_points", "steps", "dt", "ensemble_size", "alpha_true")
        for field in immutable_fields:
            if stored_config[field] != asdict(config)[field]:
                raise RuntimeError(f"Shared asset mismatch for {field}: {stored_config[field]} != {asdict(config)[field]}")
        return SharedAssets(
            config=config,
            truth=torch.as_tensor(data["truth"], dtype=torch.float64, device=device),
            initial_ensemble=torch.as_tensor(data["initial_ensemble"], dtype=torch.float64, device=device),
            forecast_noise=torch.as_tensor(data["forecast_noise"], dtype=torch.float64, device=device),
            observation_noise=torch.as_tensor(data["observation_noise"], dtype=torch.float64, device=device),
            observation_indices=torch.as_tensor(data["observation_indices"], dtype=torch.int64, device=device),
            asset_path=path,
        )


def materialize_scenario(shared: SharedAssets) -> Scenario:
    config = shared.config
    observations = {
        step: shared.truth[step, shared.observation_indices] + config.obs_noise * shared.observation_noise[step - 1]
        for step in range(config.obs_interval, config.steps + 1, config.obs_interval)
    }
    physical, augmented = cyclic_localization(config, shared.observation_indices)
    return Scenario(
        config=config,
        truth=shared.truth,
        observations=observations,
        initial_ensemble=shared.initial_ensemble,
        forecast_noise=shared.forecast_noise,
        observation_indices=shared.observation_indices,
        localization=physical,
        augmented_localization=augmented,
        asset_path=shared.asset_path,
    )


class RunningMetrics:
    def __init__(self) -> None:
        self.squared_error = 0.0
        self.truth_energy = 0.0
        self.points = 0
        self.crps_values: list[float] = []
        self.coverage_values: list[float] = []
        self.width_values: list[float] = []

    def add(
        self,
        ensemble: torch.Tensor,
        truth: torch.Tensor,
        weights: torch.Tensor,
        *,
        point_estimate: torch.Tensor,
        probabilistic: bool,
    ) -> None:
        normalized = weights / weights.sum().clamp_min(1.0e-300)
        self.squared_error += float((point_estimate - truth).square().sum().detach().cpu())
        self.truth_energy += float(truth.square().sum().detach().cpu())
        self.points += int(truth.numel())
        if probabilistic:
            self.crps_values.append(float(weighted_ensemble_crps(ensemble, truth, normalized).detach().cpu()))
            coverage, width = weighted_central_interval_coverage_width(ensemble, truth, normalized, level=0.90)
            self.coverage_values.append(float(coverage.detach().cpu()))
            self.width_values.append(float(width.detach().cpu()))

    def finalize(self) -> dict[str, float]:
        return {
            "nrmse": math.sqrt(self.squared_error / max(self.truth_energy, 1.0e-30)),
            "rmse": math.sqrt(self.squared_error / max(self.points, 1)),
            "crps": float(np.mean(self.crps_values)),
            "coverage_90": float(np.mean(self.coverage_values)),
            "interval_width_90": float(np.mean(self.width_values)),
        }


def covariance(scenario: Scenario) -> torch.Tensor:
    size = int(scenario.observation_indices.numel())
    return scenario.config.obs_noise**2 * torch.eye(size, dtype=torch.float64, device=scenario.truth.device)


def entropy(weights: torch.Tensor) -> torch.Tensor:
    safe = weights.clamp_min(1.0e-300)
    return -(safe * safe.log()).sum()


def entropy_project(weights: torch.Tensor, target_entropy: float) -> torch.Tensor:
    normalized = weights / weights.sum().clamp_min(1.0e-300)
    if float(entropy(normalized)) >= target_entropy:
        return normalized
    uniform = torch.full_like(normalized, 1.0 / normalized.numel())
    low, high = 0.0, 1.0
    for _ in range(45):
        middle = 0.5 * (low + high)
        mixed = (1.0 - middle) * normalized + middle * uniform
        if float(entropy(mixed)) < target_entropy:
            low = middle
        else:
            high = middle
    output = (1.0 - high) * normalized + high * uniform
    return output / output.sum().clamp_min(1.0e-300)


def evidence_score(
    ensemble_observation: torch.Tensor,
    observation: torch.Tensor,
    config: L96ScalingConfig,
    dimension_weights: torch.Tensor | None,
) -> torch.Tensor:
    mean = ensemble_observation.mean(dim=0)
    residual = observation - mean
    anomalies = ensemble_observation - mean
    matrix = anomalies.mT @ anomalies / max(ensemble_observation.shape[0] - 1, 1)
    matrix = (1.0 - config.evidence_shrinkage) * matrix + config.evidence_shrinkage * torch.diag(torch.diagonal(matrix))
    matrix = matrix + (config.obs_noise**2 + 1.0e-8) * torch.eye(
        observation.numel(),
        dtype=observation.dtype,
        device=observation.device,
    )
    if dimension_weights is not None:
        weights = dimension_weights.to(dtype=observation.dtype, device=observation.device).clamp_min(1.0e-8)
        weights = observation.numel() * weights / weights.sum().clamp_min(1.0e-12)
        variances = torch.diagonal(matrix).clamp_min(1.0e-12)
        marginal = residual.square() / variances + variances.log() + math.log(2.0 * math.pi)
        return -0.5 * (weights * marginal).sum()
    factor = stable_cholesky(matrix)
    solved = torch.cholesky_solve(residual[:, None], factor).squeeze(-1)
    log_det = 2.0 * torch.log(torch.diagonal(factor)).sum()
    return -0.5 * (residual @ solved + log_det + observation.numel() * math.log(2.0 * math.pi))


def weighted_denkf_analysis(
    state_ensemble: torch.Tensor,
    weights: torch.Tensor,
    observation: torch.Tensor,
    operator: SparseObservation,
    observation_covariance: torch.Tensor,
    localization: torch.Tensor,
) -> torch.Tensor:
    weights = weights.to(dtype=state_ensemble.dtype, device=state_ensemble.device).clamp_min(1.0e-300)
    weights = weights / weights.sum().clamp_min(1.0e-300)
    predicted = operator(state_ensemble)
    state_mean = (weights[:, None] * state_ensemble).sum(dim=0)
    observed_mean = (weights[:, None] * predicted).sum(dim=0)
    state_anomalies = state_ensemble - state_mean
    observed_anomalies = predicted - observed_mean
    denominator = (1.0 - weights.square().sum()).clamp_min(torch.finfo(state_ensemble.dtype).eps)
    weighted_observed_anomalies = weights[:, None] * observed_anomalies
    cross_covariance = state_anomalies.mT @ weighted_observed_anomalies / denominator
    innovation_covariance = observed_anomalies.mT @ weighted_observed_anomalies / denominator + observation_covariance
    factor = stable_cholesky(innovation_covariance)
    gain = torch.cholesky_solve(cross_covariance.mT, factor).mT * localization
    updated_mean = state_mean + gain @ (observation - observed_mean)
    updated_anomalies = state_anomalies - 0.5 * (observed_anomalies @ gain.mT)
    return updated_mean.unsqueeze(0) + updated_anomalies


def augmented_denkf_analysis(
    state_ensemble: torch.Tensor,
    alpha_ensemble: torch.Tensor,
    observation: torch.Tensor,
    operator: SparseObservation,
    observation_covariance: torch.Tensor,
    config: L96ScalingConfig,
    localization: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    augmented = torch.cat((state_ensemble, alpha_ensemble[:, None]), dim=-1)
    updated = denkf_analysis(augmented, observation, operator, observation_covariance, localization=localization)
    return updated[:, :-1], updated[:, -1].clamp(config.alpha_min, config.alpha_max)


def weighted_augmented_denkf_analysis(
    state_ensemble: torch.Tensor,
    alpha_ensemble: torch.Tensor,
    weights: torch.Tensor,
    observation: torch.Tensor,
    operator: SparseObservation,
    observation_covariance: torch.Tensor,
    config: L96ScalingConfig,
    localization: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    augmented = torch.cat((state_ensemble, alpha_ensemble[:, None]), dim=-1)
    updated = weighted_denkf_analysis(
        augmented,
        weights,
        observation,
        operator,
        observation_covariance,
        localization,
    )
    return updated[:, :-1], updated[:, -1].clamp(config.alpha_min, config.alpha_max)


def alpha_cloud(
    center: torch.Tensor,
    bounds: tuple[float, float],
    ensemble_size: int,
    jitter_fraction: float,
    generator: torch.Generator,
) -> torch.Tensor:
    span = bounds[1] - bounds[0]
    return (
        center
        + jitter_fraction * span * torch.randn((ensemble_size,), dtype=center.dtype, device=center.device, generator=generator)
    ).clamp(bounds[0], bounds[1])


def augmented_alpha_initial(config: L96ScalingConfig, device: torch.device) -> torch.Tensor:
    return torch.linspace(config.alpha_min, config.alpha_max, config.ensemble_size, dtype=torch.float64, device=device)


def interpolate_paths(old_grid: torch.Tensor, values: torch.Tensor, new_grid: torch.Tensor) -> torch.Tensor:
    rows = []
    for alpha in new_grid:
        if bool(alpha <= old_grid[0]):
            rows.append(values[0])
            continue
        if bool(alpha >= old_grid[-1]):
            rows.append(values[-1])
            continue
        right = int(torch.searchsorted(old_grid, alpha).detach().cpu())
        left = right - 1
        fraction = ((alpha - old_grid[left]) / (old_grid[right] - old_grid[left])).to(values.dtype)
        rows.append((1.0 - fraction) * values[left] + fraction * values[right])
    return torch.stack(rows, dim=0)


def dynamic_grid_coordinates(
    grid: torch.Tensor,
    branches: torch.Tensor,
    shadow: torch.Tensor,
    alpha_members: torch.Tensor,
    scores: torch.Tensor,
    config: L96ScalingConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Optionally regrid paths around their current alpha posterior means."""
    if not config.dynamic_regrid_from_alpha_members:
        return grid, branches, shadow, alpha_members, scores, False
    centers = alpha_members.mean(dim=1).clamp(config.alpha_min, config.alpha_max)
    order = torch.argsort(centers)
    centers = centers[order]
    if centers.numel() > 1 and bool(torch.any((centers[1:] - centers[:-1]) <= 1.0e-8)):
        return grid, branches, shadow, alpha_members, scores, False
    return (
        centers,
        branches[order],
        shadow[order],
        alpha_members[order],
        scores[order],
        True,
    )


def local_grid(grid: torch.Tensor, scores: torch.Tensor, config: L96ScalingConfig) -> torch.Tensor:
    if float((scores.max() - scores.min()).detach().cpu()) < 1.0e-9:
        return grid
    center = grid[int(torch.argmax(scores))]
    target_width = min(2.0 * config.local_grid_radius, config.alpha_max - config.alpha_min)
    left = max(config.alpha_min, float(center) - config.local_grid_radius)
    right = min(config.alpha_max, float(center) + config.local_grid_radius)
    if right - left < target_width:
        if left <= config.alpha_min:
            right = min(config.alpha_max, config.alpha_min + target_width)
        else:
            left = max(config.alpha_min, config.alpha_max - target_width)
    result = torch.linspace(left, right, config.local_grid_points, dtype=grid.dtype, device=grid.device)
    result[int(torch.argmin((result - center).abs()))] = center
    kept = [result[0]]
    for item in result[1:]:
        if float(item - kept[-1]) >= config.local_grid_min_spacing:
            kept.append(item)
    return torch.stack(kept)


def maybe_regrid(
    grid: torch.Tensor,
    branches: torch.Tensor,
    shadow: torch.Tensor,
    alpha_members: torch.Tensor,
    scores: torch.Tensor,
    config: L96ScalingConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    refined = local_grid(grid, scores, config)
    if refined.shape == grid.shape and bool(torch.allclose(refined, grid)):
        return grid, branches, shadow, alpha_members, scores, False
    return (
        refined,
        interpolate_paths(grid, branches, refined),
        interpolate_paths(grid, shadow, refined),
        interpolate_paths(grid, alpha_members, refined),
        interpolate_paths(grid, scores, refined),
        True,
    )


def should_score_probabilistic(step: int, config: L96ScalingConfig) -> bool:
    return step % config.probabilistic_metric_stride == 0 or step == config.steps


def start_runtime(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    return time.perf_counter()


def finalize_runtime(started: float, device: torch.device) -> tuple[float, float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        memory = float(torch.cuda.max_memory_allocated(device) / 1024**2)
    else:
        memory = 0.0
    return float(time.perf_counter() - started), memory


def run_aug_enkf(scenario: Scenario, *, record_trace: bool) -> dict[str, Any]:
    config = scenario.config
    device = scenario.truth.device
    system = Lorenz96(config)
    ensemble = scenario.initial_ensemble.clone()
    alpha = augmented_alpha_initial(config, device)
    generator = make_generator(device, config.seed * 100_000 + 401)
    alpha = (
        alpha
        + config.aug_alpha_jitter
        * (config.alpha_max - config.alpha_min)
        * torch.randn(alpha.shape, dtype=alpha.dtype, device=device, generator=generator)
    ).clamp(config.alpha_min, config.alpha_max)
    operator = SparseObservation(scenario.observation_indices)
    observation_covariance = covariance(scenario)
    weights = torch.full((config.ensemble_size,), 1.0 / config.ensemble_size, dtype=torch.float64, device=device)
    metrics = RunningMetrics()
    mean_history: list[torch.Tensor] = []
    alpha_history: list[float] = []
    started = start_runtime(device)
    for step in range(config.steps + 1):
        estimate = ensemble.mean(dim=0)
        if record_trace:
            mean_history.append(estimate.detach().cpu())
            alpha_history.append(float(alpha.mean().detach().cpu()))
        metrics.add(
            ensemble,
            scenario.truth[step],
            weights,
            point_estimate=estimate,
            probabilistic=should_score_probabilistic(step, config),
        )
        if step == config.steps:
            break
        alpha = (
            alpha
            + config.aug_alpha_random_walk_std
            * torch.randn(alpha.shape, dtype=alpha.dtype, device=device, generator=generator)
        ).clamp(config.alpha_min, config.alpha_max)
        ensemble = system.step(ensemble, alpha, scenario.forecast_noise[step])
        if step + 1 in scenario.observations:
            augmented = torch.cat((ensemble, alpha[:, None]), dim=-1)
            updated = denkf_analysis(
                augmented,
                scenario.observations[step + 1],
                operator,
                observation_covariance,
                localization=scenario.augmented_localization,
            )
            ensemble = system.project(updated[:, :-1])
            alpha = updated[:, -1].clamp(config.alpha_min, config.alpha_max)
    elapsed, memory = finalize_runtime(started, device)
    alpha_estimate = float(alpha.mean().detach().cpu())
    result = metrics.finalize()
    result.update(
        runtime_seconds=elapsed,
        peak_gpu_memory_mb=memory,
        forward_member_steps=config.steps * config.ensemble_size,
        alpha_estimate=alpha_estimate,
        alpha_map=float(alpha.median().detach().cpu()),
        alpha_absolute_error=abs(alpha_estimate - config.alpha_true),
        alpha_spread=float(alpha.std(unbiased=True).detach().cpu()),
        max_abs_state=float(torch.max(torch.abs(ensemble)).detach().cpu()),
    )
    if record_trace:
        result["mean_states"] = torch.stack(mean_history).numpy()
        result["alpha_mean_history"] = np.asarray(alpha_history, dtype=float)
    return result


def run_bma(scenario: Scenario, *, record_trace: bool) -> dict[str, Any]:
    config = scenario.config
    device = scenario.truth.device
    system = Lorenz96(config)
    alpha_grid = torch.linspace(
        config.alpha_min,
        config.alpha_max,
        config.bma_alpha_grid_size,
        dtype=torch.float64,
        device=device,
    )
    path_count = int(alpha_grid.numel())
    branches = scenario.initial_ensemble.unsqueeze(0).repeat(path_count, 1, 1)
    log_weights = torch.zeros(path_count, dtype=torch.float64, device=device)
    path_weights = torch.softmax(log_weights, dim=0)
    operator = SparseObservation(scenario.observation_indices)
    observation_covariance = covariance(scenario)
    metrics = RunningMetrics()
    mean_history: list[torch.Tensor] = []
    weight_history: list[torch.Tensor] = []
    started = start_runtime(device)
    for step in range(config.steps + 1):
        flat = branches.reshape(-1, config.state_dim)
        flat_weights = path_weights[:, None].expand(-1, config.ensemble_size).reshape(-1) / config.ensemble_size
        estimate = (path_weights[:, None] * branches.mean(dim=1)).sum(dim=0)
        if record_trace:
            mean_history.append(estimate.detach().cpu())
            weight_history.append(path_weights.detach().cpu())
        metrics.add(
            flat,
            scenario.truth[step],
            flat_weights,
            point_estimate=estimate,
            probabilistic=should_score_probabilistic(step, config),
        )
        if step == config.steps:
            break
        branches = system.step(
            branches,
            alpha_grid,
            scenario.forecast_noise[step].unsqueeze(0).expand(path_count, -1, -1),
        )
        if step + 1 not in scenario.observations:
            continue
        observation = scenario.observations[step + 1]
        predicted = operator(branches)
        evidence = torch.stack(
            [evidence_score(predicted[index], observation, config, None) for index in range(path_count)]
        )
        log_weights = log_weights + evidence - evidence.mean()
        path_weights = torch.softmax(log_weights, dim=0)
        for index in range(path_count):
            branches[index] = system.project(
                denkf_analysis(
                    branches[index],
                    observation,
                    operator,
                    observation_covariance,
                    localization=scenario.localization,
                )
            )
    elapsed, memory = finalize_runtime(started, device)
    alpha_estimate = float((alpha_grid * path_weights).sum().detach().cpu())
    result = metrics.finalize()
    result.update(
        runtime_seconds=elapsed,
        peak_gpu_memory_mb=memory,
        forward_member_steps=config.steps * path_count * config.ensemble_size,
        alpha_estimate=alpha_estimate,
        alpha_map=float(alpha_grid[int(torch.argmax(path_weights))].detach().cpu()),
        alpha_absolute_error=abs(alpha_estimate - config.alpha_true),
        alpha_final_entropy=float(entropy(path_weights).detach().cpu()),
        max_abs_state=float(torch.max(torch.abs(branches)).detach().cpu()),
    )
    if record_trace:
        result["mean_states"] = torch.stack(mean_history).numpy()
        result["alpha_weight_history"] = torch.stack(weight_history).numpy()
    return result


def state_weights(method: Literal["pce", "apce"], alpha_weights: torch.Tensor, config: L96ScalingConfig) -> torch.Tensor:
    if method == "apce":
        return entropy_project(alpha_weights, config.apce_entropy_floor)
    return alpha_weights / alpha_weights.sum().clamp_min(1.0e-300)


def pad_history(history: list[torch.Tensor]) -> np.ndarray:
    if not history:
        return np.empty((0, 0), dtype=float)
    width = max(int(item.numel()) for item in history)
    output = np.full((len(history), width), np.nan, dtype=float)
    for index, item in enumerate(history):
        values = item.detach().cpu().numpy().reshape(-1)
        output[index, : values.size] = values
    return output


def run_pce_apce(scenario: Scenario, method: Literal["pce", "apce"], *, record_trace: bool) -> dict[str, Any]:
    config = scenario.config
    device = scenario.truth.device
    system = Lorenz96(config)
    generator = make_generator(device, config.seed * 100_000 + (801 if method == "pce" else 803))
    alpha_grid = torch.tensor(config.coarse_alpha_grid, dtype=torch.float64, device=device)
    branches = scenario.initial_ensemble.unsqueeze(0).repeat(alpha_grid.numel(), 1, 1)
    shadow = branches.clone()
    alpha_members = torch.stack(
        [
            alpha_cloud(
                alpha,
                (config.alpha_min, config.alpha_max),
                config.ensemble_size,
                config.branch_member_alpha_jitter,
                generator,
            )
            for alpha in alpha_grid
        ],
        dim=0,
    )
    log_scores = torch.zeros(alpha_grid.numel(), dtype=torch.float64, device=device)
    alpha_weights = torch.softmax(log_scores, dim=0)
    output_weights = state_weights(method, alpha_weights, config)
    operator = SparseObservation(scenario.observation_indices)
    observation_covariance = covariance(scenario)
    metrics = RunningMetrics()
    mean_history: list[torch.Tensor] = []
    weight_history: list[torch.Tensor] = []
    grid_history: list[torch.Tensor] = []
    forward_member_steps = 0
    regrid_count = 0
    dynamic_grid_update_count = 0
    started = start_runtime(device)
    for step in range(config.steps + 1):
        path_count = int(alpha_grid.numel())
        flat = branches.reshape(-1, config.state_dim)
        flat_weights = output_weights[:, None].expand(-1, config.ensemble_size).reshape(-1) / config.ensemble_size
        estimate = (output_weights[:, None] * branches.mean(dim=1)).sum(dim=0)
        if record_trace:
            mean_history.append(estimate.detach().cpu())
            weight_history.append(output_weights.detach().cpu())
            grid_history.append(alpha_grid.detach().cpu())
        metrics.add(
            flat,
            scenario.truth[step],
            flat_weights,
            point_estimate=estimate,
            probabilistic=should_score_probabilistic(step, config),
        )
        if step == config.steps:
            break
        shared_noise = scenario.forecast_noise[step].unsqueeze(0).expand(path_count, -1, -1)
        branches = system.step(branches, alpha_members, shared_noise)
        shadow = system.step(shadow, alpha_members, shared_noise)
        forward_member_steps += 2 * path_count * config.ensemble_size
        if step + 1 not in scenario.observations:
            continue
        observation = scenario.observations[step + 1]
        shadow_predicted = operator(shadow)
        dimension_weights = None
        if method == "apce":
            between = shadow_predicted.mean(dim=1).var(dim=0, unbiased=True)
            dimension_weights = config.apce_dimension_floor + config.apce_dimension_gain * between / between.max().clamp_min(1.0e-12)
        evidence = torch.stack(
            [
                evidence_score(shadow_predicted[index], observation, config, dimension_weights)
                for index in range(path_count)
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
        output_weights = state_weights(method, alpha_weights, config)
        if method == "apce" and config.apce_recycle_entropy_projected_scores:
            log_scores = output_weights.clamp_min(1.0e-300).log()
            alpha_weights = torch.softmax(log_scores, dim=0)
        alpha_grid, branches, shadow, alpha_members, log_scores, dynamic_changed = dynamic_grid_coordinates(
            alpha_grid,
            branches,
            shadow,
            alpha_members,
            log_scores,
            config,
        )
        if dynamic_changed:
            dynamic_grid_update_count += 1
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
            output_weights = state_weights(method, alpha_weights, config)
        path_count = int(alpha_grid.numel())
        local_analysis = torch.empty_like(branches)
        for index in range(path_count):
            local_analysis[index] = denkf_analysis(
                branches[index],
                observation,
                operator,
                observation_covariance,
                localization=scenario.localization,
            )
        branches = local_analysis
        if config.global_state_analysis_strength > 1.0e-12:
            flat_branches = branches.reshape(-1, config.state_dim)
            flat_weights = output_weights[:, None].expand(-1, config.ensemble_size).reshape(-1) / config.ensemble_size
            global_analysis = weighted_denkf_analysis(
                flat_branches,
                flat_weights,
                observation,
                operator,
                observation_covariance,
                scenario.localization,
            ).reshape_as(branches)
            branches = (
                (1.0 - config.global_state_analysis_strength) * branches
                + config.global_state_analysis_strength * global_analysis
            )
        if config.branch_augmented_alpha_analysis_strength > 1.0e-12:
            joint_branches = torch.empty_like(branches)
            joint_alpha = torch.empty_like(alpha_members)
            for index in range(path_count):
                updated_state, updated_alpha = augmented_denkf_analysis(
                    branches[index],
                    alpha_members[index],
                    observation,
                    operator,
                    observation_covariance,
                    config,
                    scenario.augmented_localization,
                )
                joint_branches[index] = updated_state
                joint_alpha[index] = updated_alpha
            strength = config.branch_augmented_alpha_analysis_strength
            branches = (1.0 - strength) * branches + strength * joint_branches
            alpha_members = (1.0 - strength) * alpha_members + strength * joint_alpha
        if config.global_augmented_alpha_analysis_strength > 1.0e-12:
            flat_branches = branches.reshape(-1, config.state_dim)
            flat_alpha = alpha_members.reshape(-1)
            flat_weights = output_weights[:, None].expand(-1, config.ensemble_size).reshape(-1) / config.ensemble_size
            updated_state, updated_alpha = weighted_augmented_denkf_analysis(
                flat_branches,
                flat_alpha,
                flat_weights,
                observation,
                operator,
                observation_covariance,
                config,
                scenario.augmented_localization,
            )
            strength = config.global_augmented_alpha_analysis_strength
            branches = (1.0 - strength) * branches + strength * updated_state.reshape_as(branches)
            alpha_members = (1.0 - strength) * alpha_members + strength * updated_alpha.reshape_as(alpha_members)
        branches = system.project(branches)
        alpha_members = alpha_members.clamp(config.alpha_min, config.alpha_max)
    elapsed, memory = finalize_runtime(started, device)
    branch_alpha = alpha_members.mean(dim=1)
    alpha_estimate = float((output_weights * branch_alpha).sum().detach().cpu())
    result = metrics.finalize()
    result.update(
        runtime_seconds=elapsed,
        peak_gpu_memory_mb=memory,
        forward_member_steps=int(forward_member_steps),
        alpha_estimate=alpha_estimate,
        alpha_map=float(branch_alpha[int(torch.argmax(output_weights))].detach().cpu()),
        alpha_absolute_error=abs(alpha_estimate - config.alpha_true),
        alpha_final_entropy=float(entropy(output_weights).detach().cpu()),
        alpha_evidence_entropy=float(entropy(alpha_weights).detach().cpu()),
        alpha_regrid_count=int(regrid_count),
        dynamic_grid_update_count=int(dynamic_grid_update_count),
        final_grid_points=int(alpha_grid.numel()),
        max_abs_state=float(torch.max(torch.abs(branches)).detach().cpu()),
    )
    if record_trace:
        result["mean_states"] = torch.stack(mean_history).numpy()
        result["alpha_weight_history"] = pad_history(weight_history)
        result["alpha_grid_history"] = pad_history(grid_history)
    return result


def run_method(scenario: Scenario, method: MethodName, *, record_trace: bool) -> dict[str, Any]:
    if method == "aug_enkf":
        return run_aug_enkf(scenario, record_trace=record_trace)
    if method == "bma_static":
        return run_bma(scenario, record_trace=record_trace)
    if method in {"pce", "apce"}:
        return run_pce_apce(scenario, method, record_trace=record_trace)
    raise ValueError(method)


def numerical_status(result: dict[str, Any], scenario: Scenario) -> str:
    required = ("nrmse", "rmse", "crps", "coverage_90", "interval_width_90", "max_abs_state")
    if not all(math.isfinite(float(result[key])) for key in required):
        return "nonfinite"
    truth_scale = float(torch.max(torch.abs(scenario.truth)).detach().cpu())
    if truth_scale and float(result["max_abs_state"]) > scenario.config.max_valid_amplitude_ratio * truth_scale:
        return "diverged"
    return "valid"


def trace_path(output: Path, method: MethodName, interval: int, seed: int) -> Path:
    return output / "artifacts" / "method_traces" / "lorenz96_1024" / f"time{interval}" / method / f"seed_{seed}.npz"


def save_trace(output: Path, method: MethodName, scenario: Scenario, result: dict[str, Any]) -> str:
    path = trace_path(output, method, scenario.config.obs_interval, scenario.config.seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "times": np.arange(scenario.config.steps + 1, dtype=float) * scenario.config.dt,
        "observation_indices": scenario.observation_indices.detach().cpu().numpy(),
        "alpha_true": np.asarray(scenario.config.alpha_true),
        "obs_interval": np.asarray(scenario.config.obs_interval),
        "state_dim": np.asarray(scenario.config.state_dim),
    }
    for key in ("mean_states", "alpha_mean_history", "alpha_weight_history", "alpha_grid_history"):
        if key in result:
            payload[key] = result[key]
    np.savez_compressed(path, **payload)
    return str(path)


def run_json_path(output: Path, method: MethodName, interval: int, seed: int) -> Path:
    return output / "artifacts" / "run_json" / "lorenz96_1024" / f"time{interval}" / method / f"seed_{seed}.json"


def completed_payload(
    scenario: Scenario,
    method: MethodName,
    result: dict[str, Any],
    output: Path,
    record_trace: bool,
) -> dict[str, Any]:
    status = numerical_status(result, scenario)
    trace = save_trace(output, method, scenario, result) if record_trace else ""
    scalars = {key: value for key, value in result.items() if isinstance(value, (str, float, int))}
    return {
        "run_id": f"lorenz96_1024_t{scenario.config.obs_interval}_{method}_seed{scenario.config.seed}",
        "status": "completed",
        "numerical_status": status,
        "case": "lorenz96_1024",
        "method": method,
        "label": METHOD_LABELS[method],
        "seed": scenario.config.seed,
        "state_dim": scenario.config.state_dim,
        "observed_points": scenario.config.observed_points,
        "spatial_downsampling_factor": scenario.config.state_dim // scenario.config.observed_points,
        "observation_indices": ",".join(str(int(value)) for value in scenario.observation_indices.detach().cpu().tolist()),
        "obs_interval": scenario.config.obs_interval,
        "observation_count": len(scenario.observations),
        "dt": scenario.config.dt,
        "steps": scenario.config.steps,
        "ensemble_size": scenario.config.ensemble_size,
        "alpha_true": scenario.config.alpha_true,
        "coarse_alpha_grid": ",".join(f"{value:.4g}" for value in scenario.config.coarse_alpha_grid),
        "bma_alpha_grid_size": scenario.config.bma_alpha_grid_size,
        "pce_temperature": scenario.config.pce_temperature,
        "apce_temperature": scenario.config.apce_temperature,
        "apce_forgetting": scenario.config.apce_forgetting,
        "apce_entropy_floor": scenario.config.apce_entropy_floor,
        "apce_recycle_entropy_projected_scores": scenario.config.apce_recycle_entropy_projected_scores,
        "apce_dimension_floor": scenario.config.apce_dimension_floor,
        "apce_dimension_gain": scenario.config.apce_dimension_gain,
        "branch_member_alpha_jitter": scenario.config.branch_member_alpha_jitter,
        "branch_augmented_alpha_analysis_strength": scenario.config.branch_augmented_alpha_analysis_strength,
        "global_augmented_alpha_analysis_strength": scenario.config.global_augmented_alpha_analysis_strength,
        "global_state_analysis_strength": scenario.config.global_state_analysis_strength,
        "dynamic_regrid_from_alpha_members": scenario.config.dynamic_regrid_from_alpha_members,
        "localization_scale": scenario.config.localization_scale,
        "localization_compact_support": 2.0 * scenario.config.localization_scale,
        "probabilistic_metric_stride": scenario.config.probabilistic_metric_stride,
        "asset_npz": str(scenario.asset_path),
        "trace_npz": trace,
        **scalars,
    }


def parse_method(text: str) -> MethodName:
    if text not in METHODS:
        raise ValueError(f"--method must be one of {METHODS}")
    return text  # type: ignore[return-value]


def config_from_args(args: argparse.Namespace) -> L96ScalingConfig:
    if args.obs_interval < 1 or args.obs_interval > args.steps:
        raise ValueError("--obs-interval must lie in [1, --steps]")
    if args.observed_points < 1:
        raise ValueError("--observed-points must be positive")
    if args.observed_points > args.state_dim:
        raise ValueError("--observed-points cannot exceed state_dim")
    if args.state_dim % args.observed_points != 0:
        raise ValueError("--observed-points must evenly divide state_dim")
    return L96ScalingConfig(
        seed=args.seed,
        state_dim=args.state_dim,
        observed_points=args.observed_points,
        obs_interval=args.obs_interval,
        steps=args.steps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 1024-dimensional Lorenz-96 benchmark.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--state-dim", type=int, default=1024)
    parser.add_argument("--observed-points", type=int, default=128)
    parser.add_argument("--obs-interval", type=int, default=8)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--method", default="apce")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prepare-asset-only", action="store_true")
    parser.add_argument("--no-record-trace", action="store_true")
    args = parser.parse_args()
    config = config_from_args(args)
    device = torch.device(args.device)
    asset_root = args.asset_root or (args.output / "shared_assets")
    if args.prepare_asset_only:
        path = create_shared_assets(config, asset_root, device)
        print(json.dumps({"status": "asset_ready", "asset": str(path)}, ensure_ascii=False))
        return
    method = parse_method(args.method)
    run_path = run_json_path(args.output, method, config.obs_interval, config.seed)
    try:
        shared = load_shared_assets(config, asset_root, device)
        scenario = materialize_scenario(shared)
        result = run_method(scenario, method, record_trace=not args.no_record_trace)
        payload = completed_payload(scenario, method, result, args.output, not args.no_record_trace)
        write_json(run_path, payload)
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "numerical_status": payload["numerical_status"],
                    "method": method,
                    "seed": config.seed,
                    "obs_interval": config.obs_interval,
                    "nrmse": payload["nrmse"],
                    "crps": payload["crps"],
                    "alpha_mae": payload["alpha_absolute_error"],
                },
                ensure_ascii=False,
            )
        )
    except Exception as error:  # noqa: BLE001
        payload = {
            "run_id": f"lorenz96_1024_t{config.obs_interval}_{method}_seed{config.seed}",
            "status": "failed",
            "case": "lorenz96_1024",
            "method": method,
            "seed": config.seed,
            "obs_interval": config.obs_interval,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(run_path, payload)
        raise


if __name__ == "__main__":
    main()
