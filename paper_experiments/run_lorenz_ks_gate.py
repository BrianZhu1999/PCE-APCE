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

from hilda_da.alpha import liu_quantile
from hilda_da.baselines import denkf_analysis, letkf_analysis
from hilda_da.metrics import weighted_central_interval_coverage_width, weighted_ensemble_crps
from hilda_da.observations import SparseObservation


CaseName = Literal["lorenz96", "ks"]
MethodName = Literal[
    "misspecified_forecast",
    "denkf",
    "letkf",
    "oracle_alpha",
    "pce",
    "apce",
]

METHODS: tuple[MethodName, ...] = (
    "misspecified_forecast",
    "denkf",
    "letkf",
    "oracle_alpha",
    "pce",
    "apce",
)

METHOD_LABELS = {
    "misspecified_forecast": "Misspecified forecast",
    "denkf": "DEnKF",
    "letkf": "LETKF",
    "oracle_alpha": "Oracle-alpha",
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
    obs_stride: int
    ensemble_size: int
    obs_noise: float
    state_dim: int
    alpha_true: float = 0.12
    fixed_alpha: float = 0.50
    alpha_grid: tuple[float, ...] = (0.08, 0.22, 0.36, 0.50, 0.64, 0.78, 0.92)
    pce_temperature: float = 0.42
    apce_temperature: float = 0.52
    apce_min_temperature: float = 0.14
    apce_forgetting: float = 0.985
    apce_entropy_floor: float = 0.72
    evidence_shrinkage: float = 0.20


@dataclass
class Scenario:
    config: CaseConfig
    times: torch.Tensor
    coordinates: torch.Tensor
    truth: torch.Tensor
    observations: dict[int, torch.Tensor]
    observation_indices: torch.Tensor
    initial_ensemble: torch.Tensor
    forecast_noise: torch.Tensor
    alpha_grid: torch.Tensor


class TrajectoryMetrics:
    def __init__(self) -> None:
        self.squared_error = 0.0
        self.truth_square = 0.0
        self.crps: list[float] = []
        self.coverage: list[float] = []
        self.width: list[float] = []
        self.points = 0

    def add(self, ensemble: torch.Tensor, truth: torch.Tensor, weights: torch.Tensor) -> None:
        weights = weights / weights.sum()
        estimate = (weights.unsqueeze(-1) * ensemble).sum(dim=0)
        self.squared_error += float((estimate - truth).square().sum())
        self.truth_square += float(truth.square().sum())
        self.points += int(truth.numel())
        self.crps.append(float(weighted_ensemble_crps(ensemble, truth, weights)))
        coverage, width = weighted_central_interval_coverage_width(
            ensemble, truth, weights, level=0.90
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


def torch_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(seed)
    return generator


def smooth_periodic_noise(noise: torch.Tensor, passes: int = 2) -> torch.Tensor:
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


@dataclass(frozen=True)
class Lorenz96Config:
    dim: int = 40
    forcing_base: float = 8.0
    epistemic_scale: float = 1.55
    stochastic_scale: float = 0.035


class Lorenz96System:
    def __init__(self, config: Lorenz96Config | None = None) -> None:
        self.config = config or Lorenz96Config()
        self.state_dim = self.config.dim
        self.coordinates = torch.arange(self.config.dim, dtype=torch.float64)

    def drift(self, state: torch.Tensor, alpha_quantile: torch.Tensor) -> torch.Tensor:
        forcing = self.config.forcing_base + self.config.epistemic_scale * alpha_quantile
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


@dataclass(frozen=True)
class KSConfig:
    nx: int = 64
    length: float = 22.0
    stochastic_scale: float = 0.020
    epistemic_scale: float = 0.32
    damping_base: float = 1.0


class KuramotoSivashinskySystem:
    def __init__(self, config: KSConfig | None = None, *, dtype: torch.dtype, device: torch.device, dt: float) -> None:
        self.config = config or KSConfig()
        self.state_dim = self.config.nx
        self.dtype = dtype
        self.device = device
        self.dt = dt
        self.coordinates = torch.linspace(
            0.0,
            self.config.length,
            self.config.nx + 1,
            dtype=dtype,
            device=device,
        )[:-1]
        self.wave = 2.0 * math.pi * torch.fft.fftfreq(
            self.config.nx,
            d=self.config.length / self.config.nx,
            dtype=dtype,
            device=device,
        )
        self.linear = self.wave.square() - self.wave.pow(4)
        self._prepare_etdrk4()
        self.forcing_profile = torch.sin(2.0 * math.pi * self.coordinates / self.config.length)

    def _prepare_etdrk4(self) -> None:
        dt = self.dt
        linear = self.linear
        self.E = torch.exp(dt * linear)
        self.E2 = torch.exp(0.5 * dt * linear)
        roots = torch.exp(
            1j
            * math.pi
            * (torch.arange(1, 17, dtype=self.dtype, device=self.device) - 0.5)
            / 16.0
        )
        lr = dt * linear.to(torch.complex128).unsqueeze(1) + roots.to(torch.complex128).unsqueeze(0)
        self.Q = (dt * torch.mean((torch.exp(lr / 2.0) - 1.0) / lr, dim=1)).real.to(self.dtype)
        self.f1 = (
            dt
            * torch.mean(
                (-4.0 - lr + torch.exp(lr) * (4.0 - 3.0 * lr + lr.square())) / lr.pow(3),
                dim=1,
            )
        ).real.to(self.dtype)
        self.f2 = (
            dt
            * torch.mean((2.0 + lr + torch.exp(lr) * (-2.0 + lr)) / lr.pow(3), dim=1)
        ).real.to(self.dtype)
        self.f3 = (
            dt
            * torch.mean(
                (-4.0 - 3.0 * lr - lr.square() + torch.exp(lr) * (4.0 - lr)) / lr.pow(3),
                dim=1,
            )
        ).real.to(self.dtype)

    def nonlinear_spectrum(self, spectrum: torch.Tensor, alpha_quantile: torch.Tensor) -> torch.Tensor:
        field = torch.fft.ifft(spectrum, dim=-1).real
        nonlinear = -0.5j * self.wave * torch.fft.fft(field.square(), dim=-1)
        forcing = (
            self.config.epistemic_scale
            * alpha_quantile
            * torch.fft.fft(self.forcing_profile.expand_as(field), dim=-1)
        )
        return nonlinear + forcing

    def step_deterministic(self, state: torch.Tensor, alpha_quantile: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.fft(state, dim=-1)
        nv = self.nonlinear_spectrum(spectrum, alpha_quantile)
        a = self.E2 * spectrum + self.Q * nv
        na = self.nonlinear_spectrum(a, alpha_quantile)
        b = self.E2 * spectrum + self.Q * na
        nb = self.nonlinear_spectrum(b, alpha_quantile)
        c = self.E2 * a + self.Q * (2.0 * nb - nv)
        nc = self.nonlinear_spectrum(c, alpha_quantile)
        next_spectrum = self.E * spectrum + self.f1 * nv + 2.0 * self.f2 * (na + nb) + self.f3 * nc
        return self.project(torch.fft.ifft(next_spectrum, dim=-1).real)

    def diffusion(self, state: torch.Tensor) -> torch.Tensor:
        return torch.full_like(state, self.config.stochastic_scale)

    def project(self, state: torch.Tensor) -> torch.Tensor:
        state = state - state.mean(dim=-1, keepdim=True)
        return torch.nan_to_num(state, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)


def config_for_case(name: CaseName, seed: int) -> CaseConfig:
    if name == "lorenz96":
        return CaseConfig(
            name=name,
            seed=seed,
            steps=300,
            dt=0.010,
            obs_interval=5,
            obs_stride=2,  # Figure 4 freeze: 20 observed states, indices 0,2,...,38
            ensemble_size=32,
            obs_noise=0.45,
            state_dim=40,
            pce_temperature=0.36,
            apce_temperature=0.50,
            evidence_shrinkage=0.22,
            apce_entropy_floor=0.70,
        )
    if name == "ks":
        return CaseConfig(
            name=name,
            seed=seed,
            steps=260,
            dt=0.050,
            obs_interval=5,
            obs_stride=0,
            ensemble_size=28,
            obs_noise=0.23,
            state_dim=64,
            pce_temperature=0.42,
            apce_temperature=0.55,
            evidence_shrinkage=0.18,
            apce_entropy_floor=0.74,
        )
    raise ValueError(name)


def make_system(config: CaseConfig, device: torch.device, dtype: torch.dtype) -> Lorenz96System | KuramotoSivashinskySystem:
    if config.name == "lorenz96":
        return Lorenz96System(Lorenz96Config(dim=config.state_dim))
    if config.name == "ks":
        return KuramotoSivashinskySystem(KSConfig(nx=config.state_dim), dtype=dtype, device=device, dt=config.dt)
    raise ValueError(config.name)


def rk4_step(system: Lorenz96System, state: torch.Tensor, dt: float, alpha_quantile: torch.Tensor) -> torch.Tensor:
    k1 = system.drift(state, alpha_quantile)
    k2 = system.drift(state + 0.5 * dt * k1, alpha_quantile)
    k3 = system.drift(state + 0.5 * dt * k2, alpha_quantile)
    k4 = system.drift(state + dt * k3, alpha_quantile)
    return system.project(state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0)


def step_with_noise(
    system: Lorenz96System | KuramotoSivashinskySystem,
    state: torch.Tensor,
    dt: float,
    alpha: float,
    noise: torch.Tensor,
) -> torch.Tensor:
    quantile = liu_quantile(torch.tensor(alpha, dtype=state.dtype, device=state.device))
    if isinstance(system, Lorenz96System):
        deterministic = rk4_step(system, state, dt, quantile)
    else:
        deterministic = system.step_deterministic(state, quantile)
    return system.project(deterministic + math.sqrt(dt) * system.diffusion(state) * noise)


def spinup_initial(system: Lorenz96System | KuramotoSivashinskySystem, config: CaseConfig, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    dtype = torch.float64
    if config.name == "lorenz96":
        state = 8.0 + 0.25 * torch.randn(config.state_dim, dtype=dtype, device=device, generator=generator)
        spin_steps = 900
    else:
        assert isinstance(system, KuramotoSivashinskySystem)
        x = system.coordinates
        state = (
            0.65 * torch.cos(2.0 * math.pi * x / system.config.length)
            + 0.35 * torch.sin(4.0 * math.pi * x / system.config.length)
            + 0.08 * smooth_periodic_noise(torch.randn(config.state_dim, dtype=dtype, device=device, generator=generator), passes=4)
        )
        state = system.project(state)
        spin_steps = 260
    zero = torch.zeros_like(state)
    for _ in range(spin_steps):
        state = step_with_noise(system, state, config.dt, config.alpha_true, zero)
    return state.detach()


def observation_indices(config: CaseConfig, device: torch.device) -> torch.Tensor:
    if config.name == "lorenz96":
        stride = max(int(config.obs_stride), 1)
        return torch.arange(0, config.state_dim, stride, dtype=torch.int64, device=device)
    return torch.linspace(0, config.state_dim - 1, 12, dtype=torch.float64, device=device).round().to(torch.int64).unique()


def generate_scenario(config: CaseConfig, device: torch.device) -> Scenario:
    dtype = torch.float64
    system = make_system(config, device, dtype)
    generator = torch_generator(device, config.seed)
    state0 = spinup_initial(system, config, generator, device)
    obs_idx = observation_indices(config, device)
    truth_noise = torch.randn((config.steps, config.state_dim), dtype=dtype, device=device, generator=generator)
    forecast_noise = torch.randn(
        (config.steps, config.ensemble_size, config.state_dim),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    initial_noise = torch.randn(
        (config.ensemble_size, config.state_dim),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    obs_noise = torch.randn(
        (config.steps // config.obs_interval + 1, obs_idx.numel()),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    if config.name == "ks":
        truth_noise = smooth_periodic_noise(truth_noise, passes=4)
        forecast_noise = smooth_periodic_noise(forecast_noise, passes=4)
        initial_noise = smooth_periodic_noise(initial_noise, passes=4)
    else:
        truth_noise = smooth_periodic_noise(truth_noise, passes=1)
        forecast_noise = smooth_periodic_noise(forecast_noise, passes=1)
    initial_scale = 0.42 if config.name == "lorenz96" else 0.20
    initial_ensemble = system.project(state0.unsqueeze(0) + initial_scale * initial_noise)
    truth = torch.empty((config.steps + 1, config.state_dim), dtype=dtype, device=device)
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
    coordinates = (
        system.coordinates.to(dtype=dtype, device=device)
        if hasattr(system, "coordinates")
        else torch.arange(config.state_dim, dtype=dtype, device=device)
    )
    return Scenario(
        config=config,
        times=torch.arange(config.steps + 1, dtype=dtype, device=device) * config.dt,
        coordinates=coordinates,
        truth=truth,
        observations=observations,
        observation_indices=obs_idx,
        initial_ensemble=initial_ensemble,
        forecast_noise=forecast_noise,
        alpha_grid=torch.tensor(config.alpha_grid, dtype=dtype, device=device),
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
        return weights / weights.sum()
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


def run_fixed_method(
    scenario: Scenario,
    method: MethodName,
    device: torch.device,
    *,
    record_trace: bool = False,
) -> dict[str, Any]:
    config = scenario.config
    system = make_system(config, device, torch.float64)
    ensemble = scenario.initial_ensemble.clone()
    operator = SparseObservation(scenario.observation_indices)
    covariance = config.obs_noise**2 * torch.eye(
        scenario.observation_indices.numel(),
        dtype=ensemble.dtype,
        device=device,
    )
    weights = torch.full((config.ensemble_size,), 1.0 / config.ensemble_size, dtype=ensemble.dtype, device=device)
    metrics = TrajectoryMetrics()
    alpha = config.alpha_true if method == "oracle_alpha" else config.fixed_alpha
    trace_mean_states: list[torch.Tensor] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(config.steps + 1):
        if record_trace:
            trace_mean_states.append(ensemble.mean(dim=0).detach().cpu())
        metrics.add(ensemble, scenario.truth[step], weights)
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
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = metrics.finalize()
    result.update(
        runtime_seconds=float(time.perf_counter() - started),
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
    system = make_system(config, device, torch.float64)
    path_count = scenario.alpha_grid.numel()
    branches = scenario.initial_ensemble.unsqueeze(0).repeat(path_count, 1, 1)
    shadow = branches.clone()
    log_weights = torch.zeros(path_count, dtype=branches.dtype, device=device)
    weights = torch.softmax(log_weights, dim=0)
    operator = SparseObservation(scenario.observation_indices)
    covariance = config.obs_noise**2 * torch.eye(
        scenario.observation_indices.numel(),
        dtype=branches.dtype,
        device=device,
    )
    metrics = TrajectoryMetrics()
    trace_mean_states: list[torch.Tensor] = []
    trace_alpha_weights: list[torch.Tensor] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(config.steps + 1):
        flat = branches.reshape(-1, branches.shape[-1])
        flat_weights = weights.unsqueeze(1).expand(-1, config.ensemble_size).reshape(-1) / config.ensemble_size
        if record_trace:
            branch_means = branches.mean(dim=1)
            trace_mean_states.append((weights.unsqueeze(-1) * branch_means).sum(dim=0).detach().cpu())
            trace_alpha_weights.append(weights.detach().cpu())
        metrics.add(flat, scenario.truth[step], flat_weights)
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
        elif method == "apce":
            entropy_ratio = float(entropy(weights) / math.log(path_count))
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
            target_entropy = config.apce_entropy_floor + 0.18 * (1.0 - progress)
            weights = entropy_project(weights, target_entropy)
            log_weights = weights.clamp_min(1.0e-300).log()
        for path_index in range(path_count):
            branches[path_index] = denkf_analysis(branches[path_index], observation, operator, covariance)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    alpha_estimate = float((scenario.alpha_grid * weights).sum())
    result = metrics.finalize()
    result.update(
        runtime_seconds=float(time.perf_counter() - started),
        forward_member_steps=2 * config.steps * path_count * config.ensemble_size,
        peak_gpu_memory_mb=(
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda" else 0.0
        ),
        alpha_estimate=alpha_estimate,
        alpha_absolute_error=abs(alpha_estimate - config.alpha_true),
        alpha_final_entropy=float(entropy(weights)),
    )
    if record_trace:
        result["mean_states"] = torch.stack(trace_mean_states).numpy()
        result["alpha_weight_history"] = torch.stack(trace_alpha_weights).numpy()
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
    for case in ("lorenz96", "ks"):
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


def gate_decisions(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    baselines = ("misspecified_forecast", "denkf", "letkf")
    for case in ("lorenz96", "ks"):
        case_rows = {row["method"]: row for row in summary if row["case"] == case}
        for method in ("pce", "apce"):
            nrmse_wins = all(case_rows[method]["nrmse"] < case_rows[base]["nrmse"] for base in baselines)
            crps_wins = all(case_rows[method]["crps"] < case_rows[base]["crps"] for base in baselines)
            decisions.append(
                {
                    "case": case,
                    "method": method,
                    "wins_all_fixed_baselines_on_nrmse": bool(nrmse_wins),
                    "wins_all_fixed_baselines_on_crps": bool(crps_wins),
                    "quick_gate_pass": bool(nrmse_wins and crps_wins),
                    "nrmse_excess_over_oracle": float(case_rows[method]["nrmse"] - case_rows["oracle_alpha"]["nrmse"]),
                }
            )
    return decisions


def export_representative(output: Path, base_seed: int, device: torch.device, l96_obs_stride: int | None = None) -> None:
    source_dir = output / "representative_source"
    source_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {}
    for case in ("lorenz96", "ks"):
        config = config_for_case(case, base_seed)
        if case == "lorenz96" and l96_obs_stride is not None:
            config = CaseConfig(**{**asdict(config), "obs_stride": int(l96_obs_stride)})
        scenario = generate_scenario(config, device)
        arrays: dict[str, np.ndarray] = {
            "times": scenario.times.detach().cpu().numpy(),
            "coordinates": scenario.coordinates.detach().cpu().numpy(),
            "observation_indices": scenario.observation_indices.detach().cpu().numpy(),
            "truth_states": scenario.truth.detach().cpu().numpy(),
        }
        for method in ("misspecified_forecast", "denkf", "letkf", "pce", "apce"):
            result = run_method(scenario, method, device, record_trace=True)
            arrays[f"{method}_mean_states"] = np.asarray(result["mean_states"])
            if "alpha_weight_history" in result:
                arrays[f"{method}_alpha_weight_history"] = np.asarray(result["alpha_weight_history"])
        path = source_dir / f"{case}_representative_seed_{base_seed}.npz"
        np.savez_compressed(path, **arrays)
        manifest[case] = {
            "source": str(path.name),
            "seed": base_seed,
            "state_dim": config.state_dim,
            "steps": config.steps,
            "dt": config.dt,
            "obs_interval": config.obs_interval,
        }
    (source_dir / "representative_source_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_report(output: Path, summary: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> None:
    lines = [
        "# Lorenz-96 / Kuramoto--Sivashinsky APCE-PCE quick-gate report",
        "",
        "This is a five-paired-seed pressure test for uncertain-equation assimilation. It is not a final NCS statistical claim.",
        "",
        "| Case | Method | nRMSE | RMSE | CRPS | 90% coverage | Width | alpha error | Runtime (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {case} | {label} | {nrmse:.4%} | {rmse:.6e} | {crps:.6e} | {coverage_90:.2%} | {interval_width_90:.6e} | {alpha_absolute_error:.5f} | {runtime_seconds:.3f} |".format(**row)
        )
    lines.extend(["", "## Quick-gate decisions", ""])
    for decision in decisions:
        status = "PASS" if decision["quick_gate_pass"] else "FAIL"
        lines.append(
            f"- {decision['case']} / {decision['method'].upper()}: {status}; "
            f"nRMSE wins fixed baselines = {decision['wins_all_fixed_baselines_on_nrmse']}; "
            f"CRPS wins fixed baselines = {decision['wins_all_fixed_baselines_on_crps']}; "
            f"nRMSE excess over Oracle-alpha = {decision['nrmse_excess_over_oracle']:.6e}."
        )
    (output / "LORENZ96_KS_APCE_PCE_GATE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_suite(n_seeds: int, base_seed: int, output: Path, device: torch.device, l96_obs_stride: int | None = None) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    total = 2 * n_seeds * len(METHODS)
    completed = 0
    for case in ("lorenz96", "ks"):
        for seed_index in range(n_seeds):
            config = config_for_case(case, base_seed + seed_index)
            if case == "lorenz96" and l96_obs_stride is not None:
                config = CaseConfig(**{**asdict(config), "obs_stride": int(l96_obs_stride)})
            scenario = generate_scenario(config, device)
            for method in METHODS:
                result = run_method(scenario, method, device)
                row = {
                    "case": case,
                    "seed": config.seed,
                    "method": method,
                    "label": METHOD_LABELS[method],
                    "state_dim": int(scenario.truth.shape[-1]),
                    **result,
                }
                records.append(row)
                completed += 1
                print(
                    f"[{completed}/{total}] case={case} seed={config.seed} "
                    f"method={method} nrmse={row['nrmse']:.4%} crps={row['crps']:.4e}",
                    flush=True,
                )
    fieldnames = sorted({key for row in records for key in row})
    with (output / "run_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    summary = summarize(records, base_seed)
    decisions = gate_decisions(summary)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in summary for key in row}))
        writer.writeheader()
        writer.writerows(summary)
    payload = {
        "n_seeds": n_seeds,
        "base_seed": base_seed,
        "device": str(device),
        "methods": list(METHODS),
        "cases": ["lorenz96", "ks"],
        "case_configs": {
            case: asdict(
                CaseConfig(**{**asdict(config_for_case(case, base_seed)), "obs_stride": int(l96_obs_stride)})
                if case == "lorenz96" and l96_obs_stride is not None
                else config_for_case(case, base_seed)
            )
            for case in ("lorenz96", "ks")
        },
        "summary": summary,
        "decisions": decisions,
    }
    (output / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output, summary, decisions)
    export_representative(output, base_seed, device, l96_obs_stride=l96_obs_stride)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="5-seed Lorenz-96/KS APCE-PCE quick gate.")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=2026080600)
    parser.add_argument("--output", type=Path, default=Path("results_lorenz_ks_gate_5seeds"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--l96-obs-stride", type=int, default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    result = run_suite(args.n_seeds, args.base_seed, args.output, device, l96_obs_stride=args.l96_obs_stride)
    print(json.dumps({"decisions": result["decisions"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
