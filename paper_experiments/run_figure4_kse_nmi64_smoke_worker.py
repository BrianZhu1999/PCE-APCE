from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hilda_da.alpha_refinement import (
    apce_calibration_parameters,
    evidence_gap_confidence,
    torch_local_alpha_grid,
    torch_refined_alpha_map,
    torch_regrid_paths,
)
from hilda_da.baselines import denkf_analysis, letkf_analysis
from hilda_da.metrics import weighted_central_interval_coverage_width, weighted_ensemble_crps
from hilda_da.math_utils import stable_cholesky
from hilda_da.observations import SparseObservation, SpectralObservation
from hilda_da.strong_baselines import IEnSFConfig, iensf_analysis


MethodName = Literal["denkf", "letkf", "iensf", "aug_enkf", "bma_static", "pce", "apce"]

METHODS: tuple[MethodName, ...] = ("denkf", "letkf", "iensf", "aug_enkf", "bma_static", "pce", "apce")
METHOD_LABELS = {
    "denkf": "DEnKF",
    "letkf": "LETKF",
    "iensf": "IEnSF",
    "aug_enkf": "Aug-EnKF",
    "bma_static": "BMA",
    "pce": "PCE",
    "apce": "APCE",
}

DEFAULT_INPUT = Path("<HILDA_RESULTS_ROOT>/external/S3GM_NMI_2024/KSE_test.npy")
DEFAULT_OUTPUT = Path("<HILDA_RESULTS_ROOT>/results/figure4_kse_nmi64_fullmethods_smoke_5seeds_20260814")


@dataclass(frozen=True)
class KSE64Config:
    seed: int
    sample_index: int
    steps: int = 99
    saved_dt: float = 0.5
    obs_interval: int = 1
    obs_geometry: str = "physical"
    solver_substeps: int = 5
    length: float = 32.0 * math.pi
    state_dim: int = 1024
    observed_points: int = 16
    spectral_mode_count: int = 0
    ensemble_size: int = 20
    obs_noise: float = 0.035
    init_noise: float = 0.18
    process_noise: float = 0.006
    fixed_mu: float = 2.5
    mu_lower: float = 1.0
    mu_upper: float = 5.0
    coarse_mu_grid: tuple[float, ...] = (1.0, 1.6666666667, 2.3333333333, 3.0, 3.6666666667, 4.3333333333, 5.0)
    bma_mu_grid: tuple[float, ...] = tuple(float(x) for x in np.linspace(1.0, 5.0, 21))
    pce_temperature: float = 0.62
    apce_temperature: float = 0.50
    apce_min_temperature: float = 0.16
    apce_forgetting: float = 0.975
    apce_entropy_floor: float = 0.28
    evidence_shrinkage: float = 0.18
    local_grid_points: int = 11
    local_grid_topk: int = 3
    local_grid_min_spacing: float = 0.015
    state_weight_power: float = 4.0
    state_map_blend: float = 0.65
    state_map_blend_confidence_power: float = 0.70
    state_point_estimator_mode: str = "interpolated_map"
    branch_member_alpha_jitter: float = 0.020
    branch_member_alpha_jitter_confidence_power: float = 1.0
    branch_augmented_alpha_analysis_strength: float = 1.0
    global_augmented_alpha_analysis_strength: float = 0.30
    global_analysis_strength: float = 0.70
    global_analysis_confidence_power: float = 0.70
    analysis_iterations: int = 1
    dimension_weight_floor: float = 0.35
    dimension_weight_gain: float = 0.65


@dataclass
class KSE64Scenario:
    config: KSE64Config
    truth: torch.Tensor
    observations: dict[int, torch.Tensor]
    observation_indices: torch.Tensor
    observation_geometry: str
    observation_dim: int
    observation_scale: float
    initial_ensemble: torch.Tensor
    forecast_noise: torch.Tensor
    initial_mu_ensemble: torch.Tensor
    true_mu: float
    truth_std: float


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
        ensemble: torch.Tensor,
        truth: torch.Tensor,
        weights: torch.Tensor,
        *,
        point_estimate: torch.Tensor | None = None,
    ) -> None:
        weights = weights.to(dtype=ensemble.dtype, device=ensemble.device).clamp_min(1.0e-300)
        weights = weights / weights.sum().clamp_min(1.0e-300)
        estimate = point_estimate if point_estimate is not None else (weights.unsqueeze(-1) * ensemble).sum(dim=0)
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


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def smooth_periodic_noise(noise: torch.Tensor, passes: int = 2) -> torch.Tensor:
    output = noise
    for _ in range(passes):
        output = (
            0.18 * torch.roll(output, 3, dims=-1)
            + 0.20 * torch.roll(output, 2, dims=-1)
            + 0.22 * torch.roll(output, 1, dims=-1)
            + 0.20 * output
            + 0.22 * torch.roll(output, -1, dims=-1)
            + 0.20 * torch.roll(output, -2, dims=-1)
            + 0.18 * torch.roll(output, -3, dims=-1)
        )
    return output / output.std(dim=-1, keepdim=True).clamp_min(1.0e-12)


def periodic_spectral_upsample(samples: torch.Tensor, target_n: int) -> torch.Tensor:
    """Zero-pad Fourier modes for uniformly spaced periodic sensor samples."""
    if samples.ndim == 1:
        samples = samples.unsqueeze(0)
    observed_n = int(samples.shape[-1])
    spec = torch.fft.rfft(samples, dim=-1)
    target_spec = torch.zeros(
        (*samples.shape[:-1], target_n // 2 + 1),
        dtype=spec.dtype,
        device=samples.device,
    )
    copy_n = min(spec.shape[-1], target_spec.shape[-1])
    target_spec[..., :copy_n] = spec[..., :copy_n]
    recon = torch.fft.irfft(target_spec, n=target_n, dim=-1) * (target_n / observed_n)
    return recon.squeeze(0) if recon.shape[0] == 1 else recon


def spectral_real_inverse(real_modes: torch.Tensor, observed_n: int) -> torch.Tensor:
    """Invert a real-part-only Fourier observation back to coarse periodic samples."""
    if real_modes.ndim == 1:
        real_modes = real_modes.unsqueeze(0)
    mode_count = observed_n // 2 + 1
    if real_modes.shape[-1] > mode_count:
        real_modes = real_modes[..., :mode_count]
    real = torch.zeros((*real_modes.shape[:-1], mode_count), dtype=torch.float64, device=real_modes.device)
    real[..., : real_modes.shape[-1]] = real_modes.to(dtype=torch.float64, device=real_modes.device)
    spectrum = torch.complex(real, torch.zeros_like(real))
    recon = torch.fft.irfft(spectrum, n=observed_n, dim=-1)
    return recon.squeeze(0) if recon.shape[0] == 1 else recon


class KSEETDRK4:
    def __init__(self, *, state_dim: int, length: float, sub_dt: float, device: torch.device, dtype: torch.dtype) -> None:
        self.state_dim = state_dim
        self.length = length
        self.sub_dt = sub_dt
        self.device = device
        self.dtype = dtype
        self.wave = 2.0 * math.pi * torch.fft.fftfreq(
            state_dim,
            d=length / state_dim,
            dtype=dtype,
            device=device,
        )
        self.wave2 = self.wave.square()
        self.wave4 = self.wave2.square()
        self.roots = torch.exp(
            1j
            * math.pi
            * (torch.arange(1, 17, dtype=dtype, device=device) - 0.5)
            / 16.0
        ).to(torch.complex128)

    def _coefficients(self, mu: torch.Tensor) -> tuple[torch.Tensor, ...]:
        mu = mu.to(dtype=self.dtype, device=self.device).reshape(-1, 1)
        linear = self.wave2.unsqueeze(0) - mu * self.wave4.unsqueeze(0)
        dt = self.sub_dt
        E = torch.exp(dt * linear)
        E2 = torch.exp(0.5 * dt * linear)
        lr = dt * linear.to(torch.complex128).unsqueeze(-1) + self.roots.reshape(1, 1, -1)
        Q = (dt * torch.mean((torch.exp(lr / 2.0) - 1.0) / lr, dim=-1)).real.to(self.dtype)
        f1 = (
            dt
            * torch.mean((-4.0 - lr + torch.exp(lr) * (4.0 - 3.0 * lr + lr.square())) / lr.pow(3), dim=-1)
        ).real.to(self.dtype)
        f2 = (dt * torch.mean((2.0 + lr + torch.exp(lr) * (-2.0 + lr)) / lr.pow(3), dim=-1)).real.to(self.dtype)
        f3 = (
            dt
            * torch.mean((-4.0 - 3.0 * lr - lr.square() + torch.exp(lr) * (4.0 - lr)) / lr.pow(3), dim=-1)
        ).real.to(self.dtype)
        return E, E2, Q, f1, f2, f3

    def _nonlinear(self, spectrum: torch.Tensor) -> torch.Tensor:
        field = torch.fft.ifft(spectrum, dim=-1).real
        return -0.5j * self.wave.unsqueeze(0) * torch.fft.fft(field.square(), dim=-1)

    def step_saved(self, state: torch.Tensor, mu: torch.Tensor, *, substeps: int, noise: torch.Tensor | None, noise_scale: float) -> torch.Tensor:
        squeezed = False
        if state.ndim == 1:
            state = state.unsqueeze(0)
            squeezed = True
        mu = torch.as_tensor(mu, dtype=self.dtype, device=self.device)
        if mu.ndim == 0:
            mu = mu.expand(state.shape[0])
        mu = mu.reshape(-1)
        state = state.to(dtype=self.dtype, device=self.device)
        for _ in range(substeps):
            E, E2, Q, f1, f2, f3 = self._coefficients(mu)
            v = torch.fft.fft(state, dim=-1)
            Nv = self._nonlinear(v)
            a = E2 * v + Q.to(torch.complex128) * Nv
            Na = self._nonlinear(a)
            b = E2 * v + Q.to(torch.complex128) * Na
            Nb = self._nonlinear(b)
            c = E2 * a + Q.to(torch.complex128) * (2.0 * Nb - Nv)
            Nc = self._nonlinear(c)
            v = E * v + f1.to(torch.complex128) * Nv + 2.0 * f2.to(torch.complex128) * (Na + Nb) + f3.to(torch.complex128) * Nc
            state = torch.fft.ifft(v, dim=-1).real
        if noise is not None and noise_scale > 0.0:
            state = state + noise_scale * noise
        state = torch.nan_to_num(state, nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0)
        return state.squeeze(0) if squeezed else state


def sample_true_mu(sample_index: int) -> float:
    # Official KSE_test.npy contains three unseen ICs for each of the main-text
    # visual/test parameters 1.1, 2.5 and 3.2, in repeating order.  This mapping
    # was verified by one-step ETDRK4 prediction against the official file.
    return (1.1, 2.5, 3.2)[sample_index % 3]


def load_truth(input_path: Path, sample_index: int, steps: int, device: torch.device) -> torch.Tensor:
    raw = np.load(input_path, mmap_mode="r")
    if raw.shape != (9, 106, 1024, 1):
        raise ValueError(f"Unexpected KSE_test.npy shape: {raw.shape}")
    if sample_index < 0 or sample_index >= raw.shape[0]:
        raise ValueError(f"sample_index must be in [0,{raw.shape[0]-1}]")
    truth = np.asarray(raw[sample_index, : steps + 1, :, 0], dtype=np.float64).copy()
    return torch.as_tensor(truth, dtype=torch.float64, device=device)


def spectral_mode_count(config: KSE64Config) -> int:
    if int(config.spectral_mode_count) > 0:
        return min(int(config.spectral_mode_count), int(config.observed_points) // 2 + 1)
    return int(config.observed_points) // 2 + 1


def make_observation_operator(config: KSE64Config, observation_indices: torch.Tensor) -> SparseObservation | SpectralObservation:
    geometry = str(config.obs_geometry).lower()
    if geometry == "physical":
        return SparseObservation(observation_indices)
    if geometry == "spectral":
        return SpectralObservation(observation_indices, mode_count=spectral_mode_count(config), real_only=True)
    raise ValueError(f"Unknown observation geometry: {config.obs_geometry}")


def observation_covariance(scenario: KSE64Scenario, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    config = scenario.config
    return (config.obs_noise * scenario.observation_scale) ** 2 * torch.eye(
        scenario.observation_dim,
        dtype=dtype,
        device=device,
    )


def generate_scenario(config: KSE64Config, input_path: Path, device: torch.device) -> KSE64Scenario:
    truth = load_truth(input_path, config.sample_index, config.steps, device)
    generator = torch.Generator(device=device).manual_seed(config.seed)
    obs_stride = config.state_dim // config.observed_points
    observation_indices = torch.arange(0, config.state_dim, obs_stride, dtype=torch.int64, device=device)
    truth_std = float(truth.std())
    operator = make_observation_operator(config, observation_indices)
    clean_observation_stack = torch.stack(
        [operator(truth[step].unsqueeze(0)).squeeze(0) for step in range(config.steps + 1)],
        dim=0,
    )
    geometry = str(config.obs_geometry).lower()
    observation_dim = int(clean_observation_stack.shape[-1])
    observation_scale = truth_std if geometry == "physical" else float(clean_observation_stack.std())
    observation_scale = max(float(observation_scale), 1.0e-12)
    obs_noise_values = torch.randn(
        (config.steps + 1, observation_dim),
        dtype=truth.dtype,
        device=device,
        generator=generator,
    )
    obs_interval = max(int(config.obs_interval), 1)
    observations: dict[int, torch.Tensor] = {}
    for step in range(0, config.steps + 1, obs_interval):
        observations[step] = clean_observation_stack[step] + config.obs_noise * observation_scale * obs_noise_values[step]
    if geometry == "spectral":
        coarse_init = spectral_real_inverse(observations[0], config.observed_points)
        init_center = periodic_spectral_upsample(coarse_init, config.state_dim)
    else:
        init_center = periodic_spectral_upsample(observations[0], config.state_dim)
    init_noise = smooth_periodic_noise(
        torch.randn((config.ensemble_size, config.state_dim), dtype=truth.dtype, device=device, generator=generator),
        passes=3,
    )
    initial_ensemble = (init_center.unsqueeze(0) + config.init_noise * truth_std * init_noise).clamp(-8.0, 8.0)
    forecast_noise = smooth_periodic_noise(
        torch.randn((config.steps, config.ensemble_size, config.state_dim), dtype=truth.dtype, device=device, generator=generator),
        passes=2,
    )
    initial_mu_ensemble = torch.linspace(config.mu_lower, config.mu_upper, config.ensemble_size, dtype=truth.dtype, device=device)
    return KSE64Scenario(
        config=config,
        truth=truth,
        observations=observations,
        observation_indices=observation_indices,
        observation_geometry=geometry,
        observation_dim=observation_dim,
        observation_scale=observation_scale,
        initial_ensemble=initial_ensemble,
        forecast_noise=forecast_noise,
        initial_mu_ensemble=initial_mu_ensemble,
        true_mu=sample_true_mu(config.sample_index),
        truth_std=truth_std,
    )


def trace_common_payload(scenario: KSE64Scenario) -> dict[str, Any]:
    observation_steps = torch.as_tensor(sorted(scenario.observations), dtype=torch.int64)
    observations = torch.stack([scenario.observations[int(step)] for step in observation_steps.tolist()], dim=0)
    return {
        "trace_truth": scenario.truth.detach().cpu().numpy(),
        "trace_observations": observations.detach().cpu().numpy(),
        "trace_observation_steps": observation_steps.detach().cpu().numpy(),
        "trace_observation_indices": scenario.observation_indices.detach().cpu().numpy(),
        "trace_initial_ensemble": scenario.initial_ensemble.detach().cpu().numpy(),
        "trace_forecast_noise": scenario.forecast_noise.detach().cpu().numpy(),
        "trace_initial_mu_ensemble": scenario.initial_mu_ensemble.detach().cpu().numpy(),
        "trace_true_mu": np.asarray(scenario.true_mu, dtype=np.float64),
        "trace_observation_geometry": np.asarray(scenario.observation_geometry),
        "trace_observation_dim": np.asarray(int(scenario.observation_dim), dtype=np.int64),
        "trace_observation_scale": np.asarray(float(scenario.observation_scale), dtype=np.float64),
        "trace_spectral_mode_count": np.asarray(spectral_mode_count(scenario.config), dtype=np.int64),
        "trace_temporal_obs_interval": np.asarray(int(scenario.config.obs_interval), dtype=np.int64),
        "trace_spatial_downsampling_factor": np.asarray(int(scenario.config.state_dim // scenario.config.observed_points), dtype=np.int64),
    }


def evidence_score(
    ensemble_observation: torch.Tensor,
    observation: torch.Tensor,
    obs_noise_std: float,
    shrinkage: float,
    dimension_weights: torch.Tensor | None,
) -> torch.Tensor:
    mean = ensemble_observation.mean(dim=0)
    anomalies = ensemble_observation - mean
    covariance = anomalies.mT @ anomalies / max(ensemble_observation.shape[0] - 1, 1)
    covariance = (1.0 - shrinkage) * covariance + shrinkage * torch.diag(torch.diagonal(covariance))
    covariance = covariance + (obs_noise_std**2 + 1.0e-9) * torch.eye(
        observation.numel(), dtype=observation.dtype, device=observation.device
    )
    residual = observation - mean
    if dimension_weights is not None:
        weights = dimension_weights.to(dtype=observation.dtype, device=observation.device).clamp_min(1.0e-8)
        weights = observation.numel() * weights / weights.sum().clamp_min(1.0e-12)
        variances = torch.diagonal(covariance).clamp_min(1.0e-12)
        marginal_terms = residual.square() / variances + variances.log() + math.log(2.0 * math.pi)
        return -0.5 * torch.sum(weights * marginal_terms)
    factor = stable_cholesky(covariance)
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
    return output / output.sum().clamp_min(1.0e-300)


def weighted_denkf_analysis(
    state_ensemble: torch.Tensor,
    weights: torch.Tensor,
    observation: torch.Tensor,
    observation_operator: Any,
    observation_covariance: torch.Tensor,
) -> torch.Tensor:
    weights = weights.to(dtype=state_ensemble.dtype, device=state_ensemble.device).clamp_min(1.0e-300)
    weights = weights / weights.sum().clamp_min(1.0e-300)
    predicted = observation_operator(state_ensemble)
    x_mean = torch.sum(weights.unsqueeze(-1) * state_ensemble, dim=0)
    z_mean = torch.sum(weights.unsqueeze(-1) * predicted, dim=0)
    x_anomalies = state_ensemble - x_mean
    z_anomalies = predicted - z_mean
    denom = (1.0 - weights.square().sum()).clamp_min(torch.finfo(state_ensemble.dtype).eps)
    weighted_z = weights.unsqueeze(-1) * z_anomalies
    cross_covariance = x_anomalies.mT @ weighted_z / denom
    innovation_covariance = z_anomalies.mT @ weighted_z / denom + observation_covariance
    factor = stable_cholesky(innovation_covariance)
    gain = torch.cholesky_solve(cross_covariance.mT, factor).mT
    updated_mean = x_mean + gain @ (observation - z_mean)
    updated_anomalies = x_anomalies - 0.5 * (z_anomalies @ gain.mT)
    return updated_mean.unsqueeze(0) + updated_anomalies


def augmented_mu_denkf_analysis(
    state_ensemble: torch.Tensor,
    mu_ensemble: torch.Tensor,
    observation: torch.Tensor,
    operator: Any,
    covariance: torch.Tensor,
    lower: float,
    upper: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    augmented = torch.cat([state_ensemble, mu_ensemble[:, None]], dim=-1)
    updated = denkf_analysis(augmented, observation, operator, covariance)
    return updated[:, :-1], updated[:, -1].clamp(lower, upper)


def weighted_augmented_mu_denkf_analysis(
    state_ensemble: torch.Tensor,
    mu_ensemble: torch.Tensor,
    weights: torch.Tensor,
    observation: torch.Tensor,
    operator: Any,
    covariance: torch.Tensor,
    lower: float,
    upper: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    augmented = torch.cat([state_ensemble, mu_ensemble[:, None]], dim=-1)
    updated = weighted_denkf_analysis(augmented, weights, observation, operator, covariance)
    return updated[:, :-1], updated[:, -1].clamp(lower, upper)


def member_mu_cloud(mu: torch.Tensor, *, mu_grid: torch.Tensor, ensemble_size: int, jitter_fraction: float, confidence_power: float, log_scores: torch.Tensor) -> torch.Tensor:
    mu_tensor = torch.as_tensor(mu, dtype=mu_grid.dtype, device=mu_grid.device)
    if ensemble_size <= 1 or jitter_fraction <= 0.0:
        return mu_tensor.expand(ensemble_size).clone()
    confidence, _ = evidence_gap_confidence(log_scores)
    effective_fraction = jitter_fraction * float((1.0 - confidence) ** max(confidence_power, 1.0e-8))
    offsets = torch.linspace(-0.5, 0.5, ensemble_size, dtype=mu_grid.dtype, device=mu_grid.device)
    return (mu_tensor + effective_fraction * (float(mu_grid[-1]) - float(mu_grid[0])) * offsets).clamp(float(mu_grid[0]), float(mu_grid[-1]))


def evidence_gap_state_weights(base_weights: torch.Tensor, log_scores: torch.Tensor, config: KSE64Config) -> torch.Tensor:
    weights = base_weights.clamp_min(1.0e-300)
    weights = weights / weights.sum().clamp_min(1.0e-300)
    power = max(float(config.state_weight_power), 1.0e-8)
    if abs(power - 1.0) > 1.0e-12:
        weights = weights.pow(power)
        weights = weights / weights.sum().clamp_min(1.0e-300)
    if config.state_map_blend > 0.0 and log_scores.numel() > 1:
        confidence, _ = evidence_gap_confidence(log_scores)
        blend = float(np.clip(config.state_map_blend * confidence ** max(config.state_map_blend_confidence_power, 1.0e-8), 0.0, 1.0))
        if blend > 1.0e-12:
            one_hot = torch.zeros_like(weights)
            one_hot[int(torch.argmax(log_scores))] = 1.0
            weights = (1.0 - blend) * weights + blend * one_hot
    return weights / weights.sum().clamp_min(1.0e-300)


def interpolate_branch_mean(mu_grid: torch.Tensor, branch_means: torch.Tensor, mu_value: float) -> torch.Tensor:
    if mu_grid.numel() == 1:
        return branch_means[0]
    value = float(np.clip(mu_value, float(mu_grid[0]), float(mu_grid[-1])))
    right = int(torch.searchsorted(mu_grid, torch.tensor(value, dtype=mu_grid.dtype, device=mu_grid.device)).clamp(1, mu_grid.numel() - 1))
    left = right - 1
    denom = (mu_grid[right] - mu_grid[left]).clamp_min(torch.finfo(mu_grid.dtype).eps)
    fraction = (value - float(mu_grid[left])) / float(denom)
    return branch_means[left] * (1.0 - fraction) + branch_means[right] * fraction


def pce_point_estimate(mu_grid: torch.Tensor, branch_means: torch.Tensor, state_weights: torch.Tensor, log_scores: torch.Tensor, config: KSE64Config) -> torch.Tensor:
    mode = config.state_point_estimator_mode.lower()
    if mode in {"mean", "weighted_mean"}:
        weights = state_weights / state_weights.sum().clamp_min(1.0e-300)
        return (weights.unsqueeze(-1) * branch_means).sum(dim=0)
    if mode == "map":
        return branch_means[int(torch.argmax(log_scores))]
    mu_hat = torch_refined_alpha_map(mu_grid, log_scores)
    return interpolate_branch_mean(mu_grid, branch_means, mu_hat)


def apply_fixed_analysis(method: MethodName, ensemble: torch.Tensor, observation: torch.Tensor, operator: Any, covariance: torch.Tensor, seed: int, step: int, device: torch.device) -> torch.Tensor:
    if method == "denkf":
        return denkf_analysis(ensemble, observation, operator, covariance)
    if method == "letkf":
        return letkf_analysis(ensemble, observation, operator, covariance)
    if method == "iensf":
        generator = torch.Generator(device=device).manual_seed(seed * 100_000 + step)
        return iensf_analysis(
            ensemble,
            observation,
            operator,
            covariance,
            IEnSFConfig(sampling_time_step_count=10, refinement_iterations=1, endpoint_epsilon=1.0e-3),
            generator,
        )
    raise ValueError(method)


def run_fixed_method(scenario: KSE64Scenario, method: MethodName, device: torch.device, *, record_trace: bool = False) -> dict[str, Any]:
    config = scenario.config
    solver = KSEETDRK4(
        state_dim=config.state_dim,
        length=config.length,
        sub_dt=config.saved_dt / config.solver_substeps,
        device=device,
        dtype=torch.float64,
    )
    ensemble = scenario.initial_ensemble.clone()
    weights = torch.full((config.ensemble_size,), 1.0 / config.ensemble_size, dtype=ensemble.dtype, device=device)
    operator = make_observation_operator(config, scenario.observation_indices)
    covariance = observation_covariance(scenario, device, ensemble.dtype)
    metrics = TrajectoryMetrics()
    traces: list[torch.Tensor] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(config.steps + 1):
        mean_state = ensemble.mean(dim=0)
        if record_trace:
            traces.append(mean_state.detach().cpu())
        metrics.add(ensemble, scenario.truth[step], weights, point_estimate=mean_state)
        if step == config.steps:
            break
        ensemble = solver.step_saved(
            ensemble,
            torch.full((config.ensemble_size,), config.fixed_mu, dtype=ensemble.dtype, device=device),
            substeps=config.solver_substeps,
            noise=scenario.forecast_noise[step],
            noise_scale=config.process_noise * scenario.truth_std,
        )
        if step + 1 not in scenario.observations:
            continue
        ensemble = apply_fixed_analysis(method, ensemble, scenario.observations[step + 1], operator, covariance, config.seed, step + 1, device)
        ensemble = torch.nan_to_num(ensemble, nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = metrics.finalize()
    result.update(
        runtime_seconds=float(time.perf_counter() - started),
        peak_gpu_memory_mb=float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
        mu_estimate=float(config.fixed_mu),
        mu_map=float(config.fixed_mu),
        mu_absolute_error=abs(float(config.fixed_mu) - scenario.true_mu),
        forward_member_steps=int(config.steps * config.ensemble_size),
    )
    if record_trace:
        result["mean_states"] = torch.stack(traces).numpy()
    return result


def run_aug_enkf(scenario: KSE64Scenario, device: torch.device, *, record_trace: bool = False) -> dict[str, Any]:
    config = scenario.config
    solver = KSEETDRK4(state_dim=config.state_dim, length=config.length, sub_dt=config.saved_dt / config.solver_substeps, device=device, dtype=torch.float64)
    ensemble = scenario.initial_ensemble.clone()
    mu_ensemble = scenario.initial_mu_ensemble.clone()
    weights = torch.full((config.ensemble_size,), 1.0 / config.ensemble_size, dtype=ensemble.dtype, device=device)
    operator = make_observation_operator(config, scenario.observation_indices)
    covariance = observation_covariance(scenario, device, ensemble.dtype)
    metrics = TrajectoryMetrics()
    traces: list[torch.Tensor] = []
    mu_history: list[torch.Tensor] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(config.steps + 1):
        mean_state = ensemble.mean(dim=0)
        if record_trace:
            traces.append(mean_state.detach().cpu())
            mu_history.append(mu_ensemble.detach().cpu())
        metrics.add(ensemble, scenario.truth[step], weights, point_estimate=mean_state)
        if step == config.steps:
            break
        ensemble = solver.step_saved(
            ensemble,
            mu_ensemble,
            substeps=config.solver_substeps,
            noise=scenario.forecast_noise[step],
            noise_scale=config.process_noise * scenario.truth_std,
        )
        if step + 1 not in scenario.observations:
            continue
        ensemble, mu_ensemble = augmented_mu_denkf_analysis(
            ensemble,
            mu_ensemble,
            scenario.observations[step + 1],
            operator,
            covariance,
            config.mu_lower,
            config.mu_upper,
        )
        ensemble = torch.nan_to_num(ensemble, nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    mu_est = float(mu_ensemble.mean())
    result = metrics.finalize()
    result.update(
        runtime_seconds=float(time.perf_counter() - started),
        peak_gpu_memory_mb=float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
        mu_estimate=mu_est,
        mu_map=float(mu_ensemble.median()),
        mu_absolute_error=abs(mu_est - scenario.true_mu),
        forward_member_steps=int(config.steps * config.ensemble_size),
    )
    if record_trace:
        result["mean_states"] = torch.stack(traces).numpy()
        result["mu_history"] = torch.stack(mu_history).numpy()
    return result


def run_bma(scenario: KSE64Scenario, device: torch.device, *, record_trace: bool = False) -> dict[str, Any]:
    config = scenario.config
    solver = KSEETDRK4(state_dim=config.state_dim, length=config.length, sub_dt=config.saved_dt / config.solver_substeps, device=device, dtype=torch.float64)
    mu_grid = torch.tensor(config.bma_mu_grid, dtype=torch.float64, device=device)
    path_count = int(mu_grid.numel())
    branches = scenario.initial_ensemble.unsqueeze(0).repeat(path_count, 1, 1)
    log_scores = torch.zeros(path_count, dtype=torch.float64, device=device)
    weights_path = torch.softmax(log_scores, dim=0)
    operator = make_observation_operator(config, scenario.observation_indices)
    covariance = observation_covariance(scenario, device, torch.float64)
    metrics = TrajectoryMetrics()
    traces: list[torch.Tensor] = []
    weight_history: list[torch.Tensor] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(config.steps + 1):
        flat = branches.reshape(-1, config.state_dim)
        flat_weights = weights_path.unsqueeze(1).expand(-1, config.ensemble_size).reshape(-1) / config.ensemble_size
        branch_means = branches.mean(dim=1)
        estimate = (weights_path.unsqueeze(-1) * branch_means).sum(dim=0)
        if record_trace:
            traces.append(estimate.detach().cpu())
            weight_history.append(weights_path.detach().cpu())
        metrics.add(flat, scenario.truth[step], flat_weights, point_estimate=estimate)
        if step == config.steps:
            break
        for i in range(path_count):
            branches[i] = solver.step_saved(
                branches[i],
                torch.full((config.ensemble_size,), float(mu_grid[i]), dtype=torch.float64, device=device),
                substeps=config.solver_substeps,
                noise=scenario.forecast_noise[step],
                noise_scale=config.process_noise * scenario.truth_std,
            )
        if step + 1 not in scenario.observations:
            continue
        predicted_obs = torch.stack([operator(branches[i]) for i in range(path_count)])
        evidence = torch.stack(
            [
                evidence_score(
                    predicted_obs[i],
                    scenario.observations[step + 1],
                    config.obs_noise * scenario.observation_scale,
                    config.evidence_shrinkage,
                    None,
                )
                for i in range(path_count)
            ]
        )
        log_scores = log_scores + 0.75 * (evidence - evidence.mean())
        weights_path = torch.softmax(log_scores, dim=0)
        for i in range(path_count):
            branches[i] = denkf_analysis(branches[i], scenario.observations[step + 1], operator, covariance)
            branches[i] = torch.nan_to_num(branches[i], nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    mu_est = float((weights_path * mu_grid).sum())
    result = metrics.finalize()
    result.update(
        runtime_seconds=float(time.perf_counter() - started),
        peak_gpu_memory_mb=float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
        mu_estimate=mu_est,
        mu_map=float(mu_grid[int(torch.argmax(log_scores))]),
        mu_absolute_error=abs(mu_est - scenario.true_mu),
        final_entropy=float(entropy(weights_path)),
        forward_member_steps=int(config.steps * path_count * config.ensemble_size),
    )
    if record_trace:
        result["mean_states"] = torch.stack(traces).numpy()
        result["weight_history"] = torch.stack(weight_history).numpy()
    return result


def run_pce_apce(scenario: KSE64Scenario, method: MethodName, device: torch.device, *, record_trace: bool = False) -> dict[str, Any]:
    config = scenario.config
    solver = KSEETDRK4(state_dim=config.state_dim, length=config.length, sub_dt=config.saved_dt / config.solver_substeps, device=device, dtype=torch.float64)
    mu_grid = torch.tensor(config.coarse_mu_grid, dtype=torch.float64, device=device)
    bounds = (float(config.mu_lower), float(config.mu_upper))
    branches = scenario.initial_ensemble.unsqueeze(0).repeat(mu_grid.numel(), 1, 1)
    shadow = branches.clone()
    log_scores = torch.zeros(mu_grid.numel(), dtype=torch.float64, device=device)
    alpha_members = torch.stack(
        [
            member_mu_cloud(
                mu,
                mu_grid=mu_grid,
                ensemble_size=config.ensemble_size,
                jitter_fraction=config.branch_member_alpha_jitter,
                confidence_power=config.branch_member_alpha_jitter_confidence_power,
                log_scores=log_scores,
            )
            for mu in mu_grid
        ],
        dim=0,
    )
    path_weights = torch.softmax(log_scores, dim=0)
    state_weights = evidence_gap_state_weights(path_weights, log_scores, config)
    operator = make_observation_operator(config, scenario.observation_indices)
    covariance = observation_covariance(scenario, device, torch.float64)
    metrics = TrajectoryMetrics()
    traces: list[torch.Tensor] = []
    weight_history: list[torch.Tensor] = []
    mu_grid_history: list[torch.Tensor] = []
    mu_est_history: list[float] = []
    regrid_count = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    forward_member_steps = 0
    for step in range(config.steps + 1):
        flat = branches.reshape(-1, config.state_dim)
        flat_weights = state_weights.unsqueeze(1).expand(-1, config.ensemble_size).reshape(-1) / config.ensemble_size
        branch_means = branches.mean(dim=1)
        estimate = pce_point_estimate(mu_grid, branch_means, state_weights, log_scores, config)
        if record_trace:
            traces.append(estimate.detach().cpu())
            weight_history.append(state_weights.detach().cpu())
            mu_grid_history.append(mu_grid.detach().cpu())
            mu_est_history.append(torch_refined_alpha_map(mu_grid, log_scores))
        metrics.add(flat, scenario.truth[step], flat_weights, point_estimate=estimate)
        if step == config.steps:
            break
        for i in range(int(mu_grid.numel())):
            branches[i] = solver.step_saved(
                branches[i],
                alpha_members[i],
                substeps=config.solver_substeps,
                noise=scenario.forecast_noise[step],
                noise_scale=config.process_noise * scenario.truth_std,
            )
            shadow[i] = solver.step_saved(
                shadow[i],
                alpha_members[i],
                substeps=config.solver_substeps,
                noise=scenario.forecast_noise[step],
                noise_scale=config.process_noise * scenario.truth_std,
            )
        forward_member_steps += int(2 * mu_grid.numel() * config.ensemble_size)
        if step + 1 not in scenario.observations:
            continue
        shadow_obs = torch.stack([operator(shadow[i]) for i in range(int(mu_grid.numel()))])
        dimension_weights = None
        if method == "apce":
            between = shadow_obs.mean(dim=1).var(dim=0, unbiased=True)
            dimension_weights = config.dimension_weight_floor + config.dimension_weight_gain * between / between.max().clamp_min(1.0e-12)
        evidence = torch.stack(
            [
                evidence_score(
                    shadow_obs[i],
                    scenario.observations[step + 1],
                    config.obs_noise * scenario.observation_scale,
                    config.evidence_shrinkage,
                    dimension_weights,
                )
                for i in range(int(mu_grid.numel()))
            ]
        )
        centered = evidence - evidence.mean()
        if method == "pce":
            log_scores = log_scores + config.pce_temperature * centered
            path_weights = torch.softmax(log_scores, dim=0)
            state_weights = evidence_gap_state_weights(path_weights, log_scores, config)
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
            log_scores = calibration.forgetting * log_scores + calibration.temperature * centered
            path_weights = torch.softmax(log_scores, dim=0)
            calibrated = entropy_project(path_weights, calibration.entropy_floor)
            state_weights = evidence_gap_state_weights(calibrated, log_scores, config)
        refined = torch_local_alpha_grid(
            mu_grid,
            log_scores,
            points=config.local_grid_points,
            bounds=bounds,
            topk=config.local_grid_topk,
            min_spacing=config.local_grid_min_spacing,
        )
        if refined.shape != mu_grid.shape or not torch.allclose(refined, mu_grid):
            branches = torch_regrid_paths(mu_grid, branches, refined)
            shadow = torch_regrid_paths(mu_grid, shadow, refined)
            alpha_members = torch_regrid_paths(mu_grid, alpha_members, refined)
            log_scores = torch_regrid_paths(mu_grid, log_scores, refined)
            mu_grid = refined
            path_weights = torch.softmax(log_scores, dim=0)
            state_weights = evidence_gap_state_weights(path_weights, log_scores, config)
            regrid_count += 1
        # Figure 3 latest mechanism: local branch analysis plus Aug-EnKF style
        # member-wise and global weighted state--mu correction.
        local = torch.empty_like(branches)
        for i in range(int(mu_grid.numel())):
            local[i] = denkf_analysis(branches[i], scenario.observations[step + 1], operator, covariance)
        confidence, _ = evidence_gap_confidence(log_scores)
        global_strength = float(np.clip(config.global_analysis_strength * confidence ** max(config.global_analysis_confidence_power, 1.0e-8), 0.0, 1.0))
        if global_strength > 1.0e-12:
            flat_forecast = branches.reshape(-1, config.state_dim)
            flat_weights_analysis = state_weights.unsqueeze(1).expand(-1, config.ensemble_size).reshape(-1) / config.ensemble_size
            global_analysis = weighted_denkf_analysis(flat_forecast, flat_weights_analysis, scenario.observations[step + 1], operator, covariance).reshape_as(branches)
            branches = (1.0 - global_strength) * local + global_strength * global_analysis
        else:
            branches = local
        if config.branch_augmented_alpha_analysis_strength > 1.0e-12:
            joint_branches = torch.empty_like(branches)
            joint_mu = torch.empty_like(alpha_members)
            for i in range(int(mu_grid.numel())):
                jb, jm = augmented_mu_denkf_analysis(branches[i], alpha_members[i], scenario.observations[step + 1], operator, covariance, bounds[0], bounds[1])
                joint_branches[i] = jb
                joint_mu[i] = jm
            strength = float(config.branch_augmented_alpha_analysis_strength)
            branches = (1.0 - strength) * branches + strength * joint_branches
            alpha_members = (1.0 - strength) * alpha_members + strength * joint_mu
        if config.global_augmented_alpha_analysis_strength > 1.0e-12:
            flat_branches = branches.reshape(-1, config.state_dim)
            flat_mu = alpha_members.reshape(-1)
            flat_weights_analysis = state_weights.unsqueeze(1).expand(-1, config.ensemble_size).reshape(-1) / config.ensemble_size
            gjb, gjm = weighted_augmented_mu_denkf_analysis(
                flat_branches,
                flat_mu,
                flat_weights_analysis,
                scenario.observations[step + 1],
                operator,
                covariance,
                bounds[0],
                bounds[1],
            )
            strength = float(config.global_augmented_alpha_analysis_strength)
            branches = (1.0 - strength) * branches + strength * gjb.reshape_as(branches)
            alpha_members = (1.0 - strength) * alpha_members + strength * gjm.reshape_as(alpha_members)
        branches = torch.nan_to_num(branches, nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0)
        alpha_members = alpha_members.clamp(bounds[0], bounds[1])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    mu_est = float(torch_refined_alpha_map(mu_grid, log_scores))
    result = metrics.finalize()
    result.update(
        runtime_seconds=float(time.perf_counter() - started),
        peak_gpu_memory_mb=float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
        mu_estimate=mu_est,
        mu_map=float(mu_grid[int(torch.argmax(log_scores))]),
        mu_absolute_error=abs(mu_est - scenario.true_mu),
        final_entropy=float(entropy(state_weights)),
        evidence_entropy=float(entropy(torch.softmax(log_scores, dim=0))),
        alpha_regrid_count=int(regrid_count),
        final_path_count=int(mu_grid.numel()),
        forward_member_steps=int(forward_member_steps),
    )
    if record_trace:
        result["mean_states"] = torch.stack(traces).numpy()
        result["weight_history"] = [w.numpy() for w in weight_history]
        result["mu_grid_history"] = [g.numpy() for g in mu_grid_history]
        result["mu_estimate_history"] = np.asarray(mu_est_history, dtype=float)
    return result


def run_one(method: MethodName, scenario: KSE64Scenario, device: torch.device, record_trace: bool) -> dict[str, Any]:
    if method in {"denkf", "letkf", "iensf"}:
        result = run_fixed_method(scenario, method, device, record_trace=record_trace)
    if method == "aug_enkf":
        result = run_aug_enkf(scenario, device, record_trace=record_trace)
    elif method == "bma_static":
        result = run_bma(scenario, device, record_trace=record_trace)
    elif method in {"pce", "apce"}:
        result = run_pce_apce(scenario, method, device, record_trace=record_trace)
    elif method in {"denkf", "letkf", "iensf"}:
        pass
    else:
        raise ValueError(method)
    if record_trace:
        result.update(trace_common_payload(scenario))
    return result


def save_trace_npz(path: Path, result: dict[str, Any]) -> None:
    arrays: dict[str, Any] = {}
    for key in [
        "mean_states",
        "mu_history",
        "weight_history",
        "mu_grid_history",
        "mu_estimate_history",
        "trace_truth",
        "trace_observations",
        "trace_observation_steps",
        "trace_observation_indices",
        "trace_initial_ensemble",
        "trace_forecast_noise",
        "trace_initial_mu_ensemble",
        "trace_true_mu",
        "trace_observation_geometry",
        "trace_observation_dim",
        "trace_observation_scale",
        "trace_spectral_mode_count",
        "trace_temporal_obs_interval",
        "trace_spatial_downsampling_factor",
    ]:
        if key in result:
            value = result.pop(key)
            if isinstance(value, list):
                arrays[key] = np.asarray(value, dtype=object)
            else:
                arrays[key] = value
    if arrays:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)


def write_summary(output_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted((output_root / "runs").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(payload)
    if not rows:
        return
    summary_dir = output_root / "source_data"
    summary_dir.mkdir(parents=True, exist_ok=True)
    run_csv = summary_dir / "run_source_data.csv"
    keys = sorted({key for row in rows for key in row if key not in {"config"}})
    with run_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: clean_json(row.get(key, "")) for key in keys})
    method_rows: list[dict[str, Any]] = []
    groups = sorted(
        {
            (
                str(row.get("observation_geometry", "physical")),
                int(row.get("downsampling_factor", 64)),
                int(row.get("temporal_obs_interval", 1)),
            )
            for row in rows
        }
    )
    present_methods = [method for method in METHODS if any(row.get("method") == method for row in rows)]
    for geometry, factor, temporal in groups:
        for method in present_methods:
            subset = [
                row
                for row in rows
                if str(row.get("observation_geometry", "physical")) == geometry
                and int(row.get("downsampling_factor", 64)) == factor
                and int(row.get("temporal_obs_interval", 1)) == temporal
                and row.get("method") == method
                and row.get("valid")
            ]
            total_subset = [
                row
                for row in rows
                if int(row.get("downsampling_factor", 64)) == factor
                and int(row.get("temporal_obs_interval", 1)) == temporal
                and str(row.get("observation_geometry", "physical")) == geometry
                and row.get("method") == method
            ]
            item: dict[str, Any] = {
                "observation_geometry": geometry,
                "downsampling_factor": factor,
                "temporal_obs_interval": temporal,
                "observed_points": 1024 // factor,
                "observation_dim": int(total_subset[0].get("observation_dim", 1024 // factor)) if total_subset else "",
                "method": method,
                "label": METHOD_LABELS[method],
                "n_valid": len(subset),
                "n_total": len(total_subset),
            }
            for key in ["nrmse", "crps", "mu_absolute_error", "coverage_90", "interval_width_90", "runtime_seconds"]:
                vals = np.asarray([float(row[key]) for row in subset if row.get(key) not in (None, "")], dtype=float)
                item[f"{key}_mean"] = float(vals.mean()) if vals.size else math.nan
                item[f"{key}_std"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
            method_rows.append(item)
    method_csv = summary_dir / "method_summary.csv"
    keys = sorted({key for row in method_rows for key in row})
    with method_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(method_rows)
    seed_indices = sorted({int(row.get("seed_index", -1)) for row in rows if row.get("seed_index", "") not in (None, "")})
    spatial_factors = sorted({int(row.get("downsampling_factor", 64)) for row in rows})
    temporal_intervals = sorted({int(row.get("temporal_obs_interval", 1)) for row in rows})
    geometries = sorted({str(row.get("observation_geometry", "physical")) for row in rows})
    protocol = "figure4-kse-nmi-official"
    if len(geometries) == 1:
        protocol += f"-{geometries[0]}"
    if len(spatial_factors) == 1:
        protocol += f"-space{spatial_factors[0]}x"
    if len(temporal_intervals) == 1:
        protocol += f"-time{temporal_intervals[0]}x"
    protocol += f"-{len(present_methods)}methods-{len(seed_indices)}seeds"
    manifest = {
        "protocol": protocol,
        "output_root": str(output_root),
        "methods": present_methods,
        "observation_geometries": geometries,
        "spatial_downsampling_factors": spatial_factors,
        "temporal_obs_intervals": temporal_intervals,
        "seed_indices": seed_indices,
        "run_count": len(rows),
        "valid_count": sum(1 for row in rows if row.get("valid")),
        "run_source_data": str(run_csv),
        "run_source_data_sha256": file_sha256(run_csv),
        "method_summary": str(method_csv),
        "method_summary_sha256": file_sha256(method_csv),
        "note": "PCE/APCE use the Figure 3 augmented-local refinement mechanism on official NMI KSE dynamics.",
    }
    write_json(summary_dir / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official NMI-KSE sparse-observation full-method smoke.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--method", choices=METHODS, default=None)
    parser.add_argument("--seed-index", type=int, default=None)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--seed-base", type=int, default=2026081400)
    parser.add_argument("--downsampling-factor", type=int, default=64)
    parser.add_argument("--temporal-obs-interval", type=int, default=1)
    parser.add_argument("--obs-geometry", choices=("physical", "spectral"), default="physical")
    parser.add_argument("--spectral-mode-count", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--record-trace", action="store_true")
    parser.add_argument("--write-summary", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        write_summary(output_root)
        print(json.dumps({"status": "summary_written", "output_root": str(output_root)}, ensure_ascii=False))
        return
    if args.method is None or args.seed_index is None:
        raise ValueError("--method and --seed-index are required unless --summary-only is used")
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    if args.sample_index is None:
        sample_index = int(args.seed_index % 9)
    else:
        sample_index = int(args.sample_index)
    seed = int(args.seed_base + args.seed_index)
    if 1024 % int(args.downsampling_factor) != 0:
        raise ValueError("--downsampling-factor must divide 1024")
    if int(args.temporal_obs_interval) < 1:
        raise ValueError("--temporal-obs-interval must be >= 1")
    observed_points = 1024 // int(args.downsampling_factor)
    config = KSE64Config(
        seed=seed,
        sample_index=sample_index,
        obs_geometry=str(args.obs_geometry),
        observed_points=observed_points,
        spectral_mode_count=int(args.spectral_mode_count),
        obs_interval=int(args.temporal_obs_interval),
    )
    temporal_tag = f"t{int(args.temporal_obs_interval)}"
    obs_tag = str(args.obs_geometry).lower()
    run_id = f"kse_nmi{args.downsampling_factor}x_{temporal_tag}_{obs_tag}_{args.method}_seed{args.seed_index:02d}_sample{sample_index:02d}"
    run_path = output_root / "runs" / f"{run_id}.json"
    trace_path = output_root / "traces" / f"{run_id}.npz"
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "run_id": run_id,
        "case": f"kse_nmi_official_{args.downsampling_factor}x_{temporal_tag}_{obs_tag}",
        "method": args.method,
        "label": METHOD_LABELS[args.method],
        "seed_index": int(args.seed_index),
        "seed": seed,
        "sample_index": sample_index,
        "true_mu": sample_true_mu(sample_index),
        "observed_points": int(config.observed_points),
        "downsampling_factor": int(config.state_dim // config.observed_points),
        "temporal_obs_interval": int(config.obs_interval),
        "observation_geometry": obs_tag,
        "spectral_mode_count": spectral_mode_count(config),
        "valid": False,
        "hostname": os.uname().nodename if hasattr(os, "uname") else "",
        "device": str(device),
        "input_path": str(args.input),
        "config": asdict(config),
    }
    try:
        scenario = generate_scenario(config, args.input, device)
        payload.update(
            observation_dim=int(scenario.observation_dim),
            observation_scale=float(scenario.observation_scale),
        )
        result = run_one(args.method, scenario, device, args.record_trace)
        if args.record_trace:
            save_trace_npz(trace_path, result)
            payload["trace_npz"] = str(trace_path)
            payload["trace_npz_sha256"] = file_sha256(trace_path) if trace_path.exists() else ""
        payload.update(result)
        payload["valid"] = bool(np.isfinite(float(payload["nrmse"])))
        payload["status"] = "completed"
    except Exception as exc:  # noqa: BLE001 - run-level failure must be recorded
        payload["status"] = "failed"
        payload["failure_type"] = type(exc).__name__
        payload["failure_message"] = str(exc)
    payload["wall_seconds"] = float(time.perf_counter() - started)
    payload["script_path"] = str(Path(__file__).resolve())
    payload["script_sha256"] = file_sha256(Path(__file__).resolve())
    write_json(run_path, payload)
    if args.write_summary:
        write_summary(output_root)
    print(json.dumps(clean_json(payload), ensure_ascii=False))


if __name__ == "__main__":
    main()
