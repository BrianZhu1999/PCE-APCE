from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hilda_da.alpha import liu_quantile
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
from hilda_da.observations import SparseObservation
from hilda_da.strong_baselines import (
    EnSFConfig,
    IEnSFConfig,
    ensf_analysis,
    ensf_lr_analysis,
    ensf_lr_ridge_analysis,
    iensf_analysis,
)
from hilda_da.systems.applied_odes import final_figure3_case_names, final_figure3_case_spec
from hilda_da.systems.base import HybridSystem


CaseName = str
MethodName = Literal[
    "denkf",
    "letkf",
    "iensf",
    "ensf",
    "ensf_lr",
    "ensf_lr_ridge",
    "aug_enkf",
    "bma_static",
    "pce",
    "apce",
]

CASES: tuple[CaseName, ...] = final_figure3_case_names()
DEFAULT_METHODS: tuple[MethodName, ...] = (
    "denkf",
    "letkf",
    "iensf",
    "aug_enkf",
    "bma_static",
    "pce",
    "apce",
)
ADMISSION_METHODS: tuple[MethodName, ...] = (
    "denkf",
    "letkf",
    "iensf",
    "aug_enkf",
    "bma_static",
    "pce",
    "apce",
)
METHOD_LABELS = {
    "denkf": "DEnKF",
    "letkf": "LETKF",
    "iensf": "IEnSF",
    "ensf": "EnSF",
    "ensf_lr": "EnSF-LR",
    "ensf_lr_ridge": "EnSF-LR-Ridge",
    "aug_enkf": "Aug-EnKF",
    "bma_static": "BMA",
    "pce": "PCE",
    "apce": "APCE",
}

SOURCE_HASH_FILES = (
    PROJECT_ROOT / "paper_experiments" / "run_figure3_applied_ode.py",
    PROJECT_ROOT / "hilda_da" / "systems" / "applied_odes.py",
    PROJECT_ROOT / "hilda_da" / "alpha_refinement.py",
    PROJECT_ROOT / "hilda_da" / "baselines.py",
    PROJECT_ROOT / "hilda_da" / "strong_baselines.py",
    PROJECT_ROOT / "hilda_da" / "metrics.py",
    PROJECT_ROOT / "experiments" / "figure3_tuning_matrix.json",
)


@dataclass(frozen=True)
class CaseConfig:
    name: CaseName
    seed: int
    steps: int
    dt: float
    obs_interval: int
    ensemble_size: int
    obs_noise: float
    alpha_true: float = 0.16
    fixed_alpha: float = 0.50
    alpha_grid: tuple[float, ...] = (0.08, 0.18, 0.28, 0.40, 0.52, 0.66, 0.82)
    pce_temperature: float = 0.62
    apce_temperature: float = 0.54
    apce_min_temperature: float = 0.08
    apce_forgetting: float = 0.975
    apce_entropy_floor: float = 0.34
    evidence_shrinkage: float = 0.22
    dimension_weight_floor: float = 0.35
    dimension_weight_gain: float = 0.65
    local_grid_points: int = 7
    local_grid_topk: int = 3
    local_grid_min_spacing: float = 1.0e-3
    state_weight_power: float = 1.0
    state_map_blend: float = 0.0
    state_map_blend_confidence_power: float = 1.0
    branch_collapse_strength: float = 0.0
    branch_collapse_confidence_power: float = 1.0
    branch_member_alpha_jitter: float = 0.0
    branch_member_alpha_jitter_confidence_power: float = 1.0
    branch_augmented_alpha_analysis_strength: float = 0.0
    global_augmented_alpha_analysis_strength: float = 0.0
    state_point_estimator_mode: str = "mean"
    analysis_iterations: int = 1
    global_analysis_strength: float = 0.0
    global_analysis_confidence_power: float = 1.0
    max_valid_amplitude_ratio: float = 100.0


@dataclass
class Scenario:
    config: CaseConfig
    truth: torch.Tensor
    observations: dict[int, torch.Tensor]
    observation_indices: torch.Tensor
    primary_indices: torch.Tensor
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

    def add(
        self,
        ensemble: torch.Tensor,
        truth: torch.Tensor,
        weights: torch.Tensor,
        *,
        point_estimate: torch.Tensor | None = None,
    ) -> None:
        weights = weights / weights.sum()
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


def config_for_case(name: CaseName, seed: int) -> CaseConfig:
    spec = final_figure3_case_spec(name)
    return CaseConfig(
        name=spec.name,
        seed=seed,
        steps=spec.default_steps,
        dt=spec.default_dt,
        obs_interval=spec.default_obs_interval,
        ensemble_size=spec.default_ensemble_size,
        obs_noise=spec.default_obs_noise,
        alpha_true=spec.alpha_true,
        alpha_grid=spec.alpha_grid,
    )


def load_tuning_profile(matrix_path: Path, profile_name: str) -> dict[str, Any]:
    """Load a pre-registered PCE/APCE tuning profile.

    The returned profile is deliberately applied only to PCE/APCE runs.  Strong
    baselines keep the frozen Figure 3 benchmark configuration even when they
    are included in a diagnostic command.
    """

    name = profile_name.strip() or "baseline"
    if name in {"baseline", "none"}:
        return {
            "name": name,
            "description": "Frozen Figure 3 baseline configuration.",
            "pce_apce_overrides": {},
            "method_overrides": {},
            "case_overrides": {},
        }
    if not matrix_path.is_file():
        raise FileNotFoundError(f"Missing tuning matrix: {matrix_path}")
    payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    profiles = payload.get("profiles", payload)
    if name not in profiles:
        raise KeyError(f"Unknown tuning profile {name!r}; available: {sorted(profiles)}")
    profile = dict(profiles[name])
    profile["name"] = name
    profile.setdefault("pce_apce_overrides", profile.get("overrides", {}))
    profile.setdefault("method_overrides", {})
    profile.setdefault("case_overrides", {})
    allowed = set(CaseConfig.__dataclass_fields__)
    for source, overrides in [("pce_apce_overrides", profile["pce_apce_overrides"])]:
        unknown = sorted(set(overrides) - allowed)
        if unknown:
            raise ValueError(f"{source} in profile {name!r} contains unsupported CaseConfig fields: {unknown}")
    for case, overrides in profile["case_overrides"].items():
        if case not in CASES:
            raise ValueError(f"Profile {name!r} has override for unknown case: {case}")
        if not isinstance(overrides, dict):
            raise TypeError(f"case_overrides[{case!r}] in profile {name!r} must be a mapping")
        method_keys = {"both", "pce", "apce"}
        flat_keys = set(overrides) - method_keys
        unknown = sorted(flat_keys - allowed)
        if unknown:
            raise ValueError(f"case_overrides[{case!r}] in profile {name!r} contains unsupported fields: {unknown}")
        for method_key in method_keys:
            if method_key in overrides:
                nested = overrides[method_key]
                if not isinstance(nested, dict):
                    raise TypeError(f"case_overrides[{case!r}][{method_key!r}] in profile {name!r} must be a mapping")
                nested_unknown = sorted(set(nested) - allowed)
                if nested_unknown:
                    raise ValueError(
                        f"case_overrides[{case!r}][{method_key!r}] in profile {name!r} contains unsupported fields: {nested_unknown}"
                    )
    for method, overrides in profile["method_overrides"].items():
        if method not in {"pce", "apce"}:
            raise ValueError(f"Profile {name!r} has override for unsupported method: {method}")
        unknown = sorted(set(overrides) - allowed)
        if unknown:
            raise ValueError(f"method_overrides[{method!r}] in profile {name!r} contains unsupported fields: {unknown}")
    return profile


def _normalize_config_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    int_fields = {"steps", "obs_interval", "ensemble_size", "local_grid_points", "local_grid_topk"}
    tuple_fields = {"alpha_grid"}
    for key, value in overrides.items():
        if key in tuple_fields:
            output[key] = tuple(float(item) for item in value)
        elif key in int_fields:
            output[key] = int(value)
        elif isinstance(value, (int, float)):
            output[key] = float(value)
        else:
            output[key] = value
    return output


def apply_tuning_profile(config: CaseConfig, profile: dict[str, Any], case: CaseName, method: MethodName) -> CaseConfig:
    if method not in {"pce", "apce"}:
        return config
    overrides: dict[str, Any] = {}
    overrides.update(profile.get("pce_apce_overrides", {}))
    overrides.update(profile.get("method_overrides", {}).get(method, {}))
    case_overrides = profile.get("case_overrides", {}).get(case, {})
    if isinstance(case_overrides, dict):
        flat_case_overrides = {key: value for key, value in case_overrides.items() if key not in {"both", "pce", "apce"}}
        overrides.update(flat_case_overrides)
        overrides.update(case_overrides.get("both", {}))
        overrides.update(case_overrides.get(method, {}))
    else:
        overrides.update(case_overrides)
    if not overrides:
        return config
    return replace(config, **_normalize_config_overrides(overrides))


def make_system(config: CaseConfig) -> HybridSystem:
    return final_figure3_case_spec(config.name).system_factory()


def initial_state(config: CaseConfig, device: torch.device) -> torch.Tensor:
    return final_figure3_case_spec(config.name).initial_state_factory(device)


def initial_scale(config: CaseConfig) -> torch.Tensor:
    return final_figure3_case_spec(config.name).initial_scale_factory()


def observation_indices(config: CaseConfig, device: torch.device) -> torch.Tensor:
    return final_figure3_case_spec(config.name).observation_indices_factory(device)


def primary_indices(config: CaseConfig, state_dim: int, device: torch.device) -> torch.Tensor:
    return torch.arange(state_dim, dtype=torch.int64, device=device)


def step_with_noise(
    system: HybridSystem,
    state: torch.Tensor,
    time_value: float,
    dt: float,
    alpha: float | torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    alpha_tensor = torch.as_tensor(alpha, dtype=state.dtype, device=state.device)
    quantile = liu_quantile(alpha_tensor)
    k1 = system.drift(state, time_value, quantile)
    k2 = system.drift(state + 0.5 * dt * k1, time_value + 0.5 * dt, quantile)
    k3 = system.drift(state + 0.5 * dt * k2, time_value + 0.5 * dt, quantile)
    k4 = system.drift(state + dt * k3, time_value + dt, quantile)
    deterministic = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    stochastic = math.sqrt(dt) * system.diffusion(state, time_value) * noise
    return system.project(deterministic + stochastic)


def generate_scenario(config: CaseConfig, device: torch.device) -> Scenario:
    dtype = torch.float64
    system = make_system(config)
    generator = torch.Generator(device=device).manual_seed(config.seed)
    state0 = initial_state(config, device)
    state_dim = int(state0.numel())
    obs_idx = observation_indices(config, device)
    primary_idx = primary_indices(config, state_dim, device)
    truth_noise = torch.randn((config.steps, state_dim), dtype=dtype, device=device, generator=generator)
    forecast_noise = torch.randn(
        (config.steps, config.ensemble_size, state_dim),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    init_noise = torch.randn((config.ensemble_size, state_dim), dtype=dtype, device=device, generator=generator)
    obs_noise_values = torch.randn(
        (config.steps // config.obs_interval + 2, obs_idx.numel()),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    scale = initial_scale(config).to(device=device, dtype=dtype)
    initial_ensemble = system.project(state0.unsqueeze(0) + scale.unsqueeze(0) * init_noise)
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
    obs_row = 0
    for step in range(config.obs_interval, config.steps + 1, config.obs_interval):
        observations[step] = truth[step, obs_idx] + config.obs_noise * obs_noise_values[obs_row]
        obs_row += 1
    return Scenario(
        config=config,
        truth=truth,
        observations=observations,
        observation_indices=obs_idx,
        primary_indices=primary_idx,
        initial_ensemble=initial_ensemble,
        forecast_noise=forecast_noise,
        alpha_grid=torch.tensor(config.alpha_grid, dtype=dtype, device=device),
    )


def primary(values: torch.Tensor, scenario: Scenario) -> torch.Tensor:
    return values.index_select(-1, scenario.primary_indices)


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


def pce_state_output_weights(
    base_weights: torch.Tensor,
    log_scores: torch.Tensor,
    config: CaseConfig,
) -> torch.Tensor:
    """Separate cognitive identification weights from state-output weights.

    The alpha MAP/log-score trajectory remains the cognitive-coordinate
    estimator.  This layer only controls how sharply the branch ensemble is
    exposed to state metrics and downstream traces.  Defaults are the identity,
    so frozen baseline runs are unchanged.
    """

    weights = base_weights.clamp_min(1.0e-300)
    power = max(float(config.state_weight_power), 1.0e-8)
    if abs(power - 1.0) > 1.0e-12:
        weights = weights.pow(power)
        weights = weights / weights.sum().clamp_min(1.0e-300)
    blend_ceiling = float(config.state_map_blend)
    if blend_ceiling > 0.0 and log_scores.numel() > 1:
        confidence, _ = evidence_gap_confidence(log_scores)
        confidence_power = max(float(config.state_map_blend_confidence_power), 1.0e-8)
        blend = float(np.clip(blend_ceiling * (confidence**confidence_power), 0.0, 1.0))
        if blend > 1.0e-12:
            one_hot = torch.zeros_like(weights)
            one_hot[int(torch.argmax(log_scores))] = 1.0
            weights = (1.0 - blend) * weights + blend * one_hot
    return weights / weights.sum().clamp_min(1.0e-300)


def interpolate_branch_mean(alpha_grid: torch.Tensor, branch_means: torch.Tensor, alpha_value: float) -> torch.Tensor:
    if alpha_grid.numel() == 1:
        return branch_means[0]
    alpha_value = float(np.clip(alpha_value, float(alpha_grid[0]), float(alpha_grid[-1])))
    right = int(torch.searchsorted(alpha_grid, torch.tensor(alpha_value, dtype=alpha_grid.dtype, device=alpha_grid.device)).clamp(1, alpha_grid.numel() - 1))
    left = right - 1
    denom = (alpha_grid[right] - alpha_grid[left]).clamp_min(torch.finfo(alpha_grid.dtype).eps)
    frac = (alpha_value - float(alpha_grid[left])) / float(denom)
    return branch_means[left] * (1.0 - frac) + branch_means[right] * frac


def interpolate_branch_quadratic(alpha_grid: torch.Tensor, branch_means: torch.Tensor, alpha_value: float) -> torch.Tensor:
    if alpha_grid.numel() < 3:
        return interpolate_branch_mean(alpha_grid, branch_means, alpha_value)
    alpha_value = float(np.clip(alpha_value, float(alpha_grid[0]), float(alpha_grid[-1])))
    center = int(torch.argmin((alpha_grid - alpha_value).abs()))
    if center <= 0:
        indices = torch.tensor([0, 1, 2], device=alpha_grid.device)
    elif center >= alpha_grid.numel() - 1:
        indices = torch.tensor([alpha_grid.numel() - 3, alpha_grid.numel() - 2, alpha_grid.numel() - 1], device=alpha_grid.device)
    else:
        indices = torch.tensor([center - 1, center, center + 1], device=alpha_grid.device)
    x = alpha_grid.index_select(0, indices).detach().cpu().numpy()
    y = branch_means.index_select(0, indices).detach().cpu().numpy()
    try:
        coeffs = np.polyfit(x, y, deg=2)
        estimate = np.polyval(coeffs, alpha_value)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return interpolate_branch_mean(alpha_grid, branch_means, alpha_value)
    if not np.all(np.isfinite(estimate)):
        return interpolate_branch_mean(alpha_grid, branch_means, alpha_value)
    return torch.as_tensor(estimate, dtype=branch_means.dtype, device=branch_means.device)


def pce_point_estimate(
    alpha_grid: torch.Tensor,
    branch_means: torch.Tensor,
    state_weights: torch.Tensor,
    log_scores: torch.Tensor,
    config: CaseConfig,
) -> torch.Tensor:
    mode = str(config.state_point_estimator_mode).strip().lower()
    if mode in {"mean", "weighted_mean"}:
        weights = state_weights.to(dtype=branch_means.dtype, device=branch_means.device).clamp_min(1.0e-300)
        weights = weights / weights.sum().clamp_min(1.0e-300)
        return (weights.unsqueeze(-1) * branch_means).sum(dim=0)
    if mode == "map":
        return branch_means[int(torch.argmax(log_scores))]
    if mode in {"interpolated_map", "continuous_map", "map_interp"}:
        alpha_hat = continuous_alpha(alpha_grid, log_scores)
        return interpolate_branch_mean(alpha_grid, branch_means, alpha_hat)
    if mode in {"quadratic_map", "continuous_quadratic_map", "quadratic_interp"}:
        alpha_hat = continuous_alpha(alpha_grid, log_scores)
        return interpolate_branch_quadratic(alpha_grid, branch_means, alpha_hat)
    raise ValueError(f"Unknown state_point_estimator_mode: {config.state_point_estimator_mode!r}")


def collapse_branch_cloud(
    branches: torch.Tensor,
    shadow: torch.Tensor,
    log_scores: torch.Tensor,
    config: CaseConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optionally pull all alpha branches toward the dominant path.

    This is a conservative coherence step: it does not change the benchmark
    inputs, only the internal branch geometry used for state propagation and
    trace export.
    """

    strength = max(float(config.branch_collapse_strength), 0.0)
    if strength <= 0.0 or branches.shape[0] <= 1:
        return branches, shadow
    confidence, _ = evidence_gap_confidence(log_scores)
    confidence_power = max(float(config.branch_collapse_confidence_power), 1.0e-8)
    blend = float(np.clip(strength * (confidence**confidence_power), 0.0, 1.0))
    if blend <= 1.0e-12:
        return branches, shadow
    dominant = int(torch.argmax(log_scores))
    dominant_branch = branches[dominant : dominant + 1]
    dominant_shadow = shadow[dominant : dominant + 1]
    branches = (1.0 - blend) * branches + blend * dominant_branch
    shadow = (1.0 - blend) * shadow + blend * dominant_shadow
    return branches, shadow


def weighted_denkf_analysis(
    state_ensemble: torch.Tensor,
    weights: torch.Tensor,
    observation: torch.Tensor,
    observation_operator: SparseObservation,
    observation_covariance: torch.Tensor,
) -> torch.Tensor:
    """Weighted deterministic EnKF used only inside PCE/APCE branch mixtures."""

    weights = weights.to(dtype=state_ensemble.dtype, device=state_ensemble.device)
    weights = weights.clamp_min(1.0e-300)
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


def augmented_alpha_denkf_analysis(
    state_ensemble: torch.Tensor,
    alpha_ensemble: torch.Tensor,
    observation: torch.Tensor,
    observation_operator: SparseObservation,
    observation_covariance: torch.Tensor,
    *,
    alpha_lower: float,
    alpha_upper: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Joint DEnKF update for state and member-wise alpha inside one branch."""

    if alpha_ensemble.ndim != 1 or alpha_ensemble.shape[0] != state_ensemble.shape[0]:
        raise ValueError("alpha_ensemble must be one-dimensional and match the ensemble size")
    augmented = torch.cat([state_ensemble, alpha_ensemble[:, None]], dim=-1)
    updated = denkf_analysis(augmented, observation, observation_operator, observation_covariance)
    return updated[:, :-1], updated[:, -1].clamp(alpha_lower, alpha_upper)


def weighted_augmented_alpha_denkf_analysis(
    state_ensemble: torch.Tensor,
    alpha_ensemble: torch.Tensor,
    weights: torch.Tensor,
    observation: torch.Tensor,
    observation_operator: SparseObservation,
    observation_covariance: torch.Tensor,
    *,
    alpha_lower: float,
    alpha_upper: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted global state--alpha DEnKF used only inside PCE/APCE.

    This keeps the benchmark fixed but lets the cognitive branch weights guide
    a single cross-alpha state--parameter correction.  The observation
    operator still acts on the physical state coordinates only; the last
    augmented coordinate is the member-wise cognitive parameter.
    """

    if alpha_ensemble.ndim != 1 or alpha_ensemble.shape[0] != state_ensemble.shape[0]:
        raise ValueError("alpha_ensemble must be one-dimensional and match the ensemble size")
    augmented = torch.cat([state_ensemble, alpha_ensemble[:, None]], dim=-1)
    updated = weighted_denkf_analysis(augmented, weights, observation, observation_operator, observation_covariance)
    return updated[:, :-1], updated[:, -1].clamp(alpha_lower, alpha_upper)


def pad_1d_history(history: list[torch.Tensor]) -> np.ndarray:
    if not history:
        return np.empty((0, 0), dtype=float)
    max_len = max(int(item.numel()) for item in history)
    output = np.full((len(history), max_len), np.nan, dtype=float)
    for index, item in enumerate(history):
        values = item.detach().cpu().numpy().reshape(-1)
        output[index, : values.size] = values
    return output


def history_lengths(history: list[torch.Tensor]) -> np.ndarray:
    return np.asarray([int(item.numel()) for item in history], dtype=np.int64)


def analysis_generator(config: CaseConfig, step: int, device: torch.device) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(config.seed * 100_000 + step)


def apply_analysis(
    method: MethodName,
    ensemble: torch.Tensor,
    observation: torch.Tensor,
    operator: SparseObservation,
    covariance: torch.Tensor,
    config: CaseConfig,
    step: int,
    device: torch.device,
) -> torch.Tensor:
    if method == "denkf":
        return denkf_analysis(ensemble, observation, operator, covariance)
    if method == "letkf":
        return letkf_analysis(ensemble, observation, operator, covariance)
    if method == "iensf":
        return iensf_analysis(
            ensemble,
            observation,
            operator,
            covariance,
            IEnSFConfig(sampling_time_step_count=20, refinement_iterations=2),
            analysis_generator(config, step, device),
        )
    if method == "ensf":
        return ensf_analysis(
            ensemble,
            observation,
            operator,
            covariance,
            EnSFConfig(sampling_time_step_count=20),
            analysis_generator(config, step, device),
        )
    if method == "ensf_lr":
        return ensf_lr_analysis(
            ensemble,
            observation,
            operator,
            covariance,
            EnSFConfig(sampling_time_step_count=20),
            analysis_generator(config, step, device),
        )
    if method == "ensf_lr_ridge":
        return ensf_lr_ridge_analysis(
            ensemble,
            observation,
            operator,
            covariance,
            EnSFConfig(sampling_time_step_count=20),
            analysis_generator(config, step, device),
        )
    raise ValueError(method)


def estimate_physical_metrics(config: CaseConfig, estimate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    nonnegative_cases = {
        "chemical",
        "pk_infusion",
        "sir",
        "sis",
        "seiar",
        "logistic",
        "gordon_schaefer",
        "lotka_volterra",
        "robertson",
    }
    positivity_violation_rate = float(np.mean(estimate < -1.0e-8)) if config.name in nonnegative_cases else 0.0
    if config.name == "chemical":
        invariant = estimate[:, 0] + 2.0 * estimate[:, 1]
        truth_invariant = truth[:, 0] + 2.0 * truth[:, 1]
        return {
            "positivity_violation_rate": positivity_violation_rate,
            "physical_validity_error": float(np.mean(np.abs(invariant - truth_invariant))),
            "peak_time_error": 0.0,
            "auc_relative_error": 0.0,
        }
    if config.name == "pk_infusion":
        est_x = estimate[:, 0]
        truth_x = truth[:, 0]
        peak_time_error = abs(int(np.argmax(est_x)) - int(np.argmax(truth_x))) * config.dt
        auc_truth = float(np.trapz(truth_x, dx=config.dt))
        auc_est = float(np.trapz(est_x, dx=config.dt))
        return {
            "positivity_violation_rate": positivity_violation_rate,
            "physical_validity_error": abs(auc_est - auc_truth) / max(abs(auc_truth), 1.0e-12),
            "peak_time_error": float(peak_time_error),
            "auc_relative_error": abs(auc_est - auc_truth) / max(abs(auc_truth), 1.0e-12),
        }
    if config.name in {"sir", "sis", "seiar"}:
        conservation = np.abs(estimate.sum(axis=1) - 1.0)
        infected_index = 2 if config.name == "seiar" else 1
        peak_time_error = abs(int(np.argmax(estimate[:, infected_index])) - int(np.argmax(truth[:, infected_index]))) * config.dt
        return {
            "positivity_violation_rate": positivity_violation_rate,
            "physical_validity_error": float(np.mean(conservation)),
            "peak_time_error": float(peak_time_error),
            "auc_relative_error": 0.0,
        }
    if config.name in {"logistic", "gordon_schaefer"}:
        capacity = 1.0
        boundedness_error = np.maximum(estimate[:, 0] - capacity, 0.0) + np.maximum(-estimate[:, 0], 0.0)
        peak_time_error = abs(int(np.argmax(estimate[:, 0])) - int(np.argmax(truth[:, 0]))) * config.dt
        return {
            "positivity_violation_rate": positivity_violation_rate,
            "physical_validity_error": float(np.mean(boundedness_error)),
            "peak_time_error": float(peak_time_error),
            "auc_relative_error": 0.0,
        }
    if config.name == "robertson":
        conservation = np.abs(estimate.sum(axis=1) - 1.0)
        peak_time_error = abs(int(np.argmax(estimate[:, 1])) - int(np.argmax(truth[:, 1]))) * config.dt
        return {
            "positivity_violation_rate": positivity_violation_rate,
            "physical_validity_error": float(np.mean(conservation)),
            "peak_time_error": float(peak_time_error),
            "auc_relative_error": 0.0,
        }
    if config.name in {"rl_circuit", "van_der_pol", "duffing", "lotka_volterra", "fhn", "pendulum"}:
        amplitude = np.linalg.norm(estimate, axis=1)
        truth_amplitude = np.linalg.norm(truth, axis=1)
        peak_time_error = abs(int(np.argmax(amplitude)) - int(np.argmax(truth_amplitude))) * config.dt
        amplitude_error = np.mean(np.abs(amplitude - truth_amplitude)) / max(float(np.mean(np.abs(truth_amplitude))), 1.0e-12)
        return {
            "positivity_violation_rate": positivity_violation_rate,
            "physical_validity_error": float(amplitude_error),
            "peak_time_error": float(peak_time_error),
            "auc_relative_error": 0.0,
        }
    raise ValueError(config.name)


def run_fixed_method(
    scenario: Scenario,
    method: MethodName,
    device: torch.device,
    *,
    record_trace: bool = False,
) -> dict[str, Any]:
    config = scenario.config
    system = make_system(config)
    ensemble = scenario.initial_ensemble.clone()
    operator = SparseObservation(scenario.observation_indices)
    covariance = config.obs_noise**2 * torch.eye(scenario.observation_indices.numel(), dtype=ensemble.dtype, device=device)
    metrics = TrajectoryMetrics()
    weights = torch.full((config.ensemble_size,), 1.0 / config.ensemble_size, dtype=ensemble.dtype, device=device)
    estimates: list[torch.Tensor] = []
    trace_states: list[torch.Tensor] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(config.steps + 1):
        mean_state = ensemble.mean(dim=0)
        estimates.append(mean_state.detach().cpu())
        if record_trace:
            trace_states.append(mean_state.detach().cpu())
        metrics.add(primary(ensemble, scenario), primary(scenario.truth[step], scenario), weights)
        if step == config.steps:
            break
        ensemble = step_with_noise(system, ensemble, step * config.dt, config.dt, config.fixed_alpha, scenario.forecast_noise[step])
        if step + 1 not in scenario.observations:
            continue
        observation = scenario.observations[step + 1]
        ensemble = apply_analysis(method, ensemble, observation, operator, covariance, config, step + 1, device)
        ensemble = system.project(ensemble)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = metrics.finalize()
    estimate_array = torch.stack(estimates).numpy()
    result.update(
        runtime_seconds=float(time.perf_counter() - started),
        forward_member_steps=config.steps * config.ensemble_size,
        peak_gpu_memory_mb=float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
        alpha_estimate=float(config.fixed_alpha),
        alpha_absolute_error=abs(config.fixed_alpha - config.alpha_true),
        max_abs_state=float(np.max(np.abs(estimate_array))),
        **estimate_physical_metrics(config, estimate_array, scenario.truth.detach().cpu().numpy()),
    )
    if record_trace:
        result["mean_states"] = torch.stack(trace_states).numpy()
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
    alpha_grid = scenario.alpha_grid.clone()
    global_bounds = (float(scenario.alpha_grid[0]), float(scenario.alpha_grid[-1]))
    path_count = int(alpha_grid.numel())
    branches = scenario.initial_ensemble.unsqueeze(0).repeat(path_count, 1, 1)
    shadow = branches.clone()
    alpha_log_scores = torch.zeros(path_count, dtype=branches.dtype, device=device)
    alpha_members = torch.stack(
        [
            branch_member_alpha_cloud(
                alpha,
                alpha_grid=alpha_grid,
                ensemble_size=config.ensemble_size,
                jitter_fraction=float(config.branch_member_alpha_jitter),
                confidence_power=float(config.branch_member_alpha_jitter_confidence_power),
                log_scores=alpha_log_scores,
            )
            for alpha in alpha_grid
        ],
        dim=0,
    )
    alpha_weights = torch.softmax(alpha_log_scores, dim=0)
    state_weights = pce_state_output_weights(alpha_weights, alpha_log_scores, config)
    operator = SparseObservation(scenario.observation_indices)
    covariance = config.obs_noise**2 * torch.eye(scenario.observation_indices.numel(), dtype=branches.dtype, device=device)
    metrics = TrajectoryMetrics()
    estimates: list[torch.Tensor] = []
    trace_states: list[torch.Tensor] = []
    trace_weights: list[torch.Tensor] = []
    trace_alpha_grid: list[torch.Tensor] = []
    trace_alpha_estimate: list[float] = []
    regrid_count = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    forward_member_steps = 0
    for step in range(config.steps + 1):
        path_count = int(alpha_grid.numel())
        flat = branches.reshape(-1, branches.shape[-1])
        flat_weights = state_weights.unsqueeze(1).expand(-1, config.ensemble_size).reshape(-1) / config.ensemble_size
        branch_means = branches.mean(dim=1)
        estimate = pce_point_estimate(alpha_grid, branch_means, state_weights, alpha_log_scores, config)
        estimates.append(estimate.detach().cpu())
        if record_trace:
            trace_states.append(estimate.detach().cpu())
            trace_weights.append(state_weights.detach().cpu())
            trace_alpha_grid.append(alpha_grid.detach().cpu())
            trace_alpha_estimate.append(continuous_alpha(alpha_grid, alpha_log_scores))
        metrics.add(
            primary(flat, scenario),
            primary(scenario.truth[step], scenario),
            flat_weights,
            point_estimate=primary(estimate.unsqueeze(0), scenario).squeeze(0),
        )
        if step == config.steps:
            break
        for path_index in range(path_count):
            member_alpha = alpha_members[path_index]
            branches[path_index] = step_with_noise(
                system,
                branches[path_index],
                step * config.dt,
                config.dt,
                member_alpha,
                scenario.forecast_noise[step],
            )
            shadow[path_index] = step_with_noise(
                system,
                shadow[path_index],
                step * config.dt,
                config.dt,
                member_alpha,
                scenario.forecast_noise[step],
            )
        forward_member_steps += 2 * path_count * config.ensemble_size
        if step + 1 not in scenario.observations:
            continue
        observation = scenario.observations[step + 1]
        shadow_observations = torch.stack([operator(path_branch) for path_branch in shadow])
        dimension_weights = None
        if method == "apce":
            between = shadow_observations.mean(dim=1).var(dim=0, unbiased=True)
            dimension_weights = config.dimension_weight_floor + config.dimension_weight_gain * between / between.max().clamp_min(1.0e-12)
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
            state_weights = pce_state_output_weights(alpha_weights, alpha_log_scores, config)
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
            calibrated_weights = entropy_project(alpha_weights, calibration.entropy_floor)
            state_weights = pce_state_output_weights(calibrated_weights, alpha_log_scores, config)
        else:
            raise ValueError(method)
        refined_grid = torch_local_alpha_grid(
            alpha_grid,
            alpha_log_scores,
            points=config.local_grid_points,
            bounds=global_bounds,
            topk=config.local_grid_topk,
            min_spacing=config.local_grid_min_spacing,
        )
        if refined_grid.shape != alpha_grid.shape or not torch.allclose(refined_grid, alpha_grid):
            branches = torch_regrid_paths(alpha_grid, branches, refined_grid)
            shadow = torch_regrid_paths(alpha_grid, shadow, refined_grid)
            alpha_members = torch_regrid_paths(alpha_grid, alpha_members, refined_grid)
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
                calibrated_weights = entropy_project(alpha_weights, calibration.entropy_floor)
                state_weights = pce_state_output_weights(calibrated_weights, alpha_log_scores, config)
            else:
                state_weights = pce_state_output_weights(alpha_weights, alpha_log_scores, config)
            alpha_grid = refined_grid
            path_count = int(alpha_grid.numel())
            regrid_count += 1
        if method in {"pce", "apce"} and float(config.branch_collapse_strength) > 0.0:
            branches, shadow = collapse_branch_cloud(branches, shadow, alpha_log_scores, config)
        analysis_iterations = max(int(config.analysis_iterations), 1)
        global_strength = float(np.clip(config.global_analysis_strength, 0.0, 1.0))
        alpha_aug_strength = float(np.clip(config.branch_augmented_alpha_analysis_strength, 0.0, 1.0))
        global_alpha_aug_strength = float(np.clip(config.global_augmented_alpha_analysis_strength, 0.0, 1.0))
        if global_strength > 1.0e-12:
            confidence, _ = evidence_gap_confidence(alpha_log_scores)
            global_strength *= float(np.clip(confidence ** max(float(config.global_analysis_confidence_power), 1.0e-8), 0.0, 1.0))
        for _ in range(analysis_iterations):
            forecast_branches = branches
            local_branches = torch.empty_like(forecast_branches)
            for path_index in range(int(alpha_grid.numel())):
                local_branches[path_index] = denkf_analysis(forecast_branches[path_index], observation, operator, covariance)
            if global_strength > 1.0e-12:
                flat_forecast = forecast_branches.reshape(-1, forecast_branches.shape[-1])
                flat_analysis_weights = (
                    state_weights.unsqueeze(1).expand(-1, config.ensemble_size).reshape(-1) / config.ensemble_size
                )
                global_branches = weighted_denkf_analysis(
                    flat_forecast,
                    flat_analysis_weights,
                    observation,
                    operator,
                    covariance,
                ).reshape_as(forecast_branches)
                branches = (1.0 - global_strength) * local_branches + global_strength * global_branches
            else:
                branches = local_branches
            if alpha_aug_strength > 1.0e-12:
                joint_branches = torch.empty_like(branches)
                joint_alpha_members = torch.empty_like(alpha_members)
                for path_index in range(int(alpha_grid.numel())):
                    aug_branch, aug_alpha = augmented_alpha_denkf_analysis(
                        branches[path_index],
                        alpha_members[path_index],
                        observation,
                        operator,
                        covariance,
                        alpha_lower=global_bounds[0],
                        alpha_upper=global_bounds[1],
                    )
                    joint_branches[path_index] = aug_branch
                    joint_alpha_members[path_index] = aug_alpha
                branches = (1.0 - alpha_aug_strength) * branches + alpha_aug_strength * joint_branches
                alpha_members = (1.0 - alpha_aug_strength) * alpha_members + alpha_aug_strength * joint_alpha_members
            if global_alpha_aug_strength > 1.0e-12:
                flat_branches = branches.reshape(-1, branches.shape[-1])
                flat_alpha_members = alpha_members.reshape(-1)
                flat_analysis_weights = (
                    state_weights.unsqueeze(1).expand(-1, config.ensemble_size).reshape(-1) / config.ensemble_size
                )
                global_joint_branches, global_joint_alpha = weighted_augmented_alpha_denkf_analysis(
                    flat_branches,
                    flat_alpha_members,
                    flat_analysis_weights,
                    observation,
                    operator,
                    covariance,
                    alpha_lower=global_bounds[0],
                    alpha_upper=global_bounds[1],
                )
                global_joint_branches = global_joint_branches.reshape_as(branches)
                global_joint_alpha = global_joint_alpha.reshape_as(alpha_members)
                branches = (1.0 - global_alpha_aug_strength) * branches + global_alpha_aug_strength * global_joint_branches
                alpha_members = (
                    (1.0 - global_alpha_aug_strength) * alpha_members
                    + global_alpha_aug_strength * global_joint_alpha
                )
            for path_index in range(int(alpha_grid.numel())):
                branches[path_index] = system.project(branches[path_index])
                alpha_members[path_index] = alpha_members[path_index].clamp(global_bounds[0], global_bounds[1])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = metrics.finalize()
    estimate_array = torch.stack(estimates).numpy()
    alpha_estimate = continuous_alpha(alpha_grid, alpha_log_scores)
    result.update(
        runtime_seconds=float(time.perf_counter() - started),
        forward_member_steps=int(forward_member_steps),
        peak_gpu_memory_mb=float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
        alpha_estimate=alpha_estimate,
        alpha_absolute_error=abs(alpha_estimate - config.alpha_true),
        alpha_map=float(alpha_grid[int(torch.argmax(alpha_log_scores))]),
        alpha_final_entropy=float(entropy(state_weights)),
        alpha_evidence_entropy=float(entropy(alpha_weights)),
        alpha_regrid_count=int(regrid_count),
        alpha_grid_min=float(alpha_grid.min().detach().cpu()),
        alpha_grid_max=float(alpha_grid.max().detach().cpu()),
        alpha_grid_points=int(alpha_grid.numel()),
        max_abs_state=float(np.max(np.abs(estimate_array))),
        **estimate_physical_metrics(config, estimate_array, scenario.truth.detach().cpu().numpy()),
    )
    if record_trace:
        result["mean_states"] = torch.stack(trace_states).numpy()
        result["alpha_weight_history"] = pad_1d_history(trace_weights)
        result["alpha_weight_history_lengths"] = history_lengths(trace_weights)
        result["alpha_grid_history"] = pad_1d_history(trace_alpha_grid)
        result["alpha_grid_history_lengths"] = history_lengths(trace_alpha_grid)
        result["alpha_estimate_history"] = np.asarray(trace_alpha_estimate, dtype=float)
    return result


def tile_alpha_members(
    alpha_grid: torch.Tensor,
    ensemble_size: int,
    generator: torch.Generator,
    jitter: float = 0.012,
) -> torch.Tensor:
    repeats = int(math.ceil(ensemble_size / int(alpha_grid.numel())))
    tiled = alpha_grid.repeat(repeats)[:ensemble_size].clone()
    if jitter > 0.0:
        tiled = tiled + jitter * torch.randn(
            (ensemble_size,),
            dtype=alpha_grid.dtype,
            device=alpha_grid.device,
            generator=generator,
        )
    return tiled.clamp(float(alpha_grid[0]), float(alpha_grid[-1]))


def member_alpha_offsets(ensemble_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if ensemble_size <= 1:
        return torch.zeros((ensemble_size,), dtype=dtype, device=device)
    offsets = torch.linspace(-1.0, 1.0, ensemble_size, dtype=dtype, device=device)
    offsets = offsets - offsets.mean()
    return offsets / offsets.abs().max().clamp_min(1.0e-12)


def branch_member_alpha_values(
    alpha: torch.Tensor,
    *,
    alpha_grid: torch.Tensor,
    ensemble_size: int,
    jitter_fraction: float,
    confidence_power: float,
    log_scores: torch.Tensor,
) -> torch.Tensor:
    alpha_tensor = torch.as_tensor(alpha, dtype=alpha_grid.dtype, device=alpha_grid.device)
    if ensemble_size <= 1 or jitter_fraction <= 0.0:
        return alpha_tensor
    confidence, _ = evidence_gap_confidence(log_scores)
    effective_fraction = jitter_fraction * float((1.0 - confidence) ** max(confidence_power, 1.0e-8))
    if effective_fraction <= 1.0e-12:
        return alpha_tensor
    lower = float(alpha_grid[0])
    upper = float(alpha_grid[-1])
    span = max(upper - lower, 1.0e-12)
    offsets = member_alpha_offsets(ensemble_size, alpha_grid.device, alpha_grid.dtype)
    return (alpha_tensor + effective_fraction * span * offsets).clamp(lower, upper)


def branch_member_alpha_cloud(
    alpha: torch.Tensor,
    *,
    alpha_grid: torch.Tensor,
    ensemble_size: int,
    jitter_fraction: float,
    confidence_power: float,
    log_scores: torch.Tensor,
) -> torch.Tensor:
    cloud = branch_member_alpha_values(
        alpha,
        alpha_grid=alpha_grid,
        ensemble_size=ensemble_size,
        jitter_fraction=jitter_fraction,
        confidence_power=confidence_power,
        log_scores=log_scores,
    )
    if cloud.ndim == 0:
        return cloud.expand(ensemble_size).clone()
    return cloud


def random_walk_alpha(
    alpha: torch.Tensor,
    generator: torch.Generator,
    lower: float,
    upper: float,
    std: float = 0.004,
) -> torch.Tensor:
    updated = alpha + std * torch.randn(alpha.shape, dtype=alpha.dtype, device=alpha.device, generator=generator)
    return updated.clamp(lower, upper)


def run_aug_enkf_method(
    scenario: Scenario,
    device: torch.device,
    *,
    record_trace: bool = False,
) -> dict[str, Any]:
    config = scenario.config
    system = make_system(config)
    ensemble = scenario.initial_ensemble.clone()
    generator = torch.Generator(device=device).manual_seed(config.seed * 1000 + 741_001)
    alpha = tile_alpha_members(scenario.alpha_grid, config.ensemble_size, generator, jitter=0.012)
    operator = SparseObservation(scenario.observation_indices)
    covariance = config.obs_noise**2 * torch.eye(
        scenario.observation_indices.numel(), dtype=ensemble.dtype, device=device
    )
    weights = torch.full((config.ensemble_size,), 1.0 / config.ensemble_size, dtype=ensemble.dtype, device=device)
    metrics = TrajectoryMetrics()
    estimates: list[torch.Tensor] = []
    trace_states: list[torch.Tensor] = []
    trace_alpha: list[torch.Tensor] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(config.steps + 1):
        mean_state = ensemble.mean(dim=0)
        estimates.append(mean_state.detach().cpu())
        if record_trace:
            trace_states.append(mean_state.detach().cpu())
            trace_alpha.append(alpha.mean().detach().cpu())
        metrics.add(primary(ensemble, scenario), primary(scenario.truth[step], scenario), weights)
        if step == config.steps:
            break
        alpha = random_walk_alpha(
            alpha,
            generator,
            lower=float(scenario.alpha_grid[0]),
            upper=float(scenario.alpha_grid[-1]),
            std=0.004,
        )
        ensemble = step_with_noise(
            system,
            ensemble,
            step * config.dt,
            config.dt,
            alpha,
            scenario.forecast_noise[step],
        )
        if step + 1 not in scenario.observations:
            continue
        augmented = torch.cat([ensemble, alpha[:, None]], dim=-1)
        updated = denkf_analysis(augmented, scenario.observations[step + 1], operator, covariance)
        ensemble = system.project(updated[:, :-1])
        alpha = updated[:, -1].clamp(float(scenario.alpha_grid[0]), float(scenario.alpha_grid[-1]))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = metrics.finalize()
    estimate_array = torch.stack(estimates).numpy()
    alpha_estimate = float(alpha.mean())
    result.update(
        runtime_seconds=float(time.perf_counter() - started),
        forward_member_steps=config.steps * config.ensemble_size,
        peak_gpu_memory_mb=float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
        alpha_estimate=alpha_estimate,
        alpha_absolute_error=abs(alpha_estimate - config.alpha_true),
        alpha_spread=float(alpha.std(unbiased=True)) if alpha.numel() > 1 else 0.0,
        max_abs_state=float(np.max(np.abs(estimate_array))),
        **estimate_physical_metrics(config, estimate_array, scenario.truth.detach().cpu().numpy()),
    )
    if record_trace:
        result["mean_states"] = torch.stack(trace_states).numpy()
        result["alpha_member_history"] = torch.stack(trace_alpha).numpy()
    return result


def run_bma_static_method(
    scenario: Scenario,
    device: torch.device,
    *,
    record_trace: bool = False,
) -> dict[str, Any]:
    config = scenario.config
    system = make_system(config)
    path_count = scenario.alpha_grid.numel()
    branches = scenario.initial_ensemble.unsqueeze(0).repeat(path_count, 1, 1)
    log_weights = torch.zeros(path_count, dtype=branches.dtype, device=device)
    path_weights = torch.softmax(log_weights, dim=0)
    operator = SparseObservation(scenario.observation_indices)
    covariance = config.obs_noise**2 * torch.eye(
        scenario.observation_indices.numel(), dtype=branches.dtype, device=device
    )
    metrics = TrajectoryMetrics()
    estimates: list[torch.Tensor] = []
    trace_states: list[torch.Tensor] = []
    trace_weights: list[torch.Tensor] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(config.steps + 1):
        flat = branches.reshape(-1, branches.shape[-1])
        flat_weights = path_weights.unsqueeze(1).expand(-1, config.ensemble_size).reshape(-1) / config.ensemble_size
        branch_means = branches.mean(dim=1)
        estimate = (path_weights.unsqueeze(-1) * branch_means).sum(dim=0)
        estimates.append(estimate.detach().cpu())
        if record_trace:
            trace_states.append(estimate.detach().cpu())
            trace_weights.append(path_weights.detach().cpu())
        metrics.add(primary(flat, scenario), primary(scenario.truth[step], scenario), flat_weights)
        if step == config.steps:
            break
        for path_index, alpha in enumerate(scenario.alpha_grid):
            branches[path_index] = step_with_noise(
                system,
                branches[path_index],
                step * config.dt,
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
    result = metrics.finalize()
    estimate_array = torch.stack(estimates).numpy()
    alpha_estimate = continuous_alpha(scenario.alpha_grid, path_weights)
    result.update(
        runtime_seconds=float(time.perf_counter() - started),
        forward_member_steps=config.steps * path_count * config.ensemble_size,
        peak_gpu_memory_mb=float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
        alpha_estimate=alpha_estimate,
        alpha_absolute_error=abs(alpha_estimate - config.alpha_true),
        alpha_final_entropy=float(entropy(path_weights)),
        alpha_map=float(scenario.alpha_grid[int(torch.argmax(path_weights))]),
        max_abs_state=float(np.max(np.abs(estimate_array))),
        **estimate_physical_metrics(config, estimate_array, scenario.truth.detach().cpu().numpy()),
    )
    if record_trace:
        result["mean_states"] = torch.stack(trace_states).numpy()
        result["alpha_weight_history"] = torch.stack(trace_weights).numpy()
    return result


def run_method(scenario: Scenario, method: MethodName, device: torch.device, *, record_trace: bool = False) -> dict[str, Any]:
    if method in {"pce", "apce"}:
        return run_pce_method(scenario, method, device, record_trace=record_trace)
    if method == "aug_enkf":
        return run_aug_enkf_method(scenario, device, record_trace=record_trace)
    if method == "bma_static":
        return run_bma_static_method(scenario, device, record_trace=record_trace)
    return run_fixed_method(scenario, method, device, record_trace=record_trace)


def classify_numerical_status(row: dict[str, Any], truth: torch.Tensor, config: CaseConfig) -> str:
    keys = ("nrmse", "rmse", "crps", "coverage_90", "interval_width_90", "max_abs_state")
    if not all(math.isfinite(float(row[key])) for key in keys):
        return "nonfinite"
    truth_scale = float(torch.max(torch.abs(truth)).detach().cpu())
    if truth_scale > 0 and float(row["max_abs_state"]) > config.max_valid_amplitude_ratio * truth_scale:
        return "diverged"
    if float(row["positivity_violation_rate"]) > 0.02:
        return "physical_violation"
    if float(row["nrmse"]) > 1.0 or float(row["physical_validity_error"]) > 1.0:
        return "bounded_poor"
    return "valid"


def run_one(
    case: CaseName,
    method: MethodName,
    seed: int,
    device: torch.device,
    record_trace: bool,
    tuning_profile: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    config = apply_tuning_profile(config_for_case(case, seed), tuning_profile, case, method)
    scenario = generate_scenario(config, device)
    result = run_method(scenario, method, device, record_trace=record_trace)
    core_runtime = float(result.get("runtime_seconds", 0.0))
    status = classify_numerical_status(result, scenario.truth, config)
    row = {
        "case": case,
        "method": method,
        "label": METHOD_LABELS[method],
        "tuning_profile": tuning_profile.get("name", "baseline"),
        "seed": seed,
        "state_dim": int(scenario.truth.shape[-1]),
        "observed_variables": ",".join(str(int(i)) for i in scenario.observation_indices.detach().cpu().tolist()),
        "observation_count": int(scenario.observation_indices.numel()),
        "observation_noise": config.obs_noise,
        "dt": config.dt,
        "steps": config.steps,
        "assimilation_interval": config.obs_interval,
        "ensemble_size": config.ensemble_size,
        "alpha_true": config.alpha_true,
        "alpha_grid": ",".join(f"{value:.4g}" for value in config.alpha_grid),
        "numerical_status": status,
        "status": "completed",
        "core_runtime_seconds": core_runtime,
        "trace_io_runtime_seconds": 0.0,
        "elapsed_seconds_wall": float(time.perf_counter() - started),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        **{key: value for key, value in result.items() if not isinstance(value, np.ndarray)},
    }
    if record_trace:
        observation_steps = np.asarray(sorted(scenario.observations), dtype=np.int64)
        observation_values = np.stack(
            [scenario.observations[int(step)].detach().cpu().numpy() for step in observation_steps],
            axis=0,
        )
        row["_trace_payload"] = {
            key: value for key, value in result.items() if isinstance(value, np.ndarray)
        }
        row["_trace_payload"].update(
            initial_ensemble=scenario.initial_ensemble.detach().cpu().numpy(),
            forecast_noise=scenario.forecast_noise.detach().cpu().numpy(),
            truth_states=scenario.truth.detach().cpu().numpy(),
            observation_steps=observation_steps,
            observation_values=observation_values,
            observation_indices=scenario.observation_indices.detach().cpu().numpy(),
            primary_indices=scenario.primary_indices.detach().cpu().numpy(),
            alpha_grid=scenario.alpha_grid.detach().cpu().numpy(),
            config_json=np.asarray(json.dumps(asdict(config), ensure_ascii=False)),
        )
    return row


def run_json_path(
    output: Path,
    case: CaseName,
    method: MethodName,
    seed: int,
    device: torch.device,
    *,
    force: bool,
    record_trace: bool,
    tuning_profile: dict[str, Any],
) -> dict[str, Any]:
    if output.is_file() and not force:
        row = json.loads(output.read_text(encoding="utf-8"))
        trace_output = output.with_suffix(".npz")
        if (
            row.get("status") == "completed"
            and row.get("tuning_profile", "baseline") == tuning_profile.get("name", "baseline")
            and (not record_trace or trace_output.exists())
        ):
            return row
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        row = run_one(case, method, seed, device, record_trace=record_trace, tuning_profile=tuning_profile)
        serializable = {key: value for key, value in row.items() if key != "_trace_payload"}
        trace_payload = row.get("_trace_payload")
        trace_io_started = time.perf_counter()
        if record_trace and isinstance(trace_payload, dict):
            trace_output = output.with_suffix(".npz")
            np.savez_compressed(trace_output, **trace_payload)
        trace_io_seconds = float(time.perf_counter() - trace_io_started)
        row["trace_io_runtime_seconds"] = trace_io_seconds
        row["runtime_seconds"] = float(row.get("core_runtime_seconds", row.get("runtime_seconds", 0.0))) + trace_io_seconds
        serializable["trace_io_runtime_seconds"] = trace_io_seconds
        serializable["runtime_seconds"] = row["runtime_seconds"]
    except Exception as exc:
        serializable = {
            "case": case,
            "method": method,
            "seed": seed,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        output.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
        raise
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(output)
    return row


def bootstrap_ci(values: np.ndarray, seed: int, n_bootstrap: int = 5000) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, values.size, size=(n_bootstrap, values.size))].mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def summarize(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for case in sorted({row["case"] for row in records}):
        for method in sorted({row["method"] for row in records if row["case"] == case}):
            subset = [row for row in records if row["case"] == case and row["method"] == method and row["numerical_status"] == "valid"]
            item: dict[str, Any] = {
                "case": case,
                "method": method,
                "label": METHOD_LABELS[method],
                "n": len(subset),
                "seeds": ",".join(str(row["seed"]) for row in subset),
            }
            if not subset:
                item["valid"] = False
                summary.append(item)
                continue
            item["valid"] = True
            for key in (
                "nrmse",
                "rmse",
                "crps",
                "coverage_90",
                "interval_width_90",
                "alpha_absolute_error",
                "physical_validity_error",
                "peak_time_error",
                "auc_relative_error",
                "core_runtime_seconds",
                "trace_io_runtime_seconds",
                "runtime_seconds",
                "peak_gpu_memory_mb",
            ):
                values = np.asarray([float(row[key]) for row in subset], dtype=float)
                item[f"{key}_mean"] = float(values.mean())
                item[f"{key}_sd"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
                item[f"{key}_ci95"] = bootstrap_ci(values, seed + len(summary)) if values.size > 1 else [float(values[0]), float(values[0])]
            summary.append(item)
    return summary


def sign_test_p_less(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0.0]
    n = int(nonzero.size)
    if n == 0:
        return 1.0
    wins = int(np.sum(nonzero < 0.0))
    return float(sum(math.comb(n, k) for k in range(wins, n + 1)) / (2**n))


def holm_adjust(rows: list[dict[str, Any]]) -> None:
    indexed = sorted(enumerate(rows), key=lambda item: float(item[1]["p_raw"]))
    m = len(indexed)
    previous = 0.0
    for rank, (index, row) in enumerate(indexed):
        adjusted = min(1.0, max(previous, (m - rank) * float(row["p_raw"])))
        rows[index]["p_holm"] = adjusted
        previous = adjusted


def paired_comparisons(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metrics = ("nrmse", "crps", "coverage_90", "interval_width_90", "physical_validity_error", "runtime_seconds")
    for case in sorted({row["case"] for row in records}):
        case_rows = [row for row in records if row["case"] == case and row["numerical_status"] == "valid"]
        methods = sorted({row["method"] for row in case_rows})
        for reference in ("apce", "pce"):
            if reference not in methods:
                continue
            baselines = [method for method in methods if method != reference]
            for baseline in baselines:
                ref_by_seed = {row["seed"]: row for row in case_rows if row["method"] == reference}
                base_by_seed = {row["seed"]: row for row in case_rows if row["method"] == baseline}
                paired = sorted(set(ref_by_seed) & set(base_by_seed))
                if not paired:
                    continue
                for metric in metrics:
                    diff = np.asarray([float(ref_by_seed[s][metric]) - float(base_by_seed[s][metric]) for s in paired], dtype=float)
                    row = {
                        "case": case,
                        "reference": reference,
                        "baseline": baseline,
                        "metric": metric,
                        "n": int(diff.size),
                        "mean_difference_reference_minus_baseline": float(diff.mean()),
                        "ci95_low": bootstrap_ci(diff, seed + len(rows))[0] if diff.size > 1 else float(diff[0]),
                        "ci95_high": bootstrap_ci(diff, seed + len(rows))[1] if diff.size > 1 else float(diff[0]),
                        "p_raw": sign_test_p_less(diff),
                        "paired_seeds": ",".join(str(s) for s in paired),
                    }
                    rows.append(row)
    holm_adjust(rows)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hash_manifest() -> dict[str, Any]:
    files: dict[str, str] = {}
    for path in SOURCE_HASH_FILES:
        if path.is_file():
            files[path.relative_to(PROJECT_ROOT).as_posix()] = file_sha256(path)
        else:
            files[path.relative_to(PROJECT_ROOT).as_posix()] = "MISSING"
    encoded = json.dumps(files, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return {
        "aggregate_sha256": hashlib.sha256(encoded).hexdigest(),
        "files": files,
    }


def write_manifest(output: Path, args: argparse.Namespace, records: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    registry_metadata = {
        case: {
            "family": final_figure3_case_spec(case).family,
            "tier": final_figure3_case_spec(case).tier,
            "state_dim": final_figure3_case_spec(case).state_dim,
            "observation_dim": final_figure3_case_spec(case).observation_dim,
            "cognitive_parameter": final_figure3_case_spec(case).cognitive_parameter,
            "source_metadata": final_figure3_case_spec(case).source_metadata,
        }
        for case in args.cases
    }
    payload = {
        "protocol": "figure3-final-hybrid-ode-v4",
        "stage": args.stage,
        "command_line": sys.argv,
        "python_executable": sys.executable,
        "cases": args.cases,
        "methods": args.methods,
        "n_seeds": args.n_seeds,
        "base_seed": args.base_seed,
        "tuning_matrix": str(args.tuning_matrix),
        "tuning_profile": args.tuning_profile,
        "tuning_profile_payload": args.tuning_profile_payload,
        "source_hash": source_hash_manifest(),
        "case_pool_size": len(CASES),
        "method_pool_size": len(ADMISSION_METHODS),
        "registry_metadata": registry_metadata,
        "records": len(records),
        "valid_records": sum(1 for row in records if row.get("numerical_status") == "valid"),
        "case_configs": {case: asdict(config_for_case(case, args.base_seed)) for case in args.cases},
        "tuning_case_configs": {
            case: {
                method: asdict(apply_tuning_profile(config_for_case(case, args.base_seed), args.tuning_profile_payload, case, method))
                for method in ("pce", "apce")
            }
            for case in args.cases
        },
        "summary": summary,
    }
    (output / "figure3_v4_config_manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_representative_npz(output: Path, trace_rows: list[dict[str, Any]]) -> None:
    if not trace_rows:
        return
    arrays: dict[str, np.ndarray] = {}
    metadata: list[dict[str, Any]] = []
    for row in trace_rows:
        payload = row.get("_trace_payload", {})
        case = row["case"]
        method = row["method"]
        seed = row["seed"]
        for key, value in payload.items():
            arrays[f"{case}_{method}_s{seed}_{key}"] = value
        metadata.append({key: row[key] for key in ("case", "method", "seed", "label") if key in row})
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
    np.savez_compressed(output / "figure3_v4_representative_traces.npz", **arrays)


def parse_csv_choices(value: str, allowed: tuple[str, ...]) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(items) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown choices: {unknown}")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Figure 3 applied uncertain ODE APCE/PCE protocol.")
    parser.add_argument("--stage", choices=("debug", "smoke", "quick", "formal"), default="debug")
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--methods", default="")
    parser.add_argument("--n-seeds", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=2026081200)
    parser.add_argument("--tuning-matrix", type=Path, default=PROJECT_ROOT / "experiments" / "figure3_tuning_matrix.json")
    parser.add_argument("--tuning-profile", default="baseline")
    parser.add_argument("--output", type=Path, default=Path("results_figure3_applied_ode_v4"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--record-representative", action="store_true")
    parser.add_argument("--record-all-traces", action="store_true")
    parser.add_argument("--single-case", choices=CASES)
    parser.add_argument("--single-method", choices=ADMISSION_METHODS)
    parser.add_argument("--single-seed", type=int)
    parser.add_argument("--single-output", type=Path)
    args = parser.parse_args()
    args.cases = parse_csv_choices(args.cases, CASES)
    if args.methods:
        args.methods = parse_csv_choices(args.methods, ADMISSION_METHODS)
    else:
        args.methods = list(ADMISSION_METHODS if args.stage == "smoke" else DEFAULT_METHODS)
    if args.n_seeds <= 0:
        args.n_seeds = {"debug": 1, "smoke": 5, "quick": 5, "formal": 50}[args.stage]
    return args


def main() -> None:
    args = parse_args()
    args.tuning_profile_payload = load_tuning_profile(args.tuning_matrix, args.tuning_profile)
    device = torch.device(args.device)
    if args.single_case and args.single_method and args.single_seed is not None:
        output = args.single_output or (args.output / "runs" / f"fig3_{args.single_case}_{args.single_method}_s{args.single_seed}.json")
        row = run_json_path(
            output,
            args.single_case,
            args.single_method,
            args.single_seed,
            device,
            force=args.force,
            record_trace=args.record_representative,
            tuning_profile=args.tuning_profile_payload,
        )
        print(json.dumps({key: value for key, value in row.items() if key != "_trace_payload"}, ensure_ascii=False, sort_keys=True), flush=True)
        return

    args.output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    total = len(args.cases) * args.n_seeds * len(args.methods)
    completed = 0
    for case in args.cases:
        for seed_offset in range(args.n_seeds):
            seed = args.base_seed + seed_offset
            for method in args.methods:
                completed += 1
                record_trace = args.record_all_traces or (
                    args.record_representative and seed_offset == 0 and method in {
                    "denkf",
                    "letkf",
                    "iensf",
                    "aug_enkf",
                    "bma_static",
                    "pce",
                    "apce",
                    }
                )
                path = args.output / "runs" / f"fig3_{case}_{method}_s{seed}.json"
                row = run_json_path(
                    path,
                    case,
                    method,
                    seed,
                    device,
                    force=args.force,
                    record_trace=record_trace,
                    tuning_profile=args.tuning_profile_payload,
                )
                records.append({key: value for key, value in row.items() if key != "_trace_payload"})
                if "_trace_payload" in row:
                    trace_rows.append(row)
                print(
                    f"[{completed}/{total}] case={case} seed={seed} method={method} "
                    f"status={row.get('numerical_status')} nrmse={float(row.get('nrmse', float('nan'))):.4g} "
                    f"crps={float(row.get('crps', float('nan'))):.4g}",
                    flush=True,
                )
    source_data = args.output / "source_data"
    write_csv(source_data / "figure3_v4_run_source_data.csv", records)
    summary = summarize(records, args.base_seed)
    comparisons = paired_comparisons(records, args.base_seed + 10_000)
    write_csv(source_data / "figure3_v4_method_summary.csv", summary)
    write_csv(source_data / "figure3_v4_paired_comparisons.csv", comparisons)
    write_manifest(args.output, args, records, summary)
    save_representative_npz(source_data, trace_rows)
    print(json.dumps({"records": len(records), "valid": sum(1 for row in records if row["numerical_status"] == "valid")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
