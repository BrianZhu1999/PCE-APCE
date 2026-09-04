"""Figure 5: KSE sparse-sensing blackout forecast worker.

This runner deliberately imports, rather than modifies, the frozen Figure 4
official-NMI KSE worker.  The only protocol change is temporal: observations,
analysis corrections, evidence updates, and local regridding are available
through ``blackout_start_step`` and are forbidden afterwards.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import run_figure4_kse_nmi64_smoke_worker as core


MethodName = Literal["aug_enkf", "bma_static", "pce", "apce"]
METHODS: tuple[MethodName, ...] = ("aug_enkf", "bma_static", "pce", "apce")
METRIC_DIRECTIONS = {
    "forecast_nrmse": "lower",
    "forecast_crps": "lower",
    "forecast_correlation_error": "lower",
    "mu_absolute_error_at_blackout": "lower",
    "skill_horizon_time_015": "higher",
    "skill_horizon_time_020": "higher",
    "skill_horizon_time_030": "higher",
}

DEFAULT_INPUT = Path("<HILDA_RESULTS_ROOT>/external/S3GM_NMI_2024/KSE_test.npy")
DEFAULT_OUTPUT = Path(
    "<HILDA_RESULTS_ROOT>/results/"
    "figure5_kse_blackoutfull_t1_step40_smoke_5seeds_20260814_4gpu"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
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


def finite_tensor(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all().item())


def padded_history(history: list[torch.Tensor]) -> np.ndarray:
    """Pad a variable-length one-dimensional history with NaN for NPZ storage."""
    if not history:
        return np.empty((0, 0), dtype=np.float64)
    width = max(int(item.numel()) for item in history)
    output = np.full((len(history), width), np.nan, dtype=np.float64)
    for index, item in enumerate(history):
        values = item.detach().cpu().numpy().reshape(-1)
        output[index, : values.size] = values
    return output


def two_point_correlation_error(estimate: torch.Tensor, truth: torch.Tensor) -> float:
    """Mean absolute error of circular, normalized two-point correlations."""
    state_dim = int(truth.numel())
    lags = torch.linspace(
        1,
        state_dim // 2,
        32,
        dtype=torch.float64,
        device=truth.device,
    ).round().to(torch.int64)

    def correlation(field: torch.Tensor) -> torch.Tensor:
        centered = field - field.mean()
        denominator = centered.square().mean().clamp_min(1.0e-12)
        return torch.stack([(centered * torch.roll(centered, -int(lag), dims=0)).mean() / denominator for lag in lags])

    return float((correlation(estimate) - correlation(truth)).abs().mean())


class ForecastMetrics:
    """Horizon-resolved and window-aggregated metrics after sensor blackout."""

    def __init__(self, *, blackout_start_step: int, saved_dt: float) -> None:
        self.blackout_start_step = int(blackout_start_step)
        self.saved_dt = float(saved_dt)
        self.steps: list[int] = []
        self.nrmse: list[float] = []
        self.crps: list[float] = []
        self.coverage: list[float] = []
        self.width: list[float] = []
        self.correlation_error: list[float] = []
        self.squared_error = 0.0
        self.truth_square = 0.0

    def add(
        self,
        *,
        step: int,
        ensemble: torch.Tensor,
        weights: torch.Tensor,
        truth: torch.Tensor,
        estimate: torch.Tensor,
    ) -> None:
        if step <= self.blackout_start_step:
            return
        weights = weights.to(dtype=ensemble.dtype, device=ensemble.device).clamp_min(1.0e-300)
        weights = weights / weights.sum().clamp_min(1.0e-300)
        error_sq = (estimate - truth).square().sum()
        truth_sq = truth.square().sum().clamp_min(1.0e-30)
        self.steps.append(int(step))
        self.nrmse.append(float(torch.sqrt(error_sq / truth_sq)))
        self.crps.append(float(core.weighted_ensemble_crps(ensemble, truth, weights)))
        coverage, width = core.weighted_central_interval_coverage_width(ensemble, truth, weights, level=0.90)
        self.coverage.append(float(coverage))
        self.width.append(float(width))
        self.correlation_error.append(two_point_correlation_error(estimate, truth))
        self.squared_error += float(error_sq)
        self.truth_square += float(truth_sq)

    def _skill_horizon_time(self, threshold: float) -> float:
        if not self.nrmse:
            return math.nan
        passed = np.asarray(self.nrmse, dtype=float) > float(threshold)
        if passed.any():
            lead_steps = int(np.flatnonzero(passed)[0]) + 1
        else:
            lead_steps = len(self.nrmse)
        return float(lead_steps * self.saved_dt)

    def finalize(self) -> dict[str, Any]:
        return {
            "forecast_nrmse": math.sqrt(self.squared_error / max(self.truth_square, 1.0e-30)),
            "forecast_crps": float(np.mean(self.crps)) if self.crps else math.nan,
            "forecast_coverage_90": float(np.mean(self.coverage)) if self.coverage else math.nan,
            "forecast_interval_width_90": float(np.mean(self.width)) if self.width else math.nan,
            "forecast_correlation_error": float(np.mean(self.correlation_error)) if self.correlation_error else math.nan,
            "forecast_steps": self.steps,
            "forecast_lead_time": [float((step - self.blackout_start_step) * self.saved_dt) for step in self.steps],
            "lead_nrmse": self.nrmse,
            "lead_crps": self.crps,
            "lead_coverage_90": self.coverage,
            "lead_interval_width_90": self.width,
            "lead_correlation_error": self.correlation_error,
            "skill_horizon_time_015": self._skill_horizon_time(0.15),
            "skill_horizon_time_020": self._skill_horizon_time(0.20),
            "skill_horizon_time_030": self._skill_horizon_time(0.30),
        }


def mixture_spread(ensemble: torch.Tensor, weights: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
    normalized = weights.to(dtype=ensemble.dtype, device=ensemble.device).clamp_min(1.0e-300)
    normalized = normalized / normalized.sum().clamp_min(1.0e-300)
    return torch.sqrt((normalized[:, None] * (ensemble - estimate).square()).sum(dim=0).clamp_min(0.0))


def available_observations(scenario: core.KSE64Scenario, blackout_start_step: int) -> dict[int, torch.Tensor]:
    return {step: observation for step, observation in scenario.observations.items() if int(step) <= int(blackout_start_step)}


def scenario_trace_payload(
    scenario: core.KSE64Scenario,
    *,
    blackout_start_step: int,
    available_obs: dict[int, torch.Tensor],
) -> dict[str, np.ndarray]:
    observation_steps = torch.as_tensor(sorted(available_obs), dtype=torch.int64)
    observations = torch.stack([available_obs[int(step)] for step in observation_steps.tolist()], dim=0)
    return {
        "truth": scenario.truth.detach().cpu().numpy(),
        "assimilation_observations": observations.detach().cpu().numpy(),
        "assimilation_observation_steps": observation_steps.detach().cpu().numpy(),
        "sensor_indices": scenario.observation_indices.detach().cpu().numpy(),
        "initial_ensemble": scenario.initial_ensemble.detach().cpu().numpy(),
        "forecast_noise": scenario.forecast_noise.detach().cpu().numpy(),
        "initial_mu_ensemble": scenario.initial_mu_ensemble.detach().cpu().numpy(),
        "true_mu": np.asarray(scenario.true_mu, dtype=np.float64),
        "blackout_start_step": np.asarray(int(blackout_start_step), dtype=np.int64),
        "saved_dt": np.asarray(float(scenario.config.saved_dt), dtype=np.float64),
        "observation_geometry": np.asarray(scenario.observation_geometry),
        "observation_scale": np.asarray(float(scenario.observation_scale), dtype=np.float64),
        "observation_dim": np.asarray(int(scenario.observation_dim), dtype=np.int64),
    }


def record_frame(
    *,
    step: int,
    scenario: core.KSE64Scenario,
    metrics: ForecastMetrics,
    flat_ensemble: torch.Tensor,
    flat_weights: torch.Tensor,
    estimate: torch.Tensor,
    mean_history: list[torch.Tensor],
    spread_history: list[torch.Tensor],
) -> None:
    mean_history.append(estimate.detach().cpu())
    spread_history.append(mixture_spread(flat_ensemble, flat_weights, estimate).detach().cpu())
    metrics.add(
        step=step,
        ensemble=flat_ensemble,
        weights=flat_weights,
        truth=scenario.truth[step],
        estimate=estimate,
    )


def run_aug_enkf_blackout(
    scenario: core.KSE64Scenario,
    device: torch.device,
    *,
    blackout_start_step: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    config = scenario.config
    solver = core.KSEETDRK4(
        state_dim=config.state_dim,
        length=config.length,
        sub_dt=config.saved_dt / config.solver_substeps,
        device=device,
        dtype=torch.float64,
    )
    ensemble = scenario.initial_ensemble.clone()
    mu_ensemble = scenario.initial_mu_ensemble.clone()
    weights = torch.full((config.ensemble_size,), 1.0 / config.ensemble_size, dtype=torch.float64, device=device)
    operator = core.make_observation_operator(config, scenario.observation_indices)
    covariance = core.observation_covariance(scenario, device, torch.float64)
    observations = available_observations(scenario, blackout_start_step)
    metrics = ForecastMetrics(blackout_start_step=blackout_start_step, saved_dt=config.saved_dt)
    means: list[torch.Tensor] = []
    spreads: list[torch.Tensor] = []
    mu_history: list[torch.Tensor] = []
    checkpoint: dict[str, np.ndarray] = {}
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(config.steps + 1):
        estimate = ensemble.mean(dim=0)
        record_frame(
            step=step,
            scenario=scenario,
            metrics=metrics,
            flat_ensemble=ensemble,
            flat_weights=weights,
            estimate=estimate,
            mean_history=means,
            spread_history=spreads,
        )
        mu_history.append(mu_ensemble.detach().cpu())
        if step == blackout_start_step:
            checkpoint = {
                "checkpoint_ensemble": ensemble.detach().cpu().numpy(),
                "checkpoint_mu_ensemble": mu_ensemble.detach().cpu().numpy(),
            }
        if step == config.steps:
            break
        ensemble = solver.step_saved(
            ensemble,
            mu_ensemble,
            substeps=config.solver_substeps,
            noise=scenario.forecast_noise[step],
            noise_scale=config.process_noise * scenario.truth_std,
        )
        next_step = step + 1
        if next_step in observations:
            ensemble, mu_ensemble = core.augmented_mu_denkf_analysis(
                ensemble,
                mu_ensemble,
                observations[next_step],
                operator,
                covariance,
                config.mu_lower,
                config.mu_upper,
            )
            ensemble = torch.nan_to_num(ensemble, nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    mu_estimate = float(checkpoint["checkpoint_mu_ensemble"].mean()) if checkpoint else float(mu_ensemble.mean())
    summary = metrics.finalize()
    summary.update(
        runtime_seconds=float(time.perf_counter() - started),
        peak_gpu_memory_mb=float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
        mu_estimate_at_blackout=mu_estimate,
        mu_map_at_blackout=float(np.median(checkpoint["checkpoint_mu_ensemble"])) if checkpoint else float(mu_ensemble.median()),
        mu_absolute_error_at_blackout=abs(mu_estimate - scenario.true_mu),
        forward_member_steps=int(config.steps * config.ensemble_size),
        final_path_count=1,
    )
    arrays = {
        "mean_states": torch.stack(means).numpy(),
        "state_spread": torch.stack(spreads).numpy(),
        "mu_history": torch.stack(mu_history).numpy(),
        **checkpoint,
    }
    return summary, arrays


def run_bma_blackout(
    scenario: core.KSE64Scenario,
    device: torch.device,
    *,
    blackout_start_step: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    config = scenario.config
    solver = core.KSEETDRK4(
        state_dim=config.state_dim,
        length=config.length,
        sub_dt=config.saved_dt / config.solver_substeps,
        device=device,
        dtype=torch.float64,
    )
    mu_grid = torch.tensor(config.bma_mu_grid, dtype=torch.float64, device=device)
    path_count = int(mu_grid.numel())
    branches = scenario.initial_ensemble.unsqueeze(0).repeat(path_count, 1, 1)
    log_scores = torch.zeros(path_count, dtype=torch.float64, device=device)
    path_weights = torch.softmax(log_scores, dim=0)
    operator = core.make_observation_operator(config, scenario.observation_indices)
    covariance = core.observation_covariance(scenario, device, torch.float64)
    observations = available_observations(scenario, blackout_start_step)
    metrics = ForecastMetrics(blackout_start_step=blackout_start_step, saved_dt=config.saved_dt)
    means: list[torch.Tensor] = []
    spreads: list[torch.Tensor] = []
    weight_history: list[torch.Tensor] = []
    checkpoint: dict[str, np.ndarray] = {}
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(config.steps + 1):
        flat = branches.reshape(-1, config.state_dim)
        flat_weights = path_weights[:, None].expand(-1, config.ensemble_size).reshape(-1) / config.ensemble_size
        estimate = (path_weights[:, None] * branches.mean(dim=1)).sum(dim=0)
        record_frame(
            step=step,
            scenario=scenario,
            metrics=metrics,
            flat_ensemble=flat,
            flat_weights=flat_weights,
            estimate=estimate,
            mean_history=means,
            spread_history=spreads,
        )
        weight_history.append(path_weights.detach().cpu())
        if step == blackout_start_step:
            checkpoint = {
                "checkpoint_branches": branches.detach().cpu().numpy(),
                "checkpoint_mu_grid": mu_grid.detach().cpu().numpy(),
                "checkpoint_log_scores": log_scores.detach().cpu().numpy(),
                "checkpoint_path_weights": path_weights.detach().cpu().numpy(),
            }
        if step == config.steps:
            break
        for path_index in range(path_count):
            branches[path_index] = solver.step_saved(
                branches[path_index],
                torch.full((config.ensemble_size,), float(mu_grid[path_index]), dtype=torch.float64, device=device),
                substeps=config.solver_substeps,
                noise=scenario.forecast_noise[step],
                noise_scale=config.process_noise * scenario.truth_std,
            )
        next_step = step + 1
        if next_step in observations:
            predicted_obs = torch.stack([operator(branches[path_index]) for path_index in range(path_count)])
            evidence = torch.stack(
                [
                    core.evidence_score(
                        predicted_obs[path_index],
                        observations[next_step],
                        config.obs_noise * scenario.observation_scale,
                        config.evidence_shrinkage,
                        None,
                    )
                    for path_index in range(path_count)
                ]
            )
            log_scores = log_scores + 0.75 * (evidence - evidence.mean())
            path_weights = torch.softmax(log_scores, dim=0)
            for path_index in range(path_count):
                branches[path_index] = core.denkf_analysis(branches[path_index], observations[next_step], operator, covariance)
                branches[path_index] = torch.nan_to_num(branches[path_index], nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    final_weights = checkpoint.get("checkpoint_path_weights", path_weights.detach().cpu().numpy())
    mu_estimate = float(np.dot(final_weights, mu_grid.detach().cpu().numpy()))
    summary = metrics.finalize()
    summary.update(
        runtime_seconds=float(time.perf_counter() - started),
        peak_gpu_memory_mb=float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
        mu_estimate_at_blackout=mu_estimate,
        mu_map_at_blackout=float(mu_grid[int(torch.argmax(log_scores))]),
        mu_absolute_error_at_blackout=abs(mu_estimate - scenario.true_mu),
        evidence_entropy_at_blackout=float(core.entropy(torch.as_tensor(final_weights, dtype=torch.float64))),
        forward_member_steps=int(config.steps * path_count * config.ensemble_size),
        final_path_count=path_count,
    )
    arrays = {
        "mean_states": torch.stack(means).numpy(),
        "state_spread": torch.stack(spreads).numpy(),
        "path_weight_history": torch.stack(weight_history).numpy(),
        "mu_grid": mu_grid.detach().cpu().numpy(),
        **checkpoint,
    }
    return summary, arrays


def run_pce_apce_blackout(
    scenario: core.KSE64Scenario,
    method: MethodName,
    device: torch.device,
    *,
    blackout_start_step: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    config = scenario.config
    solver = core.KSEETDRK4(
        state_dim=config.state_dim,
        length=config.length,
        sub_dt=config.saved_dt / config.solver_substeps,
        device=device,
        dtype=torch.float64,
    )
    mu_grid = torch.tensor(config.coarse_mu_grid, dtype=torch.float64, device=device)
    bounds = (float(config.mu_lower), float(config.mu_upper))
    branches = scenario.initial_ensemble.unsqueeze(0).repeat(mu_grid.numel(), 1, 1)
    shadow = branches.clone()
    log_scores = torch.zeros(mu_grid.numel(), dtype=torch.float64, device=device)
    alpha_members = torch.stack(
        [
            core.member_mu_cloud(
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
    state_weights = core.evidence_gap_state_weights(path_weights, log_scores, config)
    operator = core.make_observation_operator(config, scenario.observation_indices)
    covariance = core.observation_covariance(scenario, device, torch.float64)
    observations = available_observations(scenario, blackout_start_step)
    metrics = ForecastMetrics(blackout_start_step=blackout_start_step, saved_dt=config.saved_dt)
    means: list[torch.Tensor] = []
    spreads: list[torch.Tensor] = []
    path_weight_history: list[torch.Tensor] = []
    state_weight_history: list[torch.Tensor] = []
    mu_grid_history: list[torch.Tensor] = []
    mu_estimate_history: list[float] = []
    checkpoint: dict[str, np.ndarray] = {}
    regrid_count = 0
    forward_member_steps = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(config.steps + 1):
        flat = branches.reshape(-1, config.state_dim)
        flat_weights = state_weights[:, None].expand(-1, config.ensemble_size).reshape(-1) / config.ensemble_size
        branch_means = branches.mean(dim=1)
        estimate = core.pce_point_estimate(mu_grid, branch_means, state_weights, log_scores, config)
        record_frame(
            step=step,
            scenario=scenario,
            metrics=metrics,
            flat_ensemble=flat,
            flat_weights=flat_weights,
            estimate=estimate,
            mean_history=means,
            spread_history=spreads,
        )
        path_weight_history.append(path_weights.detach().cpu())
        state_weight_history.append(state_weights.detach().cpu())
        mu_grid_history.append(mu_grid.detach().cpu())
        mu_estimate_history.append(float(core.torch_refined_alpha_map(mu_grid, log_scores)))
        if step == blackout_start_step:
            checkpoint = {
                "checkpoint_analysis_branches": branches.detach().cpu().numpy(),
                "checkpoint_shadow_branches": shadow.detach().cpu().numpy(),
                "checkpoint_alpha_members": alpha_members.detach().cpu().numpy(),
                "checkpoint_mu_grid": mu_grid.detach().cpu().numpy(),
                "checkpoint_log_scores": log_scores.detach().cpu().numpy(),
                "checkpoint_path_weights": path_weights.detach().cpu().numpy(),
                "checkpoint_state_weights": state_weights.detach().cpu().numpy(),
            }
        if step == config.steps:
            break
        path_count = int(mu_grid.numel())
        for path_index in range(path_count):
            branches[path_index] = solver.step_saved(
                branches[path_index],
                alpha_members[path_index],
                substeps=config.solver_substeps,
                noise=scenario.forecast_noise[step],
                noise_scale=config.process_noise * scenario.truth_std,
            )
            shadow[path_index] = solver.step_saved(
                shadow[path_index],
                alpha_members[path_index],
                substeps=config.solver_substeps,
                noise=scenario.forecast_noise[step],
                noise_scale=config.process_noise * scenario.truth_std,
            )
        forward_member_steps += int(2 * path_count * config.ensemble_size)
        next_step = step + 1
        if next_step not in observations:
            continue
        shadow_obs = torch.stack([operator(shadow[path_index]) for path_index in range(path_count)])
        dimension_weights = None
        if method == "apce":
            between = shadow_obs.mean(dim=1).var(dim=0, unbiased=True)
            dimension_weights = config.dimension_weight_floor + config.dimension_weight_gain * between / between.max().clamp_min(1.0e-12)
        evidence = torch.stack(
            [
                core.evidence_score(
                    shadow_obs[path_index],
                    observations[next_step],
                    config.obs_noise * scenario.observation_scale,
                    config.evidence_shrinkage,
                    dimension_weights,
                )
                for path_index in range(path_count)
            ]
        )
        centered = evidence - evidence.mean()
        if method == "pce":
            log_scores = log_scores + config.pce_temperature * centered
            path_weights = torch.softmax(log_scores, dim=0)
            state_weights = core.evidence_gap_state_weights(path_weights, log_scores, config)
        else:
            calibration = core.apce_calibration_parameters(
                centered,
                pce_temperature=config.pce_temperature,
                apce_temperature=config.apce_temperature,
                apce_min_temperature=config.apce_min_temperature,
                apce_forgetting=config.apce_forgetting,
                apce_entropy_floor=config.apce_entropy_floor,
                progress=next_step / max(blackout_start_step, 1),
            )
            log_scores = calibration.forgetting * log_scores + calibration.temperature * centered
            path_weights = torch.softmax(log_scores, dim=0)
            calibrated = core.entropy_project(path_weights, calibration.entropy_floor)
            state_weights = core.evidence_gap_state_weights(calibrated, log_scores, config)
        refined = core.torch_local_alpha_grid(
            mu_grid,
            log_scores,
            points=config.local_grid_points,
            bounds=bounds,
            topk=config.local_grid_topk,
            min_spacing=config.local_grid_min_spacing,
        )
        if refined.shape != mu_grid.shape or not torch.allclose(refined, mu_grid):
            branches = core.torch_regrid_paths(mu_grid, branches, refined)
            shadow = core.torch_regrid_paths(mu_grid, shadow, refined)
            alpha_members = core.torch_regrid_paths(mu_grid, alpha_members, refined)
            log_scores = core.torch_regrid_paths(mu_grid, log_scores, refined)
            mu_grid = refined
            path_weights = torch.softmax(log_scores, dim=0)
            state_weights = core.evidence_gap_state_weights(path_weights, log_scores, config)
            regrid_count += 1
        local = torch.empty_like(branches)
        for path_index in range(int(mu_grid.numel())):
            local[path_index] = core.denkf_analysis(branches[path_index], observations[next_step], operator, covariance)
        confidence, _ = core.evidence_gap_confidence(log_scores)
        global_strength = float(
            np.clip(
                config.global_analysis_strength * confidence ** max(config.global_analysis_confidence_power, 1.0e-8),
                0.0,
                1.0,
            )
        )
        if global_strength > 1.0e-12:
            flat_forecast = branches.reshape(-1, config.state_dim)
            analysis_weights = state_weights[:, None].expand(-1, config.ensemble_size).reshape(-1) / config.ensemble_size
            global_analysis = core.weighted_denkf_analysis(
                flat_forecast,
                analysis_weights,
                observations[next_step],
                operator,
                covariance,
            ).reshape_as(branches)
            branches = (1.0 - global_strength) * local + global_strength * global_analysis
        else:
            branches = local
        if config.branch_augmented_alpha_analysis_strength > 1.0e-12:
            joint_branches = torch.empty_like(branches)
            joint_mu = torch.empty_like(alpha_members)
            for path_index in range(int(mu_grid.numel())):
                joint_branches[path_index], joint_mu[path_index] = core.augmented_mu_denkf_analysis(
                    branches[path_index],
                    alpha_members[path_index],
                    observations[next_step],
                    operator,
                    covariance,
                    bounds[0],
                    bounds[1],
                )
            strength = float(config.branch_augmented_alpha_analysis_strength)
            branches = (1.0 - strength) * branches + strength * joint_branches
            alpha_members = (1.0 - strength) * alpha_members + strength * joint_mu
        if config.global_augmented_alpha_analysis_strength > 1.0e-12:
            flat_branches = branches.reshape(-1, config.state_dim)
            flat_mu = alpha_members.reshape(-1)
            analysis_weights = state_weights[:, None].expand(-1, config.ensemble_size).reshape(-1) / config.ensemble_size
            joint_state, joint_mu = core.weighted_augmented_mu_denkf_analysis(
                flat_branches,
                flat_mu,
                analysis_weights,
                observations[next_step],
                operator,
                covariance,
                bounds[0],
                bounds[1],
            )
            strength = float(config.global_augmented_alpha_analysis_strength)
            branches = (1.0 - strength) * branches + strength * joint_state.reshape_as(branches)
            alpha_members = (1.0 - strength) * alpha_members + strength * joint_mu.reshape_as(alpha_members)
        branches = torch.nan_to_num(branches, nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0)
        alpha_members = alpha_members.clamp(bounds[0], bounds[1])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    final_grid = checkpoint.get("checkpoint_mu_grid", mu_grid.detach().cpu().numpy())
    final_log_scores = checkpoint.get("checkpoint_log_scores", log_scores.detach().cpu().numpy())
    mu_estimate = float(core.torch_refined_alpha_map(
        torch.as_tensor(final_grid, dtype=torch.float64),
        torch.as_tensor(final_log_scores, dtype=torch.float64),
    ))
    summary = metrics.finalize()
    summary.update(
        runtime_seconds=float(time.perf_counter() - started),
        peak_gpu_memory_mb=float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
        mu_estimate_at_blackout=mu_estimate,
        mu_map_at_blackout=float(final_grid[int(np.argmax(final_log_scores))]),
        mu_absolute_error_at_blackout=abs(mu_estimate - scenario.true_mu),
        evidence_entropy_at_blackout=float(core.entropy(torch.softmax(torch.as_tensor(final_log_scores, dtype=torch.float64), dim=0))),
        state_entropy_at_blackout=float(core.entropy(torch.as_tensor(checkpoint.get("checkpoint_state_weights", state_weights.detach().cpu().numpy()), dtype=torch.float64))),
        alpha_regrid_count=int(regrid_count),
        final_path_count=int(len(final_grid)),
        forward_member_steps=int(forward_member_steps),
    )
    arrays = {
        "mean_states": torch.stack(means).numpy(),
        "state_spread": torch.stack(spreads).numpy(),
        "path_weight_history": padded_history(path_weight_history),
        "state_weight_history": padded_history(state_weight_history),
        "mu_grid_history": padded_history(mu_grid_history),
        "mu_estimate_history": np.asarray(mu_estimate_history, dtype=np.float64),
        **checkpoint,
    }
    return summary, arrays


def run_one(
    method: MethodName,
    scenario: core.KSE64Scenario,
    device: torch.device,
    *,
    blackout_start_step: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if method == "aug_enkf":
        return run_aug_enkf_blackout(scenario, device, blackout_start_step=blackout_start_step)
    if method == "bma_static":
        return run_bma_blackout(scenario, device, blackout_start_step=blackout_start_step)
    return run_pce_apce_blackout(scenario, method, device, blackout_start_step=blackout_start_step)


def save_trace_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def numeric_summary(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return math.nan, math.nan
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


def write_summary(output_root: Path) -> None:
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((output_root / "runs").glob("*.json"))]
    if not rows:
        return
    source_data = output_root / "source_data"
    source_data.mkdir(parents=True, exist_ok=True)
    run_csv = source_data / "run_source_data.csv"
    excluded = {"config"}
    fields = sorted({key for row in rows for key in row if key not in excluded})
    with run_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: clean_json(row.get(key, "")) for key in fields})

    valid_rows = [row for row in rows if row.get("status") == "completed" and bool(row.get("valid"))]
    summary_metrics = [
        "forecast_nrmse",
        "forecast_crps",
        "forecast_correlation_error",
        "forecast_coverage_90",
        "forecast_interval_width_90",
        "mu_absolute_error_at_blackout",
        "skill_horizon_time_015",
        "skill_horizon_time_020",
        "skill_horizon_time_030",
        "runtime_seconds",
        "peak_gpu_memory_mb",
    ]
    method_rows: list[dict[str, Any]] = []
    for method in METHODS:
        subset = [row for row in valid_rows if row.get("method") == method]
        all_method = [row for row in rows if row.get("method") == method]
        item: dict[str, Any] = {
            "method": method,
            "label": core.METHOD_LABELS[method],
            "n_valid": len(subset),
            "n_total": len(all_method),
        }
        for metric in summary_metrics:
            mean, std = numeric_summary([float(row.get(metric, math.nan)) for row in subset])
            item[f"{metric}_mean"] = mean
            item[f"{metric}_std"] = std
        method_rows.append(item)
    method_csv = source_data / "method_summary.csv"
    fields = sorted({key for row in method_rows for key in row})
    with method_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(method_rows)

    lead_rows: list[dict[str, Any]] = []
    for method in METHODS:
        subset = [row for row in valid_rows if row.get("method") == method]
        steps = sorted({int(step) for row in subset for step in row.get("forecast_steps", [])})
        for step in steps:
            item = {
                "method": method,
                "label": core.METHOD_LABELS[method],
                "forecast_step": step,
                "forecast_lead_time": (step - int(subset[0]["blackout_start_step"])) * float(subset[0]["saved_dt"]),
            }
            for metric_key, output_name in [
                ("lead_nrmse", "nrmse"),
                ("lead_crps", "crps"),
                ("lead_correlation_error", "correlation_error"),
            ]:
                values = []
                for row in subset:
                    row_steps = [int(value) for value in row.get("forecast_steps", [])]
                    if step in row_steps:
                        values.append(float(row[metric_key][row_steps.index(step)]))
                mean, std = numeric_summary(values)
                item[f"{output_name}_mean"] = mean
                item[f"{output_name}_std"] = std
                item[f"{output_name}_n"] = len(values)
            lead_rows.append(item)
    lead_csv = source_data / "lead_time_summary.csv"
    fields = sorted({key for row in lead_rows for key in row})
    with lead_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(lead_rows)

    by_method_seed = {(str(row["method"]), int(row["seed_index"])): row for row in valid_rows}
    paired_rows: list[dict[str, Any]] = []
    for candidate in ("pce", "apce"):
        for baseline in ("aug_enkf", "bma_static"):
            common_seeds = sorted(
                set(seed for method, seed in by_method_seed if method == candidate)
                & set(seed for method, seed in by_method_seed if method == baseline)
            )
            for metric, direction in METRIC_DIRECTIONS.items():
                differences = []
                for seed_index in common_seeds:
                    candidate_value = float(by_method_seed[(candidate, seed_index)][metric])
                    baseline_value = float(by_method_seed[(baseline, seed_index)][metric])
                    difference = baseline_value - candidate_value if direction == "lower" else candidate_value - baseline_value
                    differences.append(difference)
                mean, std = numeric_summary(differences)
                paired_rows.append(
                    {
                        "candidate": candidate,
                        "candidate_label": core.METHOD_LABELS[candidate],
                        "baseline": baseline,
                        "baseline_label": core.METHOD_LABELS[baseline],
                        "metric": metric,
                        "positive_delta_means_candidate_better": True,
                        "paired_n": len(differences),
                        "mean_delta": mean,
                        "std_delta": std,
                        "wins": int(sum(value > 0.0 for value in differences)),
                        "ties": int(sum(np.isclose(value, 0.0) for value in differences)),
                        "losses": int(sum(value < 0.0 for value in differences)),
                        "per_seed_delta": differences,
                    }
                )
    paired_csv = source_data / "paired_comparisons.csv"
    fields = sorted({key for row in paired_rows for key in row})
    with paired_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in paired_rows:
            flattened = dict(row)
            flattened["per_seed_delta"] = json.dumps(flattened["per_seed_delta"])
            writer.writerow(flattened)

    manifests = {
        "protocol": "figure5-official-nmi-kse-fullfield-fulltime-blackout-step40",
        "output_root": str(output_root),
        "methods": list(METHODS),
        "n_runs": len(rows),
        "n_valid": len(valid_rows),
        "steps": 99,
        "saved_dt": 0.5,
        "assimilation_window_steps": [0, 40],
        "forecast_window_steps": [41, 99],
        "blackout_frozen_state": [
            "state ensemble / analysis branches",
            "mu ensemble / alpha members",
            "BMA or PCE/APCE path weights",
            "PCE/APCE refined local grid and cumulative evidence",
        ],
        "blackout_forbidden_operations": ["observations", "analysis update", "evidence update", "local regridding"],
        "metrics": summary_metrics,
        "run_source_data": str(run_csv),
        "run_source_data_sha256": file_sha256(run_csv),
        "method_summary": str(method_csv),
        "method_summary_sha256": file_sha256(method_csv),
        "lead_time_summary": str(lead_csv),
        "lead_time_summary_sha256": file_sha256(lead_csv),
        "paired_comparisons": str(paired_csv),
        "paired_comparisons_sha256": file_sha256(paired_csv),
        "worker_sha256": file_sha256(Path(__file__).resolve()),
        "figure4_core_sha256": file_sha256(Path(core.__file__).resolve()),
    }
    write_json(source_data / "manifest.json", manifests)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Figure 5 official-NMI KSE blackout forecast.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--method", choices=METHODS, default=None)
    parser.add_argument("--seed-index", type=int, default=None)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--seed-base", type=int, default=2026081400)
    parser.add_argument("--downsampling-factor", type=int, default=1)
    parser.add_argument("--temporal-obs-interval", type=int, default=1)
    parser.add_argument("--blackout-start-step", type=int, default=40)
    parser.add_argument("--steps", type=int, default=99)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--record-trace", action="store_true")
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
    if int(args.downsampling_factor) <= 0 or 1024 % int(args.downsampling_factor):
        raise ValueError("--downsampling-factor must be a positive divisor of 1024")
    if int(args.temporal_obs_interval) <= 0:
        raise ValueError("--temporal-obs-interval must be positive")
    if not 0 < int(args.blackout_start_step) < int(args.steps):
        raise ValueError("--blackout-start-step must lie strictly inside [0, --steps]")

    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    sample_index = int(args.seed_index % 9) if args.sample_index is None else int(args.sample_index)
    seed = int(args.seed_base + args.seed_index)
    observed_points = 1024 // int(args.downsampling_factor)
    config = core.KSE64Config(
        seed=seed,
        sample_index=sample_index,
        steps=int(args.steps),
        obs_interval=int(args.temporal_obs_interval),
        obs_geometry="physical",
        observed_points=observed_points,
    )
    run_id = (
        f"kse_nmi{args.downsampling_factor}x_t{args.temporal_obs_interval}_"
        f"blackout{args.blackout_start_step}_{args.method}_seed{args.seed_index:02d}_sample{sample_index:02d}"
    )
    run_path = output_root / "runs" / f"{run_id}.json"
    trace_path = output_root / "traces" / f"{run_id}.npz"
    payload: dict[str, Any] = {
        "run_id": run_id,
        "case": f"kse_nmi_official_physical_{args.downsampling_factor}x_t{args.temporal_obs_interval}_blackout{args.blackout_start_step}",
        "method": args.method,
        "label": core.METHOD_LABELS[args.method],
        "seed_index": int(args.seed_index),
        "seed": seed,
        "sample_index": sample_index,
        "true_mu": core.sample_true_mu(sample_index),
        "downsampling_factor": int(args.downsampling_factor),
        "observed_points": int(observed_points),
        "temporal_obs_interval": int(args.temporal_obs_interval),
        "observation_geometry": "physical",
        "steps": int(args.steps),
        "saved_dt": float(config.saved_dt),
        "blackout_start_step": int(args.blackout_start_step),
        "assimilation_final_observation_step": int(args.blackout_start_step),
        "forecast_start_step": int(args.blackout_start_step) + 1,
        "future_observations_used": False,
        "analysis_updates_after_blackout": 0,
        "evidence_updates_after_blackout": 0,
        "regrids_after_blackout": 0,
        "valid": False,
        "status": "started",
        "hostname": os.uname().nodename if hasattr(os, "uname") else "",
        "device": str(device),
        "input_path": str(args.input),
        "config": asdict(config),
    }
    started = time.perf_counter()
    try:
        scenario = core.generate_scenario(config, args.input, device)
        payload.update(
            observation_dim=int(scenario.observation_dim),
            observation_scale=float(scenario.observation_scale),
        )
        result, trace_arrays = run_one(
            args.method,
            scenario,
            device,
            blackout_start_step=int(args.blackout_start_step),
        )
        if not finite_tensor(torch.as_tensor(result["lead_nrmse"], dtype=torch.float64)):
            raise FloatingPointError("Non-finite forecast lead-time nRMSE.")
        if args.record_trace:
            trace_arrays.update(
                scenario_trace_payload(
                    scenario,
                    blackout_start_step=int(args.blackout_start_step),
                    available_obs=available_observations(scenario, int(args.blackout_start_step)),
                )
            )
            save_trace_npz(trace_path, trace_arrays)
            payload["trace_npz"] = str(trace_path)
            payload["trace_npz_sha256"] = file_sha256(trace_path)
        payload.update(result)
        payload["valid"] = bool(np.isfinite(float(payload["forecast_nrmse"])))
        payload["status"] = "completed"
    except Exception as exc:  # noqa: BLE001 -- all failures are first-class source data
        payload["status"] = "failed"
        payload["failure_type"] = type(exc).__name__
        payload["failure_message"] = str(exc)
    payload["wall_seconds"] = float(time.perf_counter() - started)
    payload["script_path"] = str(Path(__file__).resolve())
    payload["script_sha256"] = file_sha256(Path(__file__).resolve())
    payload["figure4_core_path"] = str(Path(core.__file__).resolve())
    payload["figure4_core_sha256"] = file_sha256(Path(core.__file__).resolve())
    write_json(run_path, payload)
    print(json.dumps(clean_json(payload), ensure_ascii=False))


if __name__ == "__main__":
    main()
