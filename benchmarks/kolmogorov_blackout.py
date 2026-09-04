from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from . import kolmogorov_velocity as core

MethodName = Literal["aug_enkf", "bma_static", "pce", "apce"]
METHODS: tuple[MethodName, ...] = ("aug_enkf", "bma_static", "pce", "apce")
LOWER_IS_BETTER = (
    "forecast_nrmse",
    "forecast_crps",
    "forecast_vorticity_nrmse",
    "blackout_alpha_absolute_error",
)


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def normalized_weights(weights: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    result = weights.to(dtype=like.dtype, device=like.device).clamp_min(1.0e-300)
    return result / result.sum().clamp_min(1.0e-300)


def weighted_moments(ensemble: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weights = normalized_weights(weights, ensemble)
    mean = (weights[:, None] * ensemble).sum(dim=0)
    spread = torch.sqrt((weights[:, None] * (ensemble - mean).square()).sum(dim=0).clamp_min(0.0))
    return mean, spread


def vorticity(state: torch.Tensor, system: core.KolmogorovVelocitySystem) -> torch.Tensor:
    field = system._reshape(state)
    u_hat = torch.fft.fft2(field[..., 0, :, :], dim=(-2, -1))
    v_hat = torch.fft.fft2(field[..., 1, :, :], dim=(-2, -1))
    return torch.fft.ifft2(1j * system.kx * v_hat - 1j * system.ky * u_hat, dim=(-2, -1)).real


class ForecastMetrics:
    def __init__(self, blackout_start_step: int, save_dt: float, system: core.KolmogorovVelocitySystem) -> None:
        self.blackout_start_step = int(blackout_start_step)
        self.save_dt = float(save_dt)
        self.system = system
        self.steps: list[int] = []
        self.lead_nrmse: list[float] = []
        self.lead_crps: list[float] = []
        self.lead_coverage: list[float] = []
        self.lead_width: list[float] = []
        self.lead_vorticity_nrmse: list[float] = []
        self.squared_error = 0.0
        self.truth_square = 0.0
        self.points = 0

    def add(
        self,
        step: int,
        ensemble: torch.Tensor,
        truth: torch.Tensor,
        weights: torch.Tensor,
        estimate: torch.Tensor,
    ) -> None:
        if step <= self.blackout_start_step:
            return
        weights = normalized_weights(weights, ensemble)
        error_sq = (estimate - truth).square().sum()
        truth_sq = truth.square().sum().clamp_min(1.0e-30)
        omega_estimate = vorticity(estimate, self.system)
        omega_truth = vorticity(truth, self.system)
        omega_nrmse = torch.sqrt(
            (omega_estimate - omega_truth).square().sum()
            / omega_truth.square().sum().clamp_min(1.0e-30)
        )
        coverage, width = core.weighted_central_interval_coverage_width(
            ensemble, truth, weights, level=0.90
        )
        self.steps.append(int(step))
        self.lead_nrmse.append(float(torch.sqrt(error_sq / truth_sq).detach().cpu()))
        self.lead_crps.append(float(core.weighted_ensemble_crps(ensemble, truth, weights).detach().cpu()))
        self.lead_coverage.append(float(coverage.detach().cpu()))
        self.lead_width.append(float(width.detach().cpu()))
        self.lead_vorticity_nrmse.append(float(omega_nrmse.detach().cpu()))
        self.squared_error += float(error_sq.detach().cpu())
        self.truth_square += float(truth_sq.detach().cpu())
        self.points += int(truth.numel())

    def skill_horizon(self, threshold: float) -> float:
        values = np.asarray(self.lead_nrmse, dtype=float)
        if not values.size:
            return math.nan
        crossed = np.flatnonzero(values > threshold)
        steps = int(crossed[0]) + 1 if crossed.size else int(values.size)
        return float(steps * self.save_dt)

    def finalize(self) -> dict[str, Any]:
        return {
            "forecast_nrmse": math.sqrt(self.squared_error / max(self.truth_square, 1.0e-30)),
            "forecast_rmse": math.sqrt(self.squared_error / max(self.points, 1)),
            "forecast_crps": float(np.mean(self.lead_crps)),
            "forecast_coverage_90": float(np.mean(self.lead_coverage)),
            "forecast_coverage_90_error": abs(float(np.mean(self.lead_coverage)) - 0.90),
            "forecast_interval_width_90": float(np.mean(self.lead_width)),
            "forecast_vorticity_nrmse": float(np.mean(self.lead_vorticity_nrmse)),
            "forecast_steps": self.steps,
            "forecast_lead_time": [
                float((step - self.blackout_start_step) * self.save_dt) for step in self.steps
            ],
            "lead_nrmse": self.lead_nrmse,
            "lead_crps": self.lead_crps,
            "lead_coverage_90": self.lead_coverage,
            "lead_interval_width_90": self.lead_width,
            "lead_vorticity_nrmse": self.lead_vorticity_nrmse,
            "skill_horizon_time_010": self.skill_horizon(0.10),
            "skill_horizon_time_015": self.skill_horizon(0.15),
            "skill_horizon_time_020": self.skill_horizon(0.20),
        }


def padded_history(history: list[torch.Tensor]) -> np.ndarray:
    if not history:
        return np.empty((0, 0), dtype=np.float64)
    width = max(int(item.numel()) for item in history)
    result = np.full((len(history), width), np.nan, dtype=np.float64)
    for index, item in enumerate(history):
        values = item.detach().cpu().numpy().reshape(-1)
        result[index, : values.size] = values
    return result


def trace_payload(
    scenario: core.Scenario,
    blackout_start_step: int,
    means: list[torch.Tensor],
    spreads: list[torch.Tensor],
    extra: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    cfg = scenario.config
    assimilated = np.full((cfg.steps + 1, cfg.observed_points), np.nan, dtype=np.float64)
    for step in range(0, blackout_start_step + 1, cfg.observation_interval):
        assimilated[step] = scenario.observations[step].detach().cpu().numpy()
    return {
        "truth": scenario.truth.detach().cpu().numpy(),
        "assimilated_observations": assimilated,
        "sensor_indices": scenario.sensor_indices.detach().cpu().numpy(),
        "sensor_spatial_indices": scenario.sensor_spatial_indices.detach().cpu().numpy(),
        "times": np.arange(cfg.steps + 1, dtype=np.float64) * cfg.save_dt,
        "blackout_start_step": np.asarray(blackout_start_step, dtype=np.int64),
        "mean_states": torch.stack(means).numpy(),
        "state_spread": torch.stack(spreads).numpy(),
        **extra,
    }


def run_aug_enkf(
    scenario: core.Scenario,
    blackout_start_step: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cfg, device = scenario.config, scenario.truth.device
    system = core.KolmogorovVelocitySystem(cfg, device)
    operator = lambda state: state.index_select(-1, scenario.sensor_indices)
    covariance = core.covariance(scenario)
    generator = core.make_generator(device, cfg.seed + 401)
    alpha = torch.linspace(cfg.alpha_min, cfg.alpha_max, cfg.ensemble_size, dtype=torch.float64, device=device)
    alpha = (
        alpha
        + 0.02
        * torch.randn(alpha.shape, dtype=alpha.dtype, device=device, generator=generator)
    ).clamp(cfg.alpha_min, cfg.alpha_max)
    state = scenario.initial_ensemble.clone()
    weights = torch.full((cfg.ensemble_size,), 1.0 / cfg.ensemble_size, dtype=torch.float64, device=device)
    metrics = ForecastMetrics(blackout_start_step, cfg.save_dt, system)
    means: list[torch.Tensor] = []
    spreads: list[torch.Tensor] = []
    alpha_history: list[float] = []
    started = core.start_runtime(device)
    for step in range(cfg.steps + 1):
        if step <= blackout_start_step and step % cfg.observation_interval == 0:
            updated = core.denkf_analysis(
                torch.cat((state, alpha[:, None]), dim=-1),
                scenario.observations[step],
                operator,
                covariance,
                localization=scenario.augmented_localization,
            )
            state = system.project(updated[:, :-1])
            alpha = updated[:, -1].clamp(cfg.alpha_min, cfg.alpha_max)
        mean, spread = weighted_moments(state, weights)
        metrics.add(step, state, scenario.truth[step], weights, mean)
        means.append(mean.detach().cpu())
        spreads.append(spread.detach().cpu())
        alpha_history.append(float(alpha.mean().detach().cpu()))
        if step == cfg.steps:
            break
        if step < blackout_start_step:
            alpha = (
                alpha
                + cfg.alpha_random_walk_std
                * torch.randn(alpha.shape, dtype=alpha.dtype, device=device, generator=generator)
            ).clamp(cfg.alpha_min, cfg.alpha_max)
        state = system.step(state, alpha, scenario.forecast_noise[step])
    runtime, memory = core.finish_runtime(started, device)
    alpha_estimate = float(alpha_history[blackout_start_step])
    result = metrics.finalize()
    result.update(
        runtime_seconds=runtime,
        peak_gpu_memory_mb=memory,
        blackout_alpha_estimate=alpha_estimate,
        blackout_alpha_absolute_error=abs(alpha_estimate - cfg.alpha_true),
        max_abs_state=float(state.abs().max().detach().cpu()),
    )
    arrays = trace_payload(
        scenario,
        blackout_start_step,
        means,
        spreads,
        {"alpha_estimate_history": np.asarray(alpha_history, dtype=np.float64)},
    )
    return result, arrays


def run_bma(
    scenario: core.Scenario,
    blackout_start_step: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cfg, device = scenario.config, scenario.truth.device
    system = core.KolmogorovVelocitySystem(cfg, device)
    operator = lambda state: state.index_select(-1, scenario.sensor_indices)
    covariance = core.covariance(scenario)
    grid = torch.linspace(cfg.alpha_min, cfg.alpha_max, cfg.bma_alpha_grid_size, dtype=torch.float64, device=device)
    branches = scenario.initial_ensemble.unsqueeze(0).repeat(grid.numel(), 1, 1)
    shadow = branches.clone()
    log_weights = torch.zeros(grid.numel(), dtype=torch.float64, device=device)
    path_weights = torch.softmax(log_weights, dim=0)
    metrics = ForecastMetrics(blackout_start_step, cfg.save_dt, system)
    means: list[torch.Tensor] = []
    spreads: list[torch.Tensor] = []
    weight_history: list[torch.Tensor] = []
    started = core.start_runtime(device)
    for step in range(cfg.steps + 1):
        if step <= blackout_start_step and step % cfg.observation_interval == 0:
            predicted = torch.stack([operator(shadow[index]) for index in range(grid.numel())])
            scores = torch.stack(
                [
                    core.evidence_score(
                        predicted[index], scenario.observations[step], cfg, scenario.obs_sigma, None
                    )
                    for index in range(grid.numel())
                ]
            )
            log_weights = log_weights + scores - scores.mean()
            path_weights = torch.softmax(log_weights, dim=0)
            for index in range(grid.numel()):
                branches[index] = system.project(
                    core.denkf_analysis(
                        branches[index],
                        scenario.observations[step],
                        operator,
                        covariance,
                        localization=scenario.localization,
                    )
                )
        flat = branches.reshape(-1, cfg.state_dim)
        flat_weights = path_weights[:, None].expand(-1, cfg.ensemble_size).reshape(-1) / cfg.ensemble_size
        mean, spread = weighted_moments(flat, flat_weights)
        metrics.add(step, flat, scenario.truth[step], flat_weights, mean)
        means.append(mean.detach().cpu())
        spreads.append(spread.detach().cpu())
        weight_history.append(path_weights.detach().cpu())
        if step == cfg.steps:
            break
        noise = scenario.forecast_noise[step].unsqueeze(0).expand(grid.numel(), -1, -1)
        branches = system.step(branches, grid[:, None], noise)
        if step < blackout_start_step:
            shadow = system.step(shadow, grid[:, None], noise)
    runtime, memory = core.finish_runtime(started, device)
    blackout_weights = weight_history[blackout_start_step].to(grid)
    alpha_estimate = float((blackout_weights * grid).sum().detach().cpu())
    result = metrics.finalize()
    result.update(
        runtime_seconds=runtime,
        peak_gpu_memory_mb=memory,
        blackout_alpha_estimate=alpha_estimate,
        blackout_alpha_absolute_error=abs(alpha_estimate - cfg.alpha_true),
        blackout_evidence_entropy=float(core.entropy(blackout_weights).detach().cpu()),
        max_abs_state=float(branches.abs().max().detach().cpu()),
    )
    arrays = trace_payload(
        scenario,
        blackout_start_step,
        means,
        spreads,
        {
            "alpha_grid": grid.detach().cpu().numpy(),
            "path_weight_history": torch.stack(weight_history).numpy(),
        },
    )
    return result, arrays


def run_pce_apce(
    scenario: core.Scenario,
    method: Literal["pce", "apce"],
    blackout_start_step: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cfg, device = scenario.config, scenario.truth.device
    system = core.KolmogorovVelocitySystem(cfg, device)
    operator = lambda state: state.index_select(-1, scenario.sensor_indices)
    covariance = core.covariance(scenario)
    grid = torch.as_tensor(cfg.coarse_alpha_grid, dtype=torch.float64, device=device)
    branches = scenario.initial_ensemble.unsqueeze(0).repeat(grid.numel(), 1, 1)
    shadow = branches.clone()
    generator = core.make_generator(device, cfg.seed + (701 if method == "pce" else 1701))
    alpha_members = grid[:, None].expand(-1, cfg.ensemble_size).clone()
    alpha_members = (
        alpha_members
        + cfg.branch_member_alpha_jitter
        * torch.randn(alpha_members.shape, dtype=torch.float64, device=device, generator=generator)
    ).clamp(cfg.alpha_min, cfg.alpha_max)
    log_scores = torch.zeros(grid.numel(), dtype=torch.float64, device=device)
    alpha_weights = torch.softmax(log_scores, dim=0)
    state_weights = alpha_weights.clone()
    metrics = ForecastMetrics(blackout_start_step, cfg.save_dt, system)
    means: list[torch.Tensor] = []
    spreads: list[torch.Tensor] = []
    path_weight_history: list[torch.Tensor] = []
    state_weight_history: list[torch.Tensor] = []
    grid_history: list[torch.Tensor] = []
    alpha_history: list[float] = []
    regrid_count = 0
    started = core.start_runtime(device)
    for step in range(cfg.steps + 1):
        flat = branches.reshape(-1, cfg.state_dim)
        flat_weights = state_weights[:, None].expand(-1, cfg.ensemble_size).reshape(-1) / cfg.ensemble_size
        mean, spread = weighted_moments(flat, flat_weights)
        metrics.add(step, flat, scenario.truth[step], flat_weights, mean)
        means.append(mean.detach().cpu())
        spreads.append(spread.detach().cpu())
        path_weight_history.append(alpha_weights.detach().cpu())
        state_weight_history.append(state_weights.detach().cpu())
        grid_history.append(grid.detach().cpu())
        alpha_history.append(float((alpha_weights * grid).sum().detach().cpu()))
        if step == cfg.steps:
            break
        noise = scenario.forecast_noise[step]
        branches = torch.stack(
            [system.step(branches[index], alpha_members[index], noise) for index in range(grid.numel())]
        )
        if step < blackout_start_step:
            shadow = torch.stack(
                [system.step(shadow[index], alpha_members[index], noise) for index in range(grid.numel())]
            )
        next_step = step + 1
        if next_step > blackout_start_step or next_step % cfg.observation_interval != 0:
            continue
        observation = scenario.observations[next_step]
        shadow_prediction = torch.stack([operator(shadow[index]) for index in range(grid.numel())])
        if method == "apce":
            between = shadow_prediction.mean(dim=1).var(dim=0, unbiased=True)
            dimension_weights = (
                cfg.dimension_weight_floor
                + cfg.dimension_weight_gain * between / between.max().clamp_min(1.0e-12)
            )
        else:
            dimension_weights = None
        evidence = torch.stack(
            [
                core.evidence_score(
                    shadow_prediction[index], observation, cfg, scenario.obs_sigma, dimension_weights
                )
                for index in range(grid.numel())
            ]
        )
        centered = evidence - evidence.mean()
        if method == "pce":
            log_scores = log_scores + cfg.pce_temperature * centered
            alpha_weights = torch.softmax(log_scores, dim=0)
        else:
            normalized_entropy = core.entropy(alpha_weights) / math.log(grid.numel())
            temperature = max(
                cfg.apce_min_temperature,
                cfg.apce_temperature * float(normalized_entropy**0.75),
            )
            log_scores = cfg.apce_forgetting * log_scores + temperature * centered
            alpha_weights = core.entropy_project(
                torch.softmax(log_scores, dim=0), cfg.apce_entropy_floor
            )
        state_weights = alpha_weights.clone()
        refined = core.local_grid(grid, log_scores, cfg)
        if refined.shape != grid.shape or not torch.allclose(refined, grid):
            branches = core.interpolate_paths(grid, branches, refined)
            shadow = core.interpolate_paths(grid, shadow, refined)
            alpha_members = core.interpolate_paths(grid, alpha_members, refined)
            log_scores = core.interpolate_paths(grid, log_scores, refined)
            grid = refined
            alpha_weights = torch.softmax(log_scores, dim=0)
            state_weights = (
                core.entropy_project(alpha_weights, cfg.apce_entropy_floor)
                if method == "apce"
                else alpha_weights
            )
            regrid_count += 1
        forecast_branches = branches
        local_branches = torch.empty_like(branches)
        for index in range(grid.numel()):
            local_branches[index] = system.project(
                core.denkf_analysis(
                    forecast_branches[index],
                    observation,
                    operator,
                    covariance,
                    localization=scenario.localization,
                )
            )
        flat_forecast = forecast_branches.reshape(-1, cfg.state_dim)
        analysis_weights = (
            state_weights[:, None].expand(-1, cfg.ensemble_size).reshape(-1) / cfg.ensemble_size
        )
        if cfg.global_analysis_strength > 0:
            global_state = core.weighted_denkf(
                flat_forecast,
                analysis_weights,
                observation,
                operator,
                covariance,
                scenario.localization,
            ).reshape_as(branches)
            branches = (
                (1.0 - cfg.global_analysis_strength) * local_branches
                + cfg.global_analysis_strength * global_state
            )
        else:
            branches = local_branches
        branches = system.project(branches)
        if cfg.branch_augmented_alpha_analysis_strength > 0:
            joint_states = torch.empty_like(branches)
            joint_alpha = torch.empty_like(alpha_members)
            for index in range(grid.numel()):
                joint_states[index], joint_alpha[index] = core.alpha_aug_analysis(
                    branches[index],
                    alpha_members[index],
                    observation,
                    operator,
                    covariance,
                    cfg,
                    scenario.augmented_localization,
                    system,
                )
            strength = cfg.branch_augmented_alpha_analysis_strength
            branches = (1.0 - strength) * branches + strength * joint_states
            alpha_members = (1.0 - strength) * alpha_members + strength * joint_alpha
        if cfg.global_augmented_alpha_analysis_strength > 0:
            flat = branches.reshape(-1, cfg.state_dim)
            flat_alpha = alpha_members.reshape(-1)
            analysis_weights = (
                state_weights[:, None].expand(-1, cfg.ensemble_size).reshape(-1)
                / cfg.ensemble_size
            )
            joint = core.weighted_denkf(
                torch.cat((flat, flat_alpha[:, None]), dim=-1),
                analysis_weights,
                observation,
                operator,
                covariance,
                scenario.augmented_localization,
            )
            strength = cfg.global_augmented_alpha_analysis_strength
            branches = (
                (1.0 - strength) * branches
                + strength * system.project(joint[:, :-1].reshape_as(branches))
            )
            alpha_members = (
                (1.0 - strength) * alpha_members
                + strength
                * joint[:, -1].reshape_as(alpha_members).clamp(cfg.alpha_min, cfg.alpha_max)
            )
        branches = system.project(branches)
        alpha_members = alpha_members.clamp(cfg.alpha_min, cfg.alpha_max)
    runtime, memory = core.finish_runtime(started, device)
    blackout_alpha = float(alpha_history[blackout_start_step])
    blackout_state_weights = state_weight_history[blackout_start_step]
    result = metrics.finalize()
    result.update(
        runtime_seconds=runtime,
        peak_gpu_memory_mb=memory,
        blackout_alpha_estimate=blackout_alpha,
        blackout_alpha_absolute_error=abs(blackout_alpha - cfg.alpha_true),
        blackout_evidence_entropy=float(core.entropy(path_weight_history[blackout_start_step]).detach().cpu()),
        blackout_state_entropy=float(core.entropy(blackout_state_weights).detach().cpu()),
        alpha_regrid_count=regrid_count,
        final_grid_points=int(grid.numel()),
        max_abs_state=float(branches.abs().max().detach().cpu()),
    )
    arrays = trace_payload(
        scenario,
        blackout_start_step,
        means,
        spreads,
        {
            "path_weight_history": padded_history(path_weight_history),
            "state_weight_history": padded_history(state_weight_history),
            "alpha_grid_history": padded_history(grid_history),
            "alpha_estimate_history": np.asarray(alpha_history, dtype=np.float64),
        },
    )
    return result, arrays


def run_method(
    scenario: core.Scenario,
    method: MethodName,
    blackout_start_step: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if method == "aug_enkf":
        return run_aug_enkf(scenario, blackout_start_step)
    if method == "bma_static":
        return run_bma(scenario, blackout_start_step)
    return run_pce_apce(scenario, method, blackout_start_step)


def save_trace(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return math.nan, math.nan
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(clean_json(value), ensure_ascii=False)
                    if isinstance(value, (list, tuple, dict))
                    else clean_json(value)
                    for key, value in row.items()
                }
            )


def aggregate(output: Path) -> None:
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((output / "runs").glob("*.json"))]
    if not records:
        raise RuntimeError(f"No run records found under {output / 'runs'}")
    valid = [row for row in records if row.get("status") == "completed" and row.get("valid")]
    source = output / "source_data"
    scalar_rows = [
        {
            key: value
            for key, value in row.items()
            if not isinstance(value, (list, dict)) and key != "config"
        }
        for row in records
    ]
    run_csv = source / "kolmogorov64_blackout_run_source_data.csv"
    write_csv(run_csv, scalar_rows)
    summary_metrics = [
        "forecast_nrmse",
        "forecast_crps",
        "forecast_vorticity_nrmse",
        "forecast_coverage_90",
        "forecast_interval_width_90",
        "blackout_alpha_absolute_error",
        "skill_horizon_time_010",
        "skill_horizon_time_015",
        "skill_horizon_time_020",
    ]
    summary_rows: list[dict[str, Any]] = []
    for method in METHODS:
        subset = [row for row in valid if row["method"] == method]
        item: dict[str, Any] = {
            "method": method,
            "label": core.METHOD_LABELS[method],
            "n_valid": len(subset),
            "n_total": sum(row["method"] == method for row in records),
        }
        for metric in summary_metrics:
            mean, std = mean_std([float(row.get(metric, math.nan)) for row in subset])
            item[f"{metric}_mean"] = mean
            item[f"{metric}_std"] = std
        summary_rows.append(item)
    summary_csv = source / "kolmogorov64_blackout_method_summary.csv"
    write_csv(summary_csv, summary_rows)
    by_pair = {(row["method"], int(row["seed"])): row for row in valid}
    paired_rows: list[dict[str, Any]] = []
    for candidate in ("pce", "apce"):
        for baseline in ("aug_enkf", "bma_static"):
            seeds = sorted(
                {seed for method, seed in by_pair if method == candidate}
                & {seed for method, seed in by_pair if method == baseline}
            )
            for metric in LOWER_IS_BETTER:
                deltas = [
                    float(by_pair[(baseline, seed)][metric])
                    - float(by_pair[(candidate, seed)][metric])
                    for seed in seeds
                ]
                mean, std = mean_std(deltas)
                paired_rows.append(
                    {
                        "candidate": candidate,
                        "candidate_label": core.METHOD_LABELS[candidate],
                        "baseline": baseline,
                        "baseline_label": core.METHOD_LABELS[baseline],
                        "metric": metric,
                        "positive_delta_means_candidate_better": True,
                        "paired_n": len(deltas),
                        "mean_delta": mean,
                        "std_delta": std,
                        "wins": sum(value > 0 for value in deltas),
                        "ties": sum(np.isclose(value, 0.0) for value in deltas),
                        "losses": sum(value < 0 for value in deltas),
                        "per_seed_delta": deltas,
                    }
                )
    paired_csv = source / "kolmogorov64_blackout_paired_comparisons.csv"
    write_csv(paired_csv, paired_rows)
    lead_rows: list[dict[str, Any]] = []
    for method in METHODS:
        subset = [row for row in valid if row["method"] == method]
        if not subset:
            continue
        for lead_index, lead_time in enumerate(subset[0]["forecast_lead_time"]):
            item: dict[str, Any] = {
                "method": method,
                "label": core.METHOD_LABELS[method],
                "lead_index": lead_index + 1,
                "lead_time": lead_time,
                "n": len(subset),
            }
            for source_key, output_key in (
                ("lead_nrmse", "nrmse"),
                ("lead_crps", "crps"),
                ("lead_vorticity_nrmse", "vorticity_nrmse"),
            ):
                mean, std = mean_std([float(row[source_key][lead_index]) for row in subset])
                item[f"{output_key}_mean"] = mean
                item[f"{output_key}_std"] = std
            lead_rows.append(item)
    lead_csv = source / "kolmogorov64_blackout_lead_time_source_data.csv"
    write_csv(lead_csv, lead_rows)
    manifest = {
        "protocol": "kolmogorov-blackout-forecast",
        "case": "kolmogorov_velocity",
        "reynolds": 1500,
        "forcing_wavenumber": 2,
        "sensor_grid": "16x16",
        "observation_interval": 4,
        "blackout_start_step": 40,
        "assimilation_steps": [0, 40],
        "forecast_steps": [41, 58],
        "future_observations_used": False,
        "analysis_updates_after_blackout": 0,
        "evidence_updates_after_blackout": 0,
        "regrids_after_blackout": 0,
        "methods": list(METHODS),
        "n_runs": len(records),
        "n_valid": len(valid),
        "run_source_data": str(run_csv),
        "method_summary": str(summary_csv),
        "paired_comparisons": str(paired_csv),
        "lead_time_source_data": str(lead_csv),
    }
    write_json(source / "manifest.json", manifest)
    print(json.dumps(clean_json(manifest), ensure_ascii=False))


def run_one(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    config = core.KOL64VelocityConfig(
        seed=args.seed,
        sensor_grid=16,
        window_start=args.window_start,
        data_path=str(args.data_path),
        reynolds=1500.0,
        forcing_wavenumber=2,
        observation_interval=4,
        alpha_true=core.alpha_for_re(1500.0),
    )
    shared = core.load_shared_assets(config, args.asset_root, device)
    scenario = core.materialize(shared, device)
    run_id = f"kol64_re1500_k2_s16_t4_blackout40_{args.method}_seed{args.seed}"
    run_path = args.output / "runs" / f"{run_id}.json"
    trace_path = args.output / "traces" / f"{run_id}.npz"
    payload: dict[str, Any] = {
        "run_id": run_id,
        "case": "kolmogorov_velocity",
        "method": args.method,
        "label": core.METHOD_LABELS[args.method],
        "seed": args.seed,
        "window_start": args.window_start,
        "window_end": args.window_start + config.steps,
        "reynolds": 1500.0,
        "forcing_wavenumber": 2,
        "sensor_grid": 16,
        "observation_interval": 4,
        "steps": config.steps,
        "save_dt": config.save_dt,
        "blackout_start_step": args.blackout_start_step,
        "forecast_start_step": args.blackout_start_step + 1,
        "future_observations_used": False,
        "analysis_updates_after_blackout": 0,
        "evidence_updates_after_blackout": 0,
        "regrids_after_blackout": 0,
        "alpha_true": config.alpha_true,
        "asset_npz": str(scenario.asset_path),
        "config": asdict(config),
        "status": "started",
        "valid": False,
        "device": str(device),
    }
    started = time.perf_counter()
    try:
        result, arrays = run_method(scenario, args.method, args.blackout_start_step)
        if args.record_trace:
            save_trace(trace_path, arrays)
            payload["trace_npz"] = str(trace_path)
        payload.update(result)
        required = (
            "forecast_nrmse",
            "forecast_crps",
            "forecast_vorticity_nrmse",
            "blackout_alpha_absolute_error",
        )
        payload["valid"] = all(math.isfinite(float(payload[key])) for key in required)
        payload["status"] = "completed"
    except Exception as exc:
        payload["status"] = "failed"
        payload["failure_type"] = type(exc).__name__
        payload["failure_message"] = str(exc)
    payload["wall_seconds"] = time.perf_counter() - started
    write_json(run_path, payload)
    print(json.dumps(clean_json(payload), ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the published Kolmogorov-flow blackout forecast.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=core.DEFAULT_DATA)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--window-start", type=int)
    parser.add_argument("--blackout-start-step", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--record-trace", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.summary_only:
        aggregate(args.output)
        return
    if args.method is None or args.seed is None or args.window_start is None:
        raise ValueError("--method, --seed, and --window-start are required")
    if not 0 < args.blackout_start_step < 58:
        raise ValueError("blackout step must lie strictly between 0 and 58")
    run_one(args)


if __name__ == "__main__":
    main()
