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

import run_benchmark_v3 as v3
import run_benchmark_v4 as v4
from experiments.wave_scenario_assets import WaveScenarioAssets
from hilda_da.alpha import AlphaEvidenceTracker
from hilda_da.alpha_refinement import (
    apce_calibration_parameters,
    local_alpha_grid,
    numpy_regrid_paths,
    refined_alpha_map,
)
from hilda_da.baselines import denkf_analysis, letkf_analysis
from hilda_da.config import AlphaConfig, HILDAConfig
from hilda_da.filter import HILDAFilter
from hilda_da.metrics import (
    weighted_central_interval_coverage_width,
    weighted_ensemble_crps,
)
from hilda_da.observation_flow import analytic_posterior_mixture
from hilda_da.observations import SparseObservation
from hilda_da.strong_baselines import ensf_lr_ridge_analysis
from run_hybrid_wave import ensf_update_lr


METHODS = (
    "denkf",
    "letkf",
    "ensf_lr_ridge",
    "pce",
    "apce",
    "hilda_crn_fixed",
    "hilda_kp_repaired",
)


def configuration(assets: WaveScenarioAssets) -> v3.Config:
    observation_steps = np.flatnonzero(assets.observation_mask)
    interval = int(observation_steps[0]) if observation_steps.size else assets.n_steps + 1
    return dataclasses.replace(
        v3.make_config("quick"),
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
    cfg: v3.Config,
) -> torch.Tensor:
    result = v3.propagate_batch(
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
    generator = torch.Generator(device=device).manual_seed(assets.seed + v3.stable_offset(method))
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
            ensemble, v3.alpha_to_theta(0.50, cfg), step, assets, cfg
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
    generator = torch.Generator(device=device).manual_seed(assets.seed + v3.stable_offset(method))
    means: list[np.ndarray] = []
    for step in range(assets.n_steps + 1):
        means.append(ensemble.mean(0).detach().cpu().numpy())
        if step == assets.n_steps:
            break
        ensemble = propagate_numpy(
            ensemble, v3.alpha_to_theta(0.50, cfg), step, assets, cfg
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


def update_v4_weights(
    branches: np.ndarray,
    shadow: np.ndarray,
    observation: np.ndarray,
    cfg: v3.Config,
    scenario: v3.Scenario,
    config: v4.V4EvidenceConfig,
    log_weights: np.ndarray,
    step: int,
) -> tuple[np.ndarray, np.ndarray]:
    shadow_observations = [
        shadow[q][:, scenario.observation_indices].copy() for q in range(cfg.n_alpha)
    ]
    log_likelihood_shadow = v4.evidence_vector(
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
        analysis_evidence = v4.evidence_vector(
            analysis_observations, observation, cfg.obs_noise, config
        )
        log_likelihood = (1.0 - blend) * log_likelihood_shadow + blend * analysis_evidence
    else:
        log_likelihood = log_likelihood_shadow
    centered = log_likelihood - np.mean(log_likelihood)
    weights = v3.softmax(log_weights)
    temperature = config.temperature
    if config.adaptive_temperature:
        entropy_ratio = v4.entropy(weights) / max(math.log(cfg.n_alpha), 1.0e-12)
        temperature = float(
            np.clip(config.temperature * entropy_ratio**0.75, config.min_temperature, config.temperature)
        )
    log_weights = config.forgetting * log_weights + temperature * centered
    weights = v3.softmax(log_weights)
    weights = np.maximum(weights, config.weight_floor)
    weights /= weights.sum()
    weights = v4.entropy_project(weights, v4.entropy_target(progress, config))
    return np.log(np.maximum(weights, 1.0e-300)), weights


def update_v5_weights(
    branches: np.ndarray,
    shadow: np.ndarray,
    observation: np.ndarray,
    cfg: v3.Config,
    scenario: v3.Scenario,
    config: v4.V4EvidenceConfig,
    alpha_log_scores: np.ndarray,
    step: int,
    method: str,
) -> tuple[np.ndarray, np.ndarray, Any]:
    shadow_observations = [
        shadow[q][:, scenario.observation_indices].copy() for q in range(cfg.n_alpha)
    ]
    log_likelihood_shadow = v4.evidence_vector(
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
        analysis_evidence = v4.evidence_vector(
            analysis_observations, observation, cfg.obs_noise, config
        )
        log_likelihood = (1.0 - blend) * log_likelihood_shadow + blend * analysis_evidence
    else:
        log_likelihood = log_likelihood_shadow
    centered = log_likelihood - np.mean(log_likelihood)
    if method == "pce":
        alpha_log_scores = alpha_log_scores + config.temperature * centered
        alpha_weights = v3.softmax(alpha_log_scores)
        state_weights = alpha_weights
        calibration = None
    elif method == "apce":
        calibration = apce_calibration_parameters(
            centered,
            pce_temperature=float(v4.ABLATION_CONFIGS["A6_pce"].temperature),
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
        alpha_weights = v3.softmax(alpha_log_scores)
        state_weights = v4.entropy_project(alpha_weights, calibration.entropy_floor)
    else:
        raise ValueError(method)
    return alpha_log_scores, state_weights, calibration


def run_pce_family(
    assets: WaveScenarioAssets,
    method: str,
    device: torch.device,
) -> dict[str, Any]:
    cfg = configuration(assets)
    scenario = v3.scenario_from_assets(assets, cfg)
    ablation = "A6_pce" if method == "pce" else "A7_apce"
    config = v4.ABLATION_CONFIGS[ablation]
    alpha_grid = scenario.alpha_grid.copy()
    global_bounds = (float(scenario.alpha_grid[0]), float(scenario.alpha_grid[-1]))
    theta_grid = np.asarray([v3.alpha_to_theta(float(alpha), cfg) for alpha in alpha_grid], dtype=float)
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
            branches[q] = v3.propagate_batch(
                branches[q], float(theta), float(assets.times[step]), cfg,
                np.random.default_rng(assets.seed), stochastic=True,
                noise_draw=assets.forecast_noise[step],
            )
            shadow[q] = v3.propagate_batch(
                shadow[q], float(theta), float(assets.times[step]), cfg,
                np.random.default_rng(assets.seed), stochastic=True,
                noise_draw=assets.forecast_noise[step],
            )
        if not assets.observation_mask[step + 1]:
            continue
        observation = assets.observations[step + 1]
        alpha_log_scores, state_weights, calibration = update_v5_weights(
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
            alpha_weights = v3.softmax(alpha_log_scores)
            if method == "apce":
                state_weights = numpy_regrid_paths(alpha_grid, state_weights, refined_grid)
                state_weights = np.maximum(state_weights, 1.0e-12)
                state_weights /= state_weights.sum()
                if calibration is not None:
                    state_weights = v4.entropy_project(state_weights, calibration.entropy_floor)
            else:
                state_weights = alpha_weights
            alpha_grid = refined_grid
            theta_grid = np.asarray([v3.alpha_to_theta(float(alpha), cfg) for alpha in alpha_grid], dtype=float)
            regrid_count += 1
        paired_seed = cfg.seed + 10_000_000 + step + 1
        for q in range(cfg.n_alpha):
            branches[q] = ensf_update_lr(
                branches[q], observation, scenario.observation_indices, cfg,
                np.random.default_rng(paired_seed),
            )
    alpha_estimate = refined_alpha_map(alpha_grid, alpha_log_scores)
    alpha_weights = v3.softmax(alpha_log_scores)
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
        alpha_final_entropy=float(v4.entropy(state_weights)),
        alpha_evidence_entropy=float(v4.entropy(alpha_weights)),
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
    scenario = v3.scenario_from_assets(assets, cfg)
    ablation = "A6_pce" if method == "pce" else "A7_apce"
    config = v4.ABLATION_CONFIGS[ablation]
    alpha_grid = scenario.alpha_grid.copy()
    global_bounds = (float(scenario.alpha_grid[0]), float(scenario.alpha_grid[-1]))
    theta_grid = np.asarray([v3.alpha_to_theta(float(alpha), cfg) for alpha in alpha_grid], dtype=float)
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
            branches[q] = v3.propagate_batch(
                branches[q], float(theta), float(assets.times[step]), cfg,
                np.random.default_rng(assets.seed), stochastic=True,
                noise_draw=assets.forecast_noise[step],
            )
            shadow[q] = v3.propagate_batch(
                shadow[q], float(theta), float(assets.times[step]), cfg,
                np.random.default_rng(assets.seed), stochastic=True,
                noise_draw=assets.forecast_noise[step],
            )
        if not assets.observation_mask[step + 1]:
            continue
        observation = assets.observations[step + 1]
        alpha_log_scores, state_weights, calibration = update_v5_weights(
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
            alpha_weights = v3.softmax(alpha_log_scores)
            if method == "apce":
                state_weights = numpy_regrid_paths(alpha_grid, state_weights, refined_grid)
                state_weights = np.maximum(state_weights, 1.0e-12)
                state_weights /= state_weights.sum()
                if calibration is not None:
                    state_weights = v4.entropy_project(state_weights, calibration.entropy_floor)
            else:
                state_weights = alpha_weights
            alpha_grid = refined_grid
            theta_grid = np.asarray([v3.alpha_to_theta(float(alpha), cfg) for alpha in alpha_grid], dtype=float)
        paired_seed = cfg.seed + 10_000_000 + step + 1
        for q in range(cfg.n_alpha):
            branches[q] = ensf_update_lr(
                branches[q], observation, scenario.observation_indices, cfg,
                np.random.default_rng(paired_seed),
            )
    return np.stack(means), state_weights.copy()


def run_hilda(
    assets: WaveScenarioAssets,
    repaired: bool,
    device: torch.device,
) -> dict[str, Any]:
    cfg = configuration(assets)
    dtype = torch.float64
    alpha_config = AlphaConfig(
        alpha_min=cfg.alpha_min,
        alpha_max=cfg.alpha_max,
        initial_nodes=9,
        max_nodes=9,
        prune_threshold=0.0,
    )
    hilda = HILDAFilter(HILDAConfig(alpha=alpha_config))
    tracker = AlphaEvidenceTracker.create(alpha_config, device=device, dtype=dtype)
    base = torch.as_tensor(assets.initial_ensemble, dtype=dtype, device=device)
    branches = base.unsqueeze(0).repeat(tracker.alpha.numel(), 1, 1)
    truth = torch.as_tensor(assets.truth_states, dtype=dtype, device=device)
    indices = torch.as_tensor(assets.observation_indices, dtype=torch.int64, device=device)
    operator = SparseObservation(indices)
    covariance = cfg.obs_noise**2 * torch.eye(indices.numel(), dtype=dtype, device=device)
    metrics = MetricAccumulator(assets.nx)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(assets.n_steps + 1):
        flat, member_weights = flat_path_distribution(branches, tracker.weights)
        metrics.add(flat, truth[step], member_weights)
        if step == assets.n_steps:
            break
        propagated = []
        for alpha, branch in zip(tracker.alpha, branches, strict=True):
            propagated.append(
                propagate_numpy(
                    branch, v3.alpha_to_theta(float(alpha), cfg), step, assets, cfg
                )
            )
        branches = torch.stack(propagated)
        if not assets.observation_mask[step + 1]:
            continue
        observation = torch.as_tensor(
            assets.observations[step + 1], dtype=dtype, device=device
        )
        if repaired:
            evidence = []
            for branch in branches:
                mixture = analytic_posterior_mixture(
                    operator(branch), observation, covariance,
                    relative_floor=hilda.config.flow.eigenvalue_floor,
                )
                evidence.append(mixture.log_evidence / observation.numel())
            tracker.update(torch.stack(evidence).to(tracker.log_scores))
            branches = torch.stack(
                [
                    denkf_analysis(branch, observation, operator, covariance)
                    for branch in branches
                ]
            )
        else:
            analysis = hilda.analyze_paths(
                branches, tracker, observation, operator, covariance
            )
            branches = analysis.ensembles
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    alpha_estimate = tracker.continuous_estimate()
    result = metrics.finalize()
    result.update(
        runtime_seconds=time.perf_counter() - started,
        forward_member_steps=assets.n_steps * 9 * assets.ensemble_size,
        peak_gpu_memory_mb=(
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda" else 0.0
        ),
        alpha_estimate=alpha_estimate,
        alpha_absolute_error=abs(alpha_estimate - assets.alpha_true),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--records", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = [r for r in manifest["records"] if float(r["alpha_true"]) == 0.12][: args.records]
    rows: list[dict[str, Any]] = []
    for record in selected:
        assets = WaveScenarioAssets.load(Path(record["path"]))
        for method in METHODS:
            if method in {"denkf", "letkf", "ensf_lr_ridge"}:
                result = run_single_path(assets, method, device)
            elif method in {"pce", "apce"}:
                result = run_pce_family(assets, method, device)
            else:
                result = run_hilda(assets, method == "hilda_kp_repaired", device)
            row = {
                "asset": record["name"],
                "seed": assets.seed,
                "alpha_true": assets.alpha_true,
                "method": method,
                **result,
            }
            rows.append(row)
            print("RESULT", json.dumps(row, sort_keys=True))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "runs.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for method in METHODS:
        subset = [row for row in rows if row["method"] == method]
        summary.append({
            "method": method,
            "seeds": len(subset),
            **{
                key: float(np.mean([float(row[key]) for row in subset]))
                for key in (
                    "displacement_nrmse",
                    "displacement_rmse",
                    "crps",
                    "coverage_90",
                    "interval_width_90",
                    "runtime_seconds",
                    "forward_member_steps",
                    "peak_gpu_memory_mb",
                )
            },
            "alpha_absolute_error": (
                float(np.mean([float(row["alpha_absolute_error"]) for row in subset]))
                if subset[0]["alpha_absolute_error"] is not None else None
            ),
        })
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("SUMMARY", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
