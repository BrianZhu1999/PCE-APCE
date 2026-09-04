"""Independent Figure 2 state--alpha refinement experiment.

This module deliberately leaves the frozen Figure 2 v58 runner untouched.  It
keeps its two-pass coarse/local shadow-evidence protocol and only adds the
state--alpha corrections selected in Figure 3 profile v52.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments import run_figure2_reviewer_gate as reviewer
from experiments import run_modern_baseline_admission as modern
from experiments import run_wave_repair_validation as wave_repair
from hilda_da.alpha_refinement import evidence_gap_confidence
from hilda_da.baselines import denkf_analysis
from hilda_da.observations import SparseObservation
from hilda_da.systems.one_dimensional import Heat1D
from paper_experiments import run_spring_heat_gate as sh


PROFILE_NAME = "v52_lagcases_aug_global100"
HOOK_FIELDS = (
    "branch_member_alpha_jitter",
    "branch_member_alpha_jitter_confidence_power",
    "branch_augmented_alpha_analysis_strength",
    "global_augmented_alpha_analysis_strength",
    "global_analysis_strength",
)


@dataclass(frozen=True)
class StateAlphaHooks:
    branch_member_alpha_jitter: float = 0.0
    branch_member_alpha_jitter_confidence_power: float = 1.0
    branch_augmented_alpha_analysis_strength: float = 0.0
    global_augmented_alpha_analysis_strength: float = 0.0
    global_analysis_strength: float = 0.0

    @classmethod
    def from_profile(cls, profile: dict[str, Any], method: str) -> "StateAlphaHooks":
        if method not in {"pce", "apce"}:
            raise ValueError(f"unsupported state-alpha method: {method}")
        values: dict[str, Any] = {}
        values.update(profile.get("pce_apce_overrides", {}))
        values.update(profile.get("method_overrides", {}).get(method, {}))
        unknown = sorted(set(values) - set(HOOK_FIELDS) - {
            "pce_temperature", "apce_temperature", "apce_min_temperature",
            "apce_forgetting", "apce_entropy_floor", "evidence_shrinkage",
            "local_grid_points", "local_grid_topk", "local_grid_min_spacing",
            "state_weight_power", "state_map_blend", "state_map_blend_confidence_power",
            "state_point_estimator_mode", "analysis_iterations", "global_analysis_confidence_power",
        })
        if unknown:
            raise ValueError(f"v52 profile contains unsupported Figure 2 fields: {unknown}")
        return cls(**{key: float(values[key]) for key in HOOK_FIELDS if key in values})

    def asdict(self) -> dict[str, float]:
        return {key: float(getattr(self, key)) for key in HOOK_FIELDS}


def load_v52_profile(path: Path, profile_name: str = PROFILE_NAME) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    profiles = payload.get("profiles", payload)
    if profile_name not in profiles:
        raise KeyError(f"profile {profile_name!r} not found in {path}")
    profile = dict(profiles[profile_name])
    profile["name"] = profile_name
    profile.setdefault("pce_apce_overrides", profile.get("overrides", {}))
    profile.setdefault("method_overrides", {})
    profile.setdefault("case_overrides", {})
    if not isinstance(profile["method_overrides"], dict):
        raise TypeError("method_overrides must be a mapping")
    return profile


def _member_alpha_cloud(alpha_grid: torch.Tensor, ensemble_size: int, hooks: StateAlphaHooks, scores: torch.Tensor) -> torch.Tensor:
    confidence, _ = evidence_gap_confidence(scores)
    fraction = max(hooks.branch_member_alpha_jitter, 0.0) * (1.0 - confidence) ** max(hooks.branch_member_alpha_jitter_confidence_power, 1.0e-8)
    offsets = torch.linspace(-1.0, 1.0, ensemble_size, dtype=alpha_grid.dtype, device=alpha_grid.device)
    span = float(alpha_grid[-1] - alpha_grid[0])
    return (alpha_grid[:, None] + fraction * span * offsets[None, :]).clamp(float(alpha_grid[0]), float(alpha_grid[-1]))


def _weighted_denkf(states: torch.Tensor, weights: torch.Tensor, observation: torch.Tensor, operator: SparseObservation, covariance: torch.Tensor) -> torch.Tensor:
    weights = weights.to(dtype=states.dtype, device=states.device).clamp_min(1.0e-300)
    weights = weights / weights.sum().clamp_min(1.0e-300)
    predicted = operator(states)
    x_mean = (weights[:, None] * states).sum(0)
    z_mean = (weights[:, None] * predicted).sum(0)
    xa, za = states - x_mean, predicted - z_mean
    denom = (1.0 - weights.square().sum()).clamp_min(torch.finfo(states.dtype).eps)
    zw = weights[:, None] * za
    cross = xa.mT @ zw / denom
    innovation = za.mT @ zw / denom + covariance
    factor = torch.linalg.cholesky(innovation)
    gain = torch.cholesky_solve(cross.mT, factor).mT
    mean = x_mean + gain @ (observation - z_mean)
    anomalies = xa - 0.5 * (za @ gain.mT)
    return mean[None, :] + anomalies


def _augmented_denkf(states: torch.Tensor, alpha: torch.Tensor, observation: torch.Tensor, operator: SparseObservation, covariance: torch.Tensor, lower: float, upper: float) -> tuple[torch.Tensor, torch.Tensor]:
    updated = denkf_analysis(torch.cat([states, alpha[:, None]], dim=-1), observation, operator, covariance)
    return updated[:, :-1], updated[:, -1].clamp(lower, upper)


def _weighted_augmented_denkf(states: torch.Tensor, alpha: torch.Tensor, weights: torch.Tensor, observation: torch.Tensor, operator: SparseObservation, covariance: torch.Tensor, lower: float, upper: float) -> tuple[torch.Tensor, torch.Tensor]:
    updated = _weighted_denkf(torch.cat([states, alpha[:, None]], dim=-1), weights, observation, operator, covariance)
    return updated[:, :-1], updated[:, -1].clamp(lower, upper)


def _statealpha_analysis(branches: torch.Tensor, alpha_members: torch.Tensor, state_weights: torch.Tensor, observation: torch.Tensor, operator: SparseObservation, covariance: torch.Tensor, lower: float, upper: float, hooks: StateAlphaHooks, local_analysis) -> tuple[torch.Tensor, torch.Tensor]:
    local = torch.stack([local_analysis(branch, observation, operator, covariance) for branch in branches])
    if hooks.global_analysis_strength > 1.0e-12:
        flat_weights = state_weights[:, None].expand_as(alpha_members).reshape(-1) / alpha_members.shape[1]
        global_states = _weighted_denkf(branches.reshape(-1, branches.shape[-1]), flat_weights, observation, operator, covariance).reshape_as(branches)
        branches = (1.0 - hooks.global_analysis_strength) * local + hooks.global_analysis_strength * global_states
    else:
        branches = local
    if hooks.branch_augmented_alpha_analysis_strength > 1.0e-12:
        pairs = [_augmented_denkf(branches[i], alpha_members[i], observation, operator, covariance, lower, upper) for i in range(branches.shape[0])]
        updated_states = torch.stack([pair[0] for pair in pairs])
        updated_alpha = torch.stack([pair[1] for pair in pairs])
        strength = hooks.branch_augmented_alpha_analysis_strength
        branches = (1.0 - strength) * branches + strength * updated_states
        alpha_members = (1.0 - strength) * alpha_members + strength * updated_alpha
    if hooks.global_augmented_alpha_analysis_strength > 1.0e-12:
        flat_weights = state_weights[:, None].expand_as(alpha_members).reshape(-1) / alpha_members.shape[1]
        updated_states, updated_alpha = _weighted_augmented_denkf(branches.reshape(-1, branches.shape[-1]), alpha_members.reshape(-1), flat_weights, observation, operator, covariance, lower, upper)
        strength = hooks.global_augmented_alpha_analysis_strength
        branches = (1.0 - strength) * branches + strength * updated_states.reshape_as(branches)
        alpha_members = (1.0 - strength) * alpha_members + strength * updated_alpha.reshape_as(alpha_members)
    return branches, alpha_members


def _update_spring_heat_weights(method: str, log_weights: torch.Tensor, weights: torch.Tensor, shadow: torch.Tensor, observation: torch.Tensor, scenario: sh.Scenario) -> tuple[torch.Tensor, torch.Tensor]:
    config = scenario.config
    operator = SparseObservation(scenario.observation_indices)
    shadow_obs = torch.stack([operator(branch) for branch in shadow])
    dimension_weights = None
    if method == "apce":
        between = shadow_obs.mean(dim=1).var(dim=0, unbiased=True)
        dimension_weights = 0.35 + 0.65 * between / between.max().clamp_min(1.0e-12)
    evidence = torch.stack([sh.evidence_score(shadow_obs[i], observation, config.obs_noise, config.evidence_shrinkage, dimension_weights) for i in range(shadow.shape[0])])
    centered = evidence - evidence.mean()
    if method == "pce":
        log_weights = log_weights + config.pce_temperature * centered
        weights = torch.softmax(log_weights, dim=0)
    else:
        entropy_ratio = float(sh.entropy(weights) / math.log(max(weights.numel(), 2)))
        temperature = float(np.clip(config.apce_temperature * entropy_ratio**0.75, config.apce_min_temperature, config.apce_temperature))
        log_weights = config.apce_forgetting * log_weights + temperature * centered
        weights = sh.entropy_project(torch.softmax(log_weights, dim=0), config.apce_entropy_floor)
        log_weights = weights.clamp_min(1.0e-300).log()
    return log_weights, weights


def _spring_heat_pass(scenario: sh.Scenario, method: str, device: torch.device, hooks: StateAlphaHooks, alpha_grid_override: np.ndarray | None = None) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    config = scenario.config
    system = sh.make_system(config)
    if isinstance(system, Heat1D):
        system.grid = system.grid.to(device)
    alpha_grid = torch.as_tensor(alpha_grid_override, dtype=scenario.alpha_grid.dtype, device=device) if alpha_grid_override is not None else scenario.alpha_grid.clone()
    branches = scenario.initial_ensemble.unsqueeze(0).repeat(alpha_grid.numel(), 1, 1)
    shadow = branches.clone()
    scores = torch.zeros(alpha_grid.numel(), dtype=branches.dtype, device=device)
    weights = torch.softmax(scores, dim=0)
    alpha_members = _member_alpha_cloud(alpha_grid, config.ensemble_size, hooks, scores)
    operator = SparseObservation(scenario.observation_indices)
    covariance = config.obs_noise**2 * torch.eye(scenario.observation_indices.numel(), dtype=branches.dtype, device=device)
    metrics, traces_w, traces_a = sh.TrajectoryMetrics(), [], []
    started = time.perf_counter()
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    for step in range(config.steps + 1):
        flat = branches.reshape(-1, branches.shape[-1])
        flat_weights = weights[:, None].expand(-1, config.ensemble_size).reshape(-1) / config.ensemble_size
        metrics.add(reviewer.spring_heat_primary(flat, scenario), reviewer.spring_heat_primary(scenario.truth[step], scenario), flat_weights)
        traces_w.append(weights.detach().cpu().numpy()); traces_a.append(alpha_members.detach().cpu().numpy())
        if step == config.steps: break
        # ``step_with_noise`` broadcasts the alpha tensor over a batch, preserving
        # the baseline RK4/noise law without the prohibitively slow member loop.
        propagation_alpha = alpha_members.unsqueeze(-1) if config.name == "heat" else alpha_members
        branches = sh.step_with_noise(
            system, branches, step * config.dt, config.dt, propagation_alpha,
            scenario.forecast_noise[step].unsqueeze(0).expand_as(branches),
        )
        shadow = sh.step_with_noise(
            system, shadow, step * config.dt, config.dt, propagation_alpha,
            scenario.forecast_noise[step].unsqueeze(0).expand_as(shadow),
        )
        if step + 1 not in scenario.observations: continue
        observation = scenario.observations[step + 1]
        scores, weights = _update_spring_heat_weights(method, scores, weights, shadow, observation, scenario)
        branches, alpha_members = _statealpha_analysis(branches, alpha_members, weights, observation, operator, covariance, float(alpha_grid[0]), float(alpha_grid[-1]), hooks, denkf_analysis)
    if device.type == "cuda": torch.cuda.synchronize(device)
    result = metrics.finalize()
    result.update(runtime_seconds=float(time.perf_counter()-started), forward_member_steps=int(2*config.steps*alpha_grid.numel()*config.ensemble_size), peak_gpu_memory_mb=float(torch.cuda.max_memory_allocated(device)/1024**2) if device.type=="cuda" else 0.0)
    trace = {"alpha_weights": np.asarray(traces_w), "alpha_members": np.asarray(traces_a)}
    return result, alpha_grid.detach().cpu().numpy(), weights.detach().cpu().numpy(), scores.detach().cpu().numpy(), trace


def _wave_pass(assets: modern.WaveScenarioAssets, method: str, device: torch.device, hooks: StateAlphaHooks, alpha_grid_override: np.ndarray | None = None) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    cfg = wave_repair.configuration(assets)
    scenario = reviewer.make_local_wave_scenario(assets, alpha_grid_override) if alpha_grid_override is not None else reviewer.wave_v3.scenario_from_assets(assets, cfg)
    cfg = scenario.cfg
    alpha_grid = torch.as_tensor(scenario.alpha_grid, dtype=torch.float64, device=device)
    branches = torch.as_tensor(scenario.branch_initial, dtype=torch.float64, device=device)
    shadow = branches.clone(); scores = torch.zeros(cfg.n_alpha, dtype=torch.float64, device=device); weights = torch.softmax(scores, 0)
    alpha_members = _member_alpha_cloud(alpha_grid, cfg.ensemble_size, hooks, scores)
    truth = torch.as_tensor(assets.truth_states, dtype=torch.float64, device=device)
    indices = torch.as_tensor(assets.observation_indices, dtype=torch.int64, device=device)
    operator = SparseObservation(indices); covariance = cfg.obs_noise**2 * torch.eye(indices.numel(), dtype=torch.float64, device=device)
    metrics = wave_repair.MetricAccumulator(assets.nx); traces_w=[]; traces_a=[]; started=time.perf_counter()
    evidence_config = reviewer.wave_v4.ABLATION_CONFIGS["A6_pce" if method == "pce" else "A7_apce"]
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    for step in range(assets.n_steps + 1):
        flat=branches.reshape(-1, branches.shape[-1]); flat_weights=weights[:,None].expand_as(alpha_members).reshape(-1)/cfg.ensemble_size
        metrics.add(flat, truth[step], flat_weights); traces_w.append(weights.cpu().numpy()); traces_a.append(alpha_members.cpu().numpy())
        if step == assets.n_steps: break
        for i in range(cfg.n_alpha):
            branches[i]=reviewer.wave_memberwise_propagate(branches[i], alpha_members[i], step, assets, cfg)
            shadow[i]=reviewer.wave_memberwise_propagate(shadow[i], alpha_members[i], step, assets, cfg)
        if not assets.observation_mask[step+1]: continue
        branch_np=branches.detach().cpu().numpy(); shadow_np=shadow.detach().cpu().numpy()
        scores_np, weights_np = wave_repair.update_v4_weights(branch_np, shadow_np, assets.observations[step+1], cfg, scenario, evidence_config, scores.detach().cpu().numpy(), step+1)
        scores=torch.as_tensor(scores_np,dtype=torch.float64,device=device); weights=torch.as_tensor(weights_np,dtype=torch.float64,device=device)
        observation=torch.as_tensor(assets.observations[step+1],dtype=torch.float64,device=device)
        branches, alpha_members = _statealpha_analysis(branches, alpha_members, weights, observation, operator, covariance, float(alpha_grid[0]), float(alpha_grid[-1]), hooks, denkf_analysis)
    if device.type == "cuda": torch.cuda.synchronize(device)
    result=metrics.finalize(); result.update(nrmse=result.pop("displacement_nrmse"),rmse=result.pop("displacement_rmse"),runtime_seconds=float(time.perf_counter()-started),forward_member_steps=int(2*assets.n_steps*cfg.n_alpha*cfg.ensemble_size),peak_gpu_memory_mb=float(torch.cuda.max_memory_allocated(device)/1024**2) if device.type=="cuda" else 0.0)
    return result, scenario.alpha_grid.copy(), weights.cpu().numpy(), scores.cpu().numpy(), {"alpha_weights":np.asarray(traces_w),"alpha_members":np.asarray(traces_a)}


def run_statealpha_refined(case: str, method: str, seed: int, device: torch.device, profile: dict[str, Any]) -> dict[str, Any]:
    hooks = StateAlphaHooks.from_profile(profile, method)
    source = "pce" if method == "pce" else "apce"
    if case == "wave":
        assets = modern.make_wave_assets(seed)
        coarse, grid, weights, scores, coarse_trace = _wave_pass(assets, source, device, hooks)
        local = reviewer.adaptive_local_alpha_grid(grid, weights, scores, points=11)
        final, final_grid, final_weights, final_scores, final_trace = _wave_pass(assets, source, device, hooks, local)
        true_alpha = assets.alpha_true
    else:
        scenario = sh.generate_scenario(sh.config_for_case(case, seed), device)
        coarse, grid, weights, scores, coarse_trace = _spring_heat_pass(scenario, source, device, hooks)
        local = reviewer.adaptive_local_alpha_grid(grid, weights, scores, points=11)
        final, final_grid, final_weights, final_scores, final_trace = _spring_heat_pass(scenario, source, device, hooks, local)
        true_alpha = scenario.config.alpha_true
    alpha_estimate = reviewer.weighted_alpha_mean(final_grid, final_weights)
    final.update(alpha_estimate=float(alpha_estimate), alpha_absolute_error=abs(float(alpha_estimate)-float(true_alpha)), alpha_final_entropy=float(-np.sum(np.maximum(final_weights,1e-300)*np.log(np.maximum(final_weights,1e-300)))), coarse_pass_forward_member_steps=int(coarse["forward_member_steps"]), local_pass_forward_member_steps=int(final["forward_member_steps"]), forward_member_steps=int(coarse["forward_member_steps"]+final["forward_member_steps"]), runtime_seconds=float(coarse["runtime_seconds"]+final["runtime_seconds"]), local_alpha_grid_min=float(np.min(final_grid)), local_alpha_grid_max=float(np.max(final_grid)), local_alpha_grid_points=int(len(final_grid)), tuning_profile=profile["name"], statealpha_hooks=hooks.asdict(), refinement_source_method=source)
    final["_statealpha_trace"] = {"coarse_alpha_weights": coarse_trace["alpha_weights"], "coarse_alpha_members": coarse_trace["alpha_members"], "local_alpha_weights": final_trace["alpha_weights"], "local_alpha_members": final_trace["alpha_members"], "local_alpha_grid": final_grid, "local_log_scores": final_scores}
    return final
