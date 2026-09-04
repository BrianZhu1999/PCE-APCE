from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import run_figure2_reviewer_gate as gate  # noqa: E402
from experiments import run_modern_baseline_admission as modern  # noqa: E402
from experiments import run_wave_repair_validation as wave_repair  # noqa: E402
from hilda_da.baselines import denkf_analysis  # noqa: E402
from hilda_da.observations import SparseObservation  # noqa: E402
from paper_experiments import run_spring_heat_gate as sh  # noqa: E402


NEW_METHODS = ("aug_enkf", "bma_static", "pce_refined_v2", "apce_refined_v2")
ALL_METHODS = ("denkf", "letkf", "iensf", *NEW_METHODS)


def _wave_assets(seed: int) -> modern.WaveScenarioAssets:
    return modern.make_wave_assets(seed)


def _wave_aug_trace(assets, device: torch.device) -> np.ndarray:
    cfg = wave_repair.configuration(assets)
    dtype = torch.float64
    ensemble = torch.as_tensor(assets.initial_ensemble, dtype=dtype, device=device)
    truth = torch.as_tensor(assets.truth_states, dtype=dtype, device=device)
    indices = torch.as_tensor(assets.observation_indices, dtype=torch.int64, device=device)
    alpha_grid = torch.as_tensor(
        np.linspace(cfg.alpha_min, cfg.alpha_max, cfg.n_alpha), dtype=dtype, device=device
    )
    generator = torch.Generator(device=device).manual_seed(assets.seed + 741_101)
    alpha = gate.tile_alpha_members(alpha_grid, assets.ensemble_size, generator, jitter=0.012)
    operator = gate.AugmentedSparseObservation(indices)
    covariance = cfg.obs_noise**2 * torch.eye(indices.numel(), dtype=dtype, device=device)
    means: list[np.ndarray] = []
    for step in range(assets.n_steps + 1):
        means.append(ensemble.mean(dim=0).detach().cpu().numpy())
        if step == assets.n_steps:
            break
        alpha = gate.random_walk_alpha(
            alpha,
            generator,
            lower=float(alpha_grid[0]),
            upper=float(alpha_grid[-1]),
            std=0.004,
        )
        ensemble = gate.wave_memberwise_propagate(ensemble, alpha, step, assets, cfg)
        if not assets.observation_mask[step + 1]:
            continue
        observation = torch.as_tensor(assets.observations[step + 1], dtype=dtype, device=device)
        augmented = torch.cat([ensemble, alpha[:, None]], dim=-1)
        updated = denkf_analysis(augmented, observation, operator, covariance)
        ensemble = updated[:, :-1]
        alpha = updated[:, -1].clamp(float(alpha_grid[0]), float(alpha_grid[-1]))
    return np.stack(means)


def _wave_bma_trace(assets, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    cfg = wave_repair.configuration(assets)
    scenario = gate.wave_v3.scenario_from_assets(assets, cfg)
    branches = scenario.branch_initial.copy()
    log_weights = np.zeros(cfg.n_alpha, dtype=float)
    weights = gate.softmax_np(log_weights)
    truth = torch.as_tensor(assets.truth_states, dtype=torch.float64, device=device)
    indices = torch.as_tensor(assets.observation_indices, dtype=torch.int64, device=device)
    operator = SparseObservation(indices)
    covariance = cfg.obs_noise**2 * torch.eye(indices.numel(), dtype=torch.float64, device=device)
    means: list[np.ndarray] = []
    history: list[np.ndarray] = []
    evidence_config = gate.wave_v4.V4EvidenceConfig(
        gaussian_evidence=True, shrinkage=0.35, sensitivity_floor=1.0
    )
    for step in range(assets.n_steps + 1):
        flat = torch.as_tensor(branches.reshape(-1, branches.shape[-1]), dtype=torch.float64, device=device)
        member_weights = torch.as_tensor(
            np.repeat(weights / cfg.ensemble_size, cfg.ensemble_size),
            dtype=torch.float64,
            device=device,
        )
        means.append((member_weights[:, None] * flat).sum(dim=0).detach().cpu().numpy())
        history.append(weights.copy())
        if step == assets.n_steps:
            break
        for q, theta in enumerate(scenario.theta_grid):
            branches[q] = gate.wave_v3.propagate_batch(
                branches[q],
                float(theta),
                float(assets.times[step]),
                cfg,
                np.random.default_rng(assets.seed),
                stochastic=True,
                noise_draw=assets.forecast_noise[step],
            )
        if not assets.observation_mask[step + 1]:
            continue
        observation = assets.observations[step + 1]
        branch_observations = [branches[q][:, scenario.observation_indices].copy() for q in range(cfg.n_alpha)]
        evidence = gate.wave_v4.evidence_vector(
            branch_observations, observation, cfg.obs_noise, evidence_config
        )
        log_weights = log_weights + (evidence - np.mean(evidence))
        weights = gate.softmax_np(log_weights)
        observation_t = torch.as_tensor(observation, dtype=torch.float64, device=device)
        for q in range(cfg.n_alpha):
            branch_t = torch.as_tensor(branches[q], dtype=torch.float64, device=device)
            branches[q] = (
                denkf_analysis(branch_t, observation_t, operator, covariance)
                .detach()
                .cpu()
                .numpy()
            )
    return np.stack(means), np.stack(history)


def _wave_pce_pass_trace(
    assets,
    method: str,
    device: torch.device,
    alpha_grid_override: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cfg = wave_repair.configuration(assets)
    scenario = (
        gate.wave_v3.scenario_from_assets(assets, cfg)
        if alpha_grid_override is None
        else gate.make_local_wave_scenario(assets, np.asarray(alpha_grid_override, dtype=float))
    )
    cfg = scenario.cfg
    config = gate.wave_v4.ABLATION_CONFIGS["A6_pce" if method == "pce" else "A7_apce"]
    branches = scenario.branch_initial.copy()
    shadow = branches.copy()
    weights = np.full(cfg.n_alpha, 1.0 / cfg.n_alpha)
    log_weights = np.zeros(cfg.n_alpha)
    means: list[np.ndarray] = []
    history: list[np.ndarray] = []
    truth = torch.as_tensor(assets.truth_states, dtype=torch.float64, device=device)
    for step in range(assets.n_steps + 1):
        flat = torch.as_tensor(branches.reshape(-1, branches.shape[-1]), dtype=torch.float64, device=device)
        member_weights = torch.as_tensor(
            np.repeat(weights / cfg.ensemble_size, cfg.ensemble_size),
            dtype=torch.float64,
            device=device,
        )
        means.append((member_weights[:, None] * flat).sum(dim=0).detach().cpu().numpy())
        history.append(weights.copy())
        if step == assets.n_steps:
            break
        for q, theta in enumerate(scenario.theta_grid):
            branches[q] = gate.wave_v3.propagate_batch(
                branches[q],
                float(theta),
                float(assets.times[step]),
                cfg,
                np.random.default_rng(assets.seed),
                stochastic=True,
                noise_draw=assets.forecast_noise[step],
            )
            shadow[q] = gate.wave_v3.propagate_batch(
                shadow[q],
                float(theta),
                float(assets.times[step]),
                cfg,
                np.random.default_rng(assets.seed),
                stochastic=True,
                noise_draw=assets.forecast_noise[step],
            )
        if not assets.observation_mask[step + 1]:
            continue
        log_weights, weights = wave_repair.update_v4_weights(
            branches,
            shadow,
            assets.observations[step + 1],
            cfg,
            scenario,
            config,
            log_weights,
            step + 1,
        )
        paired_seed = cfg.seed + 10_000_000 + step + 1
        for q in range(cfg.n_alpha):
            branches[q] = wave_repair.ensf_update_lr(
                branches[q],
                assets.observations[step + 1],
                scenario.observation_indices,
                cfg,
                np.random.default_rng(paired_seed),
            )
    return np.stack(means), np.stack(history), scenario.alpha_grid.copy(), weights.copy()


def _wave_refined_v2_trace(assets, method: str, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    source = "pce" if method == "pce_refined_v2" else "apce"
    _, alpha_grid, weights, log_weights = gate.run_wave_pce_pass(assets, source, device)
    local_grid = gate.adaptive_local_alpha_grid(alpha_grid, weights, log_weights, points=11)
    trace, history, _, _ = _wave_pce_pass_trace(assets, source, device, local_grid)
    return trace, history


def _spring_heat_pce_pass_trace(
    scenario: sh.Scenario,
    method: str,
    device: torch.device,
    alpha_grid_override: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    config = scenario.config
    system = sh.make_system(config)
    if isinstance(system, sh.Heat1D):
        system.grid = system.grid.to(device)
    alpha_grid = (
        torch.as_tensor(alpha_grid_override, dtype=scenario.alpha_grid.dtype, device=device)
        if alpha_grid_override is not None
        else scenario.alpha_grid.to(device)
    )
    path_count = int(alpha_grid.numel())
    branches = scenario.initial_ensemble.unsqueeze(0).repeat(path_count, 1, 1)
    shadow = branches.clone()
    log_weights = torch.zeros(path_count, dtype=branches.dtype, device=device)
    weights = torch.softmax(log_weights, dim=0)
    operator = SparseObservation(scenario.observation_indices)
    covariance = config.obs_noise**2 * torch.eye(
        scenario.observation_indices.numel(), dtype=branches.dtype, device=device
    )
    means: list[np.ndarray] = []
    history: list[np.ndarray] = []
    for step in range(config.steps + 1):
        flat = branches.reshape(-1, branches.shape[-1])
        flat_weights = weights.unsqueeze(1).expand(-1, config.ensemble_size).reshape(-1) / config.ensemble_size
        means.append((flat_weights[:, None] * flat).sum(dim=0).detach().cpu().numpy())
        history.append(weights.detach().cpu().numpy())
        if step == config.steps:
            break
        for path_index, alpha in enumerate(alpha_grid):
            branches[path_index] = sh.step_with_noise(
                system,
                branches[path_index],
                step * config.dt,
                config.dt,
                float(alpha),
                scenario.forecast_noise[step],
            )
            shadow[path_index] = sh.step_with_noise(
                system,
                shadow[path_index],
                step * config.dt,
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
            dimension_weights = 0.35 + 0.65 * between / between.max().clamp_min(1.0e-12)
        evidence = torch.stack(
            [
                sh.evidence_score(
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
            entropy_ratio = float(sh.entropy(weights) / np.log(path_count))
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
            target_entropy = config.apce_entropy_floor + 0.20 * (1.0 - progress)
            weights = sh.entropy_project(weights, target_entropy)
            log_weights = weights.clamp_min(1.0e-300).log()
        for path_index in range(path_count):
            branches[path_index] = denkf_analysis(branches[path_index], observation, operator, covariance)
    return np.stack(means), np.stack(history), alpha_grid.detach().cpu().numpy(), weights.detach().cpu().numpy()


def _spring_heat_aug_trace(scenario: sh.Scenario, device: torch.device) -> np.ndarray:
    config = scenario.config
    system = sh.make_system(config)
    if isinstance(system, sh.Heat1D):
        system.grid = system.grid.to(device)
    ensemble = scenario.initial_ensemble.clone()
    generator = torch.Generator(device=device).manual_seed(config.seed + 741_001)
    alpha = gate.tile_alpha_members(scenario.alpha_grid.to(device), config.ensemble_size, generator, jitter=0.012)
    operator = gate.AugmentedSparseObservation(scenario.observation_indices)
    covariance = config.obs_noise**2 * torch.eye(
        scenario.observation_indices.numel(), dtype=ensemble.dtype, device=device
    )
    means: list[np.ndarray] = []
    for step in range(config.steps + 1):
        means.append(ensemble.mean(dim=0).detach().cpu().numpy())
        if step == config.steps:
            break
        alpha = gate.random_walk_alpha(
            alpha,
            generator,
            lower=float(scenario.alpha_grid[0]),
            upper=float(scenario.alpha_grid[-1]),
            std=0.004,
        )
        ensemble = gate.propagate_spring_heat_memberwise(scenario, ensemble, alpha, step)
        if step + 1 not in scenario.observations:
            continue
        augmented = torch.cat([ensemble, alpha[:, None]], dim=-1)
        updated = denkf_analysis(augmented, scenario.observations[step + 1], operator, covariance)
        ensemble = updated[:, :-1]
        alpha = updated[:, -1].clamp(float(scenario.alpha_grid[0]), float(scenario.alpha_grid[-1]))
    return np.stack(means)


def _spring_heat_bma_trace(scenario: sh.Scenario, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    config = scenario.config
    system = sh.make_system(config)
    if isinstance(system, sh.Heat1D):
        system.grid = system.grid.to(device)
    path_count = int(scenario.alpha_grid.numel())
    branches = scenario.initial_ensemble.unsqueeze(0).repeat(path_count, 1, 1)
    log_weights = torch.zeros(path_count, dtype=branches.dtype, device=device)
    weights = torch.softmax(log_weights, dim=0)
    operator = SparseObservation(scenario.observation_indices)
    covariance = config.obs_noise**2 * torch.eye(
        scenario.observation_indices.numel(), dtype=branches.dtype, device=device
    )
    means: list[np.ndarray] = []
    history: list[np.ndarray] = []
    for step in range(config.steps + 1):
        flat = branches.reshape(-1, branches.shape[-1])
        flat_weights = weights.unsqueeze(1).expand(-1, config.ensemble_size).reshape(-1) / config.ensemble_size
        means.append((flat_weights[:, None] * flat).sum(dim=0).detach().cpu().numpy())
        history.append(weights.detach().cpu().numpy())
        if step == config.steps:
            break
        for path_index, alpha in enumerate(scenario.alpha_grid):
            branches[path_index] = sh.step_with_noise(
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
                sh.evidence_score(
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
        weights = torch.softmax(log_weights, dim=0)
        for path_index in range(path_count):
            branches[path_index] = denkf_analysis(branches[path_index], observation, operator, covariance)
    return np.stack(means), np.stack(history)


def _spring_heat_refined_v2_trace(scenario: sh.Scenario, method: str, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    source = "pce" if method == "pce_refined_v2" else "apce"
    _, alpha_grid, weights, log_weights = gate.run_spring_heat_pce_pass(scenario, source, device)
    alpha_grid = np.asarray(alpha_grid, dtype=float)
    weights = np.asarray(weights, dtype=float)
    log_weights = np.asarray(log_weights, dtype=float)
    local_grid = gate.adaptive_local_alpha_grid(alpha_grid, weights, log_weights, points=11)
    trace, history, _, _ = _spring_heat_pce_pass_trace(scenario, source, device, local_grid)
    return trace, history


def export_case(case: str, seed: int, output: Path, device: torch.device) -> dict[str, str]:
    if case == "wave":
        assets = _wave_assets(seed)
        arrays: dict[str, np.ndarray] = {
            "times": assets.times,
            "truth_states": assets.truth_states,
            "observation_indices": assets.observation_indices,
            "observation_mask": assets.observation_mask,
            "alpha_true": np.asarray(assets.alpha_true),
            "seed": np.asarray(seed),
        }
        arrays["denkf_mean_states"] = np.load(
            ROOT / "figures" / "figure2_full_representative_source" / f"wave_full_representative_seed_{seed}.npz",
            allow_pickle=True,
        )["denkf_mean_states"]
        old = np.load(
            ROOT / "figures" / "figure2_full_representative_source" / f"wave_full_representative_seed_{seed}.npz",
            allow_pickle=True,
        )
        arrays["letkf_mean_states"] = old["letkf_mean_states"]
        arrays["iensf_mean_states"] = old["iensf_mean_states"]
        arrays["aug_enkf_mean_states"] = _wave_aug_trace(assets, device)
        bma_trace, bma_history = _wave_bma_trace(assets, device)
        arrays["bma_static_mean_states"] = bma_trace
        pce_trace, pce_history = _wave_refined_v2_trace(assets, "pce_refined_v2", device)
        apce_trace, apce_history = _wave_refined_v2_trace(assets, "apce_refined_v2", device)
        arrays["pce_refined_v2_mean_states"] = pce_trace
        arrays["apce_refined_v2_mean_states"] = apce_trace
        arrays["bma_static_alpha_weight_history"] = bma_history
        arrays["pce_refined_v2_alpha_weight_history"] = pce_history
        arrays["apce_refined_v2_alpha_weight_history"] = apce_history
        arrays["pce_mean_states"] = pce_trace
        arrays["apce_mean_states"] = apce_trace
        arrays["pce_alpha_weight_history"] = pce_history
        arrays["apce_alpha_weight_history"] = apce_history
    else:
        config = sh.config_for_case(case, seed)
        scenario = sh.generate_scenario(config, device)
        arrays = {
            "times": np.arange(config.steps + 1, dtype=float) * config.dt,
            "truth_states": scenario.truth.detach().cpu().numpy(),
            "observation_indices": scenario.observation_indices.detach().cpu().numpy(),
            "primary_indices": scenario.primary_indices.detach().cpu().numpy(),
            "alpha_grid": scenario.alpha_grid.detach().cpu().numpy(),
            "seed": np.asarray(seed),
        }
        old = np.load(
            ROOT / "figures" / "figure2_full_representative_source" / f"{case}_full_representative_seed_{seed}.npz",
            allow_pickle=True,
        )
        arrays["denkf_mean_states"] = old["denkf_mean_states"]
        arrays["letkf_mean_states"] = old["letkf_mean_states"]
        arrays["iensf_mean_states"] = old["iensf_mean_states"]
        arrays["aug_enkf_mean_states"] = _spring_heat_aug_trace(scenario, device)
        bma_trace, bma_history = _spring_heat_bma_trace(scenario, device)
        arrays["bma_static_mean_states"] = bma_trace
        pce_trace, pce_history = _spring_heat_refined_v2_trace(scenario, "pce_refined_v2", device)
        apce_trace, apce_history = _spring_heat_refined_v2_trace(scenario, "apce_refined_v2", device)
        arrays["pce_refined_v2_mean_states"] = pce_trace
        arrays["apce_refined_v2_mean_states"] = apce_trace
        arrays["bma_static_alpha_weight_history"] = bma_history
        arrays["pce_refined_v2_alpha_weight_history"] = pce_history
        arrays["apce_refined_v2_alpha_weight_history"] = apce_history
        arrays["pce_mean_states"] = pce_trace
        arrays["apce_mean_states"] = apce_trace
        arrays["pce_alpha_weight_history"] = pce_history
        arrays["apce_alpha_weight_history"] = apce_history
        if case == "heat":
            arrays["space"] = np.linspace(0.0, 1.0, scenario.truth.shape[-1])
        else:
            arrays["state_components"] = np.asarray(["displacement", "velocity"])
    path = output / f"{case}_v2_representative_seed_{seed}.npz"
    np.savez_compressed(path, **arrays)
    return {"case": case, "path": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export representative Figure 2 V2 traces for all new methods.")
    parser.add_argument("--seed", type=int, default=2026080700)
    parser.add_argument("--output", type=Path, default=ROOT / "figures" / "figure2_v2_representative_source")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    manifest = {
        "seed": args.seed,
        "device": str(device),
        "methods": list(ALL_METHODS),
        "cases": [export_case(case, args.seed, args.output, device) for case in ("wave", "spring", "heat")],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
