from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import run_modern_baseline_admission as modern  # noqa: E402
from experiments import run_wave_repair_validation as wave_repair  # noqa: E402
from experiments.run_figure2_formal_job import run_wave_oracle  # noqa: E402
from hilda_da.baselines import denkf_analysis  # noqa: E402
from hilda_da.observations import SparseObservation  # noqa: E402
from paper_experiments import run_spring_heat_gate as sh  # noqa: E402


METHODS = ("denkf", "letkf", "iensf", "pce", "apce", "oracle_alpha")
METHOD_LABELS = {
    "denkf": "DEnKF",
    "letkf": "LETKF",
    "iensf": "IEnSF",
    "pce": "PCE",
    "apce": "APCE",
    "oracle_alpha": "Oracle-alpha",
}


def pop_numeric_metrics(result: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in result.items():
        if isinstance(value, (int, float, np.floating)):
            metrics[key] = float(value)
    return metrics


def trace_wave_strong(method: str, seed: int, device: torch.device) -> tuple[np.ndarray, dict[str, float]]:
    assets = modern.make_wave_assets(seed)
    cfg = wave_repair.configuration(assets)
    dtype = torch.float64
    ensemble = torch.as_tensor(assets.initial_ensemble, dtype=dtype, device=device)
    truth = torch.as_tensor(assets.truth_states, dtype=dtype, device=device)
    operator = SparseObservation(torch.as_tensor(assets.observation_indices, dtype=torch.int64, device=device))
    covariance = cfg.obs_noise**2 * torch.eye(assets.observation_indices.size, dtype=dtype, device=device)
    generator = torch.Generator(device=device).manual_seed(seed + 910_000)
    metrics = modern.Metrics(assets.nx)
    means: list[np.ndarray] = []
    for step in range(assets.n_steps + 1):
        means.append(ensemble.mean(dim=0).detach().cpu().numpy())
        metrics.add(ensemble, truth[step])
        if step == assets.n_steps:
            break
        propagated = modern.wave_v3.propagate_batch(
            ensemble.detach().cpu().numpy(),
            modern.wave_v3.alpha_to_theta(0.50, cfg),
            float(assets.times[step]),
            cfg,
            np.random.default_rng(seed),
            stochastic=True,
            noise_draw=assets.forecast_noise[step],
        )
        ensemble = torch.as_tensor(propagated, dtype=dtype, device=device)
        if not assets.observation_mask[step + 1]:
            continue
        observation = torch.as_tensor(assets.observations[step + 1], dtype=dtype, device=device)
        ensemble = modern.apply_analysis(method, ensemble, observation, operator, covariance, generator)
    return np.stack(means), metrics.finish()


def trace_wave_oracle(seed: int, device: torch.device) -> tuple[np.ndarray, dict[str, float]]:
    assets = modern.make_wave_assets(seed)
    cfg = wave_repair.configuration(assets)
    dtype = torch.float64
    ensemble = torch.as_tensor(assets.initial_ensemble, dtype=dtype, device=device)
    truth = torch.as_tensor(assets.truth_states, dtype=dtype, device=device)
    operator = SparseObservation(torch.as_tensor(assets.observation_indices, dtype=torch.int64, device=device))
    covariance = cfg.obs_noise**2 * torch.eye(assets.observation_indices.size, dtype=dtype, device=device)
    weights = torch.full((assets.ensemble_size,), 1.0 / assets.ensemble_size, dtype=dtype, device=device)
    metrics = wave_repair.MetricAccumulator(assets.nx)
    means: list[np.ndarray] = []
    for step in range(assets.n_steps + 1):
        means.append(ensemble.mean(dim=0).detach().cpu().numpy())
        metrics.add(ensemble, truth[step], weights)
        if step == assets.n_steps:
            break
        ensemble = wave_repair.propagate_numpy(
            ensemble,
            wave_repair.v3.alpha_to_theta(float(assets.alpha_true), cfg),
            step,
            assets,
            cfg,
        )
        if not assets.observation_mask[step + 1]:
            continue
        observation = torch.as_tensor(assets.observations[step + 1], dtype=dtype, device=device)
        ensemble = denkf_analysis(ensemble, observation, operator, covariance)
    result = metrics.finalize()
    result.update(alpha_estimate=float(assets.alpha_true), alpha_absolute_error=0.0)
    return np.stack(means), result


def export_wave(seed: int, output: Path, device: torch.device) -> dict[str, Any]:
    assets = modern.make_wave_assets(seed)
    arrays: dict[str, np.ndarray] = {
        "times": assets.times,
        "truth_states": assets.truth_states,
        "observation_indices": assets.observation_indices,
        "observation_mask": assets.observation_mask,
        "alpha_true": np.asarray(assets.alpha_true),
        "seed": np.asarray(seed),
    }
    metrics: dict[str, dict[str, float]] = {}
    for method in METHODS:
        if method in {"denkf", "letkf"}:
            trace = wave_repair.trace_single_path(assets, method, device)
            metric_row = modern.run_wave(method, seed, device)
        elif method == "iensf":
            trace, metric_row = trace_wave_strong(method, seed, device)
        elif method in {"pce", "apce"}:
            trace, weights = wave_repair.trace_pce_family(assets, method, device)
            arrays[f"{method}_final_weights"] = weights
            metric_row = wave_repair.run_pce_family(assets, method, device)
        elif method == "oracle_alpha":
            trace, metric_row = trace_wave_oracle(seed, device)
        else:
            raise ValueError(method)
        arrays[f"{method}_mean_states"] = trace
        metrics[method] = pop_numeric_metrics(metric_row)
    path = output / f"wave_full_representative_seed_{seed}.npz"
    np.savez_compressed(path, **arrays)
    return {"source": str(path), "metrics": metrics}


def trace_spring_heat_iensf(case: str, seed: int, device: torch.device) -> tuple[np.ndarray, dict[str, float]]:
    scenario = modern.make_spring_heat(case, seed, device)
    config = scenario.config
    system = sh.make_system(config)
    if isinstance(system, sh.Heat1D):
        system.grid = system.grid.to(device)
    ensemble = scenario.initial_ensemble.clone()
    operator = SparseObservation(scenario.observation_indices)
    covariance = config.obs_noise**2 * torch.eye(
        scenario.observation_indices.numel(),
        dtype=ensemble.dtype,
        device=device,
    )
    generator = torch.Generator(device=device).manual_seed(seed + 910_000)
    weights = torch.full((config.ensemble_size,), 1.0 / config.ensemble_size, dtype=ensemble.dtype, device=device)
    metrics = sh.TrajectoryMetrics()
    means: list[np.ndarray] = []
    for step in range(config.steps + 1):
        means.append(ensemble.mean(dim=0).detach().cpu().numpy())
        metrics.add(sh.primary(ensemble, scenario), sh.primary(scenario.truth[step], scenario), weights)
        if step == config.steps:
            break
        ensemble = sh.step_with_noise(
            system,
            ensemble,
            step * config.dt,
            config.dt,
            config.fixed_alpha,
            scenario.forecast_noise[step],
        )
        if step + 1 not in scenario.observations:
            continue
        ensemble = modern.apply_analysis(
            "iensf",
            ensemble,
            scenario.observations[step + 1],
            operator,
            covariance,
            generator,
        )
    return np.stack(means), metrics.finalize()


def export_spring_heat(case: str, seed: int, output: Path, device: torch.device) -> dict[str, Any]:
    config = sh.config_for_case(case, seed)
    scenario = sh.generate_scenario(config, device)
    arrays: dict[str, np.ndarray] = {
        "times": np.arange(config.steps + 1, dtype=float) * config.dt,
        "truth_states": scenario.truth.detach().cpu().numpy(),
        "observation_indices": scenario.observation_indices.detach().cpu().numpy(),
        "primary_indices": scenario.primary_indices.detach().cpu().numpy(),
        "alpha_grid": scenario.alpha_grid.detach().cpu().numpy(),
        "seed": np.asarray(seed),
    }
    metrics: dict[str, dict[str, float]] = {}
    for method in METHODS:
        if method == "iensf":
            trace, metric_row = trace_spring_heat_iensf(case, seed, device)
        else:
            result = sh.run_method(scenario, method, device, record_trace=True)
            trace = np.asarray(result.pop("mean_states"))
            if "alpha_weight_history" in result:
                arrays[f"{method}_alpha_weight_history"] = np.asarray(result.pop("alpha_weight_history"))
            metric_row = result
        arrays[f"{method}_mean_states"] = trace
        metrics[method] = pop_numeric_metrics(metric_row)
    if case == "heat":
        arrays["space"] = np.linspace(0.0, 1.0, scenario.truth.shape[-1])
    else:
        arrays["state_components"] = np.asarray(["displacement", "velocity"])
    path = output / f"{case}_full_representative_seed_{seed}.npz"
    np.savez_compressed(path, **arrays)
    return {"source": str(path), "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export full Figure 2 representative state traces.")
    parser.add_argument("--seed", type=int, default=2026080700)
    parser.add_argument("--output", type=Path, default=ROOT / "figures" / "figure2_full_representative_source")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    manifest = {
        "seed": args.seed,
        "device": str(device),
        "methods": {method: METHOD_LABELS[method] for method in METHODS},
        "wave": export_wave(args.seed, args.output, device),
        "spring": export_spring_heat("spring", args.seed, args.output, device),
        "heat": export_spring_heat("heat", args.seed, args.output, device),
    }
    manifest_path = args.output / "figure2_full_representative_source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
