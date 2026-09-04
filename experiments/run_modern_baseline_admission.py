from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_benchmark_v3 as wave_v3
from experiments.wave_scenario_assets import WaveScenarioAssets
from hilda_da.baselines import denkf_analysis, letkf_analysis
from hilda_da.metrics import (
    weighted_central_interval_coverage_width,
    weighted_ensemble_crps,
)
from hilda_da.observations import SparseObservation
from hilda_da.strong_baselines import (
    EnSFConfig,
    IEnSFConfig,
    ensf_analysis,
    ensf_lr_analysis,
    ensf_lr_ridge_analysis,
    iensf_analysis,
)
from hilda_da.systems.one_dimensional import (
    Heat1D,
    HeatConfig,
    SpringConfig,
    SpringOscillator,
)
from paper_experiments import run_spring_heat_gate as sh


METHODS = (
    "denkf",
    "letkf",
    "ensf",
    "iensf",
    "ensf_lr",
    "ensf_lr_ridge",
)
LABELS = {
    "denkf": "DEnKF",
    "letkf": "LETKF",
    "ensf": "EnSF",
    "iensf": "IEnSF",
    "ensf_lr": "EnSF-LR",
    "ensf_lr_ridge": "EnSF-LR-Ridge",
}


def source_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "hilda_da").rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


class Metrics:
    def __init__(self, primary_dim: int) -> None:
        self.primary_dim = primary_dim
        self.sq = 0.0
        self.truth_sq = 0.0
        self.points = 0
        self.crps: list[float] = []
        self.coverage: list[float] = []
        self.width: list[float] = []

    def add(self, ensemble: torch.Tensor, truth: torch.Tensor) -> None:
        mean = ensemble.mean(0)
        primary = ensemble[:, : self.primary_dim]
        target = truth[: self.primary_dim]
        self.sq += float((mean[: self.primary_dim] - target).square().sum())
        self.truth_sq += float(target.square().sum())
        self.points += int(target.numel())
        weights = torch.full(
            (ensemble.shape[0],),
            1.0 / ensemble.shape[0],
            dtype=ensemble.dtype,
            device=ensemble.device,
        )
        self.crps.append(float(weighted_ensemble_crps(primary, target, weights)))
        coverage, width = weighted_central_interval_coverage_width(
            primary, target, weights, level=0.90
        )
        self.coverage.append(float(coverage))
        self.width.append(float(width))

    def finish(self) -> dict[str, float]:
        return {
            "nrmse": math.sqrt(self.sq / max(self.truth_sq, 1e-30)),
            "rmse": math.sqrt(self.sq / max(self.points, 1)),
            "crps": float(np.mean(self.crps)),
            "coverage_90": float(np.mean(self.coverage)),
            "interval_width_90": float(np.mean(self.width)),
        }


def make_wave_assets(seed: int) -> WaveScenarioAssets:
    cfg = dataclasses.replace(
        wave_v3.make_config("quick"),
        seed=seed,
        nx=41,
        ensemble_size=18,
        n_alpha=7,
        t_end=1.0,
        dt=0.0025,
        obs_interval=20,
        n_sensors=6,
        alpha_true=0.12,
    )
    return WaveScenarioAssets.from_legacy_scenario(wave_v3.generate_scenario(cfg))


def make_spring_heat(case: str, seed: int, device: torch.device) -> sh.Scenario:
    config = sh.config_for_case(case, seed)  # type: ignore[arg-type]
    scenario = sh.generate_scenario(config, device)
    return scenario


def apply_analysis(
    method: str,
    ensemble: torch.Tensor,
    observation: torch.Tensor,
    operator: SparseObservation,
    covariance: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    if method == "denkf":
        return denkf_analysis(ensemble, observation, operator, covariance)
    if method == "letkf":
        return letkf_analysis(ensemble, observation, operator, covariance)
    if method == "ensf":
        return ensf_analysis(
            ensemble,
            observation,
            operator,
            covariance,
            EnSFConfig(sampling_time_step_count=20),
            generator,
        )
    if method == "iensf":
        return iensf_analysis(
            ensemble,
            observation,
            operator,
            covariance,
            IEnSFConfig(
                gamma=0.5,
                variance_split_mode="variance_consistent",
                sampling_time_step_count=20,
                refinement_iterations=2,
            ),
            generator,
        )
    if method == "ensf_lr":
        return ensf_lr_analysis(
            ensemble,
            observation,
            operator,
            covariance,
            EnSFConfig(sampling_time_step_count=20),
            generator,
        )
    if method == "ensf_lr_ridge":
        return ensf_lr_ridge_analysis(
            ensemble,
            observation,
            operator,
            covariance,
            EnSFConfig(sampling_time_step_count=20),
            generator,
        )
    raise ValueError(method)


def run_spring_heat(case: str, method: str, seed: int, device: torch.device) -> dict[str, Any]:
    scenario = make_spring_heat(case, seed, device)
    config = scenario.config
    system = sh.make_system(config)
    if isinstance(system, Heat1D):
        system.grid = system.grid.to(device)
    ensemble = scenario.initial_ensemble.clone()
    operator = SparseObservation(scenario.observation_indices)
    covariance = config.obs_noise**2 * torch.eye(
        scenario.observation_indices.numel(), dtype=ensemble.dtype, device=device
    )
    generator = torch.Generator(device=device).manual_seed(seed + 910_000)
    metrics = Metrics(int(scenario.primary_indices.numel()))
    max_abs = 0.0
    analyses = 0
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(config.steps + 1):
        metrics.add(ensemble.index_select(-1, scenario.primary_indices), scenario.truth[step].index_select(-1, scenario.primary_indices))
        max_abs = max(max_abs, float(ensemble.abs().max()))
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
        analyses += 1
        ensemble = apply_analysis(
            method,
            ensemble,
            scenario.observations[step + 1],
            operator,
            covariance,
            generator,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = metrics.finish()
    truth_max_abs = float(scenario.truth.abs().max())
    max_abs_ratio = max_abs / max(truth_max_abs, 1.0e-12)
    finite = bool(torch.isfinite(ensemble).all()) and all(
        math.isfinite(value) for value in result.values()
    )
    valid = finite and max_abs_ratio <= 100.0
    result.update(
        case=case,
        method=method,
        label=LABELS[method],
        seed=seed,
        state_dim=int(scenario.truth.shape[1]),
        primary_dim=int(scenario.primary_indices.numel()),
        observation_count=int(scenario.observation_indices.numel()),
        observation_indices=",".join(str(int(v)) for v in scenario.observation_indices.detach().cpu().tolist()),
        steps=int(config.steps),
        dt=float(config.dt),
        observation_interval=int(config.obs_interval),
        ensemble_size=int(config.ensemble_size),
        observation_noise=float(config.obs_noise),
        alpha_true=float(config.alpha_true),
        fixed_alpha=float(config.fixed_alpha),
        analyses=analyses,
        valid=valid,
        validity_reason=(
            "finite_and_bounded"
            if valid
            else ("nonfinite" if not finite else "state_amplitude_over_100x_truth")
        ),
        truth_max_abs=truth_max_abs,
        max_abs_ratio=max_abs_ratio,
        max_abs_state=max_abs,
        runtime_seconds=float(time.perf_counter() - start),
        forward_member_steps=int(config.steps * config.ensemble_size),
        peak_gpu_memory_mb=(
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda" else 0.0
        ),
    )
    return result


def run_wave(method: str, seed: int, device: torch.device) -> dict[str, Any]:
    assets = make_wave_assets(seed)
    cfg = dataclasses.replace(
        wave_v3.make_config("quick"),
        seed=seed,
        nx=assets.nx,
        ensemble_size=assets.ensemble_size,
        t_end=float(assets.times[-1]),
        dt=float(assets.times[1] - assets.times[0]),
        obs_interval=20,
        n_sensors=int(assets.observation_indices.size),
        alpha_true=assets.alpha_true,
    )
    ensemble = torch.as_tensor(assets.initial_ensemble, dtype=torch.float64, device=device)
    truth = torch.as_tensor(assets.truth_states, dtype=torch.float64, device=device)
    operator = SparseObservation(torch.as_tensor(assets.observation_indices, dtype=torch.int64, device=device))
    covariance = cfg.obs_noise**2 * torch.eye(assets.observation_indices.size, dtype=torch.float64, device=device)
    generator = torch.Generator(device=device).manual_seed(seed + 910_000)
    metrics = Metrics(assets.nx)
    max_abs = 0.0
    analyses = 0
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(assets.n_steps + 1):
        metrics.add(ensemble, truth[step])
        max_abs = max(max_abs, float(ensemble.abs().max()))
        if step == assets.n_steps:
            break
        propagated = wave_v3.propagate_batch(
            ensemble.detach().cpu().numpy(),
            wave_v3.alpha_to_theta(0.50, cfg),
            float(assets.times[step]),
            cfg,
            np.random.default_rng(seed),
            stochastic=True,
            noise_draw=assets.forecast_noise[step],
        )
        ensemble = torch.as_tensor(propagated, dtype=torch.float64, device=device)
        if not assets.observation_mask[step + 1]:
            continue
        analyses += 1
        observation = torch.as_tensor(assets.observations[step + 1], dtype=torch.float64, device=device)
        ensemble = apply_analysis(method, ensemble, observation, operator, covariance, generator)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = metrics.finish()
    truth_max_abs = float(truth.abs().max())
    max_abs_ratio = max_abs / max(truth_max_abs, 1.0e-12)
    finite = bool(torch.isfinite(ensemble).all()) and all(
        math.isfinite(value) for value in result.values()
    )
    valid = finite and max_abs_ratio <= 100.0
    result.update(
        case="wave",
        method=method,
        label=LABELS[method],
        seed=seed,
        state_dim=int(assets.truth_states.shape[1]),
        primary_dim=int(assets.nx),
        observation_count=int(assets.observation_indices.size),
        observation_indices=",".join(str(int(v)) for v in assets.observation_indices.tolist()),
        steps=int(assets.n_steps),
        dt=float(cfg.dt),
        observation_interval=20,
        ensemble_size=int(assets.ensemble_size),
        observation_noise=float(cfg.obs_noise),
        alpha_true=float(assets.alpha_true),
        fixed_alpha=0.50,
        analyses=analyses,
        valid=valid,
        validity_reason=(
            "finite_and_bounded"
            if valid
            else ("nonfinite" if not finite else "state_amplitude_over_100x_truth")
        ),
        truth_max_abs=truth_max_abs,
        max_abs_ratio=max_abs_ratio,
        max_abs_state=max_abs,
        runtime_seconds=float(time.perf_counter() - start),
        forward_member_steps=int(assets.n_steps * assets.ensemble_size),
        peak_gpu_memory_mb=(
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda" else 0.0
        ),
        asset_digest=assets.array_digest,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("wave", "spring", "heat"), required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, default=2026080600)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    project_root = Path(__file__).resolve().parents[1]
    result = run_wave(args.method, args.seed, device) if args.case == "wave" else run_spring_heat(args.case, args.method, args.seed, device)
    result["source_hash"] = source_hash(project_root)
    result["torch_version"] = torch.__version__
    result["cuda_available"] = bool(torch.cuda.is_available())
    result["device_name"] = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    result["protocol"] = "modern-baseline-admission-smoke-20260807-v1"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
