from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_benchmark_v3 as v3
from experiments.wave_scenario_assets import WaveScenarioAssets
from hilda_da.baselines import denkf_analysis, letkf_analysis
from hilda_da.observations import SparseObservation
from hilda_da.strong_baselines import EnSFConfig, IEnSFConfig, ensf_analysis, ensf_lr_ridge_analysis, iensf_analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    record = manifest["records"][0]
    assets = WaveScenarioAssets.load(Path(record["path"]))
    cfg = dataclasses.replace(
        v3.make_config("quick"),
        seed=assets.seed,
        nx=assets.nx,
        ensemble_size=assets.ensemble_size,
        alpha_true=assets.alpha_true,
        t_end=float(assets.times[-1]),
        dt=float(assets.times[1] - assets.times[0]),
        n_sensors=int(assets.observation_indices.size),
        obs_interval=20,
    )
    scenario = v3.scenario_from_assets(assets, cfg)
    original = {}
    for method in ("enkf", "ensf_direct", "ensf_lr", "alpha_only", "alpha_ensf_lr_pce"):
        result = v3.run_method(scenario, method)
        original[method] = float(result["metrics"]["mean_rmse"])
        if not np.isfinite(original[method]):
            raise RuntimeError(f"non-finite original method result: {method}")

    step = int(np.flatnonzero(assets.observation_mask)[0])
    forecast = torch.as_tensor(assets.initial_ensemble, dtype=torch.float64)
    for index in range(step):
        forecast = torch.as_tensor(
            v3.propagate_batch(
                forecast.numpy(), 0.0, scenario.times[index], cfg,
                np.random.default_rng(assets.seed), stochastic=True,
                noise_draw=assets.forecast_noise[index],
            ), dtype=torch.float64,
        )
    observation = torch.as_tensor(assets.observations[step], dtype=torch.float64)
    operator = SparseObservation(torch.as_tensor(assets.observation_indices, dtype=torch.int64))
    covariance = (cfg.obs_noise ** 2) * torch.eye(assets.observation_indices.size, dtype=torch.float64)
    generator = torch.Generator().manual_seed(assets.seed)
    torch_methods = {
        "denkf": lambda: denkf_analysis(forecast.clone(), observation, operator, covariance),
        "letkf": lambda: letkf_analysis(forecast.clone(), observation, operator, covariance),
        "ensf": lambda: ensf_analysis(forecast.clone(), observation, operator, covariance, EnSFConfig(sampling_time_step_count=8), generator),
        "iensf": lambda: iensf_analysis(forecast.clone(), observation, operator, covariance, IEnSFConfig(sampling_time_step_count=8, refinement_iterations=1), generator),
        "ensf_lr_ridge": lambda: ensf_lr_ridge_analysis(forecast.clone(), observation, operator, covariance, generator=generator),
    }
    for method, call in torch_methods.items():
        value = call()
        if value.shape != forecast.shape or not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"invalid loaded-asset output: {method}")
    print("LOADED_ASSET", record["name"], assets.array_digest[:16], assets.nx)
    print("ORIGINAL_METHODS", sorted(original), "PCE", original["alpha_ensf_lr_pce"])
    print("TORCH_METHODS", sorted(torch_methods), "OBS_STEP", step)


if __name__ == "__main__":
    main()
