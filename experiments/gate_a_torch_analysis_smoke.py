from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_benchmark_v3 as v3
from experiments.wave_scenario_assets import WaveScenarioAssets
from hilda_da.alpha import AlphaEvidenceTracker
from hilda_da.baselines import denkf_analysis, letkf_analysis
from hilda_da.config import AlphaConfig, HILDAConfig
from hilda_da.filter import HILDAFilter
from hilda_da.observations import SparseObservation
from hilda_da.strong_baselines import (
    EnSFConfig,
    IEnSFConfig,
    ensf_analysis,
    ensf_lr_ridge_analysis,
    iensf_analysis,
)


def main() -> None:
    cfg = dataclasses.replace(
        v3.make_config("quick"),
        seed=2026080602,
        nx=41,
        ensemble_size=18,
        n_alpha=7,
        t_end=0.05,
        dt=0.0025,
        obs_interval=5,
        n_sensors=6,
    )
    assets = WaveScenarioAssets.from_legacy_scenario(v3.generate_scenario(cfg))
    scenario = v3.scenario_from_assets(assets, cfg)
    step = 5
    # The common forecast is the fixed-alpha baseline branch. Every analysis
    # method below receives this same state, observation and covariance.
    forecast = scenario.ensemble_initial.copy()
    for index in range(step):
        forecast = v3.propagate_batch(
            forecast, 0.0, scenario.times[index], cfg,
            np.random.default_rng(2026080602),
            stochastic=True, noise_draw=assets.forecast_noise[index],
        )
    dtype = torch.float64
    state = torch.as_tensor(forecast, dtype=dtype)
    observation = torch.as_tensor(assets.observations[step], dtype=dtype)
    obs_indices = torch.as_tensor(assets.observation_indices, dtype=torch.int64)
    operator = SparseObservation(obs_indices)
    covariance = (cfg.obs_noise ** 2) * torch.eye(obs_indices.numel(), dtype=dtype)
    generator = torch.Generator().manual_seed(2026080602)

    methods = {
        "denkf": lambda: denkf_analysis(state.clone(), observation, operator, covariance),
        "letkf": lambda: letkf_analysis(state.clone(), observation, operator, covariance),
        "ensf": lambda: ensf_analysis(
            state.clone(), observation, operator, covariance,
            config=EnSFConfig(sampling_time_step_count=8), generator=generator,
        ),
        "iensf": lambda: iensf_analysis(
            state.clone(), observation, operator, covariance,
            config=IEnSFConfig(sampling_time_step_count=8, refinement_iterations=1),
            generator=generator,
        ),
        "ensf_lr_ridge": lambda: ensf_lr_ridge_analysis(
            state.clone(), observation, operator, covariance, generator=generator,
        ),
    }
    print("TORCH_COMMON", assets.nx, state.shape, observation.shape)
    for name, call in methods.items():
        try:
            result = call()
            finite = bool(torch.isfinite(result).all())
            if result.shape != state.shape or not finite:
                raise RuntimeError(f"shape={tuple(result.shape)} finite={finite}")
            print("ANALYSIS_OK", name, tuple(result.shape))
        except Exception as exc:  # interface audit records numerical blockers
            print("ANALYSIS_BLOCKED", name, type(exc).__name__, str(exc)[:180])

    fixed_alpha = AlphaConfig(initial_nodes=9, max_nodes=9, prune_threshold=0.0)
    hilda = HILDAFilter(HILDAConfig(alpha=fixed_alpha))
    tracker = AlphaEvidenceTracker.create(hilda.config.alpha, dtype=dtype)
    branches = []
    for alpha in tracker.alpha.tolist():
        branch = scenario.ensemble_initial.copy()
        theta = v3.alpha_to_theta(alpha, cfg)
        for index in range(step):
            branch = v3.propagate_batch(
                branch, theta, scenario.times[index], cfg,
                np.random.default_rng(2026080602),
                stochastic=True, noise_draw=assets.forecast_noise[index],
            )
        branches.append(torch.as_tensor(branch, dtype=dtype))
    try:
        analyzed = hilda.analyze_paths(
            torch.stack(branches), tracker, observation, operator, covariance
        )
        finite = bool(torch.isfinite(analyzed.ensembles).all())
        if not finite:
            raise RuntimeError("HILDA output is non-finite")
        if analyzed.ensembles.shape[0] != 9:
            raise RuntimeError(f"fixed alpha path count changed: {analyzed.ensembles.shape[0]}")
        print("ANALYSIS_OK", "hilda_crn_fixed", tuple(analyzed.ensembles.shape))
    except Exception as exc:
        print("ANALYSIS_BLOCKED", "hilda", type(exc).__name__, str(exc)[:180])


if __name__ == "__main__":
    main()
