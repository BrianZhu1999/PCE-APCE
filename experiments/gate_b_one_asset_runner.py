from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
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
from hilda_da.metrics import (
    weighted_central_interval_coverage_width,
    weighted_ensemble_crps,
)
from hilda_da.observations import SparseObservation
from hilda_da.strong_baselines import (
    EnFFF2PConfig,
    EnSFConfig,
    IEnSFConfig,
    enff_f2p_analysis,
    ensf_analysis,
    ensf_lr_ridge_analysis,
    iensf_analysis,
)


def displacement_metrics(
    ensemble: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    nx: int,
) -> tuple[float, float, float, float]:
    displacement = ensemble[..., :nx]
    target_u = target[:nx]
    mean_u = (weights.unsqueeze(-1) * displacement).sum(dim=-2)
    scale = torch.sqrt(torch.mean(target_u.square())).clamp_min(1.0e-12)
    nrmse = torch.sqrt(torch.mean((mean_u - target_u).square())) / scale
    target_batch = target_u.unsqueeze(0)
    crps = weighted_ensemble_crps(displacement, target_batch, weights)
    coverage, width = weighted_central_interval_coverage_width(
        displacement, target_batch, weights, level=0.9
    )
    return float(nrmse), float(crps), float(coverage), float(width)


def run_torch_method(assets: WaveScenarioAssets, method: str) -> dict[str, float | int]:
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
    dtype = torch.float64
    operator = SparseObservation(torch.as_tensor(assets.observation_indices, dtype=torch.int64))
    covariance = (cfg.obs_noise ** 2) * torch.eye(assets.observation_indices.size, dtype=dtype)
    target_states = torch.as_tensor(assets.truth_states, dtype=dtype)
    started = time.perf_counter()
    metric_rows: list[tuple[float, float, float, float]] = []
    forward_solves = 0

    if method == "hilda":
        alpha_cfg = AlphaConfig(
            alpha_min=cfg.alpha_min,
            alpha_max=cfg.alpha_max,
            initial_nodes=9,
            max_nodes=9,
            prune_threshold=0.0,
        )
        hilda = HILDAFilter(HILDAConfig(alpha=alpha_cfg))
        tracker = AlphaEvidenceTracker.create(hilda.config.alpha, dtype=dtype)
        ensemble = torch.as_tensor(assets.initial_ensemble, dtype=dtype).unsqueeze(0).repeat(9, 1, 1)
        previous_filtering = None
    else:
        ensemble = torch.as_tensor(assets.initial_ensemble, dtype=dtype)
        previous_filtering = ensemble.clone() if method == "enff_f2p" else None
        hilda = tracker = None

    generator = torch.Generator().manual_seed(assets.seed + 3001)
    for step in range(assets.n_steps + 1):
        if method == "hilda":
            branch_weights = tracker.weights.to(dtype)
            flat = ensemble.reshape(-1, assets.nx * 2)
            flat_weights = branch_weights.unsqueeze(1).expand(-1, ensemble.shape[1]).reshape(-1)
            metric_rows.append(displacement_metrics(flat, target_states[step], flat_weights, assets.nx))
        else:
            weights = torch.full((ensemble.shape[0],), 1.0 / ensemble.shape[0], dtype=dtype)
            metric_rows.append(displacement_metrics(ensemble, target_states[step], weights, assets.nx))
        if step == assets.n_steps:
            break
        if method == "hilda":
            propagated = []
            for alpha, branch in zip(tracker.alpha, ensemble, strict=True):
                propagated.append(
                    torch.as_tensor(
                        v3.propagate_batch(
                            branch.detach().numpy(),
                            v3.alpha_to_theta(float(alpha), cfg),
                            float(scenario.times[step]),
                            cfg,
                            np.random.default_rng(assets.seed),
                            stochastic=True,
                            noise_draw=assets.forecast_noise[step],
                        ),
                        dtype=dtype,
                    )
                )
            ensemble = torch.stack(propagated)
            forward_solves += 9
        else:
            ensemble = torch.as_tensor(
                v3.propagate_batch(
                    ensemble.numpy(),
                    v3.alpha_to_theta(0.50, cfg),
                    float(scenario.times[step]),
                    cfg,
                    np.random.default_rng(assets.seed),
                    stochastic=True,
                    noise_draw=assets.forecast_noise[step],
                ),
                dtype=dtype,
            )
            forward_solves += 1

        if not assets.observation_mask[step + 1]:
            continue
        observation = torch.as_tensor(assets.observations[step + 1], dtype=dtype)
        if method == "hilda":
            analysis = hilda.analyze_paths(ensemble, tracker, observation, operator, covariance)
            ensemble = analysis.ensembles
        elif method == "denkf":
            ensemble = denkf_analysis(ensemble, observation, operator, covariance)
        elif method == "letkf":
            ensemble = letkf_analysis(ensemble, observation, operator, covariance)
        elif method == "ensf":
            ensemble = ensf_analysis(
                ensemble, observation, operator, covariance,
                config=EnSFConfig(sampling_time_step_count=8), generator=generator,
            )
        elif method == "iensf":
            ensemble = iensf_analysis(
                ensemble, observation, operator, covariance,
                config=IEnSFConfig(sampling_time_step_count=8, refinement_iterations=1),
                generator=generator,
            )
        elif method == "ensf_lr_ridge":
            ensemble = ensf_lr_ridge_analysis(
                ensemble, observation, operator, covariance, generator=generator,
            )
        elif method == "enff_f2p":
            ensemble = enff_f2p_analysis(
                previous_filtering, ensemble, observation, operator, covariance,
                config=EnFFF2PConfig(sampling_time_step_count=5), generator=generator,
            )
            previous_filtering = ensemble.clone()
        else:
            raise ValueError(method)
    rows = np.asarray(metric_rows, dtype=float)
    return {
        "mean_nrmse": float(rows[:, 0].mean()),
        "mean_crps": float(rows[:, 1].mean()),
        "mean_coverage": float(rows[:, 2].mean()),
        "mean_interval_width": float(rows[:, 3].mean()),
        "forward_solves": forward_solves,
        "runtime_seconds": float(time.perf_counter() - started),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    record = manifest["records"][0]
    assets = WaveScenarioAssets.load(Path(record["path"]))
    methods = ("denkf", "letkf", "ensf", "iensf", "ensf_lr_ridge", "enff_f2p", "hilda")
    print("GATE_B_ONE_ASSET", record["name"], assets.array_digest[:16])
    for method in methods:
        result = run_torch_method(assets, method)
        print("METHOD", method, json.dumps(result, sort_keys=True))
    cfg = dataclasses.replace(v3.make_config("quick"), seed=assets.seed, nx=assets.nx, ensemble_size=assets.ensemble_size, alpha_true=assets.alpha_true, t_end=float(assets.times[-1]), dt=float(assets.times[1] - assets.times[0]), n_sensors=int(assets.observation_indices.size), obs_interval=20)
    scenario = v3.scenario_from_assets(assets, cfg)
    for method in ("alpha_ensf_lr", "alpha_ensf_lr_pce"):
        result = v3.run_method(scenario, method)
        print("METHOD", "apce" if method == "alpha_ensf_lr" else "pce", json.dumps(result["metrics"], sort_keys=True))


if __name__ == "__main__":
    main()
