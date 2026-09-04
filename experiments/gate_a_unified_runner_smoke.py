from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_benchmark_v3 as v3
from experiments.wave_scenario_assets import WaveScenarioAssets


METHODS = [
    "deterministic",
    "enkf",
    "ensf_direct",
    "ensf_lr",
    "oracle_alpha",
    "alpha_only",
    "alpha_ensf_lr",
    "alpha_ensf_lr_pce",
]


def run_in_order(scenario: v3.Scenario, methods: list[str]) -> dict[str, float]:
    return {
        method: float(v3.run_method(scenario, method)["metrics"]["mean_rmse"])
        for method in methods
    }


def main() -> None:
    cfg = dataclasses.replace(
        v3.make_config("quick"),
        seed=2026080601,
        nx=41,
        ensemble_size=18,
        n_alpha=7,
        t_end=0.05,
        dt=0.0025,
        obs_interval=5,
        n_sensors=6,
    )
    generated = v3.generate_scenario(cfg)
    assets = WaveScenarioAssets.from_legacy_scenario(generated)
    scenario = v3.scenario_from_assets(assets, cfg)
    if assets.nx != 41 or scenario.truth_states.shape[1] != 82:
        raise AssertionError("Gate A did not use the legacy 41-node Wave state")
    forward = run_in_order(scenario, METHODS)
    reverse = run_in_order(scenario, list(reversed(METHODS)))
    for method in METHODS:
        if not np.isclose(forward[method], reverse[method], rtol=0.0, atol=1.0e-13):
            raise AssertionError(f"method-order dependence for {method}: {forward[method]} vs {reverse[method]}")
    print("UNIFIED_SCENARIO", assets.array_digest[:16], assets.nx, assets.ensemble_size)
    print("METHOD_ORDER_INVARIANT", len(METHODS))
    for method in METHODS:
        print(f"RESULT {method} {forward[method]:.10g}")


if __name__ == "__main__":
    main()
