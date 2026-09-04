from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

import run_benchmark_v3 as v3

from experiments.wave_scenario_assets import WaveScenarioAssets


def main() -> None:
    cfg = dataclasses.replace(
        v3.make_config("quick"),
        seed=2026080600,
        nx=41,
        ensemble_size=18,
        n_alpha=7,
        t_end=0.05,
        dt=0.0025,
        obs_interval=5,
        n_sensors=6,
    )
    scenario = v3.generate_scenario(cfg)
    assets = WaveScenarioAssets.from_legacy_scenario(scenario)
    print(
        "SCENARIO",
        assets.nx,
        assets.truth_states.shape,
        assets.forecast_noise.shape,
        assets.truth_noise.shape,
        int(assets.observation_mask.sum()),
        assets.array_digest[:16],
    )
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        assets.save(directory)
        restored = WaveScenarioAssets.load(directory)
    print("ROUNDTRIP", restored.array_digest == assets.array_digest)


if __name__ == "__main__":
    main()
