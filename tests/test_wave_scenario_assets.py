from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.wave_scenario_assets import WaveScenarioAssets
import run_benchmark_v3 as v3


class WaveScenarioAssetTests(unittest.TestCase):
    def make_assets(self) -> WaveScenarioAssets:
        steps, nx, members, sensors = 3, 5, 4, 2
        times = np.arange(steps + 1, dtype=float)
        truth = np.arange((steps + 1) * 2 * nx, dtype=float).reshape(steps + 1, 2 * nx)
        observations = np.zeros((steps + 1, sensors), dtype=float)
        observations[[1, 3]] = [[1.0, 2.0], [3.0, 4.0]]
        return WaveScenarioAssets(
            seed=7,
            nx=nx,
            ensemble_size=members,
            times=times,
            truth_states=truth,
            observations=observations,
            observation_mask=np.array([False, True, False, True]),
            observation_indices=np.array([1, 3]),
            initial_ensemble=np.zeros((members, 2 * nx)),
            forecast_noise=np.ones((steps, members, nx)),
            truth_noise=np.ones((steps, nx)),
            observation_noise=np.zeros((2, sensors)),
            alpha_true=0.72,
        )

    def test_round_trip_preserves_digest_and_common_noise(self) -> None:
        assets = self.make_assets()
        with tempfile.TemporaryDirectory() as temporary:
            assets.save(Path(temporary))
            restored = WaveScenarioAssets.load(Path(temporary))
        self.assertEqual(restored.array_digest, assets.array_digest)
        np.testing.assert_array_equal(restored.forecast_noise, assets.forecast_noise)
        np.testing.assert_array_equal(restored.observation_mask, assets.observation_mask)

    def test_rejects_duplicate_sensor_indices(self) -> None:
        with self.assertRaises(ValueError):
            assets = self.make_assets()
            WaveScenarioAssets(
                **{**assets.__dict__, "observation_indices": np.array([1, 1])}
            )

    def test_rejects_corrupted_digest(self) -> None:
        assets = self.make_assets()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            assets.save(path)
            metadata = (path / "metadata.json").read_text(encoding="utf-8")
            (path / "metadata.json").write_text(metadata.replace(assets.array_digest, "0" * 64), encoding="utf-8")
            with self.assertRaises(ValueError):
                WaveScenarioAssets.load(path)

    def test_legacy_adapter_preserves_frozen_inputs(self) -> None:
        cfg = v3.make_config("quick")
        cfg = cfg.__class__(**{**cfg.__dict__, "seed": 19, "nx": 5, "ensemble_size": 4, "n_sensors": 2, "t_end": 0.03, "dt": 0.01, "obs_interval": 1})
        scenario = v3.generate_scenario(cfg)
        assets = WaveScenarioAssets.from_legacy_scenario(scenario)
        adapted = v3.scenario_from_assets(assets, cfg)
        np.testing.assert_array_equal(adapted.truth_states, assets.truth_states)
        np.testing.assert_array_equal(adapted.forecast_noise, assets.forecast_noise)
        self.assertEqual(sorted(adapted.observations), [1, 2, 3])
        np.testing.assert_array_equal(adapted.observations[2], assets.observations[2])
