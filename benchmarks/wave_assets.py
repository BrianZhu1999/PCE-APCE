"""Immutable common inputs for the 41-node wave benchmark."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1


def _as_float_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(array)


@dataclass(frozen=True)
class WaveScenarioAssets:
    """Common truth/observation/random inputs shared by every method.

    ``observations`` uses one row per time step and stores zeros at unobserved
    steps. ``observation_mask`` identifies the rows that contain data.
    ``forecast_noise`` is indexed as [step, member, spatial_position] and is
    the only process-noise source permitted for the main Wave comparison.
    """

    seed: int
    nx: int
    ensemble_size: int
    times: np.ndarray
    truth_states: np.ndarray
    observations: np.ndarray
    observation_mask: np.ndarray
    observation_indices: np.ndarray
    initial_ensemble: np.ndarray
    forecast_noise: np.ndarray
    truth_noise: np.ndarray
    observation_noise: np.ndarray
    alpha_true: float

    def __post_init__(self) -> None:
        arrays = {
            "times": _as_float_array(self.times, "times"),
            "truth_states": _as_float_array(self.truth_states, "truth_states"),
            "initial_ensemble": _as_float_array(self.initial_ensemble, "initial_ensemble"),
            "forecast_noise": _as_float_array(self.forecast_noise, "forecast_noise"),
            "truth_noise": _as_float_array(self.truth_noise, "truth_noise"),
            "observation_noise": _as_float_array(self.observation_noise, "observation_noise"),
            "observations": _as_float_array(self.observations, "observations"),
        }
        mask = np.asarray(self.observation_mask, dtype=bool)
        indices = np.asarray(self.observation_indices, dtype=np.int64)
        if self.nx < 2 or self.ensemble_size < 1:
            raise ValueError("nx and ensemble_size must be positive")
        if arrays["times"].ndim != 1:
            raise ValueError("times must be one-dimensional")
        n_steps = arrays["times"].size - 1
        if n_steps < 1:
            raise ValueError("times must contain at least two entries")
        if arrays["truth_states"].shape != (n_steps + 1, 2 * self.nx):
            raise ValueError("truth_states must have shape [steps+1, 2*nx]")
        if arrays["initial_ensemble"].shape != (self.ensemble_size, 2 * self.nx):
            raise ValueError("initial_ensemble has an incompatible shape")
        if arrays["forecast_noise"].shape != (n_steps, self.ensemble_size, self.nx):
            raise ValueError("forecast_noise must have shape [steps, members, nx]")
        if arrays["truth_noise"].shape != (n_steps, self.nx):
            raise ValueError("truth_noise must have shape [steps, nx]")
        if arrays["observations"].ndim != 2 or arrays["observations"].shape[0] != n_steps + 1:
            raise ValueError("observations must have one row per time step")
        if mask.shape != (n_steps + 1,) or mask.sum() != arrays["observation_noise"].shape[0]:
            raise ValueError("observation_mask and observation_noise disagree")
        if arrays["observations"].shape[1] != indices.size:
            raise ValueError("observations and observation_indices disagree")
        if indices.size == 0 or np.any(indices < 0) or np.any(indices >= self.nx):
            raise ValueError("observation_indices are outside the displacement grid")
        if np.unique(indices).size != indices.size:
            raise ValueError("observation_indices must be unique")
        if not 0.0 < float(self.alpha_true) < 1.0:
            raise ValueError("alpha_true must lie strictly between zero and one")
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        object.__setattr__(self, "observation_mask", mask)
        object.__setattr__(self, "observation_indices", indices)

    @property
    def n_steps(self) -> int:
        return self.times.size - 1

    @classmethod
    def from_scenario(cls, scenario: Any) -> "WaveScenarioAssets":
        """Convert a generated scenario without regenerating its inputs."""
        cfg = scenario.cfg
        n_steps = scenario.times.size - 1
        # Keep the dense tensor finite; ``observation_mask`` carries sparsity.
        observations = np.zeros((n_steps + 1, scenario.observation_indices.size), dtype=np.float64)
        mask = np.zeros(n_steps + 1, dtype=bool)
        ordered_noise = []
        for step, value in sorted(scenario.observations.items()):
            observations[int(step)] = _as_float_array(value, f"observation[{step}]")
            mask[int(step)] = True
            ordered_noise.append(observations[int(step)] - scenario.truth_states[int(step), scenario.observation_indices])
        truth_noise = getattr(scenario, "truth_noise", None)
        if truth_noise is None:
            raise ValueError("Scenario must expose truth_noise")
        truth_noise = np.asarray(truth_noise, dtype=np.float64)
        # The generator stores one singleton leading member axis.
        if truth_noise.ndim == 3 and truth_noise.shape[1] == 1:
            truth_noise = truth_noise[:, 0, :]
        return cls(
            seed=int(cfg.seed),
            nx=int(cfg.nx),
            ensemble_size=int(cfg.ensemble_size),
            times=scenario.times,
            truth_states=scenario.truth_states,
            observations=observations,
            observation_mask=mask,
            observation_indices=scenario.observation_indices,
            initial_ensemble=scenario.ensemble_initial,
            forecast_noise=scenario.forecast_noise,
            truth_noise=truth_noise,
            observation_noise=np.asarray(ordered_noise, dtype=np.float64),
            alpha_true=float(cfg.alpha_true),
        )

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "arrays.npz",
            times=self.times,
            truth_states=self.truth_states,
            observations=self.observations,
            observation_mask=self.observation_mask.astype(np.uint8),
            observation_indices=self.observation_indices,
            initial_ensemble=self.initial_ensemble,
            forecast_noise=self.forecast_noise,
            truth_noise=self.truth_noise,
            observation_noise=self.observation_noise,
        )
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "seed": self.seed,
            "nx": self.nx,
            "ensemble_size": self.ensemble_size,
            "alpha_true": self.alpha_true,
            "n_steps": self.n_steps,
        }
        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> "WaveScenarioAssets":
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if int(metadata.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("Unsupported WaveScenarioAssets schema")
        with np.load(directory / "arrays.npz", allow_pickle=False) as data:
            return cls(
                seed=int(metadata["seed"]),
                nx=int(metadata["nx"]),
                ensemble_size=int(metadata["ensemble_size"]),
                times=data["times"],
                truth_states=data["truth_states"],
                observations=data["observations"],
                observation_mask=data["observation_mask"].astype(bool),
                observation_indices=data["observation_indices"],
                initial_ensemble=data["initial_ensemble"],
                forecast_noise=data["forecast_noise"],
                truth_noise=data["truth_noise"],
                observation_noise=data["observation_noise"],
                alpha_true=float(metadata["alpha_true"]),
            )
