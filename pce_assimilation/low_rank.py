from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import LowRankConfig


@dataclass(frozen=True)
class LowRankMap:
    gain: torch.Tensor
    retained_rank: int
    explained_variance: float
    ridge: float


def localized_low_rank_map(
    state_ensemble: torch.Tensor,
    observation_ensemble: torch.Tensor,
    config: LowRankConfig,
    localization: torch.Tensor | None = None,
) -> LowRankMap:
    if state_ensemble.ndim != 2 or observation_ensemble.ndim != 2:
        raise ValueError("Ensembles must have shape [ensemble, variable]")
    if state_ensemble.shape[0] != observation_ensemble.shape[0]:
        raise ValueError("State and observation ensembles must have equal size")
    ensemble_size = state_ensemble.shape[0]
    if ensemble_size < 2:
        raise ValueError("At least two ensemble members are required")
    normalization = float(ensemble_size - 1) ** -0.5
    state_anomalies = (state_ensemble - state_ensemble.mean(dim=0)).mT * normalization
    observation_anomalies = (
        observation_ensemble - observation_ensemble.mean(dim=0)
    ).mT * normalization
    left, singular_values, right_transpose = torch.linalg.svd(
        observation_anomalies,
        full_matrices=False,
    )
    energy = singular_values.square()
    if float(energy.sum()) <= torch.finfo(energy.dtype).eps:
        gain = state_ensemble.new_zeros(
            state_ensemble.shape[1], observation_ensemble.shape[1]
        )
        return LowRankMap(gain=gain, retained_rank=0, explained_variance=0.0, ridge=0.0)
    cumulative = torch.cumsum(energy, dim=0) / energy.sum()
    rank = int(torch.searchsorted(cumulative, config.explained_variance).item()) + 1
    rank = min(rank, config.max_rank, ensemble_size - 1, singular_values.numel())
    retained = singular_values[:rank]
    median_energy = retained.square().median()
    ridge_tensor = torch.maximum(
        retained.new_tensor(config.ridge_floor),
        config.ridge_relative * median_energy,
    )
    right = right_transpose[:rank].mT
    gain = (
        state_anomalies
        @ right
        @ torch.diag(retained / (retained.square() + ridge_tensor))
        @ left[:, :rank].mT
    )
    if localization is not None:
        if localization.shape != gain.shape:
            raise ValueError("Localization must have shape [state, observation]")
        gain = gain * localization
    return LowRankMap(
        gain=gain,
        retained_rank=rank,
        explained_variance=float(cumulative[rank - 1]),
        ridge=float(ridge_tensor),
    )


def propagate_observation_increment(
    observation_increment: torch.Tensor,
    low_rank_map: LowRankMap,
) -> torch.Tensor:
    return observation_increment @ low_rank_map.gain.mT


def gaspari_cohn(distance: torch.Tensor, radius: float) -> torch.Tensor:
    if radius <= 0:
        raise ValueError("Localization radius must be positive")
    scaled = 2.0 * distance.abs() / radius
    first = (
        1.0
        - 5.0 / 3.0 * scaled.square()
        + 5.0 / 8.0 * scaled.pow(3)
        + 0.5 * scaled.pow(4)
        - 0.25 * scaled.pow(5)
    )
    second = (
        4.0
        - 5.0 * scaled
        + 5.0 / 3.0 * scaled.square()
        + 5.0 / 8.0 * scaled.pow(3)
        - 0.5 * scaled.pow(4)
        + scaled.pow(5) / 12.0
        - 2.0 / (3.0 * scaled.clamp_min(torch.finfo(scaled.dtype).eps))
    )
    return torch.where(
        scaled <= 1.0,
        first,
        torch.where(scaled <= 2.0, second, torch.zeros_like(scaled)),
    ).clamp(0.0, 1.0)
