from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SparseObservation:
    indices: torch.Tensor
    transform: str = "linear"

    def __call__(self, state_ensemble: torch.Tensor) -> torch.Tensor:
        selected = state_ensemble.index_select(-1, self.indices.to(state_ensemble.device))
        if self.transform == "linear":
            return selected
        if self.transform == "atan":
            return torch.atan(selected)
        if self.transform == "square_signed":
            return selected.abs() * selected
        raise ValueError(f"Unknown observation transform: {self.transform}")


@dataclass(frozen=True)
class SpectralObservation:
    spatial_indices: torch.Tensor
    mode_count: int | None = None
    real_only: bool = True

    def __call__(self, state_ensemble: torch.Tensor) -> torch.Tensor:
        selected = state_ensemble.index_select(-1, self.spatial_indices.to(state_ensemble.device))
        spectrum = torch.fft.rfft(selected, dim=-1)
        if self.mode_count is not None:
            spectrum = spectrum[..., : self.mode_count]
        if self.real_only:
            return spectrum.real
        return torch.view_as_real(spectrum).flatten(start_dim=-2)


def evenly_spaced_indices(state_dim: int, observation_count: int) -> torch.Tensor:
    if not 1 <= observation_count <= state_dim:
        raise ValueError("observation_count must lie in [1, state_dim]")
    return torch.linspace(0, state_dim - 1, observation_count).round().to(torch.int64).unique()
