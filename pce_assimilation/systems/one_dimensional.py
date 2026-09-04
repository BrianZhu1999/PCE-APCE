from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .base import HybridSystem


@dataclass(frozen=True)
class SpringConfig:
    damping: float = 0.12
    frequency: float = 1.0
    cubic_stiffness: float = 0.08
    forcing_amplitude: float = 0.8
    forcing_frequency: float = 0.7
    stochastic_scale: float = 0.12
    epistemic_scale: float = 0.35


class SpringOscillator(HybridSystem):
    state_dim = 2

    def __init__(self, config: SpringConfig | None = None) -> None:
        self.config = config or SpringConfig()

    def drift(self, state: torch.Tensor, time: float, alpha_quantile: torch.Tensor) -> torch.Tensor:
        position, velocity = state[..., 0], state[..., 1]
        acceleration = (
            -2.0 * self.config.damping * velocity
            - self.config.frequency**2 * position
            - self.config.cubic_stiffness * position.pow(3)
            + self.config.forcing_amplitude * math.cos(self.config.forcing_frequency * time)
            + self.config.epistemic_scale * alpha_quantile
        )
        return torch.stack((velocity, acceleration), dim=-1)

    def diffusion(self, state: torch.Tensor, time: float) -> torch.Tensor:
        scale = torch.zeros_like(state)
        scale[..., 1] = self.config.stochastic_scale
        return scale


@dataclass(frozen=True)
class HeatConfig:
    nx: int = 128
    length: float = 1.0
    diffusivity: float = 0.08
    reaction: float = 0.15
    stochastic_scale: float = 0.015
    epistemic_scale: float = 0.12


class Heat1D(HybridSystem):
    def __init__(self, config: HeatConfig | None = None) -> None:
        self.config = config or HeatConfig()
        self.state_dim = self.config.nx
        self.dx = self.config.length / (self.config.nx - 1)
        self.grid = torch.linspace(0.0, self.config.length, self.config.nx)

    def _profile(self, state: torch.Tensor) -> torch.Tensor:
        grid = self.grid.to(dtype=state.dtype, device=state.device)
        return torch.sin(math.pi * grid / self.config.length)

    def drift(self, state: torch.Tensor, time: float, alpha_quantile: torch.Tensor) -> torch.Tensor:
        laplacian = torch.zeros_like(state)
        laplacian[..., 1:-1] = (
            state[..., 2:] - 2.0 * state[..., 1:-1] + state[..., :-2]
        ) / self.dx**2
        reaction = self.config.reaction * state * (1.0 - state)
        result = (
            self.config.diffusivity * laplacian
            + reaction
            + self.config.epistemic_scale * alpha_quantile * self._profile(state)
        )
        result[..., 0] = 0.0
        result[..., -1] = 0.0
        return result

    def diffusion(self, state: torch.Tensor, time: float) -> torch.Tensor:
        scale = self.config.stochastic_scale * self._profile(state).expand_as(state)
        scale[..., 0] = 0.0
        scale[..., -1] = 0.0
        return scale

    def project(self, state: torch.Tensor) -> torch.Tensor:
        projected = state.clone()
        projected[..., 0] = 0.0
        projected[..., -1] = 0.0
        return projected
