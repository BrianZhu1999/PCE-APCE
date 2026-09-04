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


@dataclass(frozen=True)
class WaveConfig:
    nx: int = 128
    length: float = 1.0
    wave_speed: float = 0.7
    damping: float = 0.08
    cubic_stiffness: float = 0.03
    stochastic_scale: float = 0.02
    epistemic_scale: float = 0.15


class Wave1D(HybridSystem):
    def __init__(self, config: WaveConfig | None = None) -> None:
        self.config = config or WaveConfig()
        self.state_dim = 2 * self.config.nx
        self.dx = self.config.length / (self.config.nx - 1)
        self.grid = torch.linspace(0.0, self.config.length, self.config.nx)

    def _split(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return state[..., : self.config.nx], state[..., self.config.nx :]

    def _profile(self, state: torch.Tensor) -> torch.Tensor:
        grid = self.grid.to(dtype=state.dtype, device=state.device)
        return torch.sin(2.0 * math.pi * grid / self.config.length)

    def drift(self, state: torch.Tensor, time: float, alpha_quantile: torch.Tensor) -> torch.Tensor:
        displacement, velocity = self._split(state)
        laplacian = torch.zeros_like(displacement)
        laplacian[..., 1:-1] = (
            displacement[..., 2:]
            - 2.0 * displacement[..., 1:-1]
            + displacement[..., :-2]
        ) / self.dx**2
        acceleration = (
            self.config.wave_speed**2 * laplacian
            - self.config.damping * velocity
            - self.config.cubic_stiffness * displacement.pow(3)
            + self.config.epistemic_scale * alpha_quantile * self._profile(state)
        )
        result = torch.cat((velocity, acceleration), dim=-1)
        result[..., 0] = 0.0
        result[..., self.config.nx - 1] = 0.0
        result[..., self.config.nx] = 0.0
        result[..., -1] = 0.0
        return result

    def diffusion(self, state: torch.Tensor, time: float) -> torch.Tensor:
        scale = torch.zeros_like(state)
        scale[..., self.config.nx :] = self.config.stochastic_scale * self._profile(state)
        scale[..., self.config.nx] = 0.0
        scale[..., -1] = 0.0
        return scale

    def project(self, state: torch.Tensor) -> torch.Tensor:
        projected = state.clone()
        projected[..., 0] = 0.0
        projected[..., self.config.nx - 1] = 0.0
        projected[..., self.config.nx] = 0.0
        projected[..., -1] = 0.0
        return projected


class _PeriodicSpectral1D(HybridSystem):
    nx: int
    length: float

    def _wavenumbers(self, state: torch.Tensor) -> torch.Tensor:
        dx = self.length / self.nx
        return 2.0 * math.pi * torch.fft.fftfreq(
            self.nx,
            d=dx,
            dtype=state.dtype,
            device=state.device,
        )

    def _derivative(self, field: torch.Tensor, order: int) -> torch.Tensor:
        wave = self._wavenumbers(field)
        spectrum = torch.fft.fft(field, dim=-1)
        cutoff = self.nx // 3
        modes = torch.fft.fftfreq(self.nx, device=field.device).abs() * self.nx
        spectrum = torch.where(modes <= cutoff, spectrum, torch.zeros_like(spectrum))
        return torch.fft.ifft((1j * wave) ** order * spectrum, dim=-1).real

    def noise(self, state: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        white = super().noise(state, generator)
        spectrum = torch.fft.fft(white, dim=-1)
        modes = torch.fft.fftfreq(self.nx, device=state.device).abs() * self.nx
        filtered = torch.where(modes <= self.nx // 8, spectrum, torch.zeros_like(spectrum))
        noise = torch.fft.ifft(filtered, dim=-1).real
        return noise / noise.std(dim=-1, keepdim=True).clamp_min(torch.finfo(state.dtype).eps)


@dataclass(frozen=True)
class BurgersConfig:
    nx: int = 256
    length: float = 2.0 * math.pi
    viscosity: float = 0.02
    stochastic_scale: float = 0.025
    epistemic_scale: float = 0.18


class Burgers1D(_PeriodicSpectral1D):
    def __init__(self, config: BurgersConfig | None = None) -> None:
        self.config = config or BurgersConfig()
        self.nx = self.config.nx
        self.length = self.config.length
        self.state_dim = self.nx

    def drift(self, state: torch.Tensor, time: float, alpha_quantile: torch.Tensor) -> torch.Tensor:
        grid = torch.linspace(0.0, self.length, self.nx + 1, dtype=state.dtype, device=state.device)[:-1]
        convection = -0.5 * self._derivative(state.square(), 1)
        diffusion = self.config.viscosity * self._derivative(state, 2)
        forcing = self.config.epistemic_scale * alpha_quantile * torch.sin(2.0 * grid)
        return convection + diffusion + forcing

    def diffusion(self, state: torch.Tensor, time: float) -> torch.Tensor:
        return torch.full_like(state, self.config.stochastic_scale)


@dataclass(frozen=True)
class AllenCahnConfig:
    nx: int = 256
    length: float = 2.0 * math.pi
    interface_width: float = 0.08
    stochastic_scale: float = 0.015
    epistemic_scale: float = 0.12


class AllenCahn1D(_PeriodicSpectral1D):
    def __init__(self, config: AllenCahnConfig | None = None) -> None:
        self.config = config or AllenCahnConfig()
        self.nx = self.config.nx
        self.length = self.config.length
        self.state_dim = self.nx

    def drift(self, state: torch.Tensor, time: float, alpha_quantile: torch.Tensor) -> torch.Tensor:
        grid = torch.linspace(0.0, self.length, self.nx + 1, dtype=state.dtype, device=state.device)[:-1]
        return (
            self.config.interface_width**2 * self._derivative(state, 2)
            + state
            - state.pow(3)
            + self.config.epistemic_scale * alpha_quantile * torch.cos(grid)
        )

    def diffusion(self, state: torch.Tensor, time: float) -> torch.Tensor:
        return torch.full_like(state, self.config.stochastic_scale)
