from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .base import HybridSystem


@dataclass(frozen=True)
class NavierStokesConfig:
    nx: int = 64
    ny: int = 64
    length_x: float = 2.0 * math.pi
    length_y: float = 2.0 * math.pi
    viscosity: float = 1e-3
    linear_drag: float = 0.02
    stochastic_scale: float = 0.01
    epistemic_scale: float = 0.10
    forcing_wavenumber: int = 4


class NavierStokes2D(HybridSystem):
    """Periodic two-dimensional vorticity equation with dealiased pseudospectral drift."""

    def __init__(self, config: NavierStokesConfig | None = None) -> None:
        self.config = config or NavierStokesConfig()
        self.state_dim = self.config.nx * self.config.ny

    def _reshape(self, state: torch.Tensor) -> torch.Tensor:
        return state.reshape(*state.shape[:-1], self.config.ny, self.config.nx)

    def _wave_numbers(self, field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        kx = 2.0 * math.pi * torch.fft.fftfreq(
            self.config.nx,
            d=self.config.length_x / self.config.nx,
            dtype=field.dtype,
            device=field.device,
        )
        ky = 2.0 * math.pi * torch.fft.fftfreq(
            self.config.ny,
            d=self.config.length_y / self.config.ny,
            dtype=field.dtype,
            device=field.device,
        )
        return torch.meshgrid(ky, kx, indexing="ij")

    def _dealias(self, spectrum: torch.Tensor) -> torch.Tensor:
        mode_x = torch.fft.fftfreq(self.config.nx, device=spectrum.device).abs() * self.config.nx
        mode_y = torch.fft.fftfreq(self.config.ny, device=spectrum.device).abs() * self.config.ny
        mask = (mode_y[:, None] <= self.config.ny // 3) & (
            mode_x[None, :] <= self.config.nx // 3
        )
        return torch.where(mask, spectrum, torch.zeros_like(spectrum))

    def drift(self, state: torch.Tensor, time: float, alpha_quantile: torch.Tensor) -> torch.Tensor:
        vorticity = self._reshape(state)
        ky, kx = self._wave_numbers(vorticity)
        squared_wave = kx.square() + ky.square()
        spectrum = self._dealias(torch.fft.fft2(vorticity, dim=(-2, -1)))
        inverse_laplacian = torch.where(
            squared_wave > 0,
            squared_wave.reciprocal(),
            torch.zeros_like(squared_wave),
        )
        streamfunction_spectrum = spectrum * inverse_laplacian
        velocity_x = torch.fft.ifft2(1j * ky * streamfunction_spectrum, dim=(-2, -1)).real
        velocity_y = torch.fft.ifft2(-1j * kx * streamfunction_spectrum, dim=(-2, -1)).real
        gradient_x = torch.fft.ifft2(1j * kx * spectrum, dim=(-2, -1)).real
        gradient_y = torch.fft.ifft2(1j * ky * spectrum, dim=(-2, -1)).real
        advection = -(velocity_x * gradient_x + velocity_y * gradient_y)
        laplacian = torch.fft.ifft2(-squared_wave * spectrum, dim=(-2, -1)).real
        grid_y = torch.linspace(
            0.0,
            self.config.length_y,
            self.config.ny + 1,
            dtype=state.dtype,
            device=state.device,
        )[:-1]
        forcing = torch.cos(self.config.forcing_wavenumber * grid_y)[:, None]
        tendency = (
            advection
            + self.config.viscosity * laplacian
            - self.config.linear_drag * vorticity
            + self.config.epistemic_scale * alpha_quantile * forcing
        )
        return tendency.reshape_as(state)

    def diffusion(self, state: torch.Tensor, time: float) -> torch.Tensor:
        return torch.full_like(state, self.config.stochastic_scale)

    def noise(self, state: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        field = self._reshape(
            torch.randn(
                state.shape,
                dtype=state.dtype,
                device=state.device,
                generator=generator,
            )
        )
        spectrum = torch.fft.fft2(field, dim=(-2, -1))
        mode_x = torch.fft.fftfreq(self.config.nx, device=state.device).abs() * self.config.nx
        mode_y = torch.fft.fftfreq(self.config.ny, device=state.device).abs() * self.config.ny
        mask = (mode_y[:, None] <= self.config.ny // 8) & (
            mode_x[None, :] <= self.config.nx // 8
        )
        filtered = torch.where(mask, spectrum, torch.zeros_like(spectrum))
        noise = torch.fft.ifft2(filtered, dim=(-2, -1)).real
        reduce_dims = tuple(range(noise.ndim - 2, noise.ndim))
        scale = noise.std(dim=reduce_dims, keepdim=True).clamp_min(torch.finfo(state.dtype).eps)
        return (noise / scale).reshape_as(state)
