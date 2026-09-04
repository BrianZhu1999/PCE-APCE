from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..alpha import liu_quantile
from .base import HybridSystem


@dataclass(frozen=True)
class EnFFNavierStokesConfig:
    nx: int = 256
    ny: int = 256
    length_x: float = 2.0
    length_y: float = 2.0
    viscosity: float = 1e-3
    forcing_amplitude: float = 5e-2
    forcing_wavenumber: float = 8.0
    pressure_iterations: int = 100
    epistemic_force_scale: float = 0.15
    stochastic_scale: float = 1e-3


class EnFFNavierStokes2D(HybridSystem):
    """EnFF-compatible periodic pressure-velocity solver with hybrid forcing."""

    def __init__(self, config: EnFFNavierStokesConfig | None = None) -> None:
        self.config = config or EnFFNavierStokesConfig()
        if min(self.config.nx, self.config.ny) < 4:
            raise ValueError("Navier-Stokes grid dimensions must be at least 4")
        if self.config.pressure_iterations < 1:
            raise ValueError("pressure_iterations must be positive")
        self.state_dim = 3 * self.config.nx * self.config.ny
        self.dx = self.config.length_x / self.config.nx
        self.dy = self.config.length_y / self.config.ny

    def _reshape(self, state: torch.Tensor) -> torch.Tensor:
        return state.reshape(*state.shape[:-1], 3, self.config.nx, self.config.ny)

    def _divergence(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return (
            (u.roll(-1, -2) - u.roll(1, -2)) / (2.0 * self.dx)
            + (v.roll(-1, -1) - v.roll(1, -1)) / (2.0 * self.dy)
        )

    def _gradient(self, field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            (field.roll(-1, -2) - field.roll(1, -2)) / (2.0 * self.dx),
            (field.roll(-1, -1) - field.roll(1, -1)) / (2.0 * self.dy),
        )

    def _laplacian(self, field: torch.Tensor) -> torch.Tensor:
        return (
            (field.roll(-1, -2) - 2.0 * field + field.roll(1, -2)) / self.dx**2
            + (field.roll(-1, -1) - 2.0 * field + field.roll(1, -1)) / self.dy**2
        )

    def _advect(self, u: torch.Tensor, v: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
        gradient_x, gradient_y = self._gradient(field)
        return u * gradient_x + v * gradient_y

    def _pressure_poisson(self, pressure: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        denominator = 2.0 * (self.dx**2 + self.dy**2)
        for _ in range(self.config.pressure_iterations):
            pressure = (
                (pressure.roll(1, -2) + pressure.roll(-1, -2)) * self.dy**2
                + (pressure.roll(1, -1) + pressure.roll(-1, -1)) * self.dx**2
                - source * self.dx**2 * self.dy**2
            ) / denominator
        return pressure

    def _forcing(self, reference: torch.Tensor) -> torch.Tensor:
        y = torch.linspace(
            0.0,
            self.config.length_y,
            self.config.ny + 1,
            dtype=reference.dtype,
            device=reference.device,
        )[:-1]
        return self.config.forcing_amplitude * torch.sin(
            2.0 * math.pi * self.config.forcing_wavenumber * y / self.config.length_y
        ).unsqueeze(0)

    def drift(
        self,
        state: torch.Tensor,
        time: float,
        alpha_quantile: torch.Tensor,
    ) -> torch.Tensor:
        pressure, u, v = self._reshape(state).unbind(dim=-3)
        multiplier = 1.0 + self.config.epistemic_force_scale * alpha_quantile
        du = (
            self.config.viscosity * self._laplacian(u)
            - self._advect(u, v, u)
            + multiplier * self._forcing(u)
        )
        dv = self.config.viscosity * self._laplacian(v) - self._advect(u, v, v)
        return torch.stack((torch.zeros_like(pressure), du, dv), dim=-3).reshape_as(state)

    def diffusion(self, state: torch.Tensor, time: float) -> torch.Tensor:
        values = self._reshape(torch.full_like(state, self.config.stochastic_scale))
        values[..., 0, :, :] = 0.0
        return values.reshape_as(state)

    def project_state(self, state: torch.Tensor, dt: float) -> torch.Tensor:
        pressure, u, v = self._reshape(state).unbind(dim=-3)
        source = self._divergence(u, v) / dt
        pressure = self._pressure_poisson(pressure, source)
        pressure_x, pressure_y = self._gradient(pressure)
        u = u - dt * pressure_x
        v = v - dt * pressure_y
        return torch.stack((pressure, u, v), dim=-3).reshape_as(state)

    def step(
        self,
        state: torch.Tensor,
        time: float,
        dt: float,
        alpha: float | torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        quantile = liu_quantile(torch.as_tensor(alpha, dtype=state.dtype, device=state.device))
        predicted = state + dt * self.drift(state, time, quantile)
        if self.config.stochastic_scale > 0.0:
            predicted = predicted + dt**0.5 * self.diffusion(state, time) * torch.randn(
                state.shape,
                dtype=state.dtype,
                device=state.device,
                generator=generator,
            )
        return self.project_state(predicted, dt)

    def taylor_green_state(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        x = torch.linspace(0.0, self.config.length_x, self.config.nx + 1, dtype=dtype, device=device)[:-1]
        y = torch.linspace(0.0, self.config.length_y, self.config.ny + 1, dtype=dtype, device=device)[:-1]
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        phase_x = 2.0 * math.pi * xx / self.config.length_x
        phase_y = 2.0 * math.pi * yy / self.config.length_y
        u = torch.sin(phase_x) * torch.cos(phase_y)
        v = -torch.cos(phase_x) * torch.sin(phase_y)
        return torch.stack((torch.zeros_like(u), u, v)).reshape(-1)
