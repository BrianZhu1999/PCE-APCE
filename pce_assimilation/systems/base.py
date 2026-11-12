from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch

from ..evidence import liu_quantile


class HybridSystem(ABC):
    state_dim: int

    @abstractmethod
    def drift(
        self,
        state: torch.Tensor,
        time: float,
        alpha_quantile: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def diffusion(self, state: torch.Tensor, time: float) -> torch.Tensor:
        raise NotImplementedError

    def project(self, state: torch.Tensor) -> torch.Tensor:
        return state

    def noise(
        self,
        state: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        return torch.randn(
            state.shape,
            dtype=state.dtype,
            device=state.device,
            generator=generator,
        )

    def step(
        self,
        state: torch.Tensor,
        time: float,
        dt: float,
        alpha: float | torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        quantile = liu_quantile(
            torch.as_tensor(alpha, dtype=state.dtype, device=state.device)
        )
        k1 = self.drift(state, time, quantile)
        k2 = self.drift(state + 0.5 * dt * k1, time + 0.5 * dt, quantile)
        k3 = self.drift(state + 0.5 * dt * k2, time + 0.5 * dt, quantile)
        k4 = self.drift(state + dt * k3, time + dt, quantile)
        deterministic = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        stochastic = math_sqrt(dt) * self.diffusion(state, time) * self.noise(state, generator)
        return self.project(deterministic + stochastic)


def math_sqrt(value: float) -> float:
    if value <= 0:
        raise ValueError("Time step must be positive")
    return value**0.5


def simulate_trajectory(
    system: HybridSystem,
    initial_state: torch.Tensor,
    steps: int,
    dt: float,
    alpha_schedule: float | Callable[[int, float], float],
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    state = initial_state
    trajectory = [state.clone()]
    for index in range(steps):
        time = index * dt
        alpha = alpha_schedule(index, time) if callable(alpha_schedule) else alpha_schedule
        state = system.step(state, time, dt, alpha, generator)
        trajectory.append(state.clone())
    return torch.stack(trajectory)
