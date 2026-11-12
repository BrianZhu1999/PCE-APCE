from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .base import HybridSystem


def _positive(values: torch.Tensor, floor: float = 0.0) -> torch.Tensor:
    return torch.clamp(values, min=floor)


@dataclass(frozen=True)
class ChemicalReactionConfig:
    """Second-order decomposition reaction with an uncertain reaction rate."""

    rate: float = 0.55
    uncertain_rate: float = 0.12
    stochastic_scale: float = 0.018
    initial_reactant: float = 1.0


class ChemicalReactionODE(HybridSystem):
    state_dim = 2

    def __init__(self, config: ChemicalReactionConfig | None = None) -> None:
        self.config = config or ChemicalReactionConfig()

    def drift(self, state: torch.Tensor, time: float, alpha_quantile: torch.Tensor) -> torch.Tensor:
        reactant = _positive(state[..., 0])
        effective_rate = torch.clamp(
            self.config.rate + self.config.uncertain_rate * alpha_quantile,
            min=0.05,
        )
        consumption = 2.0 * effective_rate * reactant.square()
        return torch.stack((-consumption, 0.5 * consumption), dim=-1)

    def diffusion(self, state: torch.Tensor, time: float) -> torch.Tensor:
        reactant = _positive(state[..., 0])
        product = _positive(state[..., 1])
        scale = torch.zeros_like(state)
        scale[..., 0] = self.config.stochastic_scale * (0.20 + reactant)
        scale[..., 1] = 0.5 * self.config.stochastic_scale * (0.20 + product)
        return scale

    def project(self, state: torch.Tensor) -> torch.Tensor:
        projected = _positive(state)
        total = self.config.initial_reactant
        reactant = torch.clamp(projected[..., 0], min=0.0, max=total)
        product = torch.clamp(projected[..., 1], min=0.0)
        invariant_product = 0.5 * (total - reactant)
        product = 0.75 * product + 0.25 * invariant_product
        return torch.stack((reactant, product), dim=-1)


@dataclass(frozen=True)
class PharmacokineticInfusionConfig:
    """Single-compartment infusion model with uncertain elimination and input."""

    infusion_rate: float = 0.52
    elimination_rate: float = 0.44
    uncertain_elimination: float = 0.075
    uncertain_infusion: float = 0.045
    stochastic_scale: float = 0.013


class PharmacokineticInfusionODE(HybridSystem):
    state_dim = 1

    def __init__(self, config: PharmacokineticInfusionConfig | None = None) -> None:
        self.config = config or PharmacokineticInfusionConfig()

    def drift(self, state: torch.Tensor, time: float, alpha_quantile: torch.Tensor) -> torch.Tensor:
        concentration = _positive(state[..., 0])
        derivative = (
            self.config.infusion_rate
            - self.config.elimination_rate * concentration
            + self.config.uncertain_elimination * concentration * alpha_quantile
            + self.config.uncertain_infusion * alpha_quantile
        )
        return derivative.unsqueeze(-1)

    def diffusion(self, state: torch.Tensor, time: float) -> torch.Tensor:
        return self.config.stochastic_scale * (0.10 + _positive(state))

    def project(self, state: torch.Tensor) -> torch.Tensor:
        return _positive(state)


@dataclass(frozen=True)
class FitzHughNagumoConfig:
    """FitzHugh-Nagumo excitable-cell oscillator with uncertain drive."""

    recovery_timescale: float = 0.08
    recovery_offset: float = 0.70
    recovery_slope: float = 0.80
    external_drive: float = 0.50
    uncertain_drive: float = 0.12
    stochastic_scale: float = 0.008


class FitzHughNagumoODE(HybridSystem):
    state_dim = 2

    def __init__(self, config: FitzHughNagumoConfig | None = None) -> None:
        self.config = config or FitzHughNagumoConfig()

    def drift(self, state: torch.Tensor, time: float, alpha_quantile: torch.Tensor) -> torch.Tensor:
        voltage = state[..., 0]
        recovery = state[..., 1]
        drive = self.config.external_drive + self.config.uncertain_drive * alpha_quantile
        d_voltage = voltage - voltage.pow(3) / 3.0 - recovery + drive
        d_recovery = self.config.recovery_timescale * (
            voltage + self.config.recovery_offset - self.config.recovery_slope * recovery
        )
        return torch.stack((d_voltage, d_recovery), dim=-1)

    def diffusion(self, state: torch.Tensor, time: float) -> torch.Tensor:
        scale = torch.zeros_like(state)
        scale[..., 0] = self.config.stochastic_scale
        return scale

    def project(self, state: torch.Tensor) -> torch.Tensor:
        return torch.clamp(state, min=-2.5, max=2.5)


@dataclass(frozen=True)
class RobertsonConfig:
    """Stiff Robertson reaction model with an uncertain reaction rate."""

    k1: float = 0.04
    k2: float = 3.0e7
    k3: float = 1.0e4
    uncertain_k2: float = 1.5e7
    stochastic_scale: float = 0.002


class RobertsonODE(HybridSystem):
    state_dim = 3

    def __init__(self, config: RobertsonConfig | None = None) -> None:
        self.config = config or RobertsonConfig()

    def drift(self, state: torch.Tensor, time: float, alpha_quantile: torch.Tensor) -> torch.Tensor:
        species_x = _positive(state[..., 0])
        species_y = _positive(state[..., 1])
        species_z = _positive(state[..., 2])
        k2 = self.config.k2 + self.config.uncertain_k2 * alpha_quantile
        dx = -self.config.k1 * species_x + self.config.k3 * species_y * species_z
        dy = self.config.k1 * species_x - k2 * species_y.square() - self.config.k3 * species_y * species_z
        dz = k2 * species_y.square()
        return torch.stack((dx, dy, dz), dim=-1)

    def diffusion(self, state: torch.Tensor, time: float) -> torch.Tensor:
        return self.config.stochastic_scale * (0.10 + _positive(state))

    def project(self, state: torch.Tensor) -> torch.Tensor:
        positive = _positive(state, floor=1.0e-12)
        total = positive.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        return positive / total


@dataclass(frozen=True)
class PendulumConfig:
    """Damped, forced pendulum with an uncertain natural frequency."""

    gravity_over_length: float = 9.81
    damping: float = 0.12
    torque_amplitude: float = 0.24
    torque_frequency: float = 0.90
    uncertain_frequency: float = 1.10
    stochastic_scale: float = 0.006


class PendulumODE(HybridSystem):
    state_dim = 2

    def __init__(self, config: PendulumConfig | None = None) -> None:
        self.config = config or PendulumConfig()

    def drift(self, state: torch.Tensor, time: float, alpha_quantile: torch.Tensor) -> torch.Tensor:
        angle = state[..., 0]
        angular_velocity = state[..., 1]
        omega_sq = self.config.gravity_over_length + self.config.uncertain_frequency * alpha_quantile
        forcing = self.config.torque_amplitude * torch.cos(
            torch.as_tensor(self.config.torque_frequency * time, dtype=state.dtype, device=state.device)
        )
        d_angle = angular_velocity
        d_velocity = -omega_sq * torch.sin(angle) - self.config.damping * angular_velocity + forcing
        return torch.stack((d_angle, d_velocity), dim=-1)

    def diffusion(self, state: torch.Tensor, time: float) -> torch.Tensor:
        scale = torch.zeros_like(state)
        scale[..., 1] = self.config.stochastic_scale
        return scale

    def project(self, state: torch.Tensor) -> torch.Tensor:
        return state


@dataclass(frozen=True)
class AppliedODECase:
    """Numerical settings for one applied ODE benchmark."""

    name: str
    family: str
    state_dim: int
    observation_dim: int
    cognitive_parameter: str
    system_factory: Callable[[], HybridSystem]
    initial_state_factory: Callable[[torch.device], torch.Tensor]
    initial_scale_factory: Callable[[], torch.Tensor]
    observation_indices_factory: Callable[[torch.device], torch.Tensor]
    default_steps: int
    default_dt: float
    default_obs_interval: int
    default_ensemble_size: int
    default_obs_noise: float
    alpha_true: float = 0.12
    alpha_grid: tuple[float, ...] = (0.08, 0.22, 0.36, 0.50, 0.64, 0.78, 0.92)


def _state(values: list[float]) -> Callable[[torch.device], torch.Tensor]:
    def factory(device: torch.device) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.float64, device=device)

    return factory


def _scale(values: list[float]) -> Callable[[], torch.Tensor]:
    def factory() -> torch.Tensor:
        return torch.tensor(values, dtype=torch.float64)

    return factory


def _observed(indices: list[int]) -> Callable[[torch.device], torch.Tensor]:
    def factory(device: torch.device) -> torch.Tensor:
        return torch.tensor(indices, dtype=torch.int64, device=device)

    return factory


APPLIED_ODE_CASES: tuple[AppliedODECase, ...] = (
    AppliedODECase(
        name="chemical",
        family="reaction kinetics",
        state_dim=2,
        observation_dim=2,
        cognitive_parameter="reaction_rate",
        system_factory=lambda: ChemicalReactionODE(
            ChemicalReactionConfig(rate=0.55, uncertain_rate=0.12, stochastic_scale=0.014)
        ),
        initial_state_factory=_state([1.0, 0.0]),
        initial_scale_factory=_scale([0.025, 0.012]),
        observation_indices_factory=_observed([0, 1]),
        default_steps=240,
        default_dt=0.012,
        default_obs_interval=6,
        default_ensemble_size=26,
        default_obs_noise=0.018,
    ),
    AppliedODECase(
        name="pk_infusion",
        family="pharmacokinetics",
        state_dim=1,
        observation_dim=1,
        cognitive_parameter="elimination_or_infusion",
        system_factory=lambda: PharmacokineticInfusionODE(
            PharmacokineticInfusionConfig(
                infusion_rate=0.52,
                elimination_rate=0.44,
                uncertain_elimination=0.075,
                uncertain_infusion=0.045,
                stochastic_scale=0.010,
            )
        ),
        initial_state_factory=_state([0.02]),
        initial_scale_factory=_scale([0.025]),
        observation_indices_factory=_observed([0]),
        default_steps=280,
        default_dt=0.020,
        default_obs_interval=5,
        default_ensemble_size=24,
        default_obs_noise=0.020,
    ),
    AppliedODECase(
        name="pendulum",
        family="nonlinear mechanics",
        state_dim=2,
        observation_dim=2,
        cognitive_parameter="natural_frequency_or_damping",
        system_factory=lambda: PendulumODE(PendulumConfig()),
        initial_state_factory=_state([0.30, 0.00]),
        initial_scale_factory=_scale([0.030, 0.030]),
        observation_indices_factory=_observed([0, 1]),
        default_steps=300,
        default_dt=0.010,
        default_obs_interval=6,
        default_ensemble_size=28,
        default_obs_noise=0.008,
    ),
    AppliedODECase(
        name="fhn",
        family="excitable dynamics",
        state_dim=2,
        observation_dim=2,
        cognitive_parameter="excitability_or_drive",
        system_factory=lambda: FitzHughNagumoODE(
            FitzHughNagumoConfig(
                recovery_timescale=0.10,
                recovery_offset=0.65,
                recovery_slope=0.78,
                external_drive=0.42,
                uncertain_drive=0.08,
                stochastic_scale=0.004,
            )
        ),
        initial_state_factory=_state([-0.85, -0.30]),
        initial_scale_factory=_scale([0.030, 0.030]),
        observation_indices_factory=_observed([0, 1]),
        default_steps=300,
        default_dt=0.008,
        default_obs_interval=6,
        default_ensemble_size=30,
        default_obs_noise=0.008,
    ),
    AppliedODECase(
        name="robertson",
        family="reaction kinetics",
        state_dim=3,
        observation_dim=2,
        cognitive_parameter="reaction_rate",
        system_factory=lambda: RobertsonODE(RobertsonConfig()),
        initial_state_factory=_state([1.0, 0.0, 0.0]),
        initial_scale_factory=_scale([0.010, 0.010, 0.010]),
        observation_indices_factory=_observed([1, 2]),
        default_steps=240,
        default_dt=0.001,
        default_obs_interval=8,
        default_ensemble_size=24,
        default_obs_noise=0.005,
    ),
)

APPLIED_ODE_CASE_NAMES: tuple[str, ...] = tuple(case.name for case in APPLIED_ODE_CASES)
APPLIED_ODE_CASE_MAP: dict[str, AppliedODECase] = {case.name: case for case in APPLIED_ODE_CASES}


def applied_ode_case_names() -> tuple[str, ...]:
    return APPLIED_ODE_CASE_NAMES


def applied_ode_case_spec(name: str) -> AppliedODECase:
    try:
        return APPLIED_ODE_CASE_MAP[name]
    except KeyError as exc:
        raise KeyError(f"Unknown applied ODE case: {name}") from exc
