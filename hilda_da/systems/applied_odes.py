from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from .base import HybridSystem


def _positive(values: torch.Tensor, floor: float = 0.0) -> torch.Tensor:
    return torch.clamp(values, min=floor)


@dataclass(frozen=True)
class ChemicalReactionConfig:
    """Uncertain chemical reaction inspired by Tang and Yang (2021).

    State variables are a reactant concentration and a lumped product
    concentration for a second-order decomposition reaction.  The cognitive
    path modulates the effective reaction rate through the Liu normal
    quantile.
    """

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
class PharmacokineticBolusConfig:
    """Mono-compartment uncertain pharmacokinetic model.

    The alpha path follows dX/dt = -k X + sigma X F(alpha), corresponding to
    the one-factor uncertain differential equation in Liu and Yang (2021).
    """

    elimination_rate: float = 0.58
    uncertain_elimination: float = 0.13
    stochastic_scale: float = 0.015


class PharmacokineticBolusODE(HybridSystem):
    state_dim = 1

    def __init__(self, config: PharmacokineticBolusConfig | None = None) -> None:
        self.config = config or PharmacokineticBolusConfig()

    def drift(self, state: torch.Tensor, time: float, alpha_quantile: torch.Tensor) -> torch.Tensor:
        concentration = _positive(state[..., 0])
        derivative = (
            -self.config.elimination_rate * concentration
            + self.config.uncertain_elimination * concentration * alpha_quantile
        )
        return derivative.unsqueeze(-1)

    def diffusion(self, state: torch.Tensor, time: float) -> torch.Tensor:
        return self.config.stochastic_scale * (0.15 + _positive(state))

    def project(self, state: torch.Tensor) -> torch.Tensor:
        return _positive(state)


@dataclass(frozen=True)
class PharmacokineticInfusionConfig:
    """Multifactor uncertain pharmacokinetic infusion model.

    A single cognitive alpha path is used for the Figure 3 comparison by
    following the diagonal alpha-path specialization of the two Liu-process
    model: dX/dt = k0 - k1 X + sigma1 X F(alpha) + sigma2 F(alpha).
    """

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
class SIRRumourConfig:
    """Uncertain SIR rumour-spreading model from Sun, Sheng and Cui (2021)."""

    beta: float = 0.82
    contact_decay: float = 0.56
    forgetting: float = 0.08
    uncertain_contact: float = 0.30
    stochastic_scale: float = 0.010


class SIRRumourODE(HybridSystem):
    state_dim = 3

    def __init__(self, config: SIRRumourConfig | None = None) -> None:
        self.config = config or SIRRumourConfig()

    def drift(self, state: torch.Tensor, time: float, alpha_quantile: torch.Tensor) -> torch.Tensor:
        ignorant = _positive(state[..., 0])
        spreader = _positive(state[..., 1])
        stifler = _positive(state[..., 2])
        contact = ignorant * spreader
        stifling = spreader * stifler
        cognitive_flux = self.config.uncertain_contact * stifling * alpha_quantile
        d_ignorant = -contact
        d_spreader = (
            self.config.beta * contact
            - self.config.contact_decay * stifling
            - self.config.forgetting * spreader
            - cognitive_flux
        )
        d_stifler = (
            self.config.contact_decay * stifling
            + self.config.forgetting * spreader
            + (1.0 - self.config.beta) * contact
            + cognitive_flux
        )
        return torch.stack((d_ignorant, d_spreader, d_stifler), dim=-1)

    def diffusion(self, state: torch.Tensor, time: float) -> torch.Tensor:
        scale = self.config.stochastic_scale * (0.10 + _positive(state))
        return scale

    def project(self, state: torch.Tensor) -> torch.Tensor:
        positive = _positive(state, floor=1.0e-8)
        total = positive.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        return positive / total


@dataclass(frozen=True)
class SISEpidemicConfig:
    """Classical SIS epidemic model with uncertain transmission."""

    transmission: float = 0.88
    recovery: float = 0.34
    uncertain_transmission: float = 0.18
    stochastic_scale: float = 0.008


class SISEpidemicODE(HybridSystem):
    state_dim = 2

    def __init__(self, config: SISEpidemicConfig | None = None) -> None:
        self.config = config or SISEpidemicConfig()

    def drift(self, state: torch.Tensor, time: float, alpha_quantile: torch.Tensor) -> torch.Tensor:
        susceptible = _positive(state[..., 0])
        infected = _positive(state[..., 1])
        transmission = self.config.transmission + self.config.uncertain_transmission * alpha_quantile
        incidence = transmission * susceptible * infected
        d_susceptible = -incidence + self.config.recovery * infected
        d_infected = incidence - self.config.recovery * infected
        return torch.stack((d_susceptible, d_infected), dim=-1)

    def diffusion(self, state: torch.Tensor, time: float) -> torch.Tensor:
        return self.config.stochastic_scale * (0.10 + _positive(state))

    def project(self, state: torch.Tensor) -> torch.Tensor:
        positive = _positive(state, floor=1.0e-8)
        total = positive.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        return positive / total


@dataclass(frozen=True)
class SIREpidemicConfig:
    """Classical SIR epidemic model with uncertain contact rate."""

    transmission: float = 0.84
    recovery: float = 0.26
    loss_of_immunity: float = 0.02
    uncertain_transmission: float = 0.20
    stochastic_scale: float = 0.008


class SIREpidemicODE(HybridSystem):
    state_dim = 3

    def __init__(self, config: SIREpidemicConfig | None = None) -> None:
        self.config = config or SIREpidemicConfig()

    def drift(self, state: torch.Tensor, time: float, alpha_quantile: torch.Tensor) -> torch.Tensor:
        susceptible = _positive(state[..., 0])
        infected = _positive(state[..., 1])
        recovered = _positive(state[..., 2])
        transmission = self.config.transmission + self.config.uncertain_transmission * alpha_quantile
        incidence = transmission * susceptible * infected
        d_susceptible = -incidence + self.config.loss_of_immunity * recovered
        d_infected = incidence - self.config.recovery * infected
        d_recovered = self.config.recovery * infected - self.config.loss_of_immunity * recovered
        return torch.stack((d_susceptible, d_infected, d_recovered), dim=-1)

    def diffusion(self, state: torch.Tensor, time: float) -> torch.Tensor:
        return self.config.stochastic_scale * (0.10 + _positive(state))

    def project(self, state: torch.Tensor) -> torch.Tensor:
        positive = _positive(state, floor=1.0e-8)
        total = positive.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        return positive / total


@dataclass(frozen=True)
class SEIREpidemicConfig:
    """SEIR epidemic model with uncertain transmission."""

    transmission: float = 0.92
    incubation: float = 0.42
    recovery: float = 0.25
    uncertain_transmission: float = 0.18
    stochastic_scale: float = 0.008


class SEIREpidemicODE(HybridSystem):
    state_dim = 4

    def __init__(self, config: SEIREpidemicConfig | None = None) -> None:
        self.config = config or SEIREpidemicConfig()

    def drift(self, state: torch.Tensor, time: float, alpha_quantile: torch.Tensor) -> torch.Tensor:
        susceptible = _positive(state[..., 0])
        exposed = _positive(state[..., 1])
        infected = _positive(state[..., 2])
        recovered = _positive(state[..., 3])
        transmission = self.config.transmission + self.config.uncertain_transmission * alpha_quantile
        incidence = transmission * susceptible * infected
        d_susceptible = -incidence
        d_exposed = incidence - self.config.incubation * exposed
        d_infected = self.config.incubation * exposed - self.config.recovery * infected
        d_recovered = self.config.recovery * infected
        return torch.stack((d_susceptible, d_exposed, d_infected, d_recovered), dim=-1)

    def diffusion(self, state: torch.Tensor, time: float) -> torch.Tensor:
        return self.config.stochastic_scale * (0.10 + _positive(state))

    def project(self, state: torch.Tensor) -> torch.Tensor:
        positive = _positive(state, floor=1.0e-8)
        total = positive.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        return positive / total


@dataclass(frozen=True)
class LogisticGrowthConfig:
    """Uncertain logistic population growth."""

    growth_rate: float = 0.62
    carrying_capacity: float = 1.0
    uncertain_growth: float = 0.14
    stochastic_scale: float = 0.008


class LogisticGrowthODE(HybridSystem):
    state_dim = 1

    def __init__(self, config: LogisticGrowthConfig | None = None) -> None:
        self.config = config or LogisticGrowthConfig()

    def drift(self, state: torch.Tensor, time: float, alpha_quantile: torch.Tensor) -> torch.Tensor:
        population = _positive(state[..., 0])
        growth = self.config.growth_rate + self.config.uncertain_growth * alpha_quantile
        derivative = growth * population * (1.0 - population / self.config.carrying_capacity)
        return derivative.unsqueeze(-1)

    def diffusion(self, state: torch.Tensor, time: float) -> torch.Tensor:
        return self.config.stochastic_scale * (0.10 + _positive(state))

    def project(self, state: torch.Tensor) -> torch.Tensor:
        return torch.clamp(_positive(state), min=0.0, max=self.config.carrying_capacity)


@dataclass(frozen=True)
class GordonSchaeferConfig:
    """Uncertain Gordon--Schaefer renewable-resource model."""

    growth_rate: float = 0.68
    carrying_capacity: float = 1.0
    catchability: float = 0.18
    effort: float = 1.0
    uncertain_catchability: float = 0.10
    stochastic_scale: float = 0.008


class GordonSchaeferODE(HybridSystem):
    state_dim = 1

    def __init__(self, config: GordonSchaeferConfig | None = None) -> None:
        self.config = config or GordonSchaeferConfig()

    def drift(self, state: torch.Tensor, time: float, alpha_quantile: torch.Tensor) -> torch.Tensor:
        biomass = _positive(state[..., 0])
        harvest_rate = self.config.catchability + self.config.uncertain_catchability * alpha_quantile
        growth = self.config.growth_rate * biomass * (1.0 - biomass / self.config.carrying_capacity)
        harvest = harvest_rate * self.config.effort * biomass
        return (growth - harvest).unsqueeze(-1)

    def diffusion(self, state: torch.Tensor, time: float) -> torch.Tensor:
        return self.config.stochastic_scale * (0.10 + _positive(state))

    def project(self, state: torch.Tensor) -> torch.Tensor:
        return torch.clamp(_positive(state), min=0.0, max=self.config.carrying_capacity)


@dataclass(frozen=True)
class RLCircuitConfig:
    """Uncertain RL electrical circuit."""

    resistance: float = 0.18
    inductance: float = 1.0
    voltage: float = 0.75
    uncertain_resistance: float = 0.08
    stochastic_scale: float = 0.008


class RLCircuitODE(HybridSystem):
    state_dim = 1

    def __init__(self, config: RLCircuitConfig | None = None) -> None:
        self.config = config or RLCircuitConfig()

    def drift(self, state: torch.Tensor, time: float, alpha_quantile: torch.Tensor) -> torch.Tensor:
        current = state[..., 0]
        resistance = self.config.resistance + self.config.uncertain_resistance * alpha_quantile
        derivative = (self.config.voltage - resistance * current) / self.config.inductance
        return derivative.unsqueeze(-1)

    def diffusion(self, state: torch.Tensor, time: float) -> torch.Tensor:
        return torch.full_like(state, self.config.stochastic_scale)

    def project(self, state: torch.Tensor) -> torch.Tensor:
        return torch.clamp(state, min=-3.0, max=3.0)


@dataclass(frozen=True)
class RLCCircuitConfig:
    """Uncertain RLC circuit with periodic forcing."""

    resistance: float = 0.14
    inductance: float = 0.90
    capacitance: float = 1.10
    drive_amplitude: float = 0.50
    drive_frequency: float = 0.80
    uncertain_resistance: float = 0.06
    stochastic_scale: float = 0.008


class RLCCircuitODE(HybridSystem):
    state_dim = 2

    def __init__(self, config: RLCCircuitConfig | None = None) -> None:
        self.config = config or RLCCircuitConfig()

    def drift(self, state: torch.Tensor, time: float, alpha_quantile: torch.Tensor) -> torch.Tensor:
        charge = state[..., 0]
        current = state[..., 1]
        resistance = self.config.resistance + self.config.uncertain_resistance * alpha_quantile
        forcing = self.config.drive_amplitude * torch.cos(
            torch.as_tensor(self.config.drive_frequency * time, dtype=state.dtype, device=state.device)
        )
        d_charge = current
        d_current = (
            forcing
            - resistance * current
            - charge / self.config.capacitance
        ) / self.config.inductance
        return torch.stack((d_charge, d_current), dim=-1)

    def diffusion(self, state: torch.Tensor, time: float) -> torch.Tensor:
        return torch.full_like(state, self.config.stochastic_scale)


@dataclass(frozen=True)
class VanDerPolConfig:
    """Van der Pol oscillator with uncertain nonlinear damping."""

    mu: float = 1.20
    uncertain_mu: float = 0.18
    stochastic_scale: float = 0.008


class VanDerPolODE(HybridSystem):
    state_dim = 2

    def __init__(self, config: VanDerPolConfig | None = None) -> None:
        self.config = config or VanDerPolConfig()

    def drift(self, state: torch.Tensor, time: float, alpha_quantile: torch.Tensor) -> torch.Tensor:
        position = state[..., 0]
        velocity = state[..., 1]
        mu = self.config.mu + self.config.uncertain_mu * alpha_quantile
        acceleration = mu * (1.0 - position.square()) * velocity - position
        return torch.stack((velocity, acceleration), dim=-1)

    def diffusion(self, state: torch.Tensor, time: float) -> torch.Tensor:
        scale = torch.zeros_like(state)
        scale[..., 1] = self.config.stochastic_scale
        return scale

    def project(self, state: torch.Tensor) -> torch.Tensor:
        return torch.clamp(state, min=-3.0, max=3.0)


@dataclass(frozen=True)
class DuffingConfig:
    """Duffing oscillator with uncertain linear stiffness."""

    damping: float = 0.18
    linear_stiffness: float = 0.80
    cubic_stiffness: float = 0.18
    forcing_amplitude: float = 0.35
    forcing_frequency: float = 0.90
    uncertain_stiffness: float = 0.12
    stochastic_scale: float = 0.008


class DuffingODE(HybridSystem):
    state_dim = 2

    def __init__(self, config: DuffingConfig | None = None) -> None:
        self.config = config or DuffingConfig()

    def drift(self, state: torch.Tensor, time: float, alpha_quantile: torch.Tensor) -> torch.Tensor:
        position = state[..., 0]
        velocity = state[..., 1]
        stiffness = self.config.linear_stiffness + self.config.uncertain_stiffness * alpha_quantile
        forcing = self.config.forcing_amplitude * torch.cos(
            torch.as_tensor(self.config.forcing_frequency * time, dtype=state.dtype, device=state.device)
        )
        acceleration = (
            -self.config.damping * velocity
            - stiffness * position
            - self.config.cubic_stiffness * position.pow(3)
            + forcing
        )
        return torch.stack((velocity, acceleration), dim=-1)

    def diffusion(self, state: torch.Tensor, time: float) -> torch.Tensor:
        scale = torch.zeros_like(state)
        scale[..., 1] = self.config.stochastic_scale
        return scale

    def project(self, state: torch.Tensor) -> torch.Tensor:
        return torch.clamp(state, min=-3.0, max=3.0)


@dataclass(frozen=True)
class LotkaVolterraConfig:
    """Uncertain Lotka--Volterra predator--prey dynamics."""

    prey_growth: float = 1.10
    predation: float = 0.90
    predator_death: float = 0.70
    conversion: float = 0.55
    uncertain_predation: float = 0.14
    stochastic_scale: float = 0.008


class LotkaVolterraODE(HybridSystem):
    state_dim = 2

    def __init__(self, config: LotkaVolterraConfig | None = None) -> None:
        self.config = config or LotkaVolterraConfig()

    def drift(self, state: torch.Tensor, time: float, alpha_quantile: torch.Tensor) -> torch.Tensor:
        prey = _positive(state[..., 0])
        predator = _positive(state[..., 1])
        predation = self.config.predation + self.config.uncertain_predation * alpha_quantile
        interaction = predation * prey * predator
        d_prey = self.config.prey_growth * prey - interaction
        d_predator = -self.config.predator_death * predator + self.config.conversion * interaction
        return torch.stack((d_prey, d_predator), dim=-1)

    def diffusion(self, state: torch.Tensor, time: float) -> torch.Tensor:
        return self.config.stochastic_scale * (0.10 + _positive(state))

    def project(self, state: torch.Tensor) -> torch.Tensor:
        return _positive(state, floor=1.0e-8)


@dataclass(frozen=True)
class FitzHughNagumoConfig:
    """FitzHugh--Nagumo excitable-cell oscillator with uncertain drive."""

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
class SEIARConfig:
    """Uncertain SEIAR benchmark used for Figure 3 final-v4."""

    transmission: float = 0.86
    asymptomatic_contact: float = 0.42
    incubation: float = 0.36
    symptomatic_recovery: float = 0.24
    asymptomatic_recovery: float = 0.18
    asymptomatic_fraction: float = 0.34
    uncertain_transmission: float = 0.12
    stochastic_scale: float = 0.007


class SEIARODE(HybridSystem):
    state_dim = 5

    def __init__(self, config: SEIARConfig | None = None) -> None:
        self.config = config or SEIARConfig()

    def drift(self, state: torch.Tensor, time: float, alpha_quantile: torch.Tensor) -> torch.Tensor:
        susceptible = _positive(state[..., 0])
        exposed = _positive(state[..., 1])
        infectious = _positive(state[..., 2])
        asymptomatic = _positive(state[..., 3])
        removed = _positive(state[..., 4])
        _ = removed
        beta = self.config.transmission + self.config.uncertain_transmission * alpha_quantile
        contact = susceptible * (infectious + self.config.asymptomatic_contact * asymptomatic)
        incidence = beta * contact
        asymptomatic_branch = self.config.asymptomatic_fraction * self.config.incubation * exposed
        symptomatic_branch = (1.0 - self.config.asymptomatic_fraction) * self.config.incubation * exposed
        d_s = -incidence
        d_e = incidence - self.config.incubation * exposed
        d_i = symptomatic_branch - self.config.symptomatic_recovery * infectious
        d_a = asymptomatic_branch - self.config.asymptomatic_recovery * asymptomatic
        d_r = self.config.symptomatic_recovery * infectious + self.config.asymptomatic_recovery * asymptomatic
        return torch.stack((d_s, d_e, d_i, d_a, d_r), dim=-1)

    def diffusion(self, state: torch.Tensor, time: float) -> torch.Tensor:
        scale = self.config.stochastic_scale * (0.10 + _positive(state))
        return scale

    def project(self, state: torch.Tensor) -> torch.Tensor:
        positive = _positive(state, floor=1.0e-8)
        total = positive.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        return positive / total


@dataclass(frozen=True)
class RobertsonConfig:
    """Stiff canonical Robertson reaction benchmark."""

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
        _ = species_z
        k2 = self.config.k2 + self.config.uncertain_k2 * alpha_quantile
        dx = -self.config.k1 * species_x + self.config.k3 * species_y * species_z
        dy = self.config.k1 * species_x - k2 * species_y.square() - self.config.k3 * species_y * species_z
        dz = k2 * species_y.square()
        return torch.stack((dx, dy, dz), dim=-1)

    def diffusion(self, state: torch.Tensor, time: float) -> torch.Tensor:
        scale = self.config.stochastic_scale * (0.10 + _positive(state))
        return scale

    def project(self, state: torch.Tensor) -> torch.Tensor:
        positive = _positive(state, floor=1.0e-12)
        total = positive.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        return positive / total


@dataclass(frozen=True)
class PendulumConfig:
    """Damped/forced pendulum benchmark with one uncertain frequency coordinate."""

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
    """Final Figure 3 registry entry."""

    name: str
    family: str
    tier: str
    state_dim: int
    primary_dim: int
    observation_dim: int
    cognitive_parameter: str
    source_metadata: dict[str, Any]
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


def _scalar_tensor(values: list[float]) -> Callable[[torch.device], torch.Tensor]:
    def factory(device: torch.device) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.float64, device=device)

    return factory


def _vector_scale(values: list[float]) -> Callable[[], torch.Tensor]:
    def factory() -> torch.Tensor:
        return torch.tensor(values, dtype=torch.float64)

    return factory


def _obs_indices(indices: list[int]) -> Callable[[torch.device], torch.Tensor]:
    def factory(device: torch.device) -> torch.Tensor:
        return torch.tensor(indices, dtype=torch.int64, device=device)

    return factory


FINAL_FIGURE3_CASES: tuple[AppliedODECase, ...] = (
    AppliedODECase(
        name="chemical",
        family="reaction kinetics",
        tier="A",
        state_dim=2,
        primary_dim=2,
        observation_dim=2,
        cognitive_parameter="reaction_rate",
        source_metadata={"paper": "Tang & Yang 2021", "doi": "10.1016/j.amc.2021.126479"},
        system_factory=lambda: ChemicalReactionODE(ChemicalReactionConfig(rate=0.55, uncertain_rate=0.12, stochastic_scale=0.014)),
        initial_state_factory=_scalar_tensor([1.0, 0.0]),
        initial_scale_factory=_vector_scale([0.025, 0.012]),
        observation_indices_factory=_obs_indices([0, 1]),
        default_steps=240,
        default_dt=0.012,
        default_obs_interval=6,
        default_ensemble_size=26,
        default_obs_noise=0.018,
    ),
    AppliedODECase(
        name="pk_infusion",
        family="pharmacokinetics",
        tier="A",
        state_dim=1,
        primary_dim=1,
        observation_dim=1,
        cognitive_parameter="elimination_or_infusion",
        source_metadata={"paper": "Liu & Yang 2021 multifactor PK", "doi": "10.1016/j.amc.2020.125722"},
        system_factory=lambda: PharmacokineticInfusionODE(
            PharmacokineticInfusionConfig(
                infusion_rate=0.52,
                elimination_rate=0.44,
                uncertain_elimination=0.075,
                uncertain_infusion=0.045,
                stochastic_scale=0.010,
            )
        ),
        initial_state_factory=_scalar_tensor([0.02]),
        initial_scale_factory=_vector_scale([0.025]),
        observation_indices_factory=_obs_indices([0]),
        default_steps=280,
        default_dt=0.020,
        default_obs_interval=5,
        default_ensemble_size=24,
        default_obs_noise=0.020,
    ),
    AppliedODECase(
        name="sir",
        family="spreading",
        tier="A",
        state_dim=3,
        primary_dim=3,
        observation_dim=3,
        cognitive_parameter="contact_or_forgetting",
        source_metadata={"paper": "Sun et al. 2021", "doi": "10.1186/s13662-021-03386-w"},
        system_factory=lambda: SIRRumourODE(
            SIRRumourConfig(beta=0.82, contact_decay=0.56, forgetting=0.08, uncertain_contact=0.30, stochastic_scale=0.008)
        ),
        initial_state_factory=_scalar_tensor([0.84, 0.13, 0.03]),
        initial_scale_factory=_vector_scale([0.020, 0.018, 0.010]),
        observation_indices_factory=_obs_indices([0, 1, 2]),
        default_steps=280,
        default_dt=0.020,
        default_obs_interval=5,
        default_ensemble_size=30,
        default_obs_noise=0.012,
    ),
    AppliedODECase(
        name="sis",
        family="spreading",
        tier="A",
        state_dim=2,
        primary_dim=2,
        observation_dim=2,
        cognitive_parameter="transmission_or_recovery",
        source_metadata={"paper": "Li et al. SIS UDE", "doi": "10.3233/JIFS-17354"},
        system_factory=lambda: SISEpidemicODE(
            SISEpidemicConfig(transmission=0.88, recovery=0.34, uncertain_transmission=0.18, stochastic_scale=0.008)
        ),
        initial_state_factory=_scalar_tensor([0.93, 0.07]),
        initial_scale_factory=_vector_scale([0.020, 0.020]),
        observation_indices_factory=_obs_indices([0, 1]),
        default_steps=240,
        default_dt=0.020,
        default_obs_interval=5,
        default_ensemble_size=28,
        default_obs_noise=0.010,
    ),
    AppliedODECase(
        name="seiar",
        family="spreading",
        tier="A",
        state_dim=5,
        primary_dim=5,
        observation_dim=3,
        cognitive_parameter="transmission_or_asymptomatic",
        source_metadata={"paper": "Jia & Chen 2021", "doi": "10.1007/s10700-020-09341-w"},
        system_factory=lambda: SEIARODE(
            SEIARConfig(
                transmission=0.86,
                asymptomatic_contact=0.42,
                incubation=0.36,
                symptomatic_recovery=0.24,
                asymptomatic_recovery=0.18,
                asymptomatic_fraction=0.34,
                uncertain_transmission=0.12,
                stochastic_scale=0.007,
            )
        ),
        initial_state_factory=_scalar_tensor([0.78, 0.09, 0.07, 0.04, 0.02]),
        initial_scale_factory=_vector_scale([0.020, 0.018, 0.016, 0.012, 0.010]),
        observation_indices_factory=_obs_indices([1, 2, 3]),
        default_steps=240,
        default_dt=0.018,
        default_obs_interval=5,
        default_ensemble_size=30,
        default_obs_noise=0.010,
    ),
    AppliedODECase(
        name="logistic",
        family="population",
        tier="A",
        state_dim=1,
        primary_dim=1,
        observation_dim=1,
        cognitive_parameter="growth_or_carrying",
        source_metadata={"paper": "Zhang & Yang 2018", "doi": "10.1007/s00500-018-03678-6"},
        system_factory=lambda: LogisticGrowthODE(
            LogisticGrowthConfig(growth_rate=0.62, carrying_capacity=1.0, uncertain_growth=0.14, stochastic_scale=0.008)
        ),
        initial_state_factory=_scalar_tensor([0.18]),
        initial_scale_factory=_vector_scale([0.025]),
        observation_indices_factory=_obs_indices([0]),
        default_steps=220,
        default_dt=0.020,
        default_obs_interval=5,
        default_ensemble_size=24,
        default_obs_noise=0.012,
    ),
    AppliedODECase(
        name="gordon_schaefer",
        family="population/resource",
        tier="A",
        state_dim=1,
        primary_dim=1,
        observation_dim=1,
        cognitive_parameter="catchability_or_harvest",
        source_metadata={"paper": "Chen & Liu 2023", "doi": "10.1016/j.amc.2023.128011"},
        system_factory=lambda: GordonSchaeferODE(
            GordonSchaeferConfig(growth_rate=0.68, carrying_capacity=1.0, catchability=0.18, effort=1.0, uncertain_catchability=0.10, stochastic_scale=0.008)
        ),
        initial_state_factory=_scalar_tensor([0.35]),
        initial_scale_factory=_vector_scale([0.025]),
        observation_indices_factory=_obs_indices([0]),
        default_steps=220,
        default_dt=0.020,
        default_obs_interval=5,
        default_ensemble_size=24,
        default_obs_noise=0.012,
    ),
    AppliedODECase(
        name="rl_circuit",
        family="engineering",
        tier="A",
        state_dim=1,
        primary_dim=1,
        observation_dim=1,
        cognitive_parameter="resistance_or_input",
        source_metadata={"paper": "Liu & Zhou 2021", "doi": "10.3390/sym13112103"},
        system_factory=lambda: RLCircuitODE(
            RLCircuitConfig(resistance=0.16, inductance=0.95, voltage=0.70, uncertain_resistance=0.05, stochastic_scale=0.004)
        ),
        initial_state_factory=_scalar_tensor([0.04]),
        initial_scale_factory=_vector_scale([0.025]),
        observation_indices_factory=_obs_indices([0]),
        default_steps=220,
        default_dt=0.012,
        default_obs_interval=5,
        default_ensemble_size=24,
        default_obs_noise=0.012,
    ),
    AppliedODECase(
        name="van_der_pol",
        family="nonlinear dynamics",
        tier="B",
        state_dim=2,
        primary_dim=2,
        observation_dim=2,
        cognitive_parameter="nonlinear_damping",
        source_metadata={"paper": "canonical ODE benchmark", "doi": "UNVERIFIED"},
        system_factory=lambda: VanDerPolODE(VanDerPolConfig(mu=1.00, uncertain_mu=0.12, stochastic_scale=0.004)),
        initial_state_factory=_scalar_tensor([1.00, 0.00]),
        initial_scale_factory=_vector_scale([0.030, 0.030]),
        observation_indices_factory=_obs_indices([0, 1]),
        default_steps=300,
        default_dt=0.008,
        default_obs_interval=6,
        default_ensemble_size=30,
        default_obs_noise=0.008,
    ),
    AppliedODECase(
        name="duffing",
        family="nonlinear dynamics",
        tier="B",
        state_dim=2,
        primary_dim=2,
        observation_dim=2,
        cognitive_parameter="nonlinear_stiffness",
        source_metadata={"paper": "canonical ODE benchmark", "doi": "UNVERIFIED"},
        system_factory=lambda: DuffingODE(
            DuffingConfig(damping=0.24, linear_stiffness=0.95, cubic_stiffness=0.20, forcing_amplitude=0.28, forcing_frequency=0.85, uncertain_stiffness=0.08, stochastic_scale=0.004)
        ),
        initial_state_factory=_scalar_tensor([0.15, 0.00]),
        initial_scale_factory=_vector_scale([0.030, 0.030]),
        observation_indices_factory=_obs_indices([0, 1]),
        default_steps=300,
        default_dt=0.008,
        default_obs_interval=6,
        default_ensemble_size=30,
        default_obs_noise=0.008,
    ),
    AppliedODECase(
        name="lotka_volterra",
        family="population/resource",
        tier="B",
        state_dim=2,
        primary_dim=2,
        observation_dim=2,
        cognitive_parameter="interaction_coefficient",
        source_metadata={"paper": "canonical ODE benchmark", "doi": "UNVERIFIED"},
        system_factory=lambda: LotkaVolterraODE(
            LotkaVolterraConfig(prey_growth=1.10, predation=0.90, predator_death=0.70, conversion=0.55, uncertain_predation=0.14, stochastic_scale=0.008)
        ),
        initial_state_factory=_scalar_tensor([0.65, 0.25]),
        initial_scale_factory=_vector_scale([0.030, 0.025]),
        observation_indices_factory=_obs_indices([0, 1]),
        default_steps=240,
        default_dt=0.012,
        default_obs_interval=5,
        default_ensemble_size=28,
        default_obs_noise=0.012,
    ),
    AppliedODECase(
        name="fhn",
        family="excitable dynamics",
        tier="B",
        state_dim=2,
        primary_dim=2,
        observation_dim=2,
        cognitive_parameter="excitability_or_drive",
        source_metadata={"paper": "canonical ODE benchmark", "doi": "UNVERIFIED"},
        system_factory=lambda: FitzHughNagumoODE(
            FitzHughNagumoConfig(recovery_timescale=0.10, recovery_offset=0.65, recovery_slope=0.78, external_drive=0.42, uncertain_drive=0.08, stochastic_scale=0.004)
        ),
        initial_state_factory=_scalar_tensor([-0.85, -0.30]),
        initial_scale_factory=_vector_scale([0.030, 0.030]),
        observation_indices_factory=_obs_indices([0, 1]),
        default_steps=300,
        default_dt=0.008,
        default_obs_interval=6,
        default_ensemble_size=30,
        default_obs_noise=0.008,
    ),
    AppliedODECase(
        name="robertson",
        family="reaction kinetics",
        tier="B",
        state_dim=3,
        primary_dim=3,
        observation_dim=2,
        cognitive_parameter="reaction_rate",
        source_metadata={"paper": "canonical stiff ODE benchmark", "doi": "UNVERIFIED"},
        system_factory=lambda: RobertsonODE(RobertsonConfig()),
        initial_state_factory=_scalar_tensor([1.0, 0.0, 0.0]),
        initial_scale_factory=_vector_scale([0.010, 0.010, 0.010]),
        observation_indices_factory=_obs_indices([1, 2]),
        default_steps=240,
        default_dt=0.001,
        default_obs_interval=8,
        default_ensemble_size=24,
        default_obs_noise=0.005,
    ),
    AppliedODECase(
        name="pendulum",
        family="engineering/physical",
        tier="B",
        state_dim=2,
        primary_dim=2,
        observation_dim=2,
        cognitive_parameter="natural_frequency_or_damping",
        source_metadata={"paper": "canonical ODE benchmark", "doi": "UNVERIFIED"},
        system_factory=lambda: PendulumODE(PendulumConfig()),
        initial_state_factory=_scalar_tensor([0.30, 0.00]),
        initial_scale_factory=_vector_scale([0.030, 0.030]),
        observation_indices_factory=_obs_indices([0, 1]),
        default_steps=300,
        default_dt=0.010,
        default_obs_interval=6,
        default_ensemble_size=28,
        default_obs_noise=0.008,
    ),
)


FINAL_FIGURE3_CASE_NAMES: tuple[str, ...] = tuple(spec.name for spec in FINAL_FIGURE3_CASES)
FINAL_FIGURE3_CASE_MAP: dict[str, AppliedODECase] = {spec.name: spec for spec in FINAL_FIGURE3_CASES}


def final_figure3_case_names() -> tuple[str, ...]:
    return FINAL_FIGURE3_CASE_NAMES


def final_figure3_case_spec(name: str) -> AppliedODECase:
    try:
        return FINAL_FIGURE3_CASE_MAP[name]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise KeyError(f"Unknown final Figure 3 case: {name}") from exc
