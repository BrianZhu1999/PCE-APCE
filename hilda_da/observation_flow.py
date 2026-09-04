from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .config import FlowConfig
from .math_utils import gaussian_logpdf, normalized_log_weights, spd_sqrt, stable_cholesky, symmetrize


@dataclass(frozen=True)
class PosteriorMixture:
    weights: torch.Tensor
    means: torch.Tensor
    covariance: torch.Tensor
    forecast_covariance: torch.Tensor
    log_evidence: torch.Tensor


def kernel_bandwidth(
    observations: torch.Tensor,
    observation_covariance: torch.Tensor,
) -> torch.Tensor:
    if observations.ndim != 2:
        raise ValueError("Forecast observations must have shape [ensemble, observation]")
    ensemble_size, dimension = observations.shape
    variance = observations.var(dim=0, unbiased=ensemble_size > 1)
    scale = (4.0 / (dimension + 2.0)) ** (1.0 / (dimension + 4.0))
    scale *= ensemble_size ** (-1.0 / (dimension + 4.0))
    floor = 1e-4 * torch.diagonal(observation_covariance)
    diagonal = scale * scale * variance + floor
    return torch.diag(diagonal.clamp_min(torch.finfo(observations.dtype).eps))


def analytic_posterior_mixture(
    forecast_observations: torch.Tensor,
    observation: torch.Tensor,
    observation_covariance: torch.Tensor,
    bandwidth: torch.Tensor | None = None,
    *,
    relative_floor: float = 1e-6,
) -> PosteriorMixture:
    if forecast_observations.ndim != 2:
        raise ValueError("Forecast observations must have shape [ensemble, observation]")
    if observation.ndim != 1:
        raise ValueError("Observation must be one-dimensional")
    if forecast_observations.shape[1] != observation.numel():
        raise ValueError("Observation dimension does not match forecast observations")
    bandwidth = bandwidth if bandwidth is not None else kernel_bandwidth(
        forecast_observations, observation_covariance
    )
    dimension = observation.numel()
    identity = torch.eye(dimension, dtype=observation.dtype, device=observation.device)
    bandwidth_factor = stable_cholesky(bandwidth, relative_floor)
    noise_factor = stable_cholesky(observation_covariance, relative_floor)
    bandwidth_precision = torch.cholesky_solve(identity, bandwidth_factor)
    noise_precision = torch.cholesky_solve(identity, noise_factor)
    posterior_precision = symmetrize(bandwidth_precision + noise_precision)
    posterior_factor = stable_cholesky(posterior_precision, relative_floor)
    posterior_covariance = torch.cholesky_solve(identity, posterior_factor)

    right_hand_side = (
        forecast_observations @ bandwidth_precision.mT
        + observation.unsqueeze(0) @ noise_precision.mT
    )
    posterior_means = right_hand_side @ posterior_covariance.mT

    predictive_covariance = symmetrize(bandwidth + observation_covariance)
    expanded_observation = observation.unsqueeze(0).expand_as(forecast_observations)
    component_log_evidence = gaussian_logpdf(
        expanded_observation,
        forecast_observations,
        predictive_covariance,
        relative_floor,
    )
    weights, _ = normalized_log_weights(component_log_evidence)
    log_evidence = torch.logsumexp(component_log_evidence, dim=0) - math.log(
        forecast_observations.shape[0]
    )
    return PosteriorMixture(
        weights=weights,
        means=posterior_means,
        covariance=posterior_covariance,
        forecast_covariance=bandwidth,
        log_evidence=log_evidence,
    )


def _sinkhorn_plan(
    centres: torch.Tensor,
    target_weights: torch.Tensor,
    config: FlowConfig,
) -> torch.Tensor:
    ensemble_size = centres.shape[0]
    source_weights = torch.full_like(target_weights, 1.0 / ensemble_size)
    cost = torch.cdist(centres, centres).square()
    positive_cost = cost[cost > 0]
    reference = positive_cost.median() if positive_cost.numel() else cost.new_tensor(1.0)
    epsilon = (config.sinkhorn_epsilon_scale * reference).clamp_min(
        torch.finfo(cost.dtype).eps
    )
    log_kernel = -cost / epsilon
    log_source = torch.log(source_weights)
    log_target = torch.log(target_weights.clamp_min(torch.finfo(target_weights.dtype).tiny))
    log_u = torch.zeros_like(source_weights)
    log_v = torch.zeros_like(target_weights)
    for _ in range(config.sinkhorn_iterations):
        log_u = log_source - torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
        log_v = log_target - torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0)
    plan = torch.exp(log_u.unsqueeze(1) + log_kernel + log_v.unsqueeze(0))
    return plan / plan.sum().clamp_min(torch.finfo(plan.dtype).tiny)


def _mixture_velocity(
    particles: torch.Tensor,
    tau: float,
    forecast_centres: torch.Tensor,
    mixture: PosteriorMixture,
    forecast_sqrt: torch.Tensor,
    posterior_sqrt: torch.Tensor,
    relative_floor: float,
) -> torch.Tensor:
    posterior_centres = mixture.means
    mean = (1.0 - tau) * forecast_centres + tau * posterior_centres
    interpolation_sqrt = (1.0 - tau) * forecast_sqrt + tau * posterior_sqrt
    covariance = interpolation_sqrt @ interpolation_sqrt.mT
    pairwise_particles = particles.unsqueeze(1)
    pairwise_means = mean.unsqueeze(0)
    component_log_density = gaussian_logpdf(
        pairwise_particles,
        pairwise_means,
        covariance,
        relative_floor,
    )
    responsibilities = torch.softmax(
        component_log_density + torch.log(mixture.weights).unsqueeze(0),
        dim=1,
    )
    differences = pairwise_particles - pairwise_means
    solved = torch.linalg.solve(
        interpolation_sqrt,
        differences.unsqueeze(-1),
    ).squeeze(-1)
    sqrt_increment = posterior_sqrt - forecast_sqrt
    component_velocity = (
        posterior_centres - forecast_centres
    ).unsqueeze(0) + solved @ sqrt_increment.mT
    return (responsibilities.unsqueeze(-1) * component_velocity).sum(dim=1)


def _match_representable_moments(
    particles: torch.Tensor,
    mixture: PosteriorMixture,
    relative_floor: float,
) -> torch.Tensor:
    target_mean = torch.sum(mixture.weights[:, None] * mixture.means, dim=0)
    target_centred = mixture.means - target_mean
    target_covariance = symmetrize(
        mixture.covariance
        + (
            mixture.weights[:, None, None]
            * target_centred[:, :, None]
            * target_centred[:, None, :]
        ).sum(dim=0)
    )
    centred = particles - particles.mean(dim=0)
    current_covariance = symmetrize(centred.mT @ centred / particles.shape[0])
    current_values, current_vectors = torch.linalg.eigh(current_covariance)
    target_values, target_vectors = torch.linalg.eigh(target_covariance)
    current_scale = current_values.abs().median().clamp_min(
        torch.finfo(current_values.dtype).eps
    )
    retained = current_values > relative_floor * current_scale
    rank = min(
        int(retained.sum()),
        particles.shape[0] - 1,
        particles.shape[1],
    )
    if rank == 0:
        return target_mean.unsqueeze(0).expand_as(particles).clone()
    current_values = current_values[-rank:].clamp_min(
        relative_floor * current_scale
    )
    current_vectors = current_vectors[:, -rank:]
    target_scale = target_values.abs().median().clamp_min(
        torch.finfo(target_values.dtype).eps
    )
    target_values = target_values[-rank:].clamp_min(relative_floor * target_scale)
    target_vectors = target_vectors[:, -rank:]
    standardized = centred @ current_vectors / torch.sqrt(current_values)
    corrected = (
        standardized
        * torch.sqrt(target_values).unsqueeze(0)
    ) @ target_vectors.mT
    corrected = corrected - corrected.mean(dim=0)
    return corrected + target_mean


def posterior_probability_flow(
    forecast_observations: torch.Tensor,
    mixture: PosteriorMixture,
    config: FlowConfig,
) -> torch.Tensor:
    """Transport forecast observations to the analytic posterior mixture."""

    plan = _sinkhorn_plan(forecast_observations, mixture.weights, config)
    source_mass = plan.sum(dim=1, keepdim=True)
    particles = (plan @ forecast_observations) / source_mass.clamp_min(
        torch.finfo(plan.dtype).tiny
    )
    forecast_sqrt = spd_sqrt(mixture.forecast_covariance, config.eigenvalue_floor)
    posterior_sqrt = spd_sqrt(mixture.covariance, config.eigenvalue_floor)
    step = 1.0 / config.steps
    for index in range(config.steps):
        tau = index * step
        velocity = _mixture_velocity(
            particles,
            tau,
            forecast_observations,
            mixture,
            forecast_sqrt,
            posterior_sqrt,
            config.eigenvalue_floor,
        )
        proposal = particles + step * velocity
        proposal_velocity = _mixture_velocity(
            proposal,
            tau + step,
            forecast_observations,
            mixture,
            forecast_sqrt,
            posterior_sqrt,
            config.eigenvalue_floor,
        )
        particles = particles + 0.5 * step * (velocity + proposal_velocity)
    if config.moment_matching:
        if not 0.0 <= config.moment_matching_strength <= 1.0:
            raise ValueError("moment_matching_strength must lie in [0,1]")
        corrected = _match_representable_moments(
            particles,
            mixture,
            config.eigenvalue_floor,
        )
        particles = torch.lerp(
            particles,
            corrected,
            config.moment_matching_strength,
        )
    return particles


def predictive_log_evidence(
    forecast_observations: torch.Tensor,
    observation: torch.Tensor,
    observation_covariance: torch.Tensor,
    bandwidth: torch.Tensor | None = None,
) -> torch.Tensor:
    mixture = analytic_posterior_mixture(
        forecast_observations,
        observation,
        observation_covariance,
        bandwidth,
    )
    return mixture.log_evidence / observation.numel()
