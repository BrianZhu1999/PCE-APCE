from __future__ import annotations

from typing import Callable

import torch

from .config import LowRankConfig
from .low_rank import localized_low_rank_map
from .math_utils import spd_sqrt, stable_cholesky, symmetrize

ObservationOperator = Callable[[torch.Tensor], torch.Tensor]


def denkf_analysis(
    state_ensemble: torch.Tensor,
    observation: torch.Tensor,
    observation_operator: ObservationOperator,
    observation_covariance: torch.Tensor,
    localization: torch.Tensor | None = None,
) -> torch.Tensor:
    """Deterministic EnKF baseline with the DEnKF half-anomaly update."""

    predicted = observation_operator(state_ensemble)
    ensemble_size = state_ensemble.shape[0]
    x_mean = state_ensemble.mean(dim=0)
    z_mean = predicted.mean(dim=0)
    x_anomalies = state_ensemble - x_mean
    z_anomalies = predicted - z_mean
    cross_covariance = x_anomalies.mT @ z_anomalies / (ensemble_size - 1)
    innovation_covariance = (
        z_anomalies.mT @ z_anomalies / (ensemble_size - 1) + observation_covariance
    )
    factor = stable_cholesky(innovation_covariance)
    gain = torch.cholesky_solve(cross_covariance.mT, factor).mT
    if localization is not None:
        gain = gain * localization
    updated_mean = x_mean + gain @ (observation - z_mean)
    updated_anomalies = x_anomalies - 0.5 * (z_anomalies @ gain.mT)
    return updated_mean.unsqueeze(0) + updated_anomalies


def letkf_analysis(
    state_ensemble: torch.Tensor,
    observation: torch.Tensor,
    observation_operator: ObservationOperator,
    observation_covariance: torch.Tensor,
) -> torch.Tensor:
    """Local ensemble transform Kalman filter for one supplied local patch."""

    predicted = observation_operator(state_ensemble)
    ensemble_size = state_ensemble.shape[0]
    state_mean = state_ensemble.mean(dim=0)
    observation_mean = predicted.mean(dim=0)
    state_anomalies = (state_ensemble - state_mean).mT
    observation_anomalies = (predicted - observation_mean).mT
    noise_factor = stable_cholesky(observation_covariance)
    whitened_observation_anomalies = torch.linalg.solve_triangular(
        noise_factor,
        observation_anomalies,
        upper=False,
    )
    whitened_innovation = torch.linalg.solve_triangular(
        noise_factor,
        (observation - observation_mean).unsqueeze(-1),
        upper=False,
    ).squeeze(-1)
    identity = torch.eye(
        ensemble_size,
        dtype=state_ensemble.dtype,
        device=state_ensemble.device,
    )
    precision = (
        (ensemble_size - 1) * identity
        + whitened_observation_anomalies.mT @ whitened_observation_anomalies
    )
    precision_factor = stable_cholesky(precision)
    analysis_covariance = torch.cholesky_solve(identity, precision_factor)
    mean_weights = analysis_covariance @ (
        whitened_observation_anomalies.mT @ whitened_innovation
    )
    transform = spd_sqrt(
        symmetrize((ensemble_size - 1) * analysis_covariance)
    )
    analysis_weights = mean_weights.unsqueeze(1) + transform
    return (state_mean.unsqueeze(1) + state_anomalies @ analysis_weights).mT


def linear_low_rank_analysis(
    state_ensemble: torch.Tensor,
    observation: torch.Tensor,
    observation_operator: ObservationOperator,
    observation_covariance: torch.Tensor,
    config: LowRankConfig | None = None,
    localization: torch.Tensor | None = None,
) -> torch.Tensor:
    """Linear EnSF-LR diagnostic core using a Gaussian observation correction."""

    predicted = observation_operator(state_ensemble)
    low_rank = localized_low_rank_map(
        state_ensemble,
        predicted,
        config or LowRankConfig(),
        localization,
    )
    observation_variance = predicted.var(dim=0, unbiased=True)
    noise_variance = torch.diagonal(observation_covariance)
    shrinkage = observation_variance / (observation_variance + noise_variance)
    target = predicted + shrinkage * (observation.unsqueeze(0) - predicted)
    return state_ensemble + (target - predicted) @ low_rank.gain.mT
