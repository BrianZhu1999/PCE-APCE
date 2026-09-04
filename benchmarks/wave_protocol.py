"""Wave scenario and predictive-evidence utilities used by the paper benchmark."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .wave_assets import WaveScenarioAssets
from .wave_model import (
    Config,
    initial_state,
    liu_normal_inverse,
    make_config,
    propagate_batch,
    smooth_noise,
)


METHOD_SEED_OFFSETS = {
    "denkf": 1_679_088_713,
    "letkf": 3_337_870_379,
    "ensf_lr_ridge": 203_002_971,
}


@dataclass
class Scenario:
    cfg: Config
    x: np.ndarray
    times: np.ndarray
    alpha_grid: np.ndarray
    theta_grid: np.ndarray
    theta_true: float
    truth_states: np.ndarray
    observations: dict[int, np.ndarray]
    observation_indices: np.ndarray
    ensemble_initial: np.ndarray
    branch_initial: np.ndarray
    branch_initial_independent: np.ndarray
    truth_noise: np.ndarray
    forecast_noise: np.ndarray


def stable_offset(name: str) -> int:
    """Return the fixed random-number offset assigned to a comparison method."""
    try:
        return METHOD_SEED_OFFSETS[name]
    except KeyError as error:
        raise ValueError(f"No deterministic seed offset is defined for {name!r}") from error


def softmax(log_weights: np.ndarray) -> np.ndarray:
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)
    return weights / np.sum(weights)


def alpha_to_theta(alpha: float, cfg: Config) -> float:
    return float(cfg.epistemic_scale * liu_normal_inverse(alpha))


def theta_to_alpha(theta: float, cfg: Config) -> float:
    z = theta / max(cfg.epistemic_scale, 1.0e-12)
    return float(1.0 / (1.0 + math.exp(-math.pi * z / math.sqrt(3.0))))


def generate_scenario(cfg: Config) -> Scenario:
    rng = np.random.default_rng(cfg.seed)
    x = np.linspace(0.0, cfg.length, cfg.nx)
    n_steps = int(round(cfg.t_end / cfg.dt))
    times = np.arange(n_steps + 1) * cfg.dt
    alpha_grid = np.linspace(cfg.alpha_min, cfg.alpha_max, cfg.n_alpha)
    theta_grid = cfg.epistemic_scale * liu_normal_inverse(alpha_grid)
    theta_true = alpha_to_theta(cfg.alpha_true, cfg)
    observation_indices = np.linspace(4, cfg.nx - 5, cfg.n_sensors, dtype=int)

    x0 = initial_state(x)
    truth = x0[None, :].copy()
    truth_states = np.zeros((n_steps + 1, 2 * cfg.nx), dtype=float)
    truth_states[0] = truth[0]
    truth_noise = np.stack(
        [smooth_noise(rng.normal(size=(1, cfg.nx))) for _ in range(n_steps)],
        axis=0,
    )
    observation_noise = {
        step: cfg.obs_noise * rng.normal(size=cfg.n_sensors)
        for step in range(cfg.obs_interval, n_steps + 1, cfg.obs_interval)
    }

    for step in range(1, n_steps + 1):
        truth = propagate_batch(
            truth,
            theta_true,
            times[step - 1],
            cfg,
            rng,
            stochastic=True,
            noise_draw=truth_noise[step - 1],
        )
        truth_states[step] = truth[0]

    observations = {
        step: truth_states[step, observation_indices] + noise
        for step, noise in observation_noise.items()
    }
    du = 0.012 * smooth_noise(rng.normal(size=(cfg.ensemble_size, cfg.nx)))
    dv = 0.025 * smooth_noise(rng.normal(size=(cfg.ensemble_size, cfg.nx)))
    ensemble_initial = x0[None, :] + np.concatenate([du, dv], axis=1)
    boundary = [0, cfg.nx - 1, cfg.nx, 2 * cfg.nx - 1]
    ensemble_initial[:, boundary] = 0.0
    branch_initial = np.repeat(ensemble_initial[None, :, :], cfg.n_alpha, axis=0)

    branch_initial_independent = np.empty_like(branch_initial)
    for index in range(cfg.n_alpha):
        du_branch = 0.012 * smooth_noise(rng.normal(size=(cfg.ensemble_size, cfg.nx)))
        dv_branch = 0.025 * smooth_noise(rng.normal(size=(cfg.ensemble_size, cfg.nx)))
        branch_initial_independent[index] = x0[None, :] + np.concatenate(
            [du_branch, dv_branch], axis=1
        )
        branch_initial_independent[index, :, boundary] = 0.0

    forecast_noise = np.stack(
        [smooth_noise(rng.normal(size=(cfg.ensemble_size, cfg.nx))) for _ in range(n_steps)],
        axis=0,
    )
    return Scenario(
        cfg=cfg,
        x=x,
        times=times,
        alpha_grid=alpha_grid,
        theta_grid=theta_grid,
        theta_true=theta_true,
        truth_states=truth_states,
        observations=observations,
        observation_indices=observation_indices,
        ensemble_initial=ensemble_initial,
        branch_initial=branch_initial,
        branch_initial_independent=branch_initial_independent,
        truth_noise=truth_noise,
        forecast_noise=forecast_noise,
    )


def scenario_from_assets(assets: WaveScenarioAssets, cfg: Config | None = None) -> Scenario:
    """Build a wave scenario from one immutable set of paired inputs."""
    base_cfg = cfg or replace(
        make_config(),
        seed=assets.seed,
        nx=assets.nx,
        ensemble_size=assets.ensemble_size,
    )
    if base_cfg.nx != assets.nx or base_cfg.ensemble_size != assets.ensemble_size:
        raise ValueError("Configuration dimensions disagree with the paired wave assets")
    x = np.linspace(0.0, base_cfg.length, assets.nx)
    alpha_grid = np.linspace(base_cfg.alpha_min, base_cfg.alpha_max, base_cfg.n_alpha)
    theta_grid = base_cfg.epistemic_scale * liu_normal_inverse(alpha_grid)
    observations = {
        int(step): assets.observations[step].copy()
        for step in np.flatnonzero(assets.observation_mask)
    }
    branch_initial = np.repeat(
        assets.initial_ensemble[None, :, :], base_cfg.n_alpha, axis=0
    )
    return Scenario(
        cfg=base_cfg,
        x=x,
        times=assets.times.copy(),
        alpha_grid=alpha_grid,
        theta_grid=theta_grid,
        theta_true=alpha_to_theta(assets.alpha_true, base_cfg),
        truth_states=assets.truth_states.copy(),
        observations=observations,
        observation_indices=assets.observation_indices.copy(),
        ensemble_initial=assets.initial_ensemble.copy(),
        branch_initial=branch_initial,
        branch_initial_independent=branch_initial.copy(),
        truth_noise=assets.truth_noise.copy(),
        forecast_noise=assets.forecast_noise.copy(),
    )


def gaussian_log_evidence(
    ensemble_observation: np.ndarray,
    observation: np.ndarray,
    obs_noise: float,
    shrinkage: float,
    dimension_weights: np.ndarray | None = None,
) -> float:
    mean = ensemble_observation.mean(axis=0)
    anomalies = ensemble_observation - mean
    denominator = max(ensemble_observation.shape[0] - 1, 1)
    covariance = anomalies.T @ anomalies / denominator
    diagonal = np.diag(np.diag(covariance))
    covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal
    covariance += (obs_noise**2 + 1.0e-7) * np.eye(observation.size)
    residual = observation - mean

    if dimension_weights is not None:
        weights = np.asarray(dimension_weights, dtype=float)
        if weights.shape != residual.shape:
            raise ValueError("dimension_weights must match the observation dimension")
        weights = np.maximum(weights, 1.0e-8)
        weights = observation.size * weights / np.sum(weights)
        variances = np.maximum(np.diag(covariance), 1.0e-12)
        marginal_terms = residual**2 / variances + np.log(variances) + math.log(
            2.0 * math.pi
        )
        return float(-0.5 * np.sum(weights * marginal_terms))

    sign, log_determinant = np.linalg.slogdet(covariance)
    if sign <= 0:
        covariance += 1.0e-6 * np.eye(observation.size)
        sign, log_determinant = np.linalg.slogdet(covariance)
    mahalanobis = float(residual @ np.linalg.solve(covariance, residual))
    return float(
        -0.5
        * (mahalanobis + log_determinant + observation.size * math.log(2.0 * math.pi))
    )


def alpha_sensitivity_weights(
    branch_observations: list[np.ndarray], floor: float
) -> np.ndarray:
    means = np.stack([item.mean(axis=0) for item in branch_observations], axis=0)
    between = np.var(means, axis=0, ddof=1)
    normalized = between / max(float(np.max(between)), 1.0e-12)
    return floor + (1.0 - floor) * normalized


def continuous_alpha_estimate(
    alpha_grid: np.ndarray, log_weights: np.ndarray
) -> float:
    index = int(np.argmax(log_weights))
    if index == 0 or index == alpha_grid.size - 1:
        return float(alpha_grid[index])
    x = alpha_grid[index - 1 : index + 2]
    y = log_weights[index - 1 : index + 2]
    a, b, _ = np.polyfit(x, y, deg=2)
    if a >= -1.0e-12:
        return float(alpha_grid[index])
    return float(np.clip(-b / (2.0 * a), x[0], x[-1]))
