from __future__ import annotations

import argparse
import csv
import json
import math
import time
import tracemalloc
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CaseName = Literal["acoustic2d", "burgers1d", "allen_cahn1d"]
MethodName = Literal["misspecified", "oracle", "pce", "apce"]

METHODS: tuple[MethodName, ...] = ("misspecified", "oracle", "pce", "apce")
METHOD_LABELS = {
    "misspecified": "Misspecified EnSF-LR",
    "oracle": "Oracle-alpha EnSF-LR",
    "pce": "PCE",
    "apce": "APCE",
}


@dataclass(frozen=True)
class CaseConfig:
    name: CaseName
    seed: int = 20260803
    nx: int = 64
    ny: int = 1
    dt: float = 0.005
    n_steps: int = 100
    obs_interval: int = 5
    n_sensors: int = 8
    ensemble_size: int = 14
    process_noise: float = 0.006
    obs_noise: float = 0.025
    alpha_true: float = 0.70
    alpha_grid: tuple[float, ...] = (0.08, 0.22, 0.36, 0.50, 0.64, 0.78, 0.92)
    reverse_steps: int = 6
    sigma_min: float = 0.02
    sigma_max: float = 0.65
    guidance: float = 0.75
    reverse_noise_scale: float = 0.45
    regression_ridge: float = 1.0e-5
    shrinkage: float = 0.35
    pce_temperature: float = 0.50
    apce_temperature: float = 0.55
    apce_min_temperature: float = 0.16
    apce_forgetting: float = 0.985
    apce_sensitivity_floor: float = 0.35


@dataclass
class Scenario:
    cfg: CaseConfig
    truth: np.ndarray
    observations: dict[int, np.ndarray]
    observation_indices: np.ndarray
    initial_ensemble: np.ndarray
    forecast_noise: np.ndarray
    theta_grid: np.ndarray
    theta_true: float


def config_for_case(name: CaseName, seed: int) -> CaseConfig:
    if name == "acoustic2d":
        return CaseConfig(
            name=name,
            seed=seed,
            nx=18,
            ny=18,
            dt=0.010,
            n_steps=80,
            obs_interval=5,
            n_sensors=12,
            ensemble_size=14,
            process_noise=0.004,
            obs_noise=0.020,
            alpha_true=0.70,
        )
    if name == "burgers1d":
        return CaseConfig(
            name=name,
            seed=seed,
            nx=64,
            dt=0.0015,
            n_steps=140,
            obs_interval=7,
            n_sensors=8,
            ensemble_size=14,
            process_noise=0.006,
            obs_noise=0.025,
            alpha_true=0.70,
        )
    if name == "allen_cahn1d":
        return CaseConfig(
            name=name,
            seed=seed,
            nx=64,
            dt=0.008,
            n_steps=120,
            obs_interval=6,
            n_sensors=8,
            ensemble_size=14,
            process_noise=0.004,
            obs_noise=0.020,
            alpha_true=0.70,
        )
    raise ValueError(name)


def liu_normal_inverse(alpha: np.ndarray | float) -> np.ndarray:
    values = np.clip(np.asarray(alpha, dtype=float), 1.0e-7, 1.0 - 1.0e-7)
    return (math.sqrt(3.0) / math.pi) * np.log(values / (1.0 - values))


def theta_to_alpha(theta: float) -> float:
    return float(1.0 / (1.0 + math.exp(-math.pi * theta / math.sqrt(3.0))))


def smooth_periodic(values: np.ndarray) -> np.ndarray:
    return 0.25 * np.roll(values, 1, axis=-1) + 0.50 * values + 0.25 * np.roll(values, -1, axis=-1)


def state_dimension(cfg: CaseConfig) -> int:
    if cfg.name == "acoustic2d":
        return 3 * cfg.nx * cfg.ny
    return cfg.nx


def primary_dimension(cfg: CaseConfig) -> int:
    return cfg.nx * cfg.ny if cfg.name == "acoustic2d" else cfg.nx


def primary_field(states: np.ndarray, cfg: CaseConfig) -> np.ndarray:
    return states[..., : primary_dimension(cfg)]


def project_boundaries(states: np.ndarray, cfg: CaseConfig) -> np.ndarray:
    output = states.copy()
    if cfg.name == "acoustic2d":
        n = cfg.nx * cfg.ny
        p = output[..., :n].reshape(*output.shape[:-1], cfg.ny, cfg.nx)
        vx = output[..., n : 2 * n].reshape(*output.shape[:-1], cfg.ny, cfg.nx)
        vy = output[..., 2 * n :].reshape(*output.shape[:-1], cfg.ny, cfg.nx)
        p[..., 0, :] = p[..., 1, :]
        p[..., -1, :] = p[..., -2, :]
        p[..., :, 0] = p[..., :, 1]
        p[..., :, -1] = p[..., :, -2]
        vx[..., :, 0] = 0.0
        vx[..., :, -1] = 0.0
        vy[..., 0, :] = 0.0
        vy[..., -1, :] = 0.0
        return np.concatenate(
            [p.reshape(*output.shape[:-1], n), vx.reshape(*output.shape[:-1], n), vy.reshape(*output.shape[:-1], n)],
            axis=-1,
        )
    if cfg.name == "allen_cahn1d":
        output[..., 0] = output[..., 1]
        output[..., -1] = output[..., -2]
    return output


def initial_state(cfg: CaseConfig) -> np.ndarray:
    if cfg.name == "acoustic2d":
        x = np.linspace(0.0, 1.0, cfg.nx)
        y = np.linspace(0.0, 1.0, cfg.ny)
        xx, yy = np.meshgrid(x, y)
        pressure = 0.45 * np.exp(-((xx - 0.30) ** 2 + (yy - 0.35) ** 2) / 0.018)
        zeros = np.zeros_like(pressure)
        return np.concatenate([pressure.ravel(), zeros.ravel(), zeros.ravel()])
    x = np.linspace(0.0, 1.0, cfg.nx, endpoint=cfg.name != "burgers1d")
    if cfg.name == "burgers1d":
        return 0.55 * np.sin(2.0 * math.pi * x) + 0.18 * np.sin(4.0 * math.pi * x + 0.3)
    return np.tanh((0.34 - x) / 0.055) - np.tanh((0.74 - x) / 0.055) - 1.0


def propagate_acoustic2d(states: np.ndarray, theta: float, t: float, cfg: CaseConfig, noise: np.ndarray) -> np.ndarray:
    n = cfg.nx * cfg.ny
    dx = 1.0 / (cfg.nx - 1)
    dy = 1.0 / (cfg.ny - 1)
    p = states[:, :n].reshape(-1, cfg.ny, cfg.nx)
    vx = states[:, n : 2 * n].reshape(-1, cfg.ny, cfg.nx)
    vy = states[:, 2 * n :].reshape(-1, cfg.ny, cfg.nx)

    dpdx = np.gradient(p, dx, axis=2, edge_order=1)
    dpdy = np.gradient(p, dy, axis=1, edge_order=1)
    dvxdx = np.gradient(vx, dx, axis=2, edge_order=1)
    dvydy = np.gradient(vy, dy, axis=1, edge_order=1)

    x = np.linspace(0.0, 1.0, cfg.nx)
    y = np.linspace(0.0, 1.0, cfg.ny)
    xx, yy = np.meshgrid(x, y)
    source_main = 1.1 * math.sin(2.0 * math.pi * 2.0 * t) * math.exp(-0.7 * t)
    source_main *= np.exp(-((xx - 0.25) ** 2 + (yy - 0.30) ** 2) / 0.012)
    source_epistemic = math.sin(2.0 * math.pi * 1.35 * t + 0.4) * math.exp(-0.25 * t)
    source_epistemic *= np.exp(-((xx - 0.70) ** 2 + (yy - 0.68) ** 2) / 0.020)

    damping = 0.045
    p_new = p + cfg.dt * (-(dvxdx + dvydy) - damping * p + source_main + theta * source_epistemic)
    p_new += cfg.process_noise * math.sqrt(cfg.dt) * noise[:, :n].reshape(-1, cfg.ny, cfg.nx)
    vx_new = vx + cfg.dt * (-dpdx - damping * vx)
    vy_new = vy + cfg.dt * (-dpdy - damping * vy)
    updated = np.concatenate([p_new.reshape(-1, n), vx_new.reshape(-1, n), vy_new.reshape(-1, n)], axis=1)
    return project_boundaries(updated, cfg)


def propagate_burgers1d(states: np.ndarray, theta: float, t: float, cfg: CaseConfig, noise: np.ndarray) -> np.ndarray:
    dx = 1.0 / cfg.nx
    u = states
    ux = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2.0 * dx)
    uxx = (np.roll(u, -1, axis=1) - 2.0 * u + np.roll(u, 1, axis=1)) / (dx * dx)
    x = np.linspace(0.0, 1.0, cfg.nx, endpoint=False)
    viscosity = 0.022 * max(0.30, 1.0 + 0.32 * theta)
    forcing = 0.18 * np.sin(2.0 * math.pi * (x - 0.15 * t))
    forcing += 0.22 * theta * np.exp(-((x - 0.72) / 0.10) ** 2) * math.sin(2.0 * math.pi * 1.2 * t)
    updated = u + cfg.dt * (-u * ux + viscosity * uxx + forcing[None, :])
    updated += cfg.process_noise * math.sqrt(cfg.dt) * smooth_periodic(noise)
    return np.clip(updated, -2.5, 2.5)


def propagate_allen_cahn1d(states: np.ndarray, theta: float, t: float, cfg: CaseConfig, noise: np.ndarray) -> np.ndarray:
    dx = 1.0 / (cfg.nx - 1)
    u = states
    uxx = np.zeros_like(u)
    uxx[:, 1:-1] = (u[:, 2:] - 2.0 * u[:, 1:-1] + u[:, :-2]) / (dx * dx)
    x = np.linspace(0.0, 1.0, cfg.nx)
    reaction_rate = max(0.35, 1.0 + 0.28 * theta)
    source = 0.12 * theta * np.exp(-((x - 0.62) / 0.12) ** 2) * math.cos(2.0 * math.pi * 0.8 * t)
    updated = u + cfg.dt * (0.004 * uxx + reaction_rate * (u - u**3) + source[None, :])
    updated += cfg.process_noise * math.sqrt(cfg.dt) * noise
    return project_boundaries(np.clip(updated, -1.8, 1.8), cfg)


def propagate(states: np.ndarray, theta: float, t: float, cfg: CaseConfig, noise: np.ndarray) -> np.ndarray:
    if cfg.name == "acoustic2d":
        return propagate_acoustic2d(states, theta, t, cfg, noise)
    if cfg.name == "burgers1d":
        return propagate_burgers1d(states, theta, t, cfg, noise)
    return propagate_allen_cahn1d(states, theta, t, cfg, noise)


def observation_indices(cfg: CaseConfig) -> np.ndarray:
    if cfg.name == "acoustic2d":
        side_x = 4
        side_y = 3
        xs = np.linspace(2, cfg.nx - 3, side_x, dtype=int)
        ys = np.linspace(2, cfg.ny - 3, side_y, dtype=int)
        return np.asarray([y * cfg.nx + x for y in ys for x in xs], dtype=int)
    return np.linspace(3, cfg.nx - 4, cfg.n_sensors, dtype=int)


def generate_scenario(cfg: CaseConfig) -> Scenario:
    rng = np.random.default_rng(cfg.seed)
    dimension = state_dimension(cfg)
    theta_grid = liu_normal_inverse(np.asarray(cfg.alpha_grid))
    theta_true = float(liu_normal_inverse(cfg.alpha_true))
    truth = np.zeros((cfg.n_steps + 1, dimension), dtype=float)
    truth[0] = initial_state(cfg)
    truth_noise = rng.normal(size=(cfg.n_steps, 1, dimension))
    for step in range(1, cfg.n_steps + 1):
        truth[step] = propagate(
            truth[step - 1 : step], theta_true, (step - 1) * cfg.dt, cfg, truth_noise[step - 1]
        )[0]

    obs_idx = observation_indices(cfg)
    observations = {
        step: truth[step, obs_idx] + cfg.obs_noise * rng.normal(size=obs_idx.size)
        for step in range(cfg.obs_interval, cfg.n_steps + 1, cfg.obs_interval)
    }

    initial = initial_state(cfg)[None, :]
    anomalies = 0.018 * rng.normal(size=(cfg.ensemble_size, dimension))
    if cfg.name != "acoustic2d":
        anomalies = smooth_periodic(anomalies)
    ensemble = project_boundaries(initial + anomalies, cfg)
    forecast_noise = rng.normal(size=(cfg.n_steps, cfg.ensemble_size, dimension))
    return Scenario(cfg, truth, observations, obs_idx, ensemble, forecast_noise, theta_grid, theta_true)


def logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True)), axis=axis)


def ensf_observed_update(prior: np.ndarray, observation: np.ndarray, cfg: CaseConfig, rng: np.random.Generator) -> np.ndarray:
    mean = prior.mean(axis=0)
    scale = np.maximum(prior.std(axis=0, ddof=1), 0.025)
    clean = (prior - mean) / scale
    ratio = cfg.sigma_max / cfg.sigma_min
    log_ratio = math.log(ratio)
    z = clean + cfg.sigma_max * rng.normal(size=clean.shape)
    pseudo_times = np.linspace(1.0, 0.0, cfg.reverse_steps + 1)
    for index in range(cfg.reverse_steps):
        tau = max(float(pseudo_times[index]), 1.0e-4)
        next_tau = max(float(pseudo_times[index + 1]), 0.0)
        delta = next_tau - tau
        sigma = cfg.sigma_min * ratio**tau
        diffusion = sigma * math.sqrt(2.0 * log_ratio)
        difference = z[:, None, :] - clean[None, :, :]
        log_weights = -0.5 * np.sum(difference * difference, axis=2) / (sigma * sigma)
        log_weights -= logsumexp(log_weights, axis=1)[:, None]
        weights = np.exp(log_weights)
        denoised = weights @ clean
        prior_score = (denoised - z) / (sigma * sigma)
        predicted = mean[None, :] + scale[None, :] * denoised
        likelihood_score = scale[None, :] * (observation[None, :] - predicted) / (cfg.obs_noise**2)
        score = np.clip(prior_score + cfg.guidance * (1.0 - tau) * likelihood_score, -40.0, 40.0)
        z = z + (-diffusion * diffusion * score) * delta
        z += cfg.reverse_noise_scale * diffusion * math.sqrt(-delta) * rng.normal(size=z.shape)
    return mean + scale * z


def ensf_lr_update(
    prior: np.ndarray,
    observation: np.ndarray,
    obs_idx: np.ndarray,
    cfg: CaseConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    updated_observed = ensf_observed_update(prior[:, obs_idx], observation, cfg, rng)
    increment = updated_observed - prior[:, obs_idx]
    all_idx = np.arange(prior.shape[1])
    unobserved = np.setdiff1d(all_idx, obs_idx)
    updated = prior.copy()
    updated[:, obs_idx] = updated_observed
    if unobserved.size:
        observed_anomaly = prior[:, obs_idx] - prior[:, obs_idx].mean(axis=0)
        unobserved_anomaly = prior[:, unobserved] - prior[:, unobserved].mean(axis=0)
        denom = max(prior.shape[0] - 1, 1)
        covariance_oo = observed_anomaly.T @ observed_anomaly / denom
        covariance_uo = unobserved_anomaly.T @ observed_anomaly / denom
        ridge = cfg.regression_ridge * (np.trace(covariance_oo) / max(covariance_oo.shape[0], 1) + 1.0)
        gain = covariance_uo @ np.linalg.pinv(covariance_oo + ridge * np.eye(covariance_oo.shape[0]))
        updated[:, unobserved] = prior[:, unobserved] + increment @ gain.T
    return project_boundaries(updated, cfg)


def gaussian_evidence(
    ensemble_observation: np.ndarray,
    observation: np.ndarray,
    cfg: CaseConfig,
    dimension_weights: np.ndarray | None,
) -> float:
    mean = ensemble_observation.mean(axis=0)
    anomaly = ensemble_observation - mean
    covariance = anomaly.T @ anomaly / max(ensemble_observation.shape[0] - 1, 1)
    covariance = (1.0 - cfg.shrinkage) * covariance + cfg.shrinkage * np.diag(np.diag(covariance))
    covariance += (cfg.obs_noise**2 + 1.0e-7) * np.eye(observation.size)
    residual = observation - mean
    if dimension_weights is not None:
        weights = np.asarray(dimension_weights, dtype=float)
        if weights.shape != residual.shape:
            raise ValueError("dimension_weights must match the observation dimension")
        weights = np.maximum(weights, 1.0e-8)
        weights = observation.size * weights / np.sum(weights)
        variances = np.maximum(np.diag(covariance), 1.0e-12)
        marginal_terms = residual * residual / variances + np.log(variances) + math.log(2.0 * math.pi)
        return float(-0.5 * np.sum(weights * marginal_terms))
    sign, log_det = np.linalg.slogdet(covariance)
    if sign <= 0:
        covariance += 1.0e-6 * np.eye(observation.size)
        _, log_det = np.linalg.slogdet(covariance)
    mahalanobis = float(residual @ np.linalg.solve(covariance, residual))
    return float(-0.5 * (mahalanobis + log_det + observation.size * math.log(2.0 * math.pi)))


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    weights = np.exp(shifted)
    return weights / weights.sum()


def entropy(weights: np.ndarray) -> float:
    values = np.maximum(weights, 1.0e-300)
    return float(-np.sum(values * np.log(values)))


def entropy_target(progress: float) -> float:
    if progress <= 0.70:
        return 1.20 + (0.72 - 1.20) * progress / 0.70
    return 0.72 + (1.25 - 0.72) * (progress - 0.70) / 0.30


def entropy_project(weights: np.ndarray, target: float) -> np.ndarray:
    if entropy(weights) >= target:
        return weights
    uniform = np.ones_like(weights) / weights.size
    low, high = 0.0, 1.0
    for _ in range(45):
        middle = 0.5 * (low + high)
        mixed = (1.0 - middle) * weights + middle * uniform
        if entropy(mixed) < target:
            low = middle
        else:
            high = middle
    mixed = (1.0 - high) * weights + high * uniform
    return mixed / mixed.sum()


def continuous_alpha(alpha_grid: np.ndarray, theta_grid: np.ndarray, weights: np.ndarray, apce: bool) -> float:
    log_weights = np.log(np.maximum(weights, 1.0e-300))
    q = int(np.argmax(log_weights))
    quadratic = float(alpha_grid[q])
    if 0 < q < alpha_grid.size - 1:
        a, b, _ = np.polyfit(alpha_grid[q - 1 : q + 2], log_weights[q - 1 : q + 2], 2)
        if a < -1.0e-12:
            quadratic = float(np.clip(-b / (2.0 * a), alpha_grid[q - 1], alpha_grid[q + 1]))
    if not apce:
        return quadratic
    posterior_mean = theta_to_alpha(float(np.sum(weights * theta_grid)))
    concentration = float(np.max(weights))
    blend = float(np.clip((concentration - 0.25) / 0.35, 0.0, 1.0))
    return blend * quadratic + (1.0 - blend) * posterior_mean


def state_rmse(estimate: np.ndarray, truth: np.ndarray, cfg: CaseConfig) -> float:
    return float(np.sqrt(np.mean((primary_field(estimate, cfg) - primary_field(truth, cfg)) ** 2)))


def run_method(scenario: Scenario, method: MethodName) -> dict[str, Any]:
    cfg = scenario.cfg
    rng = np.random.default_rng(cfg.seed + {"misspecified": 11, "oracle": 23, "pce": 37, "apce": 53}[method])
    rmse = np.zeros(cfg.n_steps + 1, dtype=float)
    weights = np.ones(len(cfg.alpha_grid), dtype=float) / len(cfg.alpha_grid)
    log_weights = np.log(weights)
    final_estimate: np.ndarray

    tracemalloc.start()
    start = time.perf_counter()

    if method in {"misspecified", "oracle"}:
        ensemble = scenario.initial_ensemble.copy()
        theta = 0.0 if method == "misspecified" else scenario.theta_true
        rmse[0] = state_rmse(ensemble.mean(axis=0), scenario.truth[0], cfg)
        for step in range(1, cfg.n_steps + 1):
            ensemble = propagate(
                ensemble, theta, (step - 1) * cfg.dt, cfg, scenario.forecast_noise[step - 1]
            )
            if step in scenario.observations:
                ensemble = ensf_lr_update(
                    ensemble, scenario.observations[step], scenario.observation_indices, cfg, rng
                )
            rmse[step] = state_rmse(ensemble.mean(axis=0), scenario.truth[step], cfg)
        final_estimate = ensemble.mean(axis=0)
        alpha_best = cfg.alpha_true if method == "oracle" else 0.50
        alpha_cont = alpha_best
        final_entropy = None
    else:
        branches = np.repeat(scenario.initial_ensemble[None, :, :], len(cfg.alpha_grid), axis=0)
        evidence_branches = branches.copy()
        rmse[0] = state_rmse(np.mean(branches, axis=(0, 1)), scenario.truth[0], cfg)
        for step in range(1, cfg.n_steps + 1):
            for q, theta in enumerate(scenario.theta_grid):
                noise = scenario.forecast_noise[step - 1]
                branches[q] = propagate(branches[q], float(theta), (step - 1) * cfg.dt, cfg, noise)
                evidence_branches[q] = propagate(
                    evidence_branches[q], float(theta), (step - 1) * cfg.dt, cfg, noise
                )
            if step in scenario.observations:
                observation = scenario.observations[step]
                branch_observations = [
                    evidence_branches[q][:, scenario.observation_indices]
                    for q in range(len(cfg.alpha_grid))
                ]
                dimension_weights = None
                if method == "apce":
                    means = np.stack([item.mean(axis=0) for item in branch_observations])
                    between = np.var(means, axis=0, ddof=1)
                    normalized = between / max(float(np.max(between)), 1.0e-12)
                    dimension_weights = cfg.apce_sensitivity_floor + (1.0 - cfg.apce_sensitivity_floor) * normalized
                evidence = np.asarray(
                    [gaussian_evidence(item, observation, cfg, dimension_weights) for item in branch_observations]
                )
                centered = evidence - evidence.mean()
                if method == "pce":
                    log_weights = log_weights + cfg.pce_temperature * centered
                else:
                    entropy_ratio = entropy(weights) / max(math.log(len(cfg.alpha_grid)), 1.0e-12)
                    temperature = float(
                        np.clip(
                            cfg.apce_temperature * entropy_ratio**0.75,
                            cfg.apce_min_temperature,
                            cfg.apce_temperature,
                        )
                    )
                    log_weights = cfg.apce_forgetting * log_weights + temperature * centered
                weights = softmax(log_weights)
                if method == "apce":
                    weights = entropy_project(weights, entropy_target(step / cfg.n_steps))
                log_weights = np.log(np.maximum(weights, 1.0e-300))
                paired_seed = cfg.seed + 10_000_000 + step
                for q in range(len(cfg.alpha_grid)):
                    branches[q] = ensf_lr_update(
                        branches[q],
                        observation,
                        scenario.observation_indices,
                        cfg,
                        np.random.default_rng(paired_seed),
                    )
            estimate = np.sum(weights[:, None] * branches.mean(axis=1), axis=0)
            rmse[step] = state_rmse(estimate, scenario.truth[step], cfg)
        final_estimate = np.sum(weights[:, None] * branches.mean(axis=1), axis=0)
        alpha_best = float(np.asarray(cfg.alpha_grid)[int(np.argmax(weights))])
        alpha_cont = continuous_alpha(
            np.asarray(cfg.alpha_grid), scenario.theta_grid, weights, method == "apce"
        )
        final_entropy = entropy(weights)

    elapsed = time.perf_counter() - start
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "method": method,
        "label": METHOD_LABELS[method],
        "mean_rmse": float(rmse.mean()),
        "final_rmse": float(rmse[-1]),
        "peak_rmse": float(rmse.max()),
        "alpha_best": float(alpha_best),
        "alpha_continuous": float(alpha_cont),
        "alpha_abs_error": float(abs(alpha_cont - cfg.alpha_true)),
        "alpha_top1_correct": bool(
            abs(
                alpha_best
                - float(
                    np.asarray(cfg.alpha_grid)[
                        int(np.argmin(np.abs(np.asarray(cfg.alpha_grid) - cfg.alpha_true)))
                    ]
                )
            )
            < 1.0e-12
        ),
        "alpha_entropy": None if final_entropy is None else float(final_entropy),
        "runtime_seconds": float(elapsed),
        "peak_memory_mb": float(peak_memory / (1024**2)),
        "rmse": rmse,
        "final_estimate": final_estimate,
    }


def bootstrap_ci(values: np.ndarray, seed: int, n_bootstrap: int = 4000) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(n_bootstrap, values.size))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarize(records: list[dict[str, Any]], base_seed: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for case in ("acoustic2d", "burgers1d", "allen_cahn1d"):
        for method in METHODS:
            subset = [item for item in records if item["case"] == case and item["method"] == method]
            rmse = np.asarray([item["mean_rmse"] for item in subset], dtype=float)
            final = np.asarray([item["final_rmse"] for item in subset], dtype=float)
            alpha_error = np.asarray([item["alpha_abs_error"] for item in subset], dtype=float)
            output.append(
                {
                    "case": case,
                    "method": method,
                    "label": METHOD_LABELS[method],
                    "n_seeds": len(subset),
                    "state_dimension": subset[0]["state_dimension"],
                    "mean_rmse": float(rmse.mean()),
                    "mean_rmse_ci95": bootstrap_ci(rmse, base_seed + len(output)),
                    "final_rmse": float(final.mean()),
                    "alpha_mae": float(alpha_error.mean()),
                    "nearest_track_accuracy_percent": float(
                        100.0 * np.mean([item["alpha_top1_correct"] for item in subset])
                    ),
                    "runtime_seconds": float(np.mean([item["runtime_seconds"] for item in subset])),
                    "peak_memory_mb": float(np.mean([item["peak_memory_mb"] for item in subset])),
                }
            )
    return output


def configure_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def save_publication_figure(figure: plt.Figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def make_summary_figure(summary: list[dict[str, Any]], output_dir: Path) -> None:
    configure_publication_style()
    cases = ["acoustic2d", "burgers1d", "allen_cahn1d"]
    case_labels = ["2D acoustics", "Burgers", "Allen-Cahn"]
    colors = {
        "misspecified": "#9A9A9A",
        "oracle": "#6AAE8B",
        "pce": "#4C78A8",
        "apce": "#E09F3E",
    }
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))
    width = 0.18
    x = np.arange(len(cases))
    for offset, method in enumerate(METHODS):
        values = [next(item for item in summary if item["case"] == case and item["method"] == method)["mean_rmse"] for case in cases]
        axes[0].bar(x + (offset - 1.5) * width, values, width=width, color=colors[method], label=METHOD_LABELS[method])
    axes[0].set_xticks(x, case_labels)
    axes[0].set_ylabel("Time-mean RMSE")
    axes[0].set_title("a  Cross-PDE reconstruction", loc="left", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=18)

    for method in ("pce", "apce"):
        values = [next(item for item in summary if item["case"] == case and item["method"] == method)["alpha_mae"] for case in cases]
        axes[1].plot(x, values, marker="o", color=colors[method], label=METHOD_LABELS[method])
    axes[1].set_xticks(x, case_labels)
    axes[1].set_ylabel(r"Continuous $\alpha$ MAE")
    axes[1].set_title("b  Epistemic identification", loc="left", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=18)

    for method in ("pce", "apce"):
        values = []
        for case in cases:
            item = next(row for row in summary if row["case"] == case and row["method"] == method)
            baseline = next(row for row in summary if row["case"] == case and row["method"] == "misspecified")
            values.append(100.0 * (1.0 - item["mean_rmse"] / baseline["mean_rmse"]))
        axes[2].plot(x, values, marker="s", color=colors[method], label=METHOD_LABELS[method])
    axes[2].axhline(0.0, color="#777777", linewidth=0.8)
    axes[2].set_xticks(x, case_labels)
    axes[2].set_ylabel("RMSE reduction (%)")
    axes[2].set_title("c  Gain over misspecified model", loc="left", fontweight="bold")
    axes[2].tick_params(axis="x", rotation=18)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=4)
    figure.tight_layout(w_pad=1.4)
    save_publication_figure(figure, output_dir / "figure_multiphysics_summary")
    plt.close(figure)


def make_representative_figure(examples: dict[str, dict[str, Any]], output_dir: Path) -> None:
    configure_publication_style()
    figure, axes = plt.subplots(3, 3, figsize=(7.2, 5.6))
    case_order = ["acoustic2d", "burgers1d", "allen_cahn1d"]
    row_labels = ["2D acoustics", "Burgers", "Allen-Cahn"]
    for row, case in enumerate(case_order):
        payload = examples[case]
        cfg = CaseConfig(**payload["config"])
        truth = np.asarray(payload["truth"])
        misspecified = np.asarray(payload["misspecified"])
        pce = np.asarray(payload["pce"])
        if case == "acoustic2d":
            shape = (cfg.ny, cfg.nx)
            arrays = [truth[: cfg.nx * cfg.ny].reshape(shape), misspecified[: cfg.nx * cfg.ny].reshape(shape), pce[: cfg.nx * cfg.ny].reshape(shape)]
            vmax = max(float(np.max(np.abs(item))) for item in arrays)
            for col, array in enumerate(arrays):
                axes[row, col].imshow(array, cmap="RdBu_r", origin="lower", vmin=-vmax, vmax=vmax)
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
        else:
            x = np.linspace(0.0, 1.0, cfg.nx, endpoint=case != "burgers1d")
            arrays = [truth, misspecified, pce]
            for col, array in enumerate(arrays):
                axes[row, col].plot(x, array, color=["#333333", "#9A9A9A", "#4C78A8"][col], linewidth=1.3)
                axes[row, col].set_xlabel("x")
        axes[row, 0].set_ylabel(row_labels[row])
    for col, title in enumerate(["Truth", "Misspecified", "PCE"]):
        axes[0, col].set_title(chr(ord("a") + col) + "  " + title, loc="left", fontweight="bold")
    figure.tight_layout()
    save_publication_figure(figure, output_dir / "figure_multiphysics_fields")
    plt.close(figure)


def run_suite(n_seeds: int, base_seed: int, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    examples: dict[str, dict[str, Any]] = {}
    total = 3 * n_seeds * len(METHODS)
    count = 0
    for case in ("acoustic2d", "burgers1d", "allen_cahn1d"):
        for seed_index in range(n_seeds):
            cfg = config_for_case(case, base_seed + seed_index)
            scenario = generate_scenario(cfg)
            seed_results: dict[str, dict[str, Any]] = {}
            for method in METHODS:
                result = run_method(scenario, method)
                seed_results[method] = result
                record = {
                    "case": case,
                    "seed": cfg.seed,
                    "state_dimension": state_dimension(cfg),
                    **{key: value for key, value in result.items() if key not in {"rmse", "final_estimate"}},
                }
                records.append(record)
                count += 1
                print(f"[{count}/{total}] case={case} seed={cfg.seed} method={method} rmse={result['mean_rmse']:.4e}")
            if seed_index == 0:
                examples[case] = {
                    "config": asdict(cfg),
                    "truth": scenario.truth[-1].tolist(),
                    "misspecified": seed_results["misspecified"]["final_estimate"].tolist(),
                    "pce": seed_results["pce"]["final_estimate"].tolist(),
                }

    fields = sorted({key for record in records for key in record})
    with (output_dir / "multiphysics_runs.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    summary = summarize(records, base_seed)
    payload = {
        "n_seeds": n_seeds,
        "base_seed": base_seed,
        "methods": list(METHODS),
        "cases": ["acoustic2d", "burgers1d", "allen_cahn1d"],
        "summary": summary,
        "representative": examples,
    }
    with (output_dir / "multiphysics_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    make_summary_figure(summary, output_dir)
    make_representative_figure(examples, output_dir)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-PDE PCE/APCE validation suite")
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=20260803)
    parser.add_argument("--output", default="results_paper_multiphysics")
    args = parser.parse_args()
    if args.n_seeds < 3:
        raise ValueError("Use at least three paired seeds.")
    result = run_suite(args.n_seeds, args.base_seed, Path(args.output))
    print(json.dumps({key: value for key, value in result.items() if key != "representative"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
