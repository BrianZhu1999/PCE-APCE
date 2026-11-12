from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Config:
    seed: int = 20260803

    # Dimensionless wave equation
    nx: int = 41
    length: float = 1.0
    wave_speed: float = 1.0
    damping: float = 0.06
    dt: float = 0.0025
    t_end: float = 1.0

    # Nested ensemble
    ensemble_size: int = 18
    n_alpha: int = 7
    alpha_min: float = 0.08
    alpha_max: float = 0.92
    alpha_true: float = 0.12

    # Uncertainty strength
    epistemic_scale: float = 1.0
    process_noise: float = 0.010

    # Sparse observations
    obs_noise: float = 0.020
    n_sensors: int = 6
    obs_interval: int = 20

    # VE-EnSF reverse diffusion
    reverse_steps: int = 8
    sigma_min: float = 0.015
    sigma_max: float = 0.70
    guidance: float = 0.80
    reverse_noise_scale: float = 0.55
    score_clip: float = 40.0

    # Outer alpha-path evidence update
    credibility_rate: float = 0.55

    # Normalization floors
    u_scale_floor: float = 0.035
    v_scale_floor: float = 0.10

    # Analysis update: "lr" implements the two-step observed-state EnSF
    # plus ensemble linear regression to unobserved variables.
    filter_variant: str = "lr"
    regression_ridge: float = 1.0e-5


def liu_normal_inverse(alpha: np.ndarray | float) -> np.ndarray:
    """Inverse uncertainty distribution of a standard normal uncertain variable."""
    a = np.clip(np.asarray(alpha, dtype=float), 1.0e-6, 1.0 - 1.0e-6)
    return (math.sqrt(3.0) / math.pi) * np.log(a / (1.0 - a))


def smooth_noise(z: np.ndarray) -> np.ndarray:
    z = z.copy()
    if z.shape[-1] >= 3:
        z[..., 1:-1] = (
            0.25 * z[..., :-2]
            + 0.50 * z[..., 1:-1]
            + 0.25 * z[..., 2:]
        )
    z[..., 0] = 0.0
    z[..., -1] = 0.0
    return z


def source_terms(x: np.ndarray, t: float, theta: float) -> np.ndarray:
    """Known source plus an epistemically uncertain secondary source."""
    base_profile = np.exp(-((x - 0.22) / 0.055) ** 2)
    epistemic_profile = np.exp(-((x - 0.68) / 0.085) ** 2)
    base_time = 1.20 * np.sin(2.0 * math.pi * 2.4 * t) * np.exp(-1.4 * t)
    epistemic_time = (
        np.sin(2.0 * math.pi * 1.25 * t + 0.35) * np.exp(-0.35 * t)
    )
    return base_time * base_profile + theta * epistemic_time * epistemic_profile


def initial_state(x: np.ndarray) -> np.ndarray:
    u0 = 0.55 * np.exp(-((x - 0.34) / 0.075) ** 2)
    v0 = np.zeros_like(x)
    u0[[0, -1]] = 0.0
    return np.concatenate([u0, v0])


def propagate_batch(
    states: np.ndarray,
    theta: float,
    t: float,
    cfg: Config,
    rng: np.random.Generator,
    stochastic: bool = True,
    noise_draw: np.ndarray | None = None,
) -> np.ndarray:
    nx = cfg.nx
    dx = cfg.length / (nx - 1)
    cfl = cfg.wave_speed * cfg.dt / dx
    if cfl > 0.95:
        raise ValueError(f"CFL={cfl:.3f} is too large.")

    u = states[:, :nx]
    v = states[:, nx:]
    lap = np.zeros_like(u)
    lap[:, 1:-1] = (u[:, 2:] - 2.0 * u[:, 1:-1] + u[:, :-2]) / (dx * dx)

    x = np.linspace(0.0, cfg.length, nx)
    forcing = source_terms(x, t, theta)[None, :]
    acceleration = cfg.wave_speed**2 * lap - cfg.damping * v + forcing

    if stochastic and cfg.process_noise > 0.0:
        d_w = smooth_noise(rng.normal(size=u.shape)) if noise_draw is None else noise_draw
        v_new = (
            v
            + cfg.dt * acceleration
            + cfg.process_noise * math.sqrt(cfg.dt) * d_w
        )
    else:
        v_new = v + cfg.dt * acceleration

    u_new = u + cfg.dt * v_new
    u_new[:, [0, -1]] = 0.0
    v_new[:, [0, -1]] = 0.0
    return np.concatenate([u_new, v_new], axis=1)


def logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(
        np.sum(np.exp(values - maximum), axis=axis, keepdims=True)
    )
    return np.squeeze(result, axis=axis)


def mixture_score_and_denoised(
    z: np.ndarray,
    clean_ensemble: np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    difference = z[:, None, :] - clean_ensemble[None, :, :]
    log_weights = -0.5 * np.sum(difference * difference, axis=2) / (sigma * sigma)
    log_weights -= logsumexp(log_weights, axis=1)[:, None]
    weights = np.exp(log_weights)
    denoised = weights @ clean_ensemble
    score = (denoised - z) / (sigma * sigma)
    return score, denoised


def _ensf_identity_observation_update(
    prior_observed: np.ndarray,
    observation: np.ndarray,
    obs_noise: float | np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
) -> np.ndarray:
    """EnSF analysis in the observed subspace with identity observation operator."""
    mean = prior_observed.mean(axis=0)
    scale = prior_observed.std(axis=0, ddof=1)
    scale = np.maximum(scale, cfg.u_scale_floor)
    clean_ensemble = (prior_observed - mean) / scale

    noise = np.broadcast_to(np.asarray(obs_noise, dtype=float), observation.shape)
    sigma_ratio = cfg.sigma_max / cfg.sigma_min
    log_sigma_ratio = math.log(sigma_ratio)
    z = clean_ensemble + cfg.sigma_max * rng.normal(size=clean_ensemble.shape)
    pseudo_times = np.linspace(1.0, 0.0, cfg.reverse_steps + 1)

    for k in range(cfg.reverse_steps):
        tau_now = max(pseudo_times[k], 1.0e-4)
        tau_next = max(pseudo_times[k + 1], 0.0)
        delta_tau = tau_next - tau_now
        sigma = cfg.sigma_min * sigma_ratio**tau_now
        diffusion = sigma * math.sqrt(2.0 * log_sigma_ratio)

        prior_score, clean_estimate = mixture_score_and_denoised(
            z, clean_ensemble, sigma
        )
        predicted_observation = mean[None, :] + scale[None, :] * clean_estimate
        residual = observation[None, :] - predicted_observation
        likelihood_score = scale[None, :] * residual / (noise[None, :] ** 2)

        posterior_score = (
            prior_score
            + cfg.guidance * (1.0 - tau_now) * likelihood_score
        )
        posterior_score = np.clip(
            posterior_score, -cfg.score_clip, cfg.score_clip
        )

        z = (
            z
            + (-diffusion * diffusion * posterior_score) * delta_tau
            + cfg.reverse_noise_scale
            * diffusion
            * math.sqrt(-delta_tau)
            * rng.normal(size=z.shape)
        )

    return mean + scale * z


def ensf_update_direct(
    prior: np.ndarray,
    observation: np.ndarray,
    observation_indices: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply the direct observation-space ensemble score update."""
    mean = prior.mean(axis=0)
    scale = prior.std(axis=0, ddof=1)
    scale[: cfg.nx] = np.maximum(scale[: cfg.nx], cfg.u_scale_floor)
    scale[cfg.nx :] = np.maximum(scale[cfg.nx :], cfg.v_scale_floor)
    clean_ensemble = (prior - mean) / scale

    sigma_ratio = cfg.sigma_max / cfg.sigma_min
    log_sigma_ratio = math.log(sigma_ratio)
    z = clean_ensemble + cfg.sigma_max * rng.normal(size=clean_ensemble.shape)
    pseudo_times = np.linspace(1.0, 0.0, cfg.reverse_steps + 1)

    for k in range(cfg.reverse_steps):
        tau_now = max(pseudo_times[k], 1.0e-4)
        tau_next = max(pseudo_times[k + 1], 0.0)
        delta_tau = tau_next - tau_now
        sigma = cfg.sigma_min * sigma_ratio**tau_now
        diffusion = sigma * math.sqrt(2.0 * log_sigma_ratio)

        prior_score, clean_estimate = mixture_score_and_denoised(
            z, clean_ensemble, sigma
        )
        predicted = (
            mean[observation_indices][None, :]
            + scale[observation_indices][None, :]
            * clean_estimate[:, observation_indices]
        )
        likelihood_score = np.zeros_like(z)
        likelihood_score[:, observation_indices] = (
            scale[observation_indices][None, :]
            * (observation[None, :] - predicted)
            / (cfg.obs_noise**2)
        )
        posterior_score = (
            prior_score
            + cfg.guidance * (1.0 - tau_now) * likelihood_score
        )
        posterior_score = np.clip(
            posterior_score, -cfg.score_clip, cfg.score_clip
        )
        z = (
            z
            + (-diffusion * diffusion * posterior_score) * delta_tau
            + cfg.reverse_noise_scale
            * diffusion
            * math.sqrt(-delta_tau)
            * rng.normal(size=z.shape)
        )

    updated = mean + scale * z
    updated[:, [0, cfg.nx - 1, cfg.nx, 2 * cfg.nx - 1]] = 0.0
    return updated


def ensf_update_lr(
    prior: np.ndarray,
    observation: np.ndarray,
    observation_indices: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
    obs_noise: float | np.ndarray | None = None,
) -> np.ndarray:
    """Two-step EnSF-LR-style update for sparse observations.

    Step 1: EnSF updates only the observed components.
    Step 2: prior ensemble cross-covariance maps observed increments to
            unobserved components by linear regression.
    """
    observation_indices = np.asarray(observation_indices, dtype=int)
    updated_observed = _ensf_identity_observation_update(
        prior[:, observation_indices],
        observation,
        cfg.obs_noise if obs_noise is None else obs_noise,
        cfg,
        rng,
    )
    observed_increment = updated_observed - prior[:, observation_indices]

    all_indices = np.arange(prior.shape[1])
    unobserved_indices = np.setdiff1d(all_indices, observation_indices)
    updated = prior.copy()
    updated[:, observation_indices] = updated_observed

    if unobserved_indices.size:
        anomalies_observed = prior[:, observation_indices] - prior[:, observation_indices].mean(axis=0)
        anomalies_unobserved = prior[:, unobserved_indices] - prior[:, unobserved_indices].mean(axis=0)
        denom = max(prior.shape[0] - 1, 1)
        covariance_oo = anomalies_observed.T @ anomalies_observed / denom
        covariance_uo = anomalies_unobserved.T @ anomalies_observed / denom
        ridge = cfg.regression_ridge * (
            np.trace(covariance_oo) / max(covariance_oo.shape[0], 1) + 1.0
        )
        regression = covariance_uo @ np.linalg.pinv(
            covariance_oo + ridge * np.eye(covariance_oo.shape[0])
        )
        updated[:, unobserved_indices] = (
            prior[:, unobserved_indices]
            + observed_increment @ regression.T
        )

    updated[:, [0, cfg.nx - 1, cfg.nx, 2 * cfg.nx - 1]] = 0.0
    return updated


def branch_compatibility(
    prior_observations: np.ndarray,
    observation: np.ndarray,
    observation_noise: float | np.ndarray,
) -> float:
    mean = prior_observations.mean(axis=0)
    covariance = (
        np.cov(prior_observations, rowvar=False)
        if prior_observations.shape[0] > 1
        else np.zeros((observation.size, observation.size))
    )
    noise = np.broadcast_to(np.asarray(observation_noise, dtype=float), observation.shape)
    covariance = np.atleast_2d(covariance) + np.diag(noise**2 + 1.0e-7)
    innovation = observation - mean
    mahalanobis = float(innovation @ np.linalg.solve(covariance, innovation))
    return float(np.exp(-0.5 * mahalanobis / max(observation.size, 1)))


def softmax(log_weights: np.ndarray) -> np.ndarray:
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)
    return weights / np.sum(weights)


def weighted_quantile_columns(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> np.ndarray:
    output = np.empty(values.shape[1], dtype=float)
    for j in range(values.shape[1]):
        order = np.argsort(values[:, j])
        sorted_values = values[order, j]
        sorted_weights = weights[order]
        cumulative = np.cumsum(sorted_weights)
        output[j] = np.interp(quantile, cumulative, sorted_values)
    return output


def make_config() -> Config:
    """Return the wave configuration used in the paper."""
    return Config()


def run(
    cfg: Config,
    output_dir: Path | None = None,
    save_data: bool = True,
) -> dict[str, Any]:
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)
    x = np.linspace(0.0, cfg.length, cfg.nx)
    number_of_steps = int(round(cfg.t_end / cfg.dt))
    times = np.arange(number_of_steps + 1) * cfg.dt
    alpha_grid = np.linspace(cfg.alpha_min, cfg.alpha_max, cfg.n_alpha)
    theta_grid = cfg.epistemic_scale * liu_normal_inverse(alpha_grid)
    theta_true = float(cfg.epistemic_scale * liu_normal_inverse(cfg.alpha_true))

    sensor_indices = np.linspace(4, cfg.nx - 5, cfg.n_sensors, dtype=int)
    observation_indices = sensor_indices.copy()
    x0 = initial_state(x)
    truth = x0[None, :].copy()
    deterministic_baseline = x0[None, :].copy()

    branches = np.empty((cfg.n_alpha, cfg.ensemble_size, 2 * cfg.nx), dtype=float)
    for q in range(cfg.n_alpha):
        branches[q] = x0[None, :]
        branches[q, :, : cfg.nx] += 0.012 * smooth_noise(rng.normal(size=(cfg.ensemble_size, cfg.nx)))
        branches[q, :, cfg.nx :] += 0.025 * smooth_noise(rng.normal(size=(cfg.ensemble_size, cfg.nx)))
        branches[q, :, [0, cfg.nx - 1, cfg.nx, 2 * cfg.nx - 1]] = 0.0

    log_credibility = np.zeros(cfg.n_alpha, dtype=float)
    trajectory_weights = softmax(log_credibility)

    rmse_hybrid = np.zeros(number_of_steps + 1)
    rmse_baseline = np.zeros(number_of_steps + 1)
    aleatoric_history = np.zeros(number_of_steps + 1)
    epistemic_history = np.zeros(number_of_steps + 1)
    truth_field = np.zeros((number_of_steps + 1, cfg.nx))
    estimate_field = np.zeros((number_of_steps + 1, cfg.nx))
    baseline_field = np.zeros((number_of_steps + 1, cfg.nx))
    truth_velocity_field = np.zeros((number_of_steps + 1, cfg.nx))
    estimate_velocity_field = np.zeros((number_of_steps + 1, cfg.nx))
    baseline_velocity_field = np.zeros((number_of_steps + 1, cfg.nx))
    credibility_times: list[float] = []
    credibility_history: list[np.ndarray] = []

    def summarize(step: int) -> None:
        branch_means = branches.mean(axis=1)
        estimate = np.sum(trajectory_weights[:, None] * branch_means, axis=0)
        true_u = truth[0, : cfg.nx]
        true_v = truth[0, cfg.nx :]
        estimated_u = estimate[: cfg.nx]
        estimated_v = estimate[cfg.nx :]
        baseline_u = deterministic_baseline[0, : cfg.nx]
        baseline_v = deterministic_baseline[0, cfg.nx :]

        rmse_hybrid[step] = float(np.sqrt(np.mean((estimated_u - true_u) ** 2)))
        rmse_baseline[step] = float(np.sqrt(np.mean((baseline_u - true_u) ** 2)))
        truth_field[step] = true_u
        estimate_field[step] = estimated_u
        baseline_field[step] = baseline_u
        truth_velocity_field[step] = true_v
        estimate_velocity_field[step] = estimated_v
        baseline_velocity_field[step] = baseline_v

        within_path_variance = np.mean(
            np.var(branches[:, :, : cfg.nx], axis=1, ddof=1), axis=1
        )
        aleatoric_history[step] = float(np.sum(trajectory_weights * within_path_variance))
        branch_u_means = branch_means[:, : cfg.nx]
        weighted_u_mean = np.sum(trajectory_weights[:, None] * branch_u_means, axis=0)
        epistemic_history[step] = float(
            np.mean(
                np.sum(
                    trajectory_weights[:, None]
                    * (branch_u_means - weighted_u_mean) ** 2,
                    axis=0,
                )
            )
        )

    summarize(0)
    start_time = time.perf_counter()

    for step in range(1, number_of_steps + 1):
        current_time = times[step - 1]
        truth = propagate_batch(truth, theta_true, current_time, cfg, rng, stochastic=True)
        deterministic_baseline = propagate_batch(deterministic_baseline, 0.0, current_time, cfg, rng, stochastic=False)

        common_noise = smooth_noise(rng.normal(size=(cfg.ensemble_size, cfg.nx)))
        for q, theta in enumerate(theta_grid):
            branches[q] = propagate_batch(
                branches[q], float(theta), current_time, cfg, rng,
                stochastic=True, noise_draw=common_noise,
            )

        if step % cfg.obs_interval == 0:
            observation = truth[0, observation_indices] + cfg.obs_noise * rng.normal(size=observation_indices.size)
            compatibility = np.empty(cfg.n_alpha)
            for q in range(cfg.n_alpha):
                compatibility[q] = branch_compatibility(
                    branches[q][:, observation_indices], observation, cfg.obs_noise
                )
                if cfg.filter_variant.lower() == "direct":
                    branches[q] = ensf_update_direct(
                        branches[q], observation, observation_indices, cfg, rng
                    )
                else:
                    branches[q] = ensf_update_lr(
                        branches[q], observation, observation_indices, cfg, rng
                    )

            log_credibility += cfg.credibility_rate * np.log(np.clip(compatibility, 1.0e-8, 1.0))
            log_credibility -= np.max(log_credibility)
            trajectory_weights = softmax(log_credibility)
            credibility_times.append(times[step])
            credibility_history.append(trajectory_weights.copy())

        summarize(step)

    elapsed_seconds = time.perf_counter() - start_time
    branch_means = branches.mean(axis=1)
    final_estimate = np.sum(trajectory_weights[:, None] * branch_means, axis=0)
    final_samples = branches[:, :, : cfg.nx].reshape(-1, cfg.nx)
    sample_weights = np.repeat(trajectory_weights / cfg.ensemble_size, cfg.ensemble_size)
    sample_weights /= sample_weights.sum()
    lower_band = weighted_quantile_columns(final_samples, sample_weights, 0.05)
    upper_band = weighted_quantile_columns(final_samples, sample_weights, 0.95)
    final_truth_u = truth[0, : cfg.nx]
    coverage_90 = float(np.mean((final_truth_u >= lower_band) & (final_truth_u <= upper_band)))
    alpha_best = float(alpha_grid[int(np.argmax(trajectory_weights))])

    metrics = {
        "elapsed_seconds": elapsed_seconds,
        "final_rmse_hybrid": float(rmse_hybrid[-1]),
        "final_rmse_baseline": float(rmse_baseline[-1]),
        "mean_rmse_hybrid": float(np.mean(rmse_hybrid)),
        "mean_rmse_baseline": float(np.mean(rmse_baseline)),
        "relative_rmse_reduction_percent": float(
            100.0 * (1.0 - np.mean(rmse_hybrid) / max(np.mean(rmse_baseline), 1.0e-12))
        ),
        "coverage_90_final": coverage_90,
        "alpha_true": cfg.alpha_true,
        "alpha_best_final": alpha_best,
        "alpha_top1_correct": bool(abs(alpha_best - cfg.alpha_true) < 1.0e-12),
        "theta_true": theta_true,
        "theta_best_final": float(theta_grid[int(np.argmax(trajectory_weights))]),
        "filter_variant": cfg.filter_variant,
    }

    result: dict[str, Any] = {
        "config": cfg,
        "metrics": metrics,
        "x": x,
        "times": times,
        "alpha_grid": alpha_grid,
        "theta_grid": theta_grid,
        "final_weights": trajectory_weights,
        "truth_field": truth_field,
        "estimate_field": estimate_field,
        "baseline_field": baseline_field,
        "truth_velocity_field": truth_velocity_field,
        "estimate_velocity_field": estimate_velocity_field,
        "baseline_velocity_field": baseline_velocity_field,
        "rmse_hybrid": rmse_hybrid,
        "rmse_baseline": rmse_baseline,
        "aleatoric": aleatoric_history,
        "epistemic": epistemic_history,
        "credibility_times": np.asarray(credibility_times),
        "credibility_history": np.asarray(credibility_history),
        "sensor_indices": sensor_indices,
        "lower_band": lower_band,
        "upper_band": upper_band,
    }

    if output_dir is not None:
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
            json.dump({"config": asdict(cfg), "metrics": metrics}, file, ensure_ascii=False, indent=2)

        if save_data:
            np.savez_compressed(
                output_dir / "simulation_data.npz",
                x=x,
                times=times,
                alpha_grid=alpha_grid,
                theta_grid=theta_grid,
                final_weights=trajectory_weights,
                truth_field=truth_field,
                estimate_field=estimate_field,
                baseline_field=baseline_field,
                truth_velocity_field=truth_velocity_field,
                estimate_velocity_field=estimate_velocity_field,
                baseline_velocity_field=baseline_velocity_field,
                rmse_hybrid=rmse_hybrid,
                rmse_baseline=rmse_baseline,
                aleatoric=aleatoric_history,
                epistemic=epistemic_history,
                credibility_times=np.asarray(credibility_times),
                credibility_history=np.asarray(credibility_history),
                sensor_indices=sensor_indices,
            )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nested alpha-path and EnSF wave assimilation")
    parser.add_argument("--output", default="results/wave_example")
    parser.add_argument("--filter", choices=["lr", "direct"], default="lr")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = replace(make_config(), filter_variant=args.filter)
    result = run(config, Path(args.output), save_data=True)
    print("\nSimulation finished. Metrics:")
    for key, value in result["metrics"].items():
        print(f"  {key}: {value}")
    print(f"\nOutput: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
