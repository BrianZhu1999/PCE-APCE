from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy import linalg

from .candidates import ModalCandidateFamily, causal_force_envelope
from .evidence import adaptive_local_grid, joint_gaussian_score, softmax, update_weights
from .identification import IdentifiedModel, psd


@dataclass
class PassResult:
    method: str
    grid: np.ndarray
    final_weights: np.ndarray
    final_log_weights: np.ndarray
    mean_physical: np.ndarray
    branch_mean_physical: np.ndarray
    evaluation_indices: np.ndarray
    support_physical: np.ndarray
    support_weights: np.ndarray
    lower90_physical: np.ndarray
    upper90_physical: np.ndarray
    weight_time: np.ndarray
    weight_history: np.ndarray
    score_history: np.ndarray
    entropy_history: np.ndarray
    temperature_history: np.ndarray
    separation_history: np.ndarray
    blackout_mask: np.ndarray
    paired_noise_digest: str


@dataclass
class TwoPassResult:
    coarse: PassResult
    local: PassResult


def initial_covariance(model: IdentifiedModel) -> np.ndarray:
    try:
        covariance = linalg.solve_discrete_lyapunov(model.a, model.q)
        covariance = psd(covariance, 1e-12)
        if not np.isfinite(covariance).all() or np.max(np.diag(covariance)) > 1e6:
            raise ValueError
        return covariance
    except Exception:
        return np.eye(model.order)


def torch_psd(matrix: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(psd(matrix, 1e-10), dtype=torch.float32, device=device)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> np.ndarray:
    output = np.empty(values.shape[1], dtype=float)
    for dimension in range(values.shape[1]):
        order = np.argsort(values[:, dimension])
        sorted_values = values[order, dimension]
        sorted_weights = weights[order]
        cumulative = np.cumsum(sorted_weights)
        output[dimension] = np.interp(probability, cumulative, sorted_values)
    return output


def denkf_update(
    states: torch.Tensor,
    observation: torch.Tensor,
    c_observed: torch.Tensor,
    d_observed: torch.Tensor,
    input_value: torch.Tensor,
    r_observed: torch.Tensor,
) -> torch.Tensor:
    mean = states.mean(dim=1, keepdim=True)
    anomalies = states - mean
    predicted = torch.einsum("kni,ji->knj", states, c_observed) + d_observed[None, None, :] * input_value
    predicted_mean = predicted.mean(dim=1, keepdim=True)
    predicted_anomalies = predicted - predicted_mean
    denominator = max(states.shape[1] - 1, 1)
    cross = torch.einsum("kni,knj->kij", anomalies, predicted_anomalies) / denominator
    covariance = torch.einsum("kni,knj->kij", predicted_anomalies, predicted_anomalies) / denominator
    covariance = covariance + r_observed[None, :, :]
    gain = torch.linalg.solve(covariance, cross.transpose(1, 2)).transpose(1, 2)
    innovation = observation[None, :] - predicted_mean[:, 0, :]
    updated_mean = mean[:, 0, :] + torch.einsum("kij,kj->ki", gain, innovation)
    updated_anomalies = anomalies - 0.5 * torch.einsum("knj,kij->kni", predicted_anomalies, gain)
    return updated_mean[:, None, :] + updated_anomalies


def candidate_matrices(
    method: str,
    family: ModalCandidateFamily,
    grid: np.ndarray,
    envelope: float,
) -> np.ndarray:
    if method == "DEnKF":
        return family.model.a[None, :, :]
    return np.stack([family.matrix(float(alpha), float(envelope)) for alpha in grid], axis=0)


def run_pass(
    method: str,
    model: IdentifiedModel,
    family: ModalCandidateFamily,
    force_physical: np.ndarray,
    acceleration_physical: np.ndarray,
    observed_indices: list[int],
    grid: np.ndarray,
    config: dict,
    device_name: str,
    initial_noise: np.ndarray,
    forecast_noise: np.ndarray,
    condition: str,
) -> PassResult:
    if method not in ("DEnKF", "BMA", "PCE", "APCE"):
        raise ValueError(method)
    device = torch.device(device_name)
    assimilation = config["assimilation"]
    ensemble_size = initial_noise.shape[1]
    candidate_count = 1 if method == "DEnKF" else len(grid)
    grid = np.asarray([0.5], dtype=float) if method == "DEnKF" else np.asarray(grid, dtype=float)
    u = model.scaled_input(force_physical)
    y = model.scaled_output(acceleration_physical)
    envelope = causal_force_envelope(
        u,
        model.sample_rate_hz,
        float(config["candidates"]["force_envelope_seconds"]),
    )
    observed = np.asarray(observed_indices, dtype=int)
    c_obs = torch.as_tensor(model.c[observed], dtype=torch.float32, device=device)
    d_obs = torch.as_tensor(model.d[observed, 0], dtype=torch.float32, device=device)
    r_obs_np = model.r[np.ix_(observed, observed)]
    r_obs = torch_psd(r_obs_np, device)
    b = torch.as_tensor(model.b[:, 0], dtype=torch.float32, device=device)
    c_all = torch.as_tensor(model.c, dtype=torch.float32, device=device)
    d_all = torch.as_tensor(model.d[:, 0], dtype=torch.float32, device=device)
    q_chol = torch.linalg.cholesky(torch_psd(model.q, device) + 1e-10 * torch.eye(model.order, device=device))
    p0 = initial_covariance(model)
    p0_chol = np.linalg.cholesky(psd(p0, 1e-10) + 1e-10 * np.eye(model.order))
    initial = np.einsum("kni,ji->knj", initial_noise[:candidate_count], p0_chol)
    analysis = torch.as_tensor(initial, dtype=torch.float32, device=device)
    shadow = analysis.clone()
    weights = np.full(candidate_count, 1.0 / candidate_count)
    log_weights = np.zeros(candidate_count)
    interval = int(assimilation["observation_interval_samples"])
    blackout = np.zeros(len(u), dtype=bool)
    if condition == "blackout":
        start, end = assimilation["blackout_fraction"]
        blackout[int(math.floor(start * len(u))):int(math.ceil(end * len(u)))] = True
    elif condition != "standard":
        raise ValueError(condition)
    mean_output = []
    branch_output = []
    evaluation_indices = []
    supports = []
    support_weights = []
    lowers = []
    uppers = []
    weight_time = []
    weight_history = []
    score_history = []
    entropy_history = []
    temperature_history = []
    separation_history = []
    digest = hashlib.sha256(forecast_noise.tobytes()).hexdigest()
    shrinkage = float(assimilation["covariance_shrinkage"])
    for index in range(len(u)):
        matrices_np = candidate_matrices(method, family, grid, float(envelope[index]))
        matrices = torch.as_tensor(matrices_np, dtype=torch.float32, device=device)
        if index > 0:
            noise = torch.as_tensor(forecast_noise[index, :candidate_count], dtype=torch.float32, device=device)
            analysis = torch.einsum("knj,kij->kni", analysis, matrices) + b[None, None, :] * float(u[index])
            analysis = analysis + torch.einsum("knj,ij->kni", noise, q_chol)
            if method in ("PCE", "APCE"):
                shadow = torch.einsum("knj,kij->kni", shadow, matrices) + b[None, None, :] * float(u[index])
                shadow = shadow + torch.einsum("knj,ij->kni", noise, q_chol)
        forecast_for_score = shadow if method in ("PCE", "APCE") else analysis
        if index % interval == 0 and not blackout[index]:
            predicted_score = torch.einsum("kni,ji->knj", forecast_for_score, c_obs) + d_obs[None, None, :] * float(u[index])
            score_samples = predicted_score.detach().cpu().numpy()
            scores = np.asarray([
                joint_gaussian_score(score_samples[candidate], y[index, observed], r_obs_np, shrinkage)
                for candidate in range(candidate_count)
            ])
            if method in ("PCE", "APCE"):
                weights, log_weights, update = update_weights(
                    method.lower(), weights, log_weights, scores, assimilation, 1.0
                )
            elif method == "BMA":
                log_weights = log_weights + 0.5 * (scores - scores.mean())
                weights = softmax(log_weights)
                update = {"temperature": 0.5, "entropy": -float(np.sum(weights * np.log(np.maximum(weights, 1e-300))))}
            else:
                update = {"temperature": 0.0, "entropy": 0.0}
            observation = torch.as_tensor(y[index, observed], dtype=torch.float32, device=device)
            analysis = denkf_update(
                analysis,
                observation,
                c_obs,
                d_obs,
                torch.tensor(float(u[index]), dtype=torch.float32, device=device),
                r_obs,
            )
            candidate_means = score_samples.mean(axis=1)
            between = float(np.mean(np.var(candidate_means, axis=0)))
            noise_scale = max(float(np.mean(np.diag(r_obs_np))), 1e-12)
            separation_history.append(between / noise_scale)
            weight_time.append(index / model.sample_rate_hz)
            weight_history.append(weights.copy())
            score_history.append(scores.copy())
            entropy_history.append(float(update["entropy"]))
            temperature_history.append(float(update["temperature"]))
        y_candidates = torch.einsum("kni,ji->knj", analysis, c_all) + d_all[None, None, :] * float(u[index])
        y_physical = y_candidates.detach().cpu().numpy() * model.output_scale[None, None, :]
        branch_output.append(y_physical.mean(axis=1))
        mixture = np.sum(weights[:, None] * y_physical.mean(axis=1), axis=0)
        mean_output.append(mixture)
        if index % interval == 0:
            flat = y_physical.reshape(candidate_count * ensemble_size, 3)
            flat_weights = np.repeat(weights / ensemble_size, ensemble_size)
            evaluation_indices.append(index)
            supports.append(flat)
            support_weights.append(flat_weights)
            lowers.append(weighted_quantile(flat, flat_weights, 0.05))
            uppers.append(weighted_quantile(flat, flat_weights, 0.95))
    return PassResult(
        method=method,
        grid=grid,
        final_weights=weights,
        final_log_weights=log_weights,
        mean_physical=np.asarray(mean_output),
        branch_mean_physical=np.asarray(branch_output),
        evaluation_indices=np.asarray(evaluation_indices, dtype=int),
        support_physical=np.asarray(supports),
        support_weights=np.asarray(support_weights),
        lower90_physical=np.asarray(lowers),
        upper90_physical=np.asarray(uppers),
        weight_time=np.asarray(weight_time),
        weight_history=np.asarray(weight_history),
        score_history=np.asarray(score_history),
        entropy_history=np.asarray(entropy_history),
        temperature_history=np.asarray(temperature_history),
        separation_history=np.asarray(separation_history),
        blackout_mask=blackout,
        paired_noise_digest=digest,
    )


def run_two_pass(
    method: str,
    model: IdentifiedModel,
    family: ModalCandidateFamily,
    force_physical: np.ndarray,
    acceleration_physical: np.ndarray,
    observed_indices: list[int],
    config: dict,
    device_name: str,
    initial_noise: np.ndarray,
    forecast_noise: np.ndarray,
    condition: str,
) -> TwoPassResult:
    coarse_grid = np.asarray(config["candidates"]["coarse_alpha"], dtype=float)
    coarse = run_pass(
        method, model, family, force_physical, acceleration_physical,
        observed_indices, coarse_grid, config, device_name,
        initial_noise, forecast_noise, condition,
    )
    if method == "DEnKF":
        return TwoPassResult(coarse, coarse)
    local_grid = adaptive_local_grid(
        coarse.grid,
        coarse.final_weights,
        coarse.final_log_weights,
        int(config["candidates"]["local_points"]),
    )
    local = run_pass(
        method, model, family, force_physical, acceleration_physical,
        observed_indices, local_grid, config, device_name,
        initial_noise, forecast_noise, condition,
    )
    return TwoPassResult(coarse, local)
