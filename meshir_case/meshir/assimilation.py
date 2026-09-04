from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import torch

from .evidence import gaussian_score, update_weights


@dataclass
class AssimilationResult:
    mean_field: np.ndarray
    lower_field: np.ndarray
    upper_field: np.ndarray
    evaluation_indices: np.ndarray
    final_weights: np.ndarray
    weight_time: np.ndarray
    weight_history: np.ndarray
    score_history: np.ndarray
    separation_history: np.ndarray
    blackout_mask: np.ndarray
    paired_noise_digest: str


def _psd(matrix: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    matrix = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, floor * max(float(np.max(values)), 1.0))
    return (vectors * values) @ vectors.T


def denkf_update(states: torch.Tensor, observation: torch.Tensor, observation_matrix: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
    mean = states.mean(dim=1, keepdim=True)
    anomalies = states - mean
    predicted = torch.einsum("knd,od->kno", states, observation_matrix)
    predicted_mean = predicted.mean(dim=1, keepdim=True)
    predicted_anomalies = predicted - predicted_mean
    denominator = max(states.shape[1] - 1, 1)
    cross = torch.einsum("knd,kno->kdo", anomalies, predicted_anomalies) / denominator
    forecast = torch.einsum("kno,knp->kop", predicted_anomalies, predicted_anomalies) / denominator
    forecast = forecast + covariance[None]
    gain = torch.linalg.solve(forecast, cross.transpose(1, 2)).transpose(1, 2)
    innovation = observation[None] - predicted_mean[:, 0]
    updated_mean = mean[:, 0] + torch.einsum("kdo,ko->kd", gain, innovation)
    updated_anomalies = anomalies - 0.5 * torch.einsum("kno,kdo->knd", predicted_anomalies, gain)
    return updated_mean[:, None] + updated_anomalies


def run_assimilation(
    method: str,
    truth_field: np.ndarray,
    candidate_paths: np.ndarray,
    fixed_path: np.ndarray,
    basis_full: np.ndarray,
    observed_indices: np.ndarray,
    rho: float,
    state_covariance: np.ndarray,
    process_covariance: np.ndarray,
    observation_covariance: np.ndarray,
    config: dict,
    analysis_end: int,
    prediction_end: int,
    condition: str,
    device_name: str,
    initial_noise: np.ndarray,
    forecast_noise: np.ndarray,
    sample_rate_hz: float,
    candidate_toa: np.ndarray | None = None,
    observed_peak_times: np.ndarray | None = None,
    toa_sigma_seconds: float = 0.0005,
) -> AssimilationResult:
    if method not in ("DEnKF", "BMA", "PCE", "APCE"):
        raise ValueError(method)
    device = torch.device(device_name)
    candidate_paths = np.asarray(candidate_paths, dtype=np.float32)
    fixed_path = np.asarray(fixed_path, dtype=np.float32)
    truth_field = np.asarray(truth_field, dtype=np.float32)
    basis_full = np.asarray(basis_full, dtype=np.float32)
    candidates = 1 if method == "DEnKF" else len(candidate_paths)
    paths = fixed_path[None] if method == "DEnKF" else candidate_paths
    time_count, state_dim = paths.shape[1], paths.shape[2]
    field_dim = basis_full.shape[0]
    ensemble_size = initial_noise.shape[1]
    observed_indices = np.asarray(observed_indices, dtype=int)
    observation_matrix = torch.as_tensor(basis_full[observed_indices], dtype=torch.float32, device=device)
    if observation_covariance.shape[0] == basis_full.shape[0]:
        observation_covariance = _psd(observation_covariance[np.ix_(observed_indices, observed_indices)])
    else:
        observation_scale = max(float(np.mean(np.diag(observation_covariance))), 1e-8)
        observation_covariance = observation_scale * np.eye(len(observed_indices))
    observation_covariance_t = torch.as_tensor(observation_covariance, dtype=torch.float32, device=device)
    state_chol = torch.as_tensor(np.linalg.cholesky(_psd(state_covariance)), dtype=torch.float32, device=device)
    process_chol = torch.as_tensor(np.linalg.cholesky(_psd(process_covariance)), dtype=torch.float32, device=device)
    path_t = torch.as_tensor(paths, dtype=torch.float32, device=device)
    initial = path_t[:, 0, None, :] + torch.einsum("knd,ed->kne", torch.as_tensor(initial_noise[:candidates], dtype=torch.float32, device=device), state_chol)
    analysis = initial
    shadow = initial.clone()
    weights = np.full(candidates, 1.0 / candidates)
    log_weights = np.zeros(candidates)
    interval = int(config["observation_interval_samples"])
    blackout = np.zeros(time_count, dtype=bool)
    if condition == "blackout":
        lo, hi = config["blackout_fraction_of_analysis"]
        blackout[int(lo * analysis_end):int(hi * analysis_end)] = True
    output = np.zeros((time_count, field_dim), dtype=np.float32)
    lower = np.zeros_like(output)
    upper = np.zeros_like(output)
    evaluation_indices = []
    weight_time, weight_history, score_history, separation_history = [], [], [], []
    digest = hashlib.sha256(forecast_noise.tobytes()).hexdigest()
    for index in range(time_count):
        if index > 0:
            noise = torch.as_tensor(forecast_noise[index, :candidates], dtype=torch.float32, device=device)
            previous = path_t[:, index - 1, None]
            current = path_t[:, index, None]
            analysis = current + float(rho) * (analysis - previous) + torch.einsum("knd,ed->kne", noise, process_chol)
            if method in ("PCE", "APCE"):
                shadow = current + float(rho) * (shadow - previous) + torch.einsum("knd,ed->kne", noise, process_chol)
        can_update = index < analysis_end and index % interval == 0 and not blackout[index]
        if can_update:
            score_state = shadow if method in ("PCE", "APCE") else analysis
            predicted = torch.einsum("knd,od->kno", score_state, observation_matrix).detach().cpu().numpy()
            scores = np.asarray([
                gaussian_score(predicted[k], truth_field[index, observed_indices], observation_covariance, float(config["score_covariance_shrinkage"]))
                for k in range(candidates)
            ])
            if candidate_toa is not None and observed_peak_times is not None:
                toa_scores = -0.5 * np.mean(((np.asarray(observed_peak_times)[None, :] - np.asarray(candidate_toa)) / max(toa_sigma_seconds, 1e-8)) ** 2, axis=1)
                scores = toa_scores
            if method != "DEnKF":
                weights, log_weights, _ = update_weights(method.lower(), weights, log_weights, scores, config)
            analysis = denkf_update(analysis, torch.as_tensor(truth_field[index, observed_indices], dtype=torch.float32, device=device), observation_matrix, observation_covariance_t)
            candidate_observations = predicted.mean(axis=1)
            separation_history.append(float(np.mean(np.var(candidate_observations, axis=0)) / max(float(np.mean(np.diag(observation_covariance))), 1e-12)))
            weight_time.append(index / sample_rate_hz)
            weight_history.append(weights.copy())
            score_history.append(scores.copy())
        state_mean = analysis.mean(dim=1).detach().cpu().numpy()
        state_variance = analysis.var(dim=1, unbiased=True).detach().cpu().numpy()
        field_candidates = np.einsum("kd,fd->kf", state_mean, basis_full)
        mixture = np.sum(weights[:, None] * field_candidates, axis=0)
        within = np.sum(weights[:, None] * np.einsum("kd,fd->kf", state_variance, basis_full ** 2), axis=0)
        between = np.sum(weights[:, None] * (field_candidates - mixture[None]) ** 2, axis=0)
        deviation = np.sqrt(np.maximum(within + between, 1e-20))
        output[index] = mixture
        if index % interval == 0:
            evaluation_indices.append(index)
            lower[index] = mixture - 1.6448536269514722 * deviation
            upper[index] = mixture + 1.6448536269514722 * deviation
    return AssimilationResult(
        mean_field=output,
        lower_field=lower,
        upper_field=upper,
        evaluation_indices=np.asarray(evaluation_indices, dtype=int),
        final_weights=weights,
        weight_time=np.asarray(weight_time),
        weight_history=np.asarray(weight_history),
        score_history=np.asarray(score_history),
        separation_history=np.asarray(separation_history),
        blackout_mask=blackout,
        paired_noise_digest=digest,
    )
