from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from .evidence import score, update
from .geometry import boundary_mask
from .model import apply_boundary, sponge_mask, step, weighted_field_mean


@dataclass
class Result:
    mean: np.ndarray
    standard_deviation: np.ndarray
    final_weights: np.ndarray
    weight_time: np.ndarray
    weight_history: np.ndarray
    separation_history: np.ndarray
    score_history: np.ndarray
    candidate_mean: np.ndarray | None
    paired_noise_digest: str


def denkf_update(
    previous: torch.Tensor,
    current: torch.Tensor,
    observation: torch.Tensor,
    observed_flat: torch.Tensor,
    variance: float,
    localization: torch.Tensor | None = None,
    inflation: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    candidates, ensembles = current.shape[:2]
    state = torch.cat([previous.reshape(candidates, ensembles, -1), current.reshape(candidates, ensembles, -1)], dim=2)
    field_size = current[0, 0].numel()
    predicted = state[:, :, field_size:].index_select(2, observed_flat)
    state_mean = state.mean(dim=1, keepdim=True)
    state_anomaly = state - state_mean
    predicted_mean = predicted.mean(dim=1, keepdim=True)
    predicted_anomaly = predicted - predicted_mean
    denominator = max(ensembles - 1, 1)
    cross = torch.einsum("knd,kno->kdo", state_anomaly, predicted_anomaly) / denominator
    if localization is not None:
        cross = cross * localization[None]
    covariance = torch.einsum("kno,knp->kop", predicted_anomaly, predicted_anomaly) / denominator
    covariance = covariance + variance * torch.eye(len(observed_flat), device=current.device)[None]
    gain = torch.linalg.solve(covariance, cross.transpose(1, 2)).transpose(1, 2)
    innovation = observation[None] - predicted_mean[:, 0]
    updated_mean = state_mean[:, 0] + torch.einsum("kdo,ko->kd", gain, innovation)
    updated_anomaly = state_anomaly - 0.5 * torch.einsum("kno,kdo->knd", predicted_anomaly, gain)
    updated_anomaly = updated_anomaly * float(inflation)
    updated = updated_mean[:, None] + updated_anomaly
    return updated[:, :, :field_size].reshape_as(previous), updated[:, :, field_size:].reshape_as(current)


def run(
    method: str,
    truth: np.ndarray,
    boundary_series: np.ndarray,
    observed_interior_flat: np.ndarray,
    speed_candidates: np.ndarray,
    config: dict,
    device_name: str,
    initial_noise: np.ndarray,
    noise_seed: int,
    process_noise_std: float,
) -> Result:
    device = torch.device(device_name)
    physics = config["physics"]
    assimilation = config["assimilation"]
    grid = config["grid"]
    dt = 1.0 / float(grid["native_sample_rate_hz"])
    analysis_end = int(round(float(grid["analysis_end_seconds"]) / dt))
    time_count = len(truth)
    boundary_np = np.asarray(boundary_series, dtype=np.float32)
    has_candidate_boundary = boundary_np.ndim == 5
    if has_candidate_boundary:
        all_boundary_count = boundary_np.shape[1]
        nominal_boundary = int(physics.get("nominal_boundary_closure_index", all_boundary_count // 2))
        if method == "DEnKF":
            boundary_np = boundary_np[:, nominal_boundary:nominal_boundary + 1]
            candidate_count = 1
        else:
            candidate_count = all_boundary_count
    else:
        candidate_count = 1 if method == "DEnKF" else len(speed_candidates)
    speeds = np.asarray([float(physics["nominal_speed_m_s"])]) if method == "DEnKF" else np.asarray(speed_candidates, dtype=float)
    if len(speeds) == 1 and candidate_count > 1:
        speeds = np.full(candidate_count, float(speeds[0]), dtype=float)
    speed_t = torch.as_tensor(speeds, dtype=torch.float32, device=device)
    ensembles = initial_noise.shape[1]
    if boundary_np.ndim == 5:
        initial0 = torch.as_tensor(boundary_np[0], dtype=torch.float32, device=device)[:, None].expand(candidate_count, ensembles, -1, -1, -1).clone()
        initial1 = torch.as_tensor(boundary_np[1], dtype=torch.float32, device=device)[:, None].expand(candidate_count, ensembles, -1, -1, -1).clone()
    else:
        initial0 = torch.as_tensor(boundary_np[0], dtype=torch.float32, device=device)[None, None].expand(candidate_count, ensembles, -1, -1, -1).clone()
        initial1 = torch.as_tensor(boundary_np[1], dtype=torch.float32, device=device)[None, None].expand(candidate_count, ensembles, -1, -1, -1).clone()
    if bool(assimilation.get("shared_candidate_noise", True)):
        initial0 += torch.as_tensor(initial_noise[:1, :, 0], dtype=torch.float32, device=device).expand_as(initial0)
        initial1 += torch.as_tensor(initial_noise[:1, :, 1], dtype=torch.float32, device=device).expand_as(initial1)
    else:
        initial0 += torch.as_tensor(initial_noise[:candidate_count, :, 0], dtype=torch.float32, device=device)
        initial1 += torch.as_tensor(initial_noise[:candidate_count, :, 1], dtype=torch.float32, device=device)
    previous_analysis, current_analysis = initial0, initial1
    previous_shadow, current_shadow = initial0.clone(), initial1.clone()
    mask = torch.as_tensor(boundary_mask(), dtype=torch.bool, device=device)
    sponge = sponge_mask(float(physics["boundary_sponge"]), device)
    observed = torch.as_tensor(observed_interior_flat, dtype=torch.long, device=device)
    field_size = int(np.prod(truth.shape[1:]))
    localization_radius = float(assimilation.get("localization_radius_m", 0.0))
    if localization_radius > 0:
        zyx = np.column_stack(np.unravel_index(np.arange(field_size), tuple(truth.shape[1:])))
        zyx_obs = zyx[np.asarray(observed_interior_flat, dtype=int)]
        distances = np.linalg.norm((zyx[:, None] - zyx_obs[None]) * float(grid["spacing_m"]), axis=2)
        taper = np.exp(-0.5 * (distances / localization_radius) ** 2).astype(np.float32)
        localization = torch.as_tensor(np.concatenate([taper, taper], axis=0), device=device)
    else:
        localization = None
    ensemble_inflation = float(assimilation.get("ensemble_inflation", 1.0))
    truth_flat = truth.reshape(len(truth), -1)
    signal_variance = float(np.var(truth_flat[:analysis_end, observed_interior_flat]))
    prearrival = min(int(0.001 * float(grid["native_sample_rate_hz"])), analysis_end)
    observation_variance = max(float(np.var(truth_flat[:prearrival, observed_interior_flat])), 1e-5 * signal_variance, 1e-14)
    weights = np.full(candidate_count, 1.0 / candidate_count)
    log_weights = np.zeros(candidate_count)
    interval = int(assimilation["observation_interval_samples"])
    output = np.zeros_like(truth, dtype=np.float32)
    deviation = np.zeros_like(truth, dtype=np.float32)
    store_candidate_history = bool(assimilation.get("store_candidate_history", False))
    candidate_mean_history = (
        np.zeros((candidate_count, time_count) + tuple(truth.shape[1:]), dtype=np.float32)
        if store_candidate_history else None
    )
    output[0], deviation[0] = weighted_field_mean(previous_analysis, weights)
    output[1], deviation[1] = weighted_field_mean(current_analysis, weights)
    if candidate_mean_history is not None:
        candidate_mean_history[:, 0] = previous_analysis.mean(dim=1).detach().cpu().numpy()
        candidate_mean_history[:, 1] = current_analysis.mean(dim=1).detach().cpu().numpy()
    weight_time, weight_history, separation_history, score_history = [], [], [], []
    noise_rng = np.random.default_rng(noise_seed)
    digest = hashlib.sha256(f"{noise_seed}:{process_noise_std}:{len(speed_candidates)}:{ensembles}".encode()).hexdigest()
    for time_index in range(1, time_count - 1):
        if bool(assimilation.get("shared_candidate_noise", True)):
            noise_base = noise_rng.standard_normal((1, ensembles, 9, 21, 21)).astype(np.float32)
            noise = torch.as_tensor(process_noise_std * noise_base, dtype=torch.float32, device=device).expand(
                candidate_count, -1, -1, -1, -1
            )
        else:
            noise_all = noise_rng.standard_normal((max(candidate_count, len(speed_candidates)), ensembles, 9, 21, 21)).astype(np.float32)
            noise = torch.as_tensor(process_noise_std * noise_all[:candidate_count], dtype=torch.float32, device=device)
        next_analysis = step(previous_analysis, current_analysis, speed_t, float(physics["amplitude_damping_s_inv"]), dt, float(grid["spacing_m"]), noise)
        next_shadow = step(previous_shadow, current_shadow, speed_t, float(physics["amplitude_damping_s_inv"]), dt, float(grid["spacing_m"]), noise)
        if time_index + 1 < analysis_end:
            boundary = torch.as_tensor(boundary_np[time_index + 1], dtype=torch.float32, device=device)
            next_analysis = apply_boundary(next_analysis, boundary, mask)
            next_shadow = apply_boundary(next_shadow, boundary, mask)
        else:
            next_analysis = next_analysis * sponge[None, None]
            next_shadow = next_shadow * sponge[None, None]
        if time_index + 1 < analysis_end and (time_index + 1) % interval == 0:
            score_state = next_shadow if method in ("PCE", "APCE") else next_analysis
            samples = score_state.reshape(candidate_count, ensembles, -1).index_select(2, observed).detach().cpu().numpy()
            scores = np.asarray([
                score(samples[candidate], truth_flat[time_index + 1, observed_interior_flat], observation_variance, float(assimilation["score_covariance_shrinkage"]))
                for candidate in range(candidate_count)
            ])
            if method != "DEnKF":
                weights, log_weights = update(method.lower(), weights, log_weights, scores, assimilation)
            previous_analysis, next_analysis = denkf_update(
                current_analysis, next_analysis,
                torch.as_tensor(truth_flat[time_index + 1, observed_interior_flat], dtype=torch.float32, device=device),
                observed, observation_variance, localization, ensemble_inflation,
            )
            candidate_means = samples.mean(axis=1)
            separation_history.append(float(np.mean(np.var(candidate_means, axis=0)) / observation_variance))
            weight_time.append((time_index + 1) * dt)
            weight_history.append(weights.copy())
            score_history.append(scores.copy())
        else:
            previous_analysis = current_analysis
        current_analysis = next_analysis
        previous_shadow, current_shadow = current_shadow, next_shadow
        output[time_index + 1], deviation[time_index + 1] = weighted_field_mean(current_analysis, weights)
        if candidate_mean_history is not None:
            candidate_mean_history[:, time_index + 1] = current_analysis.mean(dim=1).detach().cpu().numpy()
    return Result(
        output,
        deviation,
        weights,
        np.asarray(weight_time),
        np.asarray(weight_history),
        np.asarray(separation_history),
        np.asarray(score_history),
        candidate_mean_history,
        digest,
    )
