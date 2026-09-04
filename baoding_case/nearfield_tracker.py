#!/usr/bin/env python3
"""Near-field 3-D helicopter tracking after the historical-DOA admission gate."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path

import torch

try:
    from . import ALPHA_GRID
    from .nearfield_audit import (
        apply_calibration,
        calibrate,
        fused_plane1_gps,
        parse_historical_doa,
        robust_triangulate,
        truth_at,
    )
    from .run_baoding import angle_residual, parse_nod, sha256, signed_deg
except ImportError:
    from __init__ import ALPHA_GRID
    from nearfield_audit import (
        apply_calibration,
        calibrate,
        fused_plane1_gps,
        parse_historical_doa,
        robust_triangulate,
        truth_at,
    )
    from run_baoding import angle_residual, parse_nod, sha256, signed_deg


METHODS = ("denkf", "aug_enkf", "bma", "pce", "apce")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_frontend_observations(path: Path) -> dict[str, dict[int, dict[int, tuple[float, float, int]]]]:
    output: dict[str, dict[int, dict[int, tuple[float, float, int]]]] = {}
    with path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            segment = row["segment"]
            second = int(float(row["time_s"]))
            node = int(row["node_id"])
            output.setdefault(segment, {}).setdefault(second, {})[node] = (
                float(row["azimuth_deg"]), float(row["elevation_deg"]), 1,
            )
    return output


def circular_mean(values: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.atan2(torch.sin(values).mean(dim=dim), torch.cos(values).mean(dim=dim))


def predict_angles(states: torch.Tensor, nodes: dict[int, torch.Tensor], node_ids: list[int]) -> torch.Tensor:
    position = states[..., :3]
    output = []
    for node in node_ids:
        delta = position - nodes[node].to(states)
        output.append(torch.atan2(delta[..., 1], delta[..., 0]))
        output.append(torch.atan2(delta[..., 2], torch.linalg.vector_norm(delta[..., :2], dim=-1).clamp_min(1e-6)))
    return torch.stack(output, dim=-1)


def observation_tensor(frame: dict[int, tuple[float, float, float]], node_ids: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    values, variance = [], []
    for node in node_ids:
        azimuth, elevation, sigma = frame[node]
        values.extend((math.radians(azimuth), math.radians(elevation)))
        variance.extend((math.radians(sigma) ** 2, math.radians(max(2.0, 0.65 * sigma)) ** 2))
    return (
        torch.tensor(values, dtype=torch.float64, device=device),
        torch.tensor(variance, dtype=torch.float64, device=device),
    )


def denkf_update(states: torch.Tensor, observation: torch.Tensor, observation_variance: torch.Tensor, nodes: dict[int, torch.Tensor], node_ids: list[int]) -> torch.Tensor:
    forecast_observation = predict_angles(states, nodes, node_ids)
    state_mean = states.mean(dim=-2)
    observation_mean = circular_mean(forecast_observation, dim=-2)
    state_anomaly = states - state_mean.unsqueeze(-2)
    observation_anomaly = angle_residual(forecast_observation, observation_mean.unsqueeze(-2))
    denominator = max(states.shape[-2] - 1, 1)
    cross_covariance = state_anomaly.transpose(-2, -1) @ observation_anomaly / denominator
    innovation_covariance = observation_anomaly.transpose(-2, -1) @ observation_anomaly / denominator
    innovation_covariance = innovation_covariance + torch.diag_embed(observation_variance)
    gain = torch.linalg.solve(innovation_covariance.transpose(-2, -1), cross_covariance.transpose(-2, -1)).transpose(-2, -1)
    mean_increment = torch.einsum("...ij,...j->...i", gain, angle_residual(observation, observation_mean))
    anomaly_increment = 0.5 * torch.einsum("...nj,...ij->...ni", observation_anomaly, gain)
    return state_mean.unsqueeze(-2) + mean_increment.unsqueeze(-2) + state_anomaly - anomaly_increment


def anchor_denkf_update(states: torch.Tensor, anchor: torch.Tensor, covariance_diag: torch.Tensor) -> torch.Tensor:
    """Deterministic EnKF update for an acoustic 3-D RANSAC anchor."""
    predicted = states[:, :3]
    state_mean, anchor_mean = states.mean(dim=0), predicted.mean(dim=0)
    state_anomaly, anchor_anomaly = states - state_mean, predicted - anchor_mean
    denominator = max(states.shape[0] - 1, 1)
    cross = state_anomaly.T @ anchor_anomaly / denominator
    covariance = anchor_anomaly.T @ anchor_anomaly / denominator + torch.diag(covariance_diag)
    gain = torch.linalg.solve(covariance.T, cross.T).T
    mean_increment = gain @ (anchor - anchor_mean)
    anomaly_increment = 0.5 * anchor_anomaly @ gain.T
    return state_mean + mean_increment + state_anomaly - anomaly_increment


def process_scale(alpha: torch.Tensor, q_min: float, q_max: float) -> torch.Tensor:
    ratio = torch.tensor(q_max / q_min, dtype=torch.float64, device=alpha.device)
    return q_min * torch.pow(ratio, alpha)


def propagate(states: torch.Tensor, alpha: torch.Tensor, dt: float, standard_noise: torch.Tensor, q_min: float, q_max: float, max_range_m: float, max_speed_mps: float) -> torch.Tensor:
    acceleration = standard_noise * process_scale(alpha, q_min, q_max).unsqueeze(-1)
    output = states.clone()
    output[..., :3] = output[..., :3] + output[..., 3:6] * dt + 0.5 * acceleration * dt * dt
    output[..., 3:6] = output[..., 3:6] + acceleration * dt
    return physical_project(output, max_range_m, max_speed_mps)


def physical_project(states: torch.Tensor, max_range_m: float = 30_000.0, max_speed_mps: float = 150.0) -> torch.Tensor:
    output = states.clone()
    position_norm = torch.linalg.vector_norm(output[..., :3], dim=-1, keepdim=True).clamp_min(1.0)
    velocity_norm = torch.linalg.vector_norm(output[..., 3:6], dim=-1, keepdim=True).clamp_min(1.0)
    output[..., :3] = output[..., :3] * torch.clamp(max_range_m / position_norm, max=1.0)
    output[..., 3:6] = output[..., 3:6] * torch.clamp(max_speed_mps / velocity_norm, max=1.0)
    return output


def score_candidates(prediction: torch.Tensor, observation: torch.Tensor, variance: torch.Tensor, method: str, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = circular_mean(prediction, dim=1)
    residual = angle_residual(mean, observation.unsqueeze(0))
    ensemble_anomaly = angle_residual(prediction, mean.unsqueeze(1))
    predictive_variance = ensemble_anomaly.square().mean(dim=1) + variance.unsqueeze(0)
    if method == "apce":
        between = angle_residual(mean, circular_mean(mean, dim=0).unsqueeze(0)).square().mean(dim=0)
        dimension_weight = 0.35 + 0.65 * between / between.mean().clamp_min(1e-12)
        dimension_weight = dimension_weight / dimension_weight.mean()
    else:
        dimension_weight = torch.ones_like(variance)
    score = -0.5 * (dimension_weight * (residual.square() / predictive_variance + predictive_variance.log())).mean(dim=-1)
    if method == "bma":
        logits = logits + score
        return logits, torch.softmax(logits - logits.max(), dim=0)
    if method == "pce":
        logits = logits + score / 0.66
        return logits, torch.softmax(logits - logits.max(), dim=0)
    logits = 0.975 * logits + score / 0.58
    weights = torch.softmax(logits - logits.max(), dim=0)
    uniform = torch.ones_like(weights) / len(weights)
    for _ in range(32):
        entropy = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum()
        if float(entropy) >= 0.34:
            break
        weights = 0.5 * (weights + uniform)
    return logits, weights / weights.sum()


def weighted_quantile(values: torch.Tensor, weights: torch.Tensor, quantile: float) -> torch.Tensor:
    order = torch.argsort(values)
    values = values[order]
    cumulative = torch.cumsum(weights[order], dim=0)
    index = torch.searchsorted(cumulative, torch.tensor(quantile, dtype=weights.dtype, device=weights.device)).clamp(max=len(values) - 1)
    return values[int(index)]


def crps(samples: torch.Tensor, weights: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    first = (weights[:, None] * (samples - truth).abs()).sum(dim=0)
    second = (
        weights[:, None, None]
        * weights[None, :, None]
        * (samples[:, None] - samples[None, :]).abs()
    ).sum(dim=(0, 1))
    return first - 0.5 * second


def smooth_positions(gps: dict[int, torch.Tensor], seconds: list[int], radius: int = 3) -> dict[int, torch.Tensor]:
    output = {}
    for second in seconds:
        window = [gps[key] for key in gps if abs(key - second) <= radius]
        if window:
            output[second] = torch.stack(window).mean(dim=0)
    return output


def calibrate_process_noise(gps: dict[int, torch.Tensor], seconds: list[int]) -> tuple[float, float]:
    smooth = smooth_positions(gps, seconds)
    keys = sorted(smooth)
    accelerations = []
    for left, middle, right in zip(keys, keys[1:], keys[2:]):
        if middle - left > 2 or right - middle > 2:
            continue
        velocity_left = (smooth[middle] - smooth[left]) / max(middle - left, 1)
        velocity_right = (smooth[right] - smooth[middle]) / max(right - middle, 1)
        accelerations.append(float(torch.linalg.vector_norm(velocity_right - velocity_left)))
    if not accelerations:
        return 0.25, 4.0
    accelerations.sort()
    q_min = max(0.10, accelerations[int(0.20 * (len(accelerations) - 1))])
    q_max = max(4.0 * q_min, accelerations[int(0.90 * (len(accelerations) - 1))])
    return q_min, min(q_max, 20.0)


def initial_acoustic_state(observations: dict[int, dict[int, tuple[float, float, int]]], calibration: dict, nodes: dict[int, torch.Tensor], threshold: float) -> tuple[int, torch.Tensor]:
    positions = []
    for second in sorted(observations):
        frame = apply_calibration(observations[second], calibration)
        position, inliers, _ = robust_triangulate(frame, nodes, threshold)
        if position is not None and len(inliers) >= 3:
            positions.append((second, position))
    if len(positions) < 10:
        raise RuntimeError("calibration segment has fewer than ten acoustic triangulations")
    selected = positions[-10:]
    time_end = selected[-1][0]
    position = torch.stack([item[1] for item in selected]).median(dim=0).values
    velocities = [
        (right[1] - left[1]) / max(right[0] - left[0], 1)
        for left, right in zip(selected, selected[1:])
    ]
    velocity = torch.stack(velocities).median(dim=0).values
    return time_end, torch.cat((position, velocity))


def calibration_anchor_contract(observations: dict[int, dict[int, tuple[float, float, int]]], calibration: dict, nodes: dict[int, torch.Tensor], threshold: float) -> dict[str, float]:
    anchors = []
    for second in sorted(observations):
        frame = apply_calibration(observations[second], calibration)
        position, inliers, _ = robust_triangulate(frame, nodes, threshold)
        if position is not None and len(inliers) >= 3:
            anchors.append((second, position))
    if len(anchors) < 10:
        raise RuntimeError("calibration segment has insufficient acoustic anchors")
    center = torch.stack(list(nodes.values())).mean(dim=0)
    ranges = sorted(float(torch.linalg.vector_norm(position - center)) for _, position in anchors)
    speeds = []
    for (left_t, left), (right_t, right) in zip(anchors, anchors[1:]):
        if right_t > left_t:
            speeds.append(float(torch.linalg.vector_norm(right - left) / (right_t - left_t)))
    speeds.sort()
    p90_range = ranges[min(len(ranges) - 1, int(0.90 * len(ranges)))]
    p95_speed = speeds[min(len(speeds) - 1, int(0.95 * len(speeds)))] if speeds else 20.0
    return {
        "calibration_anchor_count": float(len(anchors)),
        "range_limit_m": max(800.0, 1.30 * p90_range + 100.0),
        "speed_limit_mps": min(80.0, max(50.0, 1.10 * p95_speed)),
        "relocalization_distance_m": max(150.0, 2.0 * p95_speed),
        "minimum_position_spread_m": max(35.0, 0.5 * p95_speed),
        "minimum_analysis_inliers": 5.0,
    }


def calibration_anchor_variance(observations: dict[int, dict[int, tuple[float, float, int]]], gps: dict[int, torch.Tensor], calibration: dict, nodes: dict[int, torch.Tensor], threshold: float) -> torch.Tensor:
    residuals = []
    for second, raw_frame in observations.items():
        frame = apply_calibration(raw_frame, calibration)
        anchor, inliers, _ = robust_triangulate(frame, nodes, threshold)
        truth = truth_at(gps, second, int(calibration["delay_s"]))
        if anchor is not None and truth is not None and len(inliers) >= 5:
            residuals.append(anchor - truth)
    if len(residuals) < 10:
        return torch.tensor((50.0**2, 90.0**2, 50.0**2), dtype=torch.float64)
    residuals = torch.stack(residuals).abs()
    scales = []
    for dimension in range(3):
        values = torch.sort(residuals[:, dimension]).values
        scales.append(max(20.0, float(values[min(len(values) - 1, int(0.90 * len(values)))])))
    return torch.tensor(tuple(value * value for value in scales), dtype=torch.float64)


def analysis_anchor_guard(states: torch.Tensor, anchor: torch.Tensor, anchor_velocity: torch.Tensor, contract: dict[str, float]) -> tuple[torch.Tensor, float, bool]:
    mean_position = states[..., :3].mean(dim=-2)
    distance = float(torch.linalg.vector_norm(mean_position - anchor))
    if distance <= contract["relocalization_distance_m"]:
        return physical_project(states, contract["range_limit_m"], contract["speed_limit_mps"]), distance, False
    output = states.clone()
    position_anomaly = output[..., :3] - mean_position.unsqueeze(-2)
    spread = position_anomaly.square().mean(dim=-2).sqrt().mean().clamp_min(1e-6)
    scale = max(1.0, contract["minimum_position_spread_m"] / float(spread))
    output[..., :3] = anchor.unsqueeze(-2) + scale * position_anomaly
    velocity_mean = output[..., 3:6].mean(dim=-2)
    output[..., 3:6] += 0.75 * (anchor_velocity - velocity_mean).unsqueeze(-2)
    return physical_project(output, contract["range_limit_m"], contract["speed_limit_mps"]), distance, True


def evaluation_frames(observations: dict[int, dict[int, tuple[float, float, int]]], calibration: dict, nodes: dict[int, torch.Tensor], threshold: float) -> dict[int, tuple[dict[int, tuple[float, float, float]], list[int], torch.Tensor | None]]:
    output = {}
    for second, raw_frame in observations.items():
        frame = apply_calibration(raw_frame, calibration)
        anchor, inliers, _ = robust_triangulate(frame, nodes, threshold)
        output[second] = (frame, inliers, anchor)
    return output


def initialize_from_anchor(
    anchor: torch.Tensor,
    anchor_variance: torch.Tensor,
    ensemble_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create paired analysis/shadow ensembles without carrying state across gaps."""
    position_scale = anchor_variance.clamp_min(20.0 ** 2).sqrt().to(device)
    initial = torch.zeros((ensemble_size, 6), dtype=torch.float64, device=device)
    initial[:, :3] = anchor.unsqueeze(0) + torch.randn(
        (ensemble_size, 3), dtype=torch.float64, device=device, generator=generator
    ) * position_scale.unsqueeze(0)
    initial[:, 3:6] = torch.randn(
        (ensemble_size, 3), dtype=torch.float64, device=device, generator=generator
    ) * 15.0
    alpha_grid = torch.tensor(ALPHA_GRID, dtype=torch.float64, device=device)
    branches = initial.unsqueeze(0).repeat(len(ALPHA_GRID), 1, 1)
    shadows = branches.clone()
    single = initial.clone()
    augmented = torch.cat(
        (initial, torch.full((ensemble_size, 1), 0.50, dtype=torch.float64, device=device)),
        dim=1,
    )
    return branches, shadows, single, augmented, alpha_grid


def run_method(
    method: str,
    seed: int,
    nodes: dict[int, torch.Tensor],
    calibration_observations: dict[int, dict[int, tuple[float, float, int]]],
    evaluation_observations: dict[int, dict[int, tuple[float, float, int]]],
    gps: dict[int, torch.Tensor],
    calibration: dict,
    threshold: float,
    q_min: float,
    q_max: float,
    output: Path,
    device_name: str,
    source_provenance: dict,
    anchor_fused: bool = False,
    observation_only: bool = False,
) -> dict:
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(seed)
    center = torch.stack(list(nodes.values())).mean(dim=0)
    centered_nodes = {node: value - center for node, value in nodes.items()}
    calibration_centered = {second: frame for second, frame in calibration_observations.items()}
    initial_time, initial_state_global = initial_acoustic_state(calibration_centered, calibration, nodes, threshold)
    anchor_contract = calibration_anchor_contract(calibration_centered, calibration, nodes, threshold)
    anchor_variance = calibration_anchor_variance(calibration_centered, gps, calibration, nodes, threshold).to(device)
    ensemble_size = 48
    initial_state = initial_state_global.clone()
    initial_state[:3] -= center
    initial = initial_state.to(device).repeat(ensemble_size, 1)
    initial[:, :3] += torch.randn((ensemble_size, 3), dtype=torch.float64, device=device, generator=generator) * 100.0
    initial[:, 3:6] += torch.randn((ensemble_size, 3), dtype=torch.float64, device=device, generator=generator) * 15.0
    alpha_grid = torch.tensor(ALPHA_GRID, dtype=torch.float64, device=device)
    branches = initial.unsqueeze(0).repeat(len(ALPHA_GRID), 1, 1)
    shadows = branches.clone()
    single = initial.clone()
    augmented = torch.cat((initial, torch.full((ensemble_size, 1), 0.50, dtype=torch.float64, device=device)), dim=1)
    logits = torch.zeros(len(ALPHA_GRID), dtype=torch.float64, device=device)
    branch_weights = torch.ones(len(ALPHA_GRID), dtype=torch.float64, device=device) / len(ALPHA_GRID)
    frames = evaluation_frames(evaluation_observations, calibration, nodes, threshold)
    records = []
    availability = []
    observation_segment_id = 0
    active_segment = False
    previous = initial_time
    previous_anchor: torch.Tensor | None = None
    start = time.monotonic()
    for second in sorted(frames):
        frame, node_ids, anchor_global = frames[second]
        accepted = anchor_global is not None and len(node_ids) >= int(anchor_contract["minimum_analysis_inliers"])
        availability.append({
            "time_s": second,
            "inlier_nodes": len(node_ids),
            "accepted_acoustic_frame": bool(accepted),
            "has_anchor": anchor_global is not None,
        })
        if observation_only and not accepted:
            active_segment = False
            previous_anchor = None
            continue
        if not accepted:
            # Preserve the legacy archive protocol when observation-only mode is off.
            anchor_global = anchor_global if anchor_global is not None else torch.zeros(3, dtype=torch.float64)
            node_ids = list(node_ids)
        first_in_segment = observation_only and not active_segment
        dt = max(1.0, float(second - previous))
        anchor = anchor_global.to(device) - center.to(device)
        anchor_velocity = torch.zeros(3, dtype=torch.float64, device=device)
        if previous_anchor is not None and not first_in_segment:
            anchor_velocity = (anchor - previous_anchor) / dt
            anchor_velocity = anchor_velocity * torch.clamp(anchor_contract["speed_limit_mps"] / torch.linalg.vector_norm(anchor_velocity).clamp_min(1.0), max=1.0)
        if first_in_segment:
            observation_segment_id += 1
            branches, shadows, single, augmented, alpha_grid = initialize_from_anchor(
                anchor, anchor_variance, ensemble_size, generator, device
            )
            logits = torch.zeros(len(ALPHA_GRID), dtype=torch.float64, device=device)
            branch_weights = torch.ones(len(ALPHA_GRID), dtype=torch.float64, device=device) / len(ALPHA_GRID)
            previous = second
            dt = 1.0
        if not first_in_segment:
            standard_noise = torch.randn((ensemble_size, 3), dtype=torch.float64, device=device, generator=generator)
            if method == "denkf":
                single = propagate(single, torch.full((ensemble_size,), 0.50, dtype=torch.float64, device=device), dt, standard_noise, q_min, q_max, anchor_contract["range_limit_m"], anchor_contract["speed_limit_mps"])
            elif method == "aug_enkf":
                alpha_members = augmented[:, 6].clamp(0.0, 1.0)
                augmented[:, :6] = propagate(augmented[:, :6], alpha_members, dt, standard_noise, q_min, q_max, anchor_contract["range_limit_m"], anchor_contract["speed_limit_mps"])
                augmented[:, 6] = (alpha_members + 0.01 * torch.randn((ensemble_size,), dtype=torch.float64, device=device, generator=generator)).clamp(0.0, 1.0)
            else:
                for index in range(len(ALPHA_GRID)):
                    branches[index] = propagate(branches[index], alpha_grid[index].expand(ensemble_size), dt, standard_noise, q_min, q_max, anchor_contract["range_limit_m"], anchor_contract["speed_limit_mps"])
                    shadows[index] = propagate(shadows[index], alpha_grid[index].expand(ensemble_size), dt, standard_noise, q_min, q_max, anchor_contract["range_limit_m"], anchor_contract["speed_limit_mps"])
        analysis_update = accepted
        if analysis_update:
            observation, variance = observation_tensor(frame, node_ids, device)
            centered_node_subset = {node: centered_nodes[node].to(device) for node in node_ids}
        if method == "denkf" and analysis_update:
            single = denkf_update(single, observation, variance, centered_node_subset, node_ids)
            if anchor_fused:
                single = anchor_denkf_update(single, anchor, anchor_variance)
            single, anchor_distance, relocalized = analysis_anchor_guard(single, anchor, anchor_velocity, anchor_contract)
            samples = single
            weights = torch.ones(ensemble_size, dtype=torch.float64, device=device) / ensemble_size
            alpha_estimate = torch.tensor(0.50, dtype=torch.float64, device=device)
            entropy = 0.0
        elif method == "aug_enkf" and analysis_update:
            predicted = predict_angles(augmented[:, :6], centered_node_subset, node_ids)
            state_mean = augmented.mean(dim=0)
            observation_mean = circular_mean(predicted, dim=0)
            state_anomaly = augmented - state_mean
            observation_anomaly = angle_residual(predicted, observation_mean)
            cross = state_anomaly.T @ observation_anomaly / (ensemble_size - 1)
            innovation_covariance = observation_anomaly.T @ observation_anomaly / (ensemble_size - 1) + torch.diag(variance)
            gain = torch.linalg.solve(innovation_covariance.T, cross.T).T
            mean_increment = gain @ angle_residual(observation, observation_mean)
            augmented = state_mean + mean_increment + state_anomaly - 0.5 * observation_anomaly @ gain.T
            if anchor_fused:
                augmented[:, :6] = anchor_denkf_update(augmented[:, :6], anchor, anchor_variance)
            augmented[:, :6], anchor_distance, relocalized = analysis_anchor_guard(augmented[:, :6], anchor, anchor_velocity, anchor_contract)
            augmented[:, 6] = augmented[:, 6].clamp(0.0, 1.0)
            samples = augmented[:, :6]
            weights = torch.ones(ensemble_size, dtype=torch.float64, device=device) / ensemble_size
            alpha_estimate = augmented[:, 6].mean()
            entropy = 0.0
        elif method in ("bma", "pce", "apce") and analysis_update:
            evidence_states = shadows if method in ("pce", "apce") else branches
            prediction = predict_angles(evidence_states, centered_node_subset, node_ids)
            logits, branch_weights = score_candidates(prediction, observation, variance, method, logits)
            guard_distances, guard_flags = [], []
            for index in range(len(ALPHA_GRID)):
                branches[index] = denkf_update(branches[index], observation, variance, centered_node_subset, node_ids)
                if anchor_fused:
                    branches[index] = anchor_denkf_update(branches[index], anchor, anchor_variance)
                branches[index], guard_distance, guard_flag = analysis_anchor_guard(branches[index], anchor, anchor_velocity, anchor_contract)
                guard_distances.append(guard_distance); guard_flags.append(guard_flag)
            samples = branches.reshape(-1, 6)
            weights = branch_weights.repeat_interleave(ensemble_size) / ensemble_size
            alpha_estimate = (branch_weights * alpha_grid).sum()
            entropy = float(-(branch_weights.clamp_min(1e-12) * branch_weights.clamp_min(1e-12).log()).sum())
            anchor_distance = float(sum(guard_distances) / len(guard_distances))
            relocalized = any(guard_flags)
        else:
            if method == "denkf":
                samples = single
                weights = torch.ones(ensemble_size, dtype=torch.float64, device=device) / ensemble_size
                alpha_estimate = torch.tensor(0.50, dtype=torch.float64, device=device)
                entropy = 0.0
            elif method == "aug_enkf":
                samples = augmented[:, :6]
                weights = torch.ones(ensemble_size, dtype=torch.float64, device=device) / ensemble_size
                alpha_estimate = augmented[:, 6].mean()
                entropy = 0.0
            else:
                samples = branches.reshape(-1, 6)
                weights = branch_weights.repeat_interleave(ensemble_size) / ensemble_size
                alpha_estimate = (branch_weights * alpha_grid).sum()
                entropy = float(-(branch_weights.clamp_min(1e-12) * branch_weights.clamp_min(1e-12).log()).sum())
            anchor_distance = float(torch.linalg.vector_norm(samples[:, :3].mean(dim=0) - anchor))
            relocalized = False
        mean = (weights[:, None] * samples).sum(dim=0)
        mean_global = mean.clone()
        mean_global[:3] += center.to(device)
        truth_global = truth_at(gps, second, calibration["delay_s"])
        truth_centered = truth_global.to(device) - center.to(device) if truth_global is not None else None
        lower = torch.stack([weighted_quantile(samples[:, dim], weights, 0.05) for dim in range(3)])
        upper = torch.stack([weighted_quantile(samples[:, dim], weights, 0.95) for dim in range(3)])
        row = {
            "time_s": second,
            "method": method,
            "seed": seed,
            "inlier_nodes": len(node_ids),
            "analysis_update": bool(analysis_update),
            "evidence_update": bool(analysis_update and method in ("bma", "pce", "apce")),
            "observation_segment_id": observation_segment_id if observation_only else 0,
            "accepted_acoustic_frame": bool(accepted),
            "anchor_x": float(anchor_global[0]),
            "anchor_y": float(anchor_global[1]),
            "anchor_z": float(anchor_global[2]),
            "anchor_distance_m": float(anchor_distance),
            "relocalized": bool(relocalized),
            "px": float(mean_global[0]),
            "py": float(mean_global[1]),
            "pz": float(mean_global[2]),
            "vx": float(mean[3]),
            "vy": float(mean[4]),
            "vz": float(mean[5]),
            "alpha_estimate": float(alpha_estimate),
            "evidence_entropy": entropy,
        }
        for dimension, name in enumerate(("px", "py", "pz")):
            row[f"{name}_lo"] = float(lower[dimension] + center.to(device)[dimension])
            row[f"{name}_hi"] = float(upper[dimension] + center.to(device)[dimension])
        if truth_centered is not None:
            error = mean[:3] - truth_centered
            row.update({
                "truth_x": float(truth_global[0]),
                "truth_y": float(truth_global[1]),
                "truth_z": float(truth_global[2]),
                "position_error_m": float(torch.linalg.vector_norm(error)),
                "east_abs_error_m": abs(float(error[0])),
                "north_abs_error_m": abs(float(error[1])),
                "up_abs_error_m": abs(float(error[2])),
                "crps_position_m": float(crps(samples[:, :3], weights, truth_centered).mean()),
                "coverage_90": float(((truth_centered >= lower) & (truth_centered <= upper)).double().mean()),
                "interval_width_m": float((upper - lower).mean()),
            })
        records.append(row)
        previous = second
        previous_anchor = anchor
        if observation_only:
            active_segment = True
    runtime = time.monotonic() - start
    payload = {
        "status": "valid" if records and all(math.isfinite(row.get("position_error_m", 0.0)) for row in records) else "invalid",
        "method": method,
        "seed": seed,
        "role": "numerical-sensitivity run on one held-out physical trajectory",
        "alpha_definition": "operational maneuver-intensity coordinate controlling acceleration process noise",
        "alpha_grid": list(ALPHA_GRID),
        "q_min_accel_mps2": q_min,
        "q_max_accel_mps2": q_max,
        "anchor_guard_contract": anchor_contract,
        "anchor_fused": anchor_fused,
        "anchor_variance_diag_m2": [float(value) for value in anchor_variance.detach().cpu()],
        "observation_only": observation_only,
        "evaluation_frame_count": len(frames),
        "accepted_frame_count": len(records),
        "accepted_frame_fraction": len(records) / max(len(frames), 1),
        "observation_segment_count": observation_segment_id,
        "availability": availability,
        "runtime_s": runtime,
        "records": records,
        "runner_sha256": sha256(Path(__file__)),
        "source_provenance": source_provenance,
    }
    write_json(output / "runs" / f"{method}_seed_{seed}.json", payload)
    return payload


def aggregate(output: Path) -> dict:
    rows = []
    for path in sorted((output / "runs").glob("*_seed_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = [row for row in payload["records"] if row.get("position_error_m") is not None]
        if not records:
            continue
        errors = [row["position_error_m"] for row in records]
        rows.append({
            "method": payload["method"],
            "seed": payload["seed"],
            "status": payload["status"],
            "frames": len(records),
            "evaluation_frames": payload.get("evaluation_frame_count", len(records)),
            "accepted_frame_fraction": payload.get("accepted_frame_fraction", 1.0),
            "observation_segment_count": payload.get("observation_segment_count", 0),
            "position_rmse_m": math.sqrt(sum(value * value for value in errors) / len(errors)),
            "position_median_error_m": statistics.median(errors),
            "position_p90_error_m": sorted(errors)[min(len(errors) - 1, int(0.90 * len(errors)))],
            "crps_position_m": statistics.mean(row["crps_position_m"] for row in records),
            "coverage_90": statistics.mean(row["coverage_90"] for row in records),
            "interval_width_m": statistics.mean(row["interval_width_m"] for row in records),
            "alpha_mean": statistics.mean(row["alpha_estimate"] for row in records),
            "runtime_s": payload["runtime_s"],
        })
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = list(rows[0]) if rows else ["method"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    method_summary = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        if not selected:
            continue
        method_summary.append({
            "method": method,
            "runs": len(selected),
            "accepted_frame_fraction_mean": statistics.mean(row["accepted_frame_fraction"] for row in selected),
            "observation_segment_count_mean": statistics.mean(row["observation_segment_count"] for row in selected),
            "position_rmse_m_mean": statistics.mean(row["position_rmse_m"] for row in selected),
            "position_rmse_median": statistics.median(row["position_rmse_m"] for row in selected),
            "crps_position_m_mean": statistics.mean(row["crps_position_m"] for row in selected),
            "coverage_90_mean": statistics.mean(row["coverage_90"] for row in selected),
            "interval_width_m_mean": statistics.mean(row["interval_width_m"] for row in selected),
            "alpha_mean": statistics.mean(row["alpha_mean"] for row in selected),
            "runtime_s_mean": statistics.mean(row["runtime_s"] for row in selected),
        })
    with (output / "method_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = list(method_summary[0]) if method_summary else ["method"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(method_summary)
    manifest = {
        "valid_runs": sum(row["status"] == "valid" for row in rows),
        "expected_runs": 25,
        "methods": [row["method"] for row in method_summary],
        "protocol": "observation-gated segmented acoustic reconstruction" if any(
            json.loads(path.read_text(encoding="utf-8")).get("observation_only", False)
            for path in sorted((output / "runs").glob("*_seed_*.json"))
        ) else "legacy continuous/guarded tracking",
        "seeds_are_numerical_sensitivity_not_independent_experiments": True,
        "formal_admission": False,
    }
    write_json(output / "tracking_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-root", type=Path, default=Path("<PRIVATE_DATA_ROOT>/2017保定实验/2017保定实验"))
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--frontend", type=Path)
    parser.add_argument("--raw-admission", type=Path)
    parser.add_argument("--calibration-segment", default="danyuan_panxuan_2")
    parser.add_argument("--evaluation-segment", default="danyuan_panxuan_3")
    parser.add_argument("--calibration-seconds", type=int)
    parser.add_argument("--anchor-fused", action="store_true")
    parser.add_argument("--observation-only", action="store_true", help="omit frames without a >=5-node 3-D acoustic anchor and restart after gaps")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        print(json.dumps(aggregate(args.output), ensure_ascii=False, indent=2))
        return
    if args.method is None or args.seed is None:
        raise SystemExit("--method and --seed are required unless --aggregate is set")
    archive = args.remote_root / "20171107保定实验"
    project = archive / "project/20171107baoding"
    nodes_raw = parse_nod(archive / "GPS_data/20171107baoding.nod")
    nodes = {node: torch.tensor((value["x"], value["y"], value["z"]), dtype=torch.float64) for node, value in nodes_raw.items()}
    gps = fused_plane1_gps(archive / "GPS_data/GPS1_plane1.gps", archive / "GPS_data/GPS2_plane1.gps")
    if args.frontend is not None:
        if args.raw_admission is None:
            raise RuntimeError("--raw-admission is required with --frontend")
        raw_gate_path = args.raw_admission / "raw_frontend_gate.json"
        raw_gate = json.loads(raw_gate_path.read_text(encoding="utf-8"))
        if not raw_gate["raw_wav_frontend_admitted"]:
            raise RuntimeError("raw-WAV frontend admission gate did not pass")
        frontend_calibration_path = args.frontend / "frontend_calibration.json"
        frontend_calibration = json.loads(frontend_calibration_path.read_text(encoding="utf-8"))
        loaded = load_frontend_observations(args.frontend / "observations.csv")
        if args.calibration_segment not in loaded:
            raise RuntimeError(f"calibration segment not found: {args.calibration_segment}")
        if args.evaluation_segment not in loaded:
            raise RuntimeError(f"evaluation segment not found: {args.evaluation_segment}")
        calibration_observations = loaded[args.calibration_segment]
        evaluation_observations = loaded[args.evaluation_segment]
        if args.calibration_segment == args.evaluation_segment and args.calibration_seconds is not None:
            start_time = min(calibration_observations)
            split_time = start_time + args.calibration_seconds
            calibration_observations = {
                second: frame for second, frame in calibration_observations.items()
                if second < split_time
            }
            evaluation_observations = {
                second: frame for second, frame in evaluation_observations.items()
                if second >= split_time
            }
        calibration = {
            "delay_s": int(frontend_calibration["selected_delay_s"]),
            "nodes": {},
        }
        eligible = set(int(node) for node in raw_gate["eligible_nodes"])
        for node, metrics in raw_gate["node_calibration"].items():
            node_id = int(node)
            calibration["nodes"][node_id] = {
                "azimuth_sign": 1.0,
                "azimuth_offset_deg": 0.0,
                "elevation_offset_deg": 0.0,
                "observation_sigma_deg": max(2.0, float(metrics["joint_median_error_deg"])),
                "eligible": node_id in eligible,
            }
        threshold = float(raw_gate["selected_ransac_threshold_deg"])
        source_provenance = {
            "observation_source": "corrected direct-WAV MUSIC frontend",
            "calibration_segment": args.calibration_segment,
            "evaluation_segment": args.evaluation_segment,
            "calibration_seconds": args.calibration_seconds,
            "frontend": str(args.frontend),
            "frontend_manifest_sha256": sha256(args.frontend / "frontend_manifest.json"),
            "frontend_calibration_sha256": sha256(frontend_calibration_path),
            "raw_admission_sha256": sha256(raw_gate_path),
        }
    else:
        gate = json.loads((args.admission / "nearfield_gate.json").read_text(encoding="utf-8"))
        if not gate["three_dimensional_tracking_admitted"]:
            raise RuntimeError("near-field 3-D admission gate did not pass")
        calibration = json.loads((args.admission / "nearfield_calibration.json").read_text(encoding="utf-8"))
        calibration["nodes"] = {int(node): config for node, config in calibration["nodes"].items()}
        calibration_observations = parse_historical_doa(project / "danyuan_panxuan_2")
        evaluation_observations = parse_historical_doa(project / "danyuan_panxuan_3")
        threshold = float(gate["selected_ransac_threshold_deg"])
        source_provenance = {
            "observation_source": "historical DOA product",
            "admission_sha256": sha256(args.admission / "nearfield_gate.json"),
        }
    q_min, q_max = calibrate_process_noise(gps, sorted(calibration_observations))
    run_method(
        args.method,
        args.seed,
        nodes,
        calibration_observations,
        evaluation_observations,
        gps,
        calibration,
        threshold,
        q_min,
        q_max,
        args.output,
        args.device,
        source_provenance,
        args.anchor_fused,
        args.observation_only,
    )


if __name__ == "__main__":
    main()
