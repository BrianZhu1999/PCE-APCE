#!/usr/bin/env python3
"""Reliability-alpha IMM tracker for the admitted Baoding near-field task.

This diagnostic keeps GPS outside assimilation. It uses only calibration-window
GPS to freeze DOA and acoustic-anchor covariance. Evaluation receives corrected
WAV-MUSIC DOA, RANSAC inlier masks, and acoustic 3-D anchors.
"""
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
    from .nearfield_tracker import (
        crps, evaluation_frames, initial_acoustic_state, load_frontend_observations,
        physical_project, predict_angles, truth_at, weighted_quantile,
    )
    from .nearfield_audit import apply_calibration, robust_triangulate
    from .run_baoding import angle_residual, parse_gps, parse_nod, sha256
except ImportError:
    from nearfield_tracker import (
        crps, evaluation_frames, initial_acoustic_state, load_frontend_observations,
        physical_project, predict_angles, truth_at, weighted_quantile,
    )
    from nearfield_audit import apply_calibration, robust_triangulate
    from run_baoding import angle_residual, parse_gps, parse_nod, sha256


METHODS = ("denkf", "aug_enkf", "bma", "pce", "apce")
R_SCALE_GRID = (0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00)
MODE_TRANSITION = torch.tensor(((0.96, 0.04), (0.04, 0.96)), dtype=torch.float64)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def circular_mean(values: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.atan2(torch.sin(values).mean(dim=dim), torch.cos(values).mean(dim=dim))


def observation_tensor(frame: dict[int, tuple[float, float, float]], nodes: list[int], device: torch.device, multiplier: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    values, variance = [], []
    for node in nodes:
        azimuth, elevation, sigma = frame[node]
        values.extend((math.radians(azimuth), math.radians(elevation)))
        variance.extend((math.radians(sigma * multiplier) ** 2, math.radians(max(2.0, 0.65 * sigma) * multiplier) ** 2))
    return torch.tensor(values, dtype=torch.float64, device=device), torch.tensor(variance, dtype=torch.float64, device=device)


def angle_denkf(states: torch.Tensor, observation: torch.Tensor, variance: torch.Tensor, nodes: dict[int, torch.Tensor], node_ids: list[int]) -> torch.Tensor:
    prediction = predict_angles(states, nodes, node_ids)
    state_mean, observation_mean = states.mean(dim=0), circular_mean(prediction, dim=0)
    state_anomaly = states - state_mean
    observation_anomaly = angle_residual(prediction, observation_mean)
    denom = max(states.shape[0] - 1, 1)
    cross = state_anomaly.T @ observation_anomaly / denom
    covariance = observation_anomaly.T @ observation_anomaly / denom + torch.diag(variance)
    gain = torch.linalg.solve(covariance.T, cross.T).T
    mean_increment = gain @ angle_residual(observation, observation_mean)
    anomaly_increment = 0.5 * observation_anomaly @ gain.T
    return state_mean + mean_increment + state_anomaly - anomaly_increment


def anchor_denkf(states: torch.Tensor, anchor: torch.Tensor, covariance_diag: torch.Tensor) -> torch.Tensor:
    prediction = states[:, :3]
    state_mean, observation_mean = states.mean(dim=0), prediction.mean(dim=0)
    state_anomaly, observation_anomaly = states - state_mean, prediction - observation_mean
    denom = max(states.shape[0] - 1, 1)
    cross = state_anomaly.T @ observation_anomaly / denom
    covariance = observation_anomaly.T @ observation_anomaly / denom + torch.diag(covariance_diag)
    gain = torch.linalg.solve(covariance.T, cross.T).T
    mean_increment = gain @ (anchor - observation_mean)
    anomaly_increment = 0.5 * observation_anomaly @ gain.T
    return state_mean + mean_increment + state_anomaly - anomaly_increment


def imm_propagate(states: torch.Tensor, mode_weights: torch.Tensor, dt: float, standard_noise: torch.Tensor, q_cv: float, q_turn: float, range_limit: float, speed_limit: float) -> tuple[torch.Tensor, torch.Tensor]:
    transition = MODE_TRANSITION.to(states)
    next_weights = transition.T @ mode_weights
    mixed = []
    for destination in range(2):
        source_weight = mode_weights * transition[:, destination] / next_weights[destination].clamp_min(1e-12)
        mixed.append((source_weight[:, None, None] * states).sum(dim=0))
    mixed_states = torch.stack(mixed)
    propagated = []
    for mode, scale in enumerate((q_cv, q_turn)):
        acceleration = standard_noise * scale
        output = mixed_states[mode].clone()
        output[:, :3] += output[:, 3:6] * dt + 0.5 * acceleration * dt * dt
        output[:, 3:6] += acceleration * dt
        propagated.append(physical_project(output, range_limit, speed_limit))
    return torch.stack(propagated), next_weights / next_weights.sum()


def mode_likelihood(states: torch.Tensor, observation: torch.Tensor, variance: torch.Tensor, nodes: dict[int, torch.Tensor], node_ids: list[int]) -> torch.Tensor:
    values = []
    for mode in range(2):
        prediction = predict_angles(states[mode], nodes, node_ids)
        mean = circular_mean(prediction, dim=0)
        anomaly = angle_residual(prediction, mean)
        predictive_variance = anomaly.square().mean(dim=0) + variance
        residual = angle_residual(mean, observation)
        values.append(-0.5 * (residual.square() / predictive_variance + predictive_variance.log()).mean())
    return torch.stack(values)


def update_branch(states: torch.Tensor, mode_weights: torch.Tensor, observation: torch.Tensor, variance: torch.Tensor, anchor: torch.Tensor | None, anchor_variance: torch.Tensor | None, nodes: dict[int, torch.Tensor], node_ids: list[int], r_scale: float, quality: str) -> tuple[torch.Tensor, torch.Tensor]:
    if quality == "high":
        log_mode = mode_likelihood(states, observation, variance * r_scale, nodes, node_ids)
        mode_weights = torch.softmax(mode_weights.clamp_min(1e-12).log() + log_mode, dim=0)
        updated = torch.stack([angle_denkf(states[mode], observation, variance * r_scale, nodes, node_ids) for mode in range(2)])
        if anchor is not None and anchor_variance is not None:
            updated = torch.stack([anchor_denkf(updated[mode], anchor, anchor_variance * r_scale) for mode in range(2)])
        return updated, mode_weights
    if quality == "weak_anchor" and anchor is not None and anchor_variance is not None:
        updated = torch.stack([anchor_denkf(states[mode], anchor, anchor_variance * r_scale * 16.0) for mode in range(2)])
        return updated, mode_weights
    if quality == "weak_angle":
        updated = torch.stack([angle_denkf(states[mode], observation, variance * r_scale * 25.0, nodes, node_ids) for mode in range(2)])
        return updated, mode_weights
    return states, mode_weights


def mixture_prediction(states: torch.Tensor, mode_weights: torch.Tensor, nodes: dict[int, torch.Tensor], node_ids: list[int]) -> torch.Tensor:
    # [mode, ensemble, observation] -> candidate x ensemble x observation
    prediction = torch.stack([predict_angles(states[mode], nodes, node_ids) for mode in range(2)])
    return (mode_weights[:, None, None] * prediction).sum(dim=0)


def evidence_update(method: str, shadow_states: torch.Tensor, mode_weights: torch.Tensor, observation: torch.Tensor, variance: torch.Tensor, nodes: dict[int, torch.Tensor], node_ids: list[int], scales: torch.Tensor, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    candidate_means = torch.stack([circular_mean(mixture_prediction(shadow_states[index], mode_weights[index], nodes, node_ids), dim=0) for index in range(len(scales))])
    between = angle_residual(candidate_means, circular_mean(candidate_means, dim=0).unsqueeze(0)).square().mean(dim=0)
    scores = []
    for index, r_scale in enumerate(scales):
        prediction = mixture_prediction(shadow_states[index], mode_weights[index], nodes, node_ids)
        mean = circular_mean(prediction, dim=0)
        anomaly = angle_residual(prediction, mean)
        pred_variance = anomaly.square().mean(dim=0) + variance * r_scale
        residual = angle_residual(mean, observation)
        if method == "apce":
            dim_weight = 0.35 + 0.65 * between / between.mean().clamp_min(1e-12)
            dim_weight = dim_weight / dim_weight.mean()
        else:
            dim_weight = torch.ones_like(variance)
        scores.append(-0.5 * (dim_weight * (residual.square() / pred_variance + pred_variance.log())).mean())
    score = torch.stack(scores)
    if method == "bma":
        logits = logits + score
        return logits, torch.softmax(logits - logits.max(), dim=0)
    if method == "pce":
        logits = logits + score
        return logits, torch.softmax(logits - logits.max(), dim=0)
    logits = 0.99 * logits + score
    weights = torch.softmax(logits - logits.max(), dim=0)
    uniform = torch.ones_like(weights) / len(weights)
    entropy_target = 0.75
    for _ in range(48):
        entropy = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum()
        if float(entropy) >= entropy_target:
            break
        weights = 0.5 * (weights + uniform)
    return logits, weights / weights.sum()


def anchor_calibration(observations: dict[int, dict[int, tuple[float, float, int]]], gps: dict[int, torch.Tensor], calibration: dict, nodes: dict[int, torch.Tensor], threshold: float) -> tuple[torch.Tensor, dict[str, float]]:
    errors, anchors = [], []
    for second, raw_frame in observations.items():
        frame = apply_calibration(raw_frame, calibration)
        anchor, inliers, _ = robust_triangulate(frame, nodes, threshold)
        truth = truth_at(gps, second, calibration["delay_s"])
        if anchor is not None and truth is not None and len(inliers) >= 5:
            errors.append(anchor - truth); anchors.append((second, anchor))
    if len(errors) < 10:
        raise RuntimeError("insufficient calibration anchors")
    stack = torch.stack(errors)
    sigma = []
    for dim in range(3):
        values = torch.sort(stack[:, dim].abs()).values
        sigma.append(max(20.0, float(values[min(len(values) - 1, int(0.90 * len(values)))])))
    ranges = sorted(float(torch.linalg.vector_norm(anchor - torch.stack(list(nodes.values())).mean(dim=0))) for _, anchor in anchors)
    speeds = []
    for (left_t, left), (right_t, right) in zip(anchors, anchors[1:]):
        if right_t > left_t:
            speeds.append(float(torch.linalg.vector_norm(right - left) / (right_t - left_t)))
    speeds.sort()
    p90_range = ranges[min(len(ranges) - 1, int(0.90 * len(ranges)))]
    p95_speed = speeds[min(len(speeds) - 1, int(0.95 * len(speeds)))] if speeds else 50.0
    contract = {
        "anchor_count": float(len(anchors)), "range_limit_m": max(800.0, 1.30 * p90_range + 100.0),
        "speed_limit_mps": min(80.0, max(50.0, 1.10 * p95_speed)), "anchor_sigma_x_m": sigma[0], "anchor_sigma_y_m": sigma[1], "anchor_sigma_z_m": sigma[2],
    }
    return torch.tensor([value * value for value in sigma], dtype=torch.float64), contract


def calibration_motion_noise(gps: dict[int, torch.Tensor], observations: dict[int, dict[int, tuple[float, float, int]]], delay: int) -> tuple[float, float]:
    keys = [second for second in sorted(observations) if truth_at(gps, second, delay) is not None]
    positions = [(second, truth_at(gps, second, delay)) for second in keys]
    accelerations = []
    for (t0, p0), (t1, p1), (t2, p2) in zip(positions, positions[1:], positions[2:]):
        if t1 - t0 != 1 or t2 - t1 != 1:
            continue
        accelerations.append(float(torch.linalg.vector_norm((p2 - p1) - (p1 - p0))))
    accelerations.sort()
    if not accelerations:
        return 0.3, 2.0
    return max(0.2, accelerations[int(0.20 * (len(accelerations) - 1))]), max(1.0, accelerations[int(0.85 * (len(accelerations) - 1))])


def flatten_branch(states: torch.Tensor, mode_weights: torch.Tensor, candidate_weights: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    # states [candidate, mode, ensemble, 6] or [mode, ensemble, 6]
    if states.ndim == 3:
        samples = states.reshape(-1, 6)
        weight = mode_weights.repeat_interleave(states.shape[1]) / states.shape[1]
        return samples, weight
    candidate_count, _, ensemble_size, _ = states.shape
    samples, weights = [], []
    if candidate_weights is None:
        candidate_weights = torch.ones(candidate_count, dtype=torch.float64, device=states.device) / candidate_count
    for candidate in range(candidate_count):
        samples.append(states[candidate].reshape(-1, 6))
        weights.append(candidate_weights[candidate] * mode_weights[candidate].repeat_interleave(ensemble_size) / ensemble_size)
    return torch.cat(samples), torch.cat(weights)


def run_method(method: str, seed: int, nodes: dict[int, torch.Tensor], calibration_obs: dict[int, dict[int, tuple[float, float, int]]], evaluation_obs: dict[int, dict[int, tuple[float, float, int]]], gps: dict[int, torch.Tensor], calibration: dict, threshold: float, output: Path, device_name: str, provenance: dict) -> dict:
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(seed)
    center = torch.stack(list(nodes.values())).mean(dim=0)
    centered_nodes = {node: value.to(device) - center.to(device) for node, value in nodes.items()}
    anchor_variance, contract = anchor_calibration(calibration_obs, gps, calibration, nodes, threshold)
    anchor_variance = anchor_variance.to(device)
    q_cv, q_turn = calibration_motion_noise(gps, calibration_obs, int(calibration["delay_s"]))
    initial_time, initial_global = initial_acoustic_state(calibration_obs, calibration, nodes, threshold)
    initial = initial_global.to(device).clone(); initial[:3] -= center.to(device)
    ensemble_size = 48
    base = initial.repeat(ensemble_size, 1)
    base[:, :3] += torch.randn((ensemble_size, 3), dtype=torch.float64, device=device, generator=generator) * torch.sqrt(anchor_variance).unsqueeze(0)
    base[:, 3:6] += torch.randn((ensemble_size, 3), dtype=torch.float64, device=device, generator=generator) * 8.0
    frames = evaluation_frames(evaluation_obs, calibration, nodes, threshold)
    scales = torch.tensor(R_SCALE_GRID, dtype=torch.float64, device=device)
    candidate_count = len(scales)
    if method in ("bma", "pce", "apce"):
        analysis = base.unsqueeze(0).unsqueeze(0).repeat(candidate_count, 2, 1, 1)
        shadow = analysis.clone()
        mode_weights = torch.full((candidate_count, 2), 0.5, dtype=torch.float64, device=device)
        shadow_mode_weights = mode_weights.clone()
        candidate_logits = torch.zeros(candidate_count, dtype=torch.float64, device=device)
        candidate_weights = torch.ones(candidate_count, dtype=torch.float64, device=device) / candidate_count
    else:
        analysis = base.unsqueeze(0).repeat(2, 1, 1)
        mode_weights = torch.full((2,), 0.5, dtype=torch.float64, device=device)
        alpha_member = torch.full((ensemble_size,), 0.5, dtype=torch.float64, device=device)
    previous_time = initial_time
    records = []
    start = time.monotonic()
    for second, (frame, inlier_nodes, anchor_global) in sorted(frames.items()):
        dt = max(1.0, float(second - previous_time))
        anchor = anchor_global.to(device) - center.to(device)
        quality = "high" if len(inlier_nodes) >= 5 else "weak_anchor" if len(inlier_nodes) == 4 else "weak_angle"
        node_ids = inlier_nodes
        quality_multiplier = 1.0 if quality == "high" else 4.0 if quality == "weak_anchor" else 25.0
        observation, base_variance = observation_tensor(frame, node_ids, device, quality_multiplier)
        node_subset = {node: centered_nodes[node] for node in node_ids}
        noise = torch.randn((ensemble_size, 3), dtype=torch.float64, device=device, generator=generator)
        analysis_update = quality == "high"
        weak_update = quality != "high"
        if method in ("bma", "pce", "apce"):
            for candidate in range(candidate_count):
                analysis[candidate], mode_weights[candidate] = imm_propagate(analysis[candidate], mode_weights[candidate], dt, noise, q_cv, q_turn, contract["range_limit_m"], contract["speed_limit_mps"])
                shadow[candidate], shadow_mode_weights[candidate] = imm_propagate(shadow[candidate], shadow_mode_weights[candidate], dt, noise, q_cv, q_turn, contract["range_limit_m"], contract["speed_limit_mps"])
            if quality == "high":
                # Shadow mode probabilities are updated only from their own
                # predictive likelihood, never from analysis state correction.
                for candidate in range(candidate_count):
                    shadow_mode_weights[candidate] = torch.softmax(shadow_mode_weights[candidate].clamp_min(1e-12).log() + mode_likelihood(shadow[candidate], observation, base_variance * scales[candidate], node_subset, node_ids), dim=0)
                candidate_logits, candidate_weights = evidence_update(method, shadow, shadow_mode_weights, observation, base_variance, node_subset, node_ids, scales, candidate_logits)
            for candidate in range(candidate_count):
                analysis[candidate], mode_weights[candidate] = update_branch(analysis[candidate], mode_weights[candidate], observation, base_variance, anchor if quality != "weak_angle" else None, anchor_variance if quality != "weak_angle" else None, node_subset, node_ids, float(scales[candidate]), quality)
            samples, weights = flatten_branch(analysis, mode_weights, candidate_weights)
            scale_estimate = float((candidate_weights * scales).sum())
            alpha_estimate = float((candidate_weights * (torch.log(scales / 0.5) / math.log(8.0))).sum())
            entropy = float(-(candidate_weights.clamp_min(1e-12) * candidate_weights.clamp_min(1e-12).log()).sum())
        else:
            analysis, mode_weights = imm_propagate(analysis, mode_weights, dt, noise, q_cv, q_turn, contract["range_limit_m"], contract["speed_limit_mps"])
            if method == "aug_enkf":
                alpha_member = (alpha_member + 0.015 * torch.randn((ensemble_size,), dtype=torch.float64, device=device, generator=generator)).clamp(0.0, 1.0)
                r_scale = float(torch.exp(math.log(0.5) + alpha_member.mean() * math.log(8.0)))
            else:
                r_scale = 1.0
            analysis, mode_weights = update_branch(analysis, mode_weights, observation, base_variance, anchor if quality != "weak_angle" else None, anchor_variance if quality != "weak_angle" else None, node_subset, node_ids, r_scale, quality)
            samples, weights = flatten_branch(analysis, mode_weights)
            scale_estimate = r_scale
            alpha_estimate = 0.5 if method == "denkf" else float(alpha_member.mean())
            entropy = float(-(mode_weights.clamp_min(1e-12) * mode_weights.clamp_min(1e-12).log()).sum())
        mean = (weights[:, None] * samples).sum(dim=0)
        lower = torch.stack([weighted_quantile(samples[:, dim], weights, 0.05) for dim in range(3)])
        upper = torch.stack([weighted_quantile(samples[:, dim], weights, 0.95) for dim in range(3)])
        truth_global = truth_at(gps, second, int(calibration["delay_s"]))
        truth_centered = truth_global.to(device) - center.to(device) if truth_global is not None else None
        mean_global = mean.clone(); mean_global[:3] += center.to(device)
        row = {
            "time_s": second, "method": method, "seed": seed, "quality": quality, "inlier_nodes": len(inlier_nodes),
            "analysis_update": analysis_update, "weak_update": weak_update, "evidence_update": quality == "high" and method in ("bma", "pce", "apce"),
            "anchor_x": float(anchor_global[0]), "anchor_y": float(anchor_global[1]), "anchor_z": float(anchor_global[2]),
            "px": float(mean_global[0]), "py": float(mean_global[1]), "pz": float(mean_global[2]), "vx": float(mean[3]), "vy": float(mean[4]), "vz": float(mean[5]),
            "reliability_scale_estimate": scale_estimate, "alpha_estimate": alpha_estimate, "evidence_entropy": entropy,
        }
        for dim, name in enumerate(("px", "py", "pz")):
            row[f"{name}_lo"] = float(lower[dim] + center.to(device)[dim]); row[f"{name}_hi"] = float(upper[dim] + center.to(device)[dim])
        if truth_centered is not None:
            error = mean[:3] - truth_centered
            row.update({"truth_x": float(truth_global[0]), "truth_y": float(truth_global[1]), "truth_z": float(truth_global[2]), "position_error_m": float(torch.linalg.vector_norm(error)), "east_abs_error_m": abs(float(error[0])), "north_abs_error_m": abs(float(error[1])), "up_abs_error_m": abs(float(error[2])), "crps_position_m": float(crps(samples[:, :3], weights, truth_centered).mean()), "coverage_90": float(((truth_centered >= lower) & (truth_centered <= upper)).double().mean()), "interval_width_m": float((upper-lower).mean())})
        records.append(row); previous_time = second
    payload = {"status": "valid" if records and all(math.isfinite(row["position_error_m"]) for row in records) else "invalid", "method": method, "seed": seed, "alpha_definition": "operational acoustic-observation reliability multiplier", "reliability_scale_grid": list(R_SCALE_GRID), "motion_models": {"model_0": "constant velocity", "model_1": "random acceleration / turning"}, "anchor_contract": contract, "q_cv": q_cv, "q_turn": q_turn, "runtime_s": time.monotonic()-start, "records": records, "runner_sha256": sha256(Path(__file__)), "source_provenance": provenance}
    write_json(output / "runs" / f"{method}_seed_{seed}.json", payload)
    return payload


def aggregate(output: Path) -> dict:
    rows = []
    for path in sorted((output / "runs").glob("*_seed_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8")); records = payload["records"]; errors = [row["position_error_m"] for row in records]
        rows.append({"method": payload["method"], "seed": payload["seed"], "status": payload["status"], "frames": len(records), "position_rmse_m": math.sqrt(sum(value*value for value in errors)/len(errors)), "position_median_error_m": statistics.median(errors), "position_p90_error_m": sorted(errors)[int(.9*len(errors))], "position_p99_error_m": sorted(errors)[int(.99*len(errors))], "crps_position_m": statistics.mean(row["crps_position_m"] for row in records), "coverage_90": statistics.mean(row["coverage_90"] for row in records), "interval_width_m": statistics.mean(row["interval_width_m"] for row in records), "reliability_scale_mean": statistics.mean(row["reliability_scale_estimate"] for row in records), "runtime_s": payload["runtime_s"]})
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    method_rows=[]
    for method in METHODS:
        selected=[row for row in rows if row['method']==method]
        method_rows.append({"method":method,"runs":len(selected),"position_rmse_m_mean":statistics.mean(row['position_rmse_m'] for row in selected),"position_rmse_median":statistics.median(row['position_rmse_m'] for row in selected),"position_p99_error_m_mean":statistics.mean(row['position_p99_error_m'] for row in selected),"crps_position_m_mean":statistics.mean(row['crps_position_m'] for row in selected),"coverage_90_mean":statistics.mean(row['coverage_90'] for row in selected),"interval_width_m_mean":statistics.mean(row['interval_width_m'] for row in selected),"reliability_scale_mean":statistics.mean(row['reliability_scale_mean'] for row in selected),"runtime_s_mean":statistics.mean(row['runtime_s'] for row in selected)})
    with (output / "method_summary.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(method_rows[0]));writer.writeheader();writer.writerows(method_rows)
    manifest={"valid_runs":sum(row['status']=='valid' for row in rows),"expected_runs":25,"methods":METHODS,"seeds_are_numerical_sensitivity_not_independent_experiments":True,"formal_admission":False};write_json(output/'tracking_manifest.json',manifest);return manifest


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--remote-root',type=Path,default=Path('<PRIVATE_DATA_ROOT>/2017保定实验/2017保定实验'));parser.add_argument('--admission',type=Path,required=True);parser.add_argument('--frontend',type=Path,required=True);parser.add_argument('--raw-admission',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--method',choices=METHODS);parser.add_argument('--seed',type=int);parser.add_argument('--device',default='cuda:2');parser.add_argument('--aggregate',action='store_true');args=parser.parse_args()
    if args.aggregate: print(json.dumps(aggregate(args.output),ensure_ascii=False,indent=2));return
    if args.method is None or args.seed is None: raise SystemExit('--method and --seed are required')
    raw_gate=json.loads((args.raw_admission/'raw_frontend_gate.json').read_text(encoding='utf-8'))
    if not raw_gate['raw_wav_frontend_admitted']: raise RuntimeError('raw frontend admission failed')
    frontend_cal=json.loads((args.frontend/'frontend_calibration.json').read_text(encoding='utf-8'))
    loaded=load_frontend_observations(args.frontend/'observations.csv'); cal_obs=loaded['danyuan_panxuan_2']; eval_obs=loaded['danyuan_panxuan_3']
    eligibility=set(int(node) for node in raw_gate['eligible_nodes']); calibration={'delay_s':int(frontend_cal['selected_delay_s']),'nodes':{}}
    for node,metric in raw_gate['node_calibration'].items():
        node=int(node);calibration['nodes'][node]={'azimuth_sign':1.0,'azimuth_offset_deg':0.0,'elevation_offset_deg':0.0,'observation_sigma_deg':max(2.0,float(metric['joint_median_error_deg'])),'eligible':node in eligibility}
    archive=args.remote_root/'20171107保定实验'; node_raw=parse_nod(archive/'GPS_data/20171107baoding.nod');nodes={node:torch.tensor((value['x'],value['y'],value['z']),dtype=torch.float64) for node,value in node_raw.items()};gps={time_s:torch.tensor((x,y,z),dtype=torch.float64) for time_s,x,y,z in parse_gps(archive/'GPS_data/GPS1_plane1.gps')};provenance={'observation_source':'corrected direct-WAV MUSIC frontend','frontend_manifest_sha256':sha256(args.frontend/'frontend_manifest.json'),'frontend_calibration_sha256':sha256(args.frontend/'frontend_calibration.json'),'raw_admission_sha256':sha256(args.raw_admission/'raw_frontend_gate.json')}
    run_method(args.method,args.seed,nodes,cal_obs,eval_obs,gps,calibration,float(raw_gate['selected_ransac_threshold_deg']),args.output,args.device,provenance)


if __name__=='__main__': main()
