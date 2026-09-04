#!/usr/bin/env python3
"""Direct Cartesian PCE/APCE smoke runner for the admitted single-source bundle.

State is ``[E, N, U, vE, vN, vU]``.  The observation operator is the linear
``H=[I_3, 0]`` and every valid frame supplies its full ``R_tri``.  GPS is read
only after filtering to compute offline errors and coverage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import torch

try:
    from . import ALPHA_GRID
except ImportError:
    try:
        from __init__ import ALPHA_GRID
    except ImportError:
        # Allow the runner to be executed as a standalone remote script.
        ALPHA_GRID = (0.08, 0.22, 0.36, 0.50, 0.64, 0.78, 0.92)


STATE_NAMES = ("px", "py", "pz", "vx", "vy", "vz")
POS_NAMES = STATE_NAMES[:3]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_observations(path: Path, segment: str) -> dict[float, tuple[torch.Tensor, torch.Tensor, int, bool]]:
    grouped: dict[float, list[dict[str, str]]] = {}
    with path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("segment") != segment or row.get("valid", "False").lower() != "true":
                continue
            grouped.setdefault(float(row["time_s"]), []).append(row)
    output = {}
    for time_s, rows in grouped.items():
        rows.sort(key=lambda row: int(row["source_frame_index"]))
        observation = torch.tensor([[float(row[name]) for name in ("y_E", "y_N", "y_U")] for row in rows], dtype=torch.float64)
        covariance = torch.tensor(
            [[[float(row[f"R_{i}{j}"]) for j in range(3)] for i in range(3)] for row in rows],
            dtype=torch.float64,
        ).mean(dim=0)
        # The frontend writes one triangulated Cartesian point per time.  If a
        # malformed duplicate appears, require all copies to agree closely and
        # retain their average rather than silently selecting one.
        spread = float(torch.linalg.vector_norm(observation - observation.mean(dim=0), dim=1).max()) if len(rows) > 1 else 0.0
        if spread > 1e-6:
            raise RuntimeError(f"duplicate Cartesian observations disagree at {time_s}: {spread}")
        update_flags = {
            row.get("observation_update", "True").lower() == "true"
            for row in rows
        }
        if len(update_flags) != 1:
            raise RuntimeError(f"duplicate Cartesian observations disagree on update gating at {time_s}")
        output[time_s] = (
            observation.mean(dim=0), covariance, int(rows[0]["available_nodes"]),
            update_flags.pop(),
        )
    return dict(sorted(output.items()))


def load_truth(path: Path) -> dict[float, torch.Tensor]:
    output = {}
    if not path.exists():
        return output
    with path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            output[float(row["time_s"])] = torch.tensor([float(row[name]) for name in ("px", "py", "pz")], dtype=torch.float64)
    return output


def nearest_truth(truth: dict[float, torch.Tensor], time_s: float) -> torch.Tensor | None:
    if not truth:
        return None
    key = min(truth, key=lambda value: abs(value - time_s))
    return truth[key] if abs(key - time_s) <= 2.0 else None


def propagate(
    states: torch.Tensor,
    alpha: torch.Tensor | float,
    dt: float,
    noise: torch.Tensor,
    q_min: float,
    q_max: float,
    turn_rate_radps: float = 0.0,
) -> torch.Tensor:
    alpha_tensor = torch.as_tensor(alpha, dtype=states.dtype, device=states.device)
    q = q_min * torch.pow(torch.as_tensor(q_max / q_min, dtype=states.dtype, device=states.device), alpha_tensor)
    while q.ndim < states.ndim - 1:
        q = q.unsqueeze(-1)
    acceleration = noise * q.unsqueeze(-1)
    output = states.clone()
    omega = float(turn_rate_radps)
    if abs(omega) < 1e-12:
        output[..., :3] = output[..., :3] + output[..., 3:] * dt
    else:
        angle = omega * dt
        sine, cosine = math.sin(angle), math.cos(angle)
        vx, vy = output[..., 3], output[..., 4]
        output[..., 0] = output[..., 0] + (sine * vx + (cosine - 1.0) * vy) / omega
        output[..., 1] = output[..., 1] + ((1.0 - cosine) * vx + sine * vy) / omega
        output[..., 3] = cosine * vx - sine * vy
        output[..., 4] = sine * vx + cosine * vy
        output[..., 2] = output[..., 2] + output[..., 5] * dt
    output[..., :3] = output[..., :3] + 0.5 * acceleration * dt * dt
    output[..., 3:] = output[..., 3:] + acceleration * dt
    speed = torch.linalg.vector_norm(output[..., 3:], dim=-1, keepdim=True).clamp_min(1e-9)
    output[..., 3:] = output[..., 3:] * torch.clamp(180.0 / speed, max=1.0)
    return output


def cartesian_update(states: torch.Tensor, observation: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
    forecast = states[..., :3]
    state_mean = states.mean(dim=-2)
    observation_mean = forecast.mean(dim=-2)
    state_anomaly = states - state_mean.unsqueeze(-2)
    observation_anomaly = forecast - observation_mean.unsqueeze(-2)
    denominator = max(states.shape[-2] - 1, 1)
    cross = state_anomaly.transpose(-2, -1) @ observation_anomaly / denominator
    innovation_covariance = observation_anomaly.transpose(-2, -1) @ observation_anomaly / denominator + covariance
    innovation_covariance = 0.5 * (innovation_covariance + innovation_covariance.transpose(-2, -1))
    jitter = torch.eye(3, dtype=states.dtype, device=states.device) * 1e-6
    gain = torch.linalg.solve((innovation_covariance + jitter).transpose(-2, -1), cross.transpose(-2, -1)).transpose(-2, -1)
    new_mean = state_mean + torch.einsum("...ij,...j->...i", gain, observation - observation_mean)
    new_anomaly = state_anomaly - 0.5 * torch.einsum("...nj,...ij->...ni", observation_anomaly, gain)
    return new_mean.unsqueeze(-2) + new_anomaly


def branch_evidence(
    forecast: torch.Tensor,
    observation: torch.Tensor,
    covariance: torch.Tensor,
    method: str,
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    means = forecast.mean(dim=1)
    anomalies = forecast - means.unsqueeze(1)
    predictive = anomalies.transpose(-2, -1) @ anomalies / max(forecast.shape[1] - 1, 1) + covariance
    predictive = 0.5 * (predictive + predictive.transpose(-2, -1))
    residual = observation.unsqueeze(0) - means
    scores = []
    if method == "apce":
        between = means.var(dim=0, unbiased=False)
        dimension_weight = 0.35 + 0.65 * between / between.mean().clamp_min(1e-12)
        dimension_weight = dimension_weight / dimension_weight.mean()
    else:
        dimension_weight = torch.ones(3, dtype=forecast.dtype, device=forecast.device)
    for branch in range(forecast.shape[0]):
        if method == "apce":
            diagonal = predictive[branch].diagonal().clamp_min(1e-6)
            score = -0.5 * (dimension_weight * residual[branch].square() / diagonal + diagonal.log()).sum()
        else:
            system = predictive[branch] + torch.eye(3, dtype=forecast.dtype, device=forecast.device) * 1e-6
            solved = torch.linalg.solve(system, residual[branch])
            sign, logdet = torch.linalg.slogdet(system)
            score = -0.5 * (residual[branch] * solved).sum() - 0.5 * logdet if float(sign) > 0.0 else torch.tensor(-1e12, dtype=forecast.dtype, device=forecast.device)
        scores.append(score)
    score_tensor = torch.stack(scores)
    if method == "bma":
        logits = logits + score_tensor
    elif method == "pce":
        logits = logits + score_tensor / 0.66
    else:
        logits = 0.975 * logits + score_tensor / 0.58
    weights = torch.softmax(logits - logits.max(), dim=0)
    if method == "apce":
        uniform = torch.ones_like(weights) / len(weights)
        for _ in range(32):
            entropy = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum()
            if float(entropy) >= 0.34:
                break
            weights = 0.5 * (weights + uniform)
        weights = weights / weights.sum()
    return logits, weights


def weighted_quantile(values: torch.Tensor, weights: torch.Tensor, quantile: float) -> torch.Tensor:
    order = torch.argsort(values)
    sorted_values, sorted_weights = values[order], weights[order]
    cumulative = torch.cumsum(sorted_weights, dim=0)
    index = torch.searchsorted(cumulative, torch.as_tensor(quantile, dtype=weights.dtype, device=weights.device)).clamp(max=len(values) - 1)
    return sorted_values[int(index)]


def crps(samples: torch.Tensor, weights: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    first = (weights[:, None] * (samples - truth).abs()).sum(dim=0)
    pairwise = (weights[:, None, None] * weights[None, :, None] * (samples[:, None] - samples[None, :]).abs()).sum(dim=(0, 1))
    return first - 0.5 * pairwise


def clip_innovation(
    states: torch.Tensor,
    observation: torch.Tensor,
    covariance: torch.Tensor,
    chi2_threshold: float | None,
) -> tuple[torch.Tensor, float, float]:
    if chi2_threshold is None or chi2_threshold <= 0.0:
        return observation, 0.0, 1.0
    flattened = states.reshape(-1, states.shape[-1])
    forecast = flattened[:, :3]
    mean = forecast.mean(dim=0)
    anomaly = forecast - mean
    predictive = anomaly.transpose(0, 1) @ anomaly / max(len(forecast) - 1, 1) + covariance
    predictive = 0.5 * (predictive + predictive.transpose(0, 1))
    residual = observation - mean
    jitter = torch.eye(3, dtype=states.dtype, device=states.device) * 1e-6
    solved = torch.linalg.solve(predictive + jitter, residual)
    nis = float((residual * solved).sum().clamp_min(0.0))
    clip_factor = min(1.0, math.sqrt(float(chi2_threshold) / max(nis, 1e-12)))
    return mean + residual * clip_factor, nis, clip_factor


def run_track(
    frontend: Path,
    output: Path,
    method: str,
    seed: int,
    device_name: str,
    segment: str,
    ensemble_size: int,
    q_min: float,
    q_max: float,
    position_init_std: float,
    velocity_init_std: float,
    observation_covariance_scale: float,
    turn_rate_radps: float,
    innovation_chi2_threshold: float | None = None,
) -> dict:
    if device_name in {"cuda:0", "cuda:1"}:
        raise ValueError("project GPU policy permits only cuda:2 and cuda:3")
    device = torch.device(device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    observations = load_observations(frontend / "observations_cartesian.csv", segment)
    truth = load_truth(frontend / "gps_truth.csv")
    if len(observations) < 3:
        raise RuntimeError("fewer than three valid Cartesian observations")
    generator = torch.Generator(device=device).manual_seed(seed)
    times = sorted(observations)
    trusted_times = [time_s for time_s in times if observations[time_s][3]]
    if len(trusted_times) < 3:
        raise RuntimeError("fewer than three update-admitted Cartesian observations")
    initial = [observations[t][0] for t in trusted_times[: min(10, len(trusted_times))]]
    # The filter starts at ``times[0]``.  Using a median over the first ten
    # seconds would place the state several seconds behind a moving target.
    initial_position = initial[0].to(device)
    velocities = [
        (observations[right][0] - observations[left][0]) / max(right - left, 1e-6)
        for left, right in zip(trusted_times[:9], trusted_times[1:10])
        if right > left
    ]
    initial_velocity = torch.stack(velocities).median(dim=0).values.to(device) if velocities else torch.zeros(3, dtype=torch.float64, device=device)
    state0 = torch.cat((initial_position, initial_velocity))
    init_noise = torch.cat((
        torch.randn((ensemble_size, 3), generator=generator, device=device, dtype=torch.float64) * position_init_std,
        torch.randn((ensemble_size, 3), generator=generator, device=device, dtype=torch.float64) * velocity_init_std,
    ), dim=1)
    single = state0.unsqueeze(0) + init_noise
    branches = single.unsqueeze(0).repeat(len(ALPHA_GRID), 1, 1)
    logits = torch.zeros(len(ALPHA_GRID), dtype=torch.float64, device=device)
    records = []
    previous_time = times[0]
    for time_s in times:
        observation, covariance, available_nodes, observation_update = observations[time_s]
        observation = observation.to(device)
        covariance = covariance.to(device) * observation_covariance_scale
        if time_s != times[0]:
            dt = max(0.1, min(3.0, time_s - previous_time))
            if method == "denkf":
                noise = torch.randn((ensemble_size, 3), generator=generator, device=device, dtype=torch.float64)
                single = propagate(single, 0.50, dt, noise, q_min, q_max, turn_rate_radps)
            elif method == "aug_enkf":
                noise = torch.randn((ensemble_size, 3), generator=generator, device=device, dtype=torch.float64)
                single = propagate(single, 0.50, dt, noise, q_min, q_max, turn_rate_radps)
            else:
                noise = torch.randn((len(ALPHA_GRID), ensemble_size, 3), generator=generator, device=device, dtype=torch.float64)
                branches = propagate(branches, torch.tensor(ALPHA_GRID, dtype=torch.float64, device=device), dt, noise, q_min, q_max, turn_rate_radps)
        robust_observation = observation
        innovation_nis = 0.0
        innovation_clip_factor = 1.0
        if observation_update:
            innovation_states = single if method in ("denkf", "aug_enkf") else branches
            robust_observation, innovation_nis, innovation_clip_factor = clip_innovation(
                innovation_states, observation, covariance, innovation_chi2_threshold
            )
        if method in ("denkf", "aug_enkf") and observation_update:
            single = cartesian_update(single, robust_observation, covariance)
            samples = single
            weights = torch.ones(ensemble_size, dtype=torch.float64, device=device) / ensemble_size
            alpha_estimate = torch.tensor(0.50, dtype=torch.float64, device=device)
            evidence_entropy = 0.0
        elif method not in ("denkf", "aug_enkf") and observation_update:
            forecasts = branches[..., :3]
            logits, branch_weights = branch_evidence(forecasts, robust_observation, covariance, method, logits)
            for branch in range(len(ALPHA_GRID)):
                branches[branch] = cartesian_update(branches[branch], robust_observation, covariance)
            samples = branches.reshape(-1, 6)
            weights = branch_weights.repeat_interleave(ensemble_size) / ensemble_size
            alpha_tensor = torch.tensor(ALPHA_GRID, dtype=torch.float64, device=device)
            alpha_estimate = (branch_weights * alpha_tensor).sum()
            evidence_entropy = float(-(branch_weights.clamp_min(1e-12) * branch_weights.clamp_min(1e-12).log()).sum())
        elif method in ("denkf", "aug_enkf"):
            samples = single
            weights = torch.ones(ensemble_size, dtype=torch.float64, device=device) / ensemble_size
            alpha_estimate = torch.tensor(0.50, dtype=torch.float64, device=device)
            evidence_entropy = 0.0
        else:
            branch_weights = torch.softmax(logits - logits.max(), dim=0)
            samples = branches.reshape(-1, 6)
            weights = branch_weights.repeat_interleave(ensemble_size) / ensemble_size
            alpha_tensor = torch.tensor(ALPHA_GRID, dtype=torch.float64, device=device)
            alpha_estimate = (branch_weights * alpha_tensor).sum()
            evidence_entropy = float(-(branch_weights.clamp_min(1e-12) * branch_weights.clamp_min(1e-12).log()).sum())
        mean = (samples * weights[:, None]).sum(dim=0)
        lower = torch.stack([weighted_quantile(samples[:, dim], weights, 0.05) for dim in range(6)])
        upper = torch.stack([weighted_quantile(samples[:, dim], weights, 0.95) for dim in range(6)])
        truth_position = nearest_truth(truth, time_s)
        if truth_position is not None:
            truth_position = truth_position.to(device)
            position_error = float(torch.linalg.vector_norm(mean[:3] - truth_position))
            crps_position = float(crps(samples[:, :3], weights, truth_position).mean())
            coverage = float(((truth_position >= lower[:3]) & (truth_position <= upper[:3])).double().mean())
        else:
            position_error = crps_position = coverage = None
        row = {
            "time_s": time_s, "method": method, "seed": seed, "valid_nodes": available_nodes,
            "observation_update": observation_update,
            "innovation_nis": innovation_nis,
            "innovation_clip_factor": innovation_clip_factor,
            **{name: float(mean[index]) for index, name in enumerate(STATE_NAMES)},
            "alpha_estimate": float(alpha_estimate), "evidence_entropy": evidence_entropy,
            "position_error_m": position_error, "crps_position_m": crps_position,
            "coverage_90": coverage, "interval_width_m": float((upper[:3] - lower[:3]).mean()),
        }
        records.append(row)
        previous_time = time_s
    payload = {
        "status": "valid", "method": method, "seed": seed, "segment": segment, "state_definition": "[E,N,U,vE,vN,vU]",
        "observation_operator": "H=[I3,0]", "source_frontend": str(frontend),
        "source_frontend_sha256": sha256(frontend / "frontend_manifest.json"),
        "runner_sha256": sha256(Path(__file__)),
        "gps_role": "offline evaluation only; no GPS in initialization, propagation, update, or branch scores",
        "ensemble_size": ensemble_size, "alpha_grid": list(ALPHA_GRID), "q_min_accel_mps2": q_min, "q_max_accel_mps2": q_max,
        "position_init_std_m": position_init_std, "velocity_init_std_mps": velocity_init_std,
        "observation_covariance_scale": observation_covariance_scale,
        "turn_rate_radps": turn_rate_radps,
        "innovation_chi2_threshold": innovation_chi2_threshold,
        "records": records,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / f"{method}_seed_{seed}.json", payload)
    return payload


def aggregate(output: Path, frontend: Path, segment: str) -> None:
    summaries = []
    for path in sorted(output.glob("*_seed_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = [row for row in payload.get("records", []) if row.get("position_error_m") is not None]
        if not rows:
            continue
        errors = [float(row["position_error_m"]) for row in rows]
        summaries.append({
            "method": payload["method"], "seed": payload["seed"], "frames": len(rows),
            "position_rmse_m": math.sqrt(sum(value * value for value in errors) / len(errors)),
            "position_median_error_m": statistics.median(errors),
            "position_p90_error_m": sorted(errors)[min(len(errors) - 1, int(0.90 * len(errors)))],
            "crps_position_m": statistics.mean(float(row["crps_position_m"]) for row in rows),
            "coverage_90": statistics.mean(float(row["coverage_90"]) for row in rows),
            "interval_width_m": statistics.mean(float(row["interval_width_m"]) for row in rows),
        })
    observation = load_observations(frontend / "observations_cartesian.csv", segment)
    truth = load_truth(frontend / "gps_truth.csv")
    acoustic_errors = [
        float(torch.linalg.vector_norm(observation[time_s][0] - truth_point))
        for time_s in observation
        if observation[time_s][3]
        if (truth_point := nearest_truth(truth, time_s)) is not None
    ]
    if acoustic_errors:
        summaries.append({
            "method": "acoustic_triangulation", "seed": "fixed", "frames": len(acoustic_errors),
            "position_rmse_m": math.sqrt(sum(value * value for value in acoustic_errors) / len(acoustic_errors)),
            "position_median_error_m": statistics.median(acoustic_errors),
            "position_p90_error_m": sorted(acoustic_errors)[min(len(acoustic_errors) - 1, int(0.90 * len(acoustic_errors)))],
            "crps_position_m": None, "coverage_90": None, "interval_width_m": None,
        })
    with (output / "method_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = list(summaries[0]) if summaries else ["method"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(summaries)
    write_json(output / "aggregate_manifest.json", {
        "source_frontend": str(frontend), "segment": segment, "source_frontend_sha256": sha256(frontend / "frontend_manifest.json"),
        "gps_role": "offline evaluation only", "summaries": summaries,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("track", "aggregate"), required=True)
    parser.add_argument("--frontend", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segment", default="danyuan_panxuan_3")
    parser.add_argument("--method", choices=("denkf", "aug_enkf", "bma", "pce", "apce"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--ensemble-size", type=int, default=48)
    parser.add_argument("--q-min", type=float, default=0.5)
    parser.add_argument("--q-max", type=float, default=12.0)
    parser.add_argument("--position-init-std", type=float, default=100.0)
    parser.add_argument("--velocity-init-std", type=float, default=20.0)
    parser.add_argument("--observation-covariance-scale", type=float, default=1.0)
    parser.add_argument("--turn-rate-radps", type=float, default=0.0)
    parser.add_argument("--innovation-chi2-threshold", type=float)
    args = parser.parse_args()
    if args.stage == "track":
        if args.method is None or args.seed is None:
            raise SystemExit("--method and --seed are required for track")
        run_track(args.frontend, args.output / "runs", args.method, args.seed, args.device, args.segment, args.ensemble_size, args.q_min, args.q_max, args.position_init_std, args.velocity_init_std, args.observation_covariance_scale, args.turn_rate_radps, args.innovation_chi2_threshold)
    else:
        aggregate(args.output / "runs", args.frontend, args.segment)


if __name__ == "__main__":
    main()
