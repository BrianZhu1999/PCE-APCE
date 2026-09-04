from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_benchmark_v3 as v3
import run_benchmark_v4 as v4
from run_hybrid_wave import Config, initial_state, make_config, propagate_batch, smooth_noise, source_terms


def configure_style() -> None:
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


def save_figure(figure: plt.Figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for record in records for key in record})
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def bootstrap_ci(values: np.ndarray, seed: int, n_bootstrap: int = 4000) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(n_bootstrap, values.size))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> np.ndarray:
    order = np.argsort(values, axis=0)
    sorted_values = np.take_along_axis(values, order, axis=0)
    broadcast_weights = np.broadcast_to(weights[:, None], values.shape)
    sorted_weights = np.take_along_axis(broadcast_weights, order, axis=0)
    cumulative = np.cumsum(sorted_weights, axis=0)
    indices = np.argmax(cumulative >= probability, axis=0)
    return sorted_values[indices, np.arange(values.shape[1])]


def weighted_crps(samples: np.ndarray, weights: np.ndarray, truth: np.ndarray) -> np.ndarray:
    first = np.sum(weights[:, None] * np.abs(samples - truth[None, :]), axis=0)
    pairwise = np.abs(samples[:, None, :] - samples[None, :, :])
    second = 0.5 * np.sum(weights[:, None, None] * weights[None, :, None] * pairwise, axis=(0, 1))
    return first - second


def run_calibration(n_seeds: int, base_seed: int, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    levels = [0.50, 0.70, 0.80, 0.90, 0.95]
    records: list[dict[str, Any]] = []
    crps_records: list[dict[str, Any]] = []
    for seed_index in range(n_seeds):
        cfg = replace(make_config("quick"), seed=base_seed + seed_index, filter_variant="lr")
        scenario = v3.generate_scenario(cfg)
        for method in ("A6_pce", "A7_apce"):
            result = v4.run_ablation(scenario, method, return_details=True)
            branches = np.asarray(result["final_branches"])
            branch_weights = np.asarray(result["final_weights"])
            samples = branches[:, :, : cfg.nx].reshape(-1, cfg.nx)
            weights = np.repeat(branch_weights / cfg.ensemble_size, cfg.ensemble_size)
            truth = np.asarray(result["final_truth"])[: cfg.nx]
            crps_value = float(np.mean(weighted_crps(samples, weights, truth)))
            crps_records.append({"seed": cfg.seed, "method": method, "crps": crps_value})
            for level in levels:
                tail = 1.0 - level
                lower = weighted_quantile(samples, weights, tail / 2.0)
                upper = weighted_quantile(samples, weights, 1.0 - tail / 2.0)
                covered = (truth >= lower) & (truth <= upper)
                width = upper - lower
                interval_score = width.copy()
                interval_score += (2.0 / tail) * (lower - truth) * (truth < lower)
                interval_score += (2.0 / tail) * (truth - upper) * (truth > upper)
                records.append(
                    {
                        "seed": cfg.seed,
                        "method": method,
                        "nominal_coverage": level,
                        "picp": float(np.mean(covered)),
                        "mpiw": float(np.mean(width)),
                        "interval_score": float(np.mean(interval_score)),
                        "crps": crps_value,
                    }
                )
        print(f"[calibration {seed_index + 1}/{n_seeds}] seed={cfg.seed}")

    write_csv(records, output_dir / "calibration_runs.csv")
    summary: list[dict[str, Any]] = []
    for method in ("A6_pce", "A7_apce"):
        for level in levels:
            subset = [row for row in records if row["method"] == method and row["nominal_coverage"] == level]
            summary.append(
                {
                    "method": method,
                    "nominal_coverage": level,
                    "picp": float(np.mean([row["picp"] for row in subset])),
                    "mpiw": float(np.mean([row["mpiw"] for row in subset])),
                    "interval_score": float(np.mean([row["interval_score"] for row in subset])),
                    "crps": float(np.mean([row["crps"] for row in subset])),
                }
            )
    payload = {"n_seeds": n_seeds, "levels": levels, "summary": summary}
    (output_dir / "calibration_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    configure_style()
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))
    colors = {"A6_pce": "#4C78A8", "A7_apce": "#E09F3E"}
    labels = {"A6_pce": "PCE", "A7_apce": "APCE"}
    axes[0].plot(levels, levels, color="#777777", linestyle="--", linewidth=1.0, label="Ideal")
    for method in ("A6_pce", "A7_apce"):
        subset = [row for row in summary if row["method"] == method]
        axes[0].plot(levels, [row["picp"] for row in subset], marker="o", color=colors[method], label=labels[method])
        axes[1].plot(levels, [row["mpiw"] for row in subset], marker="s", color=colors[method], label=labels[method])
        axes[2].plot(levels, [row["interval_score"] for row in subset], marker="^", color=colors[method], label=labels[method])
    axes[0].set(xlabel="Nominal coverage", ylabel="Empirical coverage", title="a  Reliability")
    axes[1].set(xlabel="Nominal coverage", ylabel="Mean interval width", title="b  Sharpness")
    axes[2].set(xlabel="Nominal coverage", ylabel="Interval score", title="c  Proper score")
    axes[0].legend()
    for axis in axes:
        axis.set_title(axis.get_title(), loc="left", fontweight="bold")
    figure.tight_layout(w_pad=1.5)
    save_figure(figure, output_dir / "figure_calibration")
    plt.close(figure)
    return payload


def custom_source_terms(x: np.ndarray, t: float, theta: float, mismatch: str) -> np.ndarray:
    if mismatch != "source_shape":
        return source_terms(x, t, theta)
    base_profile = np.exp(-((x - 0.18) / 0.075) ** 2)
    epistemic_profile = np.exp(-((x - 0.73) / 0.12) ** 2)
    base_time = 1.10 * np.sin(2.0 * math.pi * 2.15 * t + 0.10) * np.exp(-1.15 * t)
    epistemic_time = np.sin(2.0 * math.pi * 1.10 * t + 0.50) * np.exp(-0.25 * t)
    return base_time * base_profile + theta * epistemic_time * epistemic_profile


def propagate_truth_custom(
    states: np.ndarray,
    theta: float,
    t: float,
    cfg: Config,
    noise: np.ndarray,
    mismatch: str,
) -> np.ndarray:
    nx = cfg.nx
    dx = cfg.length / (nx - 1)
    u = states[:, :nx]
    velocity = states[:, nx:]
    laplacian = np.zeros_like(u)
    laplacian[:, 1:-1] = (u[:, 2:] - 2.0 * u[:, 1:-1] + u[:, :-2]) / (dx * dx)
    x = np.linspace(0.0, cfg.length, nx)
    forcing = custom_source_terms(x, t, theta, mismatch)[None, :]
    acceleration = cfg.wave_speed**2 * laplacian - cfg.damping * velocity + forcing
    if mismatch == "non_gaussian_process":
        process_draw = np.sign(noise) * np.log1p(np.abs(noise)) * math.sqrt(2.0)
    else:
        process_draw = noise
    velocity_new = velocity + cfg.dt * acceleration + cfg.process_noise * math.sqrt(cfg.dt) * process_draw
    u_new = u + cfg.dt * velocity_new
    if mismatch == "boundary":
        u_new[:, 0] = u_new[:, 1]
        u_new[:, -1] = u_new[:, -2]
        velocity_new[:, 0] = velocity_new[:, 1]
        velocity_new[:, -1] = velocity_new[:, -2]
    else:
        u_new[:, [0, -1]] = 0.0
        velocity_new[:, [0, -1]] = 0.0
    return np.concatenate([u_new, velocity_new], axis=1)


def generate_mismatch_scenario(cfg: Config, mismatch: str) -> v3.Scenario:
    base = v3.generate_scenario(cfg)
    rng = np.random.default_rng(cfg.seed + 90_000)
    n_steps = int(round(cfg.t_end / cfg.dt))
    times = np.arange(n_steps + 1) * cfg.dt
    theta_true = v3.alpha_to_theta(cfg.alpha_true, cfg)

    if mismatch == "matched":
        return base
    if mismatch == "fine_grid":
        truth_cfg = replace(cfg, nx=81)
    elif mismatch == "wave_speed":
        truth_cfg = replace(cfg, wave_speed=1.08)
    elif mismatch == "damping":
        truth_cfg = replace(cfg, damping=0.10)
    else:
        truth_cfg = cfg

    fine_x = np.linspace(0.0, truth_cfg.length, truth_cfg.nx)
    coarse_x = np.linspace(0.0, cfg.length, cfg.nx)
    truth_state = initial_state(fine_x)[None, :]
    if mismatch == "boundary":
        truth_state[:, 0] = truth_state[:, 1]
        truth_state[:, truth_cfg.nx - 1] = truth_state[:, truth_cfg.nx - 2]
    truth_coarse = np.zeros((n_steps + 1, 2 * cfg.nx), dtype=float)

    def map_to_coarse(state: np.ndarray) -> np.ndarray:
        if truth_cfg.nx == cfg.nx:
            return state[0].copy()
        displacement = np.interp(coarse_x, fine_x, state[0, : truth_cfg.nx])
        velocity = np.interp(coarse_x, fine_x, state[0, truth_cfg.nx :])
        return np.concatenate([displacement, velocity])

    truth_coarse[0] = map_to_coarse(truth_state)
    for step in range(1, n_steps + 1):
        draw = smooth_noise(rng.normal(size=(1, truth_cfg.nx)))
        if mismatch in {"fine_grid", "wave_speed", "damping"}:
            truth_state = propagate_batch(
                truth_state,
                theta_true,
                times[step - 1],
                truth_cfg,
                rng,
                stochastic=True,
                noise_draw=draw,
            )
        else:
            truth_state = propagate_truth_custom(
                truth_state, theta_true, times[step - 1], truth_cfg, draw, mismatch
            )
        truth_coarse[step] = map_to_coarse(truth_state)

    observations: dict[int, np.ndarray] = {}
    for step in range(cfg.obs_interval, n_steps + 1, cfg.obs_interval):
        if mismatch == "non_gaussian_observation":
            noise = cfg.obs_noise * rng.standard_t(df=3, size=base.observation_indices.size) / math.sqrt(3.0)
        else:
            noise = cfg.obs_noise * rng.normal(size=base.observation_indices.size)
        observations[step] = truth_coarse[step, base.observation_indices] + noise
    return replace(base, truth_states=truth_coarse, observations=observations, theta_true=theta_true)


def run_mismatch(n_seeds: int, base_seed: int, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mismatch_types = [
        "matched",
        "fine_grid",
        "wave_speed",
        "damping",
        "source_shape",
        "boundary",
        "non_gaussian_process",
        "non_gaussian_observation",
    ]
    methods = ["enkf", "ensf_lr", "A6_pce", "A7_apce"]
    records: list[dict[str, Any]] = []
    total = len(mismatch_types) * n_seeds
    count = 0
    for mismatch in mismatch_types:
        for seed_index in range(n_seeds):
            cfg = replace(make_config("quick"), seed=base_seed + seed_index, filter_variant="lr")
            scenario = generate_mismatch_scenario(cfg, mismatch)
            for method in methods:
                if method in {"enkf", "ensf_lr"}:
                    result = v3.run_method(scenario, method)["metrics"]
                else:
                    result = v4.run_ablation(scenario, method)
                records.append(
                    {
                        "mismatch": mismatch,
                        "seed": cfg.seed,
                        "method": method,
                        "mean_rmse": float(result["mean_rmse"]),
                        "final_rmse": float(result["final_rmse"]),
                    }
                )
            count += 1
            print(f"[mismatch {count}/{total}] condition={mismatch} seed={cfg.seed}")
    write_csv(records, output_dir / "mismatch_runs.csv")
    summary: list[dict[str, Any]] = []
    for mismatch in mismatch_types:
        for method in methods:
            values = np.asarray(
                [row["mean_rmse"] for row in records if row["mismatch"] == mismatch and row["method"] == method]
            )
            summary.append(
                {
                    "mismatch": mismatch,
                    "method": method,
                    "mean_rmse": float(values.mean()),
                    "ci95": bootstrap_ci(values, base_seed + len(summary)),
                }
            )
    payload = {"n_seeds": n_seeds, "conditions": mismatch_types, "summary": summary}
    (output_dir / "mismatch_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    configure_style()
    figure, axis = plt.subplots(figsize=(7.2, 3.0))
    colors = {"enkf": "#9A9A9A", "ensf_lr": "#7F8FA6", "A6_pce": "#4C78A8", "A7_apce": "#E09F3E"}
    labels = {"enkf": "EnKF", "ensf_lr": "EnSF-LR", "A6_pce": "PCE", "A7_apce": "APCE"}
    x = np.arange(len(mismatch_types))
    width = 0.19
    for offset, method in enumerate(methods):
        values = [next(row for row in summary if row["mismatch"] == condition and row["method"] == method)["mean_rmse"] for condition in mismatch_types]
        axis.bar(x + (offset - 1.5) * width, values, width=width, color=colors[method], label=labels[method])
    short_labels = ["Matched", "Fine/coarse", "$c$", "Damping", "Source", "Boundary", "Non-Gauss. Q", "Non-Gauss. R"]
    axis.set_xticks(x, short_labels, rotation=22, ha="right")
    axis.set_ylabel("Time-mean RMSE")
    axis.set_title("Robustness under truth-filter mismatch", loc="left", fontweight="bold")
    axis.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.17))
    figure.tight_layout()
    save_figure(figure, output_dir / "figure_model_mismatch")
    plt.close(figure)
    return payload


def gaspari_cohn(distance: np.ndarray, radius: float) -> np.ndarray:
    x = np.abs(distance) / max(radius, 1.0e-12)
    result = np.zeros_like(x)
    first = x <= 1.0
    second = (x > 1.0) & (x <= 2.0)
    xf = x[first]
    xs = x[second]
    result[first] = 1.0 - 5.0 / 3.0 * xf**2 + 5.0 / 8.0 * xf**3 + 0.5 * xf**4 - 0.25 * xf**5
    result[second] = 4.0 - 5.0 * xs + 5.0 / 3.0 * xs**2 + 5.0 / 8.0 * xs**3 - 0.5 * xs**4 + 1.0 / 12.0 * xs**5 - 2.0 / (3.0 * xs)
    return np.clip(result, 0.0, 1.0)


def localized_enkf_update(
    prior: np.ndarray,
    observation: np.ndarray,
    obs_idx: np.ndarray,
    obs_noise: float,
    inflation: float,
    radius: float,
    rng: np.random.Generator,
    nx: int,
) -> np.ndarray:
    mean = prior.mean(axis=0)
    inflated = mean + inflation * (prior - mean)
    predicted = inflated[:, obs_idx]
    x_anomaly = inflated - inflated.mean(axis=0)
    y_anomaly = predicted - predicted.mean(axis=0)
    denom = max(prior.shape[0] - 1, 1)
    covariance_xy = x_anomaly.T @ y_anomaly / denom
    state_positions = np.concatenate([np.linspace(0.0, 1.0, nx), np.linspace(0.0, 1.0, nx)])
    sensor_positions = state_positions[obs_idx]
    localization = gaspari_cohn(state_positions[:, None] - sensor_positions[None, :], radius)
    covariance_xy *= localization
    covariance_yy = y_anomaly.T @ y_anomaly / denom + (obs_noise**2 + 1.0e-7) * np.eye(observation.size)
    gain = covariance_xy @ np.linalg.pinv(covariance_yy)
    perturbed = observation[None, :] + obs_noise * rng.normal(size=predicted.shape)
    updated = inflated + (perturbed - predicted) @ gain.T
    updated[:, [0, nx - 1, nx, 2 * nx - 1]] = 0.0
    return updated


def run_tuned_enkf(scenario: v3.Scenario, candidate: dict[str, Any]) -> dict[str, float]:
    cfg = scenario.cfg
    rng = np.random.default_rng(cfg.seed + 701)
    ensemble = scenario.ensemble_initial.copy()
    rmse = np.zeros(scenario.times.size)
    rmse[0] = v3.evaluate_estimate(ensemble.mean(axis=0), scenario.truth_states[0], cfg.nx)
    for step in range(1, scenario.times.size):
        ensemble = propagate_batch(
            ensemble,
            0.0,
            scenario.times[step - 1],
            cfg,
            rng,
            stochastic=True,
            noise_draw=scenario.forecast_noise[step - 1],
        )
        if step in scenario.observations:
            ensemble = localized_enkf_update(
                ensemble,
                scenario.observations[step],
                scenario.observation_indices,
                cfg.obs_noise,
                candidate["inflation"],
                candidate["radius"],
                rng,
                cfg.nx,
            )
        rmse[step] = v3.evaluate_estimate(ensemble.mean(axis=0), scenario.truth_states[step], cfg.nx)
    return {"mean_rmse": float(rmse.mean()), "final_rmse": float(rmse[-1])}


def run_tuned_joint_enkf(scenario: v3.Scenario, candidate: dict[str, Any]) -> dict[str, float]:
    cfg = scenario.cfg
    rng = np.random.default_rng(cfg.seed + 907)
    ensemble = scenario.ensemble_initial.copy()
    theta_min, theta_max = float(np.min(scenario.theta_grid)), float(np.max(scenario.theta_grid))
    theta = np.linspace(theta_min, theta_max, cfg.ensemble_size)
    rng.shuffle(theta)
    rmse = np.zeros(scenario.times.size)
    rmse[0] = v3.evaluate_estimate(ensemble.mean(axis=0), scenario.truth_states[0], cfg.nx)
    for step in range(1, scenario.times.size):
        theta = np.clip(theta + candidate["random_walk"] * rng.normal(size=theta.size), theta_min, theta_max)
        ensemble = v3.propagate_memberwise_theta(
            ensemble, theta, scenario.times[step - 1], cfg, rng, scenario.forecast_noise[step - 1]
        )
        if step in scenario.observations:
            mean = ensemble.mean(axis=0)
            ensemble = mean + candidate["inflation"] * (ensemble - mean)
            ensemble, theta = v3.joint_parameter_enkf_update(
                ensemble,
                theta,
                scenario.observations[step],
                scenario.observation_indices,
                cfg.obs_noise,
                rng,
                (theta_min, theta_max),
            )
        rmse[step] = v3.evaluate_estimate(ensemble.mean(axis=0), scenario.truth_states[step], cfg.nx)
    return {"mean_rmse": float(rmse.mean()), "final_rmse": float(rmse[-1])}


def tuning_candidates() -> dict[str, list[dict[str, Any]]]:
    return {
        "enkf": [
            {"ensemble_size": ensemble, "inflation": inflation, "radius": radius}
            for ensemble, inflation, radius in [
                (18, 1.00, 0.18), (18, 1.05, 0.30), (18, 1.10, 0.45),
                (30, 1.00, 0.18), (30, 1.05, 0.30), (30, 1.10, 0.45),
            ]
        ],
        "ensf_direct": [
            {"ensemble_size": e, "guidance": g, "reverse_steps": r, "reverse_noise_scale": n}
            for e, g, r, n in [(18, 0.65, 6, 0.45), (18, 0.80, 8, 0.55), (18, 0.95, 12, 0.40), (30, 0.65, 6, 0.45), (30, 0.80, 8, 0.55), (30, 0.95, 12, 0.40)]
        ],
        "ensf_lr": [
            {"ensemble_size": e, "guidance": g, "reverse_steps": r, "reverse_noise_scale": n, "regression_ridge": ridge}
            for e, g, r, n, ridge in [(18, 0.65, 6, 0.45, 1e-6), (18, 0.80, 8, 0.55, 1e-5), (18, 0.95, 12, 0.40, 1e-4), (30, 0.65, 6, 0.45, 1e-6), (30, 0.80, 8, 0.55, 1e-5), (30, 0.95, 12, 0.40, 1e-4)]
        ],
        "joint_param_enkf": [
            {"ensemble_size": e, "inflation": inflation, "random_walk": walk}
            for e, inflation, walk in [(18, 1.00, 0.001), (18, 1.05, 0.003), (18, 1.10, 0.010), (30, 1.00, 0.001), (30, 1.05, 0.003), (30, 1.10, 0.010)]
        ],
        "pce": [
            {"ensemble_size": e, "temperature": temperature, "shrinkage": shrinkage, "window": window}
            for e, temperature, shrinkage, window in [(18, 0.35, 0.20, 1), (18, 0.50, 0.35, 2), (18, 0.65, 0.50, 3), (30, 0.35, 0.20, 1), (30, 0.50, 0.35, 2), (30, 0.65, 0.50, 3)]
        ],
    }


def run_candidate(method: str, candidate: dict[str, Any], seed: int) -> dict[str, float]:
    config_updates = {key: value for key, value in candidate.items() if key in Config.__dataclass_fields__}
    cfg = replace(make_config("quick"), seed=seed, filter_variant="lr", **config_updates)
    scenario = v3.generate_scenario(cfg)
    if method == "enkf":
        return run_tuned_enkf(scenario, candidate)
    if method == "joint_param_enkf":
        return run_tuned_joint_enkf(scenario, candidate)
    if method in {"ensf_direct", "ensf_lr"}:
        return v3.run_method(scenario, method)["metrics"]
    evidence = v3.AlphaEvidenceConfig(
        temperature=candidate["temperature"],
        shrinkage=candidate["shrinkage"],
        window=candidate["window"],
    )
    return v3.run_method(scenario, "alpha_ensf_lr_pce", evidence)["metrics"]


def run_fair_tuning(
    calibration_seeds: int,
    test_seeds: int,
    base_seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = tuning_candidates()
    tuning_records: list[dict[str, Any]] = []
    best: dict[str, dict[str, Any]] = {}
    for method, method_candidates in candidates.items():
        candidate_scores: list[float] = []
        for candidate_index, candidate in enumerate(method_candidates):
            values = []
            for seed_index in range(calibration_seeds):
                result = run_candidate(method, candidate, base_seed + seed_index)
                values.append(result["mean_rmse"])
                tuning_records.append(
                    {
                        "phase": "calibration",
                        "method": method,
                        "candidate": candidate_index,
                        "seed": base_seed + seed_index,
                        "mean_rmse": result["mean_rmse"],
                        "configuration": json.dumps(candidate, sort_keys=True),
                    }
                )
            candidate_scores.append(float(np.mean(values)))
            print(f"[tune] method={method} candidate={candidate_index} score={candidate_scores[-1]:.4e}")
        best_index = int(np.argmin(candidate_scores))
        best[method] = {"candidate": best_index, "configuration": method_candidates[best_index], "calibration_rmse": candidate_scores[best_index]}

    test_records: list[dict[str, Any]] = []
    test_base = base_seed + 1000
    for method, selection in best.items():
        for seed_index in range(test_seeds):
            seed = test_base + seed_index
            result = run_candidate(method, selection["configuration"], seed)
            test_records.append({**result, "phase": "test", "method": method, "seed": seed})
            print(f"[test] method={method} seed={seed} rmse={result['mean_rmse']:.4e}")
    write_csv(tuning_records + test_records, output_dir / "fair_tuning_runs.csv")
    summary: list[dict[str, Any]] = []
    for method in candidates:
        values = np.asarray([row["mean_rmse"] for row in test_records if row["method"] == method])
        finals = np.asarray([row["final_rmse"] for row in test_records if row["method"] == method])
        summary.append(
            {
                "method": method,
                "mean_rmse": float(values.mean()),
                "mean_rmse_ci95": bootstrap_ci(values, base_seed + len(summary)),
                "final_rmse": float(finals.mean()),
                "best_configuration": best[method]["configuration"],
            }
        )
    payload = {
        "equal_candidate_budget": 6,
        "calibration_seeds": calibration_seeds,
        "test_seeds": test_seeds,
        "best": best,
        "summary": summary,
    }
    (output_dir / "fair_tuning_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    configure_style()
    figure, axis = plt.subplots(figsize=(4.7, 2.7))
    labels = ["EnKF", "Original EnSF", "EnSF-LR", "Joint EnKF", "PCE"]
    colors = ["#9A9A9A", "#A8B3C1", "#7F8FA6", "#6AAE8B", "#4C78A8"]
    axis.bar(labels, [row["mean_rmse"] for row in summary], color=colors)
    axis.set_ylabel("Held-out time-mean RMSE")
    axis.set_title("Equal-budget hyperparameter tuning", loc="left", fontweight="bold")
    axis.tick_params(axis="x", rotation=24)
    figure.tight_layout()
    save_figure(figure, output_dir / "figure_fair_tuning")
    plt.close(figure)
    return payload


def run_scalability(base_seed: int, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    settings = [
        (41, 18, 7, 8), (161, 18, 7, 8), (641, 18, 7, 8),
        (161, 10, 7, 8), (161, 30, 7, 8),
        (161, 18, 3, 8), (161, 18, 11, 8),
        (161, 18, 7, 4), (161, 18, 7, 12),
    ]
    for index, (nx, ensemble, n_alpha, reverse_steps) in enumerate(settings):
        stable_dt = 0.40 / (nx - 1)
        cfg = replace(
            make_config("quick"),
            seed=base_seed + index,
            nx=nx,
            ensemble_size=ensemble,
            n_alpha=n_alpha,
            reverse_steps=reverse_steps,
            dt=stable_dt,
            t_end=80 * stable_dt,
            obs_interval=8,
            filter_variant="lr",
        )
        scenario = v3.generate_scenario(cfg)
        timings = []
        peaks = []
        result: dict[str, Any] = {}
        for _ in range(3):
            tracemalloc.start()
            start = time.perf_counter()
            result = v4.run_ablation(scenario, "A6_pce")
            timings.append(time.perf_counter() - start)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peaks.append(peak / (1024**2))
        elapsed = float(np.median(timings))
        records.append(
            {
                "nx": nx,
                "state_dimension": 2 * nx,
                "ensemble_size": ensemble,
                "n_alpha": n_alpha,
                "reverse_steps": reverse_steps,
                "runtime_seconds": elapsed,
                "peak_memory_mb": float(np.median(peaks)),
                "mean_rmse": result["mean_rmse"],
            }
        )
        print(f"[scale {index + 1}/{len(settings)}] nx={nx} J={ensemble} Na={n_alpha} Ntau={reverse_steps} time={elapsed:.3f}s")

    parallel_cfg = replace(make_config("quick"), seed=base_seed + 500, t_end=0.20, filter_variant="lr")
    scenario = v3.generate_scenario(parallel_cfg)
    states = scenario.branch_initial.copy()

    def forecast_branch(q: int) -> np.ndarray:
        return propagate_batch(
            states[q],
            float(scenario.theta_grid[q]),
            0.0,
            parallel_cfg,
            np.random.default_rng(base_seed + q),
            stochastic=True,
            noise_draw=scenario.forecast_noise[0],
        )

    parallel_records: list[dict[str, Any]] = []
    repetitions = 80
    for workers in (1, 2, 4):
        start = time.perf_counter()
        if workers == 1:
            for _ in range(repetitions):
                _ = [forecast_branch(q) for q in range(parallel_cfg.n_alpha)]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for _ in range(repetitions):
                    _ = list(pool.map(forecast_branch, range(parallel_cfg.n_alpha)))
        elapsed = time.perf_counter() - start
        parallel_records.append({"workers": workers, "runtime_seconds": elapsed})
    baseline_time = parallel_records[0]["runtime_seconds"]
    for row in parallel_records:
        row["speedup"] = baseline_time / row["runtime_seconds"]
    write_csv(records, output_dir / "scalability_runs.csv")
    write_csv(parallel_records, output_dir / "parallel_speedup.csv")
    payload = {"scaling": records, "parallel": parallel_records}
    (output_dir / "scalability_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    configure_style()
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))
    dimensions = [row for row in records if row["ensemble_size"] == 18 and row["n_alpha"] == 7 and row["reverse_steps"] == 8]
    axes[0].plot([row["state_dimension"] for row in dimensions], [row["runtime_seconds"] for row in dimensions], marker="o", color="#4C78A8")
    axes[0].set(xlabel="State dimension", ylabel="Runtime (s)", title="a  State scaling")
    ensembles = [row for row in records if row["nx"] == 161 and row["n_alpha"] == 7 and row["reverse_steps"] == 8]
    axes[1].plot([row["ensemble_size"] for row in ensembles], [row["runtime_seconds"] for row in ensembles], marker="s", color="#6AAE8B")
    axes[1].set(xlabel="Ensemble size", ylabel="Runtime (s)", title="b  Ensemble scaling")
    axes[2].bar([str(row["workers"]) for row in parallel_records], [row["speedup"] for row in parallel_records], color="#E09F3E")
    axes[2].set(xlabel="Worker threads", ylabel="Measured speedup", title="c  Branch parallelism")
    for axis in axes:
        axis.set_title(axis.get_title(), loc="left", fontweight="bold")
    figure.tight_layout(w_pad=1.5)
    save_figure(figure, output_dir / "figure_scalability")
    plt.close(figure)
    return payload


def write_reproducibility_files(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "numpy": np.__version__,
        "matplotlib": mpl.__version__,
        "base_seed": 20260803,
        "commands": {
            "multiphysics": "python paper_experiments/run_multiphysics_cases.py --n-seeds 20",
            "completion": "python paper_experiments/run_completion_suite.py --all",
            "full_alpha": "python run_generalization_v4.py --alpha-values 0.08,0.15,0.22,0.30,0.36,0.43,0.50,0.57,0.64,0.70,0.78,0.85,0.92 --obs-noise-values 0.02 --sensor-values 6 --n-seeds 10",
        },
    }
    (output_dir / "software_environment.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-completion robustness, tuning and calibration suite")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--mismatch", action="store_true")
    parser.add_argument("--tuning", action="store_true")
    parser.add_argument("--scalability", action="store_true")
    parser.add_argument("--calibration-seeds", type=int, default=30)
    parser.add_argument("--mismatch-seeds", type=int, default=10)
    parser.add_argument("--tuning-calibration-seeds", type=int, default=4)
    parser.add_argument("--tuning-test-seeds", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=20260803)
    parser.add_argument("--output", default="results_paper_completion")
    args = parser.parse_args()
    if not any((args.all, args.calibration, args.mismatch, args.tuning, args.scalability)):
        args.all = True
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"environment": write_reproducibility_files(output)}
    if args.all or args.calibration:
        payload["calibration"] = run_calibration(args.calibration_seeds, args.base_seed, output / "calibration")
    if args.all or args.mismatch:
        payload["mismatch"] = run_mismatch(args.mismatch_seeds, args.base_seed, output / "mismatch")
    if args.all or args.tuning:
        payload["tuning"] = run_fair_tuning(
            args.tuning_calibration_seeds,
            args.tuning_test_seeds,
            args.base_seed,
            output / "fair_tuning",
        )
    if args.all or args.scalability:
        payload["scalability"] = run_scalability(args.base_seed, output / "scalability")
    (output / "completion_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"completed": sorted(payload), "output": str(output.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
