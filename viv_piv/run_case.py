from __future__ import annotations

import argparse
import json
import math
import pathlib
import time
from typing import Any

import numpy as np
import torch

from .assimilation import CandidateLibrary, METHODS, Scenario, run_pass, run_two_pass
from .common import json_ready, load_config, write_json
from .io import VIVCase, list_cases
from .metrics import full_field_metrics
from .rom import DMDCCandidate, PODModel


SENSOR_LAYOUT = "adaptive_fullfield_valid_x40y20"
OBSERVATION_COVARIANCE = "full"


def _controls(time_s: np.ndarray, displacement_m: np.ndarray, diameter_m: float) -> np.ndarray:
    displacement = displacement_m / diameter_m
    velocity = np.gradient(displacement, time_s)
    return np.column_stack([np.ones_like(displacement), displacement, velocity]).astype(np.float64)


def _sensor_scalar_indices(observation_dimensions: int, density: int | None) -> np.ndarray:
    if observation_dimensions % 2:
        raise ValueError("PIV observation vector must contain paired u/v components")
    points = observation_dimensions // 2
    if density is None or density == points:
        return np.arange(observation_dimensions, dtype=np.int64)
    if density <= 0 or density > points:
        raise ValueError(f"Sensor density must be between 1 and {points}, received {density}")
    selected = np.linspace(0, points - 1, density, dtype=int)
    return np.ravel(np.column_stack([2 * selected, 2 * selected + 1])).astype(np.int64)


def _evaluation_keep_mask(pod: PODModel, sensor_archive: Any | None) -> np.ndarray:
    keep = np.ones(pod.evaluation_flat_indices.size, dtype=bool)
    if sensor_archive is not None:
        sensor_indices = np.asarray(sensor_archive["sensor_flat_indices"], dtype=np.int64)
        keep = ~np.isin(pod.evaluation_flat_indices, sensor_indices)
    if not np.any(keep):
        raise ValueError("Sensor layout consumes every registered evaluation dimension")
    return keep


def _apply_spatial_taper(
    covariance: np.ndarray,
    sensor_coordinates_mm: np.ndarray,
    diameter_m: float,
    length_d: float,
) -> np.ndarray:
    """Apply a positive-definite exponential spatial taper to u/v covariance."""
    points = np.asarray(sensor_coordinates_mm, dtype=np.float64).reshape(-1, 2)
    if covariance.shape != (2 * points.shape[0], 2 * points.shape[0]):
        raise ValueError("Spatial taper coordinates and covariance dimensions disagree")
    if diameter_m <= 0.0 or length_d <= 0.0:
        raise ValueError("Spatial taper diameter and length must be positive")
    point_ids = np.repeat(np.arange(points.shape[0]), 2)
    distance_d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1) / (diameter_m * 1000.0)
    scalar_distance_d = distance_d[np.ix_(point_ids, point_ids)]
    taper = np.exp(-scalar_distance_d / float(length_d))
    tapered = 0.5 * (covariance * taper + (covariance * taper).T)
    # The exponential kernel is positive definite; this tiny floor only
    # protects Cholesky from round-off after the Schur product.
    floor = max(float(np.max(np.diag(tapered))), 1e-14) * 1e-10
    tapered = tapered + floor * np.eye(tapered.shape[0])
    np.linalg.cholesky(tapered)
    return tapered


def _load_scenario(
    model_root: pathlib.Path,
    case_id: str,
    config: dict[str, Any],
    no_control: bool,
    sensor_density: int | None = None,
    sensor_layout: str | None = None,
    observation_covariance_mode: str = "diagonal",
    *,
    target_split: str = "test",
    reference_cases: list[str] | None = None,
    covariance_shrinkage: float | None = None,
) -> Scenario:
    pod = PODModel.load(model_root / "pod_model.npz")
    target = np.load(model_root / "coefficients" / f"case_{case_id}.npz", allow_pickle=False)
    actual_split = str(target["split"].item())
    if actual_split != target_split:
        raise ValueError(f"{case_id} is registered as {actual_split!r}, expected {target_split!r}")
    reference_cases = list(reference_cases or config["train_cases"])
    if case_id in reference_cases:
        raise ValueError(f"Target case {case_id} must not enter its own reference statistics")
    if not reference_cases:
        raise ValueError("At least one reference case is required")
    if sensor_layout is not None and sensor_density is not None:
        raise ValueError("--sensor-layout and --sensor-density are mutually exclusive")
    if sensor_layout is not None:
        target_sensor = np.load(model_root / "sensor_layouts" / sensor_layout / f"case_{case_id}.npz", allow_pickle=False)
        scalar_indices = np.arange(int(target_sensor["sensor_observations"].shape[1]), dtype=np.int64)
    else:
        target_sensor = target
        scalar_indices = _sensor_scalar_indices(int(target_sensor["sensor_observations"].shape[1]), sensor_density)
    requested_covariance_shrinkage = float(
        config["observation_covariance_shrinkage"]
        if covariance_shrinkage is None else covariance_shrinkage
    )
    use_stored_covariance = (
        observation_covariance_mode in {"full", "tapered"}
        and reference_cases == list(config["train_cases"])
        and math.isclose(
            requested_covariance_shrinkage,
            float(config["observation_covariance_shrinkage"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    train_observations: list[np.ndarray] = []
    train_residuals: list[np.ndarray] = []
    train_coefficients: list[np.ndarray] = []
    for train_case in reference_cases:
        data = np.load(model_root / "coefficients" / f"case_{train_case}.npz", allow_pickle=False)
        coefficients = np.asarray(data["coefficients"], dtype=np.float64)
        train_coefficients.append(coefficients)
        if use_stored_covariance:
            continue
        sensor_data = (
            np.load(model_root / "sensor_layouts" / sensor_layout / f"case_{train_case}.npz", allow_pickle=False)
            if sensor_layout is not None else data
        )
        observation = np.asarray(sensor_data["sensor_observations"], dtype=np.float64)[:, scalar_indices]
        mean = np.asarray(sensor_data["sensor_mean"], dtype=np.float64)[scalar_indices]
        basis = np.asarray(sensor_data["sensor_basis"], dtype=np.float64)[scalar_indices]
        train_observations.append(observation)
        train_residuals.append(observation - (mean[None, :] + coefficients @ basis.T))
    observation_noise: np.ndarray
    if not use_stored_covariance:
        signal_scale = np.std(np.concatenate(train_observations, axis=0), axis=0)
        residual_scale = np.std(np.concatenate(train_residuals, axis=0), axis=0)
        observation_noise = np.maximum(
            residual_scale,
            float(config["observation_noise_fraction"]) * signal_scale,
        ).clip(1e-7)
    observation_covariance = None
    if use_stored_covariance:
        if sensor_layout is None:
            raise ValueError("Full observation covariance requires --sensor-layout")
        covariance_archive = np.load(
            model_root / "sensor_layouts" / sensor_layout / "observation_covariance_full.npz",
            allow_pickle=False,
        )
        archive_shrinkage = float(covariance_archive["shrinkage"])
        if not math.isclose(
            archive_shrinkage,
            requested_covariance_shrinkage,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Stored observation covariance shrinkage does not match the requested value: "
                f"archive={archive_shrinkage}, requested={requested_covariance_shrinkage}"
            )
        observation_noise = np.asarray(
            covariance_archive["observation_std"], dtype=np.float64
        )[scalar_indices]
        observation_covariance = np.asarray(covariance_archive["covariance"], dtype=np.float64)
        observation_covariance = observation_covariance[np.ix_(scalar_indices, scalar_indices)]
        if observation_covariance.shape != (scalar_indices.size, scalar_indices.size):
            raise ValueError("Full observation covariance has an incompatible shape")
    elif observation_covariance_mode in {"full", "tapered"}:
        # Pseudo-held-out calibration derives every covariance only from its
        # reference cases. This makes the shrinkage choice explicit and
        # prevents a target training run from contributing to its own R.
        residual = np.concatenate(train_residuals, axis=0)
        residual = residual - residual.mean(axis=0, keepdims=True)
        observation = np.concatenate(train_observations, axis=0)
        observation = observation - observation.mean(axis=0, keepdims=True)
        empirical = residual.T @ residual / max(residual.shape[0] - 1, 1)
        residual_std = np.sqrt(np.maximum(np.diag(empirical), 1e-14))
        signal_std = np.sqrt(np.maximum(np.sum(observation**2, axis=0) / max(observation.shape[0] - 1, 1), 1e-14))
        target_std = np.maximum(
            residual_std,
            float(config["observation_noise_fraction"]) * signal_std,
        ).clip(1e-7)
        correlation = empirical / np.maximum(residual_std[:, None] * residual_std[None, :], 1e-14)
        correlation = 0.5 * (correlation + correlation.T)
        np.fill_diagonal(correlation, 1.0)
        shrinkage = requested_covariance_shrinkage
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError(f"observation covariance shrinkage must be in [0, 1], received {shrinkage}")
        observation_covariance = (1.0 - shrinkage) * correlation * target_std[:, None] * target_std[None, :]
        observation_covariance += shrinkage * np.diag(target_std**2)
        observation_covariance = 0.5 * (observation_covariance + observation_covariance.T)
        # A small numerical floor is only for Cholesky robustness; it is
        # reported neither as a learned parameter nor as extra observation noise.
        try:
            np.linalg.cholesky(observation_covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Dynamic full observation covariance is not positive definite") from exc
    if observation_covariance_mode == "tapered":
        if sensor_layout is None or "sensor_coordinates_mm" not in target_sensor:
            raise ValueError("Spatially tapered covariance requires a prepared sensor layout")
        coordinates = np.asarray(target_sensor["sensor_coordinates_mm"], dtype=np.float64)
        coordinates = coordinates[scalar_indices[::2] // 2]
        observation_covariance = _apply_spatial_taper(
            observation_covariance,
            coordinates,
            float(config["cylinder_diameter_m"]),
            float(config.get("observation_taper_length_d", 1.0)),
        )
    initial_std = np.std(np.concatenate(train_coefficients, axis=0), axis=0).clip(1e-7)
    control = _controls(
        np.asarray(target["time_s"], dtype=np.float64),
        np.asarray(target["cyl_displ_m"], dtype=np.float64),
        float(config["cylinder_diameter_m"]),
    )
    if no_control:
        control = np.column_stack([np.ones(control.shape[0]), np.zeros((control.shape[0], 2))])
    evaluation_keep = _evaluation_keep_mask(pod, target_sensor if sensor_layout is not None else None)
    evaluation_indices = pod.evaluation_flat_indices[evaluation_keep]
    return Scenario(
        case_id=case_id,
        time_s=np.asarray(target["time_s"], dtype=np.float64),
        control=control,
        observations=np.asarray(target_sensor["sensor_observations"], dtype=np.float64)[:, scalar_indices],
        sensor_mean=np.asarray(target_sensor["sensor_mean"], dtype=np.float64)[scalar_indices],
        sensor_basis=np.asarray(target_sensor["sensor_basis"], dtype=np.float64)[scalar_indices],
        evaluation_truth=np.asarray(target["evaluation_truth"], dtype=np.float64)[:, evaluation_keep],
        evaluation_mean=pod.mean[evaluation_indices].astype(np.float64),
        evaluation_basis=pod.basis[evaluation_indices].astype(np.float64),
        observation_noise=observation_noise,
        initial_std=initial_std,
        observation_covariance=observation_covariance,
        initial_ensemble_scale=float(config.get("initial_ensemble_scale", 0.15)),
        process_noise_scale=float(config.get("process_noise_scale", 1.0)),
        state_inflation=float(config.get("state_inflation", 1.0)),
    )


def _load_library(model_root: pathlib.Path, *, exclude_case_ids: set[str] | None = None) -> CandidateLibrary:
    excluded = set() if exclude_case_ids is None else {str(value) for value in exclude_case_ids}
    candidates = [DMDCCandidate.load(path) for path in sorted((model_root / "candidates").glob("candidate_*.npz"))]
    candidates = [candidate for candidate in candidates if candidate.case_id not in excluded]
    candidates.sort(key=lambda item: item.reduced_velocity)
    if not candidates:
        raise FileNotFoundError(f"No candidate files under {model_root}")
    return CandidateLibrary(candidates)


def blackout_origins(scenario: Scenario, config: dict[str, Any]) -> set[int]:
    warmup = int(round(float(config["warmup_seconds"]) / float(config["time_step_s"])))
    max_horizon = int(round(max(config["blackout_horizons_s"]) / float(config["time_step_s"])))
    spacing = int(round(float(config["blackout_min_spacing_s"]) / float(config["time_step_s"])))
    start = warmup + max_horizon
    stop = scenario.steps - max_horizon - 1
    available = np.arange(start, stop + 1, max(spacing, 1), dtype=int)
    if available.size == 0:
        return set()
    selected = np.linspace(0, available.size - 1, min(int(config["blackout_origins"]), available.size), dtype=int)
    return set(int(available[index]) for index in selected)


def run_blackouts(
    pass_result,
    scenario: Scenario,
    library: CandidateLibrary,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> list[dict[str, float]]:
    if not pass_result.blackout_states:
        return []
    dtype = torch.float64
    eval_mean = torch.as_tensor(scenario.evaluation_mean, dtype=dtype, device=device)
    eval_basis = torch.as_tensor(scenario.evaluation_basis, dtype=dtype, device=device)
    inputs = torch.as_tensor(scenario.control, dtype=dtype, device=device)
    horizons = sorted({int(round(value / float(config["time_step_s"]))) for value in config["blackout_horizons_s"]})
    rows: list[dict[str, float]] = []
    for origin, snapshot in sorted(pass_result.blackout_states.items()):
        branches = torch.as_tensor(snapshot["branches"], dtype=dtype, device=device)
        weights = torch.as_tensor(snapshot["weights"], dtype=dtype, device=device)
        grid = np.asarray(snapshot["grid"], dtype=np.float64)
        member_coordinates = snapshot.get("member_coordinates")
        if member_coordinates is not None:
            coordinate_tensor = torch.as_tensor(member_coordinates, dtype=dtype, device=device)
            matrices, controls, q_sqrt = library.parameters_torch(coordinate_tensor, device, dtype)
        else:
            matrices, controls, q_sqrt = library.parameters(grid, device, dtype)
        q_sqrt = q_sqrt * float(scenario.process_noise_scale)
        generator = torch.Generator(device=device).manual_seed(int(seed + 2000000 + origin))
        for step in range(1, max(horizons) + 1):
            noise = torch.randn((branches.shape[1], library.rank), dtype=dtype, device=device, generator=generator)
            if member_coordinates is not None:
                member_states = branches[0]
                member_states = (
                    torch.bmm(member_states.unsqueeze(1), matrices.transpose(-1, -2)).squeeze(1)
                    + torch.einsum("nrc,c->nr", controls, inputs[origin + step - 1])
                    + noise * q_sqrt
                )
                branches = member_states.unsqueeze(0)
            else:
                branches = (
                    torch.matmul(branches, matrices.transpose(-1, -2))
                    + torch.einsum("jrc,c->jr", controls, inputs[origin + step - 1]).unsqueeze(1)
                    + noise.unsqueeze(0) * q_sqrt.unsqueeze(1)
                )
            if step not in horizons:
                continue
            estimate = torch.sum(weights[:, None] * branches.mean(dim=1), dim=0)
            prediction = eval_mean + estimate @ eval_basis.mT
            truth = torch.as_tensor(scenario.evaluation_truth[origin + step], dtype=dtype, device=device)
            nrmse = torch.sqrt(torch.sum((prediction - truth) ** 2) / torch.sum(truth**2).clamp_min(1e-30))
            rows.append({
                "origin_index": int(origin),
                "origin_time_s": float(scenario.time_s[origin]),
                "horizon_s": float(step * float(config["time_step_s"])),
                "evaluation_nrmse": float(nrmse),
            })
    return rows


def save_trace(path: pathlib.Path, *, pass_result, full_trace: dict[str, np.ndarray], scenario: Scenario, blackout: list[dict[str, float]]) -> None:
    arrays: dict[str, Any] = {
        "latent_estimate": pass_result.latent_estimate,
        "candidate_grid": pass_result.grid,
        "final_weights": pass_result.final_weights,
        "final_scores": pass_result.final_scores,
        "time_s": scenario.time_s,
        "evaluation_truth": scenario.evaluation_truth,
        "truth_energy": full_trace["truth_energy"],
        "predicted_energy": full_trace["predicted_energy"],
        "blackout_rows_json": np.asarray(json.dumps(blackout)),
    }
    for key, value in pass_result.trace.items():
        arrays[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one VIV-PIV PCE/APCE or baseline assimilation case.")
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--record-trace", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    case_id = str(args.case).replace(",", "").zfill(4)[-4:]
    if case_id not in config["test_cases"]:
        raise ValueError(f"--case must be one of held-out tests: {config['test_cases']}")
    variant = f"rank{int(config['rank'])}_stride1"
    output_root = pathlib.Path(config["output_root"]) / "runs" / variant
    model_root = pathlib.Path(config["output_root"]) / "models" / variant
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    run_id = f"viv_{case_id}_{args.method}_seed{args.seed:03d}"
    prepared = np.load(
        model_root / "sensor_layouts" / SENSOR_LAYOUT / f"case_{case_id}.npz",
        allow_pickle=False,
    )
    sensor_points = int(prepared["sensor_observations"].shape[1] // 2)
    run_path = output_root / "runs" / f"{run_id}.json"
    trace_path = output_root / "traces" / f"{run_id}.npz"
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "run_id": run_id,
        "case_id": case_id,
        "method": args.method,
        "seed": int(args.seed),
        "model_variant": variant,
        "device": str(device),
        "uses_known_cylinder_displacement_input": True,
        "sensor_density_points": sensor_points,
        "sensor_layout": SENSOR_LAYOUT,
        "observation_covariance": OBSERVATION_COVARIANCE,
        "observation_covariance_shrinkage": float(config["observation_covariance_shrinkage"]),
        "observation_covariance_source": "training_conditions",
        "ensemble_size": int(config["ensemble_size"]),
        "total_analysis_members": int(config["ensemble_size"]) if args.method == "aug_enkf" else None,
        "aug_enkf_stability_grid_points": int(
            config.get("aug_enkf_stability_grid_points", max(101, 20 * (len(config["train_cases"]) - 1) + 1))
        ) if args.method == "aug_enkf" else None,
        "evidence_window_frames": int(config.get("evidence_window_frames", 1)),
        "valid": False,
        "status": "started",
    }
    try:
        scenario = _load_scenario(
            model_root,
            case_id,
            config,
            False,
            None,
            SENSOR_LAYOUT,
            OBSERVATION_COVARIANCE,
        )
        payload["evaluation_dimensions"] = int(scenario.evaluation_truth.shape[1])
        payload["evaluation_sensor_overlap_excluded"] = int(config["evaluation_points"]) - int(scenario.evaluation_truth.shape[1])
        library = _load_library(model_root)
        origins = blackout_origins(scenario, config)
        if args.method in {"pce", "apce"}:
            two_pass = run_two_pass(
                scenario, library, config, args.method, args.seed, device,
                record_trace=args.record_trace, blackout_origins=origins,
            )
            result = two_pass.local
            payload.update(
                coarse_grid=two_pass.coarse.grid,
                coarse_final_weights=two_pass.coarse.final_weights,
                coarse_final_scores=two_pass.coarse.final_scores,
                local_grid_stable=two_pass.local_stable,
                local_grid_failure=two_pass.local_failure,
            )
        else:
            result = run_pass(
                scenario, library, config, args.method, args.seed, device,
                record_trace=args.record_trace, blackout_origins=origins,
            )
        blackout = run_blackouts(result, scenario, library, config, args.seed, device)
        payload.update(result.metrics)
        path = list_cases(pathlib.Path(config["data_root"]))[case_id]
        sensor = np.load(
            model_root / "sensor_layouts" / SENSOR_LAYOUT / f"case_{case_id}.npz",
            allow_pickle=False,
        )
        excluded = np.asarray(sensor["sensor_flat_indices"], dtype=np.int64)
        field_metrics, field_trace = full_field_metrics(
            VIVCase.open(path),
            PODModel.load(model_root / "pod_model.npz"),
            result.latent_estimate,
            device,
            excluded_flat_indices=excluded,
        )
        payload.update(field_metrics)
        payload.update(
            final_candidate_grid=result.grid,
            final_weights=result.final_weights,
            final_log_scores=result.final_scores,
            final_weight_entropy=float(-np.sum(np.maximum(result.final_weights, 1e-300) * np.log(np.maximum(result.final_weights, 1e-300)))),
            effective_candidate_count=float(1.0 / np.sum(result.final_weights**2)),
            blackout_count=len(blackout),
            blackout_mean_nrmse=float(np.mean([row["evaluation_nrmse"] for row in blackout])) if blackout else math.nan,
            observation_dimensions=int(scenario.observations.shape[1]),
            candidate_count=int(result.grid.size),
            status="completed",
            valid=True,
        )
        if args.record_trace:
            save_trace(trace_path, pass_result=result, full_trace=field_trace, scenario=scenario, blackout=blackout)
            payload["trace_path"] = str(trace_path)
    except Exception as exc:
        payload.update(status="failed", failure_type=type(exc).__name__, failure_message=str(exc), valid=False)
    payload["wall_seconds"] = float(time.perf_counter() - started)
    write_json(run_path, payload)
    print(json.dumps(json_ready(payload), ensure_ascii=False))


if __name__ == "__main__":
    main()
