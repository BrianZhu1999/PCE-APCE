#!/usr/bin/env python3
"""Stage A: fixed-speed 192-point S1 reconstruction smoke.

This stage deliberately validates the reconstruction information flow before
introducing candidate boundary closures for PCE/APCE. It uses 128 geometry-only
boundary points and 64 geometry-only interior points.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import signal

from fullwave.assimilation import run
from fullwave.geometry import grid_mapping, to_grid
from fullwave.model import cfl_number
from geometry_192 import boundary_candidate_series_from_sparse, sample_192, scattered_series_from_sparse


HERE = Path(__file__).resolve().parent
METHODS = ("DEnKF", "BMA", "PCE", "APCE")


def nrmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    denominator = max(float(np.sum(truth ** 2)), 1e-20)
    return float(np.sqrt(np.sum((prediction - truth) ** 2) / denominator))


def correlation(truth: np.ndarray, prediction: np.ndarray) -> float:
    a, b = truth.reshape(-1), prediction.reshape(-1)
    if np.std(a) < 1e-20 or np.std(b) < 1e-20:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def evaluate_region(truth: np.ndarray, mean: np.ndarray, std: np.ndarray, flat: np.ndarray,
                    analysis_end: int, rate: float, prefix: str) -> dict[str, float]:
    truth_flat = truth.reshape(len(truth), -1)[:, flat]
    mean_flat = mean.reshape(len(mean), -1)[:, flat]
    std_flat = std.reshape(len(std), -1)[:, flat]
    output = {
        f"{prefix}_analysis_nrmse": nrmse(truth_flat[:analysis_end], mean_flat[:analysis_end]),
        f"{prefix}_analysis_correlation": correlation(truth_flat[:analysis_end], mean_flat[:analysis_end]),
        f"{prefix}_analysis_mae": float(np.mean(np.abs(truth_flat[:analysis_end] - mean_flat[:analysis_end]))),
    }
    for milliseconds in (1, 2, 4):
        end = min(analysis_end + int(milliseconds * rate / 1000), len(truth))
        output[f"{prefix}_forecast_{milliseconds}ms_nrmse"] = nrmse(truth_flat[analysis_end:end], mean_flat[analysis_end:end])
        output[f"{prefix}_forecast_{milliseconds}ms_correlation"] = correlation(truth_flat[analysis_end:end], mean_flat[analysis_end:end])
    lower = mean_flat - 1.6448536269514722 * std_flat
    upper = mean_flat + 1.6448536269514722 * std_flat
    output[f"{prefix}_coverage_90"] = float(np.mean((truth_flat >= lower) & (truth_flat <= upper)))
    output[f"{prefix}_mean_interval_width"] = float(np.mean(upper - lower))
    return output


def fit_delay_and_speed(full_field: np.ndarray, positions: np.ndarray, source: np.ndarray,
                        sensor_indices: np.ndarray, rate: float, timing_window: int) -> tuple[float, float]:
    peak_index = np.argmax(np.abs(full_field[:timing_window, sensor_indices]), axis=0)
    peak_time = peak_index / rate
    distances = np.linalg.norm(positions[sensor_indices] - source[None, :], axis=1)
    inverse_speed, recording_delay = np.polyfit(distances, peak_time, 1)
    if inverse_speed <= 0:
        raise ValueError("non-positive fitted inverse speed")
    return float(1.0 / inverse_speed), float(recording_delay)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rir", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda:2", "cuda:3"), default="cuda:2")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--store-candidate-history", action="store_true")
    args = parser.parse_args()
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    if args.seed is not None:
        config["assimilation"]["seed"] = int(args.seed)
    config["assimilation"]["store_candidate_history"] = bool(args.store_candidate_history)
    rate = float(config["grid"]["native_sample_rate_hz"])
    analysis_end = int(round(float(config["grid"]["analysis_end_seconds"]) * rate))
    forecast_end = int(round(float(config["grid"]["forecast_end_seconds"]) * rate))

    full_field = np.load(args.rir, mmap_mode="r")
    with np.load(args.geometry) as geometry:
        positions = np.asarray(geometry["s1_positions"], dtype=float)
        source_position = np.asarray(geometry["s1_source"][0], dtype=float)
    if full_field.ndim != 2 or full_field.shape[1] != 3969:
        raise ValueError(f"unexpected S1 field shape {full_field.shape}")
    mapping, *_ = grid_mapping(positions)
    sample = sample_192()
    boundary_sensor = mapping.reshape(-1)[sample["boundary_flat"]]
    interior_flat = sample["interior_flat"]
    all_boundary = sample["boundary_all_flat"]
    all_interior = sample["interior_all_flat"]
    heldout_interior = np.setdiff1d(all_interior, interior_flat)
    heldout_boundary = np.setdiff1d(all_boundary, sample["boundary_flat"])
    heldout_all = np.union1d(heldout_interior, heldout_boundary)
    if len(boundary_sensor) != 128 or len(interior_flat) != 64 or len(heldout_interior) != 2463:
        raise RuntimeError("192-point allocation audit failed")

    timing_window = min(int(0.25 * rate), len(full_field))
    fitted_speed, recording_delay = fit_delay_and_speed(
        full_field, positions, source_position, boundary_sensor, rate, timing_window
    )
    filter_kernel = signal.firwin(int(config["grid"]["model_filter_taps"]), float(config["grid"]["model_band_hz"]), fs=rate)
    filtered = signal.lfilter(filter_kernel, [1.0], np.asarray(full_field[:timing_window], dtype=np.float32), axis=0).astype(np.float32)
    filter_delay = (len(filter_kernel) - 1) / (2.0 * rate)
    crop_start = int(round((recording_delay + filter_delay) * rate))
    if crop_start < 0 or crop_start + forecast_end > len(filtered):
        raise ValueError(f"invalid crop {crop_start}:{crop_start + forecast_end}")
    field_tm = filtered[crop_start:crop_start + forecast_end]
    truth = to_grid(np.asarray(field_tm, dtype=np.float32), mapping)

    closure_scales = [float(value) for value in config["physics"]["boundary_closure_scales_m"]]
    closure_labels = ["linear"] + [f"gaussian_{value:.2f}m" for value in closure_scales]
    boundary_series = boundary_candidate_series_from_sparse(truth, sample["boundary_flat"], closure_scales)
    sparse_flat = np.union1d(sample["boundary_flat"], interior_flat)
    interpolation = scattered_series_from_sparse(truth, sparse_flat)
    signal_std = float(np.std(truth[:analysis_end, :, :, :][:, np.unravel_index(interior_flat, (9, 21, 21))[0], np.unravel_index(interior_flat, (9, 21, 21))[1], np.unravel_index(interior_flat, (9, 21, 21))[2]]))
    process_std = float(config["physics"]["process_noise_fraction"]) * max(signal_std, 1e-8)
    rng = np.random.default_rng(2026081900 + int(config["assimilation"]["seed"]))
    ensemble = int(config["assimilation"]["ensemble_size"])
    candidate_count = len(closure_labels)
    initial_noise = (0.1 * process_std * rng.standard_normal((candidate_count, ensemble, 2, 9, 21, 21))).astype(np.float32)
    calibrated_speed = float(fitted_speed)
    cfl = cfl_number(calibrated_speed, 1.0 / rate, float(config["grid"]["spacing_m"]))
    if cfl >= 1.0:
        raise ValueError(f"unstable CFL number {cfl}")

    args.output.mkdir(parents=True, exist_ok=True)
    baseline_metrics = {}
    baseline_metrics.update(evaluate_region(truth, interpolation, np.zeros_like(interpolation), heldout_interior, analysis_end, rate, "interior"))
    baseline_metrics.update(evaluate_region(truth, interpolation, np.zeros_like(interpolation), heldout_boundary, analysis_end, rate, "boundary"))
    (args.output / "scattered_baseline.json").write_text(json.dumps({"method": "scattered_192", **baseline_metrics}, indent=2), encoding="utf-8")

    run_config = json.loads(json.dumps(config))
    run_config["physics"]["nominal_speed_m_s"] = calibrated_speed
    records = []
    run_seed = int(config["assimilation"]["seed"])
    for method in args.methods:
        print(f"method={method} device={args.device}", flush=True)
        result = run(
            method, truth, boundary_series, interior_flat, np.asarray([calibrated_speed]), run_config,
            args.device, initial_noise, noise_seed=2026081917 + run_seed, process_noise_std=process_std,
        )
        metrics = {}
        metrics.update(evaluate_region(truth, result.mean, result.standard_deviation, heldout_interior, analysis_end, rate, "interior"))
        metrics.update(evaluate_region(truth, result.mean, result.standard_deviation, heldout_boundary, analysis_end, rate, "boundary"))
        record = {
            "case": "meshir_s1_reconstruction_192_stage_ab",
            "stage": "fixed_speed_boundary_closure_reconstruction",
            "method": method,
            "device": args.device,
            "grid_shape_zyx": [9, 21, 21],
            "observed_count": 192,
            "boundary_observed_count": 128,
            "interior_observed_count": 64,
            "heldout_interior_count": int(len(heldout_interior)),
            "heldout_boundary_count": int(len(heldout_boundary)),
            "heldout_all_count": int(len(heldout_all)),
            "candidate_parameter": "boundary completion length scale; sound speed fixed after sparse boundary calibration",
            "boundary_closure_labels": closure_labels,
            "boundary_closure_scales_m": closure_scales,
            "nominal_boundary_closure_index": int(config["physics"]["nominal_boundary_closure_index"]),
            "calibrated_speed_used_m_s": calibrated_speed,
            "fitted_speed_m_s": fitted_speed,
            "recording_delay_seconds": recording_delay,
            "filter_delay_seconds": filter_delay,
            "crop_start_sample": crop_start,
            "model_band_hz": float(config["grid"]["model_band_hz"]),
            "cfl_number": cfl,
            "boundary_interpolation_uses_only_selected_boundary": True,
            "heldout_used_for_fit_or_selection": False,
            "pce_apce_run": method in ("PCE", "APCE"),
            "final_weights": result.final_weights.tolist(),
            "median_separation_ratio": float(np.median(result.separation_history)) if len(result.separation_history) else 0.0,
            "paired_noise_digest": result.paired_noise_digest,
            "reduced_order_model_used": False,
            **metrics,
            "completed": True,
        }
        stem = args.output / method.lower()
        stem.with_suffix(".json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        arrays = dict(
            truth=truth,
            mean=result.mean,
            standard_deviation=result.standard_deviation,
            boundary_flat=sample["boundary_flat"],
            interior_flat=interior_flat,
            heldout_interior=heldout_interior,
            heldout_boundary=heldout_boundary,
            positions=positions,
            mapping=mapping,
            boundary_closure_scales_m=np.asarray(closure_scales, dtype=np.float32),
            final_weights=result.final_weights,
            weight_time=result.weight_time,
            weight_history=result.weight_history,
            separation_history=result.separation_history,
            score_history=result.score_history,
        )
        if result.candidate_mean is not None:
            arrays["candidate_mean"] = result.candidate_mean
        np.savez_compressed(stem.with_suffix(".npz"), **arrays)
        records.append(str(stem.with_suffix(".json")))
    manifest = {
        "case": "meshir_s1_reconstruction_192_stage_ab",
        "authoritative_rir": str(args.rir),
        "authoritative_geometry": str(args.geometry),
        "device": args.device,
        "completed_runs": len(records),
        "methods": list(args.methods),
        "seed": run_seed,
        "candidate_history_stored": bool(args.store_candidate_history),
        "records": records,
        "allocation": {"total": 192, "boundary": 128, "interior": 64},
        "evaluation": {"heldout_interior": int(len(heldout_interior)), "heldout_boundary": int(len(heldout_boundary))},
        "sampling_rule": "geometry-only farthest-point sampling with six-face seeds and seven interior z-layer seeds",
        "model_band_hz": float(config["grid"]["model_band_hz"]),
        "fitted_speed_m_s": fitted_speed,
        "recording_delay_seconds": recording_delay,
        "candidate_parameter": "boundary completion length scale; not Liu alpha and not sound speed",
        "boundary_closure_labels": closure_labels,
        "boundary_closure_scales_m": closure_scales,
        "pce_apce_run": True,
        "reduced_order_model_used": False,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    np.savez_compressed(args.output / "sampling_192.npz", **sample)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
