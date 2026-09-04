#!/usr/bin/env python3
"""Stage C: causal boundary candidates for the 192-point S1 task."""
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
from geometry_192 import (
    causal_boundary_candidate_series_from_sparse,
    sample_192,
    scattered_series_from_sparse,
)
from run_stage_a import evaluate_region, nrmse


HERE = Path(__file__).resolve().parent
METHODS = ("DEnKF", "BMA", "PCE", "APCE")


def fixed_speed_recording_delay(
    field: np.ndarray,
    positions: np.ndarray,
    source: np.ndarray,
    sensor_indices: np.ndarray,
    rate: float,
    sound_speed: float,
    timing_window: int,
) -> float:
    values = np.abs(np.asarray(field[:timing_window, sensor_indices], dtype=np.float32))
    peak = np.max(values, axis=0)
    onset = []
    distance = np.linalg.norm(positions[sensor_indices] - source[None], axis=1)
    for sensor in range(values.shape[1]):
        threshold = max(0.05 * float(peak[sensor]), 1e-12)
        hit = np.flatnonzero(values[:, sensor] >= threshold)
        if len(hit):
            onset.append(float(hit[0]) / rate - float(distance[sensor]) / sound_speed)
    if len(onset) < max(16, len(sensor_indices) // 2):
        raise ValueError("too few direct-arrival onsets for delay calibration")
    onset_np = np.asarray(onset, dtype=float)
    median = float(np.median(onset_np))
    mad = float(np.median(np.abs(onset_np - median)))
    if mad > 0:
        keep = np.abs(onset_np - median) <= 4.0 * 1.4826 * mad
        onset_np = onset_np[keep]
    return float(np.median(onset_np))


def candidate_boundary_nrmse(
    truth: np.ndarray,
    candidates: np.ndarray,
    heldout_boundary: np.ndarray,
    analysis_end: int,
) -> list[float]:
    target = truth[:analysis_end].reshape(analysis_end, -1)[:, heldout_boundary]
    return [
        nrmse(
            target,
            candidates[:analysis_end, candidate].reshape(analysis_end, -1)[:, heldout_boundary],
        )
        for candidate in range(candidates.shape[1])
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rir", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda:2", "cuda:3"), default="cuda:2")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--store-candidate-history", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    config = json.loads((HERE / "config_v2.json").read_text(encoding="utf-8"))
    config["assimilation"]["seed"] = int(args.seed)
    config["assimilation"]["store_candidate_history"] = bool(args.store_candidate_history)
    rate = float(config["grid"]["native_sample_rate_hz"])
    analysis_end = int(round(float(config["grid"]["analysis_end_seconds"]) * rate))
    forecast_end = int(round(float(config["grid"]["forecast_end_seconds"]) * rate))
    fixed_speed = float(config["physics"]["fixed_sound_speed_m_s"])

    full_field = np.load(args.rir, mmap_mode="r")
    with np.load(args.geometry) as geometry:
        positions = np.asarray(geometry["s1_positions"], dtype=float)
        source_position = np.asarray(geometry["s1_source"][0], dtype=float)
    mapping, *_ = grid_mapping(positions)
    grid_positions = positions[mapping.reshape(-1)]
    sample = sample_192()
    boundary_sensor = mapping.reshape(-1)[sample["boundary_flat"]]
    interior_flat = sample["interior_flat"]
    heldout_interior = np.setdiff1d(sample["interior_all_flat"], interior_flat)
    heldout_boundary = np.setdiff1d(sample["boundary_all_flat"], sample["boundary_flat"])

    timing_window = min(int(0.25 * rate), len(full_field))
    recording_delay = fixed_speed_recording_delay(
        full_field, positions, source_position, boundary_sensor, rate, fixed_speed, timing_window
    )
    filter_kernel = signal.firwin(
        int(config["grid"]["model_filter_taps"]),
        float(config["grid"]["model_band_hz"]),
        fs=rate,
    )
    filtered = signal.lfilter(
        filter_kernel, [1.0], np.asarray(full_field[:timing_window], dtype=np.float32), axis=0
    ).astype(np.float32)
    filter_delay = (len(filter_kernel) - 1) / (2.0 * rate)
    crop_start = int(round((recording_delay + filter_delay) * rate))
    if crop_start < 0 or crop_start + forecast_end > len(filtered):
        raise ValueError(f"invalid crop {crop_start}:{crop_start + forecast_end}")
    truth = to_grid(filtered[crop_start:crop_start + forecast_end], mapping)
    boundary_series, candidate_labels = causal_boundary_candidate_series_from_sparse(
        truth,
        sample["boundary_flat"],
        grid_positions,
        source_position,
        rate,
        fixed_speed,
    )
    closure_errors = candidate_boundary_nrmse(
        truth, boundary_series, heldout_boundary, analysis_end
    )
    args.output.mkdir(parents=True, exist_ok=True)
    preparation = {
        "case": "meshir_s1_reconstruction_192_stage_c",
        "fixed_sound_speed_m_s": fixed_speed,
        "recording_delay_seconds": recording_delay,
        "filter_delay_seconds": filter_delay,
        "crop_start_sample": crop_start,
        "model_band_hz": float(config["grid"]["model_band_hz"]),
        "candidate_labels": candidate_labels,
        "heldout_boundary_candidate_nrmse_diagnostic": closure_errors,
        "heldout_used_to_construct_candidates": False,
        "causal_candidates": True,
    }
    (args.output / "preparation.json").write_text(json.dumps(preparation, indent=2), encoding="utf-8")
    if args.prepare_only:
        print(json.dumps(preparation, indent=2))
        return

    sparse_flat = np.union1d(sample["boundary_flat"], interior_flat)
    interpolation = scattered_series_from_sparse(truth, sparse_flat)
    signal_std = float(np.std(truth[:analysis_end].reshape(analysis_end, -1)[:, interior_flat]))
    process_std = float(config["physics"]["process_noise_fraction"]) * max(signal_std, 1e-8)
    ensemble = int(config["assimilation"]["ensemble_size"])
    candidate_count = len(candidate_labels)
    rng = np.random.default_rng(2026082000 + int(args.seed))
    initial_noise = (
        0.1 * process_std * rng.standard_normal((candidate_count, ensemble, 2, 9, 21, 21))
    ).astype(np.float32)
    run_config = json.loads(json.dumps(config))
    run_config["physics"]["nominal_speed_m_s"] = fixed_speed
    cfl = cfl_number(fixed_speed, 1.0 / rate, float(config["grid"]["spacing_m"]))
    if cfl >= 1.0:
        raise ValueError(f"unstable CFL number {cfl}")

    baseline_metrics = {}
    baseline_metrics.update(evaluate_region(
        truth, interpolation, np.zeros_like(interpolation), heldout_interior,
        analysis_end, rate, "interior",
    ))
    baseline_metrics.update(evaluate_region(
        truth, interpolation, np.zeros_like(interpolation), heldout_boundary,
        analysis_end, rate, "boundary",
    ))
    (args.output / "scattered_baseline.json").write_text(
        json.dumps({"method": "scattered_192", **baseline_metrics}, indent=2), encoding="utf-8"
    )

    records = []
    for method in args.methods:
        print(f"method={method} device={args.device}", flush=True)
        result = run(
            method,
            truth,
            boundary_series,
            interior_flat,
            np.asarray([fixed_speed]),
            run_config,
            args.device,
            initial_noise,
            noise_seed=2026082017 + int(args.seed),
            process_noise_std=process_std,
        )
        metrics = {}
        metrics.update(evaluate_region(
            truth, result.mean, result.standard_deviation, heldout_interior,
            analysis_end, rate, "interior",
        ))
        metrics.update(evaluate_region(
            truth, result.mean, result.standard_deviation, heldout_boundary,
            analysis_end, rate, "boundary",
        ))
        record = {
            **preparation,
            "method": method,
            "device": args.device,
            "seed": int(args.seed),
            "ensemble_size": ensemble,
            "cfl_number": cfl,
            "process_noise_std": process_std,
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
            candidate_labels=np.asarray(candidate_labels),
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
        **preparation,
        "authoritative_rir": str(args.rir),
        "authoritative_geometry": str(args.geometry),
        "completed_runs": len(records),
        "methods": list(args.methods),
        "seed": int(args.seed),
        "records": records,
        "allocation": {"total": 192, "boundary": 128, "interior": 64},
        "evaluation": {"heldout_interior": len(heldout_interior), "heldout_boundary": len(heldout_boundary)},
        "candidate_parameter": "causal boundary closure; not Liu alpha and not sound speed",
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    np.savez_compressed(args.output / "sampling_192.npz", **sample)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
