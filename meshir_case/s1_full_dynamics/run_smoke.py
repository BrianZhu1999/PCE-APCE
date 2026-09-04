#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import signal

from fullwave.assimilation import run
from fullwave.geometry import grid_mapping, interpolate_sparse_series, sparse_flat_indices, to_grid
from fullwave.model import cfl_number


HERE = Path(__file__).resolve().parent
METHODS = ("DEnKF", "BMA", "PCE", "APCE")


def nrmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.sum((prediction - truth) ** 2) / max(float(np.sum(truth ** 2)), 1e-20)))


def correlation(truth: np.ndarray, prediction: np.ndarray) -> float:
    a, b = truth.reshape(-1), prediction.reshape(-1)
    if np.std(a) < 1e-20 or np.std(b) < 1e-20:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def evaluate(truth: np.ndarray, mean: np.ndarray, std: np.ndarray, heldout: np.ndarray, analysis_end: int, rate: float) -> dict[str, float]:
    truth_flat = truth.reshape(len(truth), -1)[:, heldout]
    mean_flat = mean.reshape(len(mean), -1)[:, heldout]
    std_flat = std.reshape(len(std), -1)[:, heldout]
    metrics = {
        "analysis_nrmse": nrmse(truth_flat[:analysis_end], mean_flat[:analysis_end]),
        "analysis_correlation": correlation(truth_flat[:analysis_end], mean_flat[:analysis_end]),
    }
    for milliseconds in (1, 2, 4):
        end = min(analysis_end + int(milliseconds * rate / 1000), len(truth))
        metrics[f"forecast_{milliseconds}ms_nrmse"] = nrmse(truth_flat[analysis_end:end], mean_flat[analysis_end:end])
        metrics[f"forecast_{milliseconds}ms_correlation"] = correlation(truth_flat[analysis_end:end], mean_flat[analysis_end:end])
    lower = mean_flat - 1.6448536269514722 * std_flat
    upper = mean_flat + 1.6448536269514722 * std_flat
    metrics["coverage_90"] = float(np.mean((truth_flat >= lower) & (truth_flat <= upper)))
    metrics["mean_interval_width"] = float(np.mean(upper - lower))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rir", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda:2", "cuda:3"), default="cuda:2")
    args = parser.parse_args()
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    rate = float(config["grid"]["native_sample_rate_hz"])
    analysis_end = int(round(float(config["grid"]["analysis_end_seconds"]) * rate))
    forecast_end = int(round(float(config["grid"]["forecast_end_seconds"]) * rate))
    full_field = np.load(args.rir, mmap_mode="r")
    with np.load(args.geometry) as geometry:
        positions = geometry["s1_positions"]
        source_position = geometry["s1_source"][0]
    timing_window = min(int(0.25 * rate), len(full_field))
    peak_index = np.argmax(np.abs(full_field[:timing_window]), axis=0)
    peak_time = peak_index / rate
    source_distance = np.linalg.norm(positions - source_position[None, :], axis=1)
    inverse_speed, recording_delay = np.polyfit(source_distance, peak_time, 1)
    fitted_speed = float(1.0 / inverse_speed)
    filter_kernel = signal.firwin(
        int(config["grid"]["model_filter_taps"]),
        float(config["grid"]["model_band_hz"]), fs=rate,
    )
    filtered = signal.lfilter(
        filter_kernel, [1.0], np.asarray(full_field[:timing_window], dtype=np.float32), axis=0,
    ).astype(np.float32)
    filter_delay = (len(filter_kernel) - 1) / (2.0 * rate)
    crop_start = int(round((recording_delay + filter_delay) * rate))
    if crop_start < 0 or crop_start + forecast_end > len(filtered):
        raise ValueError(f"invalid crop {crop_start}:{crop_start + forecast_end}")
    field_tm = filtered[crop_start:crop_start + forecast_end]
    mapping, x, y, z = grid_mapping(positions)
    truth = to_grid(np.asarray(field_tm, dtype=np.float32), mapping)
    interpolation = interpolate_sparse_series(truth)
    observed, observed_boundary, observed_interior = sparse_flat_indices()
    heldout = np.setdiff1d(np.arange(9 * 21 * 21), observed)
    signal_std = float(np.std(truth[:analysis_end]))
    process_std = float(config["physics"]["process_noise_fraction"]) * signal_std
    rng = np.random.default_rng(2026081900 + int(config["assimilation"]["seed"]))
    max_candidates = len(config["physics"]["candidate_speed_m_s"])
    ensemble = int(config["assimilation"]["ensemble_size"])
    initial_noise = (0.1 * process_std * rng.standard_normal((max_candidates, ensemble, 2, 9, 21, 21))).astype(np.float32)
    cfl = [cfl_number(speed, 1.0 / rate, float(config["grid"]["spacing_m"])) for speed in config["physics"]["candidate_speed_m_s"]]
    if max(cfl) >= 1.0:
        raise ValueError(f"unstable CFL numbers {cfl}")
    args.output.mkdir(parents=True, exist_ok=True)
    baseline_metrics = evaluate(truth, interpolation, np.zeros_like(interpolation), heldout, analysis_end, rate)
    (args.output / "trilinear_baseline.json").write_text(json.dumps({"method": "trilinear", **baseline_metrics}, indent=2), encoding="utf-8")
    completed = []
    for method in METHODS:
        print(f"method={method} device={args.device}", flush=True)
        result = run(
            method, truth, interpolation, observed_interior,
            np.asarray(config["physics"]["candidate_speed_m_s"], dtype=float),
            config, args.device, initial_noise,
            noise_seed=2026081917, process_noise_std=process_std,
        )
        metrics = evaluate(truth, result.mean, result.standard_deviation, heldout, analysis_end, rate)
        record = {
            "case": "meshir_s1_full_grid",
            "method": method,
            "device": args.device,
            "grid_shape_zyx": [9, 21, 21],
            "sparse_shape_zyx": [3, 7, 7],
            "observed_count": int(len(observed)),
            "boundary_observed_count": int(len(observed_boundary)),
            "interior_observed_count": int(len(observed_interior)),
            "heldout_count": int(len(heldout)),
            "candidate_speed_m_s": config["physics"]["candidate_speed_m_s"],
            "fitted_speed_m_s": fitted_speed,
            "recording_delay_seconds": float(recording_delay),
            "filter_delay_seconds": float(filter_delay),
            "crop_start_sample": int(crop_start),
            "model_band_hz": float(config["grid"]["model_band_hz"]),
            "cfl_numbers": cfl,
            "final_weights": result.final_weights.tolist(),
            "median_separation_ratio": float(np.median(result.separation_history)) if len(result.separation_history) else 0.0,
            "paired_noise_digest": result.paired_noise_digest,
            "future_observations_used": False,
            "reduced_order_model_used": False,
            **metrics,
            "completed": True,
        }
        stem = args.output / method.lower()
        stem.with_suffix(".json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        snapshot_indices = [int(0.02 * rate), int(0.04 * rate), analysis_end - 1, forecast_end - 1]
        np.savez_compressed(
            stem.with_suffix(".npz"),
            snapshot_indices=np.asarray(snapshot_indices),
            truth_snapshots=truth[snapshot_indices],
            mean_snapshots=result.mean[snapshot_indices],
            std_snapshots=result.standard_deviation[snapshot_indices],
            final_weights=result.final_weights,
            weight_time=result.weight_time,
            weight_history=result.weight_history,
            observed_indices=observed,
            observed_boundary_indices=observed_boundary,
            observed_interior_indices=observed_interior,
            positions=positions,
            mapping=mapping,
        )
        completed.append(str(stem.with_suffix(".json")))
    manifest = {
        "case": "meshir_s1_full_grid",
        "authoritative_rir": str(args.rir),
        "authoritative_geometry": str(args.geometry),
        "methods": list(METHODS),
        "completed_runs": len(completed),
        "records": completed,
        "full_state_dimension": 2 * 9 * 21 * 21,
        "fitted_speed_m_s": fitted_speed,
        "recording_delay_seconds": float(recording_delay),
        "crop_start_sample": int(crop_start),
        "model_band_hz": float(config["grid"]["model_band_hz"]),
        "reduced_order_model_used": False,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
