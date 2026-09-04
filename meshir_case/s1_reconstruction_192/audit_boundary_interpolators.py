#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.interpolate import RBFInterpolator

from fullwave.geometry import grid_mapping, to_grid
from geometry_192 import boundary_mask, sample_192
from run_stage_a import nrmse
from run_stage_c import fixed_speed_recording_delay


def idw_weights(source: np.ndarray, target: np.ndarray, neighbours: int, power: float) -> np.ndarray:
    distance = np.linalg.norm(target[:, None] - source[None], axis=2)
    count = min(int(neighbours), len(source))
    order = np.argpartition(distance, count - 1, axis=1)[:, :count]
    weights = np.zeros_like(distance)
    for row in range(len(target)):
        selected = order[row]
        exact = selected[distance[row, selected] < 1e-12]
        if len(exact):
            weights[row, exact[0]] = 1.0
        else:
            local = 1.0 / np.maximum(distance[row, selected], 1e-12) ** float(power)
            weights[row, selected] = local / local.sum()
    return weights


def rbf_weights(
    source: np.ndarray,
    target: np.ndarray,
    kernel: str,
    smoothing: float,
    epsilon: float | None,
) -> np.ndarray:
    kwargs = dict(kernel=kernel, smoothing=float(smoothing))
    if epsilon is not None:
        kwargs["epsilon"] = float(epsilon)
    interpolator = RBFInterpolator(source, np.eye(len(source)), **kwargs)
    return np.asarray(interpolator(target), dtype=np.float64)


def method_specs() -> list[dict[str, object]]:
    return [
        {"name": "idw_k4_p1", "family": "idw", "neighbours": 4, "power": 1.0},
        {"name": "idw_k8_p1", "family": "idw", "neighbours": 8, "power": 1.0},
        {"name": "idw_k8_p2", "family": "idw", "neighbours": 8, "power": 2.0},
        {"name": "idw_k16_p2", "family": "idw", "neighbours": 16, "power": 2.0},
        {"name": "rbf_thin_s0", "family": "rbf", "kernel": "thin_plate_spline", "smoothing": 0.0, "epsilon": None},
        {"name": "rbf_thin_s1e-4", "family": "rbf", "kernel": "thin_plate_spline", "smoothing": 1e-4, "epsilon": None},
        {"name": "rbf_thin_s1e-3", "family": "rbf", "kernel": "thin_plate_spline", "smoothing": 1e-3, "epsilon": None},
        {"name": "rbf_cubic_s1e-4", "family": "rbf", "kernel": "cubic", "smoothing": 1e-4, "epsilon": None},
        {"name": "rbf_linear_s1e-4", "family": "rbf", "kernel": "linear", "smoothing": 1e-4, "epsilon": None},
        {"name": "rbf_gaussian_e2", "family": "rbf", "kernel": "gaussian", "smoothing": 1e-4, "epsilon": 2.0},
        {"name": "rbf_gaussian_e4", "family": "rbf", "kernel": "gaussian", "smoothing": 1e-4, "epsilon": 4.0},
        {"name": "rbf_inverse_e4", "family": "rbf", "kernel": "inverse_multiquadric", "smoothing": 1e-4, "epsilon": 4.0},
    ]


def weights_for(spec: dict[str, object], source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if spec["family"] == "idw":
        return idw_weights(source, target, int(spec["neighbours"]), float(spec["power"]))
    return rbf_weights(
        source,
        target,
        str(spec["kernel"]),
        float(spec["smoothing"]),
        None if spec["epsilon"] is None else float(spec["epsilon"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rir", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rate, speed, analysis_end, forecast_end = 16000.0, 343.0, 1024, 1152
    full_field = np.load(args.rir, mmap_mode="r")
    with np.load(args.geometry) as geometry:
        positions = np.asarray(geometry["s1_positions"], dtype=float)
        source_position = np.asarray(geometry["s1_source"][0], dtype=float)
    mapping, *_ = grid_mapping(positions)
    grid_positions = positions[mapping.reshape(-1)]
    sample = sample_192()
    boundary_sensor = mapping.reshape(-1)[sample["boundary_flat"]]
    timing_window = min(int(0.25 * rate), len(full_field))
    delay = fixed_speed_recording_delay(
        full_field, positions, source_position, boundary_sensor, rate, speed, timing_window
    )
    kernel = signal.firwin(129, 750.0, fs=rate)
    filtered = signal.lfilter(kernel, [1.0], np.asarray(full_field[:timing_window], dtype=np.float32), axis=0)
    crop = int(round((delay + 0.004) * rate))
    truth = to_grid(np.asarray(filtered[crop:crop + forecast_end], dtype=np.float32), mapping)
    truth_flat = truth.reshape(len(truth), -1)
    boundary_all = np.flatnonzero(boundary_mask().reshape(-1))
    heldout_boundary = np.setdiff1d(boundary_all, sample["boundary_flat"])
    values = truth_flat[:analysis_end, sample["boundary_flat"]]
    specs = method_specs()
    rows = []
    for spec in specs:
        fold_errors = []
        for fold in range(4):
            validation_local = np.flatnonzero(np.arange(len(sample["boundary_flat"])) % 4 == fold)
            training_local = np.setdiff1d(np.arange(len(sample["boundary_flat"])), validation_local)
            weights = weights_for(
                spec,
                grid_positions[sample["boundary_flat"][training_local]],
                grid_positions[sample["boundary_flat"][validation_local]],
            )
            prediction = values[:, training_local] @ weights.T
            fold_errors.append(nrmse(values[:, validation_local], prediction))
        full_weights = weights_for(
            spec,
            grid_positions[sample["boundary_flat"]],
            grid_positions[heldout_boundary],
        )
        heldout_prediction = values @ full_weights.T
        heldout_error = nrmse(truth_flat[:analysis_end, heldout_boundary], heldout_prediction)
        rows.append({
            "method": str(spec["name"]),
            "cv_mean_nrmse": float(np.mean(fold_errors)),
            "cv_std_nrmse": float(np.std(fold_errors, ddof=1)),
            "heldout_boundary_nrmse_diagnostic": heldout_error,
            "fold_nrmse": fold_errors,
        })
        print(rows[-1], flush=True)
    rows.sort(key=lambda row: row["cv_mean_nrmse"])
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "boundary_interpolator_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "cv_mean_nrmse", "cv_std_nrmse", "heldout_boundary_nrmse_diagnostic"],
        )
        writer.writeheader()
        writer.writerows([{key: row[key] for key in writer.fieldnames} for row in rows])
    report = {
        "selection_basis": "four-fold pseudo-holdout within the 128 observed boundary points",
        "formal_heldout_used_for_selection": False,
        "formal_heldout_reported_for_diagnostic_only": True,
        "ranked_methods": rows,
    }
    (args.output / "boundary_interpolator_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
