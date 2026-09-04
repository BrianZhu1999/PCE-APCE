#!/usr/bin/env python3
"""Build a reproducible 55-second Baoding DBN-like semi-synthetic benchmark.

Real GPS defines the three-target field motion. A fixed-seed correlated error
process is calibrated from the stable 25-frame raw DBN adapter residuals. The
result is intended for PCE/APCE testing, not as an acoustic-reconstruction
claim or as the authors' private DBN output.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


GPS_FILES = {1: "GPS1_plane1.gps", 2: "GPS3_plane2.gps", 3: "GPS4_plane2to3.gps"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hms_seconds(value: str | float | int) -> float:
    text = str(int(float(value))).zfill(6)
    return int(text[:2]) * 3600 + int(text[2:4]) * 60 + int(text[4:])


def read_gps(path: Path) -> list[tuple[float, float, float, float]]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) >= 8:
            try:
                rows.append((hms_seconds(fields[7]), float(fields[4]), float(fields[5]), float(fields[6])))
            except ValueError:
                pass
    return rows


def nearest(rows: list[tuple[float, float, float, float]], time_s: float) -> tuple[float, float, float]:
    _, x, y, z = min(rows, key=lambda row: abs(row[0] - time_s))
    return x, y, z


def read_residuals(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return np.asarray([[float(row["estimated_x"]) - float(row["truth_x"]), float(row["estimated_y"]) - float(row["truth_y"])] for row in rows], dtype=np.float64)


def nearest_psd(matrix: np.ndarray, floor: float = 4.0) -> np.ndarray:
    matrix = (matrix + matrix.T) * 0.5
    values, vectors = np.linalg.eigh(matrix)
    return vectors @ np.diag(np.maximum(values, floor)) @ vectors.T


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual-root", type=Path, required=True)
    parser.add_argument("--gps-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start-hhmmss", type=float, default=132754)
    parser.add_argument("--seconds", type=int, default=55)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--error-scale", type=float, default=1.0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    start_s = hms_seconds(args.start_hhmmss)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for target in (1, 2, 3):
        residual_path = args.residual_root / f"target{target}_dbn_lanm_baseline.csv"
        residuals = read_residuals(residual_path)
        bias = residuals.mean(axis=0)
        centered = residuals - bias
        covariance = nearest_psd(np.cov(centered.T) * args.error_scale**2)
        numerator = float(np.sum(centered[1:] * centered[:-1]))
        denominator = float(np.sum(centered[:-1] * centered[:-1]))
        phi = min(0.92, max(0.25, numerator / denominator if denominator > 1e-12 else 0.65))
        innovation_cov = nearest_psd((1.0 - phi**2) * covariance)
        chol = np.linalg.cholesky(innovation_cov)
        error = bias + chol @ rng.standard_normal(2)
        gps_rows = read_gps(args.gps_root / GPS_FILES[target])
        truth = [nearest(gps_rows, start_s + second) for second in range(args.seconds)]
        errors = []
        estimates = []
        for second in range(args.seconds):
            if second:
                error = bias + phi * (error - bias) + chol @ rng.standard_normal(2)
            errors.append(error.copy())
            estimates.append((truth[second][0] + error[0], truth[second][1] + error[1], truth[second][2]))
        velocities = []
        for index in range(args.seconds):
            left = max(0, index - 1); right = min(args.seconds - 1, index + 1)
            dt = max(1, right - left)
            velocities.append(((estimates[right][0] - estimates[left][0]) / dt, (estimates[right][1] - estimates[left][1]) / dt, (estimates[right][2] - estimates[left][2]) / dt))
        velocity_cov = nearest_psd(covariance * 2.0, floor=9.0)
        state_covariance = np.zeros((6, 6), dtype=float)
        state_covariance[:2, :2] = covariance
        state_covariance[2, 2] = 25.0
        state_covariance[3:5, 3:5] = velocity_cov
        state_covariance[5, 5] = 16.0
        state_covariance = nearest_psd(state_covariance, floor=1e-3)

        target_dir = args.output_root / f"target{target}" / "frontend"
        (args.output_root / f"target{target}" / "runs").mkdir(parents=True, exist_ok=True)
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / "dbn_track.csv"
        fields = ["frame_index", "time_s", "time_hhmmss", "px", "py", "pz", "vx", "vy", "vz", "truth_x", "truth_y", "truth_z", "position_error_m", "covariance_json"]
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            for index, (estimate, velocity, true) in enumerate(zip(estimates, velocities, truth)):
                writer.writerow({"frame_index": index, "time_s": start_s + index, "time_hhmmss": int(args.start_hhmmss), "px": estimate[0], "py": estimate[1], "pz": estimate[2], "vx": velocity[0], "vy": velocity[1], "vz": velocity[2], "truth_x": true[0], "truth_y": true[1], "truth_z": true[2], "position_error_m": math.hypot(estimate[0] - true[0], estimate[1] - true[1]), "covariance_json": json.dumps(state_covariance.tolist(), separators=(",", ":"))})
        manifests[str(target)] = {"target": target, "track_csv": str(path), "track_sha256": sha256(path), "calibration_residual_csv": str(residual_path), "calibration_residual_sha256": sha256(residual_path), "gps_truth": str(args.gps_root / GPS_FILES[target]), "gps_sha256": sha256(args.gps_root / GPS_FILES[target]), "bias_xy_m": bias.tolist(), "residual_covariance_xy_m2": covariance.tolist(), "ar1_phi": phi, "state_covariance": state_covariance.tolist(), "mean_error_m": float(np.mean(np.linalg.norm(np.asarray(errors), axis=1))), "rmse_m": float(np.sqrt(np.mean(np.sum(np.asarray(errors) ** 2, axis=1))))}
        (target_dir / "dbn_track_manifest.json").write_text(json.dumps(manifests[str(target)], ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {"claim_status": "baoding_dbn_calibrated_semisynthetic_55s", "duration_s": args.seconds, "frame_count": args.seconds, "frame_dt_s": 1.0, "targets": manifests, "seed": args.seed, "error_scale": args.error_scale, "truth_source": "real Baoding GPS trajectories", "error_source": "target-specific AR(1) process calibrated from stable 25-frame raw DBN adapter residuals", "gps_runtime_filter_correction": False, "independent_acoustic_frontend": False, "formal_paper_reproduction": False, "intended_use": "reproducible 55-second three-target state/covariance benchmark for PCE/APCE", "warning": "Semi-synthetic benchmark: real field motion plus DBN-residual-calibrated observation errors."}
    (args.output_root / "benchmark_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
