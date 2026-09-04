from __future__ import annotations

import numpy as np

from .geometry import idw_extension


def fit_pod(field: np.ndarray, rank: int, sample_stride: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    field = np.asarray(field, dtype=np.float64)
    snapshots = field[::sample_stride]
    snapshots = snapshots - snapshots.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(snapshots, full_matrices=False)
    basis = vt[:rank].T
    coefficients = (field - field.mean(axis=0, keepdims=True)) @ basis
    temporal_mean = field.mean(axis=0)
    return basis.astype(np.float32), coefficients.astype(np.float32), temporal_mean.astype(np.float32)


def build_spatial_basis(
    calibration_positions: np.ndarray,
    calibration_basis: np.ndarray,
    all_positions: np.ndarray,
    neighbors: int,
) -> np.ndarray:
    return idw_extension(calibration_positions, calibration_basis, all_positions, neighbors).astype(np.float32)


def estimate_decay_rate(coefficients: np.ndarray, sample_rate_hz: float) -> float:
    energy = np.sum(np.asarray(coefficients, dtype=float) ** 2, axis=1)
    time = np.arange(len(energy), dtype=float) / sample_rate_hz
    mask = (time >= 0.01) & (time <= min(0.20, time[-1])) & (energy > np.max(energy) * 1e-8)
    if mask.sum() < 5:
        return 1.0
    slope = np.polyfit(time[mask], np.log(np.maximum(energy[mask], 1e-20)), 1)[0]
    return float(max(-slope, 1e-3))


def retime_candidate(
    coefficients: np.ndarray,
    time: np.ndarray,
    speed_scale: float,
    damping_scale: float,
    decay_rate: float,
) -> np.ndarray:
    output = np.empty_like(coefficients, dtype=np.float32)
    for dimension in range(coefficients.shape[1]):
        output[:, dimension] = np.interp(
            time * speed_scale, time, coefficients[:, dimension], left=coefficients[0, dimension], right=0.0
        )
    output *= np.exp(-(damping_scale - 1.0) * decay_rate * time[:, None]).astype(np.float32)
    return output


def candidate_paths(
    coefficients: np.ndarray,
    sample_rate_hz: float,
    speed_scales: list[float],
    damping_scales: list[float],
    decay_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    time = np.arange(len(coefficients), dtype=float) / sample_rate_hz
    paths = np.stack([
        retime_candidate(coefficients, time, speed, damping, decay_rate)
        for speed, damping in zip(speed_scales, damping_scales)
    ]).astype(np.float32)
    parameters = np.asarray(list(zip(speed_scales, damping_scales)), dtype=np.float32)
    return paths, parameters


def residual_statistics(
    coefficient_trajectories: list[np.ndarray],
    candidate_mean: np.ndarray,
    shrinkage: float = 0.20,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    residuals = np.concatenate([
        trajectory - candidate_mean for trajectory in coefficient_trajectories
    ], axis=0)
    covariance = np.atleast_2d(np.cov(residuals.T, bias=False))
    covariance = (1.0 - shrinkage) * covariance + shrinkage * np.diag(np.diag(covariance))
    scale = max(float(np.mean(np.diag(covariance))), 1e-12)
    covariance += 1e-5 * scale * np.eye(covariance.shape[0])
    rho = float(np.clip(
        np.sum(residuals[:-1] * residuals[1:]) / max(float(np.sum(residuals[:-1] ** 2)), 1e-20),
        0.80, 0.995,
    ))
    process = max(1.0 - rho ** 2, 1e-4) * covariance
    observation = 1e-4 * np.eye(covariance.shape[0])
    return rho, covariance.astype(np.float32), process.astype(np.float32), observation.astype(np.float32)
