from __future__ import annotations

import numpy as np
from scipy.special import erf

from .data import causal_log_energy


def nrmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.sum((prediction - truth) ** 2) / max(np.sum(truth ** 2), 1e-20)))


def mae(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.abs(prediction - truth)))


def correlation(truth: np.ndarray, prediction: np.ndarray) -> float:
    a, b = np.asarray(truth).reshape(-1), np.asarray(prediction).reshape(-1)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def log_spectral_distance(truth: np.ndarray, prediction: np.ndarray, rate: float) -> float:
    window = np.hanning(truth.shape[0])[:, None]
    a = np.fft.rfft(truth * window, axis=0)
    b = np.fft.rfft(prediction * window, axis=0)
    freq = np.fft.rfftfreq(truth.shape[0], 1.0 / rate)
    mask = (freq >= 200.0) & (freq <= min(7000.0, rate / 2 - 1))
    return float(np.sqrt(np.mean((20 * np.log10(np.maximum(np.abs(a[mask]), 1e-12)) - 20 * np.log10(np.maximum(np.abs(b[mask]), 1e-12))) ** 2)))


def energy_rmse(truth: np.ndarray, prediction: np.ndarray, rate: float) -> float:
    a = causal_log_energy(truth.T, rate, 0.005, 1e-12).T
    b = causal_log_energy(prediction.T, rate, 0.005, 1e-12).T
    return float(np.sqrt(np.mean((a - b) ** 2)))


def normal_crps(mean: np.ndarray, std: np.ndarray, observation: np.ndarray) -> np.ndarray:
    std = np.maximum(std, 1e-8)
    z = (observation - mean) / std
    phi = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    cdf = 0.5 * (1 + erf(z / np.sqrt(2)))
    return std * (z * (2 * cdf - 1) + 2 * phi - 1 / np.sqrt(np.pi))


def evaluate_field(
    truth: np.ndarray,
    mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    heldout: np.ndarray,
    analysis_end: int,
    prediction_end: int,
    rate: float,
) -> dict[str, float]:
    heldout = np.asarray(heldout, dtype=int)
    analysis = slice(0, analysis_end)
    prediction = slice(analysis_end, prediction_end)
    truth_h = truth[:, heldout]
    mean_h = mean[:, heldout]
    valid = np.any(np.abs(lower) > 0, axis=1)
    valid_indices = np.flatnonzero(valid)
    coverage = np.mean((truth[valid_indices][:, heldout] >= lower[valid_indices][:, heldout]) & (truth[valid_indices][:, heldout] <= upper[valid_indices][:, heldout])) if len(valid_indices) else np.nan
    std = np.maximum((upper[valid_indices][:, heldout] - lower[valid_indices][:, heldout]) / 3.2897072539, 1e-8) if len(valid_indices) else np.ones((1, len(heldout)))
    sampled_truth = truth[valid_indices][:, heldout] if len(valid_indices) else np.zeros_like(std)
    sampled_mean = mean[valid_indices][:, heldout] if len(valid_indices) else np.zeros_like(std)
    return {
        "reconstruction_nrmse": nrmse(truth_h[analysis], mean_h[analysis]),
        "reconstruction_mae": mae(truth_h[analysis], mean_h[analysis]),
        "reconstruction_lsd_db": log_spectral_distance(truth_h[analysis], mean_h[analysis], rate),
        "reconstruction_energy_rmse_db": energy_rmse(truth_h[analysis], mean_h[analysis], rate),
        "prediction_nrmse": nrmse(truth_h[prediction], mean_h[prediction]),
        "prediction_mae": mae(truth_h[prediction], mean_h[prediction]),
        "prediction_correlation": correlation(truth_h[prediction], mean_h[prediction]),
        "coverage_90": float(coverage),
        "crps": float(np.mean(normal_crps(sampled_mean, std, sampled_truth))),
        "mean_interval_width": float(np.mean(upper[valid_indices][:, heldout] - lower[valid_indices][:, heldout])) if len(valid_indices) else np.nan,
    }
