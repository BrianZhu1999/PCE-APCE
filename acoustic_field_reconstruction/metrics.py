"""Metrics for the held-out MeshRIR reconstruction regions."""

from __future__ import annotations

import numpy as np


def nrmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    denominator = max(float(np.sum(truth**2)), 1e-20)
    return float(np.sqrt(np.sum((prediction - truth) ** 2) / denominator))


def correlation(truth: np.ndarray, prediction: np.ndarray) -> float:
    reference = truth.reshape(-1)
    estimate = prediction.reshape(-1)
    if np.std(reference) < 1e-20 or np.std(estimate) < 1e-20:
        return 0.0
    return float(np.corrcoef(reference, estimate)[0, 1])


def evaluate_region(
    truth: np.ndarray,
    mean: np.ndarray,
    standard_deviation: np.ndarray,
    flat_indices: np.ndarray,
    analysis_end: int,
    sample_rate: float,
    prefix: str,
) -> dict[str, float]:
    truth_flat = truth.reshape(len(truth), -1)[:, flat_indices]
    mean_flat = mean.reshape(len(mean), -1)[:, flat_indices]
    std_flat = standard_deviation.reshape(len(standard_deviation), -1)[:, flat_indices]
    output = {
        f"{prefix}_analysis_nrmse": nrmse(truth_flat[:analysis_end], mean_flat[:analysis_end]),
        f"{prefix}_analysis_correlation": correlation(truth_flat[:analysis_end], mean_flat[:analysis_end]),
        f"{prefix}_analysis_mae": float(np.mean(np.abs(truth_flat[:analysis_end] - mean_flat[:analysis_end]))),
    }
    for milliseconds in (1, 2, 4):
        end = min(analysis_end + int(milliseconds * sample_rate / 1000), len(truth))
        output[f"{prefix}_forecast_{milliseconds}ms_nrmse"] = nrmse(
            truth_flat[analysis_end:end], mean_flat[analysis_end:end]
        )
        output[f"{prefix}_forecast_{milliseconds}ms_correlation"] = correlation(
            truth_flat[analysis_end:end], mean_flat[analysis_end:end]
        )
    lower = mean_flat - 1.6448536269514722 * std_flat
    upper = mean_flat + 1.6448536269514722 * std_flat
    output[f"{prefix}_coverage_90"] = float(np.mean((truth_flat >= lower) & (truth_flat <= upper)))
    output[f"{prefix}_mean_interval_width"] = float(np.mean(upper - lower))
    return output
