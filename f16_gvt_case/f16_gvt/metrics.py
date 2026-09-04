from __future__ import annotations

import numpy as np

from .assimilation import PassResult


def weighted_crps(samples: np.ndarray, weights: np.ndarray, target: float) -> float:
    order = np.argsort(samples)
    values = samples[order]
    probability = weights[order] / max(float(weights.sum()), 1e-12)
    first = float(np.sum(probability * np.abs(values - target)))
    cumulative_before = np.cumsum(probability) - probability
    half_pairwise = float(np.sum(probability * values * (2.0 * cumulative_before + probability - 1.0)))
    return first - half_pairwise


def evaluate_pass(
    result: PassResult,
    truth: np.ndarray,
    heldout_index: int,
    burnin_samples: int,
    sample_rate_hz: float,
    frequency_band_hz: tuple[float, float],
) -> dict:
    target = truth[burnin_samples:, heldout_index]
    estimate = result.mean_physical[burnin_samples:, heldout_index]
    denominator = max(float(np.linalg.norm(target)), 1e-12)
    nrmse = float(np.linalg.norm(estimate - target) / denominator)
    erms = float(np.sqrt(np.mean((estimate - target) ** 2)))
    valid = result.evaluation_indices >= burnin_samples
    indices = result.evaluation_indices[valid]
    support = result.support_physical[valid][:, :, heldout_index]
    weights = result.support_weights[valid]
    eval_truth = truth[indices, heldout_index]
    crps = [weighted_crps(support[index], weights[index], eval_truth[index]) for index in range(len(indices))]
    truth_rms = max(float(np.sqrt(np.mean(eval_truth ** 2))), 1e-12)
    lower = result.lower90_physical[valid, heldout_index]
    upper = result.upper90_physical[valid, heldout_index]
    coverage = float(np.mean((eval_truth >= lower) & (eval_truth <= upper)))
    width = float(np.mean(upper - lower) / truth_rms)
    frequency = np.fft.rfftfreq(len(target), 1.0 / sample_rate_hz)
    mask = (frequency >= frequency_band_hz[0]) & (frequency <= frequency_band_hz[1])
    truth_spectrum = np.abs(np.fft.rfft(target))[mask]
    estimate_spectrum = np.abs(np.fft.rfft(estimate))[mask]
    spectral_nrmse = float(np.linalg.norm(estimate_spectrum - truth_spectrum) / max(np.linalg.norm(truth_spectrum), 1e-12))
    branch_nrmse = []
    for candidate in range(result.branch_mean_physical.shape[1]):
        candidate_estimate = result.branch_mean_physical[burnin_samples:, candidate, heldout_index]
        branch_nrmse.append(float(np.linalg.norm(candidate_estimate - target) / denominator))
    branch_nrmse = np.asarray(branch_nrmse)
    oracle = int(np.argmin(branch_nrmse))
    blackout_eval = result.blackout_mask[burnin_samples:]
    blackout_nrmse = float(np.linalg.norm((estimate - target)[blackout_eval]) / max(np.linalg.norm(target[blackout_eval]), 1e-12)) if blackout_eval.any() else float("nan")
    return {
        "heldout_nrmse": nrmse,
        "heldout_erms": erms,
        "normalized_crps": float(np.mean(crps) / truth_rms),
        "coverage_90": coverage,
        "normalized_interval_width_90": width,
        "spectral_nrmse_6_8p5hz": spectral_nrmse,
        "blackout_heldout_nrmse": blackout_nrmse,
        "oracle_candidate_nrmse": float(branch_nrmse[oracle]),
        "oracle_candidate_index": oracle,
        "oracle_candidate_alpha": float(result.grid[oracle]),
        "candidate_nrmse": branch_nrmse.tolist(),
        "mean_candidate_separation_ratio": float(np.mean(result.separation_history)) if result.separation_history.size else 0.0,
        "minimum_candidate_separation_ratio": float(np.min(result.separation_history)) if result.separation_history.size else 0.0,
        "final_effective_candidate_count": float(1.0 / np.sum(result.final_weights ** 2)),
    }
