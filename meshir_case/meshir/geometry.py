from __future__ import annotations

import numpy as np


def idw_extension(source_positions: np.ndarray, source_values: np.ndarray, target_positions: np.ndarray, neighbors: int) -> np.ndarray:
    source_positions = np.asarray(source_positions, dtype=float)
    target_positions = np.asarray(target_positions, dtype=float)
    source_values = np.asarray(source_values, dtype=float)
    output = np.empty((len(target_positions), source_values.shape[1]), dtype=float)
    k = min(neighbors, len(source_positions))
    for index, point in enumerate(target_positions):
        distance = np.linalg.norm(source_positions - point[None, :], axis=1)
        nearest = np.argpartition(distance, k - 1)[:k]
        if np.min(distance[nearest]) < 1e-12:
            output[index] = source_values[nearest[np.argmin(distance[nearest])]]
            continue
        weight = 1.0 / np.maximum(distance[nearest], 1e-12) ** 2
        output[index] = np.sum(source_values[nearest] * weight[:, None], axis=0) / np.sum(weight)
    return output


def direct_toa(source: np.ndarray, receivers: np.ndarray, speed: float) -> np.ndarray:
    return np.linalg.norm(np.asarray(receivers) - np.asarray(source)[None, :], axis=1) / speed


def geometric_localization(observed_rir: np.ndarray, observed_positions: np.ndarray, candidate_sources: np.ndarray, speed: float, sample_rate: float, threshold_fraction: float = 0.08) -> tuple[np.ndarray, float]:
    observed_rir = np.asarray(observed_rir, dtype=float)
    window = np.abs(observed_rir[:, :int(0.02 * sample_rate)])
    peaks = []
    for channel in range(window.shape[0]):
        threshold = threshold_fraction * float(np.max(window[channel]))
        hit = np.flatnonzero(window[channel] >= threshold)
        peaks.append(float(hit[0]) / sample_rate if len(hit) else 0.02)
    peaks = np.asarray(peaks)
    errors = []
    for source in candidate_sources:
        errors.append(float(np.mean((peaks - direct_toa(source, observed_positions, speed)) ** 2)))
    index = int(np.argmin(errors))
    return np.asarray(candidate_sources[index], dtype=float), float(np.sqrt(errors[index]))
