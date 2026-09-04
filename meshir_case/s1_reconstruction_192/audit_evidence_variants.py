#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    result = np.exp(np.clip(shifted, -700.0, 0.0))
    return result / result.sum()


def entropy(weights: np.ndarray) -> float:
    safe = np.maximum(weights, 1e-300)
    return float(-np.sum(safe * np.log(safe)))


def project_entropy(weights: np.ndarray, fraction: float) -> np.ndarray:
    target = float(fraction * np.log(len(weights)))
    if entropy(weights) >= target:
        return weights
    uniform = np.full_like(weights, 1.0 / len(weights))
    lo, hi = 0.0, 1.0
    for _ in range(64):
        middle = (lo + hi) / 2.0
        candidate = (1.0 - middle) * weights + middle * uniform
        if entropy(candidate) >= target:
            hi = middle
        else:
            lo = middle
    return (1.0 - hi) * weights + hi * uniform


def update_variant(name: str, weights: np.ndarray, log_weights: np.ndarray, scores: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray]:
    delta = scores - scores.mean()
    spread = float(np.std(delta))
    if name.startswith("normalized"):
        delta = delta / max(spread, 1e-6)
    if name.startswith("gated") and spread < 0.02:
        return weights, log_weights
    if name.startswith("block4") and index % 4 != 3:
        return weights, log_weights
    temperature = {
        "pce_raw": 0.08,
        "normalized_0.02": 0.02,
        "normalized_0.04": 0.04,
        "normalized_0.08": 0.08,
        "gated_0.04": 0.04,
        "block4_0.16": 0.16,
    }[name]
    if name.startswith("gated") and spread < 0.04:
        return weights, log_weights
    if name.startswith("block4"):
        log_weights = log_weights + temperature * delta
    else:
        log_weights = log_weights + temperature * delta
    weights = softmax(log_weights)
    if name == "apce_current":
        weights = project_entropy(weights, 0.20)
        log_weights = np.log(np.maximum(weights, 1e-300))
    return weights, log_weights


def reconstruct(weights_history: list[np.ndarray], candidate_mean: np.ndarray, heldout: np.ndarray, truth: np.ndarray) -> float:
    analysis_end = min(1024, len(truth))
    prediction = np.zeros_like(truth[:analysis_end])
    uniform = np.full(candidate_mean.shape[0], 1.0 / candidate_mean.shape[0])
    update_index = 0
    for time_index in range(analysis_end):
        if update_index < len(weights_history) and time_index >= 16 * (update_index + 1):
            weights = weights_history[update_index]
            update_index += 1
        else:
            weights = uniform
        prediction[time_index] = np.sum(weights[:, None, None, None] * candidate_mean[:, time_index], axis=0)
    truth_flat = truth[:analysis_end].reshape(analysis_end, -1)[:, heldout]
    prediction_flat = prediction.reshape(analysis_end, -1)[:, heldout]
    return float(np.sqrt(np.sum((prediction_flat - truth_flat) ** 2) / max(np.sum(truth_flat ** 2), 1e-20)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.npz) as data:
        scores = np.asarray(data["score_history"], dtype=float)
        candidate_mean = np.asarray(data["candidate_mean"], dtype=np.float32)
        truth = np.asarray(data["truth"], dtype=np.float32)
        heldout = np.asarray(data["heldout_interior"], dtype=int)
    names = ("pce_raw", "normalized_0.02", "normalized_0.04", "normalized_0.08", "gated_0.04", "block4_0.16")
    records = []
    for name in names:
        weights = np.full(scores.shape[1], 1.0 / scores.shape[1])
        log_weights = np.zeros(scores.shape[1])
        history = []
        for index, score_row in enumerate(scores):
            weights, log_weights = update_variant(name, weights, log_weights, score_row, index)
            history.append(weights.copy())
        records.append({
            "variant": name,
            "analysis_nrmse": reconstruct(history, candidate_mean, heldout, truth),
            "final_weights": weights.tolist(),
            "final_entropy_fraction": entropy(weights) / np.log(len(weights)),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")
    print(json.dumps({"records": records}, indent=2))


if __name__ == "__main__":
    main()
