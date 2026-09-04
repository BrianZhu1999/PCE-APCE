from __future__ import annotations

import math
import numpy as np


def entropy(weights: np.ndarray) -> float:
    weights = np.maximum(np.asarray(weights, dtype=float), 1e-300)
    return -float(np.sum(weights * np.log(weights)))


def softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    shifted = values - np.max(values)
    output = np.exp(np.clip(shifted, -700.0, 0.0))
    return output / output.sum()


def entropy_project(weights: np.ndarray, target: float) -> np.ndarray:
    weights = np.maximum(np.asarray(weights, dtype=float), 1e-300)
    weights /= weights.sum()
    if entropy(weights) >= target:
        return weights
    uniform = np.full_like(weights, 1.0 / len(weights))
    lo, hi = 0.0, 1.0
    for _ in range(64):
        middle = 0.5 * (lo + hi)
        candidate = (1 - middle) * weights + middle * uniform
        if entropy(candidate) >= target:
            hi = middle
        else:
            lo = middle
    output = (1 - hi) * weights + hi * uniform
    return output / output.sum()


def score(samples: np.ndarray, observation: np.ndarray, variance: float, shrinkage: float) -> float:
    mean = samples.mean(axis=0)
    sample_variance = np.var(samples, axis=0, ddof=1)
    common = float(np.mean(sample_variance))
    total = (1 - shrinkage) * sample_variance + shrinkage * common + variance
    total = np.maximum(total, 1e-16)
    return -0.5 * float(np.mean((observation - mean) ** 2 / total + np.log(total)))


def update(method: str, weights: np.ndarray, log_weights: np.ndarray, scores: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray]:
    centered = scores - scores.mean()
    if method == "bma":
        log_weights = log_weights + float(config["bma_temperature"]) * centered
        weights = softmax(log_weights)
    elif method == "pce":
        log_weights = log_weights + float(config["pce_temperature"]) * centered
        weights = softmax(log_weights)
    elif method == "apce":
        ratio = entropy(weights) / max(math.log(len(weights)), 1e-12)
        temperature = float(np.clip(float(config["apce_temperature"]) * ratio ** 0.75, float(config["apce_min_temperature"]), float(config["apce_temperature"])))
        log_weights = float(config["apce_forgetting"]) * log_weights + temperature * centered
        weights = softmax(log_weights)
        weights = entropy_project(weights, float(config["apce_entropy_fraction"]) * math.log(len(weights)))
        log_weights = np.log(np.maximum(weights, 1e-300))
    else:
        raise ValueError(method)
    return weights, log_weights
