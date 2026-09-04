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
    return output / np.sum(output)


def entropy_project(weights: np.ndarray, target: float) -> np.ndarray:
    weights = np.maximum(np.asarray(weights, dtype=float), 1e-300)
    weights /= weights.sum()
    if entropy(weights) >= target:
        return weights
    uniform = np.full_like(weights, 1.0 / len(weights))
    lo, hi = 0.0, 1.0
    for _ in range(64):
        middle = 0.5 * (lo + hi)
        candidate = (1.0 - middle) * weights + middle * uniform
        if entropy(candidate) >= target:
            hi = middle
        else:
            lo = middle
    output = (1.0 - hi) * weights + hi * uniform
    return output / np.sum(output)


def gaussian_score(samples: np.ndarray, observation: np.ndarray, measurement_covariance: np.ndarray, shrinkage: float) -> float:
    covariance = np.atleast_2d(np.cov(samples.T, bias=False))
    covariance = (1.0 - shrinkage) * covariance + shrinkage * np.diag(np.diag(covariance)) + measurement_covariance
    covariance = 0.5 * (covariance + covariance.T) + 1e-8 * np.eye(len(covariance))
    residual = observation - samples.mean(axis=0)
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        return -np.inf
    return -0.5 * float(residual @ np.linalg.pinv(covariance) @ residual + logdet) / len(observation)


def update_weights(method: str, weights: np.ndarray, log_weights: np.ndarray, scores: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    centered = np.asarray(scores, dtype=float) - float(np.mean(scores))
    if method == "bma":
        temperature = float(config["bma_temperature"])
        log_weights = log_weights + temperature * centered
        weights = softmax(log_weights)
    elif method == "pce":
        temperature = float(config["pce_temperature"])
        log_weights = log_weights + temperature * centered
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
    return weights, log_weights, {"entropy": entropy(weights), "temperature": temperature}
