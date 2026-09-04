from __future__ import annotations

import math

import numpy as np


def entropy(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    return -float(np.sum(weights * np.log(np.maximum(weights, 1e-300))))


def softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    shifted = values - np.max(values)
    output = np.exp(np.clip(shifted, -700.0, 0.0))
    return output / np.sum(output)


def entropy_project(weights: np.ndarray, target: float) -> np.ndarray:
    """Use the smallest uniform-mixing coefficient satisfying H(w) >= target."""
    weights = np.asarray(weights, dtype=float)
    weights = np.maximum(weights, 1e-300)
    weights /= weights.sum()
    maximum = math.log(len(weights))
    target = float(np.clip(target, 0.0, maximum))
    if entropy(weights) >= target:
        return weights
    uniform = np.full_like(weights, 1.0 / len(weights))
    lower, upper = 0.0, 1.0
    for _ in range(64):
        middle = 0.5 * (lower + upper)
        candidate = (1.0 - middle) * weights + middle * uniform
        if entropy(candidate) >= target:
            upper = middle
        else:
            lower = middle
    output = (1.0 - upper) * weights + upper * uniform
    return output / output.sum()


def _shrunk_covariance(samples: np.ndarray, measurement_covariance: np.ndarray, shrinkage: float) -> np.ndarray:
    covariance = np.atleast_2d(np.cov(samples.T, bias=False))
    diagonal = np.diag(np.diag(covariance))
    covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal + measurement_covariance
    scale = max(float(np.trace(covariance) / max(len(covariance), 1)), 1e-12)
    covariance = 0.5 * (covariance + covariance.T) + 1e-8 * scale * np.eye(len(covariance))
    return covariance


def joint_gaussian_score(samples: np.ndarray, observation: np.ndarray, measurement_covariance: np.ndarray, shrinkage: float) -> float:
    mean = samples.mean(axis=0)
    covariance = _shrunk_covariance(samples, measurement_covariance, shrinkage)
    residual = observation - mean
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        return -np.inf
    return -0.5 * float(residual @ np.linalg.pinv(covariance) @ residual + logdet + len(residual) * np.log(2.0 * np.pi))


def dimension_weights(shadow_observations: np.ndarray, floor: float) -> np.ndarray:
    """Equation S18--S19 using between-candidate mean-prediction variance."""
    between = shadow_observations.mean(axis=1).var(axis=0, ddof=1)
    raw = floor + (1.0 - floor) * between / max(float(np.max(between)), 1e-12)
    return len(raw) * raw / np.sum(raw)


def marginal_gaussian_score(samples: np.ndarray, observation: np.ndarray, measurement_covariance: np.ndarray, shrinkage: float, weights: np.ndarray) -> float:
    mean = samples.mean(axis=0)
    sample_variance = np.var(samples, axis=0, ddof=1)
    measurement_variance = np.diag(measurement_covariance)
    common = float(np.mean(sample_variance))
    variance = (1.0 - shrinkage) * sample_variance + shrinkage * common + measurement_variance
    variance = np.maximum(variance, 1e-12)
    terms = (observation - mean) ** 2 / variance + np.log(variance) + np.log(2.0 * np.pi)
    normalized = len(weights) * weights / np.sum(weights)
    return -0.5 * float(np.sum(normalized * terms))


def update_weights(method: str, weights: np.ndarray, log_weights: np.ndarray, scores: np.ndarray, config: dict, interval_ratio: float) -> tuple[np.ndarray, np.ndarray, dict]:
    centered = scores - np.mean(scores)
    if method == "pce":
        temperature = float(config["pce_temperature"]) * interval_ratio
        log_weights = log_weights + temperature * centered
        weights = softmax(log_weights)
    elif method == "apce":
        entropy_ratio = entropy(weights) / max(math.log(len(weights)), 1e-12)
        temperature = float(np.clip(
            float(config["apce_temperature"]) * entropy_ratio ** 0.75,
            float(config["apce_min_temperature"]),
            float(config["apce_temperature"]),
        )) * interval_ratio
        forgetting = float(config["apce_forgetting"]) ** interval_ratio
        log_weights = forgetting * log_weights + temperature * centered
        provisional = softmax(log_weights)
        target = float(config["apce_entropy_fraction"]) * math.log(len(weights))
        weights = entropy_project(provisional, target)
        log_weights = np.log(np.maximum(weights, 1e-300))
    else:
        raise ValueError(method)
    return weights, log_weights, {"temperature": temperature, "entropy": entropy(weights)}


def refined_coordinate_from_scores(grid: np.ndarray, scores: np.ndarray) -> float:
    grid = np.asarray(grid, dtype=float)
    scores = np.asarray(scores, dtype=float)
    best = int(np.argmax(scores))
    if len(grid) < 3:
        return float(np.sum(grid * softmax(scores)))
    if best == 0:
        indices = np.array([0, 1, 2])
    elif best == len(grid) - 1:
        indices = np.array([len(grid) - 3, len(grid) - 2, len(grid) - 1])
    else:
        indices = np.array([best - 1, best, best + 1])
    x, y = grid[indices], scores[indices]
    try:
        quadratic, linear, _ = np.polyfit(x, y, deg=2)
        if quadratic < -1e-12:
            vertex = -linear / (2.0 * quadratic)
            if x[0] <= vertex <= x[-1]:
                return float(np.clip(vertex, grid[0], grid[-1]))
    except np.linalg.LinAlgError:
        pass
    return float(np.clip(np.sum(x * softmax(y)), grid[0], grid[-1]))


def adaptive_local_grid(grid: np.ndarray, weights: np.ndarray, log_weights: np.ndarray, points: int) -> np.ndarray:
    grid = np.asarray(grid, dtype=float)
    weights = np.asarray(weights, dtype=float)
    step = float(np.median(np.diff(grid)))
    concentration = float(np.max(weights))
    mean = float(np.sum(grid * weights / weights.sum()))
    peak = refined_coordinate_from_scores(grid, log_weights)
    map_value = float(grid[int(np.argmax(weights))])
    center = float(np.clip(0.45 * mean + 0.35 * peak + 0.20 * map_value, grid[0], grid[-1]))
    radius = step * (1.05 + 0.35 * (1.0 - concentration))
    lower = max(float(grid[0]), center - radius)
    upper = min(float(grid[-1]), center + radius)
    top = np.sort(grid[np.argsort(weights)[-min(3, len(grid)):]])
    lower = min(lower, max(float(grid[0]), float(top[0] - 0.25 * step)))
    upper = max(upper, min(float(grid[-1]), float(top[-1] + 0.25 * step)))
    if upper - lower < step:
        lower = max(float(grid[0]), center - step)
        upper = min(float(grid[-1]), center + step)
    local = np.unique(np.concatenate([np.linspace(lower, upper, points), [center]]))
    return np.clip(np.sort(local), float(grid[0]), float(grid[-1]))
