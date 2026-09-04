"""Evidence profiles shared by the published wave benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import wave_protocol


@dataclass(frozen=True)
class EvidenceConfig:
    paired_initial: bool = True
    paired_sampler: bool = True
    shadow_bank: bool = True
    gaussian_evidence: bool = True
    shrinkage: float = 0.35
    temperature: float = 0.50
    forgetting: float = 1.0
    sensitivity_floor: float = 1.0
    adaptive_temperature: bool = False
    min_temperature: float = 0.15
    analysis_blend_start: float = 0.55
    analysis_blend_max: float = 0.0
    entropy_floor_start: float = 0.0
    entropy_floor_mid: float = 0.0
    entropy_floor_end: float = 0.0
    entropy_turn: float = 0.70
    weight_floor: float = 1.0e-8
    continuous_estimator: str = "quadratic"


PCE_CONFIG = EvidenceConfig()
APCE_CONFIG = EvidenceConfig(
    temperature=0.45,
    forgetting=0.975,
    sensitivity_floor=0.35,
    adaptive_temperature=True,
    min_temperature=0.08,
    analysis_blend_start=0.55,
    analysis_blend_max=0.12,
    entropy_floor_start=0.38,
    entropy_floor_mid=0.30,
    entropy_floor_end=0.22,
    entropy_turn=0.70,
    continuous_estimator="hybrid",
)
EVIDENCE_CONFIGS = {"pce": PCE_CONFIG, "apce": APCE_CONFIG}


def entropy(weights: np.ndarray) -> float:
    values = np.maximum(weights, 1.0e-300)
    return float(-np.sum(values * np.log(values)))


def entropy_project(weights: np.ndarray, target: float) -> np.ndarray:
    """Mix minimally with a uniform distribution to reach target entropy."""
    if target <= 0.0 or entropy(weights) >= target:
        return weights
    uniform = np.ones_like(weights) / weights.size
    low, high = 0.0, 1.0
    for _ in range(50):
        middle = 0.5 * (low + high)
        mixed = (1.0 - middle) * weights + middle * uniform
        if entropy(mixed) < target:
            low = middle
        else:
            high = middle
    mixed = (1.0 - high) * weights + high * uniform
    return mixed / mixed.sum()


def entropy_target(progress: float, config: EvidenceConfig) -> float:
    if config.entropy_floor_start <= 0.0:
        return 0.0
    turn = float(np.clip(config.entropy_turn, 1.0e-6, 1.0 - 1.0e-6))
    if progress <= turn:
        ratio = progress / turn
        return config.entropy_floor_start + ratio * (
            config.entropy_floor_mid - config.entropy_floor_start
        )
    ratio = (progress - turn) / (1.0 - turn)
    return config.entropy_floor_mid + ratio * (
        config.entropy_floor_end - config.entropy_floor_mid
    )


def alpha_continuous_estimate(
    alpha_grid: np.ndarray,
    theta_grid: np.ndarray,
    weights: np.ndarray,
    scenario_config: Any,
    estimator: str,
) -> float:
    log_weights = np.log(np.maximum(weights, 1.0e-300))
    quadratic = wave_protocol.continuous_alpha_estimate(alpha_grid, log_weights)
    theta_mean = float(np.sum(weights * theta_grid))
    posterior_mean = wave_protocol.theta_to_alpha(theta_mean, scenario_config)
    if estimator == "quadratic":
        return quadratic
    if estimator == "theta_mean":
        return posterior_mean
    concentration = float(np.max(weights))
    blend = float(np.clip((concentration - 0.25) / 0.35, 0.0, 1.0))
    return float(blend * quadratic + (1.0 - blend) * posterior_mean)


def evidence_vector(
    branch_observations: list[np.ndarray],
    observation: np.ndarray,
    observation_noise: float,
    config: EvidenceConfig,
) -> np.ndarray:
    dimension_weights = wave_protocol.alpha_sensitivity_weights(
        branch_observations,
        config.sensitivity_floor,
    )
    return np.asarray(
        [
            wave_protocol.gaussian_log_evidence(
                item,
                observation,
                observation_noise,
                config.shrinkage,
                dimension_weights,
            )
            for item in branch_observations
        ],
        dtype=float,
    )
