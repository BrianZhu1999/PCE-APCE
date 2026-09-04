from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AlphaConfig:
    alpha_min: float = 0.01
    alpha_max: float = 0.99
    initial_nodes: int = 9
    max_nodes: int = 25
    min_spacing: float = 2e-3
    evidence_decay: float = 0.95
    evidence_gain: float = 1.0
    interpolation_tolerance: float = 0.10
    prune_threshold: float = 1e-4
    prune_patience: int = 3


@dataclass(frozen=True)
class FlowConfig:
    steps: int = 8
    max_back_projection_iterations: int = 4
    innovation_tolerance: float = 1e-3
    increment_tolerance: float = 1e-3
    divergence_ratio: float = 1.25
    eigenvalue_floor: float = 1e-6
    max_patch_observations: int = 128
    sinkhorn_iterations: int = 100
    sinkhorn_epsilon_scale: float = 0.05
    moment_matching: bool = False
    moment_matching_strength: float = 1.0


@dataclass(frozen=True)
class LowRankConfig:
    explained_variance: float = 0.995
    max_rank: int = 64
    ridge_relative: float = 1e-3
    ridge_floor: float = 1e-6


@dataclass(frozen=True)
class AssimilationConfig:
    alpha: AlphaConfig = field(default_factory=AlphaConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    low_rank: LowRankConfig = field(default_factory=LowRankConfig)
    evidence_dtype: str = "float64"
