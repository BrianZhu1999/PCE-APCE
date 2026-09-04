from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .config import AlphaConfig


def liu_quantile(alpha: torch.Tensor | float) -> torch.Tensor:
    values = torch.as_tensor(alpha)
    if not values.is_floating_point():
        values = values.to(torch.get_default_dtype())
    if bool(((values <= 0) | (values >= 1)).any()):
        raise ValueError("Liu alpha coordinates must lie strictly inside (0, 1)")
    return math.sqrt(3.0) / math.pi * torch.log(values / (1.0 - values))


def initial_alpha_grid(config: AlphaConfig, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    nodes, _ = np.polynomial.legendre.leggauss(config.initial_nodes)
    mapped = config.alpha_min + 0.5 * (nodes + 1.0) * (config.alpha_max - config.alpha_min)
    return torch.as_tensor(mapped, dtype=dtype)


def _pchip_slopes(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Fritsch-Carlson slopes for a shape-preserving cubic interpolant."""

    if x.numel() < 2:
        raise ValueError("At least two alpha nodes are required for interpolation")
    h = x[1:] - x[:-1]
    delta = (y[1:] - y[:-1]) / h
    if x.numel() == 2:
        return torch.stack((delta[0], delta[0]))

    slopes = torch.zeros_like(y)
    same_sign = delta[:-1] * delta[1:] > 0
    w1 = 2.0 * h[1:] + h[:-1]
    w2 = h[1:] + 2.0 * h[:-1]
    harmonic = (w1 + w2) / (w1 / delta[:-1] + w2 / delta[1:])
    slopes[1:-1] = torch.where(same_sign, harmonic, torch.zeros_like(harmonic))

    left = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
    left = torch.where(left * delta[0] <= 0, torch.zeros_like(left), left)
    left = torch.where(
        (delta[0] * delta[1] < 0) & (left.abs() > 3.0 * delta[0].abs()),
        3.0 * delta[0],
        left,
    )
    right = ((2.0 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (
        h[-1] + h[-2]
    )
    right = torch.where(right * delta[-1] <= 0, torch.zeros_like(right), right)
    right = torch.where(
        (delta[-1] * delta[-2] < 0) & (right.abs() > 3.0 * delta[-1].abs()),
        3.0 * delta[-1],
        right,
    )
    slopes[0] = left
    slopes[-1] = right
    return slopes


def _pchip_values(
    x: torch.Tensor,
    y: torch.Tensor,
    query: torch.Tensor,
) -> torch.Tensor:
    slopes = _pchip_slopes(x, y)
    right = torch.searchsorted(x, query).clamp(1, x.numel() - 1)
    left = right - 1
    width = x[right] - x[left]
    position = (query - x[left]) / width
    position_squared = position.square()
    position_cubed = position_squared * position
    return (
        (2.0 * position_cubed - 3.0 * position_squared + 1.0) * y[left]
        + (position_cubed - 2.0 * position_squared + position) * width * slopes[left]
        + (-2.0 * position_cubed + 3.0 * position_squared) * y[right]
        + (position_cubed - position_squared) * width * slopes[right]
    )


@dataclass(frozen=True)
class AlphaAdaptation:
    ensembles: torch.Tensor
    source_indices: tuple[int, ...]
    added_alpha: torch.Tensor
    removed_alpha: torch.Tensor


@dataclass
class AlphaEvidenceTracker:
    alpha: torch.Tensor
    log_scores: torch.Tensor
    low_evidence_counts: torch.Tensor
    config: AlphaConfig

    @classmethod
    def create(
        cls,
        config: AlphaConfig,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> "AlphaEvidenceTracker":
        alpha = initial_alpha_grid(config, dtype=dtype).to(device)
        return cls(
            alpha=alpha,
            log_scores=torch.zeros_like(alpha),
            low_evidence_counts=torch.zeros_like(alpha, dtype=torch.int64),
            config=config,
        )

    @property
    def weights(self) -> torch.Tensor:
        shifted = self.log_scores - self.log_scores.max()
        return torch.softmax(shifted, dim=0)

    def update(self, normalized_log_evidence: torch.Tensor) -> torch.Tensor:
        if normalized_log_evidence.shape != self.log_scores.shape:
            raise ValueError("Evidence must have one value per active alpha path")
        self.log_scores.mul_(self.config.evidence_decay).add_(
            self.config.evidence_gain * normalized_log_evidence
        )
        weights = self.weights
        below = weights < self.config.prune_threshold
        self.low_evidence_counts = torch.where(
            below,
            self.low_evidence_counts + 1,
            torch.zeros_like(self.low_evidence_counts),
        )
        return weights

    def continuous_estimate(self, samples: int = 4097) -> float:
        """Return a grid-independent maximizer of the PCHIP evidence profile."""

        dense = torch.linspace(
            float(self.alpha[0]),
            float(self.alpha[-1]),
            samples,
            dtype=self.alpha.dtype,
            device=self.alpha.device,
        )
        interpolated = _pchip_values(self.alpha, self.log_scores, dense)
        return float(dense[torch.argmax(interpolated)])

    def _candidate_intervals(self) -> list[int]:
        if self.alpha.numel() < 2:
            return []
        midpoint = 0.5 * (self.alpha[:-1] + self.alpha[1:])
        cubic_midpoint = _pchip_values(self.alpha, self.log_scores, midpoint)
        linear_midpoint = 0.5 * (self.log_scores[:-1] + self.log_scores[1:])
        errors = (cubic_midpoint - linear_midpoint).abs()

        candidates: list[int] = []
        score_range = self.log_scores.max() - self.log_scores.min()
        numerical_zero = torch.finfo(self.log_scores.dtype).eps * 32.0
        if bool(score_range > numerical_zero):
            estimate = self.alpha.new_tensor(self.continuous_estimate())
            for index in range(self.alpha.numel() - 1):
                if bool((estimate >= self.alpha[index]) & (estimate <= self.alpha[index + 1])):
                    candidates.append(index)

        curved = torch.nonzero(
            errors > self.config.interpolation_tolerance,
            as_tuple=False,
        ).flatten()
        curved_order = sorted(curved.tolist(), key=lambda index: float(-errors[index]))
        candidates.extend(index for index in curved_order if index not in candidates)
        return candidates

    def proposed_midpoints(self) -> torch.Tensor:
        if self.alpha.numel() >= self.config.max_nodes:
            return self.alpha.new_empty(0)
        remaining = self.config.max_nodes - self.alpha.numel()
        midpoints = []
        for index in self._candidate_intervals():
            spacing = self.alpha[index + 1] - self.alpha[index]
            if float(spacing) >= 2.0 * self.config.min_spacing:
                midpoints.append(0.5 * (self.alpha[index] + self.alpha[index + 1]))
            if len(midpoints) >= remaining:
                break
        return torch.stack(midpoints) if midpoints else self.alpha.new_empty(0)

    def adapt_ensembles(self, path_ensembles: torch.Tensor) -> AlphaAdaptation:
        """Atomically refine/prune alpha state and keep branch ensembles aligned."""

        if path_ensembles.ndim < 1 or path_ensembles.shape[0] != self.alpha.numel():
            raise ValueError("One state ensemble is required per active alpha path")
        old_alpha = self.alpha
        old_scores = self.log_scores
        old_counts = self.low_evidence_counts
        old_weights = self.weights
        node_count = old_alpha.numel()

        top_count = min(3, node_count)
        protected = set(torch.topk(old_weights, top_count).indices.tolist())
        protected.update((0, node_count - 1))
        minimum_active = min(5, self.config.max_nodes)
        removal_budget = max(0, node_count - minimum_active)
        removable = [
            index
            for index in range(node_count)
            if index not in protected
            and int(old_counts[index]) >= self.config.prune_patience
            and float(old_weights[index]) < self.config.prune_threshold
        ]
        removable.sort(key=lambda index: float(old_weights[index]))
        removed_indices = set(removable[:removal_budget])
        kept_indices = [index for index in range(node_count) if index not in removed_indices]

        records: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]] = [
            (
                old_alpha[index],
                old_scores[index],
                old_counts[index],
                path_ensembles[index],
                index,
            )
            for index in kept_indices
        ]
        capacity = max(0, self.config.max_nodes - len(records))
        added = []
        existing_alpha = [record[0] for record in records]
        for interval in self._candidate_intervals():
            if len(added) >= capacity:
                break
            left_alpha = old_alpha[interval]
            right_alpha = old_alpha[interval + 1]
            if float(right_alpha - left_alpha) < 2.0 * self.config.min_spacing:
                continue
            new_alpha = 0.5 * (left_alpha + right_alpha)
            if any(
                abs(float(new_alpha - value)) < self.config.min_spacing
                for value in existing_alpha
            ):
                continue
            fraction = (new_alpha - left_alpha) / (right_alpha - left_alpha)
            new_ensemble = torch.lerp(
                path_ensembles[interval],
                path_ensembles[interval + 1],
                fraction.to(path_ensembles),
            )
            new_score = torch.lerp(old_scores[interval], old_scores[interval + 1], fraction)
            source_index = interval if float(fraction) <= 0.5 else interval + 1
            records.append(
                (
                    new_alpha,
                    new_score,
                    torch.zeros((), dtype=old_counts.dtype, device=old_counts.device),
                    new_ensemble,
                    source_index,
                )
            )
            existing_alpha.append(new_alpha)
            added.append(new_alpha)

        records.sort(key=lambda record: float(record[0]))
        self.alpha = torch.stack([record[0] for record in records])
        self.log_scores = torch.stack([record[1] for record in records])
        self.low_evidence_counts = torch.stack([record[2] for record in records])
        adapted_ensembles = torch.stack([record[3] for record in records])
        source_indices = tuple(record[4] for record in records)
        return AlphaAdaptation(
            ensembles=adapted_ensembles,
            source_indices=source_indices,
            added_alpha=torch.stack(added) if added else old_alpha.new_empty(0),
            removed_alpha=(
                torch.stack([old_alpha[index] for index in sorted(removed_indices)])
                if removed_indices
                else old_alpha.new_empty(0)
            ),
        )
