from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class APCECalibration:
    """Adaptive APCE calibration parameters for one observation update."""

    confidence: float
    evidence_gap: float
    temperature: float
    forgetting: float
    entropy_floor: float


def _as_numpy_1d(values: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    output = np.asarray(values, dtype=float).reshape(-1)
    if output.size == 0:
        raise ValueError("alpha refinement requires a non-empty one-dimensional array")
    return output


def evidence_gap_confidence(scores: np.ndarray | torch.Tensor) -> tuple[float, float]:
    """Return a scale-free top-1/top-2 evidence confidence in [0, 1].

    The raw scores may be instantaneous evidence or cumulative logits.  The
    returned confidence is intentionally conservative: a small top-1/top-2 gap
    leaves APCE in calibration mode, whereas a clear gap lets APCE behave like
    the sharper PCE update.
    """

    score = _as_numpy_1d(scores)
    if score.size < 2:
        return 1.0, 0.0
    ordered = np.sort(score)
    gap = float(max(ordered[-1] - ordered[-2], 0.0))
    spread = float(np.percentile(score, 75) - np.percentile(score, 25))
    scale = max(abs(spread), float(np.std(score)), 1.0e-8)
    confidence = 1.0 - math.exp(-gap / scale)
    return float(np.clip(confidence, 0.0, 1.0)), gap


def refined_alpha_map(alpha_grid: np.ndarray | torch.Tensor, scores: np.ndarray | torch.Tensor) -> float:
    """Local MAP/quadratic alpha estimate.

    This is deliberately not a posterior weighted mean.  It uses the best grid
    point and its local neighbours to form a concave quadratic MAP estimate,
    falling back to the grid MAP when the local fit is unreliable.
    """

    alpha = _as_numpy_1d(alpha_grid)
    score = _as_numpy_1d(scores)
    if alpha.shape != score.shape:
        raise ValueError("alpha_grid and scores must have the same shape")
    if alpha.size == 1:
        return float(alpha[0])
    if float(np.max(score) - np.min(score)) < 1.0e-12:
        return float(np.median(alpha))

    best = int(np.argmax(score))
    if alpha.size < 3:
        return float(alpha[best])
    if best == 0:
        indices = np.array([0, 1, 2])
    elif best == alpha.size - 1:
        indices = np.array([alpha.size - 3, alpha.size - 2, alpha.size - 1])
    else:
        indices = np.array([best - 1, best, best + 1])

    x = alpha[indices]
    y = score[indices]
    try:
        a, b, _ = np.polyfit(x, y, deg=2)
    except np.linalg.LinAlgError:
        return float(alpha[best])
    if not np.isfinite(a) or not np.isfinite(b) or a >= -1.0e-12:
        return float(alpha[best])
    vertex = -b / (2.0 * a)
    if x[0] <= vertex <= x[-1]:
        return float(np.clip(vertex, alpha[0], alpha[-1]))
    return float(alpha[best])


def local_alpha_grid(
    alpha_grid: np.ndarray | torch.Tensor,
    scores: np.ndarray | torch.Tensor,
    *,
    points: int | None = None,
    bounds: tuple[float, float] | None = None,
    topk: int = 3,
    min_spacing: float = 1.0e-3,
) -> np.ndarray:
    """Construct the next active local grid around the MAP/high-evidence region."""

    alpha = _as_numpy_1d(alpha_grid)
    score = _as_numpy_1d(scores)
    if alpha.shape != score.shape:
        raise ValueError("alpha_grid and scores must have the same shape")
    if points is None:
        points = int(alpha.size)
    points = max(3, int(points))
    lower_bound, upper_bound = bounds if bounds is not None else (float(alpha[0]), float(alpha[-1]))
    if not lower_bound < upper_bound:
        raise ValueError("alpha bounds must be increasing")

    center = refined_alpha_map(alpha, score)
    confidence, _ = evidence_gap_confidence(score)
    diffs = np.diff(np.unique(alpha))
    median_step = float(np.median(diffs)) if diffs.size else (upper_bound - lower_bound) / max(points - 1, 1)
    global_width = upper_bound - lower_bound
    top_count = min(max(1, topk), alpha.size)
    top_values = np.sort(alpha[np.argsort(score)[-top_count:]])
    top_span = float(top_values[-1] - top_values[0]) if top_values.size > 1 else 0.0

    radius = max(
        0.5 * top_span + 0.45 * median_step,
        median_step * (0.80 + 0.70 * (1.0 - confidence)),
        0.025 * global_width,
        0.5 * (points - 1) * min_spacing,
    )
    lower = max(lower_bound, center - radius)
    upper = min(upper_bound, center + radius)
    required_width = max((points - 1) * min_spacing, 0.05 * global_width)
    if upper - lower < required_width:
        deficit = required_width - (upper - lower)
        lower = max(lower_bound, lower - 0.5 * deficit)
        upper = min(upper_bound, upper + 0.5 * deficit)
        if upper - lower < required_width:
            if lower <= lower_bound + 1.0e-12:
                upper = min(upper_bound, lower + required_width)
            else:
                lower = max(lower_bound, upper - required_width)

    output = np.linspace(lower, upper, points, dtype=float)
    return np.clip(output, lower_bound, upper_bound)


def apce_calibration_parameters(
    scores: np.ndarray | torch.Tensor,
    *,
    pce_temperature: float,
    apce_temperature: float,
    apce_min_temperature: float,
    apce_forgetting: float,
    apce_entropy_floor: float,
    progress: float,
) -> APCECalibration:
    """Turn evidence separation into APCE's calibration-only parameters."""

    confidence, gap = evidence_gap_confidence(scores)
    pce_temperature = float(pce_temperature)
    apce_temperature = float(apce_temperature)
    apce_min_temperature = float(apce_min_temperature)
    uncertain_temperature = max(apce_min_temperature, min(apce_temperature, pce_temperature))
    temperature = (1.0 - confidence) * uncertain_temperature + confidence * pce_temperature
    uncertain_forgetting = min(float(apce_forgetting), 0.985)
    forgetting = (1.0 - confidence) * uncertain_forgetting + confidence * 1.0
    progress = float(np.clip(progress, 0.0, 1.0))
    floor = float(apce_entropy_floor) * (1.0 - confidence) ** 1.35
    floor *= 0.80 + 0.20 * (1.0 - progress)
    return APCECalibration(
        confidence=float(confidence),
        evidence_gap=float(gap),
        temperature=float(max(temperature, apce_min_temperature)),
        forgetting=float(np.clip(forgetting, 0.0, 1.0)),
        entropy_floor=float(max(floor, 0.0)),
    )


def torch_refined_alpha_map(alpha_grid: torch.Tensor, scores: torch.Tensor) -> float:
    return refined_alpha_map(alpha_grid, scores)


def torch_local_alpha_grid(
    alpha_grid: torch.Tensor,
    scores: torch.Tensor,
    *,
    points: int | None = None,
    bounds: tuple[float, float] | None = None,
    topk: int = 3,
    min_spacing: float = 1.0e-3,
) -> torch.Tensor:
    grid = local_alpha_grid(
        alpha_grid,
        scores,
        points=points,
        bounds=bounds,
        topk=topk,
        min_spacing=min_spacing,
    )
    return torch.as_tensor(grid, dtype=alpha_grid.dtype, device=alpha_grid.device)


def torch_regrid_paths(
    old_alpha: torch.Tensor,
    values: torch.Tensor,
    new_alpha: torch.Tensor,
) -> torch.Tensor:
    """Linearly interpolate path-indexed tensors onto a new alpha grid."""

    if values.shape[0] != old_alpha.numel():
        raise ValueError("values must be indexed by alpha on dimension 0")
    if old_alpha.numel() == 1:
        return values.expand((new_alpha.numel(),) + values.shape[1:]).clone()
    right = torch.searchsorted(old_alpha, new_alpha).clamp(1, old_alpha.numel() - 1)
    left = right - 1
    denom = (old_alpha[right] - old_alpha[left]).clamp_min(torch.finfo(old_alpha.dtype).eps)
    fraction = ((new_alpha - old_alpha[left]) / denom).to(values)
    view_shape = (new_alpha.numel(),) + (1,) * (values.ndim - 1)
    fraction = fraction.reshape(view_shape)
    return values[left] * (1.0 - fraction) + values[right] * fraction


def numpy_regrid_paths(
    old_alpha: np.ndarray,
    values: np.ndarray,
    new_alpha: np.ndarray,
) -> np.ndarray:
    old = np.asarray(old_alpha, dtype=float)
    new = np.asarray(new_alpha, dtype=float)
    value = np.asarray(values)
    if value.shape[0] != old.size:
        raise ValueError("values must be indexed by alpha on dimension 0")
    if old.size == 1:
        return np.repeat(value, new.size, axis=0)
    right = np.searchsorted(old, new, side="left")
    right = np.clip(right, 1, old.size - 1)
    left = right - 1
    denom = np.maximum(old[right] - old[left], np.finfo(float).eps)
    fraction = (new - old[left]) / denom
    shape = (new.size,) + (1,) * (value.ndim - 1)
    fraction = fraction.reshape(shape)
    return value[left] * (1.0 - fraction) + value[right] * fraction
