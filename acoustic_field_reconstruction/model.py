from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def cfl_number(speed: float, dt: float, spacing: float) -> float:
    return float(np.sqrt(3.0) * speed * dt / spacing)


def laplacian(field: torch.Tensor) -> torch.Tensor:
    padded = F.pad(field, (1, 1, 1, 1, 1, 1), mode="replicate")
    center = padded[:, :, 1:-1, 1:-1, 1:-1]
    return (
        padded[:, :, 2:, 1:-1, 1:-1] + padded[:, :, :-2, 1:-1, 1:-1]
        + padded[:, :, 1:-1, 2:, 1:-1] + padded[:, :, 1:-1, :-2, 1:-1]
        + padded[:, :, 1:-1, 1:-1, 2:] + padded[:, :, 1:-1, 1:-1, :-2]
        - 6.0 * center
    )


def step(
    previous: torch.Tensor,
    current: torch.Tensor,
    speed: torch.Tensor,
    damping: float,
    dt: float,
    spacing: float,
    process_noise: torch.Tensor | None,
) -> torch.Tensor:
    coefficient = (speed * dt / spacing) ** 2
    output = (
        2.0 * current - previous
        + coefficient[:, None, None, None, None] * laplacian(current)
        - 2.0 * damping * dt * (current - previous)
    )
    if process_noise is not None:
        output = output + process_noise
    return output


def apply_boundary(field: torch.Tensor, boundary_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if boundary_values.ndim == 3:
        values = boundary_values[None, None]
    elif boundary_values.ndim == 4:
        values = boundary_values[:, None]
    else:
        raise ValueError(f"unexpected boundary shape {tuple(boundary_values.shape)}")
    return torch.where(mask[None, None], values, field)


def sponge_mask(value: float, device: torch.device) -> torch.Tensor:
    mask = torch.ones((9, 21, 21), dtype=torch.float32, device=device)
    mask[[0, -1], :, :] *= value
    mask[:, [0, -1], :] *= value
    mask[:, :, [0, -1]] *= value
    return mask


def weighted_field_mean(states: torch.Tensor, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean_candidate = states.mean(dim=1).detach().cpu().numpy()
    variance_candidate = states.var(dim=1, unbiased=True).detach().cpu().numpy()
    weights = np.asarray(weights, dtype=float)
    mean = np.sum(weights[:, None, None, None] * mean_candidate, axis=0)
    within = np.sum(weights[:, None, None, None] * variance_candidate, axis=0)
    between = np.sum(weights[:, None, None, None] * (mean_candidate - mean[None]) ** 2, axis=0)
    return mean.astype(np.float32), np.sqrt(np.maximum(within + between, 1e-20)).astype(np.float32)
