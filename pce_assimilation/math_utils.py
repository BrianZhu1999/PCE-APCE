from __future__ import annotations

import math

import torch


def symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.mT)


def stable_cholesky(
    matrix: torch.Tensor,
    relative_floor: float = 1e-6,
) -> torch.Tensor:
    matrix = symmetrize(matrix)
    scale = torch.diagonal(matrix).abs().median().clamp_min(torch.finfo(matrix.dtype).eps)
    jitter = relative_floor * scale
    identity = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    for multiplier in (1.0, 10.0, 100.0, 1000.0):
        factor, info = torch.linalg.cholesky_ex(matrix + multiplier * jitter * identity)
        if int(info.max()) == 0:
            return factor
    raise RuntimeError("Covariance is not positive definite after jitter escalation")


def spd_sqrt(matrix: torch.Tensor, relative_floor: float = 1e-6) -> torch.Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetrize(matrix))
    positive = eigenvalues[eigenvalues > 0]
    reference = positive.median() if positive.numel() else torch.ones((), dtype=matrix.dtype, device=matrix.device)
    floor = relative_floor * reference
    eigenvalues = eigenvalues.clamp_min(floor)
    return (eigenvectors * eigenvalues.sqrt().unsqueeze(0)) @ eigenvectors.mT


def gaussian_logpdf(
    values: torch.Tensor,
    means: torch.Tensor,
    covariance: torch.Tensor,
    relative_floor: float = 1e-6,
) -> torch.Tensor:
    """Broadcasted multivariate Gaussian log density.

    The final axis is the event dimension. Leading axes are broadcast.
    """

    factor = stable_cholesky(covariance, relative_floor)
    differences = values - means
    solved = torch.linalg.solve_triangular(
        factor,
        differences.unsqueeze(-1),
        upper=False,
    ).squeeze(-1)
    quadratic = solved.square().sum(dim=-1)
    log_determinant = 2.0 * torch.log(torch.diagonal(factor)).sum()
    dimension = covariance.shape[-1]
    return -0.5 * (dimension * math.log(2.0 * math.pi) + log_determinant + quadratic)


def normalized_log_weights(log_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    log_normalizer = torch.logsumexp(log_weights, dim=-1, keepdim=True)
    normalized_log = log_weights - log_normalizer
    return normalized_log.exp(), normalized_log


def effective_sample_size(weights: torch.Tensor) -> torch.Tensor:
    return weights.square().sum(dim=-1).reciprocal()
