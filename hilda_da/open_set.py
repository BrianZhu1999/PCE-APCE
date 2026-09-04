from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .math_utils import stable_cholesky, symmetrize


def predictive_discrepancy(
    forecast_observations: torch.Tensor,
    observation: torch.Tensor,
    observation_covariance: torch.Tensor,
) -> torch.Tensor:
    ensemble_size = forecast_observations.shape[0]
    anomalies = forecast_observations - forecast_observations.mean(dim=0)
    predictive_covariance = symmetrize(
        anomalies.mT @ anomalies / max(ensemble_size - 1, 1) + observation_covariance
    )
    factor = stable_cholesky(predictive_covariance)
    residual = observation - forecast_observations.mean(dim=0)
    whitened = torch.linalg.solve_triangular(
        factor,
        residual.unsqueeze(-1),
        upper=False,
    ).squeeze(-1)
    log_determinant = 2.0 * torch.log(torch.diagonal(factor)).sum()
    return whitened.square().sum() + log_determinant


@dataclass
class ConformalMismatchDetector:
    threshold: float
    required_exceedances: int = 3
    consecutive_exceedances: int = 0

    @classmethod
    def calibrate(
        cls,
        calibration_scores: torch.Tensor,
        false_alarm_level: float = 0.05,
        required_exceedances: int = 3,
    ) -> "ConformalMismatchDetector":
        if calibration_scores.ndim != 1 or calibration_scores.numel() < 2:
            raise ValueError("At least two one-dimensional calibration scores are required")
        if not 0.0 < false_alarm_level < 1.0:
            raise ValueError("false_alarm_level must lie strictly between zero and one")
        if required_exceedances < 1:
            raise ValueError("required_exceedances must be positive")
        if not bool(torch.isfinite(calibration_scores).all()):
            raise ValueError("Calibration scores must be finite")
        ordered = torch.sort(calibration_scores).values
        rank = math.ceil((ordered.numel() + 1) * (1.0 - false_alarm_level))
        threshold = math.inf if rank > ordered.numel() else float(ordered[rank - 1])
        return cls(threshold, required_exceedances)

    def update(self, score: float | torch.Tensor) -> bool:
        exceeds = float(score) > self.threshold
        self.consecutive_exceedances = self.consecutive_exceedances + 1 if exceeds else 0
        return self.consecutive_exceedances >= self.required_exceedances
