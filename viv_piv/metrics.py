from __future__ import annotations

import math

import numpy as np
import torch

from .io import VIVCase
from .rom import PODModel


def weighted_crps(ensemble: torch.Tensor, truth: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights / weights.sum().clamp_min(1e-30)
    first = torch.sum(weights[:, None] * torch.abs(ensemble - truth[None, :]), dim=0)
    sorted_values, order = torch.sort(ensemble, dim=0)
    expanded = weights[:, None].expand_as(ensemble)
    sorted_weights = torch.gather(expanded, 0, order)
    cumulative_before = torch.cumsum(sorted_weights, dim=0) - sorted_weights
    second = torch.sum(sorted_weights * (2.0 * cumulative_before + sorted_weights - 1.0) * sorted_values, dim=0)
    return torch.mean(first - second)


def weighted_interval(ensemble: torch.Tensor, weights: torch.Tensor, lower: float = 0.05, upper: float = 0.95) -> tuple[torch.Tensor, torch.Tensor]:
    sorted_values, order = torch.sort(ensemble, dim=0)
    expanded = (weights / weights.sum().clamp_min(1e-30))[:, None].expand_as(ensemble)
    sorted_weights = torch.gather(expanded, 0, order)
    cumulative = torch.cumsum(sorted_weights, dim=0)
    lower_idx = torch.argmax((cumulative >= lower).to(torch.int64), dim=0)
    upper_idx = torch.argmax((cumulative >= upper).to(torch.int64), dim=0)
    columns = torch.arange(ensemble.shape[1], device=ensemble.device)
    return sorted_values[lower_idx, columns], sorted_values[upper_idx, columns]


class ReducedMetricAccumulator:
    def __init__(self, truth_scale: float):
        self.truth_scale = max(float(truth_scale), 1e-12)
        self.error_square = 0.0
        self.truth_square = 0.0
        self.count = 0
        self.crps: list[float] = []
        self.coverage: list[float] = []
        self.width: list[float] = []

    def add_point(self, estimate: torch.Tensor, truth: torch.Tensor) -> None:
        self.error_square += float(torch.sum((estimate - truth) ** 2))
        self.truth_square += float(torch.sum(truth**2))
        self.count += int(truth.numel())

    def add_distribution(self, ensemble: torch.Tensor, truth: torch.Tensor, weights: torch.Tensor) -> None:
        self.crps.append(float(weighted_crps(ensemble, truth, weights)) / self.truth_scale)
        low, high = weighted_interval(ensemble, weights)
        self.coverage.append(float(torch.mean(((truth >= low) & (truth <= high)).to(torch.float64))))
        self.width.append(float(torch.mean(high - low)) / self.truth_scale)

    def finalize(self) -> dict[str, float]:
        return {
            "evaluation_nrmse": math.sqrt(self.error_square / max(self.truth_square, 1e-30)),
            "evaluation_rmse": math.sqrt(self.error_square / max(self.count, 1)),
            "normalized_crps": float(np.mean(self.crps)) if self.crps else math.nan,
            "coverage_90": float(np.mean(self.coverage)) if self.coverage else math.nan,
            "normalized_interval_width_90": float(np.mean(self.width)) if self.width else math.nan,
            "probabilistic_frames": len(self.crps),
        }


def full_field_metrics(
    case: VIVCase,
    pod: PODModel,
    latent_estimate: np.ndarray,
    device: torch.device,
    *,
    block: int = 16,
    excluded_flat_indices: np.ndarray | None = None,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    basis = torch.as_tensor(pod.basis, dtype=torch.float32, device=device)
    mean = torch.as_tensor(pod.mean, dtype=torch.float32, device=device)
    physical_error = 0.0
    physical_truth = 0.0
    fluctuation_error = 0.0
    fluctuation_truth = 0.0
    count = 0
    unobserved_physical_error = 0.0
    unobserved_physical_truth = 0.0
    unobserved_fluctuation_error = 0.0
    unobserved_fluctuation_truth = 0.0
    unobserved_count = 0
    excluded = np.asarray(excluded_flat_indices if excluded_flat_indices is not None else [], dtype=np.int64)
    excluded_t = torch.as_tensor(excluded, dtype=torch.int64, device=device)
    truth_energy = np.empty(case.time_s.size, dtype=np.float64)
    predicted_energy = np.empty(case.time_s.size, dtype=np.float64)
    for start, values, valid in case.iter_physical(block=block):
        stop = start + values.shape[0]
        z = torch.as_tensor(latent_estimate[start:stop], dtype=torch.float32, device=device)
        prediction = mean[None, :] + z @ basis.mT
        truth = torch.as_tensor(values, dtype=torch.float32, device=device)
        valid_t = torch.as_tensor(valid, dtype=torch.bool, device=device)
        residual = prediction - truth
        physical_error += float(torch.sum(residual[valid_t] ** 2))
        physical_truth += float(torch.sum(truth[valid_t] ** 2))
        truth_fluct = truth - mean[None, :]
        prediction_fluct = prediction - mean[None, :]
        fluctuation_error += float(torch.sum((prediction_fluct[valid_t] - truth_fluct[valid_t]) ** 2))
        fluctuation_truth += float(torch.sum(truth_fluct[valid_t] ** 2))
        count += int(valid_t.sum())
        unobserved_valid = valid_t.clone()
        if excluded_t.numel():
            unobserved_valid[:, excluded_t] = False
        unobserved_physical_error += float(torch.sum(residual[unobserved_valid] ** 2))
        unobserved_physical_truth += float(torch.sum(truth[unobserved_valid] ** 2))
        unobserved_fluctuation_error += float(torch.sum((prediction_fluct[unobserved_valid] - truth_fluct[unobserved_valid]) ** 2))
        unobserved_fluctuation_truth += float(torch.sum(truth_fluct[unobserved_valid] ** 2))
        unobserved_count += int(unobserved_valid.sum())
        field_truth = truth.reshape(truth.shape[0], -1, 2)
        field_prediction = prediction.reshape(prediction.shape[0], -1, 2)
        pixel_valid = valid_t.reshape(valid_t.shape[0], -1, 2)[..., 0]
        for local in range(values.shape[0]):
            good = pixel_valid[local]
            truth_energy[start + local] = float(0.5 * torch.mean(torch.sum(field_truth[local, good] ** 2, dim=1)))
            predicted_energy[start + local] = float(0.5 * torch.mean(torch.sum(field_prediction[local, good] ** 2, dim=1)))
    energy_error = predicted_energy - truth_energy
    energy_corr = float(np.corrcoef(predicted_energy, truth_energy)[0, 1]) if np.std(truth_energy) > 0 else math.nan
    metrics = {
        "full_field_physical_nrmse": math.sqrt(physical_error / max(physical_truth, 1e-30)),
        "full_field_physical_rmse": math.sqrt(physical_error / max(count, 1)),
        "full_field_fluctuation_nrmse": math.sqrt(fluctuation_error / max(fluctuation_truth, 1e-30)),
        "unobserved_full_field_physical_nrmse": math.sqrt(unobserved_physical_error / max(unobserved_physical_truth, 1e-30)),
        "unobserved_full_field_physical_rmse": math.sqrt(unobserved_physical_error / max(unobserved_count, 1)),
        "unobserved_full_field_fluctuation_nrmse": math.sqrt(unobserved_fluctuation_error / max(unobserved_fluctuation_truth, 1e-30)),
        "observed_scalar_dimensions_excluded": int(excluded.size),
        "kinetic_energy_nrmse": math.sqrt(float(np.sum(energy_error**2)) / max(float(np.sum(truth_energy**2)), 1e-30)),
        "kinetic_energy_correlation": energy_corr,
        "kinetic_energy_peak_relative_error": abs(float(predicted_energy.max() - truth_energy.max())) / max(float(truth_energy.max()), 1e-30),
    }
    traces = {"truth_energy": truth_energy, "predicted_energy": predicted_energy}
    return metrics, traces
