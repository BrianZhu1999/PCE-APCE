from __future__ import annotations

import math
import unittest

import numpy as np

import run_benchmark_v3

try:
    import torch
    from paper_experiments import run_spring_heat_gate
except Exception:  # pragma: no cover - environment-dependent import guard
    torch = None
    run_spring_heat_gate = None


class DimensionWeightedEvidenceTests(unittest.TestCase):
    def test_numpy_dimension_weighted_score_uses_marginal_composite_terms(self) -> None:
        ensemble = np.asarray(
            [
                [0.0, 0.0],
                [0.1, 0.2],
                [0.2, 0.4],
                [0.3, 0.6],
            ],
            dtype=float,
        )
        observation = np.asarray([1.0, 0.15], dtype=float)
        weights = np.asarray([9.0, 1.0], dtype=float)
        score = run_benchmark_v3.gaussian_log_evidence(
            ensemble,
            observation,
            obs_noise=0.05,
            shrinkage=1.0,
            dimension_weights=weights,
        )

        mean = ensemble.mean(axis=0)
        covariance = np.cov(ensemble, rowvar=False)
        covariance = np.diag(np.diag(covariance)) + (0.05**2 + 1.0e-7) * np.eye(2)
        residual = observation - mean
        normalized = observation.size * weights / weights.sum()
        variance = np.diag(covariance)
        expected = -0.5 * np.sum(
            normalized
            * (residual * residual / variance + np.log(variance) + math.log(2.0 * math.pi))
        )

        self.assertAlmostEqual(score, float(expected), places=12)

    def test_dimension_weights_change_relative_score_when_residuals_differ_by_dimension(self) -> None:
        ensemble = np.asarray(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.2, 0.0],
                [0.3, 0.0],
            ],
            dtype=float,
        )
        observation = np.asarray([1.0, 0.05], dtype=float)
        score_first_dimension = run_benchmark_v3.gaussian_log_evidence(
            ensemble,
            observation,
            obs_noise=0.05,
            shrinkage=1.0,
            dimension_weights=np.asarray([9.0, 1.0]),
        )
        score_second_dimension = run_benchmark_v3.gaussian_log_evidence(
            ensemble,
            observation,
            obs_noise=0.05,
            shrinkage=1.0,
            dimension_weights=np.asarray([1.0, 9.0]),
        )

        self.assertNotAlmostEqual(score_first_dimension, score_second_dimension, places=6)

    def test_torch_dimension_weighted_score_matches_manual_formula(self) -> None:
        if torch is None or run_spring_heat_gate is None:
            self.skipTest("torch runtime is unavailable in the current local environment")
        ensemble = torch.tensor(
            [
                [0.0, 0.0],
                [0.2, 0.1],
                [0.4, 0.2],
                [0.6, 0.3],
            ],
            dtype=torch.float64,
        )
        observation = torch.tensor([1.0, 0.25], dtype=torch.float64)
        weights = torch.tensor([2.0, 5.0], dtype=torch.float64)
        score = run_spring_heat_gate.evidence_score(
            ensemble,
            observation,
            obs_noise=0.05,
            shrinkage=1.0,
            dimension_weights=weights,
        )

        mean = ensemble.mean(dim=0)
        anomalies = ensemble - mean
        covariance = anomalies.mT @ anomalies / (ensemble.shape[0] - 1)
        covariance = torch.diag(torch.diagonal(covariance)) + (0.05**2 + 1.0e-8) * torch.eye(
            2,
            dtype=torch.float64,
        )
        residual = observation - mean
        normalized = observation.numel() * weights / weights.sum()
        variances = torch.diagonal(covariance)
        expected = -0.5 * torch.sum(
            normalized
            * (residual.square() / variances + variances.log() + math.log(2.0 * math.pi))
        )

        self.assertTrue(torch.allclose(score, expected, atol=1.0e-12, rtol=1.0e-12))


if __name__ == "__main__":
    unittest.main()
