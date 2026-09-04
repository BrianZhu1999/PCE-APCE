from __future__ import annotations

import math
import unittest

import torch

from hilda_da.metrics import (
    binary_auprc,
    binary_auroc,
    binary_false_alarm_rate,
    central_interval_coverage_width,
    ensemble_crps,
    multivariate_energy_score,
    observation_rmse,
    paired_bootstrap_ci,
    paired_effect_size,
    state_rmse,
    weighted_central_interval_coverage_width,
    weighted_ensemble_crps,
    weighted_multivariate_energy_score,
    weighted_path_dispersion_diagonal,
)


class EnsembleMetricTests(unittest.TestCase):
    def test_empirical_crps_matches_hand_calculation(self) -> None:
        ensemble = torch.tensor([[0.0], [2.0]], dtype=torch.float64)
        target = torch.tensor([1.0], dtype=torch.float64)
        self.assertAlmostEqual(float(ensemble_crps(ensemble, target)), 0.5, places=12)

    def test_energy_score_matches_hand_calculation_and_chunking(self) -> None:
        ensemble = torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float64)
        target = torch.tensor([1.0, 0.0], dtype=torch.float64)
        unchunked = multivariate_energy_score(ensemble, target, chunk_size=2)
        chunked = multivariate_energy_score(ensemble, target, chunk_size=1)
        self.assertAlmostEqual(float(unchunked), 0.5, places=12)
        self.assertTrue(torch.equal(unchunked, chunked))

    def test_batched_central_interval_coverage_and_width(self) -> None:
        values = torch.arange(5, dtype=torch.float64)
        ensemble = torch.stack((values, values), dim=-1).unsqueeze(0)
        target = torch.tensor([[2.0, 4.0]], dtype=torch.float64)
        coverage, width = central_interval_coverage_width(
            ensemble,
            target,
            level=0.5,
        )
        self.assertAlmostEqual(float(coverage), 0.5, places=12)
        self.assertAlmostEqual(float(width), 2.0, places=12)

    def test_weighted_scores_and_interval_match_hand_calculation(self) -> None:
        ensemble = torch.tensor([[0.0], [2.0]], dtype=torch.float64)
        target = torch.tensor([1.0], dtype=torch.float64)
        weights = torch.tensor([0.25, 0.75], dtype=torch.float64)
        self.assertAlmostEqual(
            float(weighted_ensemble_crps(ensemble, target, weights)),
            0.625,
            places=12,
        )
        self.assertAlmostEqual(
            float(weighted_multivariate_energy_score(ensemble, target, weights)),
            0.625,
            places=12,
        )
        coverage, width = weighted_central_interval_coverage_width(
            ensemble,
            target,
            weights,
            level=0.5,
        )
        self.assertEqual(float(coverage), 1.0)
        self.assertEqual(float(width), 2.0)

    def test_weighted_metrics_validate_member_weights(self) -> None:
        ensemble = torch.zeros(2, 3, dtype=torch.float64)
        target = torch.zeros(3, dtype=torch.float64)
        with self.assertRaises(ValueError):
            weighted_ensemble_crps(
                ensemble,
                target,
                torch.ones(3, dtype=torch.float64),
            )
        with self.assertRaises(ValueError):
            weighted_multivariate_energy_score(
                ensemble,
                target,
                torch.zeros(2, dtype=torch.float64),
            )
        with self.assertRaises(ValueError):
            weighted_central_interval_coverage_width(
                ensemble,
                target,
                torch.tensor([-1.0, 2.0], dtype=torch.float64),
            )

    def test_weighted_path_dispersion_matches_dense_identity(self) -> None:
        path_ensembles = torch.tensor(
            [
                [[0.0, 1.0], [2.0, 3.0], [4.0, 2.0]],
                [[5.0, -1.0], [7.0, 1.0], [6.0, 3.0]],
            ],
            dtype=torch.float64,
        )
        weights = torch.tensor([0.25, 0.75], dtype=torch.float64)
        dispersion = weighted_path_dispersion_diagonal(path_ensembles, weights)

        path_means = path_ensembles.mean(dim=1)
        combined_mean = (weights[:, None] * path_means).sum(dim=0)
        within_covariance = sum(
            weights[index]
            * (path_ensembles[index] - path_means[index]).mT
            @ (path_ensembles[index] - path_means[index])
            / path_ensembles.shape[1]
            for index in range(path_ensembles.shape[0])
        )
        between_covariance = sum(
            weights[index]
            * torch.outer(
                path_means[index] - combined_mean,
                path_means[index] - combined_mean,
            )
            for index in range(path_ensembles.shape[0])
        )
        total_covariance = sum(
            weights[index]
            * (path_ensembles[index] - combined_mean).mT
            @ (path_ensembles[index] - combined_mean)
            / path_ensembles.shape[1]
            for index in range(path_ensembles.shape[0])
        )

        self.assertTrue(
            torch.allclose(total_covariance, within_covariance + between_covariance)
        )
        self.assertTrue(torch.all(torch.linalg.eigvalsh(within_covariance) >= -1e-12))
        self.assertTrue(torch.all(torch.linalg.eigvalsh(between_covariance) >= -1e-12))
        self.assertTrue(torch.allclose(dispersion.within, within_covariance.diagonal()))
        self.assertTrue(torch.allclose(dispersion.between, between_covariance.diagonal()))
        self.assertTrue(torch.allclose(dispersion.total, total_covariance.diagonal()))
        self.assertTrue(
            torch.allclose(
                dispersion.total_trace,
                dispersion.within_trace + dispersion.between_trace,
            )
        )

    def test_weighted_path_dispersion_validates_weights(self) -> None:
        ensembles = torch.ones(2, 3, 4, dtype=torch.float64)
        with self.assertRaises(ValueError):
            weighted_path_dispersion_diagonal(
                ensembles,
                torch.tensor([1.0, -1.0], dtype=torch.float64),
            )
        with self.assertRaises(ValueError):
            weighted_path_dispersion_diagonal(
                ensembles,
                torch.tensor([0.0, 0.0], dtype=torch.float64),
            )

    def test_rmse_metrics_match_hand_calculation(self) -> None:
        prediction = torch.tensor([1.0, 3.0], dtype=torch.float64)
        target = torch.tensor([1.0, 1.0], dtype=torch.float64)
        expected = math.sqrt(2.0)
        self.assertAlmostEqual(float(state_rmse(prediction, target)), expected, places=12)
        self.assertAlmostEqual(
            float(observation_rmse(prediction, target)),
            expected,
            places=12,
        )

    def test_ensemble_shape_and_parameter_validation(self) -> None:
        ensemble = torch.zeros(2, 3, dtype=torch.float64)
        with self.assertRaises(ValueError):
            ensemble_crps(ensemble, torch.zeros(2, dtype=torch.float64))
        with self.assertRaises(ValueError):
            multivariate_energy_score(
                ensemble,
                torch.zeros(3, dtype=torch.float64),
                chunk_size=0,
            )
        with self.assertRaises(ValueError):
            central_interval_coverage_width(
                ensemble,
                torch.zeros(3, dtype=torch.float64),
                level=1.0,
            )
        with self.assertRaises(ValueError):
            ensemble_crps(
                torch.empty(0, 2, 3, dtype=torch.float64),
                torch.empty(0, 3, dtype=torch.float64),
            )
        invalid = ensemble.clone()
        invalid[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            ensemble_crps(invalid, torch.zeros(3, dtype=torch.float64))


class PairedMetricTests(unittest.TestCase):
    def test_paired_bootstrap_constant_effect_and_seed_reproducibility(self) -> None:
        second = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float64)
        first = second + 2.0
        by_seed = paired_bootstrap_ci(first, second, resamples=101, seed=7, chunk_size=13)
        by_generator = paired_bootstrap_ci(
            first,
            second,
            resamples=101,
            generator=torch.Generator().manual_seed(7),
            chunk_size=13,
        )
        for result in (by_seed, by_generator):
            self.assertEqual(float(result.estimate), 2.0)
            self.assertEqual(float(result.lower), 2.0)
            self.assertEqual(float(result.upper), 2.0)
        self.assertTrue(torch.equal(by_seed.lower, by_generator.lower))

        variable_first = torch.tensor([0.0, 1.0, 4.0, 9.0], dtype=torch.float64)
        variable_second = torch.zeros_like(variable_first)
        repeat_a = paired_bootstrap_ci(variable_first, variable_second, resamples=127, seed=19)
        repeat_b = paired_bootstrap_ci(variable_first, variable_second, resamples=127, seed=19)
        self.assertTrue(torch.equal(repeat_a.lower, repeat_b.lower))
        self.assertTrue(torch.equal(repeat_a.upper, repeat_b.upper))

    def test_paired_effect_size_is_cohens_dz(self) -> None:
        first = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        second = torch.zeros(3, dtype=torch.float64)
        self.assertAlmostEqual(float(paired_effect_size(first, second)), 2.0, places=12)
        self.assertEqual(float(paired_effect_size(second, second)), 0.0)

    def test_bootstrap_requires_explicit_random_source(self) -> None:
        values = torch.ones(3, dtype=torch.float64)
        with self.assertRaises(ValueError):
            paired_bootstrap_ci(values, values)
        with self.assertRaises(ValueError):
            paired_bootstrap_ci(
                values,
                values,
                seed=1,
                generator=torch.Generator().manual_seed(1),
            )
        with self.assertRaises(TypeError):
            paired_bootstrap_ci(values, values, seed=1.5)


class BinaryMetricTests(unittest.TestCase):
    def test_binary_metrics_match_reference_values(self) -> None:
        # Reference values match sklearn roc_auc_score and average_precision_score.
        scores = torch.tensor([0.1, 0.4, 0.35, 0.8], dtype=torch.float64)
        labels = torch.tensor([0, 0, 1, 1])
        self.assertAlmostEqual(float(binary_auroc(scores, labels)), 0.75, places=12)
        self.assertAlmostEqual(float(binary_auprc(scores, labels)), 5.0 / 6.0, places=12)
        self.assertAlmostEqual(
            float(binary_false_alarm_rate(scores, labels, threshold=0.4)),
            0.5,
            places=12,
        )

    def test_auroc_gives_half_credit_to_ties(self) -> None:
        scores = torch.ones(4, dtype=torch.float64)
        labels = torch.tensor([0, 1, 0, 1])
        self.assertEqual(float(binary_auroc(scores, labels)), 0.5)
        self.assertEqual(float(binary_auprc(scores, labels)), 0.5)

    def test_binary_validation_rejects_invalid_labels_and_single_class(self) -> None:
        scores = torch.tensor([0.1, 0.9], dtype=torch.float64)
        with self.assertRaises(ValueError):
            binary_auroc(scores, torch.tensor([0, 2]))
        with self.assertRaises(ValueError):
            binary_auprc(scores, torch.tensor([1, 1]))
        with self.assertRaises(ValueError):
            binary_false_alarm_rate(scores, torch.tensor([1, 1]))


if __name__ == "__main__":
    unittest.main()
