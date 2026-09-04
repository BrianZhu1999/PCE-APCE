from __future__ import annotations

import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from hilda_da.alpha import AlphaEvidenceTracker, liu_quantile
from hilda_da.baselines import denkf_analysis, letkf_analysis
from hilda_da.config import AlphaConfig, FlowConfig, HILDAConfig
from hilda_da.filter import AnalysisDiagnostics, HILDAFilter
from hilda_da.low_rank import gaspari_cohn, localized_low_rank_map
from hilda_da.observation_flow import (
    analytic_posterior_mixture,
    posterior_probability_flow,
)
from hilda_da.open_set import ConformalMismatchDetector
from experiments.run_assimilation import write_checkpoint


class LiuQuantileTests(unittest.TestCase):
    def test_symmetry_and_monotonicity(self) -> None:
        alpha = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], dtype=torch.float64)
        quantiles = liu_quantile(alpha)
        self.assertTrue(torch.all(quantiles[1:] > quantiles[:-1]))
        self.assertTrue(torch.allclose(quantiles, -torch.flip(quantiles, dims=[0])))
        self.assertAlmostEqual(float(quantiles[2]), 0.0, places=12)

    def test_rejects_endpoints(self) -> None:
        with self.assertRaises(ValueError):
            liu_quantile(torch.tensor([0.0, 0.5]))


class EvidenceTests(unittest.TestCase):
    def test_evidence_tracker_selects_supported_path(self) -> None:
        tracker = AlphaEvidenceTracker.create(AlphaConfig(initial_nodes=5))
        for _ in range(8):
            tracker.update(torch.tensor([-4.0, -2.0, 0.0, -2.0, -4.0]))
        self.assertEqual(int(torch.argmax(tracker.weights)), 2)
        self.assertAlmostEqual(tracker.continuous_estimate(), float(tracker.alpha[2]), places=3)

    def test_adaptation_inserts_interpolated_ensembles_and_synchronizes_state(self) -> None:
        config = AlphaConfig(initial_nodes=5, max_nodes=7)
        tracker = AlphaEvidenceTracker.create(config)
        tracker.log_scores.copy_(torch.tensor([-4.0, -1.0, 0.0, -1.0, -4.0]))
        ensembles = tracker.alpha[:, None, None].expand(-1, 3, 2).clone()

        adaptation = tracker.adapt_ensembles(ensembles)

        self.assertEqual(tracker.alpha.numel(), 7)
        self.assertEqual(adaptation.ensembles.shape[0], tracker.alpha.numel())
        self.assertEqual(tracker.log_scores.shape, tracker.alpha.shape)
        self.assertEqual(tracker.low_evidence_counts.shape, tracker.alpha.shape)
        self.assertTrue(torch.all(tracker.low_evidence_counts[1:-1] >= 0))
        for value in adaptation.added_alpha:
            index = int(torch.argmin((tracker.alpha - value).abs()))
            self.assertTrue(
                torch.allclose(adaptation.ensembles[index], value.expand_as(adaptation.ensembles[index]))
            )

    def test_pruning_requires_patience_and_preserves_protected_nodes(self) -> None:
        config = AlphaConfig(initial_nodes=7, max_nodes=7, prune_threshold=0.05)
        evidence = torch.tensor([0.0, 0.0, -100.0, -100.0, -100.0, 0.0, 0.0])

        early = AlphaEvidenceTracker.create(config)
        for _ in range(config.prune_patience - 1):
            early.update(evidence)
        early_alpha = early.alpha.clone()
        early_result = early.adapt_ensembles(early.alpha[:, None, None])
        self.assertEqual(early_result.removed_alpha.numel(), 0)
        self.assertTrue(torch.equal(early.alpha, early_alpha))

        mature = AlphaEvidenceTracker.create(config)
        for _ in range(config.prune_patience):
            mature.update(evidence)
        boundary = mature.alpha[[0, -1]].clone()
        top_three = mature.alpha[torch.topk(mature.weights, 3).indices].clone()
        mature_result = mature.adapt_ensembles(mature.alpha[:, None, None])
        self.assertGreater(mature_result.removed_alpha.numel(), 0)
        self.assertGreaterEqual(mature.alpha.numel(), 5)
        for value in torch.cat((boundary, top_three)):
            self.assertTrue(bool(torch.any(torch.isclose(mature.alpha, value))))

    def test_adaptation_obeys_capacity_and_minimum_spacing(self) -> None:
        config = AlphaConfig(initial_nodes=5, max_nodes=6, min_spacing=0.05)
        tracker = AlphaEvidenceTracker.create(config)
        tracker.log_scores.copy_(torch.tensor([-3.0, -1.0, 0.0, -1.0, -3.0]))
        adaptation = tracker.adapt_ensembles(tracker.alpha[:, None, None])
        self.assertLessEqual(tracker.alpha.numel(), config.max_nodes)
        self.assertTrue(torch.all(tracker.alpha[1:] - tracker.alpha[:-1] >= config.min_spacing))
        self.assertEqual(adaptation.ensembles.shape[0], tracker.alpha.numel())


class ObservationFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260804)
        self.forecast = torch.randn(24, 2, dtype=torch.float64) + torch.tensor(
            [2.0, -1.5], dtype=torch.float64
        )
        self.observation = torch.tensor([0.1, -0.2], dtype=torch.float64)
        self.noise = 0.2 * torch.eye(2, dtype=torch.float64)

    def test_mixture_is_normalized_and_spd(self) -> None:
        mixture = analytic_posterior_mixture(self.forecast, self.observation, self.noise)
        self.assertAlmostEqual(float(mixture.weights.sum()), 1.0, places=10)
        self.assertTrue(torch.all(torch.linalg.eigvalsh(mixture.covariance) > 0))
        self.assertTrue(torch.isfinite(mixture.log_evidence))

    def test_probability_flow_reduces_mean_innovation(self) -> None:
        mixture = analytic_posterior_mixture(self.forecast, self.observation, self.noise)
        analysis = posterior_probability_flow(
            self.forecast,
            mixture,
            FlowConfig(steps=8, sinkhorn_iterations=150),
        )
        before = torch.linalg.vector_norm(self.forecast.mean(dim=0) - self.observation)
        after = torch.linalg.vector_norm(analysis.mean(dim=0) - self.observation)
        self.assertLess(float(after), float(before))

    def test_moment_matched_flow_recovers_target_mean_and_covariance(self) -> None:
        mixture = analytic_posterior_mixture(self.forecast, self.observation, self.noise)
        analysis = posterior_probability_flow(
            self.forecast,
            mixture,
            FlowConfig(steps=8, sinkhorn_iterations=150, moment_matching=True),
        )
        target_mean = torch.sum(mixture.weights[:, None] * mixture.means, dim=0)
        centred = mixture.means - target_mean
        target_covariance = mixture.covariance + (
            mixture.weights[:, None, None]
            * centred[:, :, None]
            * centred[:, None, :]
        ).sum(dim=0)
        analysis_covariance = torch.cov(analysis.mT, correction=0)
        self.assertTrue(torch.allclose(analysis.mean(dim=0), target_mean, atol=1e-10, rtol=1e-10))
        self.assertTrue(torch.allclose(analysis_covariance, target_covariance, atol=1e-10, rtol=1e-10))


class LowRankTests(unittest.TestCase):
    def test_low_rank_map_and_localization(self) -> None:
        torch.manual_seed(7)
        state = torch.randn(20, 8, dtype=torch.float64)
        operator = torch.randn(8, 3, dtype=torch.float64)
        predicted = state @ operator
        localization = torch.ones(8, 3, dtype=torch.float64)
        localization[4:, 0] = 0.0
        mapping = localized_low_rank_map(state, predicted, HILDAConfig().low_rank, localization)
        self.assertEqual(mapping.gain.shape, (8, 3))
        self.assertLessEqual(mapping.retained_rank, 3)
        self.assertTrue(torch.all(mapping.gain[4:, 0] == 0))

    def test_gaspari_cohn_support(self) -> None:
        distance = torch.tensor([0.0, 0.5, 1.0, 2.0], dtype=torch.float64)
        taper = gaspari_cohn(distance, radius=1.0)
        self.assertAlmostEqual(float(taper[0]), 1.0, places=12)
        self.assertEqual(float(taper[-1]), 0.0)
        self.assertTrue(torch.all((taper >= 0) & (taper <= 1)))


class FilterAndBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(12)
        self.state = torch.randn(30, 6, dtype=torch.float64) + 1.0
        self.matrix = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
             [0.4, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.0, 0.4]],
            dtype=torch.float64,
        )
        self.operator = lambda ensemble: ensemble @ self.matrix
        self.observation = torch.zeros(3, dtype=torch.float64)
        self.noise = 0.1 * torch.eye(3, dtype=torch.float64)

    def _innovation(self, ensemble: torch.Tensor) -> float:
        return float(torch.linalg.vector_norm(self.operator(ensemble).mean(dim=0)))

    def test_hilda_reduces_linear_innovation(self) -> None:
        analysis, diagnostics = HILDAFilter().analyze_local(
            self.state,
            self.observation,
            self.operator,
            self.noise,
        )
        self.assertFalse(diagnostics.diverged)
        self.assertLess(self._innovation(analysis), self._innovation(self.state))

    def test_analyze_paths_returns_adapted_branch_count(self) -> None:
        config = HILDAConfig(alpha=AlphaConfig(initial_nodes=5, max_nodes=7))
        tracker = AlphaEvidenceTracker.create(config.alpha)
        path_ensembles = torch.stack(
            [self.state + float(index) for index in range(tracker.alpha.numel())]
        )
        hilda = HILDAFilter(config)

        def fake_analysis(branch, *_args, **_kwargs):
            branch_index = int(round(float(branch.mean() - self.state.mean())))
            evidence = -float((branch_index - 2) ** 2)
            diagnostics = AnalysisDiagnostics(1, 1.0, 0.5, 1, evidence, False)
            return branch, diagnostics

        hilda.analyze_local = fake_analysis
        result = hilda.analyze_paths(
            path_ensembles,
            tracker,
            self.observation,
            self.operator,
            self.noise,
        )
        self.assertEqual(result.ensembles.shape[0], tracker.alpha.numel())
        self.assertEqual(result.evidence_weights.numel(), tracker.alpha.numel())
        self.assertEqual(len(result.diagnostics), tracker.alpha.numel())

    def test_kalman_baselines_reduce_linear_innovation(self) -> None:
        denkf = denkf_analysis(self.state, self.observation, self.operator, self.noise)
        letkf = letkf_analysis(self.state, self.observation, self.operator, self.noise)
        self.assertLess(self._innovation(denkf), self._innovation(self.state))
        self.assertLess(self._innovation(letkf), self._innovation(self.state))


class OpenSetTests(unittest.TestCase):
    def test_small_calibration_set_uses_infinite_threshold(self) -> None:
        detector = ConformalMismatchDetector.calibrate(
            torch.tensor([1.0, 2.0]),
            false_alarm_level=0.05,
        )
        self.assertTrue(math.isinf(detector.threshold))
        self.assertFalse(detector.update(1.0e12))

    def test_finite_sample_conformal_rank(self) -> None:
        calibration = torch.arange(1.0, 20.0)
        detector = ConformalMismatchDetector.calibrate(
            calibration,
            false_alarm_level=0.05,
            required_exceedances=1,
        )
        self.assertEqual(detector.threshold, 19.0)
        self.assertFalse(detector.update(19.0))
        self.assertTrue(detector.update(20.0))

    def test_conformal_calibration_validates_parameters(self) -> None:
        calibration = torch.arange(1.0, 21.0)
        with self.assertRaises(ValueError):
            ConformalMismatchDetector.calibrate(calibration, false_alarm_level=0.0)
        with self.assertRaises(ValueError):
            ConformalMismatchDetector.calibrate(calibration, required_exceedances=0)
        with self.assertRaises(ValueError):
            ConformalMismatchDetector.calibrate(torch.tensor([1.0, math.inf]))

    def test_consecutive_alarm_rule(self) -> None:
        calibration = torch.arange(1.0, 101.0)
        detector = ConformalMismatchDetector.calibrate(
            calibration,
            false_alarm_level=0.05,
            required_exceedances=3,
        )
        self.assertFalse(detector.update(200.0))
        self.assertFalse(detector.update(200.0))
        self.assertTrue(detector.update(200.0))
        self.assertFalse(detector.update(0.0))


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_round_trip_is_atomic(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "checkpoint.pt"
            payload = {"next_step": 7, "state": torch.tensor([1.0, 2.0])}
            write_checkpoint(path, payload)
            restored = torch.load(path, weights_only=False)
            self.assertEqual(restored["next_step"], 7)
            self.assertTrue(torch.equal(restored["state"], payload["state"]))
            self.assertFalse(path.with_suffix(".tmp").exists())


if __name__ == "__main__":
    unittest.main()
