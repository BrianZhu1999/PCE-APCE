from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import torch

from hilda_da import strong_baselines
from hilda_da.observations import SparseObservation
from hilda_da.strong_baselines import (
    EnFFF2PConfig,
    EnSFConfig,
    IEnSFConfig,
    _construct_iensf_prior,
    enff_f2p_analysis,
    ensf_analysis,
    ensf_lr_analysis,
    ensf_lr_ridge_analysis,
    iensf_analysis,
)


class StrongBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(31)
        self.predictive = torch.randn(20, 6, dtype=torch.float64) + 0.8
        self.previous = self.predictive - 0.05 * torch.randn(20, 6, dtype=torch.float64)
        self.indices = torch.tensor([0, 2, 4], dtype=torch.int64)
        self.operator = SparseObservation(self.indices)
        self.observation = torch.zeros(3, dtype=torch.float64)
        self.covariance = 0.1 * torch.eye(3, dtype=torch.float64)

    def innovation(self, ensemble: torch.Tensor) -> float:
        return float(
            torch.linalg.vector_norm(
                self.operator(ensemble).mean(dim=0) - self.observation
            )
        )

    def test_ensf_is_finite(self) -> None:
        result = ensf_analysis(
            self.predictive,
            self.observation,
            self.operator,
            self.covariance,
            EnSFConfig(sampling_time_step_count=30),
            torch.Generator().manual_seed(9),
        )
        self.assertEqual(result.shape, self.predictive.shape)
        self.assertTrue(torch.isfinite(result).all())

    def test_ensf_lr_updates_observed_and_unobserved_state(self) -> None:
        result = ensf_lr_analysis(
            self.predictive,
            self.observation,
            self.operator,
            self.covariance,
            EnSFConfig(sampling_time_step_count=30),
            torch.Generator().manual_seed(10),
        )
        self.assertTrue(torch.isfinite(result).all())
        self.assertFalse(torch.allclose(result[:, self.indices], self.predictive[:, self.indices]))
        self.assertFalse(torch.allclose(result[:, 1], self.predictive[:, 1]))

    def test_ensf_lr_ridge_is_finite_for_rank_deficient_observation_ensemble(self) -> None:
        predictive = self.predictive.clone()
        predictive[:, 2] = predictive[:, 0]
        result = ensf_lr_ridge_analysis(
            predictive,
            self.observation,
            self.operator,
            self.covariance,
            EnSFConfig(sampling_time_step_count=12),
            torch.Generator().manual_seed(101),
        )
        self.assertTrue(torch.isfinite(result).all())

    def test_enff_f2p_guidance_reduces_innovation(self) -> None:
        result = enff_f2p_analysis(
            self.previous,
            self.predictive,
            self.observation,
            self.operator,
            self.covariance,
            EnFFF2PConfig(
                sampling_time_step_count=6,
                sigma_min=0.01,
                guidance_lambda=0.05,
            ),
            torch.Generator().manual_seed(11),
        )
        self.assertTrue(torch.isfinite(result).all())
        self.assertLess(self.innovation(result), self.innovation(self.predictive))

    def test_iensf_is_finite_and_reduces_innovation(self) -> None:
        result = iensf_analysis(
            self.predictive,
            self.observation,
            self.operator,
            self.covariance,
            IEnSFConfig(
                gamma=1.0,
                sampling_time_step_count=24,
            ),
            torch.Generator().manual_seed(12),
        )
        self.assertEqual(result.shape, self.predictive.shape)
        self.assertTrue(torch.isfinite(result).all())
        self.assertLess(self.innovation(result), self.innovation(self.predictive))

    def test_iensf_generic_operator_uses_autograd_jacobian(self) -> None:
        predictive = self.predictive[:8, :3]

        def nonlinear_operator(state: torch.Tensor) -> torch.Tensor:
            return (state[:, :1] + 0.2 * state[:, 1:2].square())

        result = iensf_analysis(
            predictive,
            torch.zeros(1, dtype=torch.float64),
            nonlinear_operator,
            0.1 * torch.eye(1, dtype=torch.float64),
            IEnSFConfig(
                sampling_time_step_count=4,
                refinement_iterations=1,
            ),
            torch.Generator().manual_seed(13),
        )
        self.assertEqual(result.shape, predictive.shape)
        self.assertTrue(torch.isfinite(result).all())

    def test_iensf_variance_split_modes_match_their_definitions(self) -> None:
        gamma = 0.4
        sample_trace = torch.cov(self.predictive.mT).trace()
        expected_ratios = {
            "variance_consistent": 1.0,
            "literal": 1.0 - gamma**2 + gamma,
        }
        for mode, expected_ratio in expected_ratios.items():
            with self.subTest(mode=mode):
                prior = _construct_iensf_prior(
                    self.predictive,
                    IEnSFConfig(gamma=gamma, variance_split_mode=mode),
                )
                component_trace = torch.cov(prior.component_means.mT).trace()
                total_trace = component_trace + prior.covariance.trace()
                self.assertAlmostEqual(
                    float(total_trace / sample_trace),
                    expected_ratio,
                    places=10,
                )

    def test_iensf_low_rank_svd_retries_on_cpu_after_backend_failure(self) -> None:
        factor = torch.tensor(
            [
                [1.0, 1.0, 0.0],
                [2.0, 2.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        original_svd = torch.linalg.svd
        calls = 0

        def fail_once_then_svd(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("simulated device SVD non-convergence")
            return original_svd(*args, **kwargs)

        with patch.object(
            torch.linalg,
            "svd",
            side_effect=fail_once_then_svd,
        ):
            covariance = strong_baselines._low_rank_from_factor(factor, 1e-12)

        reconstructed = (
            covariance.basis
            @ torch.diag(covariance.eigenvalues)
            @ covariance.basis.mT
        )
        self.assertEqual(calls, 2)
        self.assertTrue(
            torch.allclose(reconstructed, factor @ factor.mT, atol=1e-12, rtol=1e-12)
        )

    def test_iensf_low_rank_rejects_non_finite_factor(self) -> None:
        factor = torch.tensor([[1.0], [float("nan")]], dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            strong_baselines._low_rank_from_factor(factor, 1e-12)

    def test_iensf_observation_weights_include_quadratic_and_logdet(self) -> None:
        dtype = torch.float64
        component_means = torch.tensor([[-1.0], [2.0]], dtype=dtype)
        variance = torch.tensor([0.5], dtype=dtype)
        covariance = strong_baselines._LowRankCovariance(
            torch.ones(1, 1, dtype=dtype),
            variance,
        )
        prior = strong_baselines._IEnSFPrior(
            torch.tensor([0.5], dtype=dtype),
            component_means,
            covariance,
        )
        reference = strong_baselines._IEnSFReference(
            torch.zeros(1, dtype=dtype),
            covariance,
        )
        state = torch.tensor([[0.3]], dtype=dtype)
        time_value = torch.tensor(0.4, dtype=dtype)
        alpha = 1.0 - time_value
        beta_squared = time_value
        observation = torch.tensor([1.2], dtype=dtype)
        observation_covariance = torch.tensor([[0.3]], dtype=dtype)
        operator = SparseObservation(
            torch.tensor([0], dtype=torch.int64),
            transform="square_signed",
        )

        denominator = alpha.square() * variance + beta_squared
        differences = state.unsqueeze(1) - alpha * component_means.unsqueeze(0)
        component_scores = -differences / denominator
        log_prior = -0.5 * differences.square().squeeze(-1) / denominator
        reverse_means = component_means.unsqueeze(0) + (
            differences * (alpha * variance / denominator)
        )
        predicted = reverse_means.abs() * reverse_means
        conditional_variance = variance * beta_squared / denominator
        jacobian = 2.0 * reverse_means.abs()
        predictive_variance = jacobian.square() * conditional_variance + 0.3
        residual = observation - predicted
        quadratic = residual.square() / predictive_variance
        log_determinant = torch.log(predictive_variance)
        log_observation = -0.5 * (
            math.log(2.0 * math.pi) + log_determinant + quadratic
        ).squeeze(-1)
        expected_weights = torch.softmax(log_prior + log_observation, dim=1)
        expected_score = (
            expected_weights.unsqueeze(-1) * component_scores
        ).sum(dim=1)

        with patch.object(
            strong_baselines,
            "_likelihood_score",
            return_value=torch.zeros_like(state),
        ):
            actual_score = strong_baselines._iensf_posterior_score(
                state,
                time_value,
                prior,
                reference,
                observation,
                operator,
                observation_covariance,
                IEnSFConfig(max_score_component=1e6, cholesky_jitter=0.0),
            )

        quadratic_only_weights = torch.softmax(
            log_prior - 0.5 * quadratic.squeeze(-1),
            dim=1,
        )
        self.assertTrue(
            torch.allclose(expected_weights.sum(dim=1), torch.ones(1, dtype=dtype))
        )
        self.assertFalse(torch.allclose(expected_weights, quadratic_only_weights))
        self.assertTrue(torch.allclose(actual_score, expected_score, atol=1e-12, rtol=1e-12))

    def test_iensf_analytic_jacobian_scaling_matches_j_of_t(self) -> None:
        dtype = torch.float64
        inverse_sqrt_two = 1.0 / math.sqrt(2.0)
        basis = torch.tensor(
            [
                [inverse_sqrt_two, -inverse_sqrt_two],
                [inverse_sqrt_two, inverse_sqrt_two],
            ],
            dtype=dtype,
        )
        eigenvalues = torch.tensor([2.0, 0.5], dtype=dtype)
        covariance = strong_baselines._LowRankCovariance(basis, eigenvalues)
        values = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=dtype)
        alpha = torch.tensor(0.7, dtype=dtype)
        beta_squared = torch.tensor(0.3, dtype=dtype)

        coefficients = alpha * eigenvalues / (
            alpha.square() * eigenvalues + beta_squared
        )
        j_of_t = basis @ torch.diag(coefficients) @ basis.mT
        expected = values @ j_of_t
        actual = strong_baselines._apply_jacobian_scaling(
            values,
            covariance,
            alpha,
            beta_squared,
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=1e-12))

    def test_iensf_probability_flow_drift_matches_analytic_ode(self) -> None:
        dtype = torch.float64
        terminal = torch.tensor([[0.4], [-0.7]], dtype=dtype)
        predictive = torch.tensor([[-1.0], [0.0], [2.0]], dtype=dtype)
        config = IEnSFConfig(
            gamma=0.5,
            sampling_time_step_count=2,
            endpoint_epsilon=0.1,
        )
        prior = _construct_iensf_prior(predictive, config)
        reference = strong_baselines._IEnSFReference(
            predictive.mean(dim=0),
            strong_baselines._sample_covariance_low_rank(
                predictive,
                config.spectral_tolerance,
            ),
        )
        constant_score = 0.25

        def score_stub(state, *_args, **_kwargs):
            return torch.full_like(state, constant_score)

        with patch.object(
            strong_baselines,
            "_iensf_posterior_score",
            side_effect=score_stub,
        ):
            actual = strong_baselines._iensf_probability_flow(
                terminal,
                prior,
                reference,
                torch.zeros(1, dtype=dtype),
                SparseObservation(torch.tensor([0], dtype=torch.int64)),
                torch.eye(1, dtype=dtype),
                config,
            )

        expected = terminal.clone()
        path_coordinate = torch.linspace(
            torch.logit(torch.tensor(1.0 - config.endpoint_epsilon, dtype=dtype)),
            torch.logit(torch.tensor(config.endpoint_epsilon, dtype=dtype)),
            config.sampling_time_step_count + 1,
            dtype=dtype,
        )

        def analytic_drift(state, coordinate):
            time_value = torch.sigmoid(coordinate)
            alpha = 1.0 - time_value
            diffusion_squared = 1.0 + 2.0 * time_value / alpha
            return time_value * alpha * (
                -state / alpha - 0.5 * diffusion_squared * constant_score
            )

        for now, following in path_coordinate.unfold(0, 2, 1):
            step = following - now
            first = analytic_drift(expected, now)
            proposal = expected + step * first
            second = analytic_drift(proposal, following)
            expected = expected + 0.5 * step * (first + second)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=1e-12))

    def test_iensf_logit_time_keeps_low_rank_nullspace_bounded(self) -> None:
        dtype = torch.float64
        zero_mean = torch.zeros(2, 2, dtype=dtype)
        covariance = strong_baselines._LowRankCovariance(
            torch.tensor([[1.0], [0.0]], dtype=dtype),
            torch.tensor([1.0], dtype=dtype),
        )
        prior = strong_baselines._IEnSFPrior(
            torch.zeros(2, dtype=dtype),
            zero_mean,
            covariance,
        )
        reference = strong_baselines._IEnSFReference(
            torch.zeros(2, dtype=dtype),
            covariance,
        )
        config = IEnSFConfig(sampling_time_step_count=40, endpoint_epsilon=1e-3)
        with patch.object(
            strong_baselines,
            "_iensf_posterior_score",
            side_effect=lambda state, *_args, **_kwargs: torch.stack(
                (-state[:, 0], -state[:, 1] / _args[0]),
                dim=1,
            ),
        ):
            result = strong_baselines._iensf_probability_flow(
                torch.tensor([[0.2, 0.3], [-0.1, -0.4]], dtype=dtype),
                prior,
                reference,
                torch.zeros(1, dtype=dtype),
                SparseObservation(torch.tensor([0], dtype=torch.int64)),
                torch.eye(1, dtype=dtype),
                config,
            )
        self.assertTrue(torch.isfinite(result).all())
        self.assertLess(float(result.abs().max()), 10.0)

    def test_iensf_variance_consistent_gamma_squared_conserves_covariance(self) -> None:
        predictive = torch.tensor(
            [
                [-2.0, 0.5],
                [-0.5, -1.0],
                [1.0, 2.0],
                [3.0, -0.5],
            ],
            dtype=torch.float64,
        )
        gamma = 0.4
        prior = _construct_iensf_prior(
            predictive,
            IEnSFConfig(gamma=gamma, variance_split_mode="variance_consistent"),
        )
        sample_covariance = torch.cov(predictive.mT)
        component_covariance = torch.cov(prior.component_means.mT)
        within_covariance = (
            prior.covariance.basis
            @ torch.diag(prior.covariance.eigenvalues)
            @ prior.covariance.basis.mT
        )
        self.assertTrue(
            torch.allclose(
                component_covariance,
                (1.0 - gamma**2) * sample_covariance,
                atol=1e-12,
                rtol=1e-12,
            )
        )
        self.assertTrue(
            torch.allclose(
                within_covariance,
                gamma**2 * sample_covariance,
                atol=1e-12,
                rtol=1e-12,
            )
        )
        self.assertTrue(
            torch.allclose(
                component_covariance + within_covariance,
                sample_covariance,
                atol=1e-12,
                rtol=1e-12,
            )
        )

    def test_iensf_keeps_prior_fixed_and_updates_only_reference(self) -> None:
        prior_ids: list[int] = []
        prior_components: list[torch.Tensor] = []
        prior_eigenvalues: list[torch.Tensor] = []
        terminal_ids: list[int] = []
        reference_means: list[torch.Tensor] = []
        reference_eigenvalues: list[torch.Tensor] = []

        def instrumented_flow(*args, **kwargs):
            prior_ids.append(id(args[1]))
            prior_components.append(args[1].component_means.detach().clone())
            prior_eigenvalues.append(args[1].covariance.eigenvalues.detach().clone())
            terminal_ids.append(id(args[0]))
            reference_means.append(args[2].mean.detach().clone())
            reference_eigenvalues.append(args[2].covariance.eigenvalues.detach().clone())
            iteration = len(prior_ids)
            return iteration * args[0] + float(iteration)

        config = IEnSFConfig(
            gamma=1.0,
            sampling_time_step_count=8,
            refinement_iterations=4,
        )
        with patch.object(
            strong_baselines,
            "_iensf_probability_flow",
            side_effect=instrumented_flow,
        ):
            result = iensf_analysis(
                self.predictive[:10],
                self.observation,
                self.operator,
                self.covariance,
                config,
                torch.Generator().manual_seed(14),
            )
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(len(prior_ids), 4)
        self.assertEqual(len(set(prior_ids)), 1)
        self.assertEqual(len(set(terminal_ids)), 1)
        for components in prior_components[1:]:
            self.assertTrue(torch.equal(components, prior_components[0]))
        for eigenvalues in prior_eigenvalues[1:]:
            self.assertTrue(torch.equal(eigenvalues, prior_eigenvalues[0]))
        self.assertFalse(torch.allclose(reference_means[0], reference_means[-1]))
        self.assertFalse(
            torch.allclose(reference_eigenvalues[0], reference_eigenvalues[-1])
        )


if __name__ == "__main__":
    unittest.main()
