from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal

import torch

from .math_utils import stable_cholesky
from .observations import SparseObservation

ObservationOperator = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class EnSFConfig:
    sampling_time_step_count: int = 100
    epsilon_alpha: float = 1.0
    epsilon_beta: float = 0.005
    max_score_component: float = 1000.0


@dataclass(frozen=True)
class IEnSFConfig:
    """Configuration for a transparent IEnSF equation reimplementation.

    The sampler is the probability-flow ODE associated with the paper's
    reverse SDE, integrated with fixed-step Heun updates.  The paper prints
    ``Sigma = gamma * sample_covariance`` but its variance identity requires
    ``gamma**2``.  Both interpretations are exposed explicitly.
    """

    gamma: float = 0.5
    variance_split_mode: Literal["literal", "variance_consistent"] = (
        "variance_consistent"
    )
    sampling_time_step_count: int = 40
    refinement_iterations: int = 4
    reference_mean_smoothing: float = 0.5
    reference_covariance_smoothing: float = 0.5
    component_mean_inflation: float = 1.0
    endpoint_epsilon: float = 1e-3
    spectral_tolerance: float = 1e-10
    cholesky_jitter: float = 1e-8
    max_score_component: float = 1000.0


@dataclass(frozen=True)
class _LowRankCovariance:
    basis: torch.Tensor
    eigenvalues: torch.Tensor

    @property
    def dimension(self) -> int:
        return self.basis.shape[0]

    def factor(self) -> torch.Tensor:
        return self.basis * self.eigenvalues.clamp_min(0.0).sqrt().unsqueeze(0)

    def trace(self) -> torch.Tensor:
        return self.eigenvalues.sum()


@dataclass(frozen=True)
class _IEnSFPrior:
    ensemble_mean: torch.Tensor
    component_means: torch.Tensor
    covariance: _LowRankCovariance


@dataclass(frozen=True)
class _IEnSFReference:
    mean: torch.Tensor
    covariance: _LowRankCovariance


@dataclass(frozen=True)
class EnFFF2PConfig:
    sampling_time_step_count: int = 5
    sigma_min: float = 0.001
    guidance_lambda: float = 0.005
    independent_coupling: bool = False


def _observation_log_score(
    state: torch.Tensor,
    observation: torch.Tensor,
    observation_operator: ObservationOperator,
    observation_covariance: torch.Tensor,
) -> torch.Tensor:
    state_for_gradient = state.detach().requires_grad_(True)
    residual = observation_operator(state_for_gradient) - observation.unsqueeze(0)
    factor = stable_cholesky(observation_covariance)
    whitened = torch.linalg.solve_triangular(
        factor,
        residual.mT,
        upper=False,
    ).mT
    log_likelihood = -0.5 * whitened.square().sum(dim=1)
    gradient, = torch.autograd.grad(log_likelihood.sum(), state_for_gradient)
    return gradient.detach()


def _validate_iensf_config(config: IEnSFConfig) -> None:
    if not 0.0 <= config.gamma <= 1.0:
        raise ValueError("IEnSF gamma must lie in [0, 1]")
    if config.variance_split_mode not in {"literal", "variance_consistent"}:
        raise ValueError("Unknown IEnSF variance split mode")
    if config.sampling_time_step_count < 2:
        raise ValueError("IEnSF requires at least two probability-flow steps")
    if config.refinement_iterations < 1:
        raise ValueError("IEnSF requires at least one reference refinement")
    if not 0.0 <= config.reference_mean_smoothing <= 1.0:
        raise ValueError("IEnSF mean smoothing must lie in [0, 1]")
    if not 0.0 <= config.reference_covariance_smoothing <= 1.0:
        raise ValueError("IEnSF covariance smoothing must lie in [0, 1]")
    if config.component_mean_inflation <= 0.0:
        raise ValueError("IEnSF component-mean inflation must be positive")
    if not 0.0 < config.endpoint_epsilon < 0.5:
        raise ValueError("IEnSF endpoint epsilon must lie in (0, 0.5)")


def _low_rank_from_factor(
    factor: torch.Tensor,
    relative_tolerance: float,
) -> _LowRankCovariance:
    if factor.ndim != 2:
        raise ValueError("Covariance factor must have shape [state, rank]")
    if factor.shape[1] == 0:
        return _LowRankCovariance(
            factor.new_zeros((factor.shape[0], 0)),
            factor.new_zeros((0,)),
        )
    if not bool(torch.isfinite(factor).all()):
        raise ValueError("Covariance factor contains non-finite values")
    try:
        basis, singular_values, _ = torch.linalg.svd(
            factor,
            full_matrices=False,
        )
    except RuntimeError as device_error:
        # cuSOLVER can fail on small, nearly rank-deficient factors.  A CPU
        # LAPACK retry preserves the same SVD definition and only changes the
        # numerical backend used for this low-rank covariance construction.
        cpu_dtype = torch.float64 if factor.dtype in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
        } else factor.dtype
        cpu_factor = factor.detach().to(device="cpu", dtype=cpu_dtype)
        try:
            cpu_basis, cpu_singular_values, _ = torch.linalg.svd(
                cpu_factor,
                full_matrices=False,
            )
        except RuntimeError as cpu_error:
            raise RuntimeError(
                "IEnSF low-rank covariance SVD failed on both the original "
                "device and the CPU fallback"
            ) from cpu_error
        basis = cpu_basis.to(device=factor.device, dtype=factor.dtype)
        singular_values = cpu_singular_values.to(
            device=factor.device,
            dtype=factor.dtype,
        )
    threshold = relative_tolerance * singular_values.max().clamp_min(
        torch.finfo(factor.dtype).eps
    )
    retained = singular_values > threshold
    return _LowRankCovariance(
        basis[:, retained],
        singular_values[retained].square(),
    )


def _sample_covariance_low_rank(
    ensemble: torch.Tensor,
    relative_tolerance: float,
) -> _LowRankCovariance:
    if ensemble.ndim != 2 or ensemble.shape[0] < 2:
        raise ValueError("IEnSF requires an ensemble with shape [ensemble, state]")
    anomalies = ensemble - ensemble.mean(dim=0)
    factor = anomalies.mT / math.sqrt(ensemble.shape[0] - 1)
    return _low_rank_from_factor(factor, relative_tolerance)


def _construct_iensf_prior(
    predictive_ensemble: torch.Tensor,
    config: IEnSFConfig,
) -> _IEnSFPrior:
    """Construct Eq. (22), retaining the paper's literal/corrected ambiguity."""

    _validate_iensf_config(config)
    mean = predictive_ensemble.mean(dim=0)
    anomalies = predictive_ensemble - mean
    component_scale = config.component_mean_inflation * math.sqrt(
        max(0.0, 1.0 - config.gamma**2)
    )
    component_means = mean + component_scale * anomalies
    sample_covariance = _sample_covariance_low_rank(
        predictive_ensemble,
        config.spectral_tolerance,
    )
    if config.variance_split_mode == "literal":
        covariance_scale = config.gamma
    else:
        covariance_scale = config.gamma**2
    covariance = _LowRankCovariance(
        sample_covariance.basis,
        covariance_scale * sample_covariance.eigenvalues,
    )
    return _IEnSFPrior(mean, component_means, covariance)


def _shifted_solve(
    values: torch.Tensor,
    covariance: _LowRankCovariance,
    alpha: torch.Tensor,
    beta_squared: torch.Tensor,
) -> torch.Tensor:
    result = values / beta_squared
    if covariance.eigenvalues.numel() == 0:
        return result
    projected = values @ covariance.basis
    denominator = alpha.square() * covariance.eigenvalues + beta_squared
    correction = projected * (denominator.reciprocal() - beta_squared.reciprocal())
    return result + correction @ covariance.basis.mT


def _conditional_mean(
    values: torch.Tensor,
    means: torch.Tensor,
    covariance: _LowRankCovariance,
    alpha: torch.Tensor,
    beta_squared: torch.Tensor,
) -> torch.Tensor:
    differences = values - alpha * means
    if covariance.eigenvalues.numel() == 0:
        return means.expand_as(differences)
    denominator = alpha.square() * covariance.eigenvalues + beta_squared
    coefficient = alpha * covariance.eigenvalues / denominator
    return means + ((differences @ covariance.basis) * coefficient) @ covariance.basis.mT


def _conditional_eigenvalues(
    covariance: _LowRankCovariance,
    alpha: torch.Tensor,
    beta_squared: torch.Tensor,
) -> torch.Tensor:
    denominator = alpha.square() * covariance.eigenvalues + beta_squared
    return covariance.eigenvalues * beta_squared / denominator


def _apply_jacobian_scaling(
    values: torch.Tensor,
    covariance: _LowRankCovariance,
    alpha: torch.Tensor,
    beta_squared: torch.Tensor,
) -> torch.Tensor:
    if covariance.eigenvalues.numel() == 0:
        return torch.zeros_like(values)
    denominator = alpha.square() * covariance.eigenvalues + beta_squared
    coefficient = alpha * covariance.eigenvalues / denominator
    return ((values @ covariance.basis) * coefficient) @ covariance.basis.mT


def _sparse_values_and_projection(
    points: torch.Tensor,
    observation_operator: SparseObservation,
    basis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = observation_operator.indices.to(points.device)
    selected = points.index_select(-1, indices)
    if observation_operator.transform == "linear":
        values = selected
        derivative = torch.ones_like(selected)
    elif observation_operator.transform == "atan":
        values = torch.atan(selected)
        derivative = 1.0 / (1.0 + selected.square())
    elif observation_operator.transform == "square_signed":
        values = selected.abs() * selected
        derivative = 2.0 * selected.abs()
    else:
        raise ValueError(observation_operator.transform)
    selected_basis = basis.index_select(0, indices)
    projection = derivative.unsqueeze(-1) * selected_basis
    return values, projection


def _values_and_jacobian_projection(
    points: torch.Tensor,
    observation_operator: ObservationOperator,
    basis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(observation_operator, SparseObservation):
        return _sparse_values_and_projection(points, observation_operator, basis)
    leading_shape = points.shape[:-1]
    flat_points = points.reshape(-1, points.shape[-1])

    def single_operator(point: torch.Tensor) -> torch.Tensor:
        return observation_operator(point.unsqueeze(0)).squeeze(0)

    values = torch.vmap(single_operator)(flat_points)
    jacobians = torch.vmap(torch.func.jacrev(single_operator))(flat_points)
    projections = jacobians @ basis
    observation_dimension = values.shape[-1]
    return (
        values.reshape(*leading_shape, observation_dimension),
        projections.reshape(*leading_shape, observation_dimension, basis.shape[1]),
    )


def _likelihood_score(
    points: torch.Tensor,
    observation: torch.Tensor,
    observation_operator: ObservationOperator,
    observation_covariance: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(observation_operator, SparseObservation):
        return _observation_log_score(
            points,
            observation,
            observation_operator,
            observation_covariance,
        )
    indices = observation_operator.indices.to(points.device)
    selected = points.index_select(-1, indices)
    if observation_operator.transform == "linear":
        predicted = selected
        derivative = torch.ones_like(selected)
    elif observation_operator.transform == "atan":
        predicted = torch.atan(selected)
        derivative = 1.0 / (1.0 + selected.square())
    elif observation_operator.transform == "square_signed":
        predicted = selected.abs() * selected
        derivative = 2.0 * selected.abs()
    else:
        raise ValueError(observation_operator.transform)
    factor = stable_cholesky(observation_covariance)
    precision_residual = torch.cholesky_solve(
        (observation.unsqueeze(0) - predicted).unsqueeze(-1),
        factor,
    ).squeeze(-1)
    score = torch.zeros_like(points)
    score.index_copy_(1, indices, derivative * precision_residual)
    return score


def _batched_gaussian_logpdf(
    residual: torch.Tensor,
    covariance: torch.Tensor,
    relative_jitter: float,
) -> torch.Tensor:
    covariance = 0.5 * (covariance + covariance.mT)
    dimension = covariance.shape[-1]
    identity = torch.eye(
        dimension,
        dtype=covariance.dtype,
        device=covariance.device,
    )
    scale = torch.diagonal(covariance, dim1=-2, dim2=-1).abs().mean(dim=-1)
    scale = scale.clamp_min(torch.finfo(covariance.dtype).eps)
    factor = None
    for multiplier in (1.0, 10.0, 100.0, 1000.0):
        candidate, info = torch.linalg.cholesky_ex(
            covariance
            + multiplier * relative_jitter * scale[..., None, None] * identity
        )
        if int(info.max()) == 0:
            factor = candidate
            break
    if factor is None:
        raise RuntimeError("IEnSF observation covariance is not positive definite")
    whitened = torch.linalg.solve_triangular(
        factor,
        residual.unsqueeze(-1),
        upper=False,
    ).squeeze(-1)
    quadratic = whitened.square().sum(dim=-1)
    log_determinant = 2.0 * torch.log(
        torch.diagonal(factor, dim1=-2, dim2=-1)
    ).sum(dim=-1)
    return -0.5 * (
        dimension * math.log(2.0 * math.pi) + log_determinant + quadratic
    )


def _iensf_posterior_score(
    state: torch.Tensor,
    time_value: torch.Tensor,
    prior: _IEnSFPrior,
    reference: _IEnSFReference,
    observation: torch.Tensor,
    observation_operator: ObservationOperator,
    observation_covariance: torch.Tensor,
    config: IEnSFConfig,
) -> torch.Tensor:
    alpha = 1.0 - time_value
    beta_squared = time_value
    component_means = prior.component_means.unsqueeze(0)
    pairwise_state = state.unsqueeze(1)
    differences = pairwise_state - alpha * component_means
    solved = _shifted_solve(
        differences,
        prior.covariance,
        alpha,
        beta_squared,
    )
    component_scores = -solved
    log_prior = -0.5 * (differences * solved).sum(dim=-1)

    reverse_means = _conditional_mean(
        pairwise_state,
        component_means,
        prior.covariance,
        alpha,
        beta_squared,
    )
    predicted, jacobian_projection = _values_and_jacobian_projection(
        reverse_means,
        observation_operator,
        prior.covariance.basis,
    )
    conditional_eigenvalues = _conditional_eigenvalues(
        prior.covariance,
        alpha,
        beta_squared,
    )
    projected = jacobian_projection * conditional_eigenvalues.sqrt()
    predictive_covariance = projected @ projected.mT
    predictive_covariance = predictive_covariance + observation_covariance
    residual = observation - predicted
    log_observation = _batched_gaussian_logpdf(
        residual,
        predictive_covariance,
        config.cholesky_jitter,
    )
    posterior_weights = torch.softmax(log_prior + log_observation, dim=1)
    prior_score = (posterior_weights.unsqueeze(-1) * component_scores).sum(dim=1)

    reference_points = _conditional_mean(
        state,
        reference.mean.unsqueeze(0),
        reference.covariance,
        alpha,
        beta_squared,
    )
    observation_score = _likelihood_score(
        reference_points,
        observation,
        observation_operator,
        observation_covariance,
    )
    scaled_observation_score = _apply_jacobian_scaling(
        observation_score,
        prior.covariance,
        alpha,
        beta_squared,
    )
    return (prior_score + scaled_observation_score).clamp(
        min=-config.max_score_component,
        max=config.max_score_component,
    )


def _iensf_probability_flow(
    terminal_state: torch.Tensor,
    prior: _IEnSFPrior,
    reference: _IEnSFReference,
    observation: torch.Tensor,
    observation_operator: ObservationOperator,
    observation_covariance: torch.Tensor,
    config: IEnSFConfig,
) -> torch.Tensor:
    """Integrate the IEnSF probability-flow ODE with logit-time Heun.

    The analytic drift has endpoint coefficients involving ``1/t`` and
    ``1/(1-t)``.  Uniform steps in ``t`` therefore become unstable in the
    low-rank null space near ``t=0``.  Integrating the same ODE after the
    coordinate change ``s=logit(t)`` keeps both endpoint ratios bounded while
    retaining the frozen number of Heun intervals.
    """

    state = terminal_state.clone()
    path_coordinate = torch.linspace(
        torch.logit(state.new_tensor(1.0 - config.endpoint_epsilon)),
        torch.logit(state.new_tensor(config.endpoint_epsilon)),
        config.sampling_time_step_count + 1,
        dtype=state.dtype,
        device=state.device,
    )

    def drift(values: torch.Tensor, coordinate: torch.Tensor) -> torch.Tensor:
        time_value = torch.sigmoid(coordinate)
        score = _iensf_posterior_score(
            values,
            time_value,
            prior,
            reference,
            observation,
            observation_operator,
            observation_covariance,
            config,
        )
        alpha = 1.0 - time_value
        forward_drift = -values / alpha
        diffusion_squared = 1.0 + 2.0 * time_value / alpha
        return time_value * alpha * (
            forward_drift - 0.5 * diffusion_squared * score
        )

    for now, following in path_coordinate.unfold(0, 2, 1):
        ds = following - now
        first_drift = drift(state, now)
        proposal = state + ds * first_drift
        second_drift = drift(proposal, following)
        state = (state + 0.5 * ds * (first_drift + second_drift)).detach()
    return state


def _smooth_reference(
    reference: _IEnSFReference,
    posterior_ensemble: torch.Tensor,
    config: IEnSFConfig,
) -> _IEnSFReference:
    posterior_mean = posterior_ensemble.mean(dim=0)
    posterior_covariance = _sample_covariance_low_rank(
        posterior_ensemble,
        config.spectral_tolerance,
    )
    eta_mean = config.reference_mean_smoothing
    mean = (1.0 - eta_mean) * reference.mean + eta_mean * posterior_mean
    eta_covariance = config.reference_covariance_smoothing
    covariance_factor = torch.cat(
        (
            math.sqrt(1.0 - eta_covariance) * reference.covariance.factor(),
            math.sqrt(eta_covariance) * posterior_covariance.factor(),
        ),
        dim=1,
    )
    covariance = _low_rank_from_factor(
        covariance_factor,
        config.spectral_tolerance,
    )
    return _IEnSFReference(mean, covariance)


def iensf_analysis(
    predictive_ensemble: torch.Tensor,
    observation: torch.Tensor,
    observation_operator: ObservationOperator,
    observation_covariance: torch.Tensor,
    config: IEnSFConfig | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """IEnSF with a fixed prior GMM and four reference-only refinements.

    The routine implements Eq. (57) with the complete Gaussian density for
    ``w_obs``.  One terminal Gaussian ensemble is reused across refinements;
    only the auxiliary reference Gaussian is smoothed between ODE solves.
    """

    config = config or IEnSFConfig()
    _validate_iensf_config(config)
    if predictive_ensemble.ndim != 2 or predictive_ensemble.shape[0] < 2:
        raise ValueError("IEnSF requires shape [ensemble, state] with N >= 2")
    prior = _construct_iensf_prior(predictive_ensemble, config)
    reference = _IEnSFReference(
        predictive_ensemble.mean(dim=0),
        _sample_covariance_low_rank(
            predictive_ensemble,
            config.spectral_tolerance,
        ),
    )
    terminal_state = torch.randn(
        predictive_ensemble.shape,
        dtype=predictive_ensemble.dtype,
        device=predictive_ensemble.device,
        generator=generator,
    )
    posterior = predictive_ensemble
    for _ in range(config.refinement_iterations):
        posterior = _iensf_probability_flow(
            terminal_state,
            prior,
            reference,
            observation,
            observation_operator,
            observation_covariance,
            config,
        )
        reference = _smooth_reference(reference, posterior, config)
    return posterior


def ensf_analysis(
    predictive_ensemble: torch.Tensor,
    observation: torch.Tensor,
    observation_operator: ObservationOperator,
    observation_covariance: torch.Tensor,
    config: EnSFConfig | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Official Bao-path EnSF equations under a common tensor interface."""

    config = config or EnSFConfig()
    if config.sampling_time_step_count < 2:
        raise ValueError("EnSF requires at least two path-time points")
    state = torch.randn(
        predictive_ensemble.shape,
        dtype=predictive_ensemble.dtype,
        device=predictive_ensemble.device,
        generator=generator,
    )
    path_time = torch.linspace(
        1.0,
        0.0,
        config.sampling_time_step_count,
        dtype=predictive_ensemble.dtype,
        device=predictive_ensemble.device,
    )
    for now, following in path_time.unfold(0, 2, 1):
        dt = following - now
        alpha = 1.0 - now * (1.0 - config.epsilon_alpha)
        beta = torch.sqrt(config.epsilon_beta + now * (1.0 - config.epsilon_beta))
        dt_log_alpha = -(1.0 - config.epsilon_alpha) / alpha
        means = alpha * predictive_ensemble
        differences = state.unsqueeze(1) - means.unsqueeze(0)
        component_log_density = -0.5 * differences.square().sum(dim=-1) / beta.square()
        responsibilities = torch.softmax(component_log_density, dim=1)
        conditional_scores = -differences / beta.square()
        score = (responsibilities.unsqueeze(-1) * conditional_scores).sum(dim=1)
        observation_score = _observation_log_score(
            state,
            observation,
            observation_operator,
            observation_covariance,
        )
        score = score + (1.0 - now) * observation_score
        score = score.clamp(
            min=-config.max_score_component,
            max=config.max_score_component,
        )
        g_squared = (1.0 - config.epsilon_beta) - 2.0 * dt_log_alpha * beta.square()
        g = torch.sqrt(g_squared.clamp_min(torch.finfo(state.dtype).eps))
        drift = dt_log_alpha * state - g.square() * score
        noise = torch.randn(
            state.shape,
            dtype=state.dtype,
            device=state.device,
            generator=generator,
        )
        state = (state + dt * drift + torch.sqrt(dt.abs()) * g * noise).detach()
    return state


def _ensf_lr_analysis(
    predictive_ensemble: torch.Tensor,
    observation: torch.Tensor,
    observation_operator: SparseObservation,
    observation_covariance: torch.Tensor,
    config: EnSFConfig | None = None,
    generator: torch.Generator | None = None,
    *,
    ridge_stabilized: bool,
) -> torch.Tensor:
    """EnSF observed update followed by covariance regression propagation."""

    indices = observation_operator.indices.to(predictive_ensemble.device)
    observed_forecast = predictive_ensemble.index_select(1, indices)

    def observed_operator(values: torch.Tensor) -> torch.Tensor:
        if observation_operator.transform == "linear":
            return values
        if observation_operator.transform == "atan":
            return torch.atan(values)
        if observation_operator.transform == "square_signed":
            return values.abs() * values
        raise ValueError(observation_operator.transform)

    observed_analysis = ensf_analysis(
        observed_forecast,
        observation,
        observed_operator,
        observation_covariance,
        config,
        generator,
    )
    ensemble_size = predictive_ensemble.shape[0]
    state_anomalies = (
        predictive_ensemble - predictive_ensemble.mean(dim=0)
    ).mT / math.sqrt(ensemble_size - 1)
    observed_anomalies = (
        observed_forecast - observed_forecast.mean(dim=0)
    ).mT / math.sqrt(ensemble_size - 1)
    covariance_xz = state_anomalies @ observed_anomalies.mT
    covariance_zz = observed_anomalies @ observed_anomalies.mT
    if ridge_stabilized:
        diagonal = torch.diagonal(covariance_zz)
        positive_diagonal = diagonal[diagonal > torch.finfo(diagonal.dtype).eps]
        scale = (
            positive_diagonal.median()
            if positive_diagonal.numel()
            else covariance_zz.new_ones(())
        )
        ridge = torch.maximum(
            covariance_zz.new_tensor(1e-6),
            covariance_zz.new_tensor(1e-3) * scale,
        )
        identity = torch.eye(
            covariance_zz.shape[0],
            dtype=covariance_zz.dtype,
            device=covariance_zz.device,
        )
        factor = stable_cholesky(covariance_zz + ridge * identity)
        regression = torch.cholesky_solve(covariance_xz.mT, factor).mT
    else:
        regression = covariance_xz @ torch.linalg.pinv(covariance_zz)
    analysis = predictive_ensemble + (observed_analysis - observed_forecast) @ regression.mT
    analysis[:, indices] = observed_analysis
    return analysis


def ensf_lr_analysis(
    predictive_ensemble: torch.Tensor,
    observation: torch.Tensor,
    observation_operator: SparseObservation,
    observation_covariance: torch.Tensor,
    config: EnSFConfig | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Transparent EnSF-LR v1 reimplementation using the raw generalized inverse."""

    return _ensf_lr_analysis(
        predictive_ensemble,
        observation,
        observation_operator,
        observation_covariance,
        config,
        generator,
        ridge_stabilized=False,
    )


def ensf_lr_ridge_analysis(
    predictive_ensemble: torch.Tensor,
    observation: torch.Tensor,
    observation_operator: SparseObservation,
    observation_covariance: torch.Tensor,
    config: EnSFConfig | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """EnSF-LR with a fixed, scale-relative Tikhonov covariance regularizer."""

    return _ensf_lr_analysis(
        predictive_ensemble,
        observation,
        observation_operator,
        observation_covariance,
        config,
        generator,
        ridge_stabilized=True,
    )


def _f2p_velocity(
    state: torch.Tensor,
    previous_filtering: torch.Tensor,
    predictive_ensemble: torch.Tensor,
    time_value: torch.Tensor,
    sigma_min: float,
) -> torch.Tensor:
    means = (1.0 - time_value) * previous_filtering + time_value * predictive_ensemble
    mean_derivative = predictive_ensemble - previous_filtering
    differences = state.unsqueeze(1) - means.unsqueeze(0)
    component_log_density = -0.5 * differences.square().sum(dim=-1) / sigma_min**2
    responsibilities = torch.softmax(component_log_density, dim=1)
    return responsibilities @ mean_derivative


def enff_f2p_analysis(
    previous_filtering: torch.Tensor,
    predictive_ensemble: torch.Tensor,
    observation: torch.Tensor,
    observation_operator: ObservationOperator,
    observation_covariance: torch.Tensor,
    config: EnFFF2PConfig | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Training-free EnFF filtering-to-predictive flow with local energy guidance."""

    config = config or EnFFF2PConfig()
    if previous_filtering.shape != predictive_ensemble.shape:
        raise ValueError("EnFF-F2P requires matched previous and predictive ensembles")
    if config.sampling_time_step_count < 2:
        raise ValueError("EnFF-F2P requires at least two path-time points")
    if config.independent_coupling:
        permutation = torch.randperm(
            previous_filtering.shape[0],
            device=previous_filtering.device,
            generator=generator,
        )
        previous_filtering = previous_filtering[permutation]
    state = previous_filtering + config.sigma_min * torch.randn(
        predictive_ensemble.shape,
        dtype=predictive_ensemble.dtype,
        device=predictive_ensemble.device,
        generator=generator,
    )
    factor = stable_cholesky(observation_covariance)
    path_time = torch.linspace(
        0.0,
        1.0,
        config.sampling_time_step_count,
        dtype=predictive_ensemble.dtype,
        device=predictive_ensemble.device,
    )
    for now, following in path_time.unfold(0, 2, 1):
        dt = following - now
        velocity = _f2p_velocity(
            state,
            previous_filtering,
            predictive_ensemble,
            now,
            config.sigma_min,
        )
        endpoint = (state + (1.0 - now) * velocity).detach().requires_grad_(True)
        residual = observation_operator(endpoint) - observation.unsqueeze(0)
        whitened = torch.linalg.solve_triangular(
            factor,
            residual.mT,
            upper=False,
        ).mT
        energy = 0.5 * whitened.square().sum(dim=1)
        gradient, = torch.autograd.grad(energy.sum(), endpoint)
        guidance = -config.guidance_lambda * gradient
        state = (state + dt * (velocity + guidance)).detach()
    return state
