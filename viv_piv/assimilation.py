from __future__ import annotations

import dataclasses
import math
from collections import deque
from collections.abc import Callable

import numpy as np
import torch

from pce_assimilation.ensemble_filters import denkf_analysis, letkf_analysis
from pce_assimilation.comparison_filters import IEnSFConfig, iensf_analysis

from .metrics import ReducedMetricAccumulator
from .rom import DMDCCandidate, PODModel


METHODS = ("fixed", "denkf", "letkf", "iensf", "aug_enkf", "bma", "pce", "apce")


@dataclasses.dataclass
class Scenario:
    case_id: str
    time_s: np.ndarray
    control: np.ndarray
    observations: np.ndarray
    sensor_mean: np.ndarray
    sensor_basis: np.ndarray
    evaluation_truth: np.ndarray
    evaluation_mean: np.ndarray
    evaluation_basis: np.ndarray
    observation_noise: np.ndarray
    initial_std: np.ndarray
    observation_covariance: np.ndarray | None = None
    # These controls are carried with each scenario so every saved run can
    # unambiguously identify the uncertainty treatment that generated it.
    initial_ensemble_scale: float = 0.15
    process_noise_scale: float = 1.0
    state_inflation: float = 1.0

    @property
    def steps(self) -> int:
        return int(self.time_s.size)


@dataclasses.dataclass
class CandidateLibrary:
    candidates: list[DMDCCandidate]
    _tensor_cache: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = dataclasses.field(
        default_factory=dict, init=False, repr=False
    )

    @property
    def grid(self) -> np.ndarray:
        return np.asarray([item.reduced_velocity for item in self.candidates], dtype=np.float64)

    @property
    def rank(self) -> int:
        return int(self.candidates[0].a.shape[0])

    def parameters(self, coordinates: np.ndarray, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grid = self.grid
        matrices: list[np.ndarray] = []
        controls: list[np.ndarray] = []
        noises: list[np.ndarray] = []
        for coordinate in np.asarray(coordinates, dtype=float):
            if coordinate <= grid[0]:
                lower = upper = 0
                fraction = 0.0
            elif coordinate >= grid[-1]:
                lower = upper = len(grid) - 1
                fraction = 0.0
            else:
                upper = int(np.searchsorted(grid, coordinate, side="right"))
                lower = upper - 1
                fraction = (coordinate - grid[lower]) / (grid[upper] - grid[lower])
            first, second = self.candidates[lower], self.candidates[upper]
            matrix = (1.0 - fraction) * first.a + fraction * second.a
            radius = float(np.max(np.abs(np.linalg.eigvals(matrix))))
            if radius > 1.02:
                raise ValueError(f"Unstable local candidate at U_r={coordinate:.4f}: spectral radius={radius:.5f}")
            matrices.append(matrix)
            controls.append((1.0 - fraction) * first.b + fraction * second.b)
            noises.append((1.0 - fraction) * first.q_diag + fraction * second.q_diag)
        return (
            torch.as_tensor(np.asarray(matrices), dtype=dtype, device=device),
            torch.as_tensor(np.asarray(controls), dtype=dtype, device=device),
            torch.as_tensor(np.sqrt(np.maximum(np.asarray(noises), 1e-12)), dtype=dtype, device=device),
        )

    def validate_interpolation_stability(self, coordinates: np.ndarray, maximum_radius: float = 1.02) -> float:
        grid = self.grid
        largest_radius = 0.0
        for coordinate in np.asarray(coordinates, dtype=float):
            if coordinate <= grid[0]:
                lower = upper = 0
                fraction = 0.0
            elif coordinate >= grid[-1]:
                lower = upper = len(grid) - 1
                fraction = 0.0
            else:
                upper = int(np.searchsorted(grid, coordinate, side="right"))
                lower = upper - 1
                fraction = (coordinate - grid[lower]) / (grid[upper] - grid[lower])
            first, second = self.candidates[lower], self.candidates[upper]
            matrix = (1.0 - fraction) * first.a + fraction * second.a
            radius = float(np.max(np.abs(np.linalg.eigvals(matrix))))
            largest_radius = max(largest_radius, radius)
            if radius > maximum_radius:
                raise ValueError(
                    f"Unstable interpolated candidate at U_r={coordinate:.4f}: "
                    f"spectral radius={radius:.5f}"
                )
        return largest_radius

    def _stacked_tensors(
        self, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (str(device), str(dtype))
        cached = self._tensor_cache.get(key)
        if cached is None:
            cached = (
                torch.as_tensor(self.grid, dtype=dtype, device=device),
                torch.as_tensor(np.asarray([item.a for item in self.candidates]), dtype=dtype, device=device),
                torch.as_tensor(np.asarray([item.b for item in self.candidates]), dtype=dtype, device=device),
                torch.as_tensor(np.asarray([item.q_diag for item in self.candidates]), dtype=dtype, device=device),
            )
            self._tensor_cache[key] = cached
        return cached

    def parameters_torch(
        self, coordinates: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if coordinates.ndim != 1:
            raise ValueError("Candidate coordinates must be a one-dimensional tensor")
        grid, matrices, controls, noises = self._stacked_tensors(device, dtype)
        values = coordinates.to(device=device, dtype=dtype).clamp(grid[0], grid[-1])
        upper = torch.searchsorted(grid, values, right=True).clamp(1, grid.numel() - 1)
        lower = upper - 1
        fraction = ((values - grid[lower]) / (grid[upper] - grid[lower])).clamp(0.0, 1.0)
        matrix = (1.0 - fraction)[:, None, None] * matrices[lower] + fraction[:, None, None] * matrices[upper]
        control = (1.0 - fraction)[:, None, None] * controls[lower] + fraction[:, None, None] * controls[upper]
        noise = (1.0 - fraction)[:, None] * noises[lower] + fraction[:, None] * noises[upper]
        return matrix, control, torch.sqrt(noise.clamp_min(1e-12))


def stable_cholesky(matrix: torch.Tensor) -> torch.Tensor:
    identity = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    scale = torch.diagonal(matrix).abs().mean().clamp_min(1.0)
    for exponent in range(8):
        factor, info = torch.linalg.cholesky_ex(matrix + identity * scale * (10.0**exponent) * 1e-10)
        if int(info.max()) == 0:
            return factor
    raise RuntimeError("Observation covariance is not positive definite after jitter")


def stable_ensemble_cholesky(matrix: torch.Tensor) -> torch.Tensor:
    """Factor an ensemble-space system after removing round-off asymmetry.

    With very sparse observations, the whitened anomaly Gram matrix is low
    rank. It is positive semidefinite in exact arithmetic, but batched GPU
    matmul can leave tiny antisymmetric and negative-eigenvalue round-off that
    makes a direct Cholesky fail. The update equations are unchanged; this
    only applies a symmetric projection and a scale-aware numerical jitter.
    """
    # Preserve the original factorization bit-for-bit whenever it already
    # succeeds.  The stabilization path is only needed for the nearly
    # positive-semidefinite sparse-observation case.
    try:
        return torch.linalg.cholesky(matrix)
    except RuntimeError:
        pass
    symmetric = 0.5 * (matrix + matrix.mT)
    identity = torch.eye(symmetric.shape[-1], dtype=symmetric.dtype, device=symmetric.device)
    scale = torch.diagonal(symmetric, dim1=-2, dim2=-1).abs().mean(dim=-1, keepdim=True).clamp_min(1.0)
    for exponent in range(8):
        jitter = identity * scale[..., None] * (10.0**exponent) * 1e-10
        factor, info = torch.linalg.cholesky_ex(symmetric + jitter)
        if int(info.max()) == 0:
            return factor
    raise RuntimeError("Ensemble-space covariance is not positive definite after jitter")


def _finite_guard(name: str, value: torch.Tensor, step: int) -> None:
    """Fail at the first non-finite state and retain a useful numerical diagnosis."""
    if not torch.isfinite(value).all():
        finite = value[torch.isfinite(value)]
        if finite.numel():
            lo = float(finite.min())
            hi = float(finite.max())
        else:
            lo = hi = float("nan")
        raise FloatingPointError(
            f"non-finite {name} at step={step}; finite_range=[{lo:.6e}, {hi:.6e}]"
        )


def entropy(weights: torch.Tensor) -> torch.Tensor:
    safe = weights.clamp_min(1e-300)
    return -torch.sum(safe * safe.log())


def entropy_project(weights: torch.Tensor, target: float) -> torch.Tensor:
    if float(entropy(weights)) >= target:
        return weights
    uniform = torch.full_like(weights, 1.0 / weights.numel())
    low, high = 0.0, 1.0
    for _ in range(40):
        middle = 0.5 * (low + high)
        candidate = (1.0 - middle) * weights + middle * uniform
        if float(entropy(candidate)) < target:
            low = middle
        else:
            high = middle
    result = (1.0 - high) * weights + high * uniform
    return result / result.sum().clamp_min(1e-30)


def inflate_ensemble(ensemble: torch.Tensor, factor: float) -> torch.Tensor:
    if factor <= 0.0:
        raise ValueError(f"state inflation must be positive, received {factor}")
    if abs(factor - 1.0) < 1e-12:
        return ensemble
    mean = ensemble.mean(dim=-2, keepdim=True)
    return mean + factor * (ensemble - mean)


def gaussian_score(
    ensemble: torch.Tensor,
    observation: torch.Tensor,
    covariance: torch.Tensor,
    shrinkage: float,
    dimension_weights: torch.Tensor | None,
) -> torch.Tensor:
    mean = ensemble.mean(dim=0)
    anomalies = ensemble - mean
    empirical = anomalies.mT @ anomalies / max(ensemble.shape[0] - 1, 1)
    empirical = (1.0 - shrinkage) * empirical + shrinkage * torch.diag(torch.diagonal(empirical))
    total = empirical + covariance
    residual = observation - mean
    if dimension_weights is not None:
        variances = torch.diagonal(total).clamp_min(1e-12)
        weights = dimension_weights * observation.numel() / dimension_weights.sum().clamp_min(1e-30)
        terms = residual.square() / variances + variances.log() + math.log(2.0 * math.pi)
        return -0.5 * torch.sum(weights * terms) / observation.numel()
    factor = stable_cholesky(total)
    solution = torch.cholesky_solve(residual[:, None], factor).squeeze(-1)
    log_det = 2.0 * torch.log(torch.diagonal(factor)).sum()
    return -0.5 * (residual @ solution + log_det + observation.numel() * math.log(2.0 * math.pi)) / observation.numel()


def observation_patches(observation_dim: int, config: dict[str, object]) -> list[torch.Tensor]:
    if observation_dim % 2:
        return [torch.arange(observation_dim, dtype=torch.int64)]
    points = observation_dim // 2
    patch_points = max(int(config.get("evidence_patch_points", 10)), 1)
    columns = np.array_split(np.arange(points), math.ceil(points / patch_points))
    return [torch.as_tensor(np.ravel(np.column_stack([2 * part, 2 * part + 1])), dtype=torch.int64) for part in columns]


def evidence_scores(
    branch_observations: torch.Tensor,
    observation: torch.Tensor,
    covariance: torch.Tensor,
    config: dict[str, object],
    method: str,
) -> torch.Tensor:
    patches = observation_patches(observation.numel(), config)
    if observation.numel() > int(config.get("diagonal_evidence_above_dimensions", 200)):
        if bool(config.get("_full_observation_covariance", False)):
            branch_count, ensemble_size, _ = branch_observations.shape
            grouped: dict[int, list[torch.Tensor]] = {}
            for patch in patches:
                grouped.setdefault(int(patch.numel()), []).append(patch)
            score_sum = torch.zeros(branch_count, dtype=branch_observations.dtype, device=branch_observations.device)
            patch_total = 0
            for patch_dim, same_size_patches in grouped.items():
                patch_indices = torch.stack(same_size_patches).to(branch_observations.device)
                patch_count = int(patch_indices.shape[0])
                selected = branch_observations.index_select(2, patch_indices.reshape(-1)).reshape(
                    branch_count, ensemble_size, patch_count, patch_dim
                )
                means = selected.mean(dim=1)
                anomalies = selected - means[:, None, :, :]
                empirical = torch.einsum("jnpd,jnpe->jpde", anomalies, anomalies) / max(ensemble_size - 1, 1)
                shrinkage = float(config["evidence_shrinkage"])
                empirical = (1.0 - shrinkage) * empirical + shrinkage * torch.diag_embed(
                    torch.diagonal(empirical, dim1=-2, dim2=-1)
                )
                covariance_blocks = covariance[
                    patch_indices[:, :, None], patch_indices[:, None, :]
                ]
                total = empirical + covariance_blocks[None, :, :, :]
                factors = stable_cholesky(total)
                residuals = observation.index_select(0, patch_indices.reshape(-1)).reshape(
                    patch_count, patch_dim
                )[None, :, :] - means
                solved = torch.cholesky_solve(residuals[..., None], factors).squeeze(-1)
                quadratic = torch.sum(residuals * solved, dim=-1)
                log_det = 2.0 * torch.log(torch.diagonal(factors, dim1=-2, dim2=-1)).sum(dim=-1)
                base = -0.5 * (quadratic + log_det + patch_dim * math.log(2.0 * math.pi)) / patch_dim
                if method == "apce":
                    between = means.var(dim=0, unbiased=True)
                    dimension_weights = float(config["dimension_weight_floor"]) + float(config["dimension_weight_gain"]) * (
                        between / between.amax(dim=-1, keepdim=True).clamp_min(1e-12)
                    )
                    dimension_weights = dimension_weights * patch_dim / dimension_weights.sum(dim=-1, keepdim=True).clamp_min(1e-30)
                    variances = torch.diagonal(total, dim1=-2, dim2=-1).clamp_min(1e-12)
                    marginal_terms = residuals.square() / variances + variances.log() + math.log(2.0 * math.pi)
                    weighted_marginal = -0.5 * torch.mean(
                        marginal_terms * dimension_weights[None, :, :], dim=-1
                    )
                    unweighted_marginal = -0.5 * torch.mean(marginal_terms, dim=-1)
                    base = base + weighted_marginal - unweighted_marginal
                score_sum += base.sum(dim=1)
                patch_total += patch_count
            return score_sum / max(patch_total, 1)
        branch_means = branch_observations.mean(dim=1)
        branch_variances = branch_observations.var(dim=1, unbiased=True) + torch.diagonal(covariance)[None, :]
        patch_scores: list[torch.Tensor] = []
        for patch_cpu in patches:
            patch = patch_cpu.to(branch_observations.device)
            means = branch_means[:, patch]
            variances = branch_variances[:, patch].clamp_min(1e-12)
            residuals = observation[patch][None, :] - means
            if method == "apce":
                between = means.var(dim=0, unbiased=True)
                dimension_weights = float(config["dimension_weight_floor"]) + float(config["dimension_weight_gain"]) * between / between.max().clamp_min(1e-12)
                dimension_weights = dimension_weights * patch.numel() / dimension_weights.sum().clamp_min(1e-30)
            else:
                dimension_weights = torch.ones(patch.numel(), dtype=means.dtype, device=means.device)
            terms = residuals.square() / variances + variances.log() + math.log(2.0 * math.pi)
            patch_scores.append(-0.5 * torch.mean(terms * dimension_weights[None, :], dim=1))
        return torch.stack(patch_scores, dim=0).mean(dim=0)
    scores: list[torch.Tensor] = []
    for branch in range(branch_observations.shape[0]):
        terms: list[torch.Tensor] = []
        for patch_cpu in patches:
            patch = patch_cpu.to(branch_observations.device)
            dimension_weights = None
            if method == "apce":
                between = branch_observations[:, :, patch].mean(dim=1).var(dim=0, unbiased=True)
                dimension_weights = float(config["dimension_weight_floor"]) + float(config["dimension_weight_gain"]) * between / between.max().clamp_min(1e-12)
            terms.append(
                gaussian_score(
                    branch_observations[branch, :, patch],
                    observation[patch],
                    covariance.index_select(0, patch).index_select(1, patch),
                    float(config["evidence_shrinkage"]),
                    dimension_weights,
                )
            )
        scores.append(torch.stack(terms).mean())
    return torch.stack(scores)


def _observation_operator(matrix: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    return lambda states: states @ matrix.mT


def initial_ensemble(scenario: Scenario, ensemble_size: int, generator: torch.Generator, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    matrix = torch.as_tensor(scenario.sensor_basis, dtype=dtype, device=device)
    observation = torch.as_tensor(scenario.observations[0] - scenario.sensor_mean, dtype=dtype, device=device)
    ridge = 1e-5 * torch.trace(matrix @ matrix.mT) / max(matrix.shape[0], 1)
    if matrix.shape[0] > matrix.shape[1]:
        centre = torch.linalg.solve(
            matrix.mT @ matrix + ridge * torch.eye(matrix.shape[1], dtype=dtype, device=device),
            matrix.mT @ observation,
        )
    else:
        solve = torch.linalg.solve(matrix @ matrix.mT + ridge * torch.eye(matrix.shape[0], dtype=dtype, device=device), observation)
        centre = matrix.mT @ solve
    spread = torch.as_tensor(scenario.initial_std, dtype=dtype, device=device).clamp_min(1e-8)
    noise = torch.randn((ensemble_size, matrix.shape[1]), dtype=dtype, device=device, generator=generator)
    return centre[None, :] + float(scenario.initial_ensemble_scale) * noise * spread[None, :]


def denkf_analysis_diagonal_ensemble_space(
    state_ensemble: torch.Tensor,
    observation: torch.Tensor,
    observation_operator: Callable[[torch.Tensor], torch.Tensor],
    covariance: torch.Tensor,
) -> torch.Tensor:
    """Exact DEnKF update using an ensemble-size solve for diagonal R."""
    predicted = observation_operator(state_ensemble)
    ensemble_size = state_ensemble.shape[0]
    state_mean = state_ensemble.mean(dim=0)
    observation_mean = predicted.mean(dim=0)
    state_anomalies = state_ensemble - state_mean
    observation_anomalies = predicted - observation_mean
    noise_variance = torch.diagonal(covariance).clamp_min(1e-12)
    innovation_diagonal = observation_anomalies.square().sum(dim=0) / max(ensemble_size - 1, 1) + noise_variance
    # Match pce_assimilation.math_utils.stable_cholesky, which always adds this floor.
    noise_variance = noise_variance + 1e-6 * innovation_diagonal.abs().median().clamp_min(
        torch.finfo(state_ensemble.dtype).eps
    )
    noise_std = torch.sqrt(noise_variance)
    whitened_anomalies = observation_anomalies / noise_std[None, :]
    whitened_innovation = (observation - observation_mean) / noise_std
    gram = whitened_anomalies @ whitened_anomalies.mT
    system = gram + (ensemble_size - 1) * torch.eye(
        ensemble_size, dtype=state_ensemble.dtype, device=state_ensemble.device
    )
    factor = stable_ensemble_cholesky(system)
    mean_weights = torch.cholesky_solve(
        (whitened_anomalies @ whitened_innovation)[:, None], factor
    ).squeeze(-1)
    updated_mean = state_mean + state_anomalies.mT @ mean_weights
    transformed_state = torch.cholesky_solve(state_anomalies, factor)
    updated_anomalies = state_anomalies - 0.5 * (gram @ transformed_state)
    return updated_mean[None, :] + updated_anomalies


def denkf_analysis_general_ensemble_space_batch(
    state_ensembles: torch.Tensor,
    observation: torch.Tensor,
    observation_operator: Callable[[torch.Tensor], torch.Tensor],
    covariance_factor: torch.Tensor,
) -> torch.Tensor:
    """Batched DEnKF update for branches sharing one dense observation covariance."""
    predicted = torch.stack([observation_operator(branch) for branch in state_ensembles])
    branch_count, ensemble_size, observation_dim = predicted.shape
    state_means = state_ensembles.mean(dim=1)
    observation_means = predicted.mean(dim=1)
    state_anomalies = state_ensembles - state_means[:, None, :]
    observation_anomalies = predicted - observation_means[:, None, :]
    whitened = torch.linalg.solve_triangular(
        covariance_factor,
        observation_anomalies.permute(2, 0, 1).reshape(observation_dim, branch_count * ensemble_size),
        upper=False,
    ).reshape(observation_dim, branch_count, ensemble_size).permute(1, 2, 0)
    innovations = observation[None, :] - observation_means
    whitened_innovations = torch.linalg.solve_triangular(
        covariance_factor, innovations.mT, upper=False
    ).mT
    gram = whitened @ whitened.mT
    system = gram + (ensemble_size - 1) * torch.eye(
        ensemble_size, dtype=state_ensembles.dtype, device=state_ensembles.device
    )[None, :, :]
    factor = stable_ensemble_cholesky(system)
    mean_weights = torch.cholesky_solve(
        (whitened @ whitened_innovations[..., None]), factor
    ).squeeze(-1)
    updated_means = state_means + torch.einsum("jni,jn->ji", state_anomalies, mean_weights)
    transformed_states = torch.cholesky_solve(state_anomalies, factor)
    updated_anomalies = state_anomalies - 0.5 * (gram @ transformed_states)
    return updated_means[:, None, :] + updated_anomalies


def scalable_denkf_analysis(
    state_ensemble: torch.Tensor,
    observation: torch.Tensor,
    observation_operator: Callable[[torch.Tensor], torch.Tensor],
    covariance: torch.Tensor,
    config: dict[str, object],
    covariance_factor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Use the equivalent ensemble-space DEnKF update for wide observations."""
    if covariance_factor is not None:
        return denkf_analysis_general_ensemble_space_batch(
            state_ensemble[None, :, :], observation, observation_operator, covariance_factor
        )[0]
    threshold = int(config.get("diagonal_evidence_above_dimensions", 200))
    if observation.numel() > threshold:
        return denkf_analysis_diagonal_ensemble_space(
            state_ensemble, observation, observation_operator, covariance
        )
    return denkf_analysis(state_ensemble, observation, observation_operator, covariance)


def propagate(
    states: torch.Tensor,
    matrices: torch.Tensor,
    controls: torch.Tensor,
    q_sqrt: torch.Tensor,
    input_value: torch.Tensor,
    standard_noise: torch.Tensor,
) -> torch.Tensor:
    forecast = torch.matmul(states, matrices.transpose(-1, -2))
    forcing = torch.einsum("jrc,c->jr", controls, input_value).unsqueeze(1)
    return forecast + forcing + standard_noise.unsqueeze(0) * q_sqrt.unsqueeze(1)


def build_local_grid(grid: np.ndarray, scores: np.ndarray, config: dict[str, object]) -> np.ndarray:
    topk = min(int(config["local_grid_topk"]), grid.size)
    selected = np.sort(np.argsort(scores)[-topk:])
    lower = max(int(selected[0]) - 1, 0)
    upper = min(int(selected[-1]) + 1, grid.size - 1)
    points = int(config["local_grid_points"])
    local = np.linspace(grid[lower], grid[upper], points, dtype=np.float64)
    local = np.unique(np.concatenate([local, grid[selected]]))
    spacing = float(config["local_grid_min_spacing"])
    result = [float(local[0])]
    for value in local[1:]:
        if value - result[-1] >= spacing - 1e-12:
            result.append(float(value))
    return np.asarray(result, dtype=np.float64)


@dataclasses.dataclass
class PassResult:
    method: str
    grid: np.ndarray
    final_weights: np.ndarray
    final_scores: np.ndarray
    latent_estimate: np.ndarray
    metrics: dict[str, float]
    trace: dict[str, np.ndarray]
    blackout_states: dict[int, dict[str, np.ndarray]]


def run_pass(
    scenario: Scenario,
    library: CandidateLibrary,
    config: dict[str, object],
    method: str,
    seed: int,
    device: torch.device,
    *,
    grid: np.ndarray | None = None,
    record_trace: bool = False,
    blackout_origins: set[int] | None = None,
) -> PassResult:
    if method not in METHODS:
        raise ValueError(method)
    for name, value in (
        ("initial_ensemble_scale", scenario.initial_ensemble_scale),
        ("process_noise_scale", scenario.process_noise_scale),
        ("state_inflation", scenario.state_inflation),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive, received {value}")
    dtype = torch.float64
    seed = int(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    ensemble_size = int(config["ensemble_size"])
    candidate_grid = library.grid if grid is None else np.asarray(grid, dtype=np.float64)
    if method in {"fixed", "denkf", "letkf", "iensf"}:
        centre = float(np.median(library.grid))
        candidate_grid = np.asarray([library.grid[int(np.argmin(np.abs(library.grid - centre)))]], dtype=np.float64)
    if method == "aug_enkf":
        # Aug-EnKF is one augmented ensemble, not one ensemble per candidate.
        # Its member coordinates span the candidate range and evolve jointly
        # with the state under the same total ensemble budget as other filters.
        stability_points = int(
            config.get("aug_enkf_stability_grid_points", max(101, 20 * (library.grid.size - 1) + 1))
        )
        if stability_points < 2:
            raise ValueError("aug_enkf_stability_grid_points must be at least two")
        library.validate_interpolation_stability(
            np.linspace(float(library.grid[0]), float(library.grid[-1]), stability_points)
        )
        member_coordinates_np = np.linspace(
            float(library.grid[0]), float(library.grid[-1]), ensemble_size, dtype=np.float64
        )
        candidate_grid = np.asarray([float(np.mean(member_coordinates_np))], dtype=np.float64)
        aug_coordinate: torch.Tensor | None = torch.as_tensor(
            member_coordinates_np, dtype=dtype, device=device
        )
        matrices, control_matrices, q_sqrt = library.parameters_torch(aug_coordinate, device, dtype)
    else:
        matrices, control_matrices, q_sqrt = library.parameters(candidate_grid, device, dtype)
        aug_coordinate = None
    q_sqrt = q_sqrt * float(scenario.process_noise_scale)
    sensor_basis = torch.as_tensor(scenario.sensor_basis, dtype=dtype, device=device)
    evaluation_basis = torch.as_tensor(scenario.evaluation_basis, dtype=dtype, device=device)
    observation_mean = torch.as_tensor(scenario.sensor_mean, dtype=dtype, device=device)
    evaluation_mean = torch.as_tensor(scenario.evaluation_mean, dtype=dtype, device=device)
    observations = torch.as_tensor(scenario.observations, dtype=dtype, device=device) - observation_mean[None, :]
    evaluation_truth = torch.as_tensor(scenario.evaluation_truth, dtype=dtype, device=device)
    inputs = torch.as_tensor(scenario.control, dtype=dtype, device=device)
    covariance = torch.diag(torch.as_tensor(scenario.observation_noise**2, dtype=dtype, device=device))
    covariance_factor = None
    if scenario.observation_covariance is not None:
        covariance = torch.as_tensor(scenario.observation_covariance, dtype=dtype, device=device)
        covariance_factor = torch.linalg.cholesky(covariance)
    evidence_config = dict(config)
    evidence_config["_full_observation_covariance"] = covariance_factor is not None
    initial = initial_ensemble(scenario, ensemble_size, generator, device, dtype)
    branches = (
        initial.unsqueeze(0)
        if method == "aug_enkf"
        else initial.unsqueeze(0).repeat(candidate_grid.size, 1, 1)
    )
    shadows = branches.clone()
    log_scores = torch.zeros(candidate_grid.size, dtype=dtype, device=device)
    weights = torch.full((candidate_grid.size,), 1.0 / candidate_grid.size, dtype=dtype, device=device)
    metric = ReducedMetricAccumulator(float(np.std(scenario.evaluation_truth)))
    estimates = np.empty((scenario.steps, library.rank), dtype=np.float64)
    trace: dict[str, list[np.ndarray] | list[float] | list[int]] = {
        "weights": [], "scores": [], "instant_scores": [], "entropy": [], "separation": [], "erasure": [],
        "normalized_crps": [], "crps_step": [], "aug_coordinate_mean": [], "aug_coordinate_std": [],
    }
    blackout_states = {} if blackout_origins is None else {}
    warmup = int(round(float(config["warmup_seconds"]) / float(config["time_step_s"])))
    metric_stride = int(config["probabilistic_metric_stride"])
    pce_temperature = float(config["pce_temperature"]) * float(config["time_step_s"]) / float(config["reference_evidence_interval_s"])
    apce_temperature = float(config["apce_temperature"]) * float(config["time_step_s"]) / float(config["reference_evidence_interval_s"])
    apce_min_temperature = float(config["apce_min_temperature"]) * float(config["time_step_s"]) / float(config["reference_evidence_interval_s"])
    apce_forgetting = float(config["apce_forgetting"]) ** (float(config["time_step_s"]) / float(config["reference_evidence_interval_s"]))
    evidence_window_frames = int(config.get("evidence_window_frames", 1))
    if evidence_window_frames < 1:
        raise ValueError("evidence_window_frames must be at least one")
    score_history: deque[torch.Tensor] = deque(maxlen=evidence_window_frames)
    operator = _observation_operator(sensor_basis)
    for step in range(scenario.steps):
        branch_means = branches.mean(dim=1)
        state_weights = weights
        estimate = torch.sum(state_weights[:, None] * branch_means, dim=0)
        estimates[step] = estimate.detach().cpu().numpy()
        eval_estimate = evaluation_mean + estimate @ evaluation_basis.mT
        metric.add_point(eval_estimate, evaluation_truth[step])
        if step % metric_stride == 0:
            support = branches.reshape(-1, library.rank)
            member_weights = state_weights[:, None].expand(-1, ensemble_size).reshape(-1) / ensemble_size
            eval_support = evaluation_mean[None, :] + support @ evaluation_basis.mT
            metric.add_distribution(eval_support, evaluation_truth[step], member_weights)
            trace["normalized_crps"].append(metric.crps[-1])
            trace["crps_step"].append(step)
        if method == "aug_enkf":
            if aug_coordinate is None:
                raise RuntimeError("Aug-EnKF member coordinates were not initialized")
            trace["aug_coordinate_mean"].append(float(aug_coordinate.mean()))
            trace["aug_coordinate_std"].append(float(aug_coordinate.std(unbiased=True)))
        if blackout_origins is not None and step in blackout_origins:
            blackout_states[step] = {
                "branches": branches.detach().cpu().numpy(),
                "weights": weights.detach().cpu().numpy(),
                "grid": candidate_grid.copy(),
            }
            if method == "aug_enkf":
                if aug_coordinate is None:
                    raise RuntimeError("Aug-EnKF member coordinates were not initialized")
                blackout_states[step]["member_coordinates"] = aug_coordinate.detach().cpu().numpy()
        if step == scenario.steps - 1:
            break
        standard_noise = torch.randn((ensemble_size, library.rank), dtype=dtype, device=device, generator=generator)
        if method == "aug_enkf":
            member_states = branches[0]
            member_states = (
                torch.bmm(member_states.unsqueeze(1), matrices.transpose(-1, -2)).squeeze(1)
                + torch.einsum("nrc,c->nr", control_matrices, inputs[step])
                + standard_noise * q_sqrt
            )
            branches = member_states.unsqueeze(0)
        else:
            branches = propagate(branches, matrices, control_matrices, q_sqrt, inputs[step], standard_noise)
            shadows = propagate(shadows, matrices, control_matrices, q_sqrt, inputs[step], standard_noise)
        if method == "fixed":
            pass
        elif method in {"denkf", "letkf", "iensf"}:
            if method == "denkf":
                branches[0] = scalable_denkf_analysis(
                    branches[0], observations[step + 1], operator, covariance, config, covariance_factor
                )
            elif method == "letkf":
                branches[0] = letkf_analysis(branches[0], observations[step + 1], operator, covariance)
            else:
                score_generator = torch.Generator(device=device).manual_seed(seed + 100000 + step)
                branches[0] = iensf_analysis(
                    branches[0], observations[step + 1], operator, covariance,
                    IEnSFConfig(sampling_time_step_count=10, refinement_iterations=1, endpoint_epsilon=1e-3), score_generator,
                )
            branches[0] = inflate_ensemble(branches[0], float(scenario.state_inflation))
        elif method == "aug_enkf":
            if aug_coordinate is None:
                raise RuntimeError("Aug-EnKF member coordinates were not initialized")
            augmented = torch.cat([branches[0], aug_coordinate[:, None]], dim=1)
            augmented_operator = lambda values: values[:, :-1] @ sensor_basis.mT
            updated = scalable_denkf_analysis(
                augmented, observations[step + 1], augmented_operator, covariance, config, covariance_factor
            )
            updated_states = inflate_ensemble(updated[:, :-1], float(scenario.state_inflation))
            branches = updated_states.unsqueeze(0)
            aug_coordinate = updated[:, -1].clamp(
                float(library.grid[0]), float(library.grid[-1])
            )
            candidate_grid = np.asarray([float(aug_coordinate.mean())], dtype=np.float64)
            matrices, control_matrices, q_sqrt = library.parameters_torch(aug_coordinate, device, dtype)
            q_sqrt = q_sqrt * float(scenario.process_noise_scale)
            weights = torch.ones(1, dtype=dtype, device=device)
        else:
            evidence_source = shadows if method in {"pce", "apce"} else branches
            predicted_shadow = torch.stack([operator(evidence_source[index]) for index in range(candidate_grid.size)])
            _finite_guard("predicted_shadow", predicted_shadow, step + 1)
            before = torch.stack([operator(branches[index]).mean(dim=0) for index in range(candidate_grid.size)])
            evidence_method = "apce" if method == "apce" else "pce"
            scores = evidence_scores(predicted_shadow, observations[step + 1], covariance, evidence_config, evidence_method)
            _finite_guard("evidence_scores", scores, step + 1)
            instantaneous_centered = scores - scores.mean()
            _finite_guard("centered_evidence_scores", instantaneous_centered, step + 1)
            score_history.append(instantaneous_centered)
            centered = torch.stack(tuple(score_history), dim=0).mean(dim=0)
            _finite_guard("windowed_evidence_scores", centered, step + 1)
            if step + 1 >= warmup:
                if method == "apce":
                    entropy_ratio = float(entropy(weights) / math.log(max(weights.numel(), 2)))
                    temperature = max(apce_min_temperature, min(apce_temperature, apce_temperature * entropy_ratio**0.75))
                    log_scores = apce_forgetting * log_scores + temperature * centered
                    _finite_guard("apce_log_scores_pre_shift", log_scores, step + 1)
                    # Subtracting the maximum leaves softmax weights unchanged,
                    # but prevents a long run from overflowing the cumulative log score.
                    log_scores = log_scores - torch.max(log_scores)
                    weights = entropy_project(torch.softmax(log_scores, dim=0), float(config["apce_entropy_floor"]))
                else:
                    log_scores = log_scores + pce_temperature * centered
                    _finite_guard("pce_log_scores_pre_shift", log_scores, step + 1)
                    log_scores = log_scores - torch.max(log_scores)
                    weights = torch.softmax(log_scores, dim=0)
                _finite_guard("candidate_weights", weights, step + 1)
            if covariance_factor is not None:
                branches = denkf_analysis_general_ensemble_space_batch(
                    branches, observations[step + 1], operator, covariance_factor
                )
            else:
                for index in range(candidate_grid.size):
                    branches[index] = scalable_denkf_analysis(
                        branches[index], observations[step + 1], operator, covariance, config
                    )
            branches = inflate_ensemble(branches, float(scenario.state_inflation))
            _finite_guard("analysis_branches", branches, step + 1)
            after = torch.stack([operator(branches[index]).mean(dim=0) for index in range(candidate_grid.size)])
            _finite_guard("analysis_observation_means", after, step + 1)
            between = torch.mean(torch.var(predicted_shadow.mean(dim=1), dim=0, unbiased=True)) if candidate_grid.size > 1 else torch.tensor(0.0, device=device)
            sampling = torch.mean(torch.var(predicted_shadow, dim=1, unbiased=True)) / ensemble_size if candidate_grid.size > 1 else torch.tensor(1.0, device=device)
            trace["instant_scores"].append(scores.detach().cpu().numpy())
            trace["scores"].append(centered.detach().cpu().numpy())
            trace["weights"].append(weights.detach().cpu().numpy())
            trace["entropy"].append(float(entropy(weights)))
            trace["separation"].append(float(between / sampling.clamp_min(1e-12)))
            trace["erasure"].append(float(torch.mean(torch.var(after, dim=0, unbiased=True)) / torch.mean(torch.var(before, dim=0, unbiased=True)).clamp_min(1e-12)))
            if not all(math.isfinite(float(value)) for value in (trace["entropy"][-1], trace["separation"][-1], trace["erasure"][-1])):
                raise FloatingPointError(
                    f"non-finite diagnostic trace at step={step + 1}; "
                    f"entropy={trace['entropy'][-1]}, separation={trace['separation'][-1]}, erasure={trace['erasure'][-1]}"
                )
    metrics = metric.finalize()
    trace_array = {key: np.asarray(value) for key, value in trace.items()}
    if not all(np.isfinite(value).all() for value in trace_array.values() if value.size):
        raise FloatingPointError(f"{method} emitted non-finite trace values")
    return PassResult(
        method=method,
        grid=candidate_grid.copy(),
        final_weights=weights.detach().cpu().numpy(),
        final_scores=log_scores.detach().cpu().numpy(),
        latent_estimate=estimates,
        metrics=metrics,
        trace=trace_array if record_trace else {},
        blackout_states=blackout_states,
    )


@dataclasses.dataclass
class TwoPassResult:
    coarse: PassResult
    local: PassResult
    local_stable: bool
    local_failure: str


def run_two_pass(
    scenario: Scenario,
    library: CandidateLibrary,
    config: dict[str, object],
    method: str,
    seed: int,
    device: torch.device,
    *,
    record_trace: bool,
    blackout_origins: set[int],
) -> TwoPassResult:
    coarse = run_pass(scenario, library, config, method, seed, device, record_trace=False)
    local_grid = build_local_grid(library.grid, coarse.final_scores, config)
    try:
        local = run_pass(
            scenario, library, config, method, seed, device, grid=local_grid,
            record_trace=record_trace, blackout_origins=blackout_origins,
        )
        return TwoPassResult(coarse=coarse, local=local, local_stable=True, local_failure="")
    except ValueError as exc:
        fallback = run_pass(
            scenario, library, config, method, seed, device, grid=library.grid,
            record_trace=record_trace, blackout_origins=blackout_origins,
        )
        return TwoPassResult(coarse=coarse, local=fallback, local_stable=False, local_failure=str(exc))
