from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BootstrapCI:
    estimate: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor


@dataclass(frozen=True)
class PathDispersionDiagonal:
    within: torch.Tensor
    between: torch.Tensor
    total: torch.Tensor

    @property
    def within_trace(self) -> torch.Tensor:
        return self.within.sum()

    @property
    def between_trace(self) -> torch.Tensor:
        return self.between.sum()

    @property
    def total_trace(self) -> torch.Tensor:
        return self.total.sum()


def _require_tensor(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")


def _require_floating_finite(value: torch.Tensor, name: str) -> None:
    _require_tensor(value, name)
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def _validate_ensemble_target(
    ensemble: torch.Tensor,
    target: torch.Tensor,
) -> None:
    _require_floating_finite(ensemble, "ensemble")
    _require_floating_finite(target, "target")
    if ensemble.ndim < 2:
        raise ValueError("ensemble must have shape [..., members, dimensions]")
    expected_target_shape = ensemble.shape[:-2] + ensemble.shape[-1:]
    if target.shape != expected_target_shape:
        raise ValueError(
            f"target shape must be {tuple(expected_target_shape)}, got {tuple(target.shape)}"
        )
    if ensemble.numel() == 0:
        raise ValueError("ensemble and target must be non-empty")
    if ensemble.device != target.device:
        raise ValueError("ensemble and target must be on the same device")


def ensemble_crps(ensemble: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean empirical CRPS over all batch items and state dimensions.

    The empirical distribution uses equal mass for each member. The pairwise
    absolute-difference term is evaluated from sorted samples without forming
    a members-by-members-by-dimensions tensor.
    """

    _validate_ensemble_target(ensemble, target)
    members = ensemble.shape[-2]
    absolute_error = (ensemble - target.unsqueeze(-2)).abs().mean(dim=-2)
    sorted_ensemble = torch.sort(ensemble, dim=-2).values
    coefficients = torch.arange(
        1,
        members + 1,
        dtype=ensemble.dtype,
        device=ensemble.device,
    )
    coefficients = 2.0 * coefficients - members - 1.0
    pair_penalty = (
        sorted_ensemble * coefficients.unsqueeze(-1)
    ).sum(dim=-2) / (members * members)
    return (absolute_error - pair_penalty).mean()


def _normalized_member_weights(
    ensemble: torch.Tensor,
    member_weights: torch.Tensor,
) -> torch.Tensor:
    _require_floating_finite(member_weights, "member_weights")
    expected_shape = ensemble.shape[:-1]
    if member_weights.shape != expected_shape:
        raise ValueError(
            f"member_weights shape must be {tuple(expected_shape)}, got "
            f"{tuple(member_weights.shape)}"
        )
    if member_weights.device != ensemble.device:
        raise ValueError("member_weights and ensemble must be on the same device")
    if bool((member_weights < 0).any()):
        raise ValueError("member_weights must be non-negative")
    totals = member_weights.sum(dim=-1, keepdim=True)
    if bool((totals <= 0).any()):
        raise ValueError("member_weights must have positive mass")
    return member_weights.to(ensemble) / totals.to(ensemble)


def weighted_ensemble_crps(
    ensemble: torch.Tensor,
    target: torch.Tensor,
    member_weights: torch.Tensor,
) -> torch.Tensor:
    """Mean empirical CRPS for a possibly batched weighted ensemble."""

    _validate_ensemble_target(ensemble, target)
    weights = _normalized_member_weights(ensemble, member_weights)
    observation_term = (
        weights.unsqueeze(-1) * (ensemble - target.unsqueeze(-2)).abs()
    ).sum(dim=-2)
    sorted_ensemble, order = torch.sort(ensemble, dim=-2)
    expanded_weights = weights.unsqueeze(-1).expand_as(ensemble)
    sorted_weights = torch.gather(expanded_weights, dim=-2, index=order)
    cumulative = torch.cumsum(sorted_weights, dim=-2)
    coefficients = 2.0 * cumulative - sorted_weights - 1.0
    pair_penalty = (
        sorted_weights * sorted_ensemble * coefficients
    ).sum(dim=-2)
    return (observation_term - pair_penalty).mean()


def multivariate_energy_score(
    ensemble: torch.Tensor,
    target: torch.Tensor,
    *,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Mean multivariate energy score with chunked member distances."""

    _validate_ensemble_target(ensemble, target)
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    members = ensemble.shape[-2]
    observation_term = torch.linalg.vector_norm(
        ensemble - target.unsqueeze(-2),
        dim=-1,
    ).mean(dim=-1)
    pair_sum = torch.zeros(
        ensemble.shape[:-2],
        dtype=ensemble.dtype,
        device=ensemble.device,
    )
    for start in range(0, members, chunk_size):
        stop = min(start + chunk_size, members)
        distances = torch.cdist(
            ensemble[..., start:stop, :],
            ensemble,
            p=2.0,
            compute_mode="donot_use_mm_for_euclid_dist",
        )
        pair_sum = pair_sum + distances.sum(dim=(-2, -1))
    pair_term = pair_sum / (members * members)
    return (observation_term - 0.5 * pair_term).mean()


def weighted_multivariate_energy_score(
    ensemble: torch.Tensor,
    target: torch.Tensor,
    member_weights: torch.Tensor,
    *,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Mean multivariate energy score for a weighted empirical distribution."""

    _validate_ensemble_target(ensemble, target)
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    weights = _normalized_member_weights(ensemble, member_weights)
    members = ensemble.shape[-2]
    observation_term = (
        weights
        * torch.linalg.vector_norm(
            ensemble - target.unsqueeze(-2),
            dim=-1,
        )
    ).sum(dim=-1)
    pair_term = torch.zeros(
        ensemble.shape[:-2],
        dtype=ensemble.dtype,
        device=ensemble.device,
    )
    for start in range(0, members, chunk_size):
        stop = min(start + chunk_size, members)
        distances = torch.cdist(
            ensemble[..., start:stop, :],
            ensemble,
            p=2.0,
            compute_mode="donot_use_mm_for_euclid_dist",
        )
        pair_term = pair_term + (
            distances
            * weights[..., start:stop].unsqueeze(-1)
            * weights.unsqueeze(-2)
        ).sum(dim=(-2, -1))
    return (observation_term - 0.5 * pair_term).mean()


def central_interval_coverage_width(
    ensemble: torch.Tensor,
    target: torch.Tensor,
    *,
    level: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return marginal central-interval coverage and mean interval width."""

    _validate_ensemble_target(ensemble, target)
    if not math.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between zero and one")
    tail = 0.5 * (1.0 - level)
    lower = torch.quantile(ensemble, tail, dim=-2)
    upper = torch.quantile(ensemble, 1.0 - tail, dim=-2)
    coverage = ((target >= lower) & (target <= upper)).to(ensemble.dtype).mean()
    width = (upper - lower).mean()
    return coverage, width


def weighted_central_interval_coverage_width(
    ensemble: torch.Tensor,
    target: torch.Tensor,
    member_weights: torch.Tensor,
    *,
    level: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return marginal weighted empirical coverage and interval width."""

    _validate_ensemble_target(ensemble, target)
    if not math.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between zero and one")
    weights = _normalized_member_weights(ensemble, member_weights)
    sorted_ensemble, order = torch.sort(ensemble, dim=-2)
    expanded_weights = weights.unsqueeze(-1).expand_as(ensemble)
    sorted_weights = torch.gather(expanded_weights, dim=-2, index=order)
    cumulative = torch.cumsum(sorted_weights, dim=-2)
    tail = 0.5 * (1.0 - level)

    def empirical_quantile(probability: float) -> torch.Tensor:
        indices = (cumulative >= probability).to(torch.int64).argmax(dim=-2)
        return torch.gather(
            sorted_ensemble,
            dim=-2,
            index=indices.unsqueeze(-2),
        ).squeeze(-2)

    lower = empirical_quantile(tail)
    upper = empirical_quantile(1.0 - tail)
    coverage = ((target >= lower) & (target <= upper)).to(ensemble.dtype).mean()
    width = (upper - lower).mean()
    return coverage, width


def _rmse(prediction: torch.Tensor, target: torch.Tensor, name: str) -> torch.Tensor:
    _require_floating_finite(prediction, name)
    _require_floating_finite(target, "target")
    if prediction.shape != target.shape:
        raise ValueError(
            f"{name} and target must have the same shape, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if prediction.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    if prediction.device != target.device:
        raise ValueError(f"{name} and target must be on the same device")
    return torch.sqrt(torch.mean((prediction - target).square()))


def state_rmse(state_estimate: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    return _rmse(state_estimate, truth, "state_estimate")


def observation_rmse(
    predicted_observation: torch.Tensor,
    observation: torch.Tensor,
) -> torch.Tensor:
    return _rmse(predicted_observation, observation, "predicted_observation")


def weighted_path_dispersion_diagonal(
    path_ensembles: torch.Tensor,
    path_weights: torch.Tensor,
) -> PathDispersionDiagonal:
    """Decompose population dispersion across weighted epistemic paths.

    ``path_ensembles`` has shape ``[paths, members, dimensions]``. The
    population normalization ``1 / members`` makes the within-plus-between
    identity exact. Only covariance diagonals are formed, so the computation
    remains feasible for high-dimensional PDE states.
    """

    _require_floating_finite(path_ensembles, "path_ensembles")
    _require_floating_finite(path_weights, "path_weights")
    if path_ensembles.ndim != 3 or path_ensembles.numel() == 0:
        raise ValueError(
            "path_ensembles must be non-empty with shape [paths, members, dimensions]"
        )
    if path_weights.shape != path_ensembles.shape[:1]:
        raise ValueError(
            f"path_weights shape must be {(path_ensembles.shape[0],)}, got "
            f"{tuple(path_weights.shape)}"
        )
    if path_weights.device != path_ensembles.device:
        raise ValueError("path_weights and path_ensembles must be on the same device")
    if bool((path_weights < 0).any()):
        raise ValueError("path_weights must be non-negative")
    weight_sum = path_weights.sum()
    if bool(weight_sum <= 0):
        raise ValueError("path_weights must have positive mass")
    weights = path_weights.to(path_ensembles) / weight_sum.to(path_ensembles)

    path_means = path_ensembles.mean(dim=1)
    combined_mean = (weights.unsqueeze(1) * path_means).sum(dim=0)
    within = (
        weights.unsqueeze(1)
        * (path_ensembles - path_means.unsqueeze(1)).square().mean(dim=1)
    ).sum(dim=0)
    between = (
        weights.unsqueeze(1) * (path_means - combined_mean).square()
    ).sum(dim=0)
    return PathDispersionDiagonal(
        within=within,
        between=between,
        total=within + between,
    )


def _validate_paired(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    minimum_count: int,
) -> torch.Tensor:
    _require_floating_finite(first, "first")
    _require_floating_finite(second, "second")
    if first.ndim != 1 or second.ndim != 1:
        raise ValueError("paired samples must be one-dimensional")
    if first.shape != second.shape:
        raise ValueError("paired samples must have identical shapes")
    if first.numel() < minimum_count:
        raise ValueError(f"paired samples must contain at least {minimum_count} values")
    if first.device != second.device:
        raise ValueError("paired samples must be on the same device")
    return first - second.to(first)


def paired_bootstrap_ci(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    generator: torch.Generator | None = None,
    seed: int | None = None,
    chunk_size: int = 1_024,
) -> BootstrapCI:
    """Percentile CI for the paired mean effect ``first - second``."""

    differences = _validate_paired(first, second, minimum_count=1)
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if not isinstance(resamples, int) or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if (generator is None) == (seed is None):
        raise ValueError("Provide exactly one of generator or seed")
    if generator is None:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")
        generator = torch.Generator(device=differences.device)
        generator.manual_seed(seed)
    else:
        if not isinstance(generator, torch.Generator):
            raise TypeError("generator must be a torch.Generator")
        if torch.device(generator.device) != differences.device:
            raise ValueError("generator and paired samples must be on the same device")

    bootstrap_means = torch.empty(
        resamples,
        dtype=differences.dtype,
        device=differences.device,
    )
    sample_count = differences.numel()
    for start in range(0, resamples, chunk_size):
        stop = min(start + chunk_size, resamples)
        indices = torch.randint(
            sample_count,
            (stop - start, sample_count),
            device=differences.device,
            generator=generator,
        )
        bootstrap_means[start:stop] = differences[indices].mean(dim=1)
    tail = 0.5 * (1.0 - confidence)
    bounds = torch.quantile(
        bootstrap_means,
        torch.tensor(
            [tail, 1.0 - tail],
            dtype=differences.dtype,
            device=differences.device,
        ),
    )
    return BootstrapCI(
        estimate=differences.mean(),
        lower=bounds[0],
        upper=bounds[1],
    )


def paired_effect_size(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Return paired Cohen's dz for ``first - second``."""

    differences = _validate_paired(first, second, minimum_count=2)
    standard_deviation = differences.std(unbiased=True)
    if bool(standard_deviation == 0):
        if bool(differences.mean() == 0):
            return differences.new_zeros(())
        return torch.sign(differences.mean()) * differences.new_tensor(float("inf"))
    return differences.mean() / standard_deviation


def _validate_binary(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    require_both_classes: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_floating_finite(scores, "scores")
    _require_tensor(labels, "labels")
    if scores.ndim != 1 or labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("scores and labels must be one-dimensional with identical shapes")
    if scores.numel() == 0:
        raise ValueError("scores and labels must be non-empty")
    if scores.device != labels.device:
        raise ValueError("scores and labels must be on the same device")
    if labels.is_floating_point() and not bool(torch.isfinite(labels).all()):
        raise ValueError("labels must contain only finite values")
    if not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("labels must be binary values in {0, 1}")
    binary_labels = labels.to(dtype=torch.bool)
    if require_both_classes and (
        not bool(binary_labels.any()) or not bool((~binary_labels).any())
    ):
        raise ValueError("both positive and negative labels are required")
    return scores, binary_labels


def binary_auroc(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Tie-aware binary AUROC from the Mann-Whitney rank statistic."""

    scores, labels = _validate_binary(scores, labels, require_both_classes=True)
    sorted_scores, order = torch.sort(scores)
    sorted_labels = labels[order]
    _, inverse, counts = torch.unique_consecutive(
        sorted_scores,
        return_inverse=True,
        return_counts=True,
    )
    ends = torch.cumsum(counts, dim=0).to(scores.dtype)
    starts = ends - counts.to(scores.dtype) + 1.0
    average_ranks = 0.5 * (starts + ends)
    positive_rank_sum = average_ranks[inverse][sorted_labels].sum()
    positives = labels.sum().to(scores.dtype)
    negatives = (~labels).sum().to(scores.dtype)
    return (positive_rank_sum - positives * (positives + 1.0) / 2.0) / (
        positives * negatives
    )


def binary_auprc(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Step-integrated precision-recall area (average precision)."""

    scores, labels = _validate_binary(scores, labels, require_both_classes=True)
    sorted_scores, order = torch.sort(scores, descending=True)
    sorted_labels = labels[order].to(scores.dtype)
    cumulative_true = torch.cumsum(sorted_labels, dim=0)
    cumulative_count = torch.arange(
        1,
        scores.numel() + 1,
        dtype=scores.dtype,
        device=scores.device,
    )
    group_end = torch.ones(scores.numel(), dtype=torch.bool, device=scores.device)
    group_end[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    true_at_threshold = cumulative_true[group_end]
    count_at_threshold = cumulative_count[group_end]
    precision = true_at_threshold / count_at_threshold
    recall = true_at_threshold / labels.sum().to(scores.dtype)
    previous_recall = torch.cat((recall.new_zeros(1), recall[:-1]))
    return torch.sum((recall - previous_recall) * precision)


def binary_false_alarm_rate(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> torch.Tensor:
    scores, labels = _validate_binary(scores, labels, require_both_classes=False)
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    negatives = ~labels
    if not bool(negatives.any()):
        raise ValueError("at least one negative label is required")
    return (scores[negatives] >= threshold).to(scores.dtype).mean()
