from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .evidence import AlphaEvidenceTracker
from .config import AssimilationConfig
from .low_rank import localized_low_rank_map, propagate_observation_increment
from .math_utils import stable_cholesky
from .observation_flow import analytic_posterior_mixture, posterior_probability_flow

ObservationOperator = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class AnalysisDiagnostics:
    back_projection_iterations: int
    initial_innovation: float
    final_innovation: float
    retained_rank: int
    log_evidence_per_observation: float
    diverged: bool


@dataclass(frozen=True)
class MultiPathAnalysis:
    ensembles: torch.Tensor
    evidence_weights: torch.Tensor
    state_estimate: torch.Tensor
    alpha_estimate: float
    diagnostics: tuple[AnalysisDiagnostics, ...]


def _whitened_norm(residual: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
    factor = stable_cholesky(covariance)
    whitened = torch.linalg.solve_triangular(
        factor,
        residual.unsqueeze(-1),
        upper=False,
    ).squeeze(-1)
    return torch.linalg.vector_norm(whitened)


class PCEFilter:
    def __init__(self, config: AssimilationConfig | None = None) -> None:
        self.config = config or AssimilationConfig()

    def analyze_local(
        self,
        state_ensemble: torch.Tensor,
        observation: torch.Tensor,
        observation_operator: ObservationOperator,
        observation_covariance: torch.Tensor,
        localization: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, AnalysisDiagnostics]:
        current = state_ensemble.clone()
        forecast_observations = observation_operator(current)
        if forecast_observations.shape[1] > self.config.flow.max_patch_observations:
            raise ValueError(
                "Local observation dimension exceeds max_patch_observations; "
                "provide an explicit patch decomposition"
            )
        initial_residual = forecast_observations.mean(dim=0) - observation
        initial_innovation = _whitened_norm(
            initial_residual, observation_covariance
        ).clamp_min(1.0)
        mixture = analytic_posterior_mixture(
            forecast_observations,
            observation,
            observation_covariance,
            relative_floor=self.config.flow.eigenvalue_floor,
        )
        log_evidence = mixture.log_evidence / observation.numel()
        target_observations = posterior_probability_flow(
            forecast_observations,
            mixture,
            self.config.flow,
        )
        initial_target_mismatch = torch.linalg.matrix_norm(
            forecast_observations - target_observations
        ).clamp_min(torch.finfo(current.dtype).eps)
        stable_iterations = 0
        diverged = False
        retained_rank = 0
        completed_iterations = 0

        for iteration in range(self.config.flow.max_back_projection_iterations):
            predicted = observation_operator(current)
            low_rank = localized_low_rank_map(
                current,
                predicted,
                self.config.low_rank,
                localization,
            )
            retained_rank = max(retained_rank, low_rank.retained_rank)
            state_increment = propagate_observation_increment(
                target_observations - predicted,
                low_rank,
            )
            proposal = current + state_increment
            proposal_observations = observation_operator(proposal)
            current_target_mismatch = torch.linalg.matrix_norm(
                predicted - target_observations
            ).clamp_min(torch.finfo(current.dtype).eps)
            proposal_target_mismatch = torch.linalg.matrix_norm(
                proposal_observations - target_observations
            )
            if (
                proposal_target_mismatch
                > self.config.flow.divergence_ratio * current_target_mismatch
            ):
                proposal = current + 0.5 * state_increment
                proposal_observations = observation_operator(proposal)
                proposal_target_mismatch = torch.linalg.matrix_norm(
                    proposal_observations - target_observations
                )
                if (
                    proposal_target_mismatch
                    > self.config.flow.divergence_ratio * current_target_mismatch
                ):
                    diverged = True
                    break

            increment_ratio = torch.linalg.matrix_norm(proposal - current) / torch.linalg.matrix_norm(
                current
            ).clamp_min(1.0)
            target_mismatch_ratio = proposal_target_mismatch / initial_target_mismatch
            if (
                target_mismatch_ratio < self.config.flow.innovation_tolerance
                and increment_ratio < self.config.flow.increment_tolerance
            ):
                stable_iterations += 1
            else:
                stable_iterations = 0
            current = proposal
            completed_iterations = iteration + 1
            if stable_iterations >= 2:
                break

        final_observations = observation_operator(current)
        final_innovation = _whitened_norm(
            final_observations.mean(dim=0) - observation,
            observation_covariance,
        )
        diagnostics = AnalysisDiagnostics(
            back_projection_iterations=completed_iterations,
            initial_innovation=float(initial_innovation),
            final_innovation=float(final_innovation),
            retained_rank=retained_rank,
            log_evidence_per_observation=float(log_evidence),
            diverged=diverged,
        )
        return current, diagnostics

    def analyze_paths(
        self,
        path_ensembles: torch.Tensor,
        evidence_tracker: AlphaEvidenceTracker,
        observation: torch.Tensor,
        observation_operator: ObservationOperator,
        observation_covariance: torch.Tensor,
        localization: torch.Tensor | None = None,
    ) -> MultiPathAnalysis:
        if path_ensembles.ndim != 3:
            raise ValueError("Path ensembles must have shape [alpha, ensemble, state]")
        if path_ensembles.shape[0] != evidence_tracker.alpha.numel():
            raise ValueError("One state ensemble is required per active alpha path")
        analyzed = []
        diagnostics = []
        evidence = []
        for branch in path_ensembles:
            branch_analysis, branch_diagnostics = self.analyze_local(
                branch,
                observation,
                observation_operator,
                observation_covariance,
                localization,
            )
            analyzed.append(branch_analysis)
            diagnostics.append(branch_diagnostics)
            evidence.append(branch.new_tensor(branch_diagnostics.log_evidence_per_observation))
        analyzed_tensor = torch.stack(analyzed)
        evidence_tracker.update(torch.stack(evidence).to(evidence_tracker.log_scores))
        adaptation = evidence_tracker.adapt_ensembles(analyzed_tensor)
        analyzed_tensor = adaptation.ensembles
        diagnostics = [diagnostics[index] for index in adaptation.source_indices]
        weights = evidence_tracker.weights
        branch_means = analyzed_tensor.mean(dim=1)
        state_estimate = torch.sum(weights.to(branch_means).unsqueeze(1) * branch_means, dim=0)
        return MultiPathAnalysis(
            ensembles=analyzed_tensor,
            evidence_weights=weights,
            state_estimate=state_estimate,
            alpha_estimate=evidence_tracker.continuous_estimate(),
            diagnostics=tuple(diagnostics),
        )
