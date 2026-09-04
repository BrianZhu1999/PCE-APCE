from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import torch

from hilda_da.alpha import AlphaEvidenceTracker
from hilda_da.baselines import denkf_analysis
from hilda_da.config import AlphaConfig, HILDAConfig
from hilda_da.filter import HILDAFilter
from hilda_da.metrics import (
    paired_bootstrap_ci,
    paired_effect_size,
    weighted_central_interval_coverage_width,
)
from hilda_da.observation_flow import (
    analytic_posterior_mixture,
    posterior_probability_flow,
    predictive_log_evidence,
)
from hilda_da.observations import SparseObservation
from hilda_da.strong_baselines import ensf_lr_ridge_analysis
from hilda_da.systems import SpringOscillator


@dataclass(frozen=True)
class Variant:
    name: str
    mode: str
    inner: str = "hilda"
    paired_forecast_noise: bool = False
    shadow_evidence: bool = False
    adaptive_alpha: bool = False
    model_alpha: float | None = None
    moment_matching: bool = False
    moment_matching_strength: float = 1.0


VARIANTS = (
    Variant("hilda_current", "multipath", adaptive_alpha=True),
    Variant(
        "hilda_paired",
        "multipath",
        paired_forecast_noise=True,
        adaptive_alpha=True,
    ),
    Variant(
        "hilda_paired_fixed",
        "multipath",
        paired_forecast_noise=True,
    ),
    Variant(
        "hilda_paired_fixed_moment",
        "multipath",
        paired_forecast_noise=True,
        moment_matching=True,
    ),
    Variant(
        "hilda_paired_fixed_moment25",
        "multipath",
        paired_forecast_noise=True,
        moment_matching=True,
        moment_matching_strength=0.25,
    ),
    Variant(
        "hilda_paired_fixed_moment50",
        "multipath",
        paired_forecast_noise=True,
        moment_matching=True,
        moment_matching_strength=0.50,
    ),
    Variant(
        "hilda_paired_shadow_fixed",
        "multipath",
        paired_forecast_noise=True,
        shadow_evidence=True,
    ),
    Variant(
        "hilda_paired_shadow_adaptive",
        "multipath",
        paired_forecast_noise=True,
        shadow_evidence=True,
        adaptive_alpha=True,
    ),
    Variant(
        "pce_like_ensf_lr",
        "multipath",
        inner="ensf_lr",
        paired_forecast_noise=True,
        shadow_evidence=True,
    ),
    Variant("hilda_oracle", "single", model_alpha=None),
    Variant(
        "hilda_oracle_moment",
        "single",
        model_alpha=None,
        moment_matching=True,
    ),
    Variant("hilda_fixed_wrong", "single", model_alpha=0.5),
    Variant("denkf_oracle", "denkf", model_alpha=None),
    Variant("free_oracle", "free", model_alpha=None),
    Variant("free_wrong", "free", model_alpha=0.5),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Low-dimensional mechanism audit for HILDA before formal simulation"
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=2026080500)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--observation-interval", type=int, default=5)
    parser.add_argument("--ensemble-size", type=int, default=20)
    parser.add_argument("--observation-noise", type=float, default=0.05)
    parser.add_argument("--alpha-true", type=float, default=0.78)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(variant.name for variant in VARIANTS),
        default=None,
    )
    return parser.parse_args()


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _sample_standard_deviation(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = math.fsum(values) / len(values)
    return math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _summaries(rows: list[dict[str, Any]], group_key: str, metrics: list[str]) -> list[dict[str, Any]]:
    summaries = []
    for group in sorted({str(row[group_key]) for row in rows}):
        selected = [row for row in rows if str(row[group_key]) == group]
        for metric in metrics:
            values = [float(row[metric]) for row in selected if row.get(metric) is not None]
            if not values:
                continue
            summaries.append(
                {
                    group_key: group,
                    "metric": metric,
                    "runs": len(values),
                    "mean": math.fsum(values) / len(values),
                    "sample_standard_deviation": _sample_standard_deviation(values),
                }
            )
    return summaries


def posterior_moment_audit(
    seed: int,
    ensemble_size: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(seed)
    forecast = 1.2 + 0.7 * torch.randn(
        ensemble_size, 1, dtype=dtype, device=device, generator=generator
    )
    observation = torch.tensor([-0.15], dtype=dtype, device=device)
    covariance = torch.tensor([[0.09]], dtype=dtype, device=device)
    mixture = analytic_posterior_mixture(forecast, observation, covariance)
    transported = posterior_probability_flow(forecast, mixture, HILDAConfig().flow)
    matched_flow_config = replace(HILDAConfig().flow, moment_matching=True)
    moment_matched = posterior_probability_flow(forecast, mixture, matched_flow_config)

    target_mean = torch.sum(mixture.weights[:, None] * mixture.means, dim=0)
    centred = mixture.means - target_mean
    target_covariance = mixture.covariance + (
        mixture.weights[:, None, None]
        * centred[:, :, None]
        * centred[:, None, :]
    ).sum(dim=0)
    transported_mean = transported.mean(dim=0)
    transported_covariance = torch.cov(transported.mT, correction=0).reshape(1, 1)
    matched_mean = moment_matched.mean(dim=0)
    matched_covariance = torch.cov(moment_matched.mT, correction=0).reshape(1, 1)

    identity = lambda values: values
    analyzed, diagnostics = HILDAFilter().analyze_local(
        forecast,
        observation,
        identity,
        covariance,
    )
    initial_target_mismatch = torch.linalg.matrix_norm(forecast - transported).clamp_min(
        torch.finfo(dtype).eps
    )
    final_target_mismatch = torch.linalg.matrix_norm(analyzed - transported)
    return {
        "seed": seed,
        "ensemble_size": ensemble_size,
        "standardized_mean_error": float(
            torch.linalg.vector_norm(transported_mean - target_mean)
            / torch.sqrt(torch.trace(target_covariance)).clamp_min(torch.finfo(dtype).eps)
        ),
        "variance_ratio": float(
            torch.trace(transported_covariance)
            / torch.trace(target_covariance).clamp_min(torch.finfo(dtype).eps)
        ),
        "relative_covariance_error": float(
            torch.linalg.matrix_norm(transported_covariance - target_covariance)
            / torch.linalg.matrix_norm(target_covariance).clamp_min(torch.finfo(dtype).eps)
        ),
        "matched_standardized_mean_error": float(
            torch.linalg.vector_norm(matched_mean - target_mean)
            / torch.sqrt(torch.trace(target_covariance)).clamp_min(torch.finfo(dtype).eps)
        ),
        "matched_variance_ratio": float(
            torch.trace(matched_covariance)
            / torch.trace(target_covariance).clamp_min(torch.finfo(dtype).eps)
        ),
        "matched_relative_covariance_error": float(
            torch.linalg.matrix_norm(matched_covariance - target_covariance)
            / torch.linalg.matrix_norm(target_covariance).clamp_min(torch.finfo(dtype).eps)
        ),
        "identity_back_projection_mismatch_ratio": float(
            final_target_mismatch / initial_target_mismatch
        ),
        "analysis_diverged": diagnostics.diverged,
    }


def _propagate_calls(
    system: SpringOscillator,
    calls: list[tuple[torch.Tensor, float]],
    time_value: float,
    dt: float,
    generator: torch.Generator,
    *,
    paired: bool,
) -> list[torch.Tensor]:
    if not paired:
        return [
            system.step(state, time_value, dt, alpha, generator)
            for state, alpha in calls
        ]
    shared_state = generator.get_state()
    outputs = []
    final_state = None
    for state, alpha in calls:
        generator.set_state(shared_state)
        outputs.append(system.step(state, time_value, dt, alpha, generator))
        final_state = generator.get_state()
    if final_state is not None:
        generator.set_state(final_state)
    return outputs


def _analyze_ensf_lr_paths(
    branches: torch.Tensor,
    observation: torch.Tensor,
    operator: SparseObservation,
    covariance: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    shared_state = generator.get_state()
    analyzed = []
    final_state = None
    for branch in branches:
        generator.set_state(shared_state)
        analyzed.append(
            ensf_lr_ridge_analysis(
                branch,
                observation,
                operator,
                covariance,
                generator=generator,
            )
        )
        final_state = generator.get_state()
    if final_state is not None:
        generator.set_state(final_state)
    return torch.stack(analyzed)


def _entropy(weights: torch.Tensor) -> float:
    safe = weights.clamp_min(torch.finfo(weights.dtype).tiny)
    return float(-(safe * safe.log()).sum())


def run_spring_variant(
    seed: int,
    variant: Variant,
    args: argparse.Namespace,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    system = SpringOscillator()
    generator = torch.Generator(device=device).manual_seed(seed)
    truth_generator = torch.Generator(device=device).manual_seed(seed + 1)
    observation_generator = torch.Generator(device=device).manual_seed(seed + 2)
    truth = torch.tensor([0.6, 0.0], dtype=dtype, device=device)
    initial_spread = 0.08 * torch.randn(
        args.ensemble_size,
        system.state_dim,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    initial_ensemble = truth.unsqueeze(0) + initial_spread
    operator = SparseObservation(torch.tensor([0], dtype=torch.int64, device=device))
    covariance = args.observation_noise**2 * torch.eye(1, dtype=dtype, device=device)
    hilda_config = HILDAConfig(
        flow=replace(
            HILDAConfig().flow,
            moment_matching=variant.moment_matching,
            moment_matching_strength=variant.moment_matching_strength,
        )
    )
    hilda = HILDAFilter(hilda_config)

    tracker = None
    branches = None
    shadow = None
    ensemble = None
    if variant.mode == "multipath":
        alpha_config = AlphaConfig()
        tracker = AlphaEvidenceTracker.create(alpha_config, device=device, dtype=torch.float64)
        branches = initial_ensemble.unsqueeze(0).expand(tracker.alpha.numel(), -1, -1).clone()
        shadow = branches.clone() if variant.shadow_evidence else None
    else:
        ensemble = initial_ensemble.clone()

    position_errors: list[float] = []
    truth_position_rms: list[float] = []
    alpha_errors: list[float] = []
    entropies: list[float] = []
    active_paths: list[float] = []
    innovations: list[float] = []
    position_coverages: list[float] = []
    position_interval_widths: list[float] = []

    for step in range(args.steps):
        time_value = step * args.dt
        truth = system.step(
            truth,
            time_value,
            args.dt,
            args.alpha_true,
            truth_generator,
        )
        model_alpha = args.alpha_true if variant.model_alpha is None else variant.model_alpha

        if variant.mode == "multipath":
            assert tracker is not None and branches is not None
            calls = [
                (branch, float(alpha))
                for alpha, branch in zip(tracker.alpha, branches, strict=True)
            ]
            if shadow is not None:
                calls.extend(
                    (branch, float(alpha))
                    for alpha, branch in zip(tracker.alpha, shadow, strict=True)
                )
            propagated = _propagate_calls(
                system,
                calls,
                time_value,
                args.dt,
                generator,
                paired=variant.paired_forecast_noise,
            )
            branch_count = tracker.alpha.numel()
            branches = torch.stack(propagated[:branch_count])
            if shadow is not None:
                shadow = torch.stack(propagated[branch_count:])
        else:
            assert ensemble is not None
            ensemble = system.step(
                ensemble,
                time_value,
                args.dt,
                model_alpha,
                generator,
            )

        if (step + 1) % args.observation_interval != 0:
            continue
        clean_observation = operator(truth.unsqueeze(0)).squeeze(0)
        observation = clean_observation + args.observation_noise * torch.randn(
            clean_observation.shape,
            dtype=dtype,
            device=device,
            generator=observation_generator,
        )

        if variant.mode == "multipath":
            assert tracker is not None and branches is not None
            evidence_source = shadow if shadow is not None else branches
            evidence = torch.stack(
                [
                    predictive_log_evidence(
                        operator(branch),
                        observation,
                        covariance,
                    )
                    for branch in evidence_source
                ]
            ).to(tracker.log_scores)
            if variant.inner == "hilda":
                analyzed_items = [
                    hilda.analyze_local(
                        branch,
                        observation,
                        operator,
                        covariance,
                    )
                    for branch in branches
                ]
                branches = torch.stack([item[0] for item in analyzed_items])
                innovations.extend(item[1].final_innovation for item in analyzed_items)
            else:
                branches = _analyze_ensf_lr_paths(
                    branches,
                    observation,
                    operator,
                    covariance,
                    generator,
                )
                innovations.append(
                    float(torch.linalg.vector_norm(operator(branches).mean(dim=(0, 1)) - observation))
                )
            tracker.update(evidence)
            if variant.adaptive_alpha:
                if shadow is None:
                    branches = tracker.adapt_ensembles(branches).ensembles
                else:
                    state_dimension = branches.shape[-1]
                    combined = torch.cat((branches, shadow), dim=-1)
                    combined = tracker.adapt_ensembles(combined).ensembles
                    branches = combined[..., :state_dimension]
                    shadow = combined[..., state_dimension:]
            weights = tracker.weights.to(branches)
            estimate = torch.sum(weights[:, None] * branches.mean(dim=1), dim=0)
            metric_ensemble = branches.reshape(-1, branches.shape[-1])
            metric_weights = (
                weights[:, None]
                .expand(-1, branches.shape[1])
                .reshape(-1)
                / branches.shape[1]
            )
            alpha_estimate = tracker.continuous_estimate()
            alpha_errors.append(abs(alpha_estimate - args.alpha_true))
            entropies.append(_entropy(tracker.weights))
            active_paths.append(float(tracker.alpha.numel()))
        elif variant.mode == "single":
            assert ensemble is not None
            ensemble, diagnostics = hilda.analyze_local(
                ensemble,
                observation,
                operator,
                covariance,
            )
            estimate = ensemble.mean(dim=0)
            innovations.append(diagnostics.final_innovation)
            metric_ensemble = ensemble
            metric_weights = torch.full(
                (ensemble.shape[0],),
                1.0 / ensemble.shape[0],
                dtype=ensemble.dtype,
                device=ensemble.device,
            )
        elif variant.mode == "denkf":
            assert ensemble is not None
            ensemble = denkf_analysis(ensemble, observation, operator, covariance)
            estimate = ensemble.mean(dim=0)
            innovations.append(
                float(torch.linalg.vector_norm(operator(ensemble).mean(dim=0) - observation))
            )
            metric_ensemble = ensemble
            metric_weights = torch.full(
                (ensemble.shape[0],),
                1.0 / ensemble.shape[0],
                dtype=ensemble.dtype,
                device=ensemble.device,
            )
        else:
            assert ensemble is not None
            estimate = ensemble.mean(dim=0)
            metric_ensemble = ensemble
            metric_weights = torch.full(
                (ensemble.shape[0],),
                1.0 / ensemble.shape[0],
                dtype=ensemble.dtype,
                device=ensemble.device,
            )

        position_errors.append(float((estimate[0] - truth[0]).abs()))
        truth_position_rms.append(float(truth[0].abs()))
        position_coverage, position_width = weighted_central_interval_coverage_width(
            metric_ensemble[:, :1],
            truth[:1],
            metric_weights,
            level=0.90,
        )
        position_coverages.append(float(position_coverage))
        position_interval_widths.append(float(position_width))

    mean_position_rmse = math.fsum(position_errors) / len(position_errors)
    mean_truth_rms = math.fsum(truth_position_rms) / len(truth_position_rms)
    return {
        "seed": seed,
        "variant": variant.name,
        "mean_position_rmse": mean_position_rmse,
        "position_ratio_of_means_nrmse": mean_position_rmse / mean_truth_rms,
        "mean_truth_position_rms": mean_truth_rms,
        "mean_position_coverage_90": math.fsum(position_coverages) / len(position_coverages),
        "mean_position_interval_width_90": (
            math.fsum(position_interval_widths) / len(position_interval_widths)
        ),
        "mean_alpha_absolute_error": (
            math.fsum(alpha_errors) / len(alpha_errors) if alpha_errors else None
        ),
        "final_alpha_estimate": (
            tracker.continuous_estimate() if tracker is not None else None
        ),
        "mean_weight_entropy": (
            math.fsum(entropies) / len(entropies) if entropies else None
        ),
        "mean_active_paths": (
            math.fsum(active_paths) / len(active_paths) if active_paths else None
        ),
        "mean_analysis_innovation": (
            math.fsum(innovations) / len(innovations) if innovations else None
        ),
    }


def _paired_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_variant = {
        variant: {
            int(row["seed"]): float(row["mean_position_rmse"])
            for row in rows
            if row["variant"] == variant
        }
        for variant in sorted({row["variant"] for row in rows})
    }
    current = by_variant.get("hilda_current")
    if current is None:
        return []
    output = []
    for variant, samples in by_variant.items():
        if variant == "hilda_current":
            continue
        seeds = sorted(set(current) & set(samples))
        first = torch.tensor([samples[seed] for seed in seeds], dtype=torch.float64)
        second = torch.tensor([current[seed] for seed in seeds], dtype=torch.float64)
        ci = paired_bootstrap_ci(
            first,
            second,
            resamples=10_000,
            seed=20260805 + len(output),
        )
        output.append(
            {
                "variant": variant,
                "reference": "hilda_current",
                "metric": "mean_position_rmse",
                "paired_count": len(seeds),
                "mean_variant": float(first.mean()),
                "mean_reference": float(second.mean()),
                "mean_difference": float(ci.estimate),
                "bootstrap_95_ci_lower": float(ci.lower),
                "bootstrap_95_ci_upper": float(ci.upper),
                "cohen_dz": float(paired_effect_size(first, second)) if len(seeds) > 1 else None,
                "direction_improves_on_current": float(ci.estimate) < 0.0,
            }
        )
    return output


def _decision_table(
    moment_summaries: list[dict[str, Any]],
    spring_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    moment_lookup = {
        (row["ensemble_size"], row["metric"]): float(row["mean"])
        for row in moment_summaries
    }
    spring_lookup = {
        (row["variant"], row["metric"]): float(row["mean"])
        for row in spring_summaries
    }

    def ratio(first: str, second: str, metric: str = "mean_position_rmse") -> float:
        return spring_lookup[(first, metric)] / spring_lookup[(second, metric)]

    required_variants = {
        "free_wrong", "free_oracle", "hilda_oracle", "denkf_oracle",
        "hilda_paired", "hilda_current", "hilda_paired_shadow_fixed",
        "hilda_paired_shadow_adaptive", "pce_like_ensf_lr",
    }
    if not required_variants.issubset({key[0] for key in spring_lookup}):
        return []
    flow_mean = moment_lookup[("200", "standardized_mean_error")]
    flow_variance = moment_lookup[("200", "variance_ratio")]
    back_projection = moment_lookup[("200", "identity_back_projection_mismatch_ratio")]
    visibility_ratio = ratio("free_wrong", "free_oracle")
    return [
        {
            "mechanism": "posterior_flow_moments",
            "diagnostic": f"N=200 mean error={flow_mean:.6g}, variance ratio={flow_variance:.6g}",
            "pass": flow_mean < 0.10 and 0.80 <= flow_variance <= 1.20,
        },
        {
            "mechanism": "identity_back_projection",
            "diagnostic": f"N=200 target mismatch ratio={back_projection:.6g}",
            "pass": back_projection < 0.05,
        },
        {
            "mechanism": "alpha_visibility",
            "diagnostic": f"free wrong/oracle position-RMSE ratio={visibility_ratio:.6g}",
            "pass": visibility_ratio > 1.25,
        },
        {
            "mechanism": "oracle_inner_filter",
            "diagnostic": f"HILDA-oracle/DEnKF-oracle RMSE ratio={ratio('hilda_oracle', 'denkf_oracle'):.6g}",
            "pass": ratio("hilda_oracle", "denkf_oracle") <= 1.25,
        },
        {
            "mechanism": "paired_forecast_noise",
            "diagnostic": f"paired/current RMSE ratio={ratio('hilda_paired', 'hilda_current'):.6g}",
            "pass": ratio("hilda_paired", "hilda_current") <= 1.0,
        },
        {
            "mechanism": "shadow_evidence_fixed_grid",
            "diagnostic": f"paired-shadow-fixed/current RMSE ratio={ratio('hilda_paired_shadow_fixed', 'hilda_current'):.6g}",
            "pass": ratio("hilda_paired_shadow_fixed", "hilda_current") <= 1.0,
        },
        {
            "mechanism": "adaptive_alpha_grid",
            "diagnostic": f"adaptive/fixed paired-shadow RMSE ratio={ratio('hilda_paired_shadow_adaptive', 'hilda_paired_shadow_fixed'):.6g}",
            "pass": ratio("hilda_paired_shadow_adaptive", "hilda_paired_shadow_fixed") <= 1.10,
        },
        {
            "mechanism": "hilda_inner_vs_ensf_lr_under_same_outer",
            "diagnostic": f"HILDA/EnSF-LR paired-shadow-fixed RMSE ratio={ratio('hilda_paired_shadow_fixed', 'pce_like_ensf_lr'):.6g}",
            "pass": ratio("hilda_paired_shadow_fixed", "pce_like_ensf_lr") <= 1.25,
        },
    ]


def main() -> None:
    args = parse_args()
    if args.seeds < 2:
        raise ValueError("Use at least two paired seeds")
    if args.steps < args.observation_interval:
        raise ValueError("At least one observation time is required")
    if not 0.0 < args.alpha_true < 1.0:
        raise ValueError("alpha-true must lie strictly inside (0,1)")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    seeds = [args.base_seed + index for index in range(args.seeds)]
    selected_variants = (
        [variant for variant in VARIANTS if variant.name in set(args.variants)]
        if args.variants
        else list(VARIANTS)
    )
    moment_rows = [
        posterior_moment_audit(seed, ensemble_size, dtype=dtype, device=device)
        for seed in seeds
        for ensemble_size in (20, 50, 200)
    ]
    spring_rows = [
        run_spring_variant(seed, variant, args, dtype=dtype, device=device)
        for seed in seeds
        for variant in selected_variants
    ]
    moment_summaries = _summaries(
        [dict(row, ensemble_size=str(row["ensemble_size"])) for row in moment_rows],
        "ensemble_size",
        [
            "standardized_mean_error",
            "variance_ratio",
            "relative_covariance_error",
            "matched_standardized_mean_error",
            "matched_variance_ratio",
            "matched_relative_covariance_error",
            "identity_back_projection_mismatch_ratio",
        ],
    )
    spring_summaries = _summaries(
        spring_rows,
        "variant",
        [
            "mean_position_rmse",
            "position_ratio_of_means_nrmse",
            "mean_position_coverage_90",
            "mean_position_interval_width_90",
            "mean_alpha_absolute_error",
            "final_alpha_estimate",
            "mean_weight_entropy",
            "mean_active_paths",
            "mean_analysis_innovation",
        ],
    )
    comparisons = _paired_comparisons(spring_rows)
    decisions = _decision_table(moment_summaries, spring_summaries)
    payload = {
        "configuration": {
            **vars(args),
            "output_directory": str(args.output_directory),
            "dtype": str(dtype),
            "device": str(device),
            "variants": [asdict(variant) for variant in selected_variants],
        },
        "posterior_moment_runs": moment_rows,
        "posterior_moment_summaries": moment_summaries,
        "spring_runs": spring_rows,
        "spring_summaries": spring_summaries,
        "paired_comparisons": comparisons,
        "mechanism_decisions": decisions,
    }
    args.output_directory.mkdir(parents=True, exist_ok=True)
    _atomic_write(
        args.output_directory / "mechanism_audit.json",
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
    )
    _atomic_write(args.output_directory / "posterior_moment_runs.csv", _csv_text(moment_rows))
    _atomic_write(
        args.output_directory / "posterior_moment_summaries.csv",
        _csv_text(moment_summaries),
    )
    _atomic_write(args.output_directory / "spring_runs.csv", _csv_text(spring_rows))
    _atomic_write(args.output_directory / "spring_summaries.csv", _csv_text(spring_summaries))
    _atomic_write(args.output_directory / "paired_comparisons.csv", _csv_text(comparisons))
    _atomic_write(args.output_directory / "mechanism_decisions.csv", _csv_text(decisions))
    print(
        json.dumps(
            {
                "seeds": len(seeds),
                "spring_runs": len(spring_rows),
                "passed_mechanisms": sum(bool(row["pass"]) for row in decisions),
                "total_mechanisms": len(decisions),
                "output_directory": str(args.output_directory),
            }
        )
    )


if __name__ == "__main__":
    main()
