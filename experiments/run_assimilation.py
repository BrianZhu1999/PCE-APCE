from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hilda_da.alpha import AlphaEvidenceTracker
from hilda_da.baselines import denkf_analysis, letkf_analysis
from hilda_da.config import HILDAConfig
from hilda_da.filter import HILDAFilter
from hilda_da.metrics import (
    observation_rmse,
    state_rmse,
    weighted_central_interval_coverage_width,
    weighted_ensemble_crps,
    weighted_multivariate_energy_score,
)
from hilda_da.observations import SparseObservation, evenly_spaced_indices
from hilda_da.strong_baselines import (
    EnFFF2PConfig,
    IEnSFConfig,
    enff_f2p_analysis,
    ensf_analysis,
    ensf_lr_analysis,
    ensf_lr_ridge_analysis,
    iensf_analysis,
)
from hilda_da.systems import (
    AllenCahn1D,
    AllenCahnConfig,
    Burgers1D,
    BurgersConfig,
    EnFFNavierStokes2D,
    EnFFNavierStokesConfig,
    Heat1D,
    HeatConfig,
    NavierStokes2D,
    NavierStokesConfig,
    SpringOscillator,
    Wave1D,
    WaveConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible HILDA assimilation experiments")
    parser.add_argument(
        "--system",
        choices=(
            "spring", "heat", "wave", "burgers", "allen_cahn",
            "navier_stokes", "navier_stokes_enff",
        ),
        required=True,
    )
    parser.add_argument(
        "--method",
        choices=("hilda", "denkf", "letkf", "ensf", "iensf", "ensf_lr", "ensf_lr_ridge", "enff_f2p"),
        default="hilda",
    )
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--ensemble-size", type=int, default=20)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--observation-interval", type=int, default=5)
    parser.add_argument("--observation-count", type=int, default=16)
    parser.add_argument("--observation-noise", type=float, default=0.05)
    parser.add_argument("--observation-transform", choices=("linear", "atan", "square_signed"), default="linear")
    parser.add_argument("--alpha-mode", choices=("fixed", "drift", "switch"), default="fixed")
    parser.add_argument("--alpha-true", type=float, default=0.72)
    parser.add_argument("--alpha-after-switch", type=float, default=0.28)
    parser.add_argument("--fixed-model-alpha", type=float, default=0.5)
    parser.add_argument("--enff-grid-size", type=int, default=256)
    parser.add_argument("--iensf-gamma", type=float, default=None)
    parser.add_argument(
        "--iensf-variance-split-mode",
        choices=("variance_consistent", "literal"),
        default="variance_consistent",
    )
    parser.add_argument("--iensf-flow-steps", type=int, default=40)
    parser.add_argument("--iensf-refinement-iterations", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--coverage-level", type=float, default=0.9)
    parser.add_argument("--energy-score-chunk-size", type=int, default=64)
    parser.add_argument("--resume-run", type=Path, default=None)
    return parser.parse_args()


def build_system(name: str, enff_grid_size: int = 256):
    if name == "spring":
        return SpringOscillator(), 0.01
    if name == "heat":
        return Heat1D(HeatConfig(nx=128)), 2e-4
    if name == "wave":
        return Wave1D(WaveConfig(nx=128)), 5e-4
    if name == "burgers":
        return Burgers1D(BurgersConfig(nx=256)), 5e-4
    if name == "allen_cahn":
        return AllenCahn1D(AllenCahnConfig(nx=256)), 1e-3
    if name == "navier_stokes":
        return NavierStokes2D(NavierStokesConfig(nx=64, ny=64)), 2e-4
    if name == "navier_stokes_enff":
        return EnFFNavierStokes2D(
            EnFFNavierStokesConfig(nx=enff_grid_size, ny=enff_grid_size)
        ), 1e-4
    raise ValueError(name)


def initial_state(system_name: str, system, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if system_name == "spring":
        return torch.tensor([0.6, 0.0], dtype=dtype, device=device)
    if system_name in {"heat", "wave"}:
        nx = system.config.nx
        grid = torch.linspace(0.0, system.config.length, nx, dtype=dtype, device=device)
        displacement = 0.6 * torch.sin(math.pi * grid / system.config.length)
        if system_name == "heat":
            return displacement
        return torch.cat((displacement, torch.zeros_like(displacement)))
    if system_name in {"burgers", "allen_cahn"}:
        nx = system.config.nx
        grid = torch.linspace(0.0, system.config.length, nx + 1, dtype=dtype, device=device)[:-1]
        if system_name == "burgers":
            return torch.sin(grid) + 0.2 * torch.sin(2.0 * grid)
        return 0.7 * torch.tanh(torch.sin(grid) / 0.2)
    if system_name == "navier_stokes":
        ny, nx = system.config.ny, system.config.nx
        x = torch.linspace(0.0, 2.0 * math.pi, nx + 1, dtype=dtype, device=device)[:-1]
        y = torch.linspace(0.0, 2.0 * math.pi, ny + 1, dtype=dtype, device=device)[:-1]
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return (0.5 * torch.sin(xx) * torch.sin(yy) + 0.2 * torch.cos(2.0 * yy)).reshape(-1)
    if system_name == "navier_stokes_enff":
        return system.taylor_green_state(dtype=dtype, device=device)
    raise ValueError(system_name)


def alpha_at(args: argparse.Namespace, step: int) -> float:
    if args.alpha_mode == "fixed":
        return args.alpha_true
    progress = step / max(args.steps - 1, 1)
    if args.alpha_mode == "drift":
        return float(np.clip(0.5 + 0.32 * math.sin(2.0 * math.pi * progress), 0.02, 0.98))
    return args.alpha_true if progress < 0.5 else args.alpha_after_switch


def source_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "hilda_da").rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_iensf_config(args: argparse.Namespace) -> IEnSFConfig:
    gamma = args.iensf_gamma
    if gamma is None:
        gamma = 1.0 if args.observation_transform == "linear" else 0.25
    return IEnSFConfig(
        gamma=gamma,
        variance_split_mode=args.iensf_variance_split_mode,
        sampling_time_step_count=args.iensf_flow_steps,
        refinement_iterations=args.iensf_refinement_iterations,
    )


def resolved_config(args: argparse.Namespace, dt: float, state_dim: int) -> dict[str, Any]:
    result = vars(args).copy()
    result.pop("checkpoint_interval")
    result.pop("resume_run")
    result.pop("run_id")
    result["output_root"] = str(result["output_root"])
    result["dt"] = dt
    result["state_dim"] = state_dim
    result["hilda"] = asdict(HILDAConfig())
    if args.method == "iensf":
        result["iensf"] = asdict(build_iensf_config(args))
    return result


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _resume_signature(configuration: dict[str, Any]) -> dict[str, Any]:
    signature = configuration.copy()
    signature.pop("output_root", None)
    return signature


def write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.ensemble_size < 4:
        raise ValueError("ensemble-size must be at least 4")
    if args.observation_interval < 1:
        raise ValueError("observation-interval must be positive")
    if args.checkpoint_interval < 1:
        raise ValueError("checkpoint-interval must be positive")
    if not 0.0 < args.coverage_level < 1.0:
        raise ValueError("coverage-level must lie strictly between zero and one")
    if args.energy_score_chunk_size < 1:
        raise ValueError("energy-score-chunk-size must be positive")
    if not 0.0 < args.fixed_model_alpha < 1.0:
        raise ValueError("fixed-model-alpha must lie strictly between zero and one")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    system, default_dt = build_system(args.system, args.enff_grid_size)
    dt = args.dt if args.dt is not None else default_dt
    if args.observation_count > 128:
        raise ValueError("This entry point currently accepts at most 128 observations per local patch")

    project_root = Path(__file__).resolve().parents[1]
    configuration = resolved_config(args, dt, system.state_dim)
    if args.resume_run is None:
        configuration_hash = hashlib.sha256(
            json.dumps(configuration, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        run_id = args.run_id or f"{args.system}_{args.method}_s{args.seed}_{configuration_hash}"
        if Path(run_id).name != run_id or run_id in {".", ".."}:
            raise ValueError("run-id must be a single directory name")
        run_directory = args.output_root / run_id
        if run_directory.exists():
            raise FileExistsError(f"Immutable run already exists: {run_directory}")
        run_directory.mkdir(parents=True)
    else:
        run_directory = args.resume_run.resolve()
        stored_configuration_path = run_directory / "config.json"
        checkpoint_path = run_directory / "checkpoint.pt"
        if not stored_configuration_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError("Resume requires config.json and checkpoint.pt")
        stored_configuration = json.loads(stored_configuration_path.read_text(encoding="utf-8"))
        if _resume_signature(stored_configuration) != _resume_signature(configuration):
            raise ValueError("Resume arguments do not match the immutable run configuration")
        configuration = stored_configuration
        run_id = run_directory.name
    if args.resume_run is None:
        write_json(run_directory / "config.json", configuration)

    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    truth_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    observation_generator = torch.Generator(device=device).manual_seed(args.seed + 2)

    truth = initial_state(args.system, system, dtype=dtype, device=device)
    indices = evenly_spaced_indices(system.state_dim, args.observation_count).to(device)
    observation_operator = SparseObservation(indices, args.observation_transform)
    observation_covariance = args.observation_noise**2 * torch.eye(
        indices.numel(), dtype=dtype, device=device
    )
    initial_spread = 0.08 * torch.randn(
        args.ensemble_size,
        system.state_dim,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    if args.system == "navier_stokes_enff":
        projected = system.project_state(truth.unsqueeze(0) + initial_spread, dt)
        initial_spread = projected - truth.unsqueeze(0)

    hilda = HILDAFilter()
    tracker = AlphaEvidenceTracker.create(hilda.config.alpha, device=device, dtype=torch.float64)
    if args.method == "hilda":
        ensemble = truth.unsqueeze(0).unsqueeze(0) + initial_spread.unsqueeze(0)
        ensemble = ensemble.expand(tracker.alpha.numel(), -1, -1).clone()
    else:
        ensemble = truth.unsqueeze(0) + initial_spread
    previous_filtering = ensemble.clone() if args.method == "enff_f2p" else None

    metrics: list[dict[str, Any]] = []
    truth_history = [truth.detach().cpu()]
    estimate_history = []
    alpha_history = []
    start_step = 0
    elapsed_before_resume = 0.0
    resume_count = 0
    if args.resume_run is not None:
        checkpoint = torch.load(
            run_directory / "checkpoint.pt",
            map_location=device,
            weights_only=False,
        )
        if checkpoint.get("completed", False):
            raise RuntimeError("Run is already complete and cannot be resumed")
        start_step = int(checkpoint["next_step"])
        truth = checkpoint["truth"].to(device=device, dtype=dtype)
        ensemble = checkpoint["ensemble"].to(device=device, dtype=dtype)
        previous_filtering = checkpoint["previous_filtering"]
        if previous_filtering is not None:
            previous_filtering = previous_filtering.to(device=device, dtype=dtype)
        metrics = checkpoint["metrics"]
        truth_history = [item.detach().cpu() for item in checkpoint["truth_history"]]
        estimate_history = [
            item.detach().cpu() for item in checkpoint["estimate_history"]
        ]
        alpha_history = checkpoint["alpha_history"]
        generator.set_state(checkpoint["generator_state"].cpu())
        truth_generator.set_state(checkpoint["truth_generator_state"].cpu())
        observation_generator.set_state(checkpoint["observation_generator_state"].cpu())
        if args.method == "hilda":
            tracker.alpha = checkpoint["tracker_alpha"].to(device=device, dtype=torch.float64)
            tracker.log_scores = checkpoint["tracker_log_scores"].to(
                device=device, dtype=torch.float64
            )
            tracker.low_evidence_counts = checkpoint["tracker_low_evidence_counts"].to(
                device=device, dtype=torch.int64
            )
        elapsed_before_resume = float(checkpoint.get("elapsed_seconds", 0.0))
        resume_count = int(checkpoint.get("resume_count", 0)) + 1
    started = time.time()
    failed = False
    failure_message = None
    completed = False
    next_step = start_step

    def checkpoint_payload(*, is_completed: bool) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "completed": is_completed,
            "next_step": next_step,
            "truth": truth.detach(),
            "ensemble": ensemble.detach(),
            "previous_filtering": (
                previous_filtering.detach() if previous_filtering is not None else None
            ),
            "metrics": metrics,
            "truth_history": truth_history,
            "estimate_history": estimate_history,
            "alpha_history": alpha_history,
            "generator_state": generator.get_state(),
            "truth_generator_state": truth_generator.get_state(),
            "observation_generator_state": observation_generator.get_state(),
            "tracker_alpha": tracker.alpha.detach() if args.method == "hilda" else None,
            "tracker_log_scores": (
                tracker.log_scores.detach() if args.method == "hilda" else None
            ),
            "tracker_low_evidence_counts": (
                tracker.low_evidence_counts.detach() if args.method == "hilda" else None
            ),
            "elapsed_seconds": elapsed_before_resume + time.time() - started,
            "resume_count": resume_count,
        }
    try:
        for step in range(start_step, args.steps):
            time_value = step * dt
            true_alpha = alpha_at(args, step)
            truth = system.step(truth, time_value, dt, true_alpha, truth_generator)
            if args.method == "hilda":
                propagated = []
                for alpha, branch in zip(tracker.alpha, ensemble, strict=True):
                    propagated.append(
                        system.step(branch, time_value, dt, float(alpha), generator)
                    )
                ensemble = torch.stack(propagated)
            else:
                ensemble = system.step(
                    ensemble,
                    time_value,
                    dt,
                    args.fixed_model_alpha,
                    generator,
                )

            next_step = step + 1
            if (step + 1) % args.observation_interval != 0:
                continue
            clean_observation = observation_operator(truth.unsqueeze(0)).squeeze(0)
            observation = clean_observation + args.observation_noise * torch.randn(
                clean_observation.shape,
                dtype=dtype,
                device=device,
                generator=observation_generator,
            )
            if args.method == "hilda":
                analysis = hilda.analyze_paths(
                    ensemble,
                    tracker,
                    observation,
                    observation_operator,
                    observation_covariance,
                )
                ensemble = analysis.ensembles
                estimate = analysis.state_estimate
                alpha_estimate = analysis.alpha_estimate
                innovation = float(np.mean([item.final_innovation for item in analysis.diagnostics]))
                metric_ensemble = analysis.ensembles.reshape(
                    -1, analysis.ensembles.shape[-1]
                )
                metric_weights = (
                    analysis.evidence_weights.to(metric_ensemble)
                    .unsqueeze(1)
                    .expand(-1, analysis.ensembles.shape[1])
                    .reshape(-1)
                    / analysis.ensembles.shape[1]
                )
            elif args.method == "denkf":
                ensemble = denkf_analysis(
                    ensemble, observation, observation_operator, observation_covariance
                )
                estimate = ensemble.mean(dim=0)
                alpha_estimate = None
                innovation = float(torch.linalg.vector_norm(observation_operator(ensemble).mean(0) - observation))
            elif args.method == "letkf":
                ensemble = letkf_analysis(
                    ensemble, observation, observation_operator, observation_covariance
                )
                estimate = ensemble.mean(dim=0)
                alpha_estimate = None
                innovation = float(torch.linalg.vector_norm(observation_operator(ensemble).mean(0) - observation))
            elif args.method == "ensf":
                ensemble = ensf_analysis(
                    ensemble,
                    observation,
                    observation_operator,
                    observation_covariance,
                    generator=generator,
                )
                estimate = ensemble.mean(dim=0)
                alpha_estimate = None
                innovation = float(torch.linalg.vector_norm(observation_operator(ensemble).mean(0) - observation))
            elif args.method in {"ensf_lr", "ensf_lr_ridge"}:
                lr_analysis = (
                    ensf_lr_ridge_analysis
                    if args.method == "ensf_lr_ridge"
                    else ensf_lr_analysis
                )
                ensemble = lr_analysis(
                    ensemble,
                    observation,
                    observation_operator,
                    observation_covariance,
                    generator=generator,
                )
                estimate = ensemble.mean(dim=0)
                alpha_estimate = None
                innovation = float(torch.linalg.vector_norm(observation_operator(ensemble).mean(0) - observation))
            elif args.method == "iensf":
                ensemble = iensf_analysis(
                    ensemble,
                    observation,
                    observation_operator,
                    observation_covariance,
                    config=build_iensf_config(args),
                    generator=generator,
                )
                estimate = ensemble.mean(dim=0)
                alpha_estimate = None
                innovation = float(torch.linalg.vector_norm(observation_operator(ensemble).mean(0) - observation))
            else:
                guidance_lambda = 0.001 if args.system == "navier_stokes" else 0.005
                ensemble = enff_f2p_analysis(
                    previous_filtering,
                    ensemble,
                    observation,
                    observation_operator,
                    observation_covariance,
                    EnFFF2PConfig(guidance_lambda=guidance_lambda),
                    generator,
                )
                previous_filtering = ensemble.clone()
                estimate = ensemble.mean(dim=0)
                alpha_estimate = None
                innovation = float(torch.linalg.vector_norm(observation_operator(ensemble).mean(0) - observation))
            if args.method != "hilda":
                metric_ensemble = ensemble
                metric_weights = torch.full(
                    (ensemble.shape[0],),
                    1.0 / ensemble.shape[0],
                    dtype=ensemble.dtype,
                    device=ensemble.device,
                )
            coverage, interval_width = weighted_central_interval_coverage_width(
                metric_ensemble,
                truth,
                metric_weights,
                level=args.coverage_level,
            )
            predicted_observation = observation_operator(estimate.unsqueeze(0)).squeeze(0)
            metrics.append(
                {
                    "step": step + 1,
                    "time": (step + 1) * dt,
                    "state_rmse": float(state_rmse(estimate, truth)),
                    "observation_rmse": float(
                        observation_rmse(predicted_observation, clean_observation)
                    ),
                    "crps": float(
                        weighted_ensemble_crps(metric_ensemble, truth, metric_weights)
                    ),
                    "energy_score": float(
                        weighted_multivariate_energy_score(
                            metric_ensemble,
                            truth,
                            metric_weights,
                            chunk_size=args.energy_score_chunk_size,
                        )
                    ),
                    "coverage": float(coverage),
                    "interval_width": float(interval_width),
                    "coverage_level": args.coverage_level,
                    "alpha_true": true_alpha,
                    "alpha_estimate": alpha_estimate,
                    "alpha_absolute_error": (
                        abs(alpha_estimate - true_alpha)
                        if alpha_estimate is not None
                        else None
                    ),
                    "fixed_model_alpha": (
                        args.fixed_model_alpha if args.method != "hilda" else None
                    ),
                    "analysis_innovation": innovation,
                }
            )
            truth_history.append(truth.detach().cpu())
            estimate_history.append(estimate.detach().cpu())
            if alpha_estimate is not None:
                alpha_history.append(alpha_estimate)
            if len(metrics) % args.checkpoint_interval == 0:
                write_checkpoint(
                    run_directory / "checkpoint.pt",
                    checkpoint_payload(is_completed=False),
                )
        completed = True
        write_checkpoint(
            run_directory / "checkpoint.pt",
            checkpoint_payload(is_completed=True),
        )
    except Exception as error:
        failed = True
        failure_message = f"{type(error).__name__}: {error}"
        raise
    finally:
        elapsed = elapsed_before_resume + time.time() - started
        provenance = {
            "run_id": run_id,
            "configuration": configuration,
            "source_sha256": source_hash(project_root),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "elapsed_seconds": elapsed,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "completed": completed,
            "resume_count": resume_count,
            "resumed_from_step": start_step if args.resume_run is not None else None,
            "failed": failed,
            "failure_message": failure_message,
        }
        write_json(run_directory / "config.json", configuration)
        write_json(run_directory / "provenance.json", provenance)
        write_json(run_directory / "metrics.json", metrics)
        if metrics:
            write_json(
                run_directory / "summary.json",
                {
                    "mean_state_rmse": float(np.mean([item["state_rmse"] for item in metrics])),
                    "mean_observation_rmse": float(
                        np.mean([item["observation_rmse"] for item in metrics])
                    ),
                    "mean_crps": float(np.mean([item["crps"] for item in metrics])),
                    "mean_energy_score": float(
                        np.mean([item["energy_score"] for item in metrics])
                    ),
                    "mean_coverage": float(
                        np.mean([item["coverage"] for item in metrics])
                    ),
                    "mean_interval_width": float(
                        np.mean([item["interval_width"] for item in metrics])
                    ),
                    "coverage_level": args.coverage_level,
                    "mean_alpha_absolute_error": (
                        float(
                            np.mean(
                                [
                                    item["alpha_absolute_error"]
                                    for item in metrics
                                    if item["alpha_absolute_error"] is not None
                                ]
                            )
                        )
                        if any(
                            item["alpha_absolute_error"] is not None
                            for item in metrics
                        )
                        else None
                    ),
                    "analysis_times": len(metrics),
                },
            )
        torch.save(
            {
                "truth": torch.stack(truth_history),
                "estimate": torch.stack(estimate_history) if estimate_history else torch.empty(0),
                "alpha_estimate": torch.tensor(alpha_history),
            },
            run_directory / "trajectories.pt",
        )

    print(json.dumps({"run_id": run_id, "output": str(run_directory), "metrics": len(metrics)}))


if __name__ == "__main__":
    main()
