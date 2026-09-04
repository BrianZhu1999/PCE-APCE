"""Run one strict internal uncertainty-calibration trial on a prepared fold."""
from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Any

import numpy as np
import torch

from .assimilation import run_two_pass
from .common import json_ready, load_config, software_environment, write_json
from .run_case import _load_library, _load_scenario


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one VIV-PIV internal pseudo-holdout calibration trial.")
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--method", choices=("pce", "apce"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--layout", default="20x40")
    parser.add_argument("--initial-ensemble-scale", type=float, required=True)
    parser.add_argument("--process-noise-scale", type=float, required=True)
    parser.add_argument("--state-inflation", type=float, required=True)
    parser.add_argument("--covariance-shrinkage", type=float, required=True)
    parser.add_argument("--probabilistic-metric-stride", type=int, default=None)
    parser.add_argument("--evidence-window-frames", type=int, default=1)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--trace-output", type=pathlib.Path, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = load_config(args.config)
    case_id = str(args.case).replace(",", "").zfill(4)[-4:]
    reference_cases = [str(value) for value in config["train_cases"] if str(value) != case_id]
    if case_id not in config["train_cases"]:
        raise ValueError(f"Calibration case must be a configured training case, received {case_id}")
    for name, value in (
        ("initial_ensemble_scale", args.initial_ensemble_scale),
        ("process_noise_scale", args.process_noise_scale),
        ("state_inflation", args.state_inflation),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    if not 0.0 <= args.covariance_shrinkage <= 1.0:
        raise ValueError("covariance shrinkage must lie in [0, 1]")
    if args.probabilistic_metric_stride is not None and args.probabilistic_metric_stride < 1:
        raise ValueError("probabilistic metric stride must be at least one")
    if args.evidence_window_frames < 1:
        raise ValueError("evidence window must be at least one frame")
    config = dict(config)
    config.update(
        initial_ensemble_scale=float(args.initial_ensemble_scale),
        process_noise_scale=float(args.process_noise_scale),
        state_inflation=float(args.state_inflation),
        observation_covariance_shrinkage=float(args.covariance_shrinkage),
        evidence_window_frames=int(args.evidence_window_frames),
    )
    if args.probabilistic_metric_stride is not None:
        config["probabilistic_metric_stride"] = int(args.probabilistic_metric_stride)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "protocol": "strict leave-one-training-case-out uncertainty calibration",
        "case_id": case_id,
        "method": args.method,
        "seed": int(args.seed),
        "variant": args.variant,
        "layout": args.layout,
        "reference_cases": reference_cases,
        "candidate_excludes_target": True,
        "initial_ensemble_scale": float(args.initial_ensemble_scale),
        "process_noise_scale": float(args.process_noise_scale),
        "state_inflation": float(args.state_inflation),
        "observation_covariance_shrinkage": float(args.covariance_shrinkage),
        "observation_covariance": "full, target-excluded reference estimate",
        "probabilistic_metric_stride": int(config["probabilistic_metric_stride"]),
        "evidence_window_frames": int(args.evidence_window_frames),
        "valid": False,
        "status": "started",
    }
    started = time.perf_counter()
    try:
        model_root = pathlib.Path(config["output_root"]) / "models" / args.variant
        scenario = _load_scenario(
            model_root, case_id, config, False, sensor_layout=args.layout,
            observation_covariance_mode="full", target_split="test",
            reference_cases=reference_cases, covariance_shrinkage=float(args.covariance_shrinkage),
        )
        library = _load_library(model_root, exclude_case_ids={case_id})
        if case_id in [candidate.case_id for candidate in library.candidates]:
            raise RuntimeError("Pseudo-holdout leaked into the candidate library")
        device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
        result = run_two_pass(
            scenario, library, config, args.method, args.seed, device,
            record_trace=args.trace_output is not None, blackout_origins=set(),
        )
        if args.trace_output is not None:
            args.trace_output.parent.mkdir(parents=True, exist_ok=True)
            trace_arrays: dict[str, Any] = {
                "latent_estimate": result.local.latent_estimate,
                "candidate_grid": result.local.grid,
                "final_weights": result.local.final_weights,
                "final_scores": result.local.final_scores,
                "time_s": scenario.time_s,
                "evaluation_truth": scenario.evaluation_truth,
            }
            trace_arrays.update(result.local.trace)
            np.savez_compressed(args.trace_output, **trace_arrays)
        payload.update(
            result.local.metrics,
            candidate_count=int(result.local.grid.size),
            candidate_grid=result.local.grid,
            final_weights=result.local.final_weights,
            local_grid_stable=bool(result.local_stable),
            local_grid_failure=result.local_failure,
            trace_output=None if args.trace_output is None else str(args.trace_output),
            valid=True,
            status="completed",
        )
    except Exception as exc:  # Persist a worker diagnostic for every failed grid point.
        payload.update(status="failed", failure_type=type(exc).__name__, failure_message=str(exc), valid=False)
    payload["wall_seconds"] = float(time.perf_counter() - started)
    payload["environment"] = software_environment()
    write_json(output, payload)
    print(json.dumps(json_ready(payload), ensure_ascii=False))
    if not payload["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
