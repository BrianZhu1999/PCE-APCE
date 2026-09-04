from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path("<HILDA_RESULTS_ROOT>/code")
OBS_RUNNER = Path(
    "<HILDA_RESULTS_ROOT>/results/"
    "figure3_selected6_observation_insufficient_screen_5seeds_20260813/"
    "run_obs_insufficient.py"
)
FORMAL_ROOT = Path(
    "<HILDA_RESULTS_ROOT>/results/"
    "figure3_selected5_freq1to8_formal_50seeds_allmethods_20260813"
)
DEFAULT_TUNING_MATRIX = Path(
    "<HILDA_RESULTS_ROOT>/results/"
    "figure3_selected6_targeted_auglocal_tuning_5seeds_20260813/"
    "tuning_matrix_targeted_20260813.json"
)


def load_modules():
    import importlib.util

    sys.path.insert(0, str(PROJECT_ROOT))
    spec = importlib.util.spec_from_file_location("figure3_obs_retry_source", OBS_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load observation runner from {OBS_RUNNER}")
    obsrun = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = obsrun
    spec.loader.exec_module(obsrun)

    from hilda_da.strong_baselines import IEnSFConfig, iensf_analysis
    from paper_experiments import run_figure3_applied_ode as figure3

    for factor in range(1, 9):
        obsrun.SCENARIOS[f"freq{factor}"] = {
            "obs_interval_factor": factor,
            "obs_noise_factor": 1.0,
        }
    return obsrun, figure3, IEnSFConfig, iensf_analysis


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_iensf_config(
    figure3,
    IEnSFConfig,
    iensf_analysis,
    *,
    sampling_steps: int,
    refinements: int,
    max_score_component: float,
    endpoint_epsilon: float,
) -> None:
    original = figure3.apply_analysis

    def retry_apply_analysis(
        method,
        ensemble,
        observation,
        operator,
        covariance,
        config,
        step,
        device,
    ):
        if method != "iensf":
            return original(
                method,
                ensemble,
                observation,
                operator,
                covariance,
                config,
                step,
                device,
            )
        return iensf_analysis(
            ensemble,
            observation,
            operator,
            covariance,
            IEnSFConfig(
                sampling_time_step_count=sampling_steps,
                refinement_iterations=refinements,
                endpoint_epsilon=endpoint_epsilon,
                max_score_component=max_score_component,
            ),
            figure3.analysis_generator(config, step, device),
        )

    figure3.apply_analysis = retry_apply_analysis


def failed_tasks(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for factor in range(1, 9):
        for path in sorted((root / f"freq{factor}" / "runs").glob(
            f"fig3obs_freq{factor}_pendulum_iensf_s*.json"
        )):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if row.get("numerical_status") == "valid":
                continue
            rows.append(
                {
                    "scenario": f"freq{factor}",
                    "seed": int(row["seed"]),
                    "original_status": row.get("numerical_status", ""),
                    "original_json": str(path),
                    "original_npz": str(path.with_suffix(".npz")),
                }
            )
    return rows


def select_tasks(tasks: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = tasks
    if args.scenario:
        selected = [row for row in selected if row["scenario"] == args.scenario]
    if args.seed is not None:
        selected = [row for row in selected if row["seed"] == args.seed]
    if args.shard_count > 1:
        selected = [
            row for index, row in enumerate(selected)
            if index % args.shard_count == args.shard_index
        ]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, default=FORMAL_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tuning-matrix", type=Path, default=DEFAULT_TUNING_MATRIX)
    parser.add_argument("--tuning-profile", default="v52_lagcases_aug_global100")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sampling-steps", type=int, default=40)
    parser.add_argument("--refinements", type=int, default=4)
    parser.add_argument("--max-score-component", type=float, default=1000.0)
    parser.add_argument("--endpoint-epsilon", type=float, default=1.0e-3)
    parser.add_argument("--scenario")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--record-all-traces", action="store_true")
    args = parser.parse_args()

    obsrun, figure3, IEnSFConfig, iensf_analysis = load_modules()
    install_iensf_config(
        figure3,
        IEnSFConfig,
        iensf_analysis,
        sampling_steps=args.sampling_steps,
        refinements=args.refinements,
        max_score_component=args.max_score_component,
        endpoint_epsilon=args.endpoint_epsilon,
    )
    profile = obsrun.load_tuning_profile(args.tuning_matrix, args.tuning_profile)
    tasks = select_tasks(failed_tasks(args.formal_root), args)
    device = torch.device(args.device)
    run_dir = args.output / f"shard{args.shard_index}_gpu{device.index or 0}" / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for index, task in enumerate(tasks, 1):
        scenario = task["scenario"]
        seed = int(task["seed"])
        destination = run_dir / f"fig3retry_{scenario}_pendulum_iensf_s{seed}.json"
        started = time.perf_counter()
        row = obsrun.run_one(
            "pendulum",
            "iensf",
            seed,
            scenario,
            device,
            profile,
            args.record_all_traces,
        )
        payload = {
            key: value
            for key, value in row.items()
            if key != "_trace_payload"
        }
        payload.update(
            {
                "retry_reason": "IEnSF probability-flow numerical stabilization",
                "retry_of_json": task["original_json"],
                "original_numerical_status": task["original_status"],
                "iensf_sampling_time_step_count": args.sampling_steps,
                "iensf_refinement_iterations": args.refinements,
                "iensf_max_score_component": args.max_score_component,
                "iensf_endpoint_epsilon": args.endpoint_epsilon,
                "retry_wall_seconds": float(time.perf_counter() - started),
            }
        )
        if args.record_all_traces and "_trace_payload" in row:
            np.savez_compressed(destination.with_suffix(".npz"), **row["_trace_payload"])
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(destination)
        results.append(payload)
        print(
            f"[{index}/{len(tasks)}] {scenario} seed={seed} "
            f"status={payload.get('numerical_status')} "
            f"nrmse={float(payload.get('nrmse', float('nan'))):.6g}",
            flush=True,
        )

    source_dir = args.output / "source_data"
    write_csv(
        source_dir / f"retry_shard_{args.shard_index}.csv",
        results,
    )
    manifest = {
        "protocol": "figure3-pendulum-iensf-same-protocol-numerical-retry",
        "formal_root": str(args.formal_root),
        "output": str(args.output),
        "tasks": len(tasks),
        "valid": sum(row.get("numerical_status") == "valid" for row in results),
        "device": str(device),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "iensf_config": {
            "sampling_time_step_count": args.sampling_steps,
            "refinement_iterations": args.refinements,
            "max_score_component": args.max_score_component,
            "endpoint_epsilon": args.endpoint_epsilon,
        },
        "source_files": {
            str(PROJECT_ROOT / "hilda_da/strong_baselines.py"):
                sha256_file(PROJECT_ROOT / "hilda_da/strong_baselines.py"),
            str(PROJECT_ROOT / "paper_experiments/run_figure3_applied_ode.py"):
                sha256_file(PROJECT_ROOT / "paper_experiments/run_figure3_applied_ode.py"),
            str(OBS_RUNNER): sha256_file(OBS_RUNNER),
            str(Path(__file__)): sha256_file(Path(__file__)),
        },
    }
    (args.output / f"retry_manifest_shard_{args.shard_index}.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
