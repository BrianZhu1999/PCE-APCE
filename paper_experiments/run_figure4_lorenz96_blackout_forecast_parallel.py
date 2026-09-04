from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "paper_experiments" / "run_figure4_lorenz96_blackout_forecast.py"
AGGREGATOR = ROOT / "paper_experiments" / "aggregate_figure4_lorenz96_blackout_forecast.py"
DEFAULT_OUTPUT = ROOT / "results" / "figure4_lorenz96_1024_obs128_t8_blackout200_smoke_5seeds_20260815"
DEFAULT_METHODS = ("aug_enkf", "bma_static", "pce", "apce")
DEFAULT_PROFILE_BY_METHOD = {
    "aug_enkf": "baseline",
    "bma_static": "baseline",
    "pce": "pce_temp_020+alpha_conservative",
    "apce": "apce_floor_045+alpha_conservative",
}


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def run_json_path(output: Path, method: str, interval: int, seed: int) -> Path:
    return output / "artifacts" / "run_json" / "lorenz96_1024" / f"time{interval}" / method / f"seed_{seed}.json"


def is_completed(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return payload.get("status") == "completed" and payload.get("numerical_status") == "valid"


def parse_csv_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_csv_strings(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def prepare_assets(
    output: Path,
    seeds: list[int],
    devices: list[str],
    *,
    state_dim: int,
    observed_points: int,
    obs_interval: int,
    steps: int,
    blackout_start_step: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        log_path = output / "logs_parallel" / f"prepare_asset_seed{seed}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(WORKER),
            "--seed",
            str(seed),
            "--state-dim",
            str(state_dim),
            "--observed-points",
            str(observed_points),
            "--obs-interval",
            str(obs_interval),
            "--steps",
            str(steps),
            "--blackout-start-step",
            str(blackout_start_step),
            "--tuning-profile",
            "baseline",
            "--output",
            str(output),
            "--device",
            devices[index % len(devices)],
            "--prepare-asset-only",
        ]
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write("COMMAND: " + " ".join(cmd) + "\n\n")
            handle.flush()
            proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, text=True)
        results.append(
            {
                "seed": seed,
                "device": devices[index % len(devices)],
                "returncode": int(proc.returncode),
                "runtime_seconds": float(time.perf_counter() - started),
                "log_path": str(log_path),
            }
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Shared asset preparation failed for seed {seed}; see {log_path}")
    return results


def run_task(task: dict[str, Any]) -> dict[str, Any]:
    output = Path(task["output"])
    method = str(task["method"])
    interval = int(task["obs_interval"])
    seed = int(task["seed"])
    run_path = run_json_path(output, method, interval, seed)
    log_path = output / "logs_parallel" / f"lorenz96_1024_blackout_t{interval}_{method}_seed{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if is_completed(run_path):
        return {
            **task,
            "status": "skipped_completed",
            "returncode": 0,
            "run_json": str(run_path),
            "log_path": str(log_path),
        }
    cmd = [
        sys.executable,
        str(WORKER),
        "--seed",
        str(seed),
        "--state-dim",
        str(task["state_dim"]),
        "--observed-points",
        str(task["observed_points"]),
        "--obs-interval",
        str(interval),
        "--steps",
        str(task["steps"]),
        "--blackout-start-step",
        str(task["blackout_start_step"]),
        "--method",
        method,
        "--tuning-profile",
        str(task["tuning_profile"]),
        "--output",
        str(output),
        "--device",
        str(task["device"]),
    ]
    if task.get("asset_root"):
        cmd.extend(["--asset-root", str(task["asset_root"])])
    if not bool(task["record_trace"]):
        cmd.append("--no-record-trace")
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND: " + " ".join(cmd) + "\n\n")
        handle.flush()
        proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, text=True)
    return {
        **task,
        "status": "completed_process" if proc.returncode == 0 else "failed_process",
        "returncode": int(proc.returncode),
        "runtime_seconds": float(time.perf_counter() - started),
        "run_json": str(run_path),
        "run_json_completed": is_completed(run_path),
        "log_path": str(log_path),
    }


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel launcher for Figure 4 Lorenz-96 blackout forecast runs.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed-base", type=int, default=2026080600)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--state-dim", type=int, default=1024)
    parser.add_argument("--observed-points", type=int, default=128)
    parser.add_argument("--obs-interval", type=int, default=8)
    parser.add_argument("--blackout-start-step", type=int, default=200)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--devices", default="cuda:2,cuda:3")
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--record-trace", action="store_true")
    parser.add_argument("--no-prepare-assets-first", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument("--profiles", default="")
    args = parser.parse_args()

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    devices = parse_csv_strings(args.devices)
    methods = parse_csv_strings(args.methods)
    seeds = [int(args.seed_base) + index for index in range(int(args.seed_count))]
    profile_by_method = dict(DEFAULT_PROFILE_BY_METHOD)
    if args.profiles:
        for item in parse_csv_strings(args.profiles):
            method, profile = item.split("=", 1)
            profile_by_method[method.strip()] = profile.strip()
    asset_root = args.asset_root or (output / "shared_assets")

    asset_preparation: list[dict[str, Any]] = []
    if not bool(args.no_prepare_assets_first):
        asset_preparation = prepare_assets(
            output,
            seeds,
            devices,
            state_dim=int(args.state_dim),
            observed_points=int(args.observed_points),
            obs_interval=int(args.obs_interval),
            steps=int(args.steps),
            blackout_start_step=int(args.blackout_start_step),
        )

    tasks: list[dict[str, Any]] = []
    for seed in seeds:
        for method in methods:
            tasks.append(
                {
                    "case": "lorenz96_1024",
                    "method": method,
                    "seed": seed,
                    "obs_interval": int(args.obs_interval),
                    "blackout_start_step": int(args.blackout_start_step),
                    "output": str(output),
                    "device": devices[len(tasks) % len(devices)],
                    "record_trace": bool(args.record_trace),
                    "state_dim": int(args.state_dim),
                    "observed_points": int(args.observed_points),
                    "steps": int(args.steps),
                    "tuning_profile": profile_by_method.get(method, "baseline"),
                    "asset_root": str(asset_root),
                }
            )

    manifest_path = output / "parallel_launcher_manifest.json"
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with futures.ThreadPoolExecutor(max_workers=max(1, int(args.max_parallel))) as executor:
        future_to_task = {executor.submit(run_task, task): task for task in tasks}
        for future in futures.as_completed(future_to_task):
            result = future.result()
            results.append(result)
            partial = {
                "created_at_unix": time.time(),
                "runtime_seconds_so_far": time.perf_counter() - started,
                "output": str(output),
                "worker": str(WORKER),
                "aggregator": str(AGGREGATOR),
                "total_tasks": len(tasks),
                "finished_tasks": len(results),
                "seed_base": int(args.seed_base),
                "seed_count": int(args.seed_count),
                "seeds": seeds,
                "obs_interval": int(args.obs_interval),
                "blackout_start_step": int(args.blackout_start_step),
                "methods": methods,
                "devices": devices,
                "max_parallel": int(args.max_parallel),
                "record_trace": bool(args.record_trace),
                "asset_root": str(asset_root),
                "profiles": profile_by_method,
                "asset_preparation": asset_preparation,
                "results": results,
            }
            write_manifest(manifest_path, partial)
            print(json.dumps(clean_json(result), ensure_ascii=False), flush=True)

    aggregate_returncode = None
    aggregate_log = output / "logs_parallel" / "aggregate.log"
    if not bool(args.skip_aggregate):
        cmd = [sys.executable, str(AGGREGATOR), "--output", str(output)]
        with aggregate_log.open("w", encoding="utf-8") as handle:
            handle.write("COMMAND: " + " ".join(cmd) + "\n\n")
            handle.flush()
            aggregate_returncode = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, text=True).returncode

    final = {
        "created_at_unix": time.time(),
        "runtime_seconds": time.perf_counter() - started,
        "output": str(output),
        "worker": str(WORKER),
        "aggregator": str(AGGREGATOR),
        "total_tasks": len(tasks),
        "finished_tasks": len(results),
        "completed_or_skipped_processes": sum(1 for row in results if int(row.get("returncode", 1)) == 0),
        "process_failures": sum(1 for row in results if int(row.get("returncode", 1)) != 0),
        "aggregate_returncode": aggregate_returncode,
        "aggregate_log": str(aggregate_log) if aggregate_returncode is not None else "",
        "seed_base": int(args.seed_base),
        "seed_count": int(args.seed_count),
        "seeds": seeds,
        "obs_interval": int(args.obs_interval),
        "blackout_start_step": int(args.blackout_start_step),
        "methods": methods,
        "devices": devices,
        "max_parallel": int(args.max_parallel),
        "record_trace": bool(args.record_trace),
        "asset_root": str(asset_root),
        "profiles": profile_by_method,
        "asset_preparation": asset_preparation,
        "results": results,
    }
    write_manifest(manifest_path, final)
    print(json.dumps(clean_json(final), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
