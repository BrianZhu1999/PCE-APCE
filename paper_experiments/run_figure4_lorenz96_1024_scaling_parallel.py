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
WORKER = ROOT / "paper_experiments" / "run_figure4_lorenz96_1024_scaling.py"
AGGREGATOR = ROOT / "paper_experiments" / "aggregate_figure4_lorenz96_1024_scaling.py"
DEFAULT_OUTPUT = ROOT / "results" / "figure4_lorenz96_1024_obs32_time2468_smoke_5seeds_20260815"
DEFAULT_METHODS = ("aug_enkf", "bma_static", "pce", "apce")
DEFAULT_INTERVALS = (2, 4, 6, 8)


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


def run_task(task: dict[str, Any]) -> dict[str, Any]:
    output = Path(task["output"])
    method = str(task["method"])
    interval = int(task["obs_interval"])
    seed = int(task["seed"])
    run_path = run_json_path(output, method, interval, seed)
    log_path = output / "logs_parallel" / f"lorenz96_1024_t{interval}_{method}_seed{seed}.log"
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


def prepare_assets(
    output: Path,
    seeds: list[int],
    intervals: list[int],
    devices: list[str],
    *,
    state_dim: int,
    observed_points: int,
    steps: int,
    tuning_profile: str,
    asset_root: Path | None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    interval = int(intervals[0])
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
            str(interval),
            "--steps",
            str(steps),
            "--tuning-profile",
            tuning_profile,
            "--output",
            str(output),
            "--device",
            devices[index % len(devices)],
            "--prepare-asset-only",
        ]
        if asset_root is not None:
            cmd.extend(["--asset-root", str(asset_root)])
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write("COMMAND: " + " ".join(cmd) + "\n\n")
            handle.flush()
            proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, text=True)
        results.append(
            {
                "seed": seed,
                "obs_interval_used_for_asset": interval,
                "device": devices[index % len(devices)],
                "returncode": int(proc.returncode),
                "runtime_seconds": float(time.perf_counter() - started),
                "log_path": str(log_path),
            }
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Shared asset preparation failed for seed {seed}; see {log_path}")
    return results


def parse_csv_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_csv_strings(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel launcher for Figure 4 Lorenz-96 D=1024 scaling runs.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed-base", type=int, default=2026080600)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--state-dim", type=int, default=1024)
    parser.add_argument("--observed-points", type=int, default=32)
    parser.add_argument("--obs-intervals", default=",".join(str(item) for item in DEFAULT_INTERVALS))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--tuning-profile", default="baseline")
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument("--devices", default="cpu")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--no-record-trace", action="store_true")
    parser.add_argument("--no-prepare-assets-first", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    args = parser.parse_args()

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    intervals = parse_csv_ints(args.obs_intervals)
    methods = parse_csv_strings(args.methods)
    devices = parse_csv_strings(args.devices)
    seeds = [int(args.seed_base) + index for index in range(int(args.seed_count))]
    asset_preparation: list[dict[str, Any]] = []
    if not bool(args.no_prepare_assets_first):
        asset_preparation = prepare_assets(
            output,
            seeds,
            intervals,
            devices,
            state_dim=int(args.state_dim),
            observed_points=int(args.observed_points),
            steps=int(args.steps),
            tuning_profile=str(args.tuning_profile),
            asset_root=args.asset_root,
        )
    tasks: list[dict[str, Any]] = []
    for interval in intervals:
        for seed in seeds:
            for method in methods:
                tasks.append(
                    {
                        "case": "lorenz96_1024",
                        "method": method,
                        "seed": seed,
                        "obs_interval": int(interval),
                        "output": str(output),
                        "device": devices[len(tasks) % len(devices)],
                        "record_trace": not bool(args.no_record_trace),
                        "state_dim": int(args.state_dim),
                        "observed_points": int(args.observed_points),
                        "steps": int(args.steps),
                        "tuning_profile": str(args.tuning_profile),
                        "asset_root": str(args.asset_root) if args.asset_root is not None else "",
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
                "obs_intervals": intervals,
                "methods": methods,
                "devices": devices,
                "max_parallel": int(args.max_parallel),
                "record_trace": not bool(args.no_record_trace),
                "tuning_profile": str(args.tuning_profile),
                "asset_root": str(args.asset_root) if args.asset_root is not None else "",
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
        "obs_intervals": intervals,
        "methods": methods,
        "devices": devices,
        "max_parallel": int(args.max_parallel),
        "record_trace": not bool(args.no_record_trace),
        "tuning_profile": str(args.tuning_profile),
        "asset_root": str(args.asset_root) if args.asset_root is not None else "",
        "asset_preparation": asset_preparation,
        "results": results,
    }
    write_manifest(manifest_path, final)
    print(json.dumps(clean_json(final), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
