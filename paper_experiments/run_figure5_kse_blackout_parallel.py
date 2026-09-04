"""Four-GPU launcher for the Figure 5 KSE blackout-forecast smoke/formal runs."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "paper_experiments" / "run_figure5_kse_blackout_forecast.py"
DEFAULT_INPUT = Path("<HILDA_RESULTS_ROOT>/external/S3GM_NMI_2024/KSE_test.npy")
DEFAULT_OUTPUT = Path(
    "<HILDA_RESULTS_ROOT>/results/"
    "figure5_kse_blackoutfull_t1_step40_smoke_5seeds_20260814_4gpu"
)
DEFAULT_METHODS = ("aug_enkf", "bma_static", "pce", "apce")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def is_completed(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return payload.get("status") == "completed" and bool(payload.get("valid"))


def run_task(task: dict[str, Any]) -> dict[str, Any]:
    output_root = Path(task["output_root"])
    run_path = output_root / "runs" / f"{task['run_id']}.json"
    log_path = output_root / "logs_parallel" / f"{task['run_id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if is_completed(run_path):
        return {**task, "status": "skipped_completed", "returncode": 0, "log_path": str(log_path)}
    command = [
        sys.executable,
        str(WORKER),
        "--input",
        str(task["input"]),
        "--output-root",
        str(output_root),
        "--method",
        str(task["method"]),
        "--seed-index",
        str(task["seed_index"]),
        "--sample-index",
        str(task["sample_index"]),
        "--seed-base",
        str(task["seed_base"]),
        "--downsampling-factor",
        str(task["downsampling_factor"]),
        "--temporal-obs-interval",
        str(task["temporal_obs_interval"]),
        "--blackout-start-step",
        str(task["blackout_start_step"]),
        "--steps",
        str(task["steps"]),
        "--device",
        str(task["device"]),
        "--record-trace",
    ]
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND: " + " ".join(command) + "\n\n")
        handle.flush()
        process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True)
    return {
        **task,
        "status": "completed_process" if process.returncode == 0 else "failed_process",
        "returncode": int(process.returncode),
        "runtime_seconds": float(time.perf_counter() - started),
        "run_json_exists": run_path.exists(),
        "run_json_completed": is_completed(run_path),
        "log_path": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Figure 5 KSE blackout forecast on all four GPUs.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--downsampling-factor", type=int, default=1)
    parser.add_argument("--temporal-obs-interval", type=int, default=1)
    parser.add_argument("--blackout-start-step", type=int, default=40)
    parser.add_argument("--steps", type=int, default=99)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--seed-base", type=int, default=2026081400)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--devices", default="cuda:2,cuda:3")
    parser.add_argument("--max-parallel", type=int, default=16)
    args = parser.parse_args()

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    methods = [item.strip() for item in str(args.methods).split(",") if item.strip()]
    devices = [item.strip() for item in str(args.devices).split(",") if item.strip()]
    if not methods or not devices:
        raise ValueError("At least one method and one device are required.")
    tasks: list[dict[str, Any]] = []
    for seed_index in range(int(args.seed_count)):
        sample_index = seed_index % 9
        for method_index, method in enumerate(methods):
            run_id = (
                f"kse_nmi{args.downsampling_factor}x_t{args.temporal_obs_interval}_"
                f"blackout{args.blackout_start_step}_{method}_seed{seed_index:02d}_sample{sample_index:02d}"
            )
            tasks.append(
                {
                    "run_id": run_id,
                    "input": str(args.input),
                    "output_root": str(output_root),
                    "method": method,
                    "seed_index": seed_index,
                    "sample_index": sample_index,
                    "seed_base": int(args.seed_base),
                    "downsampling_factor": int(args.downsampling_factor),
                    "temporal_obs_interval": int(args.temporal_obs_interval),
                    "blackout_start_step": int(args.blackout_start_step),
                    "steps": int(args.steps),
                    "device": devices[(seed_index * len(methods) + method_index) % len(devices)],
                }
            )

    manifest_path = output_root / "parallel_launcher_manifest.json"
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with futures.ThreadPoolExecutor(max_workers=max(1, int(args.max_parallel))) as executor:
        pending = {executor.submit(run_task, task): task for task in tasks}
        for future in futures.as_completed(pending):
            result = future.result()
            results.append(result)
            write_json(
                manifest_path,
                {
                    "status": "running",
                    "created_at_unix": time.time(),
                    "runtime_seconds_so_far": float(time.perf_counter() - started),
                    "output_root": str(output_root),
                    "total_tasks": len(tasks),
                    "finished_tasks": len(results),
                    "max_parallel": int(args.max_parallel),
                    "devices": devices,
                    "worker": str(WORKER),
                    "worker_sha256": file_sha256(WORKER),
                    "results": results,
                },
            )
            print(json.dumps(clean_json(result), ensure_ascii=False), flush=True)

    summary_command = [sys.executable, str(WORKER), "--summary-only", "--output-root", str(output_root)]
    summary_log = output_root / "logs_parallel" / "summary.log"
    with summary_log.open("w", encoding="utf-8") as handle:
        summary_returncode = subprocess.run(summary_command, stdout=handle, stderr=subprocess.STDOUT, text=True).returncode
    final = {
        "status": "completed",
        "created_at_unix": time.time(),
        "runtime_seconds": float(time.perf_counter() - started),
        "output_root": str(output_root),
        "total_tasks": len(tasks),
        "finished_tasks": len(results),
        "completed_or_skipped": sum(1 for row in results if int(row.get("returncode", 1)) == 0),
        "process_failures": sum(1 for row in results if int(row.get("returncode", 1)) != 0),
        "summary_returncode": int(summary_returncode),
        "max_parallel": int(args.max_parallel),
        "devices": devices,
        "worker": str(WORKER),
        "worker_sha256": file_sha256(WORKER),
        "launcher_sha256": file_sha256(Path(__file__).resolve()),
        "results": results,
    }
    write_json(manifest_path, final)
    print(json.dumps(clean_json(final), ensure_ascii=False))


if __name__ == "__main__":
    main()
