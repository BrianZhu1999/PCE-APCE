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
WORKER = ROOT / "paper_experiments" / "run_figure4_kse_nmi64_smoke_worker.py"
DEFAULT_INPUT = Path("<HILDA_RESULTS_ROOT>/external/S3GM_NMI_2024/KSE_test.npy")
DEFAULT_OUTPUT = Path(
    "<HILDA_RESULTS_ROOT>/results/figure4_kse_spectral32_t2_core4_smoke_5seeds_20260814_4gpu"
)
DEFAULT_METHODS = ("aug_enkf", "bma_static", "pce", "apce")


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


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
    cmd = [
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
        "--obs-geometry",
        str(task["obs_geometry"]),
        "--spectral-mode-count",
        str(task["spectral_mode_count"]),
        "--device",
        str(task["device"]),
        "--record-trace",
    ]
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
        "run_json_exists": run_path.exists(),
        "run_json_completed": is_completed(run_path),
        "log_path": str(log_path),
    }


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel 5-seed KSE spectral-observation smoke.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--downsampling-factor", type=int, default=32)
    parser.add_argument("--temporal-obs-interval", type=int, default=2)
    parser.add_argument("--spectral-mode-count", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--seed-base", type=int, default=2026081400)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--devices", default="cuda:2,cuda:3")
    parser.add_argument("--max-parallel", type=int, default=16)
    args = parser.parse_args()

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    devices = [x.strip() for x in args.devices.split(",") if x.strip()]
    if not devices:
        raise ValueError("--devices must contain at least one device")

    observed_points = 1024 // int(args.downsampling_factor)
    mode_count = int(args.spectral_mode_count) if int(args.spectral_mode_count) > 0 else observed_points // 2 + 1
    tasks: list[dict[str, Any]] = []
    for seed_index in range(int(args.seed_count)):
        sample_index = int(seed_index % 9)
        for method in methods:
            run_id = (
                f"kse_nmi{args.downsampling_factor}x_t{args.temporal_obs_interval}_spectral_"
                f"{method}_seed{seed_index:02d}_sample{sample_index:02d}"
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
                    "obs_geometry": "spectral",
                    "spectral_mode_count": int(mode_count),
                    "device": devices[len(tasks) % len(devices)],
                }
            )

    manifest_path = output_root / "parallel_launcher_manifest.json"
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
                "output_root": str(output_root),
                "total_tasks": len(tasks),
                "finished_tasks": len(results),
                "max_parallel": int(args.max_parallel),
                "devices": devices,
                "results": results,
            }
            write_manifest(manifest_path, partial)
            print(json.dumps(clean_json(result), ensure_ascii=False), flush=True)

    summary_cmd = [sys.executable, str(WORKER), "--summary-only", "--output-root", str(output_root)]
    summary_log = output_root / "logs_parallel" / "summary.log"
    with summary_log.open("w", encoding="utf-8") as handle:
        summary_returncode = subprocess.run(summary_cmd, stdout=handle, stderr=subprocess.STDOUT, text=True).returncode

    final = {
        "created_at_unix": time.time(),
        "runtime_seconds": time.perf_counter() - started,
        "output_root": str(output_root),
        "total_tasks": len(tasks),
        "finished_tasks": len(results),
        "completed_or_skipped": sum(1 for row in results if int(row.get("returncode", 1)) == 0),
        "process_failures": sum(1 for row in results if int(row.get("returncode", 1)) != 0),
        "summary_returncode": int(summary_returncode),
        "max_parallel": int(args.max_parallel),
        "devices": devices,
        "methods": methods,
        "obs_geometry": "spectral",
        "downsampling_factor": int(args.downsampling_factor),
        "temporal_obs_interval": int(args.temporal_obs_interval),
        "observed_spatial_points": int(observed_points),
        "spectral_mode_count": int(mode_count),
        "results": results,
    }
    write_manifest(manifest_path, final)
    print(json.dumps(clean_json(final), ensure_ascii=False))


if __name__ == "__main__":
    main()
