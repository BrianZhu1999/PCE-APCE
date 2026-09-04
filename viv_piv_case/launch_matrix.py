"""Launch a paired VIV-PIV method/case/seed matrix across fixed GPUs."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .common import load_config, write_json


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_seeds(value: str) -> list[int]:
    output: list[int] = []
    for token in parse_list(value):
        if "-" in token:
            start, stop = token.split("-", 1)
            output.extend(range(int(start), int(stop) + 1))
        else:
            output.append(int(token))
    return sorted(set(output))


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch VIV-PIV paired method matrix.")
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--methods", required=True, help="Comma-separated method ids")
    parser.add_argument("--cases", default=None, help="Comma-separated external case ids")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--record-trace", action="store_true")
    parser.add_argument("--skip-full-field", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-control", action="store_true")
    parser.add_argument("--sensor-density", type=int, default=None)
    parser.add_argument("--sensor-layout", default=None)
    parser.add_argument("--observation-covariance", choices=("diagonal", "full", "tapered"), default="diagonal")
    parser.add_argument("--covariance-shrinkage", type=float, default=None)
    parser.add_argument("--ensemble-size", type=int, default=None)
    parser.add_argument("--evidence-window-frames", type=int, default=None)
    parser.add_argument("--variant", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    methods = parse_list(args.methods)
    cases = parse_list(args.cases) if args.cases else list(config["test_cases"])
    seeds = parse_seeds(args.seeds)
    gpus = parse_list(args.gpus)
    if not gpus:
        raise ValueError("At least one GPU id is required")
    if args.ensemble_size is not None and args.ensemble_size < 2:
        raise ValueError("--ensemble-size must be at least two")
    if args.covariance_shrinkage is not None and not 0.0 <= args.covariance_shrinkage <= 1.0:
        raise ValueError("--covariance-shrinkage must lie in [0, 1]")
    if args.evidence_window_frames is not None and args.evidence_window_frames < 1:
        raise ValueError("--evidence-window-frames must be at least one")
    variant = args.variant or f"rank{int(config['rank'])}_stride1"
    run_root = pathlib.Path(config["output_root"]) / "runs" / variant
    log_root = run_root / "logs" / "matrix"
    log_root.mkdir(parents=True, exist_ok=True)
    tasks = [(case, method, seed) for seed in seeds for case in cases for method in methods]
    partitions = [tasks[index::len(gpus)] for index in range(len(gpus))]

    def worker(gpu: str, local_tasks: list[tuple[str, str, int]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for case, method, seed in local_tasks:
            density_suffix = f"_sensors{args.sensor_density:03d}" if args.sensor_density is not None else ""
            layout_suffix = f"_layout{args.sensor_layout}" if args.sensor_layout is not None else ""
            ensemble_suffix = f"_ens{args.ensemble_size:03d}" if args.ensemble_size is not None else ""
            covariance_suffix = "_covfull" if args.observation_covariance == "full" else ("_covtaper" if args.observation_covariance == "tapered" else "")
            shrinkage_suffix = f"_shr{int(round(1000.0 * args.covariance_shrinkage)):03d}" if args.covariance_shrinkage is not None else ""
            evidence_window_suffix = f"_ew{args.evidence_window_frames:02d}" if args.evidence_window_frames is not None else ""
            run_id = f"viv_{case}_{method}_seed{seed:03d}" + density_suffix + layout_suffix + ensemble_suffix + covariance_suffix + shrinkage_suffix + evidence_window_suffix + ("_no_control" if args.no_control else "")
            result_path = run_root / "runs" / f"{run_id}.json"
            if args.skip_existing and result_path.exists():
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                if payload.get("status") == "completed" and payload.get("valid"):
                    rows.append({"run_id": run_id, "gpu": gpu, "returncode": 0, "status": "skipped_completed"})
                    continue
            command = [
                sys.executable,
                "-m",
                "viv_piv_case.run_case",
                "--config",
                str(args.config),
                "--case",
                case,
                "--method",
                method,
                "--seed",
                str(seed),
                "--model-variant",
                variant,
                "--device",
                f"cuda:{gpu}",
            ]
            if args.record_trace:
                command.append("--record-trace")
            if args.skip_full_field:
                command.append("--skip-full-field")
            if args.no_control:
                command.append("--no-control")
            if args.sensor_density is not None:
                command.extend(["--sensor-density", str(args.sensor_density)])
            if args.sensor_layout is not None:
                command.extend(["--sensor-layout", args.sensor_layout])
            if args.ensemble_size is not None:
                command.extend(["--ensemble-size", str(args.ensemble_size)])
            if args.covariance_shrinkage is not None:
                command.extend(["--covariance-shrinkage", str(args.covariance_shrinkage)])
            if args.evidence_window_frames is not None:
                command.extend(["--evidence-window-frames", str(args.evidence_window_frames)])
            command.extend(["--observation-covariance", args.observation_covariance])
            environment = os.environ.copy()
            log_path = log_root / f"{run_id}.log"
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment, check=False)
            payload = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
            rows.append({
                "run_id": run_id,
                "gpu": gpu,
                "returncode": completed.returncode,
                "status": payload.get("status", "missing_result"),
                "valid": payload.get("valid", False),
                "log": str(log_path),
            })
        return rows

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu, part) for gpu, part in zip(gpus, partitions)]
        rows = [row for future in futures for row in future.result()]
    failed = [row for row in rows if row.get("returncode") != 0 or row.get("status") not in {"completed", "skipped_completed"} or row.get("valid") is False]
    manifest = {
        "variant": variant,
        "methods": methods,
        "cases": cases,
        "seeds": seeds,
        "gpus": gpus,
        "no_control": args.no_control,
        "sensor_density": args.sensor_density,
        "sensor_layout": args.sensor_layout,
        "observation_covariance": args.observation_covariance,
        "observation_covariance_shrinkage": args.covariance_shrinkage,
        "ensemble_size": args.ensemble_size,
        "evidence_window_frames": args.evidence_window_frames,
        "task_count": len(tasks),
        "failed_count": len(failed),
        "rows": rows,
    }
    protocol_suffix = (f"_sensors{args.sensor_density:03d}" if args.sensor_density is not None else "")
    protocol_suffix += f"_layout{args.sensor_layout}" if args.sensor_layout is not None else ""
    protocol_suffix += f"_ens{args.ensemble_size:03d}" if args.ensemble_size is not None else ""
    protocol_suffix += "_covfull" if args.observation_covariance == "full" else ("_covtaper" if args.observation_covariance == "tapered" else "")
    protocol_suffix += f"_shr{int(round(1000.0 * args.covariance_shrinkage)):03d}" if args.covariance_shrinkage is not None else ""
    protocol_suffix += f"_ew{args.evidence_window_frames:02d}" if args.evidence_window_frames is not None else ""
    protocol_suffix += "_no_control" if args.no_control else ""
    name = f"matrix_{'-'.join(methods)}_seeds{min(seeds):03d}-{max(seeds):03d}{protocol_suffix}.json"
    write_json(run_root / "manifests" / name, manifest)
    print(json.dumps({"task_count": len(tasks), "failed_count": len(failed), "manifest": str(run_root / 'manifests' / name)}, ensure_ascii=False))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
