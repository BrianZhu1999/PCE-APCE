"""Run the paper VIV-PIV case-method-seed matrix on GPUs 2 and 3."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .common import load_config, write_json


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_seeds(value: str) -> list[int]:
    seeds: list[int] = []
    for token in parse_list(value):
        if "-" in token:
            start, stop = token.split("-", 1)
            seeds.extend(range(int(start), int(stop) + 1))
        else:
            seeds.append(int(token))
    return sorted(set(seeds))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the VIV-PIV reconstruction matrix.")
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--methods", default="pce,apce,aug_enkf,bma")
    parser.add_argument("--cases", default=None)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--gpus", default="2,3")
    parser.add_argument("--record-trace", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    methods = parse_list(args.methods)
    cases = parse_list(args.cases) if args.cases else list(config["test_cases"])
    seeds = parse_seeds(args.seeds)
    gpus = parse_list(args.gpus)
    unknown_methods = sorted(set(methods) - set(config["methods"]))
    unknown_cases = sorted(set(cases) - set(config["test_cases"]))
    if unknown_methods:
        raise ValueError(f"Unknown methods: {unknown_methods}")
    if unknown_cases:
        raise ValueError(f"Unknown held-out cases: {unknown_cases}")
    if not seeds:
        raise ValueError("At least one seed is required")
    if not gpus or any(gpu not in {"2", "3"} for gpu in gpus):
        raise ValueError("--gpus must select cuda:2, cuda:3, or both")

    variant = f"rank{int(config['rank'])}_stride1"
    run_root = pathlib.Path(config["output_root"]) / "runs" / variant
    log_root = run_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    tasks = [(case, method, seed) for seed in seeds for case in cases for method in methods]
    partitions = [tasks[index::len(gpus)] for index in range(len(gpus))]

    def worker(gpu: str, local_tasks: list[tuple[str, str, int]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for case, method, seed in local_tasks:
            run_id = f"viv_{case}_{method}_seed{seed:03d}"
            result_path = run_root / "runs" / f"{run_id}.json"
            if args.skip_existing and result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if result.get("status") == "completed" and result.get("valid"):
                    rows.append(
                        {"run_id": run_id, "gpu": gpu, "returncode": 0, "status": "completed"}
                    )
                    continue

            command = [
                sys.executable,
                "-m",
                "viv_piv.run_case",
                "--config",
                str(args.config),
                "--case",
                case,
                "--method",
                method,
                "--seed",
                str(seed),
                "--device",
                f"cuda:{gpu}",
            ]
            if args.record_trace:
                command.append("--record-trace")
            log_path = log_root / f"{run_id}.log"
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            result = (
                json.loads(result_path.read_text(encoding="utf-8"))
                if result_path.exists()
                else {}
            )
            rows.append(
                {
                    "run_id": run_id,
                    "gpu": gpu,
                    "returncode": completed.returncode,
                    "status": result.get("status", "missing_result"),
                    "valid": result.get("valid", False),
                    "log": str(log_path),
                }
            )
        return rows

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu, part) for gpu, part in zip(gpus, partitions)]
        rows = [row for future in futures for row in future.result()]

    failed = [
        row
        for row in rows
        if row.get("returncode") != 0
        or row.get("status") != "completed"
        or row.get("valid") is False
    ]
    summary = {
        "methods": methods,
        "cases": cases,
        "seeds": seeds,
        "gpus": gpus,
        "task_count": len(tasks),
        "failed_count": len(failed),
        "runs": rows,
    }
    summary_path = run_root / "matrix_summary.json"
    write_json(summary_path, summary)
    print(
        json.dumps(
            {
                "task_count": len(tasks),
                "failed_count": len(failed),
                "summary": str(summary_path),
            },
            ensure_ascii=False,
        )
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
