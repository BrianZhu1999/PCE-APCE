from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.run_workstation_batch import classify_existing_run


METHODS = ("hilda", "denkf", "letkf", "ensf", "iensf", "ensf_lr_ridge", "enff_f2p")
SYSTEMS = (
    "spring",
    "heat",
    "wave",
    "burgers",
    "allen_cahn",
    "navier_stokes",
    "navier_stokes_enff",
)


@dataclass(frozen=True)
class SmokeJob:
    job_id: str
    arguments: tuple[str, ...]


def build_smoke_jobs(output_root: Path) -> list[SmokeJob]:
    jobs = []
    for system in SYSTEMS:
        observation_count = 1 if system == "spring" else 4
        for method in METHODS:
            job_id = f"cpu_smoke_{system}_{method}"
            arguments = [
                "--system", system,
                "--method", method,
                "--seed", "2026080514",
                "--ensemble-size", "4",
                "--steps", "2",
                "--observation-interval", "1",
                "--observation-count", str(observation_count),
                "--observation-noise", "0.05",
                "--observation-transform", "atan",
                "--dtype", "float32",
                "--coverage-level", "0.9",
                "--energy-score-chunk-size", "16",
                "--checkpoint-interval", "1",
                "--output-root", str(output_root),
            ]
            if system == "navier_stokes_enff":
                arguments.extend(("--enff-grid-size", "8"))
            jobs.append(SmokeJob(job_id=job_id, arguments=tuple(arguments)))
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run every method-system CPU smoke")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    return parser.parse_args()


def write_status(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    runner = args.project_root / "experiments" / "run_assimilation.py"
    if not runner.is_file():
        raise FileNotFoundError(runner)
    jobs = build_smoke_jobs(args.output_root)
    finished: dict[str, dict[str, Any]] = {}
    failures = []
    log_root = args.status.parent / f"{args.status.stem}_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        run_directory = args.output_root / job.job_id
        existing = classify_existing_run(run_directory)
        if existing == "completed":
            finished[job.job_id] = {"status": "skipped_completed"}
            continue
        if existing == "invalid":
            finished[job.job_id] = {"status": "invalid_existing_run"}
            failures.append(job.job_id)
            continue
        command = [
            args.python_executable,
            str(runner),
            *job.arguments,
            "--device", "cpu",
        ]
        if existing == "resume":
            command.extend(("--resume-run", str(run_directory)))
        else:
            command.extend(("--run-id", job.job_id))
        log_path = log_root / f"{job.job_id}.log"
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.run(
                command,
                cwd=args.project_root,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        status = "completed" if process.returncode == 0 else "failed"
        finished[job.job_id] = {
            "status": status,
            "exit_code": process.returncode,
            "log": str(log_path),
        }
        if process.returncode != 0:
            failures.append(job.job_id)
        write_status(
            args.status,
            {
                "schema_version": 1,
                "expected_jobs": len(jobs),
                "finished": finished,
                "failures": failures,
            },
        )
    if failures:
        raise SystemExit(f"CPU compatibility smoke failures: {', '.join(failures)}")
    print(json.dumps({"completed": len(finished), "failures": 0}))


if __name__ == "__main__":
    main()
