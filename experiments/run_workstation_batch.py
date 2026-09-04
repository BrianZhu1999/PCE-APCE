from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BatchJob:
    job_id: str
    arguments: tuple[str, ...]


def argument_value(arguments: tuple[str, ...], flag: str, default: str) -> str:
    try:
        index = arguments.index(flag)
    except ValueError:
        return default
    if index + 1 >= len(arguments):
        raise ValueError(f"Missing value after {flag}")
    return arguments[index + 1]


def run_directory(job: BatchJob, project_root: Path) -> Path:
    output_root = Path(argument_value(job.arguments, "--output-root", "results"))
    if not output_root.is_absolute():
        output_root = project_root / output_root
    return output_root / job.job_id


def classify_existing_run(path: Path) -> str:
    if not path.exists():
        return "new"
    provenance_path = path / "provenance.json"
    checkpoint_path = path / "checkpoint.pt"
    config_path = path / "config.json"
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("completed") is True:
            return "completed"
    if checkpoint_path.is_file() and config_path.is_file():
        return "resume"
    return "invalid"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an immutable assimilation manifest across GPUs")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--runner",
        type=Path,
        default=None,
        help="Experiment entry point; relative paths are resolved under project-root",
    )
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--status-directory", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--fail-on-completed",
        action="store_true",
        help="Treat pre-existing completed runs as errors instead of skipping them",
    )
    return parser.parse_args()


def resolve_runner(project_root: Path, requested: Path | None) -> Path:
    runner = requested or Path("experiments/run_assimilation.py")
    if not runner.is_absolute():
        runner = project_root / runner
    return runner.resolve()


def load_manifest(path: Path) -> list[BatchJob]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("jobs"), list):
        raise ValueError("Manifest must contain a jobs array")
    jobs = []
    identifiers: set[str] = set()
    for item in raw["jobs"]:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ValueError("Every job needs a string id")
        if item["id"] in identifiers:
            raise ValueError(f"Duplicate job id: {item['id']}")
        arguments = item.get("arguments")
        if not isinstance(arguments, list) or not all(isinstance(value, str) for value in arguments):
            raise ValueError(f"Job {item['id']} needs a string arguments array")
        identifiers.add(item["id"])
        jobs.append(BatchJob(item["id"], tuple(arguments)))
    return jobs


def write_status(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("At least one GPU id is required")
    jobs = load_manifest(args.manifest)
    runner = resolve_runner(args.project_root, args.runner)
    if not runner.is_file():
        raise FileNotFoundError(runner)
    args.status_directory.mkdir(parents=True, exist_ok=True)
    status_path = args.status_directory / "batch_status.json"
    queue = list(jobs)
    active: dict[str, tuple[BatchJob, subprocess.Popen[str], Path]] = {}
    finished: dict[str, dict[str, Any]] = {}
    started = time.time()

    while queue or active:
        available = [gpu for gpu in gpus if gpu not in active]
        while queue and available:
            gpu = available.pop(0)
            job = queue.pop(0)
            job_run_directory = run_directory(job, args.project_root)
            existing = classify_existing_run(job_run_directory)
            if existing == "completed" and not args.fail_on_completed:
                finished[job.job_id] = {
                    "status": "skipped_completed",
                    "run_directory": str(job_run_directory),
                }
                continue
            if existing == "invalid" or (existing == "completed" and args.fail_on_completed):
                finished[job.job_id] = {
                    "status": "invalid_existing_run",
                    "run_directory": str(job_run_directory),
                }
                continue
            command = [
                args.python_executable,
                str(runner),
                *job.arguments,
            ]
            if existing == "resume":
                command.extend(("--resume-run", str(job_run_directory)))
            else:
                command.extend(("--run-id", job.job_id))
            command.extend(("--device", "cpu"))
            log_path = args.status_directory / f"{job.job_id}.log"
            if args.dry_run:
                finished[job.job_id] = {"status": "dry_run", "gpu": gpu, "command": command}
                available.append(gpu)
                continue
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment["PYTHONPATH"] = str(args.project_root)
            log_handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=args.project_root,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active[gpu] = (job, process, log_handle)

        for gpu, (job, process, log_handle) in list(active.items()):
            exit_code = process.poll()
            if exit_code is None:
                continue
            log_handle.close()
            finished[job.job_id] = {
                "status": "completed" if exit_code == 0 else "failed",
                "gpu": gpu,
                "exit_code": exit_code,
                "log": str(log_path_for(args.status_directory, job.job_id)),
            }
            del active[gpu]

        write_status(
            status_path,
            {
                "manifest": str(args.manifest.resolve()),
                "started_unix": started,
                "updated_unix": time.time(),
                "pending": [job.job_id for job in queue],
                "active": {gpu: job.job_id for gpu, (job, _, _) in active.items()},
                "finished": finished,
            },
        )
        if queue or active:
            time.sleep(args.poll_seconds)

    failures = [
        job_id
        for job_id, value in finished.items()
        if value["status"] in {"failed", "invalid_existing_run"}
    ]
    if failures:
        raise SystemExit(f"Failed jobs: {', '.join(failures)}")


def log_path_for(status_directory: Path, job_id: str) -> Path:
    return status_directory / f"{job_id}.log"


if __name__ == "__main__":
    main()
