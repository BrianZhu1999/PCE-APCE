from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def load_manifest(path: Path) -> list[dict[str, object]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    jobs = raw.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("manifest.jobs must be a list")
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--status-directory", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    jobs = load_manifest(args.manifest)
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("at least one GPU is required")
    args.status_directory.mkdir(parents=True, exist_ok=True)
    pending = list(jobs)
    active: dict[str, tuple[dict[str, object], subprocess.Popen[str], object, str]] = {}
    finished: dict[str, dict[str, object]] = {}
    status_path = args.status_directory / "batch_status.json"
    started = time.time()

    while pending or active:
        free = [gpu for gpu in gpus if gpu not in active]
        while pending and free:
            gpu = free.pop(0)
            job = pending.pop(0)
            job_id = str(job["id"])
            arguments = [str(value) for value in job["arguments"]]  # type: ignore[index]
            output_arg = arguments[arguments.index("--output") + 1]
            output_path = Path(output_arg)
            if output_path.is_file() and not args.force:
                try:
                    existing = json.loads(output_path.read_text(encoding="utf-8"))
                    finished[job_id] = {
                        "status": "skipped_completed",
                        "output": output_arg,
                        "valid": existing.get("valid"),
                    }
                except json.JSONDecodeError:
                    finished[job_id] = {"status": "invalid_existing_output", "output": output_arg}
                free.append(gpu)
                continue
            log_path = args.status_directory / f"{job_id}.log"
            log_handle = log_path.open("w", encoding="utf-8")
            command = [
                sys.executable,
                str(args.runner),
                *arguments,
                "--device",
                "cuda:2",
            ]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment["PYTHONPATH"] = str(args.project_root)
            process = subprocess.Popen(
                command,
                cwd=args.project_root,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active[gpu] = (job, process, log_handle, str(log_path))

        for gpu, (job, process, log_handle, log_path) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log_handle.close()  # type: ignore[union-attr]
            job_id = str(job["id"])
            arguments = [str(value) for value in job["arguments"]]  # type: ignore[index]
            output_arg = arguments[arguments.index("--output") + 1]
            output_path = Path(output_arg)
            finished[job_id] = {
                "status": "completed" if code == 0 else "failed",
                "exit_code": code,
                "output": output_arg,
                "log": log_path,
                "valid": (
                    json.loads(output_path.read_text(encoding="utf-8")).get("valid")
                    if code == 0 and output_path.is_file()
                    else None
                ),
            }
            del active[gpu]

        status_path.write_text(
            json.dumps(
                {
                    "manifest": str(args.manifest),
                    "runner": str(args.runner),
                    "started_unix": started,
                    "updated_unix": time.time(),
                    "pending": [str(job["id"]) for job in pending],
                    "active": {gpu: str(job["id"]) for gpu, (job, _, _, _) in active.items()},
                    "finished": finished,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        if pending or active:
            time.sleep(args.poll_seconds)

    failures = [
        job_id
        for job_id, value in finished.items()
        if value.get("status") in {"failed", "invalid_existing_output"}
    ]
    if failures:
        raise SystemExit("failed jobs: " + ", ".join(failures))


if __name__ == "__main__":
    main()
