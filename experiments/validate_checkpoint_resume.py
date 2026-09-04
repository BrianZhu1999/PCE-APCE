from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate externally interrupted checkpoint resume")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--minimum-checkpoint-step", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    return parser.parse_args()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    runner = args.project_root / "experiments" / "run_assimilation.py"
    interrupted_directory = args.output_root / "interrupted"
    uninterrupted_directory = args.output_root / "uninterrupted"
    if interrupted_directory.exists() or uninterrupted_directory.exists():
        raise FileExistsError("Validation output directories must not already exist")
    args.output_root.mkdir(parents=True, exist_ok=True)

    common = [
        sys.executable,
        str(runner),
        "--system", "spring",
        "--method", "hilda",
        "--seed", "2026080507",
        "--ensemble-size", "8",
        "--steps", "50",
        "--observation-interval", "1",
        "--observation-count", "1",
        "--observation-noise", "0.05",
        "--checkpoint-interval", "1",
        "--device", args.device,
        "--output-root", str(args.output_root),
    ]
    interrupted_command = [*common, "--run-id", "interrupted"]
    interrupted_log_path = args.output_root / "interrupted_process.log"
    with interrupted_log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            interrupted_command,
            cwd=args.project_root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        checkpoint_path = interrupted_directory / "checkpoint.pt"
        deadline = time.monotonic() + args.timeout_seconds
        checkpoint_step = 0
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Interrupted candidate completed before external termination")
            if checkpoint_path.is_file():
                try:
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                    checkpoint_step = int(checkpoint["next_step"])
                except (EOFError, OSError, RuntimeError):
                    checkpoint_step = 0
                if checkpoint_step >= args.minimum_checkpoint_step:
                    break
            time.sleep(0.05)
        else:
            process.terminate()
            process.wait(timeout=10)
            raise TimeoutError("No usable checkpoint appeared before the validation timeout")
        process.terminate()
        interrupted_exit_code = process.wait(timeout=10)
    if interrupted_exit_code == 0:
        raise RuntimeError("External termination unexpectedly returned success")

    resume_command = [*common, "--resume-run", str(interrupted_directory)]
    subprocess.run(resume_command, cwd=args.project_root, check=True)
    uninterrupted_command = [*common, "--run-id", "uninterrupted"]
    subprocess.run(uninterrupted_command, cwd=args.project_root, check=True)

    resumed = torch.load(interrupted_directory / "trajectories.pt", map_location="cpu", weights_only=False)
    uninterrupted = torch.load(
        uninterrupted_directory / "trajectories.pt", map_location="cpu", weights_only=False
    )
    comparisons = {
        key: bool(torch.equal(resumed[key], uninterrupted[key]))
        for key in ("truth", "estimate", "alpha_estimate")
    }
    resumed_metrics = json.loads(
        (interrupted_directory / "metrics.json").read_text(encoding="utf-8")
    )
    uninterrupted_metrics = json.loads(
        (uninterrupted_directory / "metrics.json").read_text(encoding="utf-8")
    )
    metrics_exact_match = resumed_metrics == uninterrupted_metrics
    interrupted_provenance = json.loads(
        (interrupted_directory / "provenance.json").read_text(encoding="utf-8")
    )
    report = {
        "schema_version": 1,
        "checkpoint_step_before_termination": checkpoint_step,
        "interrupted_exit_code": interrupted_exit_code,
        "resume_count": interrupted_provenance["resume_count"],
        "resumed_from_step": interrupted_provenance["resumed_from_step"],
        "tensor_exact_match": comparisons,
        "metrics_exact_match": metrics_exact_match,
        "passed": (
            all(comparisons.values())
            and metrics_exact_match
            and interrupted_provenance["resume_count"] == 1
        ),
    }
    atomic_json(args.output_root / "validation_report.json", report)
    if not report["passed"]:
        raise AssertionError(report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
