from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from run_figure2_reviewer_gate import LABELS, run_case_method_seed


FORMAL_PROTOCOL = "figure2-reviewer-gate-50paired-seeds-20260810-v2-local-mixture"


def write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def parse_tasks(path: Path) -> list[tuple[str, str, int]]:
    tasks: list[tuple[str, str, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        case, method, seed_text = [item.strip() for item in line.split(",")]
        tasks.append((case, method, int(seed_text)))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable Figure 2 V2 formal worker.")
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--protocol", default=FORMAL_PROTOCOL)
    args = parser.parse_args()

    device = torch.device(args.device)
    tasks = parse_tasks(args.task_file)
    completed = 0
    failed = 0
    started = time.perf_counter()
    for index, (case, method, seed) in enumerate(tasks, start=1):
        out_path = args.output / case / method / f"seed_{seed}.json"
        if out_path.exists():
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
                if existing.get("status") == "completed" and existing.get("valid", False):
                    completed += 1
                    print(f"SKIP {index}/{len(tasks)} {case} {method} {seed}", flush=True)
                    continue
            except Exception:
                pass

        print(f"RUN {index}/{len(tasks)} {case} {method} {seed}", flush=True)
        job_started = time.perf_counter()
        try:
            row = run_case_method_seed(case, method, seed, device, protocol=args.protocol)
            row["worker_device"] = str(device)
            row["worker_elapsed_seconds"] = float(time.perf_counter() - job_started)
            write_atomic(out_path, row)
            completed += 1
            print(
                f"OK {case} {method} {seed} nrmse={row.get('nrmse')} crps={row.get('crps')}",
                flush=True,
            )
        except Exception as exc:
            failed += 1
            failure = {
                "case": case,
                "method": method,
                "label": LABELS.get(method, method),
                "seed": seed,
                "status": "failed",
                "valid": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "worker_device": str(device),
                "worker_elapsed_seconds": float(time.perf_counter() - job_started),
            }
            write_atomic(out_path, failure)
            print(f"FAILED {case} {method} {seed}: {type(exc).__name__}: {exc}", flush=True)

    status = {
        "status": "completed" if failed == 0 else "completed_with_failures",
        "tasks": len(tasks),
        "completed": completed,
        "failed": failed,
        "elapsed_seconds": float(time.perf_counter() - started),
        "device": str(device),
        "protocol": args.protocol,
    }
    write_atomic(args.output / "worker_status.json", status)
    print(json.dumps(status, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
