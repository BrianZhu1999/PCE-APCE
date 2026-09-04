from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "paper_experiments" / "run_figure4_kse_nmi64_smoke_worker.py"
DEFAULT_INPUT = Path(r"<LOCAL_PATH>图4绘制\S3GM_NMI_2024\KSE_test.npy")
DEFAULT_OUTPUT = ROOT / "results" / "figure4_kse_nmi_official_sparse32_temporal_sweep_5seeds_20260814"
DEFAULT_METHODS = ("denkf", "letkf", "iensf", "aug_enkf", "bma_static", "pce", "apce")


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def run_command(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND: " + " ".join(cmd) + "\n\n")
        handle.flush()
        proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, text=True)
    return int(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KSE 32x spatial + temporal sparsity sweep.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--downsampling-factor", type=int, default=32)
    parser.add_argument("--temporal-intervals", default="2,4,6,8")
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--seed-base", type=int, default=2026081400)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--record-trace", action="store_true", default=True)
    parser.add_argument("--no-record-trace", action="store_false", dest="record_trace")
    args = parser.parse_args()

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    log_root = output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    temporal_intervals = [int(item) for item in args.temporal_intervals.split(",") if item.strip()]
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    seeds = list(range(int(args.seed_count)))

    launch_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for temporal_interval in temporal_intervals:
        for seed_index in seeds:
            sample_index = int(seed_index)
            for method in methods:
                run_id = f"kse_nmi{args.downsampling_factor}x_t{temporal_interval}_{method}_seed{seed_index:02d}_sample{sample_index:02d}"
                log_path = log_root / f"{run_id}.log"
                cmd = [
                    sys.executable,
                    str(WORKER),
                    "--input",
                    str(args.input),
                    "--output-root",
                    str(output_root),
                    "--method",
                    method,
                    "--seed-index",
                    str(seed_index),
                    "--sample-index",
                    str(sample_index),
                    "--seed-base",
                    str(args.seed_base),
                    "--downsampling-factor",
                    str(args.downsampling_factor),
                    "--temporal-obs-interval",
                    str(temporal_interval),
                    "--device",
                    str(args.device),
                    "--record-trace",
                    "--write-summary",
                ]
                returncode = run_command(cmd, log_path)
                launch_rows.append(
                    {
                        "run_id": run_id,
                        "method": method,
                        "seed_index": seed_index,
                        "sample_index": sample_index,
                        "temporal_obs_interval": temporal_interval,
                        "downsampling_factor": int(args.downsampling_factor),
                        "returncode": returncode,
                        "log_path": str(log_path),
                    }
                )

    summary_cmd = [
        sys.executable,
        str(WORKER),
        "--summary-only",
        "--output-root",
        str(output_root),
    ]
    summary_log = log_root / "summary.log"
    summary_returncode = run_command(summary_cmd, summary_log)

    manifest = {
        "created_at_unix": time.time(),
        "runtime_seconds": time.perf_counter() - started,
        "input_path": str(args.input),
        "output_root": str(output_root),
        "downsampling_factor": int(args.downsampling_factor),
        "temporal_intervals": temporal_intervals,
        "seed_count": int(args.seed_count),
        "seed_base": int(args.seed_base),
        "device": str(args.device),
        "methods": methods,
        "record_trace": bool(args.record_trace),
        "summary_returncode": summary_returncode,
        "run_rows": launch_rows,
        "failure_count": sum(1 for row in launch_rows if int(row["returncode"]) != 0),
        "success_count": sum(1 for row in launch_rows if int(row["returncode"]) == 0),
    }
    manifest_path = output_root / "launcher_manifest.json"
    manifest_path.write_text(json.dumps(clean_json(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(clean_json(manifest), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
