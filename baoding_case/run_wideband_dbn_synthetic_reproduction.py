#!/usr/bin/env python3
"""Run the upstream wideband DBN synthetic example without interactive plots.

The algorithm and update order follow the public MTT_WB_DBN ``main.py``.
This runner only removes plotting, fixes the random seed, and emits auditable
trajectory and metric files for modern Python/NumPy environments.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np


UPSTREAM_COMMIT = "8d05b864246f226b28b6baec0958deaefad41046"
SOURCE_FILES = ("main.py", "Tracking.py", "SysSetting.py", "WavGen.py", "README.md")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_finite(states: list[np.ndarray], frame: int, stage: str) -> None:
    for target, state in enumerate(states, 1):
        if not np.isfinite(np.asarray(state, dtype=float)).all():
            raise FloatingPointError(
                f"non-finite state at frame={frame}, target={target}, stage={stage}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--outer-iterations", type=int, default=5)
    parser.add_argument("--newton-iterations", type=int, default=4)
    parser.add_argument("--motion-noise-sigma", type=float, default=10.0)
    args = parser.parse_args()

    code_root = args.code_root.resolve()
    missing = [name for name in SOURCE_FILES if not (code_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing upstream files: {missing}")
    sys.path.insert(0, str(code_root))

    np.random.seed(args.seed)
    wav_module = importlib.import_module("WavGen")
    tracking_module = importlib.import_module("Tracking")

    wave = wav_module.WavGen()
    wave.WavTraGen()
    settings = wave.SysSet
    truth = np.asarray(wave.xk, dtype=float).reshape(
        settings.M, settings.BinNum, 4
    )

    tracker = tracking_module.DbnTracking(sigma=args.motion_noise_sigma)
    tracker.Niter = args.newton_iterations
    records: list[dict[str, float | int]] = []
    frame_ospa: list[float] = []

    with np.errstate(over="raise", divide="raise", invalid="raise"):
        for frame, data in enumerate(wave.WanSignals):
            tracker.input(data)
            tracker.Predictxk()
            require_finite(tracker.predictedxk, frame, "prediction")
            for iteration in range(args.outer_iterations):
                tracker.UpdateLambda_0()
                tracker.UpdateLambda()
                tracker.UpdateMu()
                tracker.UpdateSk()
                tracker.Updatexk()
                require_finite(tracker.xk, frame, f"outer_iteration_{iteration + 1}")

            squared_errors = []
            for target in range(settings.M):
                state = np.asarray(tracker.xk[target], dtype=float).reshape(4)
                true_state = truth[target, frame]
                error = float(np.linalg.norm(state[:2] - true_state[:2]))
                squared_errors.append(error * error)
                records.append(
                    {
                        "frame_index": frame,
                        "time_s": frame * float(settings.dt),
                        "target": target + 1,
                        "estimated_x": float(state[0]),
                        "estimated_y": float(state[1]),
                        "estimated_vx": float(state[2]),
                        "estimated_vy": float(state[3]),
                        "truth_x": float(true_state[0]),
                        "truth_y": float(true_state[1]),
                        "truth_vx": float(true_state[2]),
                        "truth_vy": float(true_state[3]),
                        "position_error_m": error,
                    }
                )
            frame_ospa.append(math.sqrt(sum(squared_errors) / len(squared_errors)))

    args.output_root.mkdir(parents=True, exist_ok=True)
    trajectory_csv = args.output_root / "synthetic_trajectory.csv"
    with trajectory_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    target_metrics = {}
    for target in range(1, settings.M + 1):
        errors = [
            float(row["position_error_m"])
            for row in records
            if row["target"] == target
        ]
        target_metrics[str(target)] = {
            "frames": len(errors),
            "rmse_m": math.sqrt(sum(value * value for value in errors) / len(errors)),
            "mean_error_m": sum(errors) / len(errors),
            "max_error_m": max(errors),
        }

    manifest = {
        "claim_status": "upstream_synthetic_reproduction",
        "paper": "Zhang et al., IEEE IoT Journal 9(6), 2022",
        "doi": "10.1109/JIOT.2021.3108528",
        "upstream_repository": "https://github.com/zhangwenqiong2017/MTT_WB_DBN",
        "upstream_commit": UPSTREAM_COMMIT,
        "source_sha256": {
            name: sha256(code_root / name) for name in SOURCE_FILES
        },
        "runner_role": "noninteractive transcription of upstream main.py",
        "compatibility_changes": [
            "interactive plotting omitted",
            "random seed fixed and recorded",
            "trajectory and metrics exported",
        ],
        "seed": args.seed,
        "outer_iterations": args.outer_iterations,
        "newton_iterations": args.newton_iterations,
        "motion_noise_sigma": args.motion_noise_sigma,
        "frame_count": int(settings.BinNum),
        "target_count": int(settings.M),
        "node_count": int(settings.N),
        "sensor_count": int(settings.P),
        "sample_rate_hz": float(settings.fs),
        "snapshot_length": int(settings.SnapshotLen),
        "frame_dt_s": float(settings.dt),
        "frequencies_hz": [float(value) for value in settings.fr],
        "target_metrics": target_metrics,
        "mean_ospa_order2_m": sum(frame_ospa) / len(frame_ospa),
        "max_ospa_order2_m": max(frame_ospa),
        "trajectory_csv": str(trajectory_csv),
    }
    metrics_path = args.output_root / "synthetic_metrics.json"
    metrics_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
