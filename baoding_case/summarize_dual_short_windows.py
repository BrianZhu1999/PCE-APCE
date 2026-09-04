#!/usr/bin/env python3
"""Summarize short dual-source window replays against GPS offline truth."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


SEEDS = (2026082601, 2026082602, 2026082603, 2026082604, 2026082605)


def rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def points(path: Path, names=("px", "py", "pz")) -> tuple[np.ndarray, np.ndarray]:
    data = rows(path)
    times = np.asarray([int(round(float(row["time_s"]))) for row in data])
    xyz = np.asarray([[float(row[name]) for name in names] for row in data], dtype=float)
    return times, xyz


def unwrap_sweep(xyz: np.ndarray) -> float:
    xy = xyz[:, :2]
    design = np.column_stack((2.0 * xy[:, 0], 2.0 * xy[:, 1], np.ones(len(xy))))
    rhs = np.sum(np.square(xy), axis=1)
    center_x, center_y, _ = np.linalg.lstsq(design, rhs, rcond=None)[0]
    center = np.asarray([center_x, center_y])
    phase = np.unwrap(np.arctan2(xyz[:, 1] - center[1], xyz[:, 0] - center[0]))
    return float(np.degrees(abs(phase[-1] - phase[0])))


def aggregate_target(root: Path, window: str, target: int, method: str) -> dict[str, float | int]:
    frontend = root / window / "frontend" / f"target{target}" / "frontend"
    truth_times, truth = points(frontend / "gps_truth.csv")
    run_dir = root / window / "formal_matrix" / f"target{target}" / "runs"
    estimates = []
    widths = []
    for seed in SEEDS:
        payload = json.loads((run_dir / f"{method}_seed_{seed}.json").read_text(encoding="utf-8"))
        records = payload["records"]
        run_times = np.asarray([int(round(float(row["time_s"]))) for row in records])
        if not np.array_equal(run_times, truth_times):
            raise RuntimeError(f"timeline mismatch: {window} target {target} {seed}")
        estimates.append(np.asarray([[float(row[k]) for k in ("px", "py", "pz")] for row in records]))
        widths.append(np.asarray([float(row["interval_width_m"]) for row in records]))
    estimates = np.asarray(estimates)
    median_estimate = np.median(estimates, axis=0)
    errors = np.linalg.norm(median_estimate - truth, axis=1)
    steps = np.linalg.norm(np.diff(median_estimate, axis=0), axis=1)
    return {
        "frames": int(len(truth_times)),
        "duration_s": int(truth_times[-1] - truth_times[0]),
        "start_time_s": int(truth_times[0]),
        "end_time_s": int(truth_times[-1]),
        "gps_sweep_deg": unwrap_sweep(truth),
        "rmse_m": float(np.sqrt(np.mean(errors**2))),
        "median_error_m": float(np.median(errors)),
        "p90_error_m": float(np.percentile(errors, 90.0)),
        "max_error_m": float(np.max(errors)),
        "max_step_m": float(np.max(steps)),
        "median_interval_width_m": float(np.median(np.median(np.asarray(widths), axis=0))),
        "mean_interval_width_m": float(np.mean(widths)),
        "seed_rmse_m": [float(np.sqrt(np.mean(np.linalg.norm(run - truth, axis=1) ** 2))) for run in estimates],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", nargs="+", default=("window50", "window60", "window70"))
    args = parser.parse_args()
    output: dict[str, object] = {
        "gps_role": "offline truth scoring only; no GPS in window selection or state update",
        "method": "APCE median over five seeds",
        "windows": {},
    }
    for window in args.windows:
        target_metrics = {str(target): aggregate_target(args.root, window, target, "apce") for target in (1, 2)}
        median_estimates: dict[int, np.ndarray] = {}
        truth_positions: dict[int, np.ndarray] = {}
        for target in (1, 2):
            frontend = args.root / window / "frontend" / f"target{target}" / "frontend"
            _, truth_positions[target] = points(frontend / "gps_truth.csv")
            run_dir = args.root / window / "formal_matrix" / f"target{target}" / "runs"
            estimates = []
            for seed in SEEDS:
                payload = json.loads((run_dir / f"apce_seed_{seed}.json").read_text(encoding="utf-8"))
                estimates.append(np.asarray([
                    [float(row[key]) for key in ("px", "py", "pz")]
                    for row in payload["records"]
                ]))
            median_estimates[target] = np.median(np.asarray(estimates), axis=0)
        direct = (
            np.linalg.norm(median_estimates[1] - truth_positions[1], axis=1)
            + np.linalg.norm(median_estimates[2] - truth_positions[2], axis=1)
        )
        swapped = (
            np.linalg.norm(median_estimates[1] - truth_positions[2], axis=1)
            + np.linalg.norm(median_estimates[2] - truth_positions[1], axis=1)
        )
        output["windows"][window] = {
            "targets": target_metrics,
            "mean_rmse_m": float(np.mean([target_metrics[str(target)]["rmse_m"] for target in (1, 2)])),
            "worst_rmse_m": float(np.max([target_metrics[str(target)]["rmse_m"] for target in (1, 2)])),
            "max_step_m": float(np.max([target_metrics[str(target)]["max_step_m"] for target in (1, 2)])),
            "minimum_gps_sweep_deg": float(np.min([target_metrics[str(target)]["gps_sweep_deg"] for target in (1, 2)])),
            "maximum_gps_sweep_deg": float(np.max([target_metrics[str(target)]["gps_sweep_deg"] for target in (1, 2)])),
            "identity_match_fraction": float(np.mean(direct <= swapped)),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
