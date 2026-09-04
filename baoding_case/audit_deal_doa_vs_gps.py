#!/usr/bin/env python3
"""Offline audit of supplied Baoding measured DOA files against GPS DOA.

The GPS stream is used only as an offline scoring reference.  Per-frame target
identity is assigned by the minimum joint azimuth/elevation residual, matching
the role of the supplied ``lianhe_doa_0.m`` without discarding outliers.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def circular_error_deg(value: float, reference: float) -> float:
    return (value - reference + 180.0) % 360.0 - 180.0


def load_table(path: Path) -> np.ndarray:
    rows = [[float(value) for value in line.split()]
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip()]
    if not rows:
        raise RuntimeError(f"empty table: {path}")
    return np.asarray(rows, dtype=np.float64)


def interpolate_truth(measured_times: np.ndarray, gps: np.ndarray) -> np.ndarray:
    order = np.argsort(gps[:, -1], kind="stable")
    times = gps[order, -1]
    values = gps[order]
    unique, indices = np.unique(times, return_index=True)
    values = values[indices]
    return np.column_stack([np.interp(measured_times, unique, values[:, col])
                            for col in (1, 2, 3, 4, 5, 6)])


def audit_node(measured: np.ndarray, gps: np.ndarray) -> dict:
    truth = interpolate_truth(measured[:, -1], gps)
    frame_errors = []
    for index in range(len(measured)):
        measured_az = measured[index, [1, 3, 5]]
        measured_el = measured[index, [2, 4, 6]]
        truth_az = truth[index, [0, 2, 4]]
        truth_el = truth[index, [1, 3, 5]]
        cost = np.zeros((3, 3), dtype=np.float64)
        for row in range(3):
            for column in range(3):
                cost[row, column] = math.hypot(
                    circular_error_deg(measured_az[row], truth_az[column]),
                    measured_el[row] - truth_el[column],
                )
        rows, columns = linear_sum_assignment(cost)
        azimuth = np.asarray([circular_error_deg(measured_az[row], truth_az[column])
                              for row, column in zip(rows, columns)])
        elevation = np.asarray([measured_el[row] - truth_el[column]
                                for row, column in zip(rows, columns)])
        frame_errors.append(np.concatenate((azimuth, elevation, np.hypot(azimuth, elevation))))
    errors = np.asarray(frame_errors)
    joint = errors[:, 6:9]
    return {
        "frames": int(len(measured)),
        "time_start_hhmmss": float(measured[0, -1]),
        "time_end_hhmmss": float(measured[-1, -1]),
        "azimuth_mae_deg": float(np.mean(np.abs(errors[:, :3]))),
        "azimuth_rmse_deg": float(np.sqrt(np.mean(errors[:, :3] ** 2))),
        "elevation_mae_deg": float(np.mean(np.abs(errors[:, 3:6]))),
        "elevation_rmse_deg": float(np.sqrt(np.mean(errors[:, 3:6] ** 2))),
        "joint_median_deg": float(np.median(joint)),
        "joint_p90_deg": float(np.percentile(joint, 90)),
        "joint_p95_deg": float(np.percentile(joint, 95)),
        "frame_all_targets_le10deg": float(np.mean(np.max(joint, axis=1) <= 10.0)),
        "frame_all_targets_le20deg": float(np.mean(np.max(joint, axis=1) <= 20.0)),
        "joint_max_deg": float(np.max(joint)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ips", nargs="+", type=int,
                        default=[5, 40, 43, 47, 48, 49, 54])
    args = parser.parse_args()
    result = {}
    for ip in args.ips:
        measured = load_table(args.root / f"deal_doa_{ip}_132619-133018.txt")
        gps = load_table(args.root / f"gps_doa_{ip}.txt")
        result[str(ip)] = audit_node(measured, gps)
    payload = {
        "claim_status": "offline_measured_doa_vs_gps_audit",
        "root": str(args.root),
        "gps_role": "offline scoring only; no runtime correction or relabeling",
        "assignment": "per-frame minimum joint azimuth/elevation residual",
        "nodes": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
