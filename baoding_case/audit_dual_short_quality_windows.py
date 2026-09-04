#!/usr/bin/env python3
"""Rank dual-target fixed-length windows using only A6 acoustic diagnostics.

GPS is intentionally absent from candidate selection.  The output can later be
joined with truth only for offline geometric and error reporting.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def position_covariance_min_eigenvalue(row: dict[str, str]) -> float:
    covariance = np.asarray(
        [[float(row[f"cov_{i}{j}"]) for j in range(3)] for i in range(3)],
        dtype=float,
    )
    return float(np.min(np.linalg.eigvalsh(0.5 * (covariance + covariance.T))))


def is_gate_failure(row: dict[str, str], minimum_inliers: int, maximum_reprojection: float) -> bool:
    return (
        str(row["valid"]).lower() != "true"
        or int(row["inlier_nodes"]) < minimum_inliers
        or float(row["reprojection_rms_deg"]) > maximum_reprojection
    )


def contiguous(stamps: list[int]) -> bool:
    return all(right == left + 1 for left, right in zip(stamps, stamps[1:]))


def score_window(
    rows_by_target: dict[int, dict[int, dict[str, str]]],
    stamps: list[int],
    minimum_inliers: int,
    maximum_reprojection: float,
) -> dict[str, object]:
    rows = [rows_by_target[target][stamp] for stamp in stamps for target in (1, 2)]
    reprojection = np.asarray([float(row["reprojection_rms_deg"]) for row in rows])
    condition = np.asarray([float(row["condition_number"]) for row in rows])
    inliers = np.asarray([int(row["inlier_nodes"]) for row in rows])
    steps = np.asarray([float(row["state_step_m"] or 0.0) for row in rows])
    eigenvalues = np.asarray([position_covariance_min_eigenvalue(row) for row in rows])
    gate_failures = sum(is_gate_failure(row, minimum_inliers, maximum_reprojection) for row in rows)
    moderate_failures = int(np.sum((inliers < 4) | (reprojection > 10.0)))
    high_reprojection = int(np.sum(reprojection > maximum_reprojection))
    low_inlier = int(np.sum(inliers < minimum_inliers))
    non_psd = int(np.sum(eigenvalues <= 0.0))
    extreme_state_steps = int(np.sum(steps > 250.0))
    penalty = float(
        1_000_000 * gate_failures
        + 100_000 * non_psd
        + 100_000 * extreme_state_steps
        + 10_000 * moderate_failures
        + 100 * np.percentile(reprojection, 90.0)
        + 10 * np.percentile(condition, 90.0)
        + np.max(steps)
    )
    return {
        "start_time_s": stamps[0],
        "end_time_s": stamps[-1],
        "frames": len(stamps),
        "duration_s": stamps[-1] - stamps[0],
        "gate_failure_target_seconds": int(gate_failures),
        "moderate_failure_target_seconds": moderate_failures,
        "high_reprojection_target_seconds": high_reprojection,
        "low_inlier_target_seconds": low_inlier,
        "non_psd_target_seconds": non_psd,
        "extreme_state_step_target_seconds": extreme_state_steps,
        "minimum_inliers": int(np.min(inliers)),
        "reprojection_median_deg": float(np.median(reprojection)),
        "reprojection_p90_deg": float(np.percentile(reprojection, 90.0)),
        "reprojection_max_deg": float(np.max(reprojection)),
        "condition_p90": float(np.percentile(condition, 90.0)),
        "state_step_max_m": float(np.max(steps)),
        "minimum_position_covariance_eigenvalue_m2": float(np.min(eigenvalues)),
        "acoustic_quality_penalty": penalty,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a6-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=(50, 60, 70))
    parser.add_argument("--minimum-inliers", type=int, default=2)
    parser.add_argument("--maximum-reprojection-deg", type=float, default=25.0)
    parser.add_argument("--exclude-start-s", type=int, default=46658)
    parser.add_argument("--exclude-end-s", type=int, default=46660)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    rows_by_target: dict[int, dict[int, dict[str, str]]] = {}
    for target in (1, 2):
        rows = read_csv(args.a6_root / f"target{target}_state.csv")
        rows_by_target[target] = {int(round(float(row["time_s"]))): row for row in rows}
    common = sorted(set(rows_by_target[1]) & set(rows_by_target[2]))
    payload: dict[str, object] = {
        "selection_protocol": "A6 acoustic diagnostics only; GPS excluded from candidate selection",
        "quality_gate": {
            "minimum_inliers": args.minimum_inliers,
            "maximum_reprojection_deg": args.maximum_reprojection_deg,
        },
        "excluded_acoustic_anomaly_interval_s": [args.exclude_start_s, args.exclude_end_s],
        "lengths": {},
    }
    for frames in args.frames:
        all_candidates = []
        for index in range(len(common) - frames + 1):
            stamps = common[index : index + frames]
            if not contiguous(stamps):
                continue
            all_candidates.append(score_window(
                rows_by_target, stamps, args.minimum_inliers, args.maximum_reprojection_deg
            ))
        all_candidates.sort(key=lambda row: (
            row["acoustic_quality_penalty"],
            row["gate_failure_target_seconds"],
            row["extreme_state_step_target_seconds"],
            row["moderate_failure_target_seconds"],
            row["reprojection_p90_deg"],
            row["condition_p90"],
        ))
        avoiding = [
            row for row in all_candidates
            if row["end_time_s"] < args.exclude_start_s or row["start_time_s"] > args.exclude_end_s
        ]
        payload["lengths"][str(frames)] = {
            "candidate_count": len(all_candidates),
            "top_all": all_candidates[:args.top_k],
            "top_excluding_acoustic_anomaly": avoiding[:args.top_k],
            "selected": (avoiding or all_candidates)[0],
            "selection_reason": "lowest lexicographic acoustic-quality penalty after excluding the preidentified A6 anomaly interval",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
