#!/usr/bin/env python3
"""Audit information content of the historical GPS-assisted three-peak DOA set.

GPS-derived DOAs are deliberately used only as an offline oracle and, in a
separate continuity stress test, for an initial calibration prefix. They are
never treated as an online tracking measurement. The audit distinguishes an
oracle peak assignment from a GPS-free continuation after calibration.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


DEAL_PATTERN = re.compile(r"deal_doa_(\d+)_132619-133018\.txt$")
PERMUTATIONS = tuple(itertools.permutations(range(3)))


def wrapped_abs_deg(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.abs((left - right + 180.0) % 360.0 - 180.0)


def read_rows(path: Path, columns: int) -> np.ndarray:
    rows: list[list[float]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        values = raw.split()
        if not values:
            continue
        if len(values) != columns:
            raise RuntimeError(f"unexpected column count in {path}: expected {columns}, got {len(values)}")
        rows.append([float(value) for value in values])
    if not rows:
        raise RuntimeError(f"no rows in {path}")
    return np.asarray(rows, dtype=float)


def interpolate_oracle(deal: np.ndarray, gps: np.ndarray) -> np.ndarray:
    """Reproduce the historical per-second linear interpolation convention."""
    by_time: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(deal):
        by_time[int(round(row[7]))].append(index)
    # gps_doa columns are az1, el1, az2, el2, az3, el3, d1, d2, d3,
    # not three contiguous target triples.
    gps_by_time = {
        int(round(row[10])): np.asarray(
            [
                [row[1], row[2], row[7]],
                [row[3], row[4], row[8]],
                [row[5], row[6], row[9]],
            ],
            dtype=float,
        )
        for row in gps
    }
    oracle = np.empty((len(deal), 3, 3), dtype=float)
    for time_s, indexes in by_time.items():
        current = gps_by_time.get(time_s)
        if current is None:
            raise RuntimeError(f"GPS oracle missing second {time_s}")
        following = gps_by_time.get(time_s + 1, current)
        for rank, index in enumerate(indexes):
            alpha = rank / max(len(indexes) - 1, 1)
            oracle[index] = current + alpha * (following - current)
    return oracle


def pair_cost(candidates: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    az_error = wrapped_abs_deg(candidates[:, None, 0], targets[None, :, 0])
    el_error = np.abs(candidates[:, None, 1] - targets[None, :, 1])
    combined = np.hypot(az_error, el_error)
    return az_error, el_error, combined


def oracle_permutation(candidates: np.ndarray, targets: np.ndarray) -> tuple[tuple[int, int, int], np.ndarray, np.ndarray]:
    az_error, el_error, combined = pair_cost(candidates, targets)
    best = min(PERMUTATIONS, key=lambda perm: float(sum(combined[candidate, target] for target, candidate in enumerate(perm))))
    return best, az_error, el_error


def continuity_permutation(candidates: np.ndarray, previous: np.ndarray) -> tuple[int, int, int]:
    _, _, combined = pair_cost(candidates, previous)
    return min(PERMUTATIONS, key=lambda perm: float(sum(combined[candidate, target] for target, candidate in enumerate(perm))))


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
    }


def analyze_node(node: str, deal: np.ndarray, gps: np.ndarray, calibration_seconds: float) -> tuple[dict, list[dict]]:
    oracle = interpolate_oracle(deal, gps)
    candidate_pairs = deal[:, 1:7].reshape(-1, 3, 2)
    time_s = deal[:, 7]
    start_time = float(time_s.min())
    oracle_perms: list[tuple[int, int, int]] = []
    az_errors: list[float] = []
    el_errors: list[float] = []
    combined_errors: list[float] = []
    independent_az_errors: list[float] = []
    independent_el_errors: list[float] = []
    az_collision_rows = 0
    el_collision_rows = 0
    split_pair_rows = 0
    frame_rows: list[dict] = []
    for index, (candidates, targets) in enumerate(zip(candidate_pairs, oracle[:, :, :2], strict=True)):
        perm, az_matrix, el_matrix = oracle_permutation(candidates, targets)
        oracle_perms.append(perm)
        target_az = [float(az_matrix[candidate, target]) for target, candidate in enumerate(perm)]
        target_el = [float(el_matrix[candidate, target]) for target, candidate in enumerate(perm)]
        az_errors.extend(target_az)
        el_errors.extend(target_el)
        combined_errors.extend([math.hypot(az, el) for az, el in zip(target_az, target_el, strict=True)])
        az_pick = tuple(int(np.argmin(az_matrix[:, target])) for target in range(3))
        el_pick = tuple(int(np.argmin(el_matrix[:, target])) for target in range(3))
        independent_az_errors.extend(float(az_matrix[candidate, target]) for target, candidate in enumerate(az_pick))
        independent_el_errors.extend(float(el_matrix[candidate, target]) for target, candidate in enumerate(el_pick))
        az_collision_rows += int(len(set(az_pick)) < 3)
        el_collision_rows += int(len(set(el_pick)) < 3)
        split_pair_rows += int(any(az_pick[target] != el_pick[target] for target in range(3)))
        frame_rows.append(
            {
                "node": node,
                "frame_index": index,
                "time_s": float(time_s[index]),
                "oracle_permutation_target1_target2_target3": "-".join(str(value + 1) for value in perm),
                "oracle_pair_az_mae_deg": float(np.mean(target_az)),
                "oracle_pair_el_mae_deg": float(np.mean(target_el)),
                "oracle_pair_combined_mae_deg": float(np.mean([math.hypot(az, el) for az, el in zip(target_az, target_el, strict=True)])),
                "independent_az_collision": len(set(az_pick)) < 3,
                "independent_el_collision": len(set(el_pick)) < 3,
                "independent_split_az_el_pair": any(az_pick[target] != el_pick[target] for target in range(3)),
            }
        )

    calibration_mask = time_s < start_time + calibration_seconds
    if not np.any(calibration_mask) or np.all(calibration_mask):
        raise RuntimeError(f"calibration duration {calibration_seconds}s is not usable for node {node}")
    last_calibration = int(np.where(calibration_mask)[0][-1])
    selected = candidate_pairs[last_calibration, list(oracle_perms[last_calibration])].copy()
    continuity_target_correct: list[bool] = []
    continuity_all_correct: list[bool] = []
    continuity_pair_az_error: list[float] = []
    continuity_pair_el_error: list[float] = []
    for index in range(last_calibration + 1, len(candidate_pairs)):
        perm = continuity_permutation(candidate_pairs[index], selected)
        selected = candidate_pairs[index, list(perm)]
        truth_perm = oracle_perms[index]
        target_correct = [perm[target] == truth_perm[target] for target in range(3)]
        continuity_target_correct.extend(target_correct)
        continuity_all_correct.append(all(target_correct))
        az_matrix, el_matrix, _ = pair_cost(candidate_pairs[index], oracle[index, :, :2])
        continuity_pair_az_error.extend(float(az_matrix[candidate, target]) for target, candidate in enumerate(perm))
        continuity_pair_el_error.extend(float(el_matrix[candidate, target]) for target, candidate in enumerate(perm))
        frame_rows[index].update(
            {
                "continuity_permutation_target1_target2_target3": "-".join(str(value + 1) for value in perm),
                "continuity_target_identity_accuracy": float(np.mean(target_correct)),
                "continuity_all_three_identity_correct": all(target_correct),
                "continuity_pair_az_mae_deg": float(np.mean([az_matrix[candidate, target] for target, candidate in enumerate(perm)])),
                "continuity_pair_el_mae_deg": float(np.mean([el_matrix[candidate, target] for target, candidate in enumerate(perm)])),
            }
        )

    rows = len(candidate_pairs)
    return {
        "node": node,
        "frames": rows,
        "time_start": float(time_s.min()),
        "time_end": float(time_s.max()),
        "oracle_pair_preserving_az_error_deg": quantiles(az_errors),
        "oracle_pair_preserving_el_error_deg": quantiles(el_errors),
        "oracle_pair_preserving_combined_error_deg": quantiles(combined_errors),
        "historical_independent_az_error_deg": quantiles(independent_az_errors),
        "historical_independent_el_error_deg": quantiles(independent_el_errors),
        "historical_independent_az_collision_fraction": az_collision_rows / rows,
        "historical_independent_el_collision_fraction": el_collision_rows / rows,
        "historical_independent_az_el_split_pair_fraction": split_pair_rows / rows,
        "continuity_protocol": {
            "uses_gps_after_calibration": False,
            "calibration_duration_s": calibration_seconds,
            "calibration_frames": last_calibration + 1,
            "evaluation_frames": len(continuity_all_correct),
            "per_target_identity_accuracy": float(np.mean(continuity_target_correct)),
            "all_three_identity_accuracy": float(np.mean(continuity_all_correct)),
            "pair_preserving_az_error_deg": quantiles(continuity_pair_az_error),
            "pair_preserving_el_error_deg": quantiles(continuity_pair_el_error),
        },
    }, frame_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-seconds", type=float, default=30.0)
    args = parser.parse_args()

    deal_files: dict[str, Path] = {}
    for path in args.data_dir.glob("deal_doa_*.txt"):
        match = DEAL_PATTERN.match(path.name)
        if match:
            deal_files[match.group(1)] = path
    if len(deal_files) < 3:
        raise RuntimeError(f"expected at least three historical node files, found {sorted(deal_files)}")
    args.output.mkdir(parents=True, exist_ok=True)
    node_summaries: list[dict] = []
    frame_rows: list[dict] = []
    for node in sorted(deal_files, key=int):
        gps_path = args.data_dir / f"gps_doa_{node}.txt"
        if not gps_path.exists():
            raise RuntimeError(f"missing GPS oracle file for node {node}")
        summary, rows = analyze_node(node, read_rows(deal_files[node], 8), read_rows(gps_path, 11), args.calibration_seconds)
        node_summaries.append(summary)
        frame_rows.extend(rows)

    aggregate = {
        "nodes": [entry["node"] for entry in node_summaries],
        "frames_per_node": [entry["frames"] for entry in node_summaries],
        "oracle_pair_combined_median_deg_mean_over_nodes": statistics.mean(entry["oracle_pair_preserving_combined_error_deg"]["median"] for entry in node_summaries),
        "oracle_pair_combined_p90_deg_mean_over_nodes": statistics.mean(entry["oracle_pair_preserving_combined_error_deg"]["p90"] for entry in node_summaries),
        "independent_az_collision_fraction_mean_over_nodes": statistics.mean(entry["historical_independent_az_collision_fraction"] for entry in node_summaries),
        "independent_el_collision_fraction_mean_over_nodes": statistics.mean(entry["historical_independent_el_collision_fraction"] for entry in node_summaries),
        "independent_split_pair_fraction_mean_over_nodes": statistics.mean(entry["historical_independent_az_el_split_pair_fraction"] for entry in node_summaries),
        "continuity_per_target_identity_accuracy_mean_over_nodes": statistics.mean(entry["continuity_protocol"]["per_target_identity_accuracy"] for entry in node_summaries),
        "continuity_all_three_identity_accuracy_mean_over_nodes": statistics.mean(entry["continuity_protocol"]["all_three_identity_accuracy"] for entry in node_summaries),
    }
    payload = {
        "audit": "three-source historical GPS-assisted DOA information content",
        "data_dir": str(args.data_dir),
        "gps_policy": {
            "oracle_role": "offline peak-assignment assessment only",
            "continuity_role": "initial calibration prefix only",
            "gps_used_after_calibration": False,
            "not_an_online_measurement": True,
        },
        "method": {
            "oracle_assignment": "minimum combined wrapped-azimuth/elevation error over all six one-to-one candidate permutations",
            "historical_assignment_diagnostic": "independent nearest azimuth and elevation selection per target; collision and split-pair rates are reported",
            "continuity_assignment": "after GPS-calibrated prefix, select one-to-one candidate permutation minimizing paired angular displacement from the previous assigned frame",
        },
        "aggregate": aggregate,
        "nodes": node_summaries,
    }
    fields = sorted({field for row in frame_rows for field in row})
    with (args.output / "frame_diagnostics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(frame_rows)
    (args.output / "information_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
