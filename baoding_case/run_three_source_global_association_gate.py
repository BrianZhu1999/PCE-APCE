#!/usr/bin/env python3
"""GPS-calibrated, GPS-free-afterward three-source association gate for Baoding.

The historical three-source DOA files contain three unlabelled paired
azimuth/zenith candidates at seven nodes. GPS is used only before the frozen
calibration boundary to calibrate per-node angle transforms and initialise a
three-target constant-velocity predictor. Every post-calibration assignment is
chosen from acoustic angles, pair-preserving permutations, multi-node geometry,
and previous acoustic state. GPS is then used only for offline evaluation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


FRAME_DT_S = 640.0 / 3050.0
TARGETS = (0, 1, 2)
PERMUTATIONS = tuple(itertools.permutations(range(3)))
TOP_PER_NODE = 3
TOP_GLOBAL = 48
ALPHA_POSITION = 0.65
BETA_VELOCITY = 0.05
INLIER_RESIDUAL_DEG = 20.0
MIN_INLIERS = 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def hms_seconds(value: float) -> float:
    integer = int(value)
    fraction = value - integer
    text = str(integer).zfill(6)
    return int(text[:2]) * 3600 + int(text[2:4]) * 60 + int(text[4:]) + fraction


def shift_hhmmss(value: int, offset_s: int) -> int:
    seconds = int(round(hms_seconds(value))) + offset_s
    return (seconds // 3600) * 10000 + ((seconds % 3600) // 60) * 100 + seconds % 60


def wrap_difference(left: np.ndarray | float, right: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(left) - np.asarray(right) + 180.0) % 360.0 - 180.0


def wrapped_abs(left: np.ndarray | float, right: np.ndarray | float) -> np.ndarray | float:
    return np.abs(wrap_difference(left, right))


def read_numeric(path: Path, columns: int) -> np.ndarray:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != columns:
            raise RuntimeError(f"{path} has {len(fields)} columns, expected {columns}")
        rows.append([float(value) for value in fields])
    if not rows:
        raise RuntimeError(f"no rows in {path}")
    return np.asarray(rows, dtype=float)


def parse_nodes(path: Path) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    nodes: dict[int, np.ndarray] = {}
    ip_to_node: dict[int, int] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        ip_suffix = int(fields[0].split(".")[-1])
        node = int(fields[2])
        nodes[node] = np.asarray([float(fields[3]), float(fields[4]), float(fields[5])], dtype=float)
        ip_to_node[ip_suffix] = node
    return nodes, ip_to_node


def parse_gps(path: Path) -> dict[int, np.ndarray]:
    output: dict[int, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split("\t")
        if len(fields) < 8:
            continue
        try:
            output[int(float(fields[7]))] = np.asarray([float(fields[4]), float(fields[5]), float(fields[6])], dtype=float)
        except ValueError:
            continue
    if not output:
        raise RuntimeError(f"no GPS positions in {path}")
    return output


def fuse_tracks(left: dict[int, np.ndarray], right: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    return {key: 0.5 * (left[key] + right[key]) for key in left.keys() & right.keys()}


def nearest_track_position(track: dict[int, np.ndarray], hhmmss: int) -> np.ndarray:
    if hhmmss in track:
        return track[hhmmss]
    candidates = [key for key in track if abs(hms_seconds(key) - hms_seconds(hhmmss)) <= 1.0]
    if not candidates:
        raise RuntimeError(f"GPS position unavailable at {hhmmss}")
    return track[min(candidates, key=lambda key: abs(hms_seconds(key) - hms_seconds(hhmmss)))]


def interpolate_track(deal: np.ndarray, track: dict[int, np.ndarray]) -> np.ndarray:
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(deal):
        grouped[int(round(row[7]))].append(index)
    output = np.empty((len(deal), 3), dtype=float)
    for time_s, indexes in grouped.items():
        current = nearest_track_position(track, time_s)
        following = nearest_track_position(track, shift_hhmmss(time_s, 1))
        for rank, index in enumerate(indexes):
            alpha = rank / max(1, len(indexes) - 1)
            output[index] = current + alpha * (following - current)
    return output


def gps_oracle_angles(deal: np.ndarray, gps_doa: np.ndarray) -> np.ndarray:
    """Return target-major [azimuth, zenith, range] at every acoustic frame."""
    per_second = {
        int(round(row[10])): np.asarray(
            [
                [row[1], row[2], row[7]],
                [row[3], row[4], row[8]],
                [row[5], row[6], row[9]],
            ],
            dtype=float,
        )
        for row in gps_doa
    }
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(deal):
        grouped[int(round(row[7]))].append(index)
    output = np.empty((len(deal), 3, 3), dtype=float)
    for time_s, indexes in grouped.items():
        current = per_second[time_s]
        following = per_second.get(shift_hhmmss(time_s, 1), current)
        for rank, index in enumerate(indexes):
            alpha = rank / max(1, len(indexes) - 1)
            output[index] = current + alpha * (following - current)
    return output


def angle_cost(observation: np.ndarray, truth: np.ndarray) -> float:
    return float(math.hypot(float(wrapped_abs(observation[0], truth[0])), float(observation[1] - truth[1])))


def permutation_cost(candidates: np.ndarray, targets: np.ndarray) -> tuple[tuple[int, int, int], list[float]]:
    best: tuple[tuple[int, int, int], list[float]] | None = None
    for permutation in PERMUTATIONS:
        errors = [angle_cost(candidates[permutation[target]], targets[target]) for target in TARGETS]
        if best is None or sum(errors) < sum(best[1]):
            best = (permutation, errors)
    assert best is not None
    return best


def circular_mean(values: list[float]) -> float:
    radians = np.radians(values)
    return float(math.degrees(math.atan2(float(np.sin(radians).mean()), float(np.cos(radians).mean()))) % 360.0)


def calibrate_transform(candidates: np.ndarray, oracle: np.ndarray, calibration_frames: int) -> dict:
    best: dict | None = None
    for az_sign in (1.0, -1.0):
        for zenith_sign in (1.0, -1.0):
            az_offset, zenith_offset = 0.0, 0.0
            for _ in range(5):
                az_residuals: list[float] = []
                zenith_residuals: list[float] = []
                for frame in range(calibration_frames):
                    transformed = candidates[frame].copy()
                    transformed[:, 0] = (az_sign * transformed[:, 0] + az_offset) % 360.0
                    transformed[:, 1] = zenith_sign * transformed[:, 1] + zenith_offset
                    permutation, _ = permutation_cost(transformed, oracle[frame, :, :2])
                    for target in TARGETS:
                        raw = candidates[frame, permutation[target]]
                        az_residuals.append(float(wrap_difference(oracle[frame, target, 0], az_sign * raw[0])))
                        zenith_residuals.append(float(oracle[frame, target, 1] - zenith_sign * raw[1]))
                az_offset = circular_mean(az_residuals)
                zenith_offset = float(statistics.median(zenith_residuals))
            errors: list[float] = []
            for frame in range(calibration_frames):
                transformed = candidates[frame].copy()
                transformed[:, 0] = (az_sign * transformed[:, 0] + az_offset) % 360.0
                transformed[:, 1] = zenith_sign * transformed[:, 1] + zenith_offset
                _, frame_errors = permutation_cost(transformed, oracle[frame, :, :2])
                errors.extend(frame_errors)
            candidate = {
                "azimuth_sign": az_sign,
                "azimuth_offset_deg": az_offset,
                "zenith_sign": zenith_sign,
                "zenith_offset_deg": zenith_offset,
                "calibration_pair_error_median_deg": float(statistics.median(errors)),
                "calibration_pair_error_p90_deg": float(np.quantile(errors, 0.90)),
            }
            if best is None or (candidate["calibration_pair_error_median_deg"], candidate["calibration_pair_error_p90_deg"]) < (
                best["calibration_pair_error_median_deg"], best["calibration_pair_error_p90_deg"],
            ):
                best = candidate
    assert best is not None
    return best


def apply_transform(candidates: np.ndarray, transform: dict) -> np.ndarray:
    output = candidates.copy()
    output[:, 0] = (transform["azimuth_sign"] * output[:, 0] + transform["azimuth_offset_deg"]) % 360.0
    output[:, 1] = transform["zenith_sign"] * output[:, 1] + transform["zenith_offset_deg"]
    return output


def direction(azimuth_deg: float, zenith_deg: float) -> np.ndarray:
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(90.0 - zenith_deg)
    return np.asarray(
        [math.cos(elevation) * math.cos(azimuth), math.cos(elevation) * math.sin(azimuth), math.sin(elevation)],
        dtype=float,
    )


def predicted_angles(position: np.ndarray, node: np.ndarray) -> np.ndarray:
    delta = position - node
    azimuth = math.degrees(math.atan2(delta[1], delta[0])) % 360.0
    zenith = 90.0 - math.degrees(math.atan2(delta[2], math.hypot(delta[0], delta[1])))
    return np.asarray([azimuth, zenith], dtype=float)


def solve_rays(observations: dict[int, np.ndarray], nodes: dict[int, np.ndarray]) -> tuple[np.ndarray | None, list[int], float, np.ndarray | None, float]:
    active = list(observations)
    if len(active) < MIN_INLIERS:
        return None, [], float("inf"), None, float("inf")
    for _ in range(3):
        matrix = np.zeros((3, 3), dtype=float)
        rhs = np.zeros(3, dtype=float)
        for node in active:
            vector = direction(float(observations[node][0]), float(observations[node][1]))
            projector = np.eye(3) - np.outer(vector, vector)
            matrix += projector
            rhs += projector @ nodes[node]
        condition = float(np.linalg.cond(matrix))
        if not math.isfinite(condition):
            return None, [], condition, None, float("inf")
        position = np.linalg.pinv(matrix) @ rhs
        residuals = {
            node: angle_cost(observations[node], predicted_angles(position, nodes[node]))
            for node in active
        }
        kept = [node for node, residual in residuals.items() if residual <= INLIER_RESIDUAL_DEG]
        if len(kept) < MIN_INLIERS:
            kept = [node for node, _ in sorted(residuals.items(), key=lambda item: item[1])[:MIN_INLIERS]]
        if kept == active:
            break
        active = kept
    matrix = np.zeros((3, 3), dtype=float)
    rhs = np.zeros(3, dtype=float)
    residual_m: list[float] = []
    residual_deg: list[float] = []
    for node in active:
        vector = direction(float(observations[node][0]), float(observations[node][1]))
        projector = np.eye(3) - np.outer(vector, vector)
        matrix += projector
        rhs += projector @ nodes[node]
    condition = float(np.linalg.cond(matrix))
    position = np.linalg.pinv(matrix) @ rhs
    for node in active:
        residual = angle_cost(observations[node], predicted_angles(position, nodes[node]))
        residual_deg.append(residual)
        residual_m.append(math.radians(residual) * float(np.linalg.norm(position - nodes[node])))
    sigma_m = max(float(np.median(residual_m)), 2.0)
    covariance = sigma_m * sigma_m * np.linalg.pinv(matrix)
    return position, active, condition, covariance, float(np.median(residual_deg))


def state_prediction(position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    return position + FRAME_DT_S * velocity


def node_options(
    candidates: np.ndarray,
    predicted_positions: list[np.ndarray],
    node_position: np.ndarray,
    previous: np.ndarray | None,
) -> list[dict]:
    options: list[dict] = []
    for permutation in PERMUTATIONS:
        assigned = candidates[list(permutation)]
        prediction_cost = sum(angle_cost(assigned[target], predicted_angles(predicted_positions[target], node_position)) for target in TARGETS)
        continuity_cost = 0.0 if previous is None else sum(angle_cost(assigned[target], previous[target]) for target in TARGETS)
        options.append({
            "permutation": permutation,
            "assigned": assigned,
            "cheap_cost": float(prediction_cost + 0.25 * continuity_cost),
        })
    return sorted(options, key=lambda row: row["cheap_cost"])[:TOP_PER_NODE]


def global_associate(
    node_candidates: dict[int, np.ndarray],
    nodes: dict[int, np.ndarray],
    predicted_positions: list[np.ndarray],
    previous_observations: dict[int, np.ndarray] | None,
) -> tuple[dict, dict[int, np.ndarray]]:
    ordered_nodes = sorted(node_candidates)
    candidates_per_node = [
        node_options(node_candidates[node], predicted_positions, nodes[node], None if previous_observations is None else previous_observations[node])
        for node in ordered_nodes
    ]
    cheap_hypotheses = []
    for choice in itertools.product(*candidates_per_node):
        cheap_hypotheses.append((sum(item["cheap_cost"] for item in choice), choice))
    cheap_hypotheses.sort(key=lambda item: item[0])
    best: dict | None = None
    for cheap_cost, choice in cheap_hypotheses[:TOP_GLOBAL]:
        assigned_by_node = {node: choice[index]["assigned"] for index, node in enumerate(ordered_nodes)}
        positions: list[np.ndarray | None] = []
        inliers: list[list[int]] = []
        conditions: list[float] = []
        covariances: list[np.ndarray | None] = []
        reprojection = 0.0
        for target in TARGETS:
            observations = {node: assigned_by_node[node][target] for node in ordered_nodes}
            position, target_inliers, condition, covariance, residual = solve_rays(observations, nodes)
            positions.append(position)
            inliers.append(target_inliers)
            conditions.append(condition)
            covariances.append(covariance)
            reprojection += residual + 2.0 * (len(ordered_nodes) - len(target_inliers))
        position_cost = sum(
            min(float(np.linalg.norm(positions[target] - predicted_positions[target])) / 250.0, 10.0)
            if positions[target] is not None else 10.0
            for target in TARGETS
        )
        score = float(0.35 * cheap_cost + reprojection + 8.0 * position_cost)
        record = {
            "score": score,
            "cheap_cost_deg": float(cheap_cost),
            "reprojection_cost_deg": float(reprojection),
            "position_cost_scaled": float(position_cost),
            "positions": positions,
            "inliers": inliers,
            "conditions": conditions,
            "covariances": covariances,
            "permutations": {str(node): "-".join(str(value + 1) for value in choice[index]["permutation"]) for index, node in enumerate(ordered_nodes)},
        }
        if best is None or record["score"] < best["score"]:
            best = record
            best_assigned = assigned_by_node
    assert best is not None
    return best, best_assigned


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p90": None}
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "median": float(np.median(array)), "p90": float(np.quantile(array, 0.90))}


def oracle_triangulation_upper_bound(
    candidates_by_node: dict[int, np.ndarray],
    oracle_permutations: dict[int, list[tuple[int, int, int]]],
    nodes: dict[int, np.ndarray],
    truth: list[np.ndarray],
    start_frame: int,
    frame_count: int,
) -> dict:
    """Offline ceiling with full GPS-assisted peak identity, never an online result."""
    errors = {target: [] for target in TARGETS}
    jumps = {target: [] for target in TARGETS}
    valid = {target: 0 for target in TARGETS}
    previous = [None, None, None]
    covariance_psd: list[bool] = []
    for frame in range(start_frame, frame_count):
        for target in TARGETS:
            observations = {
                node: candidates_by_node[node][frame, oracle_permutations[node][frame][target]]
                for node in candidates_by_node
            }
            position, _, _, covariance, _ = solve_rays(observations, nodes)
            covariance_psd.append(bool(covariance is not None and np.all(np.linalg.eigvalsh(covariance) >= -1e-8)))
            if position is None:
                continue
            valid[target] += 1
            errors[target].append(float(np.linalg.norm(position - truth[target][frame])))
            if previous[target] is not None:
                jumps[target].append(float(np.linalg.norm(position - previous[target])))
            previous[target] = position
    return {
        "gps_role": "full-sequence offline oracle assignment only; not a valid runtime input",
        "frames": frame_count - start_frame,
        "covariance_psd_fraction": float(np.mean(covariance_psd)) if covariance_psd else 0.0,
        "targets": {
            f"target{target + 1}": {
                "valid_fraction": valid[target] / max(frame_count - start_frame, 1),
                "position_error_m": metrics(errors[target]),
                "frame_jump_m": metrics(jumps[target]),
            }
            for target in TARGETS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--nod", type=Path, required=True)
    parser.add_argument("--gps-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-seconds", type=float, default=30.0)
    args = parser.parse_args()

    nodes, ip_to_node = parse_nodes(args.nod)
    deal_by_node: dict[int, np.ndarray] = {}
    oracle_by_node: dict[int, np.ndarray] = {}
    for deal_path in sorted(args.data_dir.glob("deal_doa_*.txt")):
        suffix = int(deal_path.name.split("_")[2])
        node = ip_to_node[suffix]
        deal = read_numeric(deal_path, 8)
        gps_doa = read_numeric(args.data_dir / f"gps_doa_{suffix}.txt", 11)
        deal_by_node[node] = deal
        oracle_by_node[node] = gps_oracle_angles(deal, gps_doa)
    active_nodes = sorted(deal_by_node)
    if len(active_nodes) != 7:
        raise RuntimeError(f"expected seven historical nodes, found {active_nodes}")
    frame_count = min(len(deal_by_node[node]) for node in active_nodes)
    deal_by_node = {node: deal_by_node[node][:frame_count] for node in active_nodes}
    oracle_by_node = {node: oracle_by_node[node][:frame_count] for node in active_nodes}
    start_hhmmss = int(round(deal_by_node[active_nodes[0]][0, 7]))
    calibration_frames = int(round(args.calibration_seconds / FRAME_DT_S))
    if calibration_frames < 10 or calibration_frames >= frame_count:
        raise RuntimeError("calibration duration is invalid for the available frames")

    transforms: dict[int, dict] = {}
    candidates_by_node: dict[int, np.ndarray] = {}
    oracle_permutations: dict[int, list[tuple[int, int, int]]] = {}
    for node in active_nodes:
        raw_candidates = deal_by_node[node][:, 1:7].reshape(frame_count, 3, 2)
        transforms[node] = calibrate_transform(raw_candidates, oracle_by_node[node], calibration_frames)
        candidates_by_node[node] = np.asarray([apply_transform(frame, transforms[node]) for frame in raw_candidates])
        oracle_permutations[node] = [
            permutation_cost(candidates_by_node[node][frame], oracle_by_node[node][frame, :, :2])[0]
            for frame in range(frame_count)
        ]

    gps1 = parse_gps(args.gps_root / "GPS1_plane1.gps")
    gps2 = parse_gps(args.gps_root / "GPS2_plane1.gps")
    tracks = [
        fuse_tracks(gps1, gps2),
        parse_gps(args.gps_root / "GPS3_plane2.gps"),
        parse_gps(args.gps_root / "GPS4_plane2to3.gps"),
    ]
    truth = [interpolate_track(deal_by_node[active_nodes[0]], track) for track in tracks]
    calibration_end_hhmmss = int(round(deal_by_node[active_nodes[0]][calibration_frames - 1, 7]))
    states_position = [truth[target][calibration_frames - 1].copy() for target in TARGETS]
    velocity_start = max(0, calibration_frames - int(round(5.0 / FRAME_DT_S)))
    elapsed = max((calibration_frames - 1 - velocity_start) * FRAME_DT_S, FRAME_DT_S)
    states_velocity = [(truth[target][calibration_frames - 1] - truth[target][velocity_start]) / elapsed for target in TARGETS]

    associated_rows = {node: [] for node in active_nodes}
    target_rows = {target: [] for target in TARGETS}
    frame_rows: list[dict] = []
    previous_observations: dict[int, np.ndarray] | None = None
    identity_correct: list[bool] = []
    frame_all_identity_correct: list[bool] = []
    position_errors = {target: [] for target in TARGETS}
    valid_counts = {target: 0 for target in TARGETS}
    frame_jumps = {target: [] for target in TARGETS}
    previous_measurement = [None, None, None]
    covariance_psd = []

    for frame in range(frame_count):
        time_s = hms_seconds(start_hhmmss) + frame * FRAME_DT_S
        time_hhmmss = int(round(deal_by_node[active_nodes[0]][frame, 7]))
        node_candidates = {node: candidates_by_node[node][frame] for node in active_nodes}
        calibration_frame = frame < calibration_frames
        if calibration_frame:
            assigned = {
                node: node_candidates[node][list(oracle_permutations[node][frame])]
                for node in active_nodes
            }
            geometry = []
            for target in TARGETS:
                observations = {node: assigned[node][target] for node in active_nodes}
                geometry.append(solve_rays(observations, nodes))
            decision = {
                "score": 0.0,
                "cheap_cost_deg": 0.0,
                "reprojection_cost_deg": 0.0,
                "position_cost_scaled": 0.0,
                "permutations": {str(node): "-".join(str(value + 1) for value in oracle_permutations[node][frame]) for node in active_nodes},
                "positions": [item[0] for item in geometry],
                "inliers": [item[1] for item in geometry],
                "conditions": [item[2] for item in geometry],
                "covariances": [item[3] for item in geometry],
            }
        else:
            predicted = [state_prediction(states_position[target], states_velocity[target]) for target in TARGETS]
            decision, assigned = global_associate(node_candidates, nodes, predicted, previous_observations)
        selected_positions = decision["positions"]
        for target in TARGETS:
            measurement = selected_positions[target]
            if measurement is not None:
                predicted = state_prediction(states_position[target], states_velocity[target])
                innovation = measurement - predicted
                states_position[target] = predicted + ALPHA_POSITION * innovation
                states_velocity[target] = states_velocity[target] + (BETA_VELOCITY / FRAME_DT_S) * innovation
                error = float(np.linalg.norm(measurement - truth[target][frame]))
                position_errors[target].append(error)
                valid_counts[target] += 1
                if previous_measurement[target] is not None:
                    frame_jumps[target].append(float(np.linalg.norm(measurement - previous_measurement[target])))
                previous_measurement[target] = measurement
            covariance = decision["covariances"][target]
            covariance_psd.append(bool(covariance is not None and np.all(np.linalg.eigvalsh(covariance) >= -1e-8)))
            covariance_flat = [float(value) for value in covariance.flatten()] if covariance is not None else [None] * 9
            target_rows[target].append({
                "frame_index": frame,
                "time_s": time_s,
                "time_hhmmss": time_hhmmss,
                "calibration_frame": calibration_frame,
                "gps_used_at_runtime": calibration_frame,
                "valid": measurement is not None,
                "px": float(measurement[0]) if measurement is not None else None,
                "py": float(measurement[1]) if measurement is not None else None,
                "pz": float(measurement[2]) if measurement is not None else None,
                "vx": float(states_velocity[target][0]),
                "vy": float(states_velocity[target][1]),
                "vz": float(states_velocity[target][2]),
                "condition_number": float(decision["conditions"][target]),
                "inlier_nodes": "-".join(str(node) for node in decision["inliers"][target]),
                "position_error_m_offline": float(np.linalg.norm(measurement - truth[target][frame])) if measurement is not None else None,
                **{f"r_{index:02d}": covariance_flat[index] for index in range(9)},
            })
        selected_identity: list[bool] = []
        for node in active_nodes:
            chosen_perm = tuple(int(value) - 1 for value in decision["permutations"][str(node)].split("-"))
            oracle_perm = oracle_permutations[node][frame]
            selected_identity.extend(chosen_perm[target] == oracle_perm[target] for target in TARGETS)
            associated_rows[node].append({
                "frame_index": frame,
                "time_s": time_s,
                "time_hhmmss": time_hhmmss,
                "node_id": node,
                "calibration_frame": calibration_frame,
                "gps_used_at_runtime": calibration_frame,
                "permutation_target1_target2_target3": decision["permutations"][str(node)],
                "target1_az_deg": float(assigned[node][0][0]),
                "target1_zenith_deg": float(assigned[node][0][1]),
                "target2_az_deg": float(assigned[node][1][0]),
                "target2_zenith_deg": float(assigned[node][1][1]),
                "target3_az_deg": float(assigned[node][2][0]),
                "target3_zenith_deg": float(assigned[node][2][1]),
                "offline_identity_accuracy": float(np.mean([chosen_perm[target] == oracle_perm[target] for target in TARGETS])),
            })
        if not calibration_frame:
            identity_correct.extend(selected_identity)
            frame_all_identity_correct.append(all(selected_identity))
        frame_rows.append({
            "frame_index": frame,
            "time_s": time_s,
            "time_hhmmss": time_hhmmss,
            "calibration_frame": calibration_frame,
            "gps_used_at_runtime": calibration_frame,
            "global_score": float(decision["score"]),
            "cheap_cost_deg": float(decision["cheap_cost_deg"]),
            "reprojection_cost_deg": float(decision["reprojection_cost_deg"]),
            "position_cost_scaled": float(decision["position_cost_scaled"]),
            "offline_identity_accuracy": float(np.mean(selected_identity)),
            "offline_all_node_target_identities_correct": all(selected_identity),
        })
        previous_observations = assigned

    args.output.mkdir(parents=True, exist_ok=True)
    for node in active_nodes:
        write_csv(args.output / f"associated_node_{node}.csv", associated_rows[node])
    for target in TARGETS:
        write_csv(args.output / f"target{target + 1}_state_covariance.csv", target_rows[target])
    write_csv(args.output / "frame_diagnostics.csv", frame_rows)
    target_metrics = {
        f"target{target + 1}": {
            "valid_fraction": valid_counts[target] / frame_count,
            "position_error_m": metrics(position_errors[target]),
            "frame_jump_m": metrics(frame_jumps[target]),
        }
        for target in TARGETS
    }
    oracle_upper_bound = oracle_triangulation_upper_bound(
        candidates_by_node,
        oracle_permutations,
        nodes,
        truth,
        calibration_frames,
        frame_count,
    )
    post_frames = frame_count - calibration_frames
    gate = {
        "identity_accuracy": float(np.mean(identity_correct)) if identity_correct else 0.0,
        "all_node_target_identity_accuracy": float(np.mean(frame_all_identity_correct)) if frame_all_identity_correct else 0.0,
        "covariance_psd_fraction": float(np.mean(covariance_psd)) if covariance_psd else 0.0,
        "target_valid_fractions": {name: metrics_row["valid_fraction"] for name, metrics_row in target_metrics.items()},
        "target_p90_jump_m": {name: metrics_row["frame_jump_m"]["p90"] for name, metrics_row in target_metrics.items()},
    }
    gate["passed"] = bool(
        gate["identity_accuracy"] >= 0.90
        and all(value >= 0.90 for value in gate["target_valid_fractions"].values())
        and all(value is not None and value < 100.0 for value in gate["target_p90_jump_m"].values())
        and gate["covariance_psd_fraction"] >= 0.99
    )
    manifest = {
        "task": "three-source GPS-calibrated, GPS-free-afterward global association gate",
        "status": "passed" if gate["passed"] else "failed",
        "claim_status": "association_gate_only; no PCE/APCE result",
        "inputs": {
            "data_dir": str(args.data_dir),
            "nod": str(args.nod),
            "gps_root": str(args.gps_root),
            "nodes": active_nodes,
            "node_ips": {str(node): next(ip for ip, mapped in ip_to_node.items() if mapped == node) for node in active_nodes},
            "deal_hashes": {str(node): sha256(next(path for path in args.data_dir.glob("deal_doa_*.txt") if ip_to_node[int(path.name.split("_")[2])] == node)) for node in active_nodes},
        },
        "gps_policy": {
            "calibration_frames": calibration_frames,
            "calibration_seconds": args.calibration_seconds,
            "calibration_end_hhmmss": calibration_end_hhmmss,
            "gps_used_after_calibration": False,
            "offline_evaluation_only_after_calibration": True,
        },
        "algorithm": {
            "candidate_constraint": "preserve each historical azimuth/zenith pair",
            "within_node_assignment": "all six one-to-one target-to-candidate permutations",
            "global_search": {"top_per_node": TOP_PER_NODE, "top_global_hypotheses": TOP_GLOBAL},
            "scores": "prediction angular residual, previous paired-angle continuity, robust multi-ray reprojection, and state-position continuity",
            "state": "six-dimensional position/velocity alpha-beta update with triangulation covariance",
            "fixed_parameters": {"alpha_position": ALPHA_POSITION, "beta_velocity": BETA_VELOCITY, "inlier_residual_deg": INLIER_RESIDUAL_DEG, "min_inliers": MIN_INLIERS},
        },
        "node_transforms": {str(node): transforms[node] for node in active_nodes},
        "offline_metrics": {"post_calibration_frames": post_frames, "targets": target_metrics, "association": gate},
        "offline_oracle_upper_bound": oracle_upper_bound,
        "admission": gate,
        "script_sha256": sha256(Path(__file__)),
    }
    (args.output / "association_gate.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "offline_metrics": manifest["offline_metrics"], "gps_policy": manifest["gps_policy"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
