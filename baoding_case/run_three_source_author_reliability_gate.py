#!/usr/bin/env python3
"""Author-code-informed three-source association gate for Baoding.

This gate transfers three mechanisms visible in the supplied DBN/ReAVI code:
node-specific observation precision, a Gaussian hidden-bearing update, and a
state/covariance recursion. GPS is limited to the frozen calibration interval
and offline scoring. Post-calibration association uses only paired acoustic
DOA candidates, array-node geometry, and the predicted acoustic state.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

import run_three_source_global_association_gate as base


TARGETS = base.TARGETS
PERMUTATIONS = base.PERMUTATIONS
FRAME_DT_S = base.FRAME_DT_S
TOP_PER_NODE = 4
TOP_GLOBAL = 128
HIDDEN_BEARING_PRIOR_SIGMA_DEG = 10.0
MIN_TARGET_NODES = 3
MAX_TARGET_NODES = 5
MIN_CALIBRATION_INLIER_FRACTION = 0.25
CALIBRATION_INLIER_DEG = 20.0
ACCELERATION_STD_MPS2 = 3.0
INITIAL_POSITION_STD_M = 30.0
INITIAL_VELOCITY_STD_MPS = 8.0
MAX_INNOVATION_M = 350.0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def robust_sigma(values: np.ndarray, floor: float = 2.0, ceiling: float = 60.0) -> float:
    values = np.asarray(values, dtype=float)
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    q68 = float(np.quantile(np.abs(values), 0.68))
    return float(np.clip(max(1.4826 * mad, q68, floor), floor, ceiling))


def calibration_precision(
    candidates_by_node: dict[int, np.ndarray],
    oracle_by_node: dict[int, np.ndarray],
    oracle_permutations: dict[int, list[tuple[int, int, int]]],
    calibration_frames: int,
) -> tuple[dict[int, dict[int, dict]], dict[int, list[int]]]:
    diagnostics: dict[int, dict[int, dict]] = {}
    for node, candidates in candidates_by_node.items():
        diagnostics[node] = {}
        for target in TARGETS:
            az_residuals, zenith_residuals, pair_errors = [], [], []
            for frame in range(calibration_frames):
                candidate = candidates[frame, oracle_permutations[node][frame][target]]
                truth = oracle_by_node[node][frame, target, :2]
                azimuth_residual = float(base.wrap_difference(candidate[0], truth[0]))
                zenith_residual = float(candidate[1] - truth[1])
                az_residuals.append(azimuth_residual)
                zenith_residuals.append(zenith_residual)
                pair_errors.append(math.hypot(azimuth_residual, zenith_residual))
            az_array = np.asarray(az_residuals)
            zenith_array = np.asarray(zenith_residuals)
            pair_array = np.asarray(pair_errors)
            inlier_fraction = float(np.mean(pair_array <= CALIBRATION_INLIER_DEG))
            sigma_azimuth = robust_sigma(az_array) / math.sqrt(max(inlier_fraction, 0.05))
            sigma_zenith = robust_sigma(zenith_array) / math.sqrt(max(inlier_fraction, 0.05))
            precision_score = inlier_fraction / max(sigma_azimuth * sigma_zenith, 1e-6)
            diagnostics[node][target] = {
                "sigma_azimuth_deg": float(np.clip(sigma_azimuth, 2.0, 90.0)),
                "sigma_zenith_deg": float(np.clip(sigma_zenith, 2.0, 90.0)),
                "pair_error_median_deg": float(np.median(pair_array)),
                "pair_error_p90_deg": float(np.quantile(pair_array, 0.90)),
                "inlier_fraction_20deg": inlier_fraction,
                "precision_score": precision_score,
            }
    selected: dict[int, list[int]] = {}
    for target in TARGETS:
        ranked = sorted(
            diagnostics,
            key=lambda node: diagnostics[node][target]["precision_score"],
            reverse=True,
        )
        admitted = [
            node for node in ranked
            if diagnostics[node][target]["inlier_fraction_20deg"] >= MIN_CALIBRATION_INLIER_FRACTION
        ][:MAX_TARGET_NODES]
        if len(admitted) < MIN_TARGET_NODES:
            admitted = ranked[:MIN_TARGET_NODES]
        selected[target] = admitted
        for node in diagnostics:
            diagnostics[node][target]["selected_for_geometry"] = node in admitted
    return diagnostics, selected


def posterior_bearing(observation: np.ndarray, prediction: np.ndarray, reliability: dict) -> tuple[np.ndarray, np.ndarray, float]:
    observation = np.asarray(observation, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    sigma = np.asarray(
        [reliability["sigma_azimuth_deg"], reliability["sigma_zenith_deg"]],
        dtype=float,
    )
    observation_unwrapped = np.asarray(
        [prediction[0] + float(base.wrap_difference(observation[0], prediction[0])), observation[1]],
        dtype=float,
    )
    observation_precision = 1.0 / np.square(sigma)
    prior_precision = np.full(2, 1.0 / HIDDEN_BEARING_PRIOR_SIGMA_DEG**2)
    posterior_variance = 1.0 / (observation_precision + prior_precision)
    posterior = posterior_variance * (
        observation_precision * observation_unwrapped + prior_precision * prediction
    )
    posterior[0] %= 360.0
    innovation = np.asarray(
        [float(base.wrap_difference(observation[0], prediction[0])), observation[1] - prediction[1]],
        dtype=float,
    )
    normalized_cost = float(np.sum(np.square(innovation) / (np.square(sigma) + HIDDEN_BEARING_PRIOR_SIGMA_DEG**2)))
    return posterior, posterior_variance, normalized_cost


def weighted_ray_solution(
    observations: dict[int, np.ndarray],
    variances: dict[int, np.ndarray],
    nodes: dict[int, np.ndarray],
) -> tuple[np.ndarray | None, list[int], float, np.ndarray | None, float]:
    active = list(observations)
    if len(active) < MIN_TARGET_NODES:
        return None, [], float("inf"), None, float("inf")
    robust_weights = {node: 1.0 for node in active}
    position = None
    matrix = None
    for _ in range(5):
        matrix = np.zeros((3, 3), dtype=float)
        rhs = np.zeros(3, dtype=float)
        for node in active:
            vector = base.direction(float(observations[node][0]), float(observations[node][1]))
            projector = np.eye(3) - np.outer(vector, vector)
            angular_variance = float(np.mean(variances[node]))
            precision = robust_weights[node] / max(math.radians(math.sqrt(angular_variance)) ** 2, 1e-8)
            matrix += precision * projector
            rhs += precision * (projector @ nodes[node])
        condition = float(np.linalg.cond(matrix))
        if not math.isfinite(condition):
            return None, [], condition, None, float("inf")
        position = np.linalg.pinv(matrix) @ rhs
        changed = False
        for node in active:
            residual = base.angle_cost(observations[node], base.predicted_angles(position, nodes[node]))
            scale = max(math.sqrt(float(np.mean(variances[node]))), 2.0)
            standardized = residual / scale
            new_weight = 1.0 if standardized <= 2.5 else 2.5 / standardized
            changed = changed or abs(new_weight - robust_weights[node]) > 1e-3
            robust_weights[node] = new_weight
        if not changed:
            break
    assert position is not None and matrix is not None
    residuals = {
        node: base.angle_cost(observations[node], base.predicted_angles(position, nodes[node]))
        for node in active
    }
    kept = [node for node in active if robust_weights[node] >= 0.20]
    if len(kept) < MIN_TARGET_NODES:
        kept = sorted(active, key=lambda node: residuals[node] / max(math.sqrt(float(np.mean(variances[node]))), 2.0))[:MIN_TARGET_NODES]
    residual_m = [
        math.radians(residuals[node]) * float(np.linalg.norm(position - nodes[node]))
        for node in kept
    ]
    sigma_m = max(float(np.median(residual_m)), 3.0)
    covariance = sigma_m**2 * np.linalg.pinv(matrix)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    covariance = eigenvectors @ np.diag(np.maximum(eigenvalues, 1.0)) @ eigenvectors.T
    normalized_residual = float(np.median([
        residuals[node] / max(math.sqrt(float(np.mean(variances[node]))), 2.0)
        for node in kept
    ]))
    return position, kept, float(np.linalg.cond(matrix)), covariance, normalized_residual


def node_options(
    candidates: np.ndarray,
    predicted_positions: list[np.ndarray],
    node_position: np.ndarray,
    previous: np.ndarray | None,
    node: int,
    reliability: dict[int, dict[int, dict]],
) -> list[dict]:
    options = []
    for permutation in PERMUTATIONS:
        assigned = candidates[list(permutation)]
        cost = 0.0
        for target in TARGETS:
            prediction = base.predicted_angles(predicted_positions[target], node_position)
            _, _, target_cost = posterior_bearing(assigned[target], prediction, reliability[node][target])
            cost += target_cost
            if previous is not None:
                continuity_sigma = max(
                    reliability[node][target]["sigma_azimuth_deg"],
                    reliability[node][target]["sigma_zenith_deg"],
                    5.0,
                )
                cost += 0.15 * base.angle_cost(assigned[target], previous[target]) ** 2 / continuity_sigma**2
        options.append({"permutation": permutation, "assigned": assigned, "cheap_cost": float(cost)})
    return sorted(options, key=lambda row: row["cheap_cost"])[:TOP_PER_NODE]


def global_associate(
    node_candidates: dict[int, np.ndarray],
    nodes: dict[int, np.ndarray],
    predicted_positions: list[np.ndarray],
    predicted_covariances: list[np.ndarray],
    previous_observations: dict[int, np.ndarray] | None,
    reliability: dict[int, dict[int, dict]],
    selected_nodes: dict[int, list[int]],
) -> tuple[dict, dict[int, np.ndarray]]:
    ordered_nodes = sorted(node_candidates)
    candidates_per_node = [
        node_options(
            node_candidates[node],
            predicted_positions,
            nodes[node],
            None if previous_observations is None else previous_observations[node],
            node,
            reliability,
        )
        for node in ordered_nodes
    ]
    cheap_hypotheses = [
        (sum(item["cheap_cost"] for item in choice), choice)
        for choice in itertools.product(*candidates_per_node)
    ]
    cheap_hypotheses.sort(key=lambda item: item[0])
    best = None
    best_assigned = None
    for cheap_cost, choice in cheap_hypotheses[:TOP_GLOBAL]:
        assigned_by_node = {node: choice[index]["assigned"] for index, node in enumerate(ordered_nodes)}
        positions, inliers, conditions, covariances = [], [], [], []
        hidden_bearings: dict[int, dict[int, np.ndarray]] = {target: {} for target in TARGETS}
        hidden_variances: dict[int, dict[int, np.ndarray]] = {target: {} for target in TARGETS}
        reprojection_cost = 0.0
        position_cost = 0.0
        for target in TARGETS:
            for node in selected_nodes[target]:
                prediction = base.predicted_angles(predicted_positions[target], nodes[node])
                bearing, variance, _ = posterior_bearing(
                    assigned_by_node[node][target], prediction, reliability[node][target]
                )
                hidden_bearings[target][node] = bearing
                hidden_variances[target][node] = variance
            position, target_inliers, condition, covariance, normalized_residual = weighted_ray_solution(
                hidden_bearings[target], hidden_variances[target], nodes
            )
            positions.append(position)
            inliers.append(target_inliers)
            conditions.append(condition)
            covariances.append(covariance)
            reprojection_cost += normalized_residual + 0.5 * (len(selected_nodes[target]) - len(target_inliers))
            if position is None:
                position_cost += 25.0
            else:
                innovation = position - predicted_positions[target]
                scale = predicted_covariances[target] + (covariance if covariance is not None else np.eye(3) * 10000.0)
                position_cost += min(float(innovation.T @ np.linalg.pinv(scale) @ innovation), 25.0)
        score = float(0.20 * cheap_cost + 2.0 * reprojection_cost + position_cost)
        record = {
            "score": score,
            "cheap_cost_normalized": float(cheap_cost),
            "reprojection_cost_normalized": float(reprojection_cost),
            "position_cost_mahalanobis": float(position_cost),
            "positions": positions,
            "inliers": inliers,
            "conditions": conditions,
            "covariances": covariances,
            "hidden_bearings": hidden_bearings,
            "hidden_variances": hidden_variances,
            "permutations": {
                str(node): "-".join(str(value + 1) for value in choice[index]["permutation"])
                for index, node in enumerate(ordered_nodes)
            },
        }
        if best is None or record["score"] < best["score"]:
            best = record
            best_assigned = assigned_by_node
    assert best is not None and best_assigned is not None
    return best, best_assigned


def motion_matrices(dt: float) -> tuple[np.ndarray, np.ndarray]:
    transition = np.block([
        [np.eye(3), dt * np.eye(3)],
        [np.zeros((3, 3)), np.eye(3)],
    ])
    gain = np.vstack([0.5 * dt**2 * np.eye(3), dt * np.eye(3)])
    process = ACCELERATION_STD_MPS2**2 * gain @ gain.T
    return transition, process


def predict_state(state: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transition, process = motion_matrices(FRAME_DT_S)
    return transition @ state, transition @ covariance @ transition.T + process


def update_state(
    predicted_state: np.ndarray,
    predicted_covariance: np.ndarray,
    measurement: np.ndarray | None,
    measurement_covariance: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if measurement is None or measurement_covariance is None:
        return predicted_state, predicted_covariance, False
    innovation = measurement - predicted_state[:3]
    if float(np.linalg.norm(innovation)) > MAX_INNOVATION_M:
        return predicted_state, predicted_covariance, False
    observation = np.hstack([np.eye(3), np.zeros((3, 3))])
    innovation_covariance = observation @ predicted_covariance @ observation.T + measurement_covariance
    gain = predicted_covariance @ observation.T @ np.linalg.pinv(innovation_covariance)
    state = predicted_state + gain @ innovation
    identity = np.eye(6)
    residual = identity - gain @ observation
    covariance = residual @ predicted_covariance @ residual.T + gain @ measurement_covariance @ gain.T
    covariance = 0.5 * (covariance + covariance.T)
    return state, covariance, True


def ospa_order2(estimate: np.ndarray, truth: np.ndarray) -> float:
    costs = []
    for permutation in PERMUTATIONS:
        squared = [float(np.sum(np.square(estimate[index] - truth[permutation[index]]))) for index in TARGETS]
        costs.append(math.sqrt(sum(squared) / len(TARGETS)))
    return min(costs)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scalar_metrics(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p90": None, "maximum": None}
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "maximum": float(np.max(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--nod", type=Path, required=True)
    parser.add_argument("--gps-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-seconds", type=float, default=30.0)
    parser.add_argument("--frame-limit", type=int)
    args = parser.parse_args()

    nodes, ip_to_node = base.parse_nodes(args.nod)
    deal_by_node: dict[int, np.ndarray] = {}
    oracle_by_node: dict[int, np.ndarray] = {}
    input_paths: dict[int, Path] = {}
    for deal_path in sorted(args.data_dir.glob("deal_doa_*.txt")):
        suffix = int(deal_path.name.split("_")[2])
        node = ip_to_node[suffix]
        deal = base.read_numeric(deal_path, 8)
        gps_doa = base.read_numeric(args.data_dir / f"gps_doa_{suffix}.txt", 11)
        deal_by_node[node] = deal
        oracle_by_node[node] = base.gps_oracle_angles(deal, gps_doa)
        input_paths[node] = deal_path
    active_nodes = sorted(deal_by_node)
    if len(active_nodes) != 7:
        raise RuntimeError(f"expected seven historical nodes, found {active_nodes}")
    frame_count = min(len(deal_by_node[node]) for node in active_nodes)
    if args.frame_limit is not None:
        frame_count = min(frame_count, args.frame_limit)
    deal_by_node = {node: deal_by_node[node][:frame_count] for node in active_nodes}
    oracle_by_node = {node: oracle_by_node[node][:frame_count] for node in active_nodes}
    start_hhmmss = int(round(deal_by_node[active_nodes[0]][0, 7]))
    calibration_frames = int(round(args.calibration_seconds / FRAME_DT_S))
    if calibration_frames < 10 or calibration_frames >= frame_count:
        raise RuntimeError("calibration duration is invalid")

    transforms, candidates_by_node, oracle_permutations = {}, {}, {}
    for node in active_nodes:
        raw_candidates = deal_by_node[node][:, 1:7].reshape(frame_count, 3, 2)
        transforms[node] = base.calibrate_transform(raw_candidates, oracle_by_node[node], calibration_frames)
        candidates_by_node[node] = np.asarray([base.apply_transform(frame, transforms[node]) for frame in raw_candidates])
        oracle_permutations[node] = [
            base.permutation_cost(candidates_by_node[node][frame], oracle_by_node[node][frame, :, :2])[0]
            for frame in range(frame_count)
        ]
    reliability, selected_nodes = calibration_precision(
        candidates_by_node, oracle_by_node, oracle_permutations, calibration_frames
    )

    gps1 = base.parse_gps(args.gps_root / "GPS1_plane1.gps")
    gps2 = base.parse_gps(args.gps_root / "GPS2_plane1.gps")
    tracks = [
        base.fuse_tracks(gps1, gps2),
        base.parse_gps(args.gps_root / "GPS3_plane2.gps"),
        base.parse_gps(args.gps_root / "GPS4_plane2to3.gps"),
    ]
    truth = [base.interpolate_track(deal_by_node[active_nodes[0]], track) for track in tracks]
    velocity_start = max(0, calibration_frames - int(round(5.0 / FRAME_DT_S)))
    elapsed = max((calibration_frames - 1 - velocity_start) * FRAME_DT_S, FRAME_DT_S)
    states, state_covariances = [], []
    for target in TARGETS:
        velocity = (truth[target][calibration_frames - 1] - truth[target][velocity_start]) / elapsed
        states.append(np.concatenate([truth[target][calibration_frames - 1], velocity]))
        state_covariances.append(np.diag([INITIAL_POSITION_STD_M**2] * 3 + [INITIAL_VELOCITY_STD_MPS**2] * 3))

    target_rows = {target: [] for target in TARGETS}
    frame_rows, identity_rows = [], []
    errors = {target: [] for target in TARGETS}
    jumps = {target: [] for target in TARGETS}
    valid_updates = {target: 0 for target in TARGETS}
    previous_positions = [None, None, None]
    previous_observations = None
    covariance_psd = []
    post_identity_all, post_identity_selected = [], []
    ospa_values = []

    for frame in range(frame_count):
        calibration_frame = frame < calibration_frames
        node_candidates = {node: candidates_by_node[node][frame] for node in active_nodes}
        if calibration_frame:
            assigned = {
                node: node_candidates[node][list(oracle_permutations[node][frame])]
                for node in active_nodes
            }
            continue
        predicted_states, predicted_state_covariances = [], []
        for target in TARGETS:
            predicted_state, predicted_covariance = predict_state(states[target], state_covariances[target])
            predicted_states.append(predicted_state)
            predicted_state_covariances.append(predicted_covariance)
        decision, assigned = global_associate(
            node_candidates,
            nodes,
            [state[:3] for state in predicted_states],
            [covariance[:3, :3] for covariance in predicted_state_covariances],
            previous_observations,
            reliability,
            selected_nodes,
        )
        for target in TARGETS:
            states[target], state_covariances[target], updated = update_state(
                predicted_states[target],
                predicted_state_covariances[target],
                decision["positions"][target],
                decision["covariances"][target],
            )
            valid_updates[target] += int(updated)
            covariance_psd.append(bool(np.all(np.linalg.eigvalsh(state_covariances[target]) >= -1e-8)))
            error = float(np.linalg.norm(states[target][:3] - truth[target][frame]))
            errors[target].append(error)
            if previous_positions[target] is not None:
                jumps[target].append(float(np.linalg.norm(states[target][:3] - previous_positions[target])))
            previous_positions[target] = states[target][:3].copy()
            target_rows[target].append({
                "frame_index": frame,
                "time_hhmmss": float(deal_by_node[active_nodes[0]][frame, 7]),
                "px": float(states[target][0]),
                "py": float(states[target][1]),
                "pz": float(states[target][2]),
                "vx": float(states[target][3]),
                "vy": float(states[target][4]),
                "vz": float(states[target][5]),
                "measurement_updated": updated,
                "position_error_m_offline": error,
                "condition_number": float(decision["conditions"][target]),
                "inlier_nodes": "-".join(str(node) for node in decision["inliers"][target]),
                **{f"p_{row}{column}": float(state_covariances[target][row, column]) for row in range(6) for column in range(6)},
            })
        estimated_positions = np.asarray([states[target][:3] for target in TARGETS])
        truth_positions = np.asarray([truth[target][frame] for target in TARGETS])
        ospa_values.append(ospa_order2(estimated_positions, truth_positions))
        frame_all = []
        frame_selected = []
        for node in active_nodes:
            chosen = tuple(int(value) - 1 for value in decision["permutations"][str(node)].split("-"))
            oracle = oracle_permutations[node][frame]
            for target in TARGETS:
                correct = chosen[target] == oracle[target]
                frame_all.append(correct)
                if node in selected_nodes[target]:
                    frame_selected.append(correct)
                identity_rows.append({
                    "frame_index": frame,
                    "node": node,
                    "target": target + 1,
                    "selected_for_geometry": node in selected_nodes[target],
                    "identity_correct_offline": correct,
                    "candidate_index": chosen[target] + 1,
                })
        post_identity_all.extend(frame_all)
        post_identity_selected.extend(frame_selected)
        frame_rows.append({
            "frame_index": frame,
            "time_hhmmss": float(deal_by_node[active_nodes[0]][frame, 7]),
            "global_score": float(decision["score"]),
            "identity_accuracy_all_offline": float(np.mean(frame_all)),
            "identity_accuracy_selected_offline": float(np.mean(frame_selected)),
            "ospa_order2_m_offline": ospa_values[-1],
        })
        previous_observations = assigned

    args.output.mkdir(parents=True, exist_ok=True)
    for target in TARGETS:
        write_csv(args.output / f"target{target + 1}_state_covariance.csv", target_rows[target])
    write_csv(args.output / "frame_diagnostics.csv", frame_rows)
    write_csv(args.output / "identity_diagnostics.csv", identity_rows)
    post_frames = frame_count - calibration_frames
    target_metrics = {
        f"target{target + 1}": {
            "valid_update_fraction": valid_updates[target] / max(post_frames, 1),
            "position_error_m": scalar_metrics(errors[target]),
            "state_step_m": scalar_metrics(jumps[target]),
        }
        for target in TARGETS
    }
    admission = {
        "selected_pair_identity_accuracy": float(np.mean(post_identity_selected)),
        "all_pair_identity_accuracy": float(np.mean(post_identity_all)),
        "target_valid_update_fractions": {
            f"target{target + 1}": valid_updates[target] / max(post_frames, 1)
            for target in TARGETS
        },
        "target_p90_state_steps_m": {
            f"target{target + 1}": scalar_metrics(jumps[target])["p90"] for target in TARGETS
        },
        "covariance_psd_fraction": float(np.mean(covariance_psd)),
        "ospa_order2_m": scalar_metrics(ospa_values),
    }
    admission["passed"] = bool(
        admission["selected_pair_identity_accuracy"] >= 0.90
        and all(value >= 0.90 for value in admission["target_valid_update_fractions"].values())
        and all(value is not None and value < 100.0 for value in admission["target_p90_state_steps_m"].values())
        and admission["covariance_psd_fraction"] >= 0.99
        and admission["ospa_order2_m"]["mean"] is not None
        and admission["ospa_order2_m"]["mean"] <= 150.0
    )
    manifest = {
        "task": "author-code-informed three-source target-specific reliability gate",
        "status": "passed" if admission["passed"] else "failed",
        "claim_status": "association_and_state_gate_only; PCE/APCE prohibited unless negative controls also pass",
        "source_mechanisms": {
            "node_specific_precision": "supplied DBN_MTT.py lambda_e update, transferred as calibration-frozen node-target angular precision",
            "hidden_bearing": "supplied DBN_MTT.py theta update with 10 deg prior covariance, transferred as a Gaussian posterior bearing",
            "state_covariance": "supplied constant-velocity F/Q structure, extended from 2-D four-state to 3-D six-state Joseph update",
            "not_claimed": "not an execution of the authors' private Baoding field code and not a full ReAVI implementation",
        },
        "inputs": {
            "data_dir": str(args.data_dir),
            "nod": str(args.nod),
            "gps_root": str(args.gps_root),
            "deal_hashes": {str(node): file_sha256(path) for node, path in input_paths.items()},
            "active_nodes": active_nodes,
        },
        "protocol": {
            "frame_dt_s": FRAME_DT_S,
            "frame_count": frame_count,
            "calibration_frames": calibration_frames,
            "calibration_seconds": args.calibration_seconds,
            "gps_used_after_calibration": False,
            "post_calibration_gps_role": "offline scoring only",
            "top_per_node": TOP_PER_NODE,
            "top_global": TOP_GLOBAL,
            "hidden_bearing_prior_sigma_deg": HIDDEN_BEARING_PRIOR_SIGMA_DEG,
            "selected_nodes_per_target": {f"target{target + 1}": selected_nodes[target] for target in TARGETS},
        },
        "node_transforms": transforms,
        "calibration_reliability": {
            str(node): {f"target{target + 1}": row for target, row in target_rows_by_node.items()}
            for node, target_rows_by_node in reliability.items()
        },
        "offline_metrics": {"targets": target_metrics, "admission": admission},
        "admission": admission,
        "scripts": {
            "script": str(Path(__file__)),
            "script_sha256": file_sha256(Path(__file__)),
            "base_script_sha256": file_sha256(Path(base.__file__)),
        },
    }
    (args.output / "author_reliability_gate.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": manifest["status"], "admission": admission, "selected_nodes": selected_nodes}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
