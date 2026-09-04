#!/usr/bin/env python3
"""GPS-free A2 joint assignment and motion-regularized DOA tracker.

This is an independent frontend experiment for the Baoding two-source record.
The raw frontend contributes two unlabelled MUSIC candidates per node and
frame.  GPS is used only to audit the declared calibration interval and to
score the held-out output after all acoustic decisions have been made.  During
the evaluation interval the tracker sees only transformed DOAs, calibration-
frozen angular reliabilities, candidate strengths, and its own predicted
states.

The implementation is deliberately conservative: boundary-valued elevation
peaks are retained with inflated variance and every assignment, inlier set,
condition number and covariance eigenvalue is exported for audit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

import shuangyuan_a2_input as a2
import shuangyuan_dual_association as base


TARGETS = (1, 2)
DEFAULT_NODES = (1, 2, 3, 5, 6, 7, 8, 11, 13)
MIN_SIGMA_DEG = 2.0
MAX_SIGMA_DEG = 50.0
MAX_ABSOLUTE_RESIDUAL_DEG = 50.0
STANDARDIZED_INLIER_LIMIT = 3.5
MAX_SPEED_MPS = 120.0
VELOCITY_SMOOTHING = 0.20
BOUNDARY_VARIANCE_FACTOR = 4.0
PRIOR_SIGMA_M = 160.0
MIN_INLIERS = 5
MAX_CONDITION = 25.0
MAX_RMS_DEG = 25.0
MIN_ASSIGNMENT_MARGIN = 0.05


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def angular_residual_deg(
    position: np.ndarray,
    node_position: tuple[float, float, float],
    observation: tuple[float, float],
) -> float:
    expected = base.truth_angles(tuple(float(value) for value in position), node_position)
    daz = (observation[0] - expected[0] + 180.0) % 360.0 - 180.0
    return math.hypot(daz, observation[1] - expected[1])


def candidate_angles(candidate: a2.DOACandidate) -> tuple[float, float]:
    return candidate.azimuth_deg, candidate.zenith_deg


def robust_scale(values: Iterable[float], floor: float = 0.25) -> float:
    values = list(values)
    if not values:
        return floor
    array = np.asarray(values, dtype=float)
    median = float(np.median(array))
    mad = float(1.4826 * np.median(np.abs(array - median)))
    return max(mad, floor)


def circular_difference_deg(first: float, second: float) -> float:
    return (first - second + 180.0) % 360.0 - 180.0


@dataclass
class MotionState:
    position: np.ndarray | None = None
    velocity: np.ndarray | None = None
    time_s: float | None = None

    def predict(self, time_s: float) -> np.ndarray | None:
        if self.position is None:
            return None
        if self.velocity is None or self.time_s is None:
            return self.position.copy()
        return self.position + self.velocity * max(float(time_s) - self.time_s, 0.0)

    def update(self, position: np.ndarray, time_s: float) -> None:
        if self.position is not None and self.time_s is not None:
            dt = max(float(time_s) - self.time_s, 1e-6)
            instant = (position - self.position) / dt
            speed = float(np.linalg.norm(instant))
            if speed > MAX_SPEED_MPS:
                instant *= MAX_SPEED_MPS / speed
            if self.velocity is None:
                self.velocity = instant
            else:
                self.velocity = (1.0 - VELOCITY_SMOOTHING) * self.velocity + VELOCITY_SMOOTHING * instant
        self.position = np.asarray(position, dtype=float).copy()
        self.time_s = float(time_s)


def ray_projection(observation: tuple[float, float]) -> np.ndarray:
    direction = np.asarray(base.direction(*observation), dtype=float)
    return np.eye(3, dtype=float) - np.outer(direction, direction)


def solve_weighted_rays(
    observations: Mapping[int, a2.DOACandidate],
    sigmas: Mapping[int, float],
    nodes: Mapping[int, tuple[float, float, float]],
    selected: Iterable[int],
    robust_weights: Mapping[int, float] | None = None,
    prior_position: np.ndarray | None = None,
    prior_sigma_m: float | None = None,
) -> tuple[np.ndarray | None, float, np.ndarray | None]:
    matrix = np.zeros((3, 3), dtype=float)
    target = np.zeros(3, dtype=float)
    for node in selected:
        candidate = observations[node]
        projection = ray_projection(candidate_angles(candidate))
        sigma_rad = max(math.radians(float(sigmas[node])), math.radians(MIN_SIGMA_DEG))
        if prior_position is None:
            weight = 1.0 / sigma_rad**2
        else:
            range_m = max(float(np.linalg.norm(prior_position - np.asarray(nodes[node], dtype=float))), 50.0)
            weight = 1.0 / (range_m * sigma_rad) ** 2
        if robust_weights is not None:
            weight *= max(float(robust_weights.get(node, 1.0)), 1e-3)
        matrix += weight * projection
        target += weight * projection @ np.asarray(nodes[node], dtype=float)
    if prior_position is not None and prior_sigma_m is not None:
        precision = 1.0 / max(float(prior_sigma_m), 1.0) ** 2
        matrix += precision * np.eye(3)
        target += precision * np.asarray(prior_position, dtype=float)
    condition = float(np.linalg.cond(matrix)) if np.all(np.isfinite(matrix)) else float("inf")
    if not math.isfinite(condition) or condition > 1e10:
        return None, condition, None
    try:
        position = np.linalg.solve(matrix, target)
        covariance = np.linalg.pinv(matrix, rcond=1e-10)
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 1e-6)
        covariance = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        return position, condition, covariance
    except np.linalg.LinAlgError:
        return None, condition, None


def robust_triangulate(
    observations: Mapping[int, a2.DOACandidate],
    sigmas: Mapping[int, float],
    nodes: Mapping[int, tuple[float, float, float]],
    prior_position: np.ndarray | None,
    prior_sigma_m: float = PRIOR_SIGMA_M,
) -> tuple[np.ndarray | None, list[int], float, float, np.ndarray | None]:
    available = sorted(observations)
    if len(available) < 3:
        return None, [], float("inf"), float("inf"), None
    candidates: list[tuple[float, int, float, np.ndarray, list[int], np.ndarray | None]] = []
    subsets = itertools.combinations(available, min(3, len(available)))
    for subset in subsets:
        position, condition, covariance = solve_weighted_rays(
            observations, sigmas, nodes, subset, prior_position=prior_position,
            prior_sigma_m=prior_sigma_m if prior_position is not None else None,
        )
        if position is None:
            continue
        residuals = {node: angular_residual_deg(position, nodes[node], candidate_angles(observations[node])) for node in available}
        standardized = {node: residuals[node] / max(float(sigmas[node]), MIN_SIGMA_DEG) for node in available}
        inliers = [node for node in available if standardized[node] <= STANDARDIZED_INLIER_LIMIT and residuals[node] <= MAX_ABSOLUTE_RESIDUAL_DEG]
        if len(inliers) < 3:
            continue
        clipped = float(np.mean([min(value * value, 16.0) for value in standardized.values()]))
        if prior_position is not None:
            clipped += min(float(np.linalg.norm(position - prior_position)) / prior_sigma_m, 4.0) ** 2 / len(available)
        candidates.append((clipped, -len(inliers), condition, position, inliers, covariance))
    if not candidates:
        return None, [], float("inf"), float("inf"), None
    _, _, condition, position, inliers, covariance = min(candidates, key=lambda value: value[:3])
    for _ in range(4):
        residuals = {node: angular_residual_deg(position, nodes[node], candidate_angles(observations[node])) for node in inliers}
        robust_weights = {node: min(1.0, 1.5 * float(sigmas[node]) / max(residuals[node], 1e-6)) for node in inliers}
        updated, condition, covariance = solve_weighted_rays(
            observations, sigmas, nodes, inliers, robust_weights,
            prior_position=prior_position,
            prior_sigma_m=prior_sigma_m if prior_position is not None else None,
        )
        if updated is None:
            break
        position = updated
    residual_values = [angular_residual_deg(position, nodes[node], candidate_angles(observations[node])) for node in inliers]
    rms = math.sqrt(sum(value * value for value in residual_values) / len(residual_values)) if residual_values else float("inf")
    return position, inliers, float(condition), float(rms), covariance


def learn_quality_model(
    rows_by_node: Mapping[int, list[a2.FrontendRow]],
    priors: Mapping[int, list[a2.FrozenIdentityPrior]],
    calibrations: Mapping[int, Mapping[str, float]],
    nodes: Mapping[int, tuple[float, float, float]],
    tracks: Mapping[int, Mapping[int, tuple[float, float, float]]],
    calibration_end_s: float,
    delay_s: int,
) -> dict[str, object]:
    values: dict[int, list[tuple[float, float]]] = {node: [] for node in rows_by_node}
    samples: dict[tuple[int, int], list[tuple[float, float]]] = {(node, target): [] for node in rows_by_node for target in TARGETS}
    for node, rows in rows_by_node.items():
        for row, prior in zip(rows, priors[node], strict=True):
            if row.time_s >= calibration_end_s:
                continue
            candidates = row.candidates(calibrations[node])
            for target, index in ((1, prior.target1_peak_index), (2, prior.target2_peak_index)):
                candidate = candidates[index]
                other = candidates[1 - index]
                strength, ratio = a2.candidate_quality(candidate, other)
                values[node].append((strength, ratio))
                truth = base.nearest_truth(tracks[target], base.shift_hhmmss(row.time_hhmmss, -delay_s))
                if truth is None:
                    continue
                expected = base.truth_angles(truth, nodes[node])
                daz = circular_difference_deg(candidate.azimuth_deg, expected[0])
                dzen = candidate.zenith_deg - expected[1]
                samples[(node, target)].append((0.5 * strength + 0.5 * ratio, math.hypot(daz, dzen)))
    normalization: dict[int, dict[str, float]] = {}
    for node, pairs in values.items():
        strength = [pair[0] for pair in pairs]
        ratio = [pair[1] for pair in pairs]
        normalization[node] = {
            "strength_median": float(np.median(strength)) if strength else 0.0,
            "strength_scale": robust_scale(strength),
            "ratio_median": float(np.median(ratio)) if ratio else 0.0,
            "ratio_scale": robust_scale(ratio),
        }
    node_target_sigma: dict[tuple[int, int], float] = {}
    quality_bins: dict[tuple[int, int], list[dict[str, float]]] = {}
    for key, pairs in samples.items():
        node, target = key
        residuals = [pair[1] for pair in pairs]
        raw = math.sqrt(np.mean(np.minimum(np.asarray(residuals, dtype=float), MAX_ABSOLUTE_RESIDUAL_DEG) ** 2)) if residuals else 30.0
        node_target_sigma[key] = float(np.clip(raw, MIN_SIGMA_DEG, MAX_SIGMA_DEG))
        ordered = sorted(pairs, key=lambda pair: pair[0])
        bins: list[dict[str, float]] = []
        for lower_q, upper_q in zip(np.linspace(0.0, 1.0, 5)[:-1], np.linspace(0.0, 1.0, 5)[1:], strict=True):
            if ordered:
                lo = float(np.quantile([pair[0] for pair in ordered], lower_q))
                hi = float(np.quantile([pair[0] for pair in ordered], upper_q))
                subset = [pair[1] for pair in ordered if lo <= pair[0] <= hi]
            else:
                lo, hi, subset = -6.0, 6.0, []
            sigma = math.sqrt(np.mean(np.minimum(np.asarray(subset or residuals or [30.0]), MAX_ABSOLUTE_RESIDUAL_DEG) ** 2))
            bins.append({"lower": lo, "upper": hi, "sigma_deg": float(np.clip(sigma, MIN_SIGMA_DEG, MAX_SIGMA_DEG)), "count": float(len(subset))})
        quality_bins[key] = bins
    return {
        "normalization": normalization,
        "node_target_sigma_deg": node_target_sigma,
        "quality_bins": quality_bins,
        "calibration_samples": {f"node{node}_target{target}": len(samples[(node, target)]) for node in rows_by_node for target in TARGETS},
    }


def quality_score(candidate: a2.DOACandidate, other: a2.DOACandidate, normalization: Mapping[str, float]) -> float:
    strength, ratio = a2.candidate_quality(candidate, other)
    z_strength = (strength - normalization["strength_median"]) / normalization["strength_scale"]
    z_ratio = (ratio - normalization["ratio_median"]) / normalization["ratio_scale"]
    return float(np.clip(0.5 * z_strength + 0.5 * z_ratio, -6.0, 6.0))


def candidate_sigma(
    node: int,
    target: int,
    candidate: a2.DOACandidate,
    other: a2.DOACandidate,
    model: Mapping[str, object],
) -> tuple[float, float]:
    normalization = model["normalization"][str(node)]
    score = quality_score(candidate, other, normalization)
    bins = model["quality_bins"][f"node{node}_target{target}"]
    selected = min(bins, key=lambda value: 0.0 if value["lower"] <= score <= value["upper"] else min(abs(score - value["lower"]), abs(score - value["upper"])))
    sigma = float(selected["sigma_deg"])
    if candidate.boundary:
        sigma *= BOUNDARY_VARIANCE_FACTOR
    return float(np.clip(sigma, MIN_SIGMA_DEG, MAX_SIGMA_DEG)), score


def assignment_for_node(
    candidates: tuple[a2.DOACandidate, a2.DOACandidate],
    predicted: Mapping[int, np.ndarray] | None,
    node: int,
    nodes: Mapping[int, tuple[float, float, float]],
    model: Mapping[str, object],
) -> tuple[tuple[int, int], float, float, dict[int, float], dict[int, float]]:
    costs: list[float] = []
    sigmas_by_assignment: list[dict[int, float]] = []
    scores_by_assignment: list[dict[int, float]] = []
    for permutation in ((0, 1), (1, 0)):
        total = 0.0
        sigmas: dict[int, float] = {}
        scores: dict[int, float] = {}
        for target, index in zip(TARGETS, permutation, strict=True):
            candidate, other = candidates[index], candidates[1 - index]
            sigma, score = candidate_sigma(node, target, candidate, other, model)
            sigmas[target] = sigma
            scores[target] = score
            cost = 0.15 * max(0.0, -score)
            if candidate.boundary:
                cost += 2.0
            if predicted is not None and predicted.get(target) is not None:
                expected = base.truth_angles(tuple(float(value) for value in predicted[target]), nodes[node])
                residual = math.hypot(circular_difference_deg(candidate.azimuth_deg, expected[0]), candidate.zenith_deg - expected[1])
                cost += (residual / max(sigma, MIN_SIGMA_DEG)) ** 2
            total += cost
        costs.append(total)
        sigmas_by_assignment.append(sigmas)
        scores_by_assignment.append(scores)
    winner = int(np.argmin(np.asarray(costs)))
    loser = 1 - winner
    margin = float(costs[loser] - costs[winner])
    return ((0, 1), (1, 0))[winner], float(costs[winner]), margin, sigmas_by_assignment[winner], scores_by_assignment[winner]


def calibrate_initial_states(
    rows_by_node: Mapping[int, list[a2.FrontendRow]],
    priors: Mapping[int, list[a2.FrozenIdentityPrior]],
    calibrations: Mapping[int, Mapping[str, float]],
    nodes: Mapping[int, tuple[float, float, float]],
    model: Mapping[str, object],
    calibration_end_s: float,
) -> dict[int, MotionState]:
    states = {target: MotionState() for target in TARGETS}
    histories: dict[int, list[tuple[float, np.ndarray]]] = {target: [] for target in TARGETS}
    frame_count = min(len(rows) for rows in rows_by_node.values())
    for frame_index in range(frame_count):
        reference = rows_by_node[min(rows_by_node)][frame_index]
        if reference.time_s >= calibration_end_s:
            break
        for target in TARGETS:
            observations: dict[int, a2.DOACandidate] = {}
            sigmas: dict[int, float] = {}
            for node, rows in rows_by_node.items():
                row = rows[frame_index]
                candidates = row.candidates(calibrations[node])
                prior = priors[node][frame_index]
                index = prior.target1_peak_index if target == 1 else prior.target2_peak_index
                candidate, other = candidates[index], candidates[1 - index]
                observations[node] = candidate
                sigmas[node] = candidate_sigma(node, target, candidate, other, model)[0]
            position, _, _, _, _ = robust_triangulate(observations, sigmas, nodes, None)
            if position is not None and np.all(np.isfinite(position)):
                histories[target].append((reference.time_s, position))
    for target in TARGETS:
        values = histories[target][-8:]
        if not values:
            continue
        time_s, position = values[-1]
        states[target].position = position.copy()
        states[target].time_s = float(time_s)
        if len(values) >= 2:
            times = np.asarray([item[0] for item in values], dtype=float)
            matrix = np.column_stack((times - times.mean(), np.ones(len(times))))
            velocity = np.asarray([np.linalg.lstsq(matrix, np.asarray([item[1][axis] for item in values]), rcond=None)[0][0] for axis in range(3)])
            speed = float(np.linalg.norm(velocity))
            if speed > MAX_SPEED_MPS:
                velocity *= MAX_SPEED_MPS / speed
            states[target].velocity = velocity
    return states


def load_truth_tracks(remote_root: Path) -> dict[int, dict[int, tuple[float, float, float]]]:
    gps_root = remote_root / "20171107保定实验" / "GPS_data"
    return {
        1: base.fuse(base.parse_gps(gps_root / "GPS1_plane1.gps"), base.parse_gps(gps_root / "GPS2_plane1.gps")),
        2: base.fuse(base.parse_gps(gps_root / "GPS3_plane2.gps"), base.parse_gps(gps_root / "GPS4_plane2to3.gps")),
    }


def score_rows(rows: list[dict[str, object]], track: Mapping[int, tuple[float, float, float]], delay_s: int) -> dict[str, object]:
    errors: list[float] = []
    for row in rows:
        if not bool(row["valid"]):
            continue
        truth = base.nearest_truth(track, base.shift_hhmmss(int(row["time_hhmmss"]), -delay_s))
        if truth is None:
            continue
        position = np.asarray([float(row[name]) for name in ("px", "py", "pz")], dtype=float)
        errors.append(float(np.linalg.norm(position - np.asarray(truth, dtype=float))))
    return {
        "frames": len(rows),
        "valid_frames": sum(bool(row["valid"]) for row in rows),
        "valid_fraction": sum(bool(row["valid"]) for row in rows) / len(rows) if rows else 0.0,
        "rmse_m": math.sqrt(float(np.mean(np.asarray(errors) ** 2))) if errors else None,
        "median_m": float(np.median(errors)) if errors else None,
        "p90_m": float(np.percentile(errors, 90.0)) if errors else None,
    }


def acoustic_pass(row: Mapping[str, object]) -> bool:
    return (
        bool(row["valid"])
        and int(row["inlier_nodes"]) >= MIN_INLIERS
        and float(row["condition_number"]) <= MAX_CONDITION
        and float(row["reprojection_rms_deg"]) <= MAX_RMS_DEG
        and float(row["assignment_margin_deg"]) >= MIN_ASSIGNMENT_MARGIN
    )


def contiguous_windows(rows_by_target: Mapping[int, list[dict[str, object]]], min_length: int = 5) -> list[dict[str, object]]:
    flags = [acoustic_pass(row) and acoustic_pass(rows_by_target[2][index]) for index, row in enumerate(rows_by_target[1])]
    windows: list[dict[str, object]] = []
    start: int | None = None
    for index, flag in enumerate(flags + [False]):
        if flag and start is None:
            start = index
        if not flag and start is not None:
            end = index - 1
            if end - start + 1 >= min_length:
                windows.append({
                    "start_frame_index": start,
                    "end_frame_index": end,
                    "length_frames": end - start + 1,
                    "start_time_s": float(rows_by_target[1][start]["time_s"]),
                    "end_time_s": float(rows_by_target[1][end]["time_s"]),
                })
            start = None
    return sorted(windows, key=lambda row: (-int(row["length_frames"]), int(row["start_frame_index"])))


def run(args: argparse.Namespace) -> None:
    nodes_ids = tuple(int(node) for node in args.nodes)
    rows_by_node = a2.load_frontend_bundle(args.frontend, nodes_ids)
    calibrations, delay_s = a2.load_frozen_orientation_calibrations(args.association_gate, nodes_ids)
    priors = a2.load_frozen_identity_priors(args.association, rows_by_node, calibrations)
    nodes = base.parse_nod(args.node_file)
    nodes = {node: nodes[node] for node in nodes_ids}
    tracks = load_truth_tracks(args.remote_root)
    model = learn_quality_model(rows_by_node, priors, calibrations, nodes, tracks, args.calibration_end_s, delay_s)
    states = calibrate_initial_states(rows_by_node, priors, calibrations, nodes, model, args.calibration_end_s)
    frame_count = min(len(rows) for rows in rows_by_node.values())
    output_by_target: dict[int, list[dict[str, object]]] = {target: [] for target in TARGETS}
    diagnostics: list[dict[str, object]] = []
    reference_rows = rows_by_node[nodes_ids[0]]
    for frame_index in range(frame_count):
        reference = reference_rows[frame_index]
        time_s = reference.time_s
        is_calibration = time_s < args.calibration_end_s
        predicted = {target: states[target].predict(time_s) for target in TARGETS}
        assignments: dict[int, dict[int, a2.DOACandidate]] = {target: {} for target in TARGETS}
        assignment_costs: list[float] = []
        assignment_margins: list[float] = []
        boundary_counts = {target: 0 for target in TARGETS}
        quality_scores: dict[int, list[float]] = {target: [] for target in TARGETS}
        for node in nodes_ids:
            row = rows_by_node[node][frame_index]
            candidates = row.candidates(calibrations[node])
            if is_calibration:
                prior = priors[node][frame_index]
                permutation = (prior.target1_peak_index, prior.target2_peak_index)
                cost, margin = 0.0, float("nan")
                sigmas = {}
                scores = {}
                for target, index in zip(TARGETS, permutation, strict=True):
                    candidate, other = candidates[index], candidates[1 - index]
                    sigmas[target], scores[target] = candidate_sigma(node, target, candidate, other, model)
            else:
                permutation, cost, margin, sigmas, scores = assignment_for_node(candidates, predicted, node, nodes, model)
            assignment_costs.append(float(cost))
            if math.isfinite(margin):
                assignment_margins.append(float(margin))
            for target, index in zip(TARGETS, permutation, strict=True):
                candidate = candidates[index]
                assignments[target][node] = candidate
                quality_scores[target].append(float(scores[target]))
                if candidate.boundary:
                    boundary_counts[target] += 1
        frame_result: dict[int, tuple[np.ndarray | None, list[int], float, float, np.ndarray | None]] = {}
        for target in TARGETS:
            sigmas = {}
            for node in nodes_ids:
                candidate = assignments[target][node]
                other = rows_by_node[node][frame_index].candidates(calibrations[node])[1 - candidate.peak_index]
                sigmas[node] = candidate_sigma(node, target, candidate, other, model)[0]
            result = robust_triangulate(assignments[target], sigmas, nodes, predicted[target])
            frame_result[target] = result
            position, inliers, condition, rms, covariance = result
            output_by_target[target].append({
                "frame_index": frame_index,
                "time_s": time_s,
                "time_hhmmss": reference.time_hhmmss,
                "calibration_frame": is_calibration,
                "valid": position is not None,
                "px": float(position[0]) if position is not None else "",
                "py": float(position[1]) if position is not None else "",
                "pz": float(position[2]) if position is not None else "",
                "inlier_nodes": len(inliers),
                "inlier_node_ids": ";".join(str(node) for node in inliers),
                "condition_number": float(condition),
                "reprojection_rms_deg": float(rms),
                "covariance_min_eigenvalue_m2": float(np.min(np.linalg.eigvalsh(covariance))) if covariance is not None else "",
                "covariance_psd": bool(covariance is not None and np.min(np.linalg.eigvalsh(covariance)) >= -1e-8),
                "boundary_candidate_fraction": boundary_counts[target] / len(nodes_ids),
                "mean_quality_score": float(np.mean(quality_scores[target])) if quality_scores[target] else "",
            })
            if position is not None:
                states[target].update(position, time_s)
        diagnostics.append({
            "frame_index": frame_index,
            "time_s": time_s,
            "time_hhmmss": reference.time_hhmmss,
            "calibration_frame": is_calibration,
            "assignment_cost_deg2": float(np.mean(assignment_costs)) if assignment_costs else "",
            "assignment_margin_deg": float(np.mean(assignment_margins)) if assignment_margins else "",
            "target1_valid": bool(frame_result[1][0] is not None),
            "target2_valid": bool(frame_result[2][0] is not None),
            "target1_inlier_nodes": len(frame_result[1][1]),
            "target2_inlier_nodes": len(frame_result[2][1]),
            "target1_condition_number": frame_result[1][2],
            "target2_condition_number": frame_result[2][2],
            "target1_reprojection_rms_deg": frame_result[1][3],
            "target2_reprojection_rms_deg": frame_result[2][3],
            "target1_boundary_candidate_fraction": boundary_counts[1] / len(nodes_ids),
            "target2_boundary_candidate_fraction": boundary_counts[2] / len(nodes_ids),
            "joint_acoustic_pass": bool(
                acoustic_pass(output_by_target[1][-1])
                and acoustic_pass(output_by_target[2][-1])
                and (is_calibration or (math.isfinite(float(np.mean(assignment_margins))) and np.mean(assignment_margins) >= MIN_ASSIGNMENT_MARGIN))
            ),
        })
    for target in TARGETS:
        rows = output_by_target[target]
        split = {"calibration": [row for row in rows if row["calibration_frame"]], "evaluation": [row for row in rows if not row["calibration_frame"]]}
        target_root = args.output / f"target{target}"
        write_csv(target_root / "a2_target_track.csv", rows)
        write_json(target_root / "a2_metrics.json", {name: score_rows(value, tracks[target], delay_s) for name, value in split.items()})
    write_csv(args.output / "a2_frame_diagnostics.csv", diagnostics)
    windows = contiguous_windows(output_by_target, min_length=args.min_window_frames)
    scored_windows: list[dict[str, object]] = []
    for window in windows:
        start, end = int(window["start_frame_index"]), int(window["end_frame_index"])
        entry = dict(window)
        for target in TARGETS:
            subset = output_by_target[target][start:end + 1]
            entry[f"target{target}_offline_score"] = score_rows(subset, tracks[target], delay_s)
        scored_windows.append(entry)
    write_json(args.output / "a2_acoustic_window_candidates.json", {
        "selection_rule": "joint acoustic gates only: both targets valid, >=5 inliers, condition <=25, angular RMS <=25 deg, assignment margin >=0.05 deg; GPS scores appended after selection",
        "candidate_count": len(scored_windows),
        "candidates": scored_windows,
    })
    provenance = {
        "task": "Baoding A2 direct-DOA candidate assignment with confidence and constant-velocity prior",
        "protocol": {
            "calibration_interval": f"[{reference_rows[0].time_s}, {args.calibration_end_s}) s",
            "evaluation_interval": f"[{next((row['time_s'] for row in reference_rows if row.time_s >= args.calibration_end_s), None)}, {reference_rows[-1].time_s}] s",
            "raw_update_interval_s": float(np.median(np.diff([row.time_s for row in reference_rows]))),
            "nodes": list(nodes_ids),
            "gps_runtime_evaluation": False,
            "gps_role": "orientation/identity/reliability calibration before evaluation and offline scoring only after output freeze",
        },
        "model": model,
        "gates": {
            "min_inliers": MIN_INLIERS,
            "max_condition_number": MAX_CONDITION,
            "max_reprojection_rms_deg": MAX_RMS_DEG,
            "min_assignment_margin_deg": MIN_ASSIGNMENT_MARGIN,
            "boundary_variance_factor": BOUNDARY_VARIANCE_FACTOR,
            "prior_sigma_m": PRIOR_SIGMA_M,
        },
        "sources": {
            "frontend": str(args.frontend),
            "frontend_manifest": a2.validate_frontend_bundle(args.frontend, nodes_ids),
            "association_gate": str(args.association_gate),
            "association_gate_sha256": sha256(args.association_gate),
            "node_file": str(args.node_file),
        },
        "outputs": {
            "diagnostics": str(args.output / "a2_frame_diagnostics.csv"),
            "windows": str(args.output / "a2_acoustic_window_candidates.json"),
        },
    }
    write_json(args.output / "a2_provenance.json", provenance)
    summary = {
        "output": str(args.output),
        "frames": frame_count,
        "calibration_frames": sum(bool(row["calibration_frame"]) for row in diagnostics),
        "evaluation_frames": sum(not bool(row["calibration_frame"]) for row in diagnostics),
        "windows": len(scored_windows),
        "longest_window": scored_windows[0] if scored_windows else None,
        "target_metrics": {
            str(target): score_rows([row for row in output_by_target[target] if not row["calibration_frame"]], tracks[target], delay_s)
            for target in TARGETS
        },
    }
    write_json(args.output / "a2_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-root", type=Path, default=Path("<PRIVATE_DATA_ROOT>/2017保定实验/2017保定实验"))
    parser.add_argument("--frontend", type=Path, required=True)
    parser.add_argument("--association", type=Path, required=True)
    parser.add_argument("--association-gate", type=Path, required=True)
    parser.add_argument("--node-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-end-s", type=float, default=46561.0)
    parser.add_argument("--min-window-frames", type=int, default=24)
    parser.add_argument("--nodes", type=int, nargs="+", default=list(DEFAULT_NODES))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
