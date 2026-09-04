#!/usr/bin/env python3
"""A2 direct DOA-map tracker for the 2017 Baoding two-source record.

This is a bounded, calibration-frozen frontend experiment.  It consumes the
two rank-paired MUSIC candidates saved by the historical replay, keeps both
candidate permutations during held-out tracking, and estimates a 6-D constant
velocity state from weighted multi-node rays.  GPS is read only by the
calibration and offline scoring helpers; the held-out update path never opens
GPS files.

The input candidates are not joint 2-D MUSIC peaks and their peak strengths
are only relative MUSIC-contrast proxies.  The output therefore must not be
described as a private Zhang-DBN reproduction or as measured acoustic SNR.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from . import shuangyuan_dual_association as base
    from .shuangyuan_a2_input import (
        DOACandidate,
        FrontendRow,
        candidate_quality,
        load_frontend_bundle,
        load_frozen_identity_priors,
        load_frozen_orientation_calibrations,
        validate_frontend_bundle,
    )
except ImportError:  # direct execution from the remote code directory
    import shuangyuan_dual_association as base
    from shuangyuan_a2_input import (
        DOACandidate,
        FrontendRow,
        candidate_quality,
        load_frontend_bundle,
        load_frozen_identity_priors,
        load_frozen_orientation_calibrations,
        validate_frontend_bundle,
    )


NODES = tuple(base.NODES)
TARGETS = (1, 2)
CALIBRATION_START_S = 46540
CALIBRATION_END_S = 46561
EVALUATION_END_S = 46741
MIN_SIGMA_DEG = 2.0
MAX_SIGMA_DEG = 75.0
BOUNDARY_INFLATION = 6.0
MAX_ABS_RESIDUAL_DEG = 65.0
STANDARDIZED_INLIER_LIMIT = 3.5
HUBER_DEG = 20.0
Q_ACCEL = 16.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(row for row in rows)


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def percentile(values: Sequence[float], quantile: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), quantile))


def wrap_rad(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def tangent_residual_deg(observed: Tuple[float, float], predicted: Tuple[float, float]) -> np.ndarray:
    """Tangent-plane [sin(zenith)*dazimuth, dzenith] residual in radians."""
    az, zen = math.radians(observed[0]), math.radians(observed[1])
    paz, pzen = math.radians(predicted[0]), math.radians(predicted[1])
    return np.asarray([math.sin(pzen) * wrap_rad(az - paz), zen - pzen], dtype=float)


def tangent_norm_deg(observed: Tuple[float, float], predicted: Tuple[float, float]) -> float:
    return math.degrees(float(np.linalg.norm(tangent_residual_deg(observed, predicted))))


def direction(azimuth_deg: float, zenith_deg: float) -> np.ndarray:
    zen = math.radians(float(np.clip(zenith_deg, 0.5, 89.5)))
    az = math.radians(azimuth_deg)
    elevation = math.pi / 2.0 - zen
    return np.asarray(
        [math.cos(elevation) * math.cos(az), math.cos(elevation) * math.sin(az), math.sin(elevation)],
        dtype=float,
    )


def truth_angles(position: Sequence[float], node: Sequence[float]) -> Tuple[float, float]:
    delta = np.asarray(position, dtype=float) - np.asarray(node, dtype=float)
    azimuth = math.degrees(math.atan2(delta[1], delta[0])) % 360.0
    zenith = 90.0 - math.degrees(math.atan2(delta[2], math.hypot(delta[0], delta[1])))
    return azimuth, zenith


def load_gps_track(path_a: Path, path_b: Path) -> Dict[int, Tuple[float, float, float]]:
    return base.fuse(base.parse_gps(path_a), base.parse_gps(path_b))


def load_truth_tracks(gps_root: Path) -> Dict[int, Dict[int, Tuple[float, float, float]]]:
    return {
        1: load_gps_track(gps_root / "GPS1_plane1.gps", gps_root / "GPS2_plane1.gps"),
        2: load_gps_track(gps_root / "GPS3_plane2.gps", gps_root / "GPS4_plane2to3.gps"),
    }


def nearest_truth(track: Mapping[int, Tuple[float, float, float]], second: int) -> Optional[Tuple[float, float, float]]:
    if second in track:
        return track[second]
    near = [key for key in track if abs(key - second) <= 1]
    if not near:
        return None
    return track[min(near, key=lambda key: abs(key - second))]


def frame_bins(bundle: Mapping[int, Sequence[FrontendRow]], start_s: int, end_s: int) -> Dict[int, List[int]]:
    reference = bundle[min(bundle)]
    bins: Dict[int, List[int]] = {}
    for index, row in enumerate(reference):
        second = int(math.floor(row.time_s + 1e-9))
        if start_s <= second < end_s:
            bins.setdefault(second, []).append(index)
    return bins


def row_candidates(row: FrontendRow, calibration: Mapping[str, float]) -> Tuple[DOACandidate, DOACandidate]:
    return row.candidates(calibration)


def quality_value(first: DOACandidate, second: DOACandidate) -> float:
    strength, ratio = candidate_quality(first, second)
    # Relative contrast proxy; no physical SNR interpretation is implied.
    return float(strength + 0.5 * ratio)


def robust_scale(values: Sequence[float], fallback: float = 10.0) -> float:
    if not values:
        return fallback
    array = np.asarray(values, dtype=float)
    med = float(np.median(array))
    mad = float(np.median(np.abs(array - med)))
    scale = 1.4826 * mad
    if scale < 1e-6:
        scale = float(np.sqrt(np.mean(np.square(array))))
    return max(scale, fallback)


def fit_quality_model(
    bundle: Mapping[int, Sequence[FrontendRow]],
    calibrations: Mapping[int, Mapping[str, float]],
    priors: Mapping[int, Sequence[Any]],
    nodes_abs: Mapping[int, Sequence[float]],
    tracks: Mapping[int, Mapping[int, Tuple[float, float, float]]],
    centre: np.ndarray,
    calibration_start_s: int,
    calibration_end_s: int,
    delay_s: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Fit frozen node-target angular scales and quality normalization."""
    errors: Dict[Tuple[int, int], List[float]] = {(node, target): [] for node in NODES for target in TARGETS}
    qualities: Dict[Tuple[int, int], List[float]] = {(node, target): [] for node in NODES for target in TARGETS}
    boundary: Dict[Tuple[int, int], int] = {(node, target): 0 for node in NODES for target in TARGETS}
    for node in NODES:
        node_local = np.asarray(nodes_abs[node], dtype=float) - centre
        for index, row in enumerate(bundle[node]):
            second = int(math.floor(row.time_s + 1e-9))
            if not (calibration_start_s <= second < calibration_end_s):
                continue
            truths: List[Optional[Tuple[float, float]]] = []
            for target in TARGETS:
                point = nearest_truth(tracks[target], base.seconds_to_hhmmss(second - delay_s))
                truths.append(truth_angles(point, node_local + centre) if point is not None else None)
            if any(value is None for value in truths):
                continue
            candidates = row_candidates(row, calibrations[node])
            prior = priors[node][index]
            candidate_indices = {1: prior.target1_peak_index, 2: prior.target2_peak_index}
            for target in TARGETS:
                candidate = candidates[candidate_indices[target]]
                alternative = candidates[1 - candidate_indices[target]]
                error = tangent_norm_deg(
                    (candidate.azimuth_deg, candidate.zenith_deg),
                    truths[target - 1],  # type: ignore[arg-type]
                )
                key = (node, target)
                errors[key].append(error)
                qualities[key].append(quality_value(candidate, alternative))
                boundary[key] += int(candidate.boundary)
    model_sigma: Dict[Tuple[int, int], float] = {}
    quality_med: Dict[Tuple[int, int], float] = {}
    quality_scale: Dict[Tuple[int, int], float] = {}
    reliability_rows: List[Dict[str, Any]] = []
    for node in NODES:
        for target in TARGETS:
            key = (node, target)
            values = errors[key]
            qs = qualities[key]
            sigma = float(np.clip(robust_scale(values, 5.0), MIN_SIGMA_DEG, 45.0))
            qmed = float(np.median(qs)) if qs else 0.0
            qscale = robust_scale(qs, 0.25)
            model_sigma[key] = sigma
            quality_med[key] = qmed
            quality_scale[key] = qscale
            reliability_rows.append({
                "node": node,
                "target": target,
                "calibration_samples": len(values),
                "calibration_angular_median_deg": float(np.median(values)) if values else "",
                "calibration_angular_p90_deg": percentile(values, 90.0) or "",
                "frozen_sigma_deg": sigma,
                "quality_median_proxy": qmed,
                "quality_mad_scale_proxy": qscale,
                "boundary_count": boundary[key],
                "quality_name": "relative MUSIC-contrast proxy",
            })
    model = {
        "node_target_sigma_deg": model_sigma,
        "quality_median": quality_med,
        "quality_scale": quality_scale,
        "min_sigma_deg": MIN_SIGMA_DEG,
        "max_sigma_deg": MAX_SIGMA_DEG,
        "boundary_inflation": BOUNDARY_INFLATION,
        "quality_feature": "log geometric peak strength + 0.5 log peak-strength ratio",
        "gps_used_only_in_calibration": True,
    }
    return model, reliability_rows


def sigma_for(
    candidate: DOACandidate,
    alternative: DOACandidate,
    node: int,
    target: int,
    model: Mapping[str, Any],
    use_quality: bool,
) -> float:
    key = (node, target)
    base_sigma = float(model["node_target_sigma_deg"].get(key, 10.0))
    if use_quality:
        quality = quality_value(candidate, alternative)
        scale = max(float(model["quality_scale"].get(key, 0.25)), 0.1)
        centred = float(model["quality_median"].get(key, 0.0))
        # Higher relative contrast receives a smaller, calibration-frozen scale.
        base_sigma *= math.exp(-0.20 * float(np.clip((quality - centred) / scale, -3.0, 3.0)))
    if candidate.boundary:
        base_sigma *= BOUNDARY_INFLATION
    return float(np.clip(base_sigma, MIN_SIGMA_DEG, MAX_SIGMA_DEG))


def cv_matrices(dt: float, q: float) -> Tuple[np.ndarray, np.ndarray]:
    f = np.eye(6, dtype=float)
    f[:3, 3:] = np.eye(3) * dt
    q1 = q * np.asarray([[dt ** 4 / 4.0, dt ** 3 / 2.0], [dt ** 3 / 2.0, dt ** 2]], dtype=float)
    process = np.zeros((6, 6), dtype=float)
    for axis in range(3):
        process[axis, axis] = q1[0, 0]
        process[axis, axis + 3] = q1[0, 1]
        process[axis + 3, axis] = q1[1, 0]
        process[axis + 3, axis + 3] = q1[1, 1]
    return f, process


def project_matrix(angles: Tuple[float, float]) -> np.ndarray:
    unit = direction(*angles)
    return np.eye(3, dtype=float) - np.outer(unit, unit)


def weighted_rays(
    observations: Sequence[Dict[str, Any]],
    nodes: Mapping[int, np.ndarray],
    prior_position: Optional[np.ndarray],
    prior_sigma_m: Optional[float],
    use_quality: bool,
    huber: bool,
    model: Mapping[str, Any],
) -> Dict[str, Any]:
    """Solve weighted ray intersection with an optional CV prior and Huber IRLS."""
    if len(observations) < 3:
        return {"position": None, "covariance": np.eye(3) * 1.0e8, "inliers": [], "condition": float("inf"), "rms": float("inf"), "objective": float("inf"), "iterations": 0}
    working = list(observations)
    position: Optional[np.ndarray] = prior_position.copy() if prior_position is not None else None
    inliers = list(range(len(working)))
    condition = float("inf")
    matrix = np.eye(3, dtype=float) * 1.0e-12
    objective = float("inf")
    iterations = 0
    for iteration in range(6 if huber else 1):
        iterations = iteration + 1
        matrix = np.zeros((3, 3), dtype=float)
        rhs = np.zeros(3, dtype=float)
        residuals: List[float] = []
        for item in working:
            node = int(item["node"])
            sigma_deg = sigma_for(item["candidate"], item["alternative"], node, int(item["target"]), model, use_quality)
            sigma_rad = max(math.radians(sigma_deg), math.radians(MIN_SIGMA_DEG))
            anchor = np.asarray(nodes[node], dtype=float)
            if position is None:
                range_m = 1500.0
            else:
                range_m = max(float(np.linalg.norm(position - anchor)), 100.0)
            weight = 1.0 / max((range_m * sigma_rad) ** 2, 1.0e-12)
            if position is not None:
                predicted = truth_angles(position, anchor)
                residual = tangent_norm_deg((item["candidate"].azimuth_deg, item["candidate"].zenith_deg), predicted)
                residuals.append(residual)
                if huber:
                    weight *= min(1.0, HUBER_DEG / max(residual, 1.0e-9))
            projection = project_matrix((item["candidate"].azimuth_deg, item["candidate"].zenith_deg))
            matrix += weight * projection
            rhs += weight * projection @ anchor
        if prior_position is not None and prior_sigma_m is not None:
            prior_weight = 1.0 / max(prior_sigma_m ** 2, 1.0)
            matrix += prior_weight * np.eye(3)
            rhs += prior_weight * prior_position
        matrix = 0.5 * (matrix + matrix.T)
        try:
            condition = float(np.linalg.cond(matrix))
            candidate_position = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            break
        if position is not None and float(np.linalg.norm(candidate_position - position)) < 0.02:
            position = candidate_position
            break
        position = candidate_position
    if position is None:
        return {"position": None, "covariance": np.eye(3) * 1.0e8, "inliers": [], "condition": condition, "rms": float("inf"), "objective": float("inf"), "iterations": iterations}
    final_residuals: List[float] = []
    standardized: List[float] = []
    for item in working:
        node = int(item["node"])
        predicted = truth_angles(position, nodes[node])
        residual = tangent_norm_deg((item["candidate"].azimuth_deg, item["candidate"].zenith_deg), predicted)
        sigma = sigma_for(item["candidate"], item["alternative"], node, int(item["target"]), model, use_quality)
        final_residuals.append(residual)
        standardized.append(residual / max(sigma, MIN_SIGMA_DEG))
    inliers = [index for index, item in enumerate(working) if standardized[index] <= STANDARDIZED_INLIER_LIMIT and final_residuals[index] <= MAX_ABS_RESIDUAL_DEG and not item["candidate"].boundary]
    if len(inliers) < 3:
        inliers = [index for index, value in enumerate(standardized) if value <= STANDARDIZED_INLIER_LIMIT and final_residuals[index] <= MAX_ABS_RESIDUAL_DEG]
    if inliers and len(inliers) < len(working):
        # Re-solve once using accepted rays, keeping the same prior.
        reduced = weighted_rays([working[index] for index in inliers], nodes, prior_position, prior_sigma_m, use_quality, False, model)
        if reduced["position"] is not None:
            position = reduced["position"]
            matrix = np.linalg.pinv(reduced["covariance"])
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    eigenvalues = np.maximum(eigenvalues, 1.0e-12)
    covariance = eigenvectors @ np.diag(1.0 / eigenvalues) @ eigenvectors.T
    covariance = 0.5 * (covariance + covariance.T)
    residual_values = [final_residuals[index] for index in inliers] if inliers else final_residuals
    objective = float(sum(min((value / MAX_ABS_RESIDUAL_DEG) ** 2, 16.0) for value in residual_values))
    rms = math.sqrt(sum(value ** 2 for value in residual_values) / len(residual_values)) if residual_values else float("inf")
    return {"position": position, "covariance": covariance, "inliers": inliers, "condition": condition, "rms": rms, "objective": objective, "iterations": iterations}


def candidate_assignment(
    frame_rows: Mapping[int, FrontendRow],
    calibrations: Mapping[int, Mapping[str, float]],
    nodes: Mapping[int, np.ndarray],
    target_positions: Sequence[Optional[np.ndarray]],
    model: Mapping[str, Any],
    variant: str,
    previous: Optional[Mapping[int, Tuple[int, int]]],
    frozen_priors: Optional[Mapping[int, Sequence[Any]]],
    frame_index: int,
) -> Tuple[Dict[int, Tuple[int, int]], Dict[int, Dict[str, Any]]]:
    """Choose identity/swap per node and return posterior margins."""
    use_quality = variant in ("strength", "combined", "a2")
    assignments: Dict[int, Tuple[int, int]] = {}
    diagnostics: Dict[int, Dict[str, Any]] = {}
    for node in sorted(frame_rows):
        candidates = row_candidates(frame_rows[node], calibrations[node])
        costs: List[float] = []
        for permutation in ((0, 1), (1, 0)):
            cost = 0.0
            for target, candidate_index in zip(TARGETS, permutation):
                candidate = candidates[candidate_index]
                alternative = candidates[1 - candidate_index]
                if target_positions[target - 1] is None:
                    prediction_cost = 0.0
                else:
                    predicted_angles = truth_angles(target_positions[target - 1], nodes[node])
                    residual = tangent_norm_deg((candidate.azimuth_deg, candidate.zenith_deg), predicted_angles)
                    sigma = sigma_for(candidate, alternative, node, target, model, use_quality)
                    prediction_cost = min((residual / max(sigma, MIN_SIGMA_DEG)) ** 2, 100.0)
                cost += prediction_cost
            if previous is not None and node in previous:
                old = previous[node]
                cost += 0.25 * sum(int(candidate_index != old[target - 1]) for target, candidate_index in zip(TARGETS, permutation))
            if variant == "a2" and frozen_priors is not None and frame_index < len(frozen_priors[node]):
                # Calibration labels are only a weak anchor; both permutations
                # remain in the posterior and this term is disabled in evaluation.
                prior = frozen_priors[node][frame_index]
                cost += 0.05 * sum(int(candidate_index != expected) for candidate_index, expected in zip(permutation, (prior.target1_peak_index, prior.target2_peak_index)))
            costs.append(cost)
        shifted = np.asarray(costs, dtype=float) - min(costs)
        probabilities = np.exp(-shifted)
        probabilities /= max(float(np.sum(probabilities)), 1.0e-12)
        chosen = 0 if costs[0] <= costs[1] else 1
        permutation = ((0, 1), (1, 0))[chosen]
        assignments[node] = permutation
        diagnostics[node] = {
            "identity_cost": costs[0],
            "swap_cost": costs[1],
            "identity_probability": float(probabilities[0]),
            "swap_probability": float(probabilities[1]),
            "chosen_probability": float(probabilities[chosen]),
            "margin": float(abs(probabilities[0] - probabilities[1])),
            "chosen": "identity" if chosen == 0 else "swap",
        }
    return assignments, diagnostics


def aggregate_frame_rows(bundle: Mapping[int, Sequence[FrontendRow]], indices: Sequence[int]) -> Dict[int, FrontendRow]:
    """Use the middle raw row per node as the 1-s representative.

    Assignment still sees all raw rows in a bin; this helper is only used for
    the initial frame metadata and keeps the output one row per second.
    """
    middle = indices[len(indices) // 2]
    return {node: rows[middle] for node, rows in bundle.items()}


def run_variant(
    variant: str,
    bundle: Mapping[int, Sequence[FrontendRow]],
    calibrations: Mapping[int, Mapping[str, float]],
    nodes: Mapping[int, np.ndarray],
    model: Mapping[str, Any],
    bins: Mapping[int, Sequence[int]],
    calibration_start_s: int,
    calibration_end_s: int,
    end_s: int,
    frozen_priors: Mapping[int, Sequence[Any]],
) -> Tuple[Dict[int, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    rows_by_target: Dict[int, List[Dict[str, Any]]] = {1: [], 2: []}
    assignment_rows: List[Dict[str, Any]] = []
    states: Dict[int, Optional[np.ndarray]] = {1: None, 2: None}
    previous_assignments: Optional[Dict[int, Tuple[int, int]]] = None
    previous_time: Optional[float] = None
    previous_covariance: Dict[int, np.ndarray] = {1: np.eye(3) * 1.0e6, 2: np.eye(3) * 1.0e6}
    use_motion = variant in ("combined", "a2")
    use_quality = variant in ("strength", "combined", "a2")
    joint_iterations = 4 if variant == "a2" else 1
    for second in sorted(bins):
        if second < calibration_start_s or second >= end_s:
            continue
        indices = list(bins[second])
        if not indices:
            continue
        time_s = float(second)
        dt = 1.0 if previous_time is None else max(time_s - previous_time, 1.0e-6)
        predicted_states: List[Optional[np.ndarray]] = []
        for target in TARGETS:
            if states[target] is None:
                predicted_states.append(None)
            else:
                predicted_states.append(states[target][:3] + states[target][3:] * dt)
        calibration_frame = calibration_start_s <= second < calibration_end_s
        local_assignment: Optional[Dict[int, Tuple[int, int]]] = None
        local_diag: Dict[int, Dict[str, Any]] = {}
        estimate_positions = predicted_states
        target_results: Dict[int, Dict[str, Any]] = {}
        for iteration in range(joint_iterations):
            # One assignment per raw subframe, then a pooled ray solve.
            all_observations: Dict[int, List[Dict[str, Any]]] = {1: [], 2: []}
            margins: List[float] = []
            selected_probabilities: List[float] = []
            for raw_index in indices:
                frame_rows = {node: bundle[node][raw_index] for node in NODES}
                if calibration_frame:
                    # GPS is allowed only in this frozen calibration interval:
                    # recover the already-fitted rank assignment as the initial
                    # calibration estimate, while retaining the same diagnostics.
                    assignments = {
                        node: (frozen_priors[node][raw_index].target1_peak_index, frozen_priors[node][raw_index].target2_peak_index)
                        for node in NODES
                    }
                    diagnostics = {node: {"margin": 1.0, "chosen_probability": 1.0, "chosen": "frozen_calibration"} for node in NODES}
                else:
                    assignments, diagnostics = candidate_assignment(
                        frame_rows, calibrations, nodes, estimate_positions, model, variant,
                        previous_assignments, None, raw_index,
                    )
                if local_assignment is None:
                    local_assignment = assignments
                    local_diag = diagnostics
                for node in NODES:
                    candidates = row_candidates(frame_rows[node], calibrations[node])
                    permutation = assignments[node]
                    for target, candidate_index in zip(TARGETS, permutation):
                        all_observations[target].append({
                            "node": node,
                            "target": target,
                            "candidate": candidates[candidate_index],
                            "alternative": candidates[1 - candidate_index],
                        })
                    margins.append(float(diagnostics[node].get("margin", 0.0)))
                    selected_probabilities.append(float(diagnostics[node].get("chosen_probability", 0.0)))
            for target in TARGETS:
                prior_position = estimate_positions[target - 1] if use_motion else None
                prior_sigma = 160.0 if use_motion and prior_position is not None else None
                result = weighted_rays(
                    all_observations[target], nodes, prior_position, prior_sigma,
                    use_quality, variant == "a2", model,
                )
                target_results[target] = result
                if result["position"] is not None:
                    estimate_positions[target - 1] = result["position"]
        if local_assignment is None:
            continue
        for target in TARGETS:
            result = target_results[target]
            old_state = states[target]
            position = result["position"]
            if position is not None:
                if old_state is None:
                    velocity = np.zeros(3, dtype=float)
                else:
                    velocity = (position - old_state[:3]) / dt
                    speed = float(np.linalg.norm(velocity))
                    if speed > 150.0:
                        velocity *= 150.0 / speed
                    velocity = 0.8 * old_state[3:] + 0.2 * velocity
                states[target] = np.concatenate([position, velocity])
                previous_covariance[target] = np.asarray(result["covariance"], dtype=float)
            state = states[target]
            position = None if state is None else state[:3]
            velocity = np.zeros(3, dtype=float) if state is None else state[3:]
            covariance = previous_covariance[target]
            full_covariance = np.zeros((6, 6), dtype=float)
            full_covariance[:3, :3] = covariance
            full_covariance[3:, 3:] = np.eye(3) * max(Q_ACCEL * dt, 4.0)
            eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (full_covariance + full_covariance.T))
            full_covariance = eigenvectors @ np.diag(np.maximum(eigenvalues, 1.0e-9)) @ eigenvectors.T
            inlier_node_ids = sorted({int(all_observations[target][i]["node"]) for i in result["inliers"]})
            target_row: Dict[str, Any] = {
                "frame_index": second - calibration_start_s,
                "time_s": time_s,
                "calibration_frame": calibration_frame,
                "px": "" if position is None else float(position[0]),
                "py": "" if position is None else float(position[1]),
                "pz": "" if position is None else float(position[2]),
                "vx": float(velocity[0]), "vy": float(velocity[1]), "vz": float(velocity[2]),
                "valid": bool(position is not None),
                "converged": bool(position is not None and math.isfinite(result["objective"])),
                "iterations": int(result["iterations"]),
                "objective": float(result["objective"]),
                "available_nodes": len(NODES),
                "inlier_nodes": len(inlier_node_ids),
                "inlier_node_ids": ";".join(str(node) for node in inlier_node_ids),
                "condition_number": float(result["condition"]),
                "reprojection_rms_deg": float(result["rms"]),
                "state_step_m": "" if old_state is None or position is None else float(np.linalg.norm(position - old_state[:3])),
                "identity_margin_mean": float(np.mean(margins)) if margins else 0.0,
                "identity_margin_min": float(np.min(margins)) if margins else 0.0,
                "frozen_assignment_probability_mean": float(np.mean(selected_probabilities)) if selected_probabilities else 0.0,
            }
            for row_index in range(6):
                for column_index in range(6):
                    target_row["cov_%d%d" % (row_index, column_index)] = float(full_covariance[row_index, column_index])
            rows_by_target[target].append(target_row)
        for node in NODES:
            diag = local_diag.get(node, {})
            assignment_rows.append({
                "time_s": time_s,
                "frame_index": second - calibration_start_s,
                "node": node,
                "calibration_frame": calibration_frame,
                "chosen": diag.get("chosen", ""),
                "identity_cost": diag.get("identity_cost", ""),
                "swap_cost": diag.get("swap_cost", ""),
                "identity_probability": diag.get("identity_probability", ""),
                "swap_probability": diag.get("swap_probability", ""),
                "chosen_probability": diag.get("chosen_probability", ""),
                "assignment_margin": diag.get("margin", ""),
            })
        previous_assignments = local_assignment
        previous_time = time_s
    return rows_by_target, assignment_rows


def add_metrics(
    rows: Sequence[Mapping[str, Any]],
    track: Mapping[int, Tuple[float, float, float]],
    centre: np.ndarray,
    calibration_start_s: int,
    calibration_end_s: int,
    evaluation_end_s: int,
    delay_s: int,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for split, lower, upper in (("calibration", calibration_start_s, calibration_end_s), ("evaluation", calibration_end_s, evaluation_end_s)):
        errors: List[float] = []
        jumps: List[float] = []
        previous: Optional[np.ndarray] = None
        total = 0
        valid = 0
        for row in rows:
            second = int(float(row["time_s"]))
            if not (lower <= second < upper):
                continue
            total += 1
            if not bool(row.get("valid")):
                previous = None
                continue
            truth = nearest_truth(track, base.seconds_to_hhmmss(second - delay_s))
            if truth is None:
                continue
            position = np.asarray([float(row["px"]), float(row["py"]), float(row["pz"])], dtype=float)
            truth_local = np.asarray(truth, dtype=float) - centre
            error = float(np.linalg.norm(position - truth_local))
            errors.append(error)
            valid += 1
            if previous is not None:
                jumps.append(float(np.linalg.norm(position - previous)))
            previous = position
        result[split] = {
            "frames": total,
            "valid_frames": valid,
            "valid_fraction": valid / total if total else 0.0,
            "rmse_position_m": math.sqrt(sum(value * value for value in errors) / len(errors)) if errors else None,
            "median_position_error_m": float(np.median(errors)) if errors else None,
            "p90_position_error_m": percentile(errors, 90.0),
            "maximum_step_m": max(jumps) if jumps else None,
        }
    return result


def covariance_psd(rows: Sequence[Mapping[str, Any]]) -> Tuple[bool, float]:
    minimum = float("inf")
    for row in rows:
        matrix = np.asarray([[finite_float(row.get("cov_%d%d" % (i, j))) for j in range(6)] for i in range(6)], dtype=float)
        if not np.all(np.isfinite(matrix)):
            return False, float("nan")
        values = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
        minimum = min(minimum, float(np.min(values)))
        if np.min(values) <= 0.0:
            return False, minimum
    return bool(rows), minimum


def serializable_quality_model(model: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert tuple-keyed calibration dictionaries to manifest-safe JSON."""
    output: Dict[str, Any] = {}
    for name in ("node_target_sigma_deg", "quality_median", "quality_scale"):
        values = model.get(name, {})
        output[name] = {
            "node%d_target%d" % (key[0], key[1]): float(value)
            for key, value in values.items()
        }
    for name in ("min_sigma_deg", "max_sigma_deg", "boundary_inflation", "quality_feature", "gps_used_only_in_calibration"):
        if name in model:
            output[name] = model[name]
    return output


def window_audit(rows_by_target: Mapping[int, Sequence[Mapping[str, Any]]], calibration_end_s: int, end_s: int) -> Dict[str, Any]:
    by_target = {target: {int(float(row["time_s"])): row for row in rows_by_target[target]} for target in TARGETS}
    eligible: List[int] = []
    for second in range(calibration_end_s, end_s):
        if any(second not in by_target[target] for target in TARGETS):
            continue
        current = [by_target[target][second] for target in TARGETS]
        if not all(bool(row.get("valid")) for row in current):
            continue
        if min(int(row.get("inlier_nodes", 0)) for row in current) < 4:
            continue
        if max(finite_float(row.get("condition_number"), float("inf")) for row in current) > 50.0:
            continue
        if min(finite_float(row.get("identity_margin_min"), 0.0) for row in current) < 0.20:
            continue
        if max(finite_float(row.get("state_step_m"), 0.0) for row in current) > 100.0:
            continue
        eligible.append(second)
    runs: List[Tuple[int, int]] = []
    if eligible:
        start = previous = eligible[0]
        for second in eligible[1:]:
            if second != previous + 1:
                runs.append((start, previous))
                start = second
            previous = second
        runs.append((start, previous))
    longest = max(runs, key=lambda item: item[1] - item[0] + 1) if runs else None
    return {
        "gps_free": True,
        "criteria": {"both_targets_valid": True, "min_inlier_nodes": 4, "max_condition_number": 50.0, "min_identity_margin": 0.20, "max_state_step_m": 100.0},
        "eligible_seconds": len(eligible),
        "contiguous_runs": [{"start_time_s": start, "end_time_s": end, "length_s": end - start + 1} for start, end in runs],
        "longest_run": None if longest is None else {"start_time_s": longest[0], "end_time_s": longest[1], "length_s": longest[1] - longest[0] + 1},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibration-frozen A2 direct DOA-map tracker")
    parser.add_argument("--remote-root", type=Path, default=Path("<PRIVATE_DATA_ROOT>/2017保定实验/2017保定实验"))
    parser.add_argument("--frontend", type=Path, required=True)
    parser.add_argument("--association", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-start-s", type=int, default=CALIBRATION_START_S)
    parser.add_argument("--calibration-end-s", type=int, default=CALIBRATION_END_S)
    parser.add_argument("--evaluation-end-s", type=int, default=EVALUATION_END_S)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    gate_path = args.association / "shuangyuan4_global_association_gate.json"
    calibrations, delay_s = load_frozen_orientation_calibrations(gate_path, NODES)
    bundle = load_frontend_bundle(args.frontend, NODES)
    bundle_manifest = validate_frontend_bundle(args.frontend, NODES)
    priors = load_frozen_identity_priors(args.association, bundle, calibrations)
    archive = args.remote_root / "20171107保定实验"
    gps_root = archive / "GPS_data"
    nodes_abs = base.parse_nod(gps_root / "20171107baoding.nod")
    missing_nodes = [node for node in NODES if node not in nodes_abs]
    if missing_nodes:
        raise RuntimeError("missing node coordinates: %s" % missing_nodes)
    centre = np.mean(np.asarray([nodes_abs[node] for node in NODES], dtype=float), axis=0)
    nodes = {node: np.asarray(nodes_abs[node], dtype=float) - centre for node in NODES}
    tracks = load_truth_tracks(gps_root)
    model, reliability_rows = fit_quality_model(
        bundle, calibrations, priors, nodes_abs, tracks, centre,
        args.calibration_start_s, args.calibration_end_s, delay_s,
    )
    bins = frame_bins(bundle, args.calibration_start_s, args.evaluation_end_s)
    variant_metrics: Dict[str, Any] = {}
    variant_outputs: Dict[str, Any] = {}
    for variant in ("equal", "strength", "combined", "a2"):
        rows_by_target, assignment_rows = run_variant(
            variant, bundle, calibrations, nodes, model, bins,
            args.calibration_start_s, args.calibration_end_s, args.evaluation_end_s, priors,
        )
        variant_outputs[variant] = {"rows": rows_by_target, "assignments": assignment_rows}
        variant_metrics[variant] = {
            str(target): add_metrics(rows_by_target[target], tracks[target], centre, args.calibration_start_s, args.calibration_end_s, args.evaluation_end_s, delay_s)
            for target in TARGETS
        }
        root = args.output / variant
        for target in TARGETS:
            write_csv(root / ("target%d_state.csv" % target), rows_by_target[target])
        write_csv(root / "assignments.csv", assignment_rows)
    main_rows = variant_outputs["a2"]["rows"]
    for target in TARGETS:
        write_csv(args.output / ("target%d_state.csv" % target), main_rows[target])
    write_csv(args.output / "a2_assignments.csv", variant_outputs["a2"]["assignments"])
    write_csv(args.output / "node_target_reliability.csv", reliability_rows)
    summary_rows: List[Dict[str, Any]] = []
    for variant, target_data in variant_metrics.items():
        for target in TARGETS:
            for split in ("calibration", "evaluation"):
                row = {"variant": variant, "target": target, "split": split}
                row.update(target_data[str(target)][split])
                summary_rows.append(row)
    write_csv(args.output / "variant_metrics.csv", summary_rows)
    psd = {str(target): covariance_psd(main_rows[target]) for target in TARGETS}
    audit = window_audit(main_rows, args.calibration_end_s, args.evaluation_end_s)
    write_json(args.output / "a2_window_audit.json", audit)
    state_manifest = {
        "task": "Baoding shuangyuan_4 A2 direct DOA-map joint assignment with CV motion prior",
        "claim_boundary": "bounded frontend prototype; not a Zhang DBN-LA-NM reproduction and not a physical SNR estimate",
        "nodes": list(NODES),
        "state_definition": "[E, N, U, vE, vN, vU] in node-centred local ENU metres",
        "source_update_interval_s": float(bundle_manifest["frame_dt_s"]),
        "output_update_interval_s": 1.0,
        "calibration_interval_s": [args.calibration_start_s, args.calibration_end_s],
        "evaluation_interval_s": [args.calibration_end_s, args.evaluation_end_s],
        "delay_s": delay_s,
        "gps_role": "GPS used only to fit frozen orientation/identity/quality scales on calibration seconds and to score outputs offline; no GPS is read by held-out state updates",
        "assignment": "per-node identity/swap posterior over both rank-paired candidates; A2 repeats assignment and weighted ray update",
        "variants": ["equal", "strength", "combined", "a2"],
        "motion_prior": {"model": "constant velocity", "q_accel_m2_s3": Q_ACCEL, "prior_sigma_m": 160.0, "huber_deg": HUBER_DEG},
        "quality_model": serializable_quality_model(model),
        "metrics": variant_metrics,
        "covariance": {"state_covariance": "6x6 block PSD; position ray normal inverse plus velocity process floor", "psd": {target: {"finite_psd": bool(value[0]), "minimum_eigenvalue": value[1]} for target, value in psd.items()}},
        "window_audit": audit,
        "input_manifest": bundle_manifest,
        "sources": {
            "frontend": str(args.frontend),
            "frontend_csv_sha256": {path.name: sha256(path) for path in sorted(args.frontend.glob("dual_doa_node_*.csv"))},
            "association": str(args.association),
            "association_gate_sha256": sha256(gate_path),
            "node_file": str(gps_root / "20171107baoding.nod"),
            "node_file_sha256": sha256(gps_root / "20171107baoding.nod"),
        },
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha256(Path(__file__).resolve()),
    }
    write_json(args.output / "a2_manifest.json", state_manifest)
    write_json(args.output / "a2_metrics.json", {"variants": variant_metrics, "psd": state_manifest["covariance"]["psd"], "window_audit": audit})
    print(json.dumps({"output": str(args.output), "evaluation": {variant: {str(target): variant_metrics[variant][str(target)]["evaluation"] for target in TARGETS} for variant in variant_metrics}, "window_audit": audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
