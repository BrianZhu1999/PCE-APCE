#!/usr/bin/env python3
"""Confidence-aware two-source localization for the Baoding field data.

The historical frontend estimates two unlabelled MUSIC peaks at every node.
This module preserves each peak's strength, learns a frozen angular-error
model from the declared calibration interval, and compares five localization
variants on the independent interval. GPS is never used in a runtime update.
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

import numpy as np

import shuangyuan_dual_association as base


VARIANTS = (
    "equal_weight",
    "node_precision",
    "strength_precision",
    "combined_precision",
    "combined_motion",
)
TARGETS = (1, 2)
MOTION_SIGMA_CANDIDATES_M = (20.0, 40.0, 80.0, 160.0)
MAX_ABSOLUTE_RESIDUAL_DEG = 50.0
STANDARDIZED_INLIER_LIMIT = 3.5
MIN_SIGMA_DEG = 2.0
MAX_SIGMA_DEG = 45.0
QUALITY_BINS = 5
QUALITY_SHRINKAGE = 12.0
VELOCITY_SMOOTHING = 0.20
MAX_SPEED_MPS = 120.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
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


def circular_difference_deg(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def angular_components(
    observation: tuple[float, float],
    expected: tuple[float, float],
) -> tuple[float, float]:
    return circular_difference_deg(observation[0], expected[0]), observation[1] - expected[1]


def angular_residual_deg(
    position: np.ndarray,
    node_position: tuple[float, float, float],
    observation: tuple[float, float],
) -> float:
    expected = base.truth_angles(tuple(float(value) for value in position), node_position)
    daz, dzen = angular_components(observation, expected)
    return math.hypot(daz, dzen)


def robust_scale(values: list[float], floor: float = 0.25) -> float:
    if not values:
        return floor
    array = np.asarray(values, dtype=float)
    median = float(np.median(array))
    mad = float(1.4826 * np.median(np.abs(array - median)))
    return max(mad, floor)


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else None


@dataclass(frozen=True)
class PeakObservation:
    node: int
    target: int
    frame_index: int
    time_s: float
    time_second: int
    azimuth_deg: float
    zenith_deg: float
    azimuth_strength: float
    zenith_strength: float
    log_geometric_strength: float
    log_peak_ratio: float
    calibration_frame: bool

    @property
    def angles(self) -> tuple[float, float]:
        return self.azimuth_deg, self.zenith_deg


def peak_index_for_associated(
    transformed: list[tuple[float, float]],
    associated: tuple[float, float],
) -> int:
    distances = [
        math.hypot(
            base.circular_error(candidate[0], associated[0]),
            candidate[1] - associated[1],
        )
        for candidate in transformed
    ]
    return int(np.argmin(np.asarray(distances, dtype=float)))


def load_observations(
    frontend: Path,
    association: Path,
    calibrations: dict[int, dict],
) -> dict[int, dict[int, list[PeakObservation]]]:
    output = {target: {} for target in TARGETS}
    for node in base.NODES:
        raw_rows = base.load_rows(frontend / f"dual_doa_node_{node}_125540_125900.csv")
        associated_rows = read_csv(association / f"associated_global_node_{node}.csv")
        if len(raw_rows) != len(associated_rows):
            raise RuntimeError(f"node {node}: raw/associated row mismatch")
        for raw, labeled in zip(raw_rows, associated_rows, strict=True):
            transformed = base.apply_transform(raw, calibrations[node])
            chosen: set[int] = set()
            for target in TARGETS:
                associated_angles = (
                    float(labeled[f"target{target}_az_deg"]),
                    float(labeled[f"target{target}_zenith_deg"]),
                )
                index = peak_index_for_associated(transformed, associated_angles)
                if index in chosen:
                    index = 1 - index
                chosen.add(index)
                other = 1 - index
                azimuth_strength = max(float(raw[f"azimuth_strength_{index + 1}"]), 1e-12)
                zenith_strength = max(float(raw[f"zenith_strength_{index + 1}"]), 1e-12)
                other_azimuth = max(float(raw[f"azimuth_strength_{other + 1}"]), 1e-12)
                other_zenith = max(float(raw[f"zenith_strength_{other + 1}"]), 1e-12)
                observation = PeakObservation(
                    node=node,
                    target=target,
                    frame_index=int(labeled["frame_index"]),
                    time_s=float(labeled["time_s"]),
                    time_second=int(float(labeled["time_second"])),
                    azimuth_deg=associated_angles[0],
                    zenith_deg=associated_angles[1],
                    azimuth_strength=azimuth_strength,
                    zenith_strength=zenith_strength,
                    log_geometric_strength=0.5 * (math.log(azimuth_strength) + math.log(zenith_strength)),
                    log_peak_ratio=0.5 * (
                        math.log(azimuth_strength / other_azimuth)
                        + math.log(zenith_strength / other_zenith)
                    ),
                    calibration_frame=str(labeled["calibration_frame"]).lower() == "true",
                )
                output[target].setdefault(observation.frame_index, []).append(observation)
    return output


def calibration_truth(
    tracks: dict[int, dict[int, tuple[float, float, float]]],
    target: int,
    time_second: int,
    delay_s: int,
) -> tuple[float, float, float] | None:
    return base.nearest_truth(tracks[target], base.shift_hhmmss(time_second, -delay_s))


def quality_normalization(
    observations: dict[int, dict[int, list[PeakObservation]]],
) -> dict[int, dict[str, float]]:
    values = {node: {"strength": [], "ratio": []} for node in base.NODES}
    for frames in observations.values():
        for frame in frames.values():
            for item in frame:
                if item.calibration_frame:
                    values[item.node]["strength"].append(item.log_geometric_strength)
                    values[item.node]["ratio"].append(item.log_peak_ratio)
    return {
        node: {
            "strength_median": float(np.median(values[node]["strength"])),
            "strength_scale": robust_scale(values[node]["strength"]),
            "ratio_median": float(np.median(values[node]["ratio"])),
            "ratio_scale": robust_scale(values[node]["ratio"]),
        }
        for node in base.NODES
    }


def quality_score(item: PeakObservation, normalization: dict[int, dict[str, float]]) -> float:
    value = normalization[item.node]
    z_strength = (item.log_geometric_strength - value["strength_median"]) / value["strength_scale"]
    z_ratio = (item.log_peak_ratio - value["ratio_median"]) / value["ratio_scale"]
    return float(np.clip(0.5 * z_strength + 0.5 * z_ratio, -6.0, 6.0))


def clipped_rms(residuals: list[float], limit: float = MAX_SIGMA_DEG) -> float:
    if not residuals:
        return 10.0
    array = np.minimum(np.asarray(residuals, dtype=float), limit)
    return float(np.sqrt(np.mean(array**2)))


def learn_reliability(
    observations: dict[int, dict[int, list[PeakObservation]]],
    nodes: dict[int, tuple[float, float, float]],
    tracks: dict[int, dict[int, tuple[float, float, float]]],
    delay_s: int,
    normalization: dict[int, dict[str, float]],
) -> tuple[dict, list[dict[str, object]]]:
    samples: list[dict[str, float | int]] = []
    for target, frames in observations.items():
        for frame in frames.values():
            for item in frame:
                if not item.calibration_frame:
                    continue
                truth = calibration_truth(tracks, target, item.time_second, delay_s)
                if truth is None:
                    continue
                expected = base.truth_angles(truth, nodes[item.node])
                daz, dzen = angular_components(item.angles, expected)
                samples.append({
                    "node": item.node,
                    "target": target,
                    "quality": quality_score(item, normalization),
                    "azimuth_residual_deg": daz,
                    "zenith_residual_deg": dzen,
                    "angular_residual_deg": math.hypot(daz, dzen),
                })
    global_residuals = [float(row["angular_residual_deg"]) for row in samples]
    global_sigma = clipped_rms(global_residuals)
    all_quality = np.asarray([float(row["quality"]) for row in samples], dtype=float)
    global_edges = np.unique(np.quantile(all_quality, np.linspace(0.0, 1.0, QUALITY_BINS + 1)))
    if len(global_edges) < 3:
        global_edges = np.asarray([-6.0, 0.0, 6.0])

    node_sigma: dict[int, float] = {}
    node_target_sigma: dict[tuple[int, int], float] = {}
    for node in base.NODES:
        residuals = [
            float(row["angular_residual_deg"])
            for row in samples if int(row["node"]) == node
        ]
        node_sigma[node] = float(np.clip(clipped_rms(residuals), MIN_SIGMA_DEG, MAX_SIGMA_DEG))
        for target in TARGETS:
            target_residuals = [
                float(row["angular_residual_deg"])
                for row in samples
                if int(row["node"]) == node and int(row["target"]) == target
            ]
            raw = clipped_rms(target_residuals)
            shrunk = math.sqrt(
                (len(target_residuals) * raw**2 + QUALITY_SHRINKAGE * node_sigma[node] ** 2)
                / max(len(target_residuals) + QUALITY_SHRINKAGE, 1.0)
            )
            node_target_sigma[(node, target)] = float(np.clip(shrunk, MIN_SIGMA_DEG, MAX_SIGMA_DEG))

    global_bins: list[dict[str, float]] = []
    combined_bins: dict[tuple[int, int], list[dict[str, float]]] = {}
    for lower, upper in zip(global_edges[:-1], global_edges[1:], strict=True):
        selected = [
            float(row["angular_residual_deg"])
            for row in samples
            if lower <= float(row["quality"]) <= upper
        ]
        raw = clipped_rms(selected)
        shrunk = math.sqrt(
            (len(selected) * raw**2 + QUALITY_SHRINKAGE * global_sigma**2)
            / max(len(selected) + QUALITY_SHRINKAGE, 1.0)
        )
        global_bins.append({
            "lower": float(lower), "upper": float(upper), "count": float(len(selected)),
            "sigma_deg": float(np.clip(shrunk, MIN_SIGMA_DEG, MAX_SIGMA_DEG)),
        })
    for node in base.NODES:
        for target in TARGETS:
            local = [row for row in samples if int(row["node"]) == node and int(row["target"]) == target]
            bins = []
            for global_bin in global_bins:
                selected = [
                    float(row["angular_residual_deg"])
                    for row in local
                    if global_bin["lower"] <= float(row["quality"]) <= global_bin["upper"]
                ]
                raw = clipped_rms(selected) if selected else node_target_sigma[(node, target)]
                prior = node_target_sigma[(node, target)]
                shrunk = math.sqrt(
                    (len(selected) * raw**2 + QUALITY_SHRINKAGE * prior**2)
                    / max(len(selected) + QUALITY_SHRINKAGE, 1.0)
                )
                bins.append({
                    "lower": global_bin["lower"], "upper": global_bin["upper"],
                    "count": float(len(selected)),
                    "sigma_deg": float(np.clip(shrunk, MIN_SIGMA_DEG, MAX_SIGMA_DEG)),
                })
            combined_bins[(node, target)] = bins

    model = {
        "global_sigma_deg": float(np.clip(global_sigma, MIN_SIGMA_DEG, MAX_SIGMA_DEG)),
        "node_sigma_deg": node_sigma,
        "node_target_sigma_deg": node_target_sigma,
        "global_quality_bins": global_bins,
        "combined_quality_bins": combined_bins,
    }
    reliability_rows = []
    for node in base.NODES:
        for target in TARGETS:
            subset = [row for row in samples if int(row["node"]) == node and int(row["target"]) == target]
            reliability_rows.append({
                "node_id": node,
                "target": target,
                "calibration_samples": len(subset),
                "node_sigma_deg": node_sigma[node],
                "node_target_sigma_deg": node_target_sigma[(node, target)],
                "median_angular_error_deg": statistics.median(float(row["angular_residual_deg"]) for row in subset),
                "p90_angular_error_deg": percentile([float(row["angular_residual_deg"]) for row in subset], 90.0),
                "median_log_geometric_strength": statistics.median(
                    item.log_geometric_strength
                    for frame in observations[target].values()
                    for item in frame
                    if item.node == node and item.calibration_frame
                ),
            })
    return model, reliability_rows


def sigma_for(
    item: PeakObservation,
    variant: str,
    model: dict,
    normalization: dict[int, dict[str, float]],
) -> float:
    if variant == "equal_weight":
        return 10.0
    if variant == "node_precision":
        return float(model["node_target_sigma_deg"][(item.node, item.target)])
    score = quality_score(item, normalization)
    bins = (
        model["global_quality_bins"]
        if variant == "strength_precision"
        else model["combined_quality_bins"][(item.node, item.target)]
    )
    selected = min(
        bins,
        key=lambda value: 0.0
        if value["lower"] <= score <= value["upper"]
        else min(abs(score - value["lower"]), abs(score - value["upper"])),
    )
    return float(selected["sigma_deg"])


def ray_projection(observation: tuple[float, float]) -> np.ndarray:
    direction = np.asarray(base.direction(*observation), dtype=float)
    return np.eye(3, dtype=float) - np.outer(direction, direction)


def solve_weighted_rays(
    frame: list[PeakObservation],
    nodes: dict[int, tuple[float, float, float]],
    sigmas: dict[int, float],
    selected: list[int],
    robust_weights: dict[int, float] | None = None,
    prior_position: np.ndarray | None = None,
    prior_sigma_m: float | None = None,
) -> tuple[np.ndarray | None, float]:
    matrix = np.zeros((3, 3), dtype=float)
    target = np.zeros(3, dtype=float)
    by_node = {item.node: item for item in frame}
    for node in selected:
        item = by_node[node]
        projection = ray_projection(item.angles)
        sigma_rad = max(math.radians(sigmas[node]), math.radians(MIN_SIGMA_DEG))
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
        prior_precision = 1.0 / max(prior_sigma_m, 1.0) ** 2
        matrix += prior_precision * np.eye(3)
        target += prior_precision * prior_position
    condition = float(np.linalg.cond(matrix))
    if not math.isfinite(condition) or condition > 1e10:
        return None, condition
    try:
        return np.linalg.solve(matrix, target), condition
    except np.linalg.LinAlgError:
        return None, condition


def robust_weighted_triangulate(
    frame: list[PeakObservation],
    nodes: dict[int, tuple[float, float, float]],
    sigmas: dict[int, float],
    prior_position: np.ndarray | None = None,
    prior_sigma_m: float | None = None,
) -> tuple[np.ndarray | None, list[int], float, float]:
    available = sorted(item.node for item in frame)
    if len(available) < 3:
        return None, [], float("inf"), float("inf")
    by_node = {item.node: item for item in frame}
    candidates = []
    for subset in itertools.combinations(available, 3):
        position, condition = solve_weighted_rays(
            frame, nodes, sigmas, list(subset), prior_position=prior_position,
            prior_sigma_m=prior_sigma_m,
        )
        if position is None:
            continue
        residuals = {
            node: angular_residual_deg(position, nodes[node], by_node[node].angles)
            for node in available
        }
        standardized = {
            node: residuals[node] / max(sigmas[node], MIN_SIGMA_DEG)
            for node in available
        }
        clipped_loss = float(np.mean([min(value**2, 16.0) for value in standardized.values()]))
        if prior_position is not None and prior_sigma_m is not None:
            clipped_loss += min(float(np.linalg.norm(position - prior_position)) / prior_sigma_m, 4.0) ** 2 / len(available)
        inliers = [
            node for node in available
            if standardized[node] <= STANDARDIZED_INLIER_LIMIT
            and residuals[node] <= MAX_ABSOLUTE_RESIDUAL_DEG
        ]
        if len(inliers) >= 3:
            candidates.append((clipped_loss, -len(inliers), condition, inliers, position))
    if not candidates:
        return None, [], float("inf"), float("inf")
    _, _, _, inliers, position = min(candidates, key=lambda value: value[:3])
    condition = float("inf")
    for _ in range(4):
        residuals = {
            node: angular_residual_deg(position, nodes[node], by_node[node].angles)
            for node in inliers
        }
        robust_weights = {
            node: min(1.0, 1.5 * sigmas[node] / max(residuals[node], 1e-6))
            for node in inliers
        }
        updated, condition = solve_weighted_rays(
            frame, nodes, sigmas, inliers, robust_weights,
            prior_position=prior_position, prior_sigma_m=prior_sigma_m,
        )
        if updated is None:
            break
        position = updated
    residual_values = [angular_residual_deg(position, nodes[node], by_node[node].angles) for node in inliers]
    rms = math.sqrt(sum(value**2 for value in residual_values) / len(residual_values))
    return position, inliers, condition, rms


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
        return self.position + self.velocity * max(time_s - self.time_s, 0.0)

    def update(self, position: np.ndarray, time_s: float) -> None:
        if self.position is not None and self.time_s is not None:
            dt = max(time_s - self.time_s, 1e-6)
            instant = (position - self.position) / dt
            speed = float(np.linalg.norm(instant))
            if speed > MAX_SPEED_MPS:
                instant *= MAX_SPEED_MPS / speed
            if self.velocity is None:
                self.velocity = instant
            else:
                self.velocity = (1.0 - VELOCITY_SMOOTHING) * self.velocity + VELOCITY_SMOOTHING * instant
        self.position = position.copy()
        self.time_s = time_s


def run_variant(
    observations: dict[int, dict[int, list[PeakObservation]]],
    nodes: dict[int, tuple[float, float, float]],
    model: dict,
    normalization: dict[int, dict[str, float]],
    variant: str,
    motion_sigma_m: float | None = None,
) -> dict[int, list[dict[str, object]]]:
    output = {target: [] for target in TARGETS}
    states = {target: MotionState() for target in TARGETS}
    for target in TARGETS:
        for frame_index in sorted(observations[target]):
            frame = observations[target][frame_index]
            time_s = frame[0].time_s
            sigmas = {
                item.node: sigma_for(item, variant, model, normalization)
                for item in frame
            }
            prior = states[target].predict(time_s) if variant == "combined_motion" else None
            position, inliers, condition, rms = robust_weighted_triangulate(
                frame, nodes, sigmas, prior_position=prior,
                prior_sigma_m=motion_sigma_m if prior is not None else None,
            )
            if position is not None:
                states[target].update(position, time_s)
            row: dict[str, object] = {
                "frame_index": frame_index,
                "time_s": time_s,
                "time_second": frame[0].time_second,
                "calibration_frame": frame[0].calibration_frame,
                "available_nodes": len(frame),
                "inlier_nodes": len(inliers),
                "inlier_node_ids": ";".join(str(node) for node in inliers),
                "condition_number": condition,
                "reprojection_rms_deg": rms,
                "valid": position is not None,
                "px": "", "py": "", "pz": "",
            }
            if position is not None:
                row.update({"px": float(position[0]), "py": float(position[1]), "pz": float(position[2])})
            output[target].append(row)
    return output


def add_offline_truth_and_metrics(
    rows: list[dict[str, object]],
    truth_track: dict[int, tuple[float, float, float]],
    delay_s: int,
) -> dict[str, dict[str, float | int | None]]:
    errors = {"calibration": [], "evaluation": []}
    jumps = {"calibration": [], "evaluation": []}
    previous: dict[str, tuple[float, np.ndarray] | None] = {"calibration": None, "evaluation": None}
    valid_counts = {"calibration": 0, "evaluation": 0}
    totals = {"calibration": 0, "evaluation": 0}
    for row in rows:
        split = "calibration" if bool(row["calibration_frame"]) else "evaluation"
        totals[split] += 1
        truth = base.nearest_truth(
            truth_track,
            base.shift_hhmmss(int(row["time_second"]), -delay_s),
        )
        row.update({"truth_x": "", "truth_y": "", "truth_z": "", "position_error_m": ""})
        if truth is None or not bool(row["valid"]):
            continue
        position = np.asarray([float(row[name]) for name in ("px", "py", "pz")], dtype=float)
        truth_array = np.asarray(truth, dtype=float)
        error = float(np.linalg.norm(position - truth_array))
        row.update({
            "truth_x": float(truth_array[0]), "truth_y": float(truth_array[1]),
            "truth_z": float(truth_array[2]), "position_error_m": error,
        })
        valid_counts[split] += 1
        errors[split].append(error)
        if previous[split] is not None:
            previous_time, previous_position = previous[split]
            if 0.0 < float(row["time_s"]) - previous_time <= 0.25:
                jumps[split].append(float(np.linalg.norm(position - previous_position)))
        previous[split] = (float(row["time_s"]), position)
    metrics = {}
    for split in ("calibration", "evaluation"):
        values = errors[split]
        metrics[split] = {
            "frames": totals[split],
            "valid_frames": valid_counts[split],
            "valid_fraction": valid_counts[split] / totals[split] if totals[split] else 0.0,
            "rmse_position_m": math.sqrt(sum(value**2 for value in values) / len(values)) if values else None,
            "median_position_error_m": statistics.median(values) if values else None,
            "p90_position_error_m": percentile(values, 90.0),
            "maximum_step_m": max(jumps[split]) if jumps[split] else None,
        }
    return metrics


def calibration_motion_score(metrics: dict[int, dict[str, dict]]) -> float:
    values = [metrics[target]["calibration"]["rmse_position_m"] for target in TARGETS]
    if any(value is None for value in values):
        return float("inf")
    return max(float(value) for value in values) + 0.25 * statistics.mean(float(value) for value in values)


def serialize_model(model: dict) -> dict:
    return {
        "global_sigma_deg": model["global_sigma_deg"],
        "node_sigma_deg": {str(key): value for key, value in model["node_sigma_deg"].items()},
        "node_target_sigma_deg": {
            f"node{node}_target{target}": value
            for (node, target), value in model["node_target_sigma_deg"].items()
        },
        "global_quality_bins": model["global_quality_bins"],
        "combined_quality_bins": {
            f"node{node}_target{target}": value
            for (node, target), value in model["combined_quality_bins"].items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-root", type=Path, default=Path("<PRIVATE_DATA_ROOT>/2017保定实验/2017保定实验"))
    parser.add_argument("--frontend", type=Path, required=True)
    parser.add_argument("--association", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gate_path = args.association / "shuangyuan4_global_association_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    calibrations = {int(key): value for key, value in gate["node_calibrations"].items()}
    delay_s = int(gate["selected_delay_s"])
    archive = args.remote_root / "20171107保定实验"
    gps_root = archive / "GPS_data"
    nodes = base.parse_nod(gps_root / "20171107baoding.nod")
    tracks = {
        1: base.fuse(base.parse_gps(gps_root / "GPS1_plane1.gps"), base.parse_gps(gps_root / "GPS2_plane1.gps")),
        2: base.fuse(base.parse_gps(gps_root / "GPS3_plane2.gps"), base.parse_gps(gps_root / "GPS4_plane2to3.gps")),
    }
    observations = load_observations(args.frontend, args.association, calibrations)
    normalization = quality_normalization(observations)
    model, reliability_rows = learn_reliability(observations, nodes, tracks, delay_s, normalization)

    motion_trials = []
    for motion_sigma_m in MOTION_SIGMA_CANDIDATES_M:
        trial_rows = run_variant(
            observations, nodes, model, normalization, "combined_motion", motion_sigma_m,
        )
        trial_metrics = {
            target: add_offline_truth_and_metrics(trial_rows[target], tracks[target], delay_s)
            for target in TARGETS
        }
        motion_trials.append({
            "motion_sigma_m": motion_sigma_m,
            "calibration_score": calibration_motion_score(trial_metrics),
            "target1_calibration_rmse_m": trial_metrics[1]["calibration"]["rmse_position_m"],
            "target2_calibration_rmse_m": trial_metrics[2]["calibration"]["rmse_position_m"],
        })
    selected_motion_sigma_m = min(motion_trials, key=lambda row: row["calibration_score"])["motion_sigma_m"]

    summary_rows = []
    variant_metrics = {}
    for variant in VARIANTS:
        rows_by_target = run_variant(
            observations, nodes, model, normalization, variant,
            selected_motion_sigma_m if variant == "combined_motion" else None,
        )
        variant_metrics[variant] = {}
        variant_root = args.output / variant
        for target in TARGETS:
            metrics = add_offline_truth_and_metrics(rows_by_target[target], tracks[target], delay_s)
            variant_metrics[variant][target] = metrics
            write_csv(variant_root / f"target{target}_triangulation_global.csv", rows_by_target[target])
            for split in ("calibration", "evaluation"):
                value = metrics[split]
                summary_rows.append({
                    "variant": variant,
                    "target": target,
                    "split": split,
                    "frames": value["frames"],
                    "valid_frames": value["valid_frames"],
                    "valid_fraction": value["valid_fraction"],
                    "rmse_position_m": value["rmse_position_m"],
                    "median_position_error_m": value["median_position_error_m"],
                    "p90_position_error_m": value["p90_position_error_m"],
                    "maximum_step_m": value["maximum_step_m"],
                    "motion_sigma_m": selected_motion_sigma_m if variant == "combined_motion" else "",
                })

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "confidence_ablation_summary.csv", summary_rows)
    write_csv(args.output / "node_target_reliability.csv", reliability_rows)
    write_csv(args.output / "motion_prior_calibration_trials.csv", motion_trials)

    confidence_rows = []
    for target, frames in observations.items():
        for frame_index in sorted(frames):
            for item in frames[frame_index]:
                confidence_rows.append({
                    "frame_index": frame_index,
                    "time_s": item.time_s,
                    "time_second": item.time_second,
                    "calibration_frame": item.calibration_frame,
                    "target": target,
                    "node_id": item.node,
                    "azimuth_deg": item.azimuth_deg,
                    "zenith_deg": item.zenith_deg,
                    "azimuth_strength": item.azimuth_strength,
                    "zenith_strength": item.zenith_strength,
                    "quality_score": quality_score(item, normalization),
                    "equal_sigma_deg": sigma_for(item, "equal_weight", model, normalization),
                    "node_sigma_deg": sigma_for(item, "node_precision", model, normalization),
                    "strength_sigma_deg": sigma_for(item, "strength_precision", model, normalization),
                    "combined_sigma_deg": sigma_for(item, "combined_precision", model, normalization),
                    "combined_precision": 1.0 / sigma_for(item, "combined_precision", model, normalization) ** 2,
                })
    write_csv(args.output / "target_node_confidence_timeseries.csv", confidence_rows)

    manifest = {
        "task": "Baoding dual-source calibration-frozen node and MUSIC-peak confidence ablation",
        "variants": list(VARIANTS),
        "selected_motion_sigma_m": selected_motion_sigma_m,
        "motion_prior_selection": {
            "rule": "minimum worst-target plus 0.25 mean target calibration RMSE",
            "candidates_m": list(MOTION_SIGMA_CANDIDATES_M),
            "trials": motion_trials,
        },
        "confidence_model": {
            "quality_feature": "equal average of node-normalized log geometric azimuth/zenith peak strength and log peak-strength ratio",
            "quality_bins": QUALITY_BINS,
            "calibration_only": True,
            "model": serialize_model(model),
            "normalization": {str(key): value for key, value in normalization.items()},
        },
        "association": "corrected global association with calibrated azimuth offsets; this A1 ablation keeps those associations frozen and does not reassign peaks",
        "motion_prior": "constant-velocity acoustic prediction added as a Gaussian prior to weighted multi-ray normal equations",
        "gps_role": "orientation/identity/reliability/motion-hyperparameter calibration on the declared calibration interval; offline scoring only after calibration",
        "evaluation_independent_of_gps_updates": True,
        "metrics": {
            variant: {str(target): values for target, values in targets.items()}
            for variant, targets in variant_metrics.items()
        },
        "sources": {
            "frontend": str(args.frontend),
            "frontend_csv_sha256": {
                path.name: sha256(path) for path in sorted(args.frontend.glob("dual_doa_node_*.csv"))
            },
            "association": str(args.association),
            "association_gate_sha256": sha256(gate_path),
            "node_coordinates": str(gps_root / "20171107baoding.nod"),
            "node_coordinates_sha256": sha256(gps_root / "20171107baoding.nod"),
        },
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha256(Path(__file__).resolve()),
    }
    write_json(args.output / "confidence_frontend_manifest.json", manifest)
    print(json.dumps({
        "output": str(args.output),
        "selected_motion_sigma_m": selected_motion_sigma_m,
        "evaluation": {
            variant: {
                str(target): variant_metrics[variant][target]["evaluation"] for target in TARGETS
            }
            for variant in VARIANTS
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
