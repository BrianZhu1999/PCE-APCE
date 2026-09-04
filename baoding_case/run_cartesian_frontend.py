#!/usr/bin/env python3
"""Build a GPS-free Cartesian observation bundle from calibrated DOA rows.

The input is the frozen single-source WAV-MUSIC replay bundle.  Each frame is
robustly triangulated from the paired azimuth/elevation observations and is
written as a local ENU observation with an explicit PSD covariance.  GPS is
copied only as an offline truth stream; it is never used to choose a frame or
to update a position.

This is deliberately a separate bundle/runner boundary.  Existing angle
domain result directories are not modified.
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
from typing import Any

import torch


NODE_IDS = (1, 2, 3, 5, 6, 7, 8, 11, 13)
R_NAMES = tuple(f"R_{row}{col}" for row in range(3) for col in range(3))
POS_NAMES = ("y_E", "y_N", "y_U")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def direction(azimuth_deg: float, elevation_deg: float) -> torch.Tensor:
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    return torch.tensor(
        (
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        ),
        dtype=torch.float64,
    )


def signed_angle_deg(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def truth_angles(position: torch.Tensor, node: torch.Tensor) -> tuple[float, float]:
    delta = position - node.to(dtype=position.dtype, device=position.device)
    azimuth = math.degrees(math.atan2(float(delta[1]), float(delta[0]))) % 360.0
    elevation = math.degrees(math.atan2(float(delta[2]), math.hypot(float(delta[0]), float(delta[1]))))
    return azimuth, elevation


def angular_residual_deg(position: torch.Tensor, node: torch.Tensor, azimuth: float, elevation: float) -> float:
    expected_azimuth, expected_elevation = truth_angles(position, node)
    return math.hypot(signed_angle_deg(azimuth - expected_azimuth), elevation - expected_elevation)


def solve_rays(
    frame: dict[int, tuple[float, float, float]],
    nodes: dict[int, torch.Tensor],
    selected: list[int],
    robust_weights: dict[int, float] | None = None,
) -> tuple[torch.Tensor | None, float]:
    matrices: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for node in selected:
        azimuth, elevation, sigma_deg = frame[node]
        ray = direction(azimuth, elevation)
        projection = torch.eye(3, dtype=torch.float64) - ray[:, None] @ ray[None, :]
        weight = 1.0 / max(math.radians(sigma_deg), math.radians(1.0)) ** 2
        if robust_weights is not None:
            weight *= max(float(robust_weights.get(node, 1.0)), 1e-3)
        matrices.append(weight * projection)
        location = nodes[node].to(dtype=torch.float64, device=projection.device)
        targets.append(weight * projection @ location)
    if len(matrices) < 3:
        return None, float("inf")
    matrix = torch.stack(matrices).sum(dim=0)
    condition = float(torch.linalg.cond(matrix))
    if not math.isfinite(condition):
        return None, condition
    try:
        position = torch.linalg.solve(matrix, torch.stack(targets).sum(dim=0))
    except RuntimeError:
        return None, condition
    return position, condition


def robust_triangulate(
    frame: dict[int, tuple[float, float, float]],
    nodes: dict[int, torch.Tensor],
    threshold_deg: float,
    condition_limit: float,
    irls_iterations: int = 0,
    irls_delta_deg: float = 8.0,
) -> tuple[torch.Tensor | None, list[int], float, float]:
    """Return position, inliers, condition number and inlier RMS."""
    available = sorted(frame)
    if len(available) < 3:
        return None, [], float("inf"), float("inf")
    candidates: list[tuple[Any, ...]] = []
    for subset in itertools.combinations(available, 3):
        position, condition = solve_rays(frame, nodes, list(subset))
        if position is None or condition > condition_limit:
            continue
        residuals = {
            node: angular_residual_deg(position, nodes[node], frame[node][0], frame[node][1])
            for node in available
        }
        inliers = [node for node in available if residuals[node] <= threshold_deg]
        if len(inliers) < 3:
            continue
        rms = math.sqrt(sum(residuals[node] ** 2 for node in inliers) / len(inliers))
        candidates.append(
            (-len(inliers), rms, condition, tuple(inliers), position)
        )
    if not candidates:
        return None, [], float("inf"), float("inf")
    _, _, _, inlier_tuple, _ = min(candidates, key=lambda item: item[:3])
    inliers = list(inlier_tuple)
    position, condition = solve_rays(frame, nodes, inliers)
    if position is None:
        return None, [], condition, float("inf")
    for _ in range(max(0, int(irls_iterations))):
        residuals_by_node = {
            node: angular_residual_deg(position, nodes[node], frame[node][0], frame[node][1])
            for node in inliers
        }
        delta = max(float(irls_delta_deg), 1e-6)
        weights = {node: (1.0 if value <= delta else delta / max(value, 1e-6)) for node, value in residuals_by_node.items()}
        updated, updated_condition = solve_rays(frame, nodes, inliers, weights)
        if updated is None:
            break
        position, condition = updated, updated_condition
    residuals = [
        angular_residual_deg(position, nodes[node], frame[node][0], frame[node][1])
        for node in inliers
    ]
    rms = math.sqrt(sum(value * value for value in residuals) / len(residuals))
    return position, inliers, condition, rms


def covariance_from_rays(
    position: torch.Tensor,
    frame: dict[int, tuple[float, float, float]],
    nodes: dict[int, torch.Tensor],
    inliers: list[int],
    rms_deg: float,
    base_sigma_deg: float,
    condition: float,
) -> torch.Tensor:
    """Approximate Cartesian covariance by tangent-plane error propagation."""
    information = torch.zeros((3, 3), dtype=torch.float64)
    propagated = torch.zeros((3, 3), dtype=torch.float64)
    for node in inliers:
        azimuth, elevation, sigma_deg = frame[node]
        sigma_rad = max(math.radians(float(sigma_deg)), math.radians(1.0))
        ray = direction(azimuth, elevation)
        projection = torch.eye(3, dtype=torch.float64) - ray[:, None] @ ray[None, :]
        weight = 1.0 / sigma_rad**2
        location = nodes[node].to(dtype=position.dtype, device=position.device)
        distance = float(torch.linalg.vector_norm(position - location))
        information += weight * projection
        # Angular error becomes a transverse displacement proportional to range.
        propagated += (distance * distance) * weight * projection
    inverse_information = torch.linalg.pinv(information, rcond=1e-10)
    covariance = inverse_information @ propagated @ inverse_information
    residual_factor = max(1.0, rms_deg / max(base_sigma_deg, 1e-6)) ** 2
    geometry_factor = 1.0 + min(25.0, math.log10(max(condition, 1.0)))
    coverage_factor = max(1.0, 5.0 / max(len(inliers), 1))
    covariance = covariance * residual_factor * geometry_factor * coverage_factor
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(0.25**2)
    return eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T


def load_observations(path: Path) -> dict[str, dict[float, dict[int, dict[str, float | bool]]]]:
    output: dict[str, dict[float, dict[int, dict[str, float | bool]]]] = {}
    with path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("valid", "True").lower() != "true":
                continue
            segment = row["segment"]
            time_s = float(row["time_s"])
            node = int(row["node_id"])
            output.setdefault(segment, {}).setdefault(time_s, {})[node] = {
                "azimuth_deg": float(row["azimuth_deg"]),
                "elevation_deg": float(row["elevation_deg"]),
                "concentration": float(row.get("concentration", 1.0)),
            }
    return output


def load_nodes(path: Path) -> dict[int, torch.Tensor]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(node): torch.tensor((float(value["x"]), float(value["y"]), float(value["z"])), dtype=torch.float64)
        for node, value in manifest["nodes"].items()
    }


def load_truth(path: Path) -> dict[float, tuple[float, float, float]]:
    output = {}
    if not path.exists():
        return output
    with path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            output[float(row["time_s"])] = tuple(float(row[name]) for name in ("px", "py", "pz"))
    return output


def nearest_truth(truth: dict[float, tuple[float, float, float]], time_s: float) -> tuple[float, float, float] | None:
    if not truth:
        return None
    key = min(truth, key=lambda value: abs(value - time_s))
    return truth[key] if abs(key - time_s) <= 2.0 else None


def offline_position_metrics(
    rows: list[dict[str, object]],
    truth: dict[float, tuple[float, float, float]],
) -> dict[str, float | int | None]:
    """Evaluate the frozen acoustic positions against GPS after the run.

    The source frontend already writes ``gps_truth.csv`` in its local ENU
    frame.  The node coordinates, in contrast, are translated here before
    triangulation, so subtracting ``center`` again would create a false 3.86e7
    m error.
    """
    errors: list[float] = []
    valid_rows = [row for row in rows if row["valid"]]
    evaluated: list[tuple[float, torch.Tensor]] = []
    for row in valid_rows:
        truth_position = nearest_truth(truth, float(row["time_s"]))
        if truth_position is None:
            continue
        estimate = torch.tensor([float(row[name]) for name in POS_NAMES], dtype=torch.float64)
        truth_local = torch.tensor(truth_position, dtype=torch.float64)
        errors.append(float(torch.linalg.vector_norm(estimate - truth_local)))
        evaluated.append((float(row["time_s"]), estimate))
    jumps = [
        float(torch.linalg.vector_norm(right - left))
        for (time_left, left), (time_right, right) in zip(evaluated, evaluated[1:])
        if time_right > time_left and time_right - time_left <= 1.1
    ]
    return {
        "evaluated_frames": len(errors),
        "median_position_error_m": statistics.median(errors) if errors else None,
        "rmse_position_error_m": math.sqrt(sum(value * value for value in errors) / len(errors)) if errors else None,
        "p90_position_error_m": sorted(errors)[min(len(errors) - 1, int(0.90 * len(errors)))] if errors else None,
        "p90_consecutive_jump_m": sorted(jumps)[min(len(jumps) - 1, int(0.90 * len(jumps)))] if jumps else None,
        "max_consecutive_jump_m": max(jumps) if jumps else None,
    }


def median_concentrations(
    observations: dict[str, dict[float, dict[int, dict[str, float | bool]]]],
    calibration_segment: str,
) -> dict[int, float]:
    values: dict[int, list[float]] = {node: [] for node in NODE_IDS}
    for frame in observations.get(calibration_segment, {}).values():
        for node, row in frame.items():
            values.setdefault(node, []).append(max(float(row["concentration"]), 1e-8))
    return {node: statistics.median(rows) if rows else 1.0 for node, rows in values.items()}


def calibrated_node_angle_covariance(
    observations: dict[str, dict[float, dict[int, dict[str, float | bool]]]],
    truth: dict[float, tuple[float, float, float]],
    nodes: dict[int, torch.Tensor],
    center: torch.Tensor,
    calibration_segment: str,
) -> tuple[dict[int, torch.Tensor], dict[int, float]]:
    """Estimate per-node azimuth/elevation covariance from calibration only."""
    local_nodes = {node: value - center for node, value in nodes.items()}
    residuals: dict[int, list[tuple[float, float]]] = {node: [] for node in local_nodes}
    for time_s, frame in observations.get(calibration_segment, {}).items():
        target = nearest_truth(truth, time_s)
        if target is None:
            continue
        position = torch.tensor(target, dtype=torch.float64)
        for node, row in frame.items():
            if node not in local_nodes:
                continue
            azimuth, elevation = truth_angles(position, local_nodes[node])
            residuals[node].append((signed_angle_deg(float(row["azimuth_deg"]) - azimuth), float(row["elevation_deg"]) - elevation))
    covariance: dict[int, torch.Tensor] = {}
    sigma: dict[int, float] = {}
    for node, values in residuals.items():
        if len(values) < 8:
            covariance[node] = torch.eye(2, dtype=torch.float64) * 8.0**2
            sigma[node] = 8.0
            continue
        array = torch.tensor(values, dtype=torch.float64)
        # MAD-centred residuals prevent the calibration's remaining fixed bias
        # from being counted twice after the upstream orientation correction.
        median = array.median(dim=0).values
        centred = array - median
        robust_scale = 1.4826 * centred.abs().median(dim=0).values
        clipped = centred.clamp(min=-3.0 * robust_scale.clamp_min(1.0), max=3.0 * robust_scale.clamp_min(1.0))
        value = clipped.T @ clipped / max(len(clipped) - 1, 1)
        value = 0.5 * (value + value.T) + torch.eye(2, dtype=torch.float64) * 0.25**2
        covariance[node] = value
        sigma[node] = float(torch.sqrt(value.trace() / 2.0).clamp(2.0, 30.0))
    return covariance, sigma


def calibrated_node_angle_bias(
    observations: dict[str, dict[float, dict[int, dict[str, float | bool]]]],
    truth: dict[float, tuple[float, float, float]],
    nodes: dict[int, torch.Tensor],
    center: torch.Tensor,
    calibration_segment: str,
) -> dict[int, tuple[float, float]]:
    """Estimate per-node azimuth/elevation bias from calibration only."""
    local_nodes = {node: value - center for node, value in nodes.items()}
    residuals: dict[int, list[tuple[float, float]]] = {node: [] for node in local_nodes}
    for time_s, frame in observations.get(calibration_segment, {}).items():
        target = nearest_truth(truth, time_s)
        if target is None:
            continue
        position = torch.tensor(target, dtype=torch.float64)
        for node, row in frame.items():
            if node not in local_nodes:
                continue
            azimuth, elevation = truth_angles(position, local_nodes[node])
            residuals[node].append((
                signed_angle_deg(float(row["azimuth_deg"]) - azimuth),
                float(row["elevation_deg"]) - elevation,
            ))
    output: dict[int, tuple[float, float]] = {}
    for node, values in residuals.items():
        output[node] = (
            statistics.median([item[0] for item in values]) if values else 0.0,
            statistics.median([item[1] for item in values]) if values else 0.0,
        )
    return output


def build_segment(
    segment: str,
    observations: dict[float, dict[int, dict[str, float | bool]]],
    nodes: dict[int, torch.Tensor],
    center: torch.Tensor,
    concentration_medians: dict[int, float],
    base_sigma_deg: float,
    threshold_deg: float,
    condition_limit: float,
    min_inliers: int,
    irls_iterations: int,
    irls_delta_deg: float,
    node_angle_sigma_deg: dict[int, float] | None = None,
    node_angle_bias_deg: dict[int, tuple[float, float]] | None = None,
    bias_alpha: float = 0.0,
) -> tuple[list[dict[str, object]], dict[str, float | int | bool]]:
    local_nodes = {node: value - center for node, value in nodes.items()}
    output: list[dict[str, object]] = []
    for source_index, time_s in enumerate(sorted(observations)):
        raw_frame = observations[time_s]
        frame: dict[int, tuple[float, float, float]] = {}
        for node, item in raw_frame.items():
            concentration = max(float(item["concentration"]), 1e-8)
            reference = max(concentration_medians.get(node, concentration), 1e-8)
            nominal_sigma = node_angle_sigma_deg.get(node, base_sigma_deg) if node_angle_sigma_deg else base_sigma_deg
            sigma_deg = nominal_sigma * math.sqrt(reference / max(concentration, 0.25 * reference))
            bias_az, bias_el = node_angle_bias_deg.get(node, (0.0, 0.0)) if node_angle_bias_deg else (0.0, 0.0)
            frame[node] = (
                (float(item["azimuth_deg"]) - bias_alpha * bias_az) % 360.0,
                float(item["elevation_deg"]) - bias_alpha * bias_el,
                max(4.0, min(30.0, sigma_deg)),
            )
        position, inliers, condition, rms = robust_triangulate(
            frame, local_nodes, threshold_deg, condition_limit, irls_iterations, irls_delta_deg
        )
        valid = position is not None and len(inliers) >= min_inliers and rms <= threshold_deg
        row: dict[str, object] = {
            "segment": segment,
            "time_s": time_s,
            "source_frame_index": source_index,
            "available_nodes": len(frame),
            "inlier_nodes": len(inliers),
            "inlier_node_ids": ";".join(str(node) for node in inliers),
            "condition_number": condition,
            "reprojection_rms_deg": rms,
            "association_cost": rms + 0.10 * math.log10(max(condition, 1.0)) if math.isfinite(rms) else float("inf"),
            "valid": bool(valid),
        }
        for name in POS_NAMES + R_NAMES:
            row[name] = None
        if valid and position is not None:
            covariance = covariance_from_rays(
                position, frame, local_nodes, inliers, rms, base_sigma_deg, condition
            )
            row.update({name: float(position[index]) for index, name in enumerate(POS_NAMES)})
            row.update({name: float(covariance[row_index, col_index]) for row_index in range(3) for col_index, name in enumerate(R_NAMES[row_index * 3:(row_index + 1) * 3])})
        output.append(row)
    valid_rows = [row for row in output if row["valid"]]
    return output, {
        "segment": segment,
        "frames": len(output),
        "valid_frames": len(valid_rows),
        "valid_fraction": len(valid_rows) / len(output) if output else 0.0,
        "median_inlier_nodes": statistics.median(int(row["inlier_nodes"]) for row in valid_rows) if valid_rows else None,
        "median_reprojection_rms_deg": statistics.median(float(row["reprojection_rms_deg"]) for row in valid_rows) if valid_rows else None,
        "p90_reprojection_rms_deg": sorted(float(row["reprojection_rms_deg"]) for row in valid_rows)[min(len(valid_rows) - 1, int(0.90 * len(valid_rows)))] if valid_rows else None,
        "finite_psd_covariance": bool(valid_rows) and all(
            all(math.isfinite(float(row[name])) for name in R_NAMES) for row in valid_rows
        ),
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "segment", "time_s", "source_frame_index", "available_nodes", "inlier_nodes", "inlier_node_ids",
        "condition_number", "reprojection_rms_deg", "association_cost", "valid", *POS_NAMES, *R_NAMES,
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-frontend", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-segment", default="danyuan_panxuan_2")
    parser.add_argument("--evaluation-segment", default="danyuan_panxuan_3")
    parser.add_argument("--base-angle-sigma-deg", type=float, default=8.0)
    parser.add_argument("--reprojection-threshold-deg", type=float, default=25.0)
    parser.add_argument("--max-condition-number", type=float, default=1e6)
    parser.add_argument("--min-inliers", type=int, default=5)
    parser.add_argument("--irls-iterations", type=int, default=0)
    parser.add_argument("--irls-delta-deg", type=float, default=8.0)
    parser.add_argument("--use-node-residual-covariance", action="store_true")
    parser.add_argument("--doa-bias-alpha", type=float, default=0.0,
                        help="fraction of calibration median DOA bias to subtract")
    args = parser.parse_args()

    observations = load_observations(args.source_frontend / "observations.csv")
    nodes_global = load_nodes(args.source_frontend / "frontend_manifest.json")
    selected_nodes = {node: nodes_global[node] for node in NODE_IDS if node in nodes_global}
    center = torch.stack(list(selected_nodes.values())).mean(dim=0)
    concentration_medians = median_concentrations(observations, args.calibration_segment)
    truth = load_truth(args.source_frontend / "gps_truth.csv")
    node_covariance, node_sigma = calibrated_node_angle_covariance(
        observations, truth, selected_nodes, center, args.calibration_segment
    )
    node_bias = calibrated_node_angle_bias(
        observations, truth, selected_nodes, center, args.calibration_segment
    )

    all_rows: list[dict[str, object]] = []
    summaries: dict[str, dict[str, object]] = {}
    for segment in (args.calibration_segment, args.evaluation_segment):
        rows, summary = build_segment(
            segment,
            observations.get(segment, {}),
            selected_nodes,
            center,
            concentration_medians,
            args.base_angle_sigma_deg,
            args.reprojection_threshold_deg,
            args.max_condition_number,
            args.min_inliers,
            args.irls_iterations,
            args.irls_delta_deg,
            node_sigma if args.use_node_residual_covariance else None,
            node_bias,
            args.doa_bias_alpha,
        )
        all_rows.extend(rows)
        summaries[segment] = summary
        summaries[segment]["offline_gps_evaluation"] = offline_position_metrics(rows, truth)

    args.output.mkdir(parents=True, exist_ok=True)
    write_rows(args.output / "observations_cartesian.csv", all_rows)
    truth_source = args.source_frontend / "gps_truth.csv"
    if truth_source.exists():
        (args.output / "gps_truth.csv").write_bytes(truth_source.read_bytes())
    manifest_source = args.source_frontend / "frontend_manifest.json"
    write_json(args.output / "frontend_manifest.json", {
        "task": "single-source Cartesian acoustic observation bundle",
        "source_frontend": str(args.source_frontend),
        "source_observations_sha256": sha256(args.source_frontend / "observations.csv"),
        "source_manifest_sha256": sha256(manifest_source),
        "coordinate_system": "local ENU metres; global node coordinates translated by arithmetic node-centre",
        "gps_truth_coordinate_system": "source frontend local ENU metres; no additional translation applied",
        "nodes": {str(node): {"x": float(value[0] - center[0]), "y": float(value[1] - center[1]), "z": float(value[2] - center[2])} for node, value in selected_nodes.items()},
        "center_xyz_global": [float(value) for value in center],
        "gps_role": "copied to gps_truth.csv for offline evaluation only; never used for frame acceptance, triangulation, covariance, or tracking update",
        "orientation_calibration": "inherited from source frontend_calibration.json",
        "covariance_model": "tangent-plane angular error propagation with residual, geometry and inlier-count inflation; eigenvalue floor 0.25 m^2",
        "parameters": {
            "base_angle_sigma_deg": args.base_angle_sigma_deg,
            "reprojection_threshold_deg": args.reprojection_threshold_deg,
            "max_condition_number": args.max_condition_number,
            "min_inliers": args.min_inliers,
            "irls_iterations": args.irls_iterations,
            "irls_delta_deg": args.irls_delta_deg,
            "use_node_residual_covariance": args.use_node_residual_covariance,
            "doa_bias_alpha": args.doa_bias_alpha,
        },
        "concentration_medians_calibration": concentration_medians,
        "node_angle_covariance_calibration_deg2": {str(node): value.tolist() for node, value in node_covariance.items()},
        "node_angle_sigma_calibration_deg": {str(node): value for node, value in node_sigma.items()},
        "node_angle_bias_calibration_deg": {
            str(node): {"azimuth": value[0], "elevation": value[1]}
            for node, value in node_bias.items()
        },
        "segments": summaries,
    })
    write_json(args.output / "cartesian_frontend_gate.json", {
        "task": "GPS-free single-source Cartesian frontend admission",
        "calibration_segment": args.calibration_segment,
        "evaluation_segment": args.evaluation_segment,
        "gps_role": "offline evaluation only",
        "admission_thresholds": {
            "evaluation_valid_fraction_min": 0.90,
            "evaluation_median_reprojection_rms_deg_max": 20.0,
            "evaluation_p90_reprojection_rms_deg_max": 30.0,
            "finite_psd_covariance": True,
        },
        "calibration": summaries.get(args.calibration_segment),
        "evaluation": summaries.get(args.evaluation_segment),
        "cartesian_frontend_admitted": bool(
            summaries.get(args.evaluation_segment, {}).get("valid_fraction", 0.0) >= 0.90
            and (summaries.get(args.evaluation_segment, {}).get("median_reprojection_rms_deg") or float("inf")) <= 20.0
            and (summaries.get(args.evaluation_segment, {}).get("p90_reprojection_rms_deg") or float("inf")) <= 30.0
            and summaries.get(args.evaluation_segment, {}).get("finite_psd_covariance", False)
        ),
        "provenance": {"runner_sha256": sha256(Path(__file__))},
    })
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
