#!/usr/bin/env python3
"""Blind near-field observability audit for the 2017 Baoding archive.

Calibration uses only ``danyuan_panxuan_2``. All reported admission metrics
come from the held-out ``danyuan_panxuan_3`` segment. Historical DOA products
are used to decide whether 3-D tracking is observable before revisiting the
raw-WAV frontend or running PCE/APCE again.
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

import torch

try:
    from .run_baoding import IP_TO_NODE, hms_seconds, parse_gps, parse_nod, sha256, signed_deg
except ImportError:
    from run_baoding import IP_TO_NODE, hms_seconds, parse_gps, parse_nod, sha256, signed_deg


def circular_mean(values: list[float]) -> float:
    return math.degrees(math.atan2(
        sum(math.sin(math.radians(value)) for value in values),
        sum(math.cos(math.radians(value)) for value in values),
    )) % 360.0


def circular_median(values: list[float]) -> float:
    return min(values, key=lambda candidate: sum(abs(signed_deg(value - candidate)) for value in values)) % 360.0


def decimal_hms_seconds(value: float) -> float:
    integer = int(value)
    fraction = value - integer
    text = str(integer).zfill(6)
    return int(text[:2]) * 3600 + int(text[2:4]) * 60 + int(text[4:]) + fraction


def parse_historical_doa(segment: Path) -> dict[int, dict[int, tuple[float, float, int]]]:
    """Return second -> node -> (azimuth, elevation, within-second samples)."""
    grouped: dict[int, dict[int, list[tuple[float, float]]]] = {}
    for path in sorted(segment.glob("*.doa")):
        stem = path.stem.split("_")
        ip = int(stem[-2])
        node = IP_TO_NODE[ip]
        for line in path.read_text(encoding="ascii", errors="replace").splitlines():
            fields = line.split()
            if len(fields) < 11:
                continue
            try:
                azimuth = float(fields[1])
                zenith = float(fields[2])
                clock = float(fields[-1])
            except ValueError:
                continue
            # The acquisition system writes startup-local timestamps in the
            # first few records. Keep only absolute HHMMSS.sss timestamps.
            if clock < 120000.0:
                continue
            second = int(decimal_hms_seconds(clock))
            grouped.setdefault(second, {}).setdefault(node, []).append((azimuth, 90.0 - zenith))
    output: dict[int, dict[int, tuple[float, float, int]]] = {}
    for second, node_rows in grouped.items():
        output[second] = {}
        for node, values in node_rows.items():
            output[second][node] = (
                circular_median([value[0] for value in values]),
                statistics.median(value[1] for value in values),
                len(values),
            )
    return output


def fused_plane1_gps(gps1: Path, gps2: Path) -> dict[int, torch.Tensor]:
    sources = []
    for path in (gps1, gps2):
        sources.append({time_s: torch.tensor((x, y, z), dtype=torch.float64) for time_s, x, y, z in parse_gps(path)})
    output = {}
    for second in sorted(set(sources[0]) | set(sources[1])):
        available = [source[second] for source in sources if second in source]
        if available:
            output[second] = torch.stack(available).mean(dim=0)
    return output


def truth_at(gps: dict[int, torch.Tensor], second: int, delay_s: int) -> torch.Tensor | None:
    target = second - delay_s
    if target in gps:
        return gps[target]
    nearby = [key for key in gps if abs(key - target) <= 1]
    return gps[min(nearby, key=lambda key: abs(key - target))] if nearby else None


def truth_angles(position: torch.Tensor, node: torch.Tensor) -> tuple[float, float]:
    delta = position - node
    azimuth = math.degrees(math.atan2(float(delta[1]), float(delta[0]))) % 360.0
    elevation = math.degrees(math.atan2(float(delta[2]), math.hypot(float(delta[0]), float(delta[1]))))
    return azimuth, elevation


def calibrate(
    observations: dict[int, dict[int, tuple[float, float, int]]],
    gps: dict[int, torch.Tensor],
    nodes: dict[int, torch.Tensor],
    delay_candidates: list[int],
) -> dict:
    candidates = []
    for delay in delay_candidates:
        node_configs = {}
        total_errors = []
        for node, location in nodes.items():
            rows = []
            for second, frame in observations.items():
                if node not in frame:
                    continue
                truth = truth_at(gps, second, delay)
                if truth is None:
                    continue
                truth_az, truth_el = truth_angles(truth, location)
                rows.append((frame[node][0], frame[node][1], truth_az, truth_el))
            orientations = []
            for sign in (1.0, -1.0):
                differences = [signed_deg(truth_az - sign * raw_az) for raw_az, _, truth_az, _ in rows]
                az_offset = circular_mean(differences) if differences else 0.0
                el_offset = statistics.median(truth_el - raw_el for _, raw_el, _, truth_el in rows) if rows else 0.0
                residuals = [
                    math.hypot(
                        signed_deg(sign * raw_az + az_offset - truth_az),
                        raw_el + el_offset - truth_el,
                    )
                    for raw_az, raw_el, truth_az, truth_el in rows
                ]
                orientations.append((statistics.median(residuals) if residuals else float("inf"), sign, az_offset, el_offset, residuals))
            median_error, sign, az_offset, el_offset, residuals = min(orientations)
            mad = statistics.median(abs(value - median_error) for value in residuals) if residuals else float("inf")
            residuals_sorted = sorted(residuals)
            p90_error = residuals_sorted[min(len(residuals_sorted) - 1, int(0.90 * len(residuals_sorted)))] if residuals_sorted else float("inf")
            sigma = max(2.0, 1.4826 * mad, p90_error)
            node_configs[node] = {
                "azimuth_sign": sign,
                "azimuth_offset_deg": az_offset,
                "elevation_offset_deg": el_offset,
                "calibration_median_joint_error_deg": median_error,
                "calibration_p90_joint_error_deg": p90_error,
                "observation_sigma_deg": sigma,
                "calibration_rows": len(rows),
                "eligible": bool(len(rows) >= 20 and median_error <= 45.0),
            }
            total_errors.extend(residuals)
        objective = statistics.median(total_errors) if total_errors else float("inf")
        candidates.append((objective, delay, node_configs))
    objective, delay, node_configs = min(candidates, key=lambda item: (item[0], abs(item[1] - 2)))
    return {"delay_s": delay, "calibration_median_joint_error_deg": objective, "nodes": node_configs}


def apply_calibration(frame: dict[int, tuple[float, float, int]], calibration: dict) -> dict[int, tuple[float, float, float]]:
    output = {}
    for node, (raw_az, raw_el, count) in frame.items():
        config = calibration["nodes"].get(node)
        if not config or not config["eligible"]:
            continue
        azimuth = (config["azimuth_sign"] * raw_az + config["azimuth_offset_deg"]) % 360.0
        elevation = raw_el + config["elevation_offset_deg"]
        output[node] = (azimuth, elevation, config["observation_sigma_deg"])
    return output


def direction(azimuth: float, elevation: float) -> torch.Tensor:
    azimuth, elevation = math.radians(azimuth), math.radians(elevation)
    return torch.tensor((
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
    ), dtype=torch.float64)


def solve_rays(frame: dict[int, tuple[float, float, float]], nodes: dict[int, torch.Tensor], selected: list[int]) -> tuple[torch.Tensor | None, float]:
    matrices, targets = [], []
    for node in selected:
        vector = direction(frame[node][0], frame[node][1])
        projection = torch.eye(3, dtype=torch.float64) - vector[:, None] @ vector[None, :]
        weight = 1.0 / max(frame[node][2], 2.0) ** 2
        matrices.append(weight * projection)
        targets.append(weight * projection @ nodes[node])
    matrix = torch.stack(matrices).sum(dim=0)
    condition = float(torch.linalg.cond(matrix))
    if not math.isfinite(condition) or condition > 1e7:
        return None, condition
    return torch.linalg.solve(matrix, torch.stack(targets).sum(dim=0)), condition


def angular_residual(position: torch.Tensor, node: int, observation: tuple[float, float, float], nodes: dict[int, torch.Tensor]) -> float:
    truth_az, truth_el = truth_angles(position, nodes[node])
    return math.hypot(signed_deg(observation[0] - truth_az), observation[1] - truth_el)


def robust_triangulate(
    frame: dict[int, tuple[float, float, float]],
    nodes: dict[int, torch.Tensor],
    threshold_deg: float,
) -> tuple[torch.Tensor | None, list[int], float]:
    available = sorted(frame)
    if len(available) < 3:
        return None, [], float("inf")
    candidates = []
    for subset in itertools.combinations(available, 3):
        position, condition = solve_rays(frame, nodes, list(subset))
        if position is None:
            continue
        residuals = {node: angular_residual(position, node, frame[node], nodes) for node in available}
        inliers = [node for node in available if residuals[node] <= threshold_deg]
        if len(inliers) < 3:
            continue
        candidates.append((-len(inliers), statistics.median(residuals[node] for node in inliers), condition, position, inliers))
    if not candidates:
        return None, [], float("inf")
    _, _, _, _, inliers = min(candidates, key=lambda item: item[:3])
    position, condition = solve_rays(frame, nodes, inliers)
    return position, inliers, condition


def evaluate(
    observations: dict[int, dict[int, tuple[float, float, int]]],
    gps: dict[int, torch.Tensor],
    nodes: dict[int, torch.Tensor],
    calibration: dict,
    threshold_deg: float,
) -> tuple[list[dict], dict]:
    rows = []
    for second in sorted(observations):
        truth = truth_at(gps, second, calibration["delay_s"])
        if truth is None:
            continue
        frame = apply_calibration(observations[second], calibration)
        position, inliers, condition = robust_triangulate(frame, nodes, threshold_deg)
        row = {"time_s": second, "available_nodes": len(frame), "inlier_nodes": len(inliers), "condition_number": condition, "valid": position is not None}
        if position is not None:
            error = float(torch.linalg.vector_norm(position - truth))
            row.update({"px": float(position[0]), "py": float(position[1]), "pz": float(position[2]), "truth_x": float(truth[0]), "truth_y": float(truth[1]), "truth_z": float(truth[2]), "position_error_m": error})
        rows.append(row)
    valid = [row for row in rows if row["valid"]]
    errors = [row["position_error_m"] for row in valid]
    metrics = {
        "frames": len(rows),
        "valid_frames": len(valid),
        "valid_fraction": len(valid) / len(rows) if rows else 0.0,
        "median_position_error_m": statistics.median(errors) if errors else None,
        "p90_position_error_m": sorted(errors)[min(len(errors) - 1, int(0.90 * len(errors)))] if errors else None,
        "rmse_position_m": math.sqrt(sum(value * value for value in errors) / len(errors)) if errors else None,
        "median_inlier_nodes": statistics.median(row["inlier_nodes"] for row in valid) if valid else None,
    }
    return rows, metrics


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-root", type=Path, default=Path("<PRIVATE_DATA_ROOT>/2017保定实验/2017保定实验"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive = args.remote_root / "20171107保定实验"
    project = archive / "project/20171107baoding"
    nod_path = archive / "GPS_data/20171107baoding.nod"
    gps1 = archive / "GPS_data/GPS1_plane1.gps"
    gps2 = archive / "GPS_data/GPS2_plane1.gps"
    nodes_raw = parse_nod(nod_path)
    nodes = {node: torch.tensor((value["x"], value["y"], value["z"]), dtype=torch.float64) for node, value in nodes_raw.items()}
    gps = fused_plane1_gps(gps1, gps2)
    calibration_observations = parse_historical_doa(project / "danyuan_panxuan_2")
    evaluation_observations = parse_historical_doa(project / "danyuan_panxuan_3")
    calibration = calibrate(calibration_observations, gps, nodes, [-1, 0, 1, 2, 3, 4])
    threshold_candidates = []
    for threshold in (5.0, 10.0, 15.0, 20.0, 30.0, 45.0):
        _, metrics = evaluate(calibration_observations, gps, nodes, calibration, threshold)
        objective = metrics["median_position_error_m"] if metrics["median_position_error_m"] is not None and metrics["valid_fraction"] >= 0.80 else float("inf")
        threshold_candidates.append((objective, threshold, metrics))
    _, threshold, calibration_metrics = min(threshold_candidates, key=lambda item: (item[0], item[1]))
    evaluation_rows, evaluation_metrics = evaluate(evaluation_observations, gps, nodes, calibration, threshold)
    eligible_nodes = [int(node) for node, config in calibration["nodes"].items() if config["eligible"]]
    gate = {
        "task": "2017 Baoding near-field single-helicopter historical-DOA observability audit",
        "calibration_segment": "danyuan_panxuan_2",
        "evaluation_segment": "danyuan_panxuan_3",
        "gps_role": "GPS1/GPS2 fusion; calibration and post-hoc evaluation only",
        "selected_delay_s": calibration["delay_s"],
        "selected_ransac_threshold_deg": threshold,
        "eligible_nodes": eligible_nodes,
        "calibration": calibration_metrics,
        "evaluation": evaluation_metrics,
        "admission_thresholds": {"valid_fraction_min": 0.90, "median_position_error_m_max": 200.0, "p90_position_error_m_max": 500.0},
    }
    gate["three_dimensional_tracking_admitted"] = bool(
        evaluation_metrics["valid_fraction"] >= 0.90
        and evaluation_metrics["median_position_error_m"] is not None
        and evaluation_metrics["median_position_error_m"] <= 200.0
        and evaluation_metrics["p90_position_error_m"] <= 500.0
    )
    gate["direction_tracking_only"] = not gate["three_dimensional_tracking_admitted"]
    gate["provenance"] = {
        "archive": str(args.remote_root),
        "nod_sha256": sha256(nod_path),
        "gps1_sha256": sha256(gps1),
        "gps2_sha256": sha256(gps2),
        "runner_sha256": sha256(Path(__file__)),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "nearfield_calibration.json").write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "nearfield_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.output / "nearfield_evaluation_timeseries.csv", evaluation_rows)
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
