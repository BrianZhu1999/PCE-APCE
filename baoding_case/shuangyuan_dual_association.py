#!/usr/bin/env python3
"""Calibrate and associate the freshly extracted shuangyuan_4 DOA pairs.

GPS is used only on the frozen calibration interval to select a common
cross-modal delay, node orientation corrections, and the initial target
ordering.  After calibration, target identity is propagated by angular
continuity alone.  A post-hoc GPS comparison and robust multi-node ray
triangulation are then used for admission auditing.
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


IP_TO_NODE = {40: 3, 43: 6, 46: 13, 47: 1, 48: 2, 49: 7, 5: 11, 54: 5, 61: 8}
NODE_TO_IP = {node: ip for ip, node in IP_TO_NODE.items()}
NODES = [1, 2, 3, 5, 6, 7, 8, 11, 13]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hms_seconds(value: float) -> float:
    integer = int(value)
    frac = value - integer
    text = str(integer).zfill(6)
    return int(text[:2]) * 3600 + int(text[2:4]) * 60 + int(text[4:]) + frac


def seconds_to_hhmmss(value: float) -> int:
    value = int(math.floor(value + 1e-9))
    hour = value // 3600
    minute = (value - hour * 3600) // 60
    second = value - hour * 3600 - minute * 60
    return hour * 10000 + minute * 100 + second


def shift_hhmmss(value: int, delta_s: int) -> int:
    return seconds_to_hhmmss(hms_seconds(value) + delta_s)


def parse_nod(path: Path) -> dict[int, tuple[float, float, float]]:
    output = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) >= 6:
            output[int(fields[2])] = (float(fields[3]), float(fields[4]), float(fields[5]))
    return output


def parse_gps(path: Path) -> dict[int, tuple[float, float, float]]:
    output = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split("\t")
        if len(fields) < 8:
            continue
        try:
            output[int(float(fields[7]))] = (float(fields[4]), float(fields[5]), float(fields[6]))
        except ValueError:
            continue
    return output


def fuse(a: dict[int, tuple[float, float, float]], b: dict[int, tuple[float, float, float]]) -> dict[int, tuple[float, float, float]]:
    return {key: tuple((a[key][i] + b[key][i]) / 2.0 for i in range(3)) for key in a.keys() & b.keys()}


def nearest_truth(track: dict[int, tuple[float, float, float]], second: int) -> tuple[float, float, float] | None:
    if second in track:
        return track[second]
    nearby = [key for key in track if abs(key - second) <= 1]
    return track[min(nearby, key=lambda key: abs(key - second))] if nearby else None


def circular_difference(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def circular_error(a: float, b: float) -> float:
    return abs(circular_difference(a, b))


def truth_angles(position: tuple[float, float, float], node: tuple[float, float, float]) -> tuple[float, float]:
    dx, dy, dz = (position[i] - node[i] for i in range(3))
    az = math.degrees(math.atan2(dy, dx)) % 360.0
    zen = 90.0 - math.degrees(math.atan2(dz, math.hypot(dx, dy)))
    return az, zen


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for index, row in enumerate(rows):
        row["_index"] = index
        for key in ("time_s", "azimuth_1_deg", "zenith_1_deg", "azimuth_2_deg", "zenith_2_deg"):
            row[key] = float(row[key])
        row["time_second"] = int(float(row.get("time_hhmmss") or seconds_to_hhmmss(row["time_s"])))
    return rows


def raw_pair(row: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    return (
        (row["azimuth_1_deg"], row["zenith_1_deg"]),
        (row["azimuth_2_deg"], row["zenith_2_deg"]),
    )


def transform_pair(
    pair: tuple[tuple[float, float], tuple[float, float]],
    az_sign: float,
    az_offset: float,
    zen_sign: float,
    zen_offset: float,
) -> list[tuple[float, float]]:
    return [
        ((az_sign * az + az_offset) % 360.0, zen_sign * zen + zen_offset)
        for az, zen in pair
    ]


def pair_cost(
    observations: list[tuple[float, float]],
    truths: list[tuple[float, float]],
) -> tuple[float, tuple[int, int], list[float]]:
    permutations = [(0, 1), (1, 0)]
    candidates = []
    for permutation in permutations:
        errors = [
            math.hypot(
                circular_error(observations[i][0], truths[permutation[i]][0]),
                observations[i][1] - truths[permutation[i]][1],
            )
            for i in range(2)
        ]
        candidates.append((sum(errors), permutation, errors))
    return min(candidates, key=lambda item: item[0])


def calibrate_node(
    rows: list[dict],
    node: tuple[float, float, float],
    target1: dict[int, tuple[float, float, float]],
    target2: dict[int, tuple[float, float, float]],
    delay_s: int,
    calibration_end: int,
) -> dict:
    calibration_rows = [row for row in rows if row["time_second"] <= calibration_end]
    best = None
    for az_sign in (1.0, -1.0):
        for zen_sign in (1.0, -1.0):
            az_offset, zen_offset = 0.0, 0.0
            assignments = []
            used = []
            for _ in range(5):
                assignments = []
                az_diffs, zen_diffs = [], []
                used = []
                for row in calibration_rows:
                    second = shift_hhmmss(row["time_second"], -delay_s)
                    p1, p2 = nearest_truth(target1, second), nearest_truth(target2, second)
                    if p1 is None or p2 is None:
                        continue
                    truths = [truth_angles(p1, node), truth_angles(p2, node)]
                    transformed = transform_pair(raw_pair(row), az_sign, az_offset, zen_sign, zen_offset)
                    _, permutation, _ = pair_cost(transformed, truths)
                    for obs_index, truth_index in enumerate(permutation):
                        az_diffs.append(circular_difference(truths[truth_index][0], az_sign * raw_pair(row)[obs_index][0]))
                        zen_diffs.append(truths[truth_index][1] - zen_sign * raw_pair(row)[obs_index][1])
                    assignments.append((row, truths, permutation))
                    used.append(row)
                if az_diffs:
                    az_offset = math.degrees(math.atan2(sum(math.sin(math.radians(x)) for x in az_diffs), sum(math.cos(math.radians(x)) for x in az_diffs))) % 360.0
                if zen_diffs:
                    zen_offset = statistics.median(zen_diffs)
            errors = []
            target_errors = [[], []]
            for row, truths, permutation in assignments:
                transformed = transform_pair(raw_pair(row), az_sign, az_offset, zen_sign, zen_offset)
                for obs_index, truth_index in enumerate(permutation):
                    error = math.hypot(
                        circular_error(transformed[obs_index][0], truths[truth_index][0]),
                        transformed[obs_index][1] - truths[truth_index][1],
                    )
                    errors.append(error)
                    target_errors[truth_index].append(error)
            objective = statistics.median(errors) if errors else float("inf")
            candidate = {
                "az_sign": az_sign,
                "az_offset_deg": az_offset,
                "zenith_sign": zen_sign,
                "zenith_offset_deg": zen_offset,
                "calibration_rows": len(errors) // 2,
                "median_joint_error_deg": objective,
                "p90_joint_error_deg": sorted(errors)[min(len(errors) - 1, int(0.90 * len(errors)))] if errors else None,
                "target1_median_error_deg": statistics.median(target_errors[0]) if target_errors[0] else None,
                "target2_median_error_deg": statistics.median(target_errors[1]) if target_errors[1] else None,
            }
            if best is None or (candidate["median_joint_error_deg"], candidate["p90_joint_error_deg"]) < (
                best["median_joint_error_deg"],
                best["p90_joint_error_deg"],
            ):
                best = candidate
    assert best is not None
    return best


def apply_transform(row: dict, calibration: dict) -> list[tuple[float, float]]:
    return transform_pair(
        raw_pair(row),
        calibration["az_sign"],
        calibration["az_offset_deg"],
        calibration["zenith_sign"],
        calibration["zenith_offset_deg"],
    )


def associate_rows(
    rows: list[dict],
    calibration: dict,
    target1: dict[int, tuple[float, float, float]],
    target2: dict[int, tuple[float, float, float]],
    node: tuple[float, float, float],
    calibration_end: int,
) -> list[dict]:
    previous = None
    output = []
    for row in rows:
        transformed = apply_transform(row, calibration)
        second = row["time_second"]
        if second <= calibration_end:
            p1, p2 = nearest_truth(target1, second), nearest_truth(target2, second)
            if p1 is not None and p2 is not None:
                truths = [truth_angles(p1, node), truth_angles(p2, node)]
                _, permutation, errors = pair_cost(transformed, truths)
                labeled = [transformed[permutation.index(i)] for i in range(2)]
                association_cost = sum(errors)
            else:
                labeled, association_cost = transformed, float("nan")
        elif previous is None:
            labeled, association_cost = transformed, 0.0
        else:
            direct = sum(
                math.hypot(circular_error(transformed[i][0], previous[i][0]), transformed[i][1] - previous[i][1])
                for i in range(2)
            )
            swapped = sum(
                math.hypot(circular_error(transformed[1 - i][0], previous[i][0]), transformed[1 - i][1] - previous[i][1])
                for i in range(2)
            )
            if swapped < direct:
                labeled, association_cost = [transformed[1], transformed[0]], swapped
            else:
                labeled, association_cost = transformed, direct
        previous = labeled
        output.append(
            {
                "node_id": row["node_id"],
                "frame_index": row["_index"],
                "time_s": row["time_s"],
                "time_second": row["time_second"],
                "target1_az_deg": labeled[0][0],
                "target1_zenith_deg": labeled[0][1],
                "target2_az_deg": labeled[1][0],
                "target2_zenith_deg": labeled[1][1],
                "association_cost_deg": association_cost,
                "calibration_frame": row["time_second"] <= calibration_end,
            }
        )
    return output


def direction(azimuth: float, zenith: float) -> tuple[float, float, float]:
    elevation = math.radians(90.0 - zenith)
    azimuth = math.radians(azimuth)
    return (
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
    )


def ray_solve(observations: dict[int, tuple[float, float]], nodes: dict[int, tuple[float, float, float]], selected: list[int]) -> tuple[tuple[float, float, float] | None, float]:
    matrices = []
    targets = []
    for node in selected:
        vector = direction(*observations[node])
        q = [[float(i == j) - vector[i] * vector[j] for j in range(3)] for i in range(3)]
        location = nodes[node]
        matrices.append(q)
        targets.append(tuple(sum(q[i][j] * location[j] for j in range(3)) for i in range(3)))
    a = [[sum(matrix[i][j] for matrix in matrices) for j in range(3)] for i in range(3)]
    b = [sum(target[i] for target in targets) for i in range(3)]
    try:
        import numpy as np
        aa, bb = np.asarray(a), np.asarray(b)
        condition = float(np.linalg.cond(aa))
        if not math.isfinite(condition) or condition > 1e8:
            return None, condition
        position = np.linalg.solve(aa, bb)
        return (float(position[0]), float(position[1]), float(position[2])), condition
    except Exception:
        return None, float("inf")


def residual(position: tuple[float, float, float], node: int, observation: tuple[float, float], nodes: dict[int, tuple[float, float, float]]) -> float:
    truth = truth_angles(position, nodes[node])
    return math.hypot(circular_error(observation[0], truth[0]), observation[1] - truth[1])


def robust_triangulate(observations: dict[int, tuple[float, float]], nodes: dict[int, tuple[float, float, float]], threshold: float = 25.0) -> tuple[tuple[float, float, float] | None, list[int], float]:
    available = sorted(observations)
    if len(available) < 3:
        return None, [], float("inf")
    candidates = []
    for subset in itertools.combinations(available, 3):
        position, condition = ray_solve(observations, nodes, list(subset))
        if position is None:
            continue
        errors = {node: residual(position, node, observations[node], nodes) for node in available}
        inliers = [node for node in available if errors[node] <= threshold]
        if len(inliers) >= 3:
            candidates.append((-len(inliers), statistics.median(errors[node] for node in inliers), condition, position, inliers))
    if not candidates:
        return None, [], float("inf")
    _, _, _, _, inliers = min(candidates, key=lambda item: item[:3])
    position, condition = ray_solve(observations, nodes, inliers)
    return position, inliers, condition


def evaluate_target(
    associated: dict[int, list[dict]],
    nodes: dict[int, tuple[float, float, float]],
    truth: dict[int, tuple[float, float, float]],
    target: int,
    delay_s: int,
) -> tuple[list[dict], dict]:
    n_frames = min(len(rows) for rows in associated.values())
    rows_out = []
    for index in range(n_frames):
        observations = {}
        time_s = None
        for node in associated:
            row = associated[node][index]
            time_s = row["time_s"] if time_s is None else time_s
            observations[node] = (row[f"target{target}_az_deg"], row[f"target{target}_zenith_deg"])
        position, inliers, condition = robust_triangulate(observations, nodes)
        truth_position = nearest_truth(truth, shift_hhmmss(seconds_to_hhmmss(time_s), -delay_s)) if time_s is not None else None
        result = {
            "frame_index": index,
            "time_s": time_s,
            "available_nodes": len(observations),
            "inlier_nodes": len(inliers),
            "condition_number": condition,
            "valid": position is not None,
        }
        if position is not None and truth_position is not None:
            result["px"], result["py"], result["pz"] = position
            result["truth_x"], result["truth_y"], result["truth_z"] = truth_position
            result["position_error_m"] = math.sqrt(sum((position[i] - truth_position[i]) ** 2 for i in range(3)))
        rows_out.append(result)
    valid = [row for row in rows_out if row["valid"] and "position_error_m" in row]
    errors = [row["position_error_m"] for row in valid]
    metrics = {
        "frames": len(rows_out),
        "valid_frames": len(valid),
        "valid_fraction": len(valid) / len(rows_out) if rows_out else 0.0,
        "median_position_error_m": statistics.median(errors) if errors else None,
        "p90_position_error_m": sorted(errors)[min(len(errors) - 1, int(0.90 * len(errors)))] if errors else None,
        "rmse_position_m": math.sqrt(sum(error * error for error in errors) / len(errors)) if errors else None,
        "median_inlier_nodes": statistics.median(row["inlier_nodes"] for row in valid) if valid else None,
    }
    return rows_out, metrics


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-root", type=Path, default=Path("<PRIVATE_DATA_ROOT>/2017保定实验/2017保定实验"))
    parser.add_argument("--frontend", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-end", type=int, default=125600)
    args = parser.parse_args()

    archive = args.remote_root / "20171107保定实验"
    project = archive / "project/20171107baoding"
    gps_root = archive / "GPS_data"
    nodes = parse_nod(gps_root / "20171107baoding.nod")
    target1 = fuse(parse_gps(gps_root / "GPS1_plane1.gps"), parse_gps(gps_root / "GPS2_plane1.gps"))
    target2 = fuse(parse_gps(gps_root / "GPS3_plane2.gps"), parse_gps(gps_root / "GPS4_plane2to3.gps"))
    args.output.mkdir(parents=True, exist_ok=True)
    fresh = {node: load_rows(args.frontend / f"dual_doa_node_{node}_125540_125900.csv") for node in NODES}
    common_frames = min(len(rows) for rows in fresh.values())
    fresh = {node: rows[:common_frames] for node, rows in fresh.items()}

    delay_candidates = []
    calibrations_by_delay = {}
    for delay in (-1, 0, 1, 2, 3, 4):
        calibrations = {
            node: calibrate_node(fresh[node], nodes[node], target1, target2, delay, args.calibration_end)
            for node in NODES
        }
        values = [item["median_joint_error_deg"] for item in calibrations.values() if math.isfinite(item["median_joint_error_deg"])]
        aggregate = statistics.median(values) if values else float("inf")
        delay_candidates.append({"delay_s": delay, "median_node_error_deg": aggregate, "node_errors_deg": values})
        calibrations_by_delay[delay] = calibrations
    selected_delay = min(delay_candidates, key=lambda row: (row["median_node_error_deg"], abs(row["delay_s"])))["delay_s"]
    calibrations = calibrations_by_delay[selected_delay]

    associated = {}
    association_rows = []
    for node in NODES:
        rows = associate_rows(fresh[node], calibrations[node], target1, target2, nodes[node], args.calibration_end)
        associated[node] = rows
        association_rows.extend(rows)
        write_csv(args.output / f"associated_node_{node}.csv", rows)

    target1_rows, target1_metrics = evaluate_target(associated, nodes, target1, 1, selected_delay)
    target2_rows, target2_metrics = evaluate_target(associated, nodes, target2, 2, selected_delay)
    write_csv(args.output / "target1_triangulation.csv", target1_rows)
    write_csv(args.output / "target2_triangulation.csv", target2_rows)
    result = {
        "task": "2017 Baoding shuangyuan_4 fresh dual-peak MUSIC and frozen association admission",
        "source_frontend": str(args.frontend),
        "source_frontend_hashes": {str(path.name): sha256(path) for path in sorted(args.frontend.glob("dual_doa_node_*.csv"))},
        "nodes": NODES,
        "calibration_interval": {"end_hhmmss": args.calibration_end, "gps_role": "delay/orientation/initial ordering only"},
        "delay_candidates": delay_candidates,
        "selected_delay_s": selected_delay,
        "node_calibrations": {str(node): calibrations[node] for node in NODES},
        "post_calibration_association": {
            "rule": "angular continuity between adjacent paired DOA frames; GPS not used after calibration interval",
            "rows_per_node": {str(node): len(associated[node]) for node in NODES},
        },
        "triangulation": {"target1": target1_metrics, "target2": target2_metrics},
        "admission": {
            "target1_position_gate": bool(
                target1_metrics["valid_fraction"] >= 0.90
                and target1_metrics["median_position_error_m"] is not None
                and target1_metrics["median_position_error_m"] <= 200.0
                and target1_metrics["p90_position_error_m"] <= 500.0
            ),
            "target2_position_gate": bool(
                target2_metrics["valid_fraction"] >= 0.90
                and target2_metrics["median_position_error_m"] is not None
                and target2_metrics["median_position_error_m"] <= 200.0
                and target2_metrics["p90_position_error_m"] <= 500.0
            ),
        },
        "provenance": {
            "nod_sha256": sha256(gps_root / "20171107baoding.nod"),
            "gps_sha256": {name: sha256(gps_root / name) for name in (
                "GPS1_plane1.gps", "GPS2_plane1.gps", "GPS3_plane2.gps", "GPS4_plane2to3.gps"
            )},
            "script_sha256": sha256(Path(__file__)),
        },
    }
    result["admission"]["dual_target_position_gate"] = bool(result["admission"]["target1_position_gate"] and result["admission"]["target2_position_gate"])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "shuangyuan4_dual_association_gate.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.output / "association_all_nodes.csv", association_rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
