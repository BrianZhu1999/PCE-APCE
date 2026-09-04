#!/usr/bin/env python3
"""Paper-aligned eight-node three-target association audit.

GPS is used only on a short calibration interval around the paper's printed
initial state (13:27:54) to determine per-node angle orientation and target
ordering. After calibration, assignments use only angular continuity and robust
multi-ray geometry.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
from pathlib import Path

import shuangyuan_dual_association as base

NODES = (1, 3, 5, 6, 7, 8, 11, 13)
AZ_PERMS = tuple(itertools.permutations(range(3)))
ZEN_PERMS = tuple(itertools.permutations(range(3)))


def hms_seconds(value: int | float) -> float:
    text = str(int(float(value))).zfill(6)
    return int(text[:2]) * 3600 + int(text[2:4]) * 60 + int(text[4:])


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def parse_gps(path: Path) -> list[tuple[float, tuple[float, float, float]]]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) >= 8:
            try:
                rows.append((hms_seconds(fields[7]), (float(fields[4]), float(fields[5]), float(fields[6]))))
            except ValueError:
                pass
    return rows


def nearest(track: list[tuple[float, tuple[float, float, float]]], time_s: float):
    return min(track, key=lambda item: abs(item[0] - time_s))[1]


def raw_angles(row: dict[str, str]) -> tuple[list[float], list[float]]:
    return ([float(row[f"azimuth_{index}_deg"]) for index in (1, 2, 3)], [float(row[f"zenith_{index}_deg"]) for index in (1, 2, 3)])


def assignment_cost(az: list[float], zen: list[float], az_perm, zen_perm, truth_angles, node_xyz) -> float:
    return sum(math.hypot(base.circular_error(az[az_perm[target]], truth_angles[target][0]), zen[zen_perm[target]] - truth_angles[target][1]) for target in range(3))


def fit_calibration(raw_rows, gps_tracks, nodes, start_index: int, stop_index: int):
    result = {}
    for node in NODES:
        best = None
        node_xyz = nodes[node]
        truth_angles_by_index = {}
        for index in range(start_index, stop_index):
            time_s = float(raw_rows[node][index]["time_s"])
            truth = [nearest(gps_tracks[target], time_s) for target in (1, 2, 3)]
            truth_angles_by_index[index] = [base.truth_angles(position, node_xyz) for position in truth]
        for sign in (1.0, -1.0):
            for az_offset in range(-180, 181, 10):
                for zen_offset in range(-30, 31, 5):
                    total = 0.0
                    for index in range(start_index, stop_index):
                        truth_angles = truth_angles_by_index[index]
                        raw_az, raw_zen = raw_angles(raw_rows[node][index])
                        transformed_az = [(sign * value + az_offset) % 360.0 for value in raw_az]
                        transformed_zen = [value + zen_offset for value in raw_zen]
                        frame_cost = min(assignment_cost(transformed_az, transformed_zen, az_perm, zen_perm, truth_angles, node_xyz) for az_perm in AZ_PERMS for zen_perm in ZEN_PERMS)
                        total += frame_cost
                    candidate = (total, sign, az_offset, zen_offset)
                    if best is None or candidate < best:
                        best = candidate
        result[node] = {"sign": best[1], "azimuth_offset_deg": best[2], "zenith_offset_deg": best[3], "calibration_cost_deg": best[0] / max(stop_index - start_index, 1), "calibration_frames": stop_index - start_index}
    return result


def choose_assignment(raw_az, raw_zen, calibration, previous_position, node_xyz):
    sign = calibration["sign"]; az_offset = calibration["azimuth_offset_deg"]; zen_offset = calibration["zenith_offset_deg"]
    az = [(sign * value + az_offset) % 360.0 for value in raw_az]
    zen = [value + zen_offset for value in raw_zen]
    best = None
    for az_perm in AZ_PERMS:
        for zen_perm in ZEN_PERMS:
            cost = 0.0
            for target in range(3):
                if previous_position[target] is None:
                    continue
                truth = base.truth_angles(previous_position[target], node_xyz)
                cost += math.hypot(base.circular_error(az[az_perm[target]], truth[0]), zen[zen_perm[target]] - truth[1])
            candidate = (cost, az_perm, zen_perm)
            if best is None or candidate < best:
                best = candidate
    return [(az[best[1][target]], zen[best[2][target]]) for target in range(3)], best[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--gps-root", type=Path, required=True)
    parser.add_argument("--nod", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--calibration-start", type=int, default=132754)
    parser.add_argument("--calibration-seconds", type=float, default=10.0)
    args = parser.parse_args()
    rows = {node: read_csv(args.input_root / f"node{node}" / f"triple_doa_node_{node}_132614.csv") for node in NODES}
    count = min(len(values) for values in rows.values())
    rows = {node: values[:count] for node, values in rows.items()}
    nodes_raw = {}
    for line in args.nod.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) >= 6:
            nodes_raw[int(fields[2])] = (float(fields[3]), float(fields[4]), float(fields[5]))
    center = tuple(sum(nodes_raw[node][axis] for node in NODES) / len(NODES) for axis in range(3))
    nodes = {node: tuple(nodes_raw[node][axis] - center[axis] for axis in range(3)) for node in NODES}
    gps_files = {1: "GPS1_plane1.gps", 2: "GPS3_plane2.gps", 3: "GPS4_plane2to3.gps"}
    gps_tracks = {target: parse_gps(args.gps_root / filename) for target, filename in gps_files.items()}
    cal_time = hms_seconds(args.calibration_start)
    times = [float(rows[NODES[0]][index]["time_s"]) for index in range(count)]
    start_index = min(range(count), key=lambda index: abs(times[index] - cal_time))
    stop_index = min(count, start_index + max(1, int(args.calibration_seconds / max(times[1] - times[0], 1e-6))))
    calibration = fit_calibration(rows, gps_tracks, nodes, start_index, stop_index)
    associated = {node: [] for node in NODES}; triangulated = {target: [] for target in (1, 2, 3)}
    previous_position = [None, None, None]
    for index in range(start_index, count):
        paired = {}
        for node in NODES:
            paired[node], _ = choose_assignment(*raw_angles(rows[node][index]), calibration[node], previous_position, nodes[node])
        positions = []
        for target in (1, 2, 3):
            position, inliers, condition = base.robust_triangulate({node: paired[node][target - 1] for node in NODES}, nodes)
            positions.append(position)
            truth = nearest(gps_tracks[target], times[index])
            error = math.dist(position, tuple(truth[axis] - center[axis] for axis in range(3))) if position is not None else None
            triangulated[target].append({"frame_index": index, "time_s": times[index], "target": target, "x": position[0] if position else None, "y": position[1] if position else None, "z": position[2] if position else None, "valid": position is not None, "inlier_nodes": len(inliers), "condition_number": condition, "gps_error_m": error})
        if all(position is not None for position in positions):
            previous_position = positions
        for node in NODES:
            out = {"frame_index": index, "time_s": times[index], "node_id": node}
            for target, (azimuth, zenith) in enumerate(paired[node], 1):
                out[f"target{target}_az_deg"] = azimuth; out[f"target{target}_zenith_deg"] = zenith
            associated[node].append(out)
    args.output_root.mkdir(parents=True, exist_ok=True)
    for node, values in associated.items():
        with (args.output_root / f"associated_paper8_node_{node}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    summary = {}
    for target, values in triangulated.items():
        with (args.output_root / f"target{target}_triangulation_paper8.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
        valid = [row for row in values if row["valid"]]
        errors = [float(row["gps_error_m"]) for row in valid]
        jumps = [math.dist((a["x"], a["y"], a["z"]), (b["x"], b["y"], b["z"])) for a, b in zip(valid, valid[1:])]
        summary[str(target)] = {"valid_fraction": len(valid) / max(len(values), 1), "median_gps_error_m": statistics.median(errors) if errors else None, "p90_gps_error_m": sorted(errors)[min(len(errors) - 1, int(0.90 * len(errors)))] if errors else None, "median_jump_m": statistics.median(jumps) if jumps else None, "p90_jump_m": sorted(jumps)[min(len(jumps) - 1, int(0.90 * len(jumps)))] if jumps else None}
    manifest = {"claim_status": "paper8_gps_calibrated_association_audit", "nodes": NODES, "excluded_node": 2, "input_root": str(args.input_root), "gps_mapping": gps_files, "calibration_start_index": start_index, "calibration_stop_index_exclusive": stop_index, "calibration_start_hhmmss": args.calibration_start, "calibration_seconds": args.calibration_seconds, "calibration": calibration, "post_calibration_gps_used": False, "summary": summary, "warning": "GPS is used only for angle orientation/target ordering calibration and audit; this is not yet a PCE/APCE gate."}
    (args.output_root / "paper8_association_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
