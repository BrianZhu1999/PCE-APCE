#!/usr/bin/env python3
"""Provisional three-source association with geometry-aware axis pairing.

The three-peak frontend ranks azimuth and zenith independently.  This module
resolves the within-node azimuth/zenith pairing sequentially: robust ray
triangulation supplies target-position proposals, and each node chooses among
the six zenith permutations using reprojection and temporal continuity.  The
result remains target-unlabelled and inspection-only.
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

DEFAULT_NODES = (1, 3, 5, 6, 7, 8, 11, 13)
PERMUTATIONS = tuple(itertools.permutations(range(3)))


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def obs_from_row(row: dict[str, str]) -> tuple[list[float], list[float]]:
    return (
        [float(row[f"azimuth_{i}_deg"]) for i in (1, 2, 3)],
        [float(row[f"zenith_{i}_deg"]) for i in (1, 2, 3)],
    )


def triangulate(paired: dict[int, list[tuple[float, float]]], nodes: dict[int, tuple[float, float, float]]):
    positions = []
    for target in range(3):
        observations = {node: values[target] for node, values in paired.items()}
        position, inliers, condition = base.robust_triangulate(observations, nodes)
        positions.append(position)
    return positions


def angular_cost(observation: tuple[float, float], position: tuple[float, float, float] | None, node_xyz: tuple[float, float, float]) -> float:
    if position is None:
        return 80.0
    truth = base.truth_angles(position, node_xyz)
    return math.hypot(base.circular_error(observation[0], truth[0]), observation[1] - truth[1])


def associate_frame(raw: dict[int, tuple[list[float], list[float]]], nodes: dict[int, tuple[float, float, float]], previous_positions, previous_observations):
    # Start with rank-wise pairing; two geometry/assignment iterations are enough
    # for the short-frame provisional audit and avoid a factorial global search.
    permutations = {node: (0, 1, 2) for node in raw}
    positions = None
    for _ in range(3):
        paired = {
            node: [(values[0][target], values[1][permutations[node][target]]) for target in range(3)]
            for node, values in raw.items()
        }
        positions = triangulate(paired, nodes)
        for node, values in raw.items():
            best = None
            for permutation in PERMUTATIONS:
                cost = 0.0
                for target in range(3):
                    cost += angular_cost((values[0][target], values[1][permutation[target]]), positions[target], nodes[node])
                    if previous_observations and node in previous_observations:
                        previous = previous_observations[node][target]
                        cost += 0.35 * math.hypot(
                            base.circular_error(values[0][target], previous[0]),
                            values[1][permutation[target]] - previous[1],
                        )
                if best is None or cost < best[0]:
                    best = (cost, permutation)
            permutations[node] = best[1]
    paired = {
        node: [(values[0][target], values[1][permutations[node][target]]) for target in range(3)]
        for node, values in raw.items()
    }
    positions = triangulate(paired, nodes)
    return paired, positions, permutations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--nod", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--nodes", default=",".join(str(node) for node in DEFAULT_NODES))
    args = parser.parse_args()
    nodes_list = tuple(int(value) for value in args.nodes.split(",") if value.strip())
    if len(nodes_list) < 3:
        raise RuntimeError("at least three nodes are required")
    rows = {node: read(args.input_root / f"node{node}" / f"triple_doa_node_{node}_132614.csv") for node in nodes_list}
    count = min(len(values) for values in rows.values())
    if args.max_frames is not None:
        count = min(count, args.max_frames)
    rows = {node: values[:count] for node, values in rows.items()}
    nodes = base.parse_nod(args.nod)
    associated = {node: [] for node in nodes_list}
    triangulated = {target: [] for target in range(3)}
    previous_positions = [None, None, None]
    previous_observations = None
    for index in range(count):
        raw = {node: obs_from_row(rows[node][index]) for node in nodes_list}
        paired, positions, permutations = associate_frame(raw, nodes, previous_positions, previous_observations)
        if all(position is not None for position in positions):
            previous_positions = positions
        previous_observations = paired
        for target in range(3):
            position = positions[target]
            triangulated[target].append({
                "frame_index": index,
                "time_s": float(rows[nodes_list[0]][index]["time_s"]),
                "target": target + 1,
                "x": position[0] if position else None,
                "y": position[1] if position else None,
                "z": position[2] if position else None,
                "valid": position is not None,
            })
        for node in nodes_list:
            out = {"frame_index": index, "time_s": float(rows[node][index]["time_s"]), "node_id": node,
                   "zenith_permutation": "".join(str(value + 1) for value in permutations[node])}
            for target, (azimuth, zenith) in enumerate(paired[node], 1):
                out[f"target{target}_az_deg"] = azimuth
                out[f"target{target}_zenith_deg"] = zenith
            associated[node].append(out)
    args.output_root.mkdir(parents=True, exist_ok=True)
    for node, values in associated.items():
        with (args.output_root / f"associated_triple_provisional_node_{node}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    for target, values in triangulated.items():
        with (args.output_root / f"target{target + 1}_triangulation_provisional.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    summary = {}
    for target, values in triangulated.items():
        valid = [row for row in values if row["valid"]]
        jumps = []
        for first, second in zip(valid, valid[1:]):
            jumps.append(math.dist((first["x"], first["y"], first["z"]), (second["x"], second["y"], second["z"])))
        summary[str(target + 1)] = {"valid_fraction": len(valid) / max(len(values), 1),
                                    "median_frame_jump_m": statistics.median(jumps) if jumps else None,
                                    "p90_frame_jump_m": sorted(jumps)[min(len(jumps) - 1, int(0.90 * len(jumps)))] if jumps else None}
    manifest = {
        "claim_status": "frontend_association_inspection_only",
        "input_root": str(args.input_root), "nodes": nodes_list, "common_frames": count,
        "association_rule": "sequential robust triangulation plus per-node six-way zenith permutation using reprojection and continuity",
        "gps_used": False, "target_identity": "provisional rank labels only",
        "summary": summary,
        "warning": "No GPS truth, target identity, PCE/APCE, or superiority gate is inferred.",
    }
    (args.output_root / "provisional_triple_association_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
