#!/usr/bin/env python3
"""Global two-target association for the fresh Baoding MUSIC peaks.

The frontend returns two *unlabelled* peaks at every node and frame.  The
previous per-node continuity rule can make locally plausible but globally
inconsistent swaps.  This module resolves the binary swaps jointly across
all nodes:

* GPS is used only on the frozen calibration interval to select delay,
  orientation corrections, and the initial target ordering;
* after the calibration boundary, all 2**9 node-wise swap hypotheses are
  scored with acoustic angular continuity and multi-node ray geometry;
* GPS is not used by the post-calibration association or triangulation.

The output is an audit artifact.  It does not run PCE/APCE and does not alter
the Figure 2/3 experiment chains.
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

import shuangyuan_dual_association as base


NODES = base.NODES
MAX_ENUM_NODES = 9
TOP_HYPOTHESES = 12
MAX_ANGULAR_RESIDUAL_DEG = 70.0
GEOMETRY_TRIM_NODES = 2
POSITION_SCALE_M = 250.0
VELOCITY_SMOOTHING = 0.35


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def angle_cost(
    observation: tuple[float, float],
    position: tuple[float, float, float] | None,
    node: tuple[float, float, float],
) -> float:
    if position is None:
        return MAX_ANGULAR_RESIDUAL_DEG
    truth = base.truth_angles(position, node)
    value = math.hypot(
        base.circular_error(observation[0], truth[0]),
        observation[1] - truth[1],
    )
    return min(float(value), MAX_ANGULAR_RESIDUAL_DEG)


def robust_geometry_cost(
    observations: dict[int, tuple[float, float]],
    nodes: dict[int, tuple[float, float, float]],
) -> tuple[tuple[float, float, float] | None, float, int, float]:
    """Robust position estimate plus a residual score.

    The global association should inherit the same robust geometry notion used
    by the final admission gate, not an all-node least-squares solve that can
    be pulled off the road by one bad peak.  We therefore score each
    hypothesis with the existing three-ray robust triangulation logic and then
    measure residuals on the returned inlier set.
    """
    position, inliers, condition = base.robust_triangulate(observations, nodes)
    if position is None:
        return None, float("inf"), 0, condition
    residuals = [
        base.residual(position, node, observations[node], nodes)
        for node in inliers
    ]
    cost = float(statistics.median(residuals)) if residuals else float("inf")
    trimmed = max(0, len(observations) - len(inliers))
    # Penalize aggressive trimming mildly; the main cost is the robust residual.
    cost += 6.0 * trimmed
    return position, cost, trimmed, condition


def enumerate_hypotheses(
    pairs: dict[int, list[tuple[float, float]]],
    nodes: dict[int, tuple[float, float, float]],
    predicted: tuple[tuple[float, float, float] | None, tuple[float, float, float] | None],
    previous_observations: dict[int, list[tuple[float, float]]] | None,
) -> tuple[dict, dict]:
    """Return the best global assignment and diagnostics."""
    ordered_nodes = sorted(pairs)
    if len(ordered_nodes) > MAX_ENUM_NODES:
        raise RuntimeError(f"global association supports at most {MAX_ENUM_NODES} nodes")
    candidates = []
    # The cheap continuity screen keeps the exhaustive search tractable while
    # retaining enough alternatives for the geometric check.
    for mask in range(1 << len(ordered_nodes)):
        target_obs = {1: {}, 2: {}}
        continuity = 0.0
        previous_cost = 0.0
        for bit, node in enumerate(ordered_nodes):
            first, second = pairs[node]
            if (mask >> bit) & 1:
                first, second = second, first
            target_obs[1][node] = first
            target_obs[2][node] = second
            continuity += angle_cost(first, predicted[0], nodes[node])
            continuity += angle_cost(second, predicted[1], nodes[node])
            if previous_observations is not None and node in previous_observations:
                previous = previous_observations[node]
                previous_cost += math.hypot(
                    base.circular_error(first[0], previous[0][0]),
                    first[1] - previous[0][1],
                )
                previous_cost += math.hypot(
                    base.circular_error(second[0], previous[1][0]),
                    second[1] - previous[1][1],
                )
        candidates.append((continuity + 0.35 * previous_cost, mask, target_obs))
    candidates.sort(key=lambda item: item[0])
    best = None
    scored = []
    for cheap_cost, mask, target_obs in candidates[:TOP_HYPOTHESES]:
        positions = {}
        geometry_cost = 0.0
        condition_max = 0.0
        for target in (1, 2):
            position, reproj, _, condition = robust_geometry_cost(target_obs[target], nodes)
            positions[target] = position
            geometry_cost += reproj if math.isfinite(reproj) else MAX_ANGULAR_RESIDUAL_DEG
            condition_max = max(condition_max, condition)
        position_cost = 0.0
        if predicted[0] is not None and positions[1] is not None:
            position_cost += min(
                math.dist(positions[1], predicted[0]) / POSITION_SCALE_M,
                10.0,
            )
        if predicted[1] is not None and positions[2] is not None:
            position_cost += min(
                math.dist(positions[2], predicted[1]) / POSITION_SCALE_M,
                10.0,
            )
        total = cheap_cost + 0.50 * geometry_cost + 10.0 * position_cost
        record = {
            "mask": int(mask),
            "cheap_cost_deg": float(cheap_cost),
            "geometry_cost_deg": float(geometry_cost),
            "position_cost_scaled": float(position_cost),
            "total_cost": float(total),
            "condition_number_max": float(condition_max),
            "positions": positions,
            "target_obs": target_obs,
        }
        scored.append(record)
        if best is None or record["total_cost"] < best["total_cost"]:
            best = record
    if best is None:
        raise RuntimeError("global association produced no hypothesis")
    return best, {
        "hypotheses_enumerated": len(candidates),
        "hypotheses_geometry_scored": len(scored),
        "top_hypotheses": [
            {
                key: value
                for key, value in record.items()
                if key not in ("positions", "target_obs")
            }
            for record in sorted(scored, key=lambda item: item["total_cost"])
        ],
    }


def label_calibration_frame(
    pairs: dict[int, list[tuple[float, float]]],
    nodes: dict[int, tuple[float, float, float]],
    target1: dict[int, tuple[float, float, float]],
    target2: dict[int, tuple[float, float, float]],
    time_second: int,
    delay_s: int,
) -> tuple[dict[int, list[tuple[float, float]]], int]:
    labeled: dict[int, list[tuple[float, float]]] = {}
    p1 = base.nearest_truth(target1, base.shift_hhmmss(time_second, -delay_s))
    p2 = base.nearest_truth(target2, base.shift_hhmmss(time_second, -delay_s))
    if p1 is None or p2 is None:
        return {node: list(pair) for node, pair in pairs.items()}, 0
    for node, pair in pairs.items():
        truths = [base.truth_angles(p1, nodes[node]), base.truth_angles(p2, nodes[node])]
        _, permutation, _ = base.pair_cost(pair, truths)
        labeled[node] = [pair[permutation.index(0)], pair[permutation.index(1)]]
    return labeled, 1


def position_from_labels(
    labeled: dict[int, list[tuple[float, float]]],
    nodes: dict[int, tuple[float, float, float]],
) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None]:
    output = []
    for target in (0, 1):
        observations = {node: pair[target] for node, pair in labeled.items()}
        position, _, _, _ = robust_geometry_cost(observations, nodes)
        output.append(position)
    return output[0], output[1]


def run_global_association(
    fresh: dict[int, list[dict]],
    nodes: dict[int, tuple[float, float, float]],
    target1: dict[int, tuple[float, float, float]],
    target2: dict[int, tuple[float, float, float]],
    calibrations: dict[int, dict],
    selected_delay: int,
    calibration_end: int,
) -> tuple[dict[int, list[dict]], dict]:
    common_frames = min(len(rows) for rows in fresh.values())
    fresh = {node: rows[:common_frames] for node, rows in fresh.items()}
    associated = {node: [] for node in sorted(fresh)}
    previous_positions = [None, None]
    velocities = [None, None]
    previous_time = None
    previous_observations = None
    frame_diagnostics = []
    for index in range(common_frames):
        time_s = float(next(iter(fresh.values()))[index]["time_s"])
        time_second = int(next(iter(fresh.values()))[index]["time_second"])
        pairs = {
            node: base.apply_transform(fresh[node][index], calibrations[node])
            for node in sorted(fresh)
        }
        if time_second <= calibration_end:
            labeled, gps_used = label_calibration_frame(
                pairs, nodes, target1, target2, time_second, selected_delay
            )
            positions = position_from_labels(labeled, nodes)
            mask = None
            diag = {
                "frame_index": index,
                "time_s": time_s,
                "time_second": time_second,
                "calibration_frame": True,
                "gps_used_for_initial_ordering": bool(gps_used),
                "global_mask": mask,
                "total_cost": 0.0,
            }
        else:
            dt = 640.0 / 3050.0 if previous_time is None else max(time_s - previous_time, 1e-6)
            predicted = []
            for target in (0, 1):
                if previous_positions[target] is None:
                    predicted.append(None)
                elif velocities[target] is None:
                    predicted.append(previous_positions[target])
                else:
                    predicted.append(tuple(
                        previous_positions[target][j] + velocities[target][j] * dt
                        for j in range(3)
                    ))
            best, diag = enumerate_hypotheses(
                pairs,
                nodes,
                (predicted[0], predicted[1]),
                previous_observations,
            )
            labeled = {
                node: [best["target_obs"][1][node], best["target_obs"][2][node]]
                for node in sorted(pairs)
            }
            positions = (best["positions"][1], best["positions"][2])
            diag.update({
                "frame_index": index,
                "time_s": time_s,
                "time_second": time_second,
                "calibration_frame": False,
                "gps_used_for_initial_ordering": False,
                "global_mask": best["mask"],
                "total_cost": best["total_cost"],
                "position_target1": best["positions"][1],
                "position_target2": best["positions"][2],
            })
        if previous_time is not None:
            dt = max(time_s - previous_time, 1e-6)
            for target in (0, 1):
                if positions[target] is not None and previous_positions[target] is not None:
                    instant = tuple(
                        (positions[target][j] - previous_positions[target][j]) / dt
                        for j in range(3)
                    )
                    if velocities[target] is None:
                        velocities[target] = instant
                    else:
                        velocities[target] = tuple(
                            (1.0 - VELOCITY_SMOOTHING) * velocities[target][j]
                            + VELOCITY_SMOOTHING * instant[j]
                            for j in range(3)
                        )
        previous_positions = [positions[0], positions[1]]
        previous_time = time_s
        previous_observations = labeled
        frame_diagnostics.append(diag)
        for node in sorted(labeled):
            pair = labeled[node]
            associated[node].append({
                "node_id": node,
                "frame_index": index,
                "time_s": time_s,
                "time_second": time_second,
                "target1_az_deg": pair[0][0],
                "target1_zenith_deg": pair[0][1],
                "target2_az_deg": pair[1][0],
                "target2_zenith_deg": pair[1][1],
                "association_cost_deg": float(diag.get("total_cost", 0.0)),
                "calibration_frame": bool(diag["calibration_frame"]),
                "global_assignment_mask": diag.get("global_mask"),
            })
    return associated, {
        "frames": common_frames,
        "calibration_end_hhmmss": calibration_end,
        "gps_role": "orientation, delay, and calibration-interval initial ordering only",
        "post_calibration_rule": "exhaustive node-wise binary assignment screened by acoustic continuity and scored by multi-node ray geometry plus position continuity",
        "frame_diagnostics": frame_diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-root", type=Path, default=Path("<PRIVATE_DATA_ROOT>/2017保定实验/2017保定实验"))
    parser.add_argument("--frontend", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-end", type=int, default=125600)
    args = parser.parse_args()

    archive = args.remote_root / "20171107保定实验"
    gps_root = archive / "GPS_data"
    nodes = base.parse_nod(gps_root / "20171107baoding.nod")
    target1 = base.fuse(
        base.parse_gps(gps_root / "GPS1_plane1.gps"),
        base.parse_gps(gps_root / "GPS2_plane1.gps"),
    )
    target2 = base.fuse(
        base.parse_gps(gps_root / "GPS3_plane2.gps"),
        base.parse_gps(gps_root / "GPS4_plane2to3.gps"),
    )
    fresh = {
        node: base.load_rows(args.frontend / f"dual_doa_node_{node}_125540_125900.csv")
        for node in NODES
    }
    common_frames = min(len(rows) for rows in fresh.values())
    fresh = {node: rows[:common_frames] for node, rows in fresh.items()}
    delay_candidates = []
    calibrations_by_delay = {}
    for delay in (-1, 0, 1, 2, 3, 4):
        calibrations = {
            node: base.calibrate_node(
                fresh[node], nodes[node], target1, target2, delay, args.calibration_end
            )
            for node in NODES
        }
        values = [
            item["median_joint_error_deg"]
            for item in calibrations.values()
            if math.isfinite(item["median_joint_error_deg"])
        ]
        aggregate = statistics.median(values) if values else float("inf")
        delay_candidates.append({
            "delay_s": delay,
            "median_node_error_deg": aggregate,
            "node_errors_deg": values,
        })
        calibrations_by_delay[delay] = calibrations
    selected_delay = min(
        delay_candidates,
        key=lambda row: (row["median_node_error_deg"], abs(row["delay_s"])),
    )["delay_s"]
    calibrations = calibrations_by_delay[selected_delay]
    associated, audit = run_global_association(
        fresh, nodes, target1, target2, calibrations, selected_delay, args.calibration_end
    )
    args.output.mkdir(parents=True, exist_ok=True)
    for node, rows in associated.items():
        write_csv(args.output / f"associated_global_node_{node}.csv", rows)
    target1_rows, target1_metrics = base.evaluate_target(
        associated, nodes, target1, 1, selected_delay
    )
    target2_rows, target2_metrics = base.evaluate_target(
        associated, nodes, target2, 2, selected_delay
    )
    write_csv(args.output / "target1_triangulation_global.csv", target1_rows)
    write_csv(args.output / "target2_triangulation_global.csv", target2_rows)
    gate = {
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
    }
    result = {
        "task": "2017 Baoding shuangyuan_4 global two-target association",
        "source_frontend": str(args.frontend),
        "source_frontend_hashes": {
            str(path.name): sha256(path)
            for path in sorted(args.frontend.glob("dual_doa_node_*.csv"))
        },
        "nodes": NODES,
        "selected_delay_s": selected_delay,
        "delay_candidates": delay_candidates,
        "node_calibrations": {str(node): calibrations[node] for node in NODES},
        "orientation_transform": {
            "azimuth_rule": "(az_sign * raw_azimuth + frozen_azimuth_offset) mod 360",
            "zenith_rule": "zenith_sign * raw_zenith + frozen_zenith_offset",
            "azimuth_offset_applied": True,
        },
        "triangulation": {"target1": target1_metrics, "target2": target2_metrics},
        "admission": gate,
        "global_association_audit": audit,
        "provenance": {
            "nod_sha256": sha256(gps_root / "20171107baoding.nod"),
            "gps_sha256": {
                name: sha256(gps_root / name)
                for name in (
                    "GPS1_plane1.gps",
                    "GPS2_plane1.gps",
                    "GPS3_plane2.gps",
                    "GPS4_plane2to3.gps",
                )
            },
            "frontend_script_sha256": sha256(
                Path(__file__).with_name("shuangyuan_dual_frontend.py")
            ),
            "dual_association_script_sha256": sha256(Path(base.__file__)),
            "association_script_sha256": sha256(Path(__file__)),
        },
    }
    result["admission"]["dual_target_position_gate"] = bool(
        result["admission"]["target1_position_gate"]
        and result["admission"]["target2_position_gate"]
    )
    (args.output / "shuangyuan4_global_association_gate.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "selected_delay_s": selected_delay,
        "triangulation": result["triangulation"],
        "admission": result["admission"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
