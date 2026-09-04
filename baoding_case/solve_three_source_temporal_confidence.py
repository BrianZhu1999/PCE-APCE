"""GPS-free temporal association for the raw three-source Baoding MUSIC audit.

This module keeps several geometrically plausible line-intersection hypotheses
per frame and selects a continuous path with a constant-velocity prior.  Node
weights are derived from the MUSIC peak strengths and are used only to soften
low-confidence node residuals.  GPS is loaded after the path is selected and is
used for offline scoring only.

The result is deliberately a diagnostic gate, not a claim of reproducing the
authors' private field implementation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from solve_three_source_bearing_triangulation import (
    PERMS,
    corrected_angles,
    line_intersection,
    predicted_angles,
    refine_positions,
    unit_vector,
)
from summarize_three_source_raw_music_xyz import load_gps, load_nod


def circ_abs(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def huber(value: float, delta: float) -> float:
    value = abs(float(value))
    if value <= delta:
        return 0.5 * value * value
    return delta * (value - 0.5 * delta)


def strength_confidences(
    measurements: dict[str, tuple[list[float], list[float]]],
    strengths: dict[str, tuple[list[float], list[float]]],
    confidence_floor: float,
) -> dict[str, np.ndarray]:
    """Return node/source confidence ratios from MUSIC peak strengths.

    The absolute pseudospectrum scale differs between nodes, so each source is
    normalized by the median strength across nodes.  The floor prevents one
    weak node from being silently discarded.
    """
    source_values = {target: [] for target in range(3)}
    for node in measurements:
        az_s, el_s = strengths[node]
        for target in range(3):
            source_values[target].append(
                math.sqrt(max(float(az_s[target]), 1e-12) * max(float(el_s[target]), 1e-12))
            )
    medians = {
        target: max(float(np.median(values)), 1e-12)
        for target, values in source_values.items()
    }
    result = {}
    for node in measurements:
        az_s, el_s = strengths[node]
        result[node] = np.asarray(
            [
                max(
                    confidence_floor,
                    min(
                        2.5,
                        math.sqrt(max(float(az_s[target]), 1e-12) * max(float(el_s[target]), 1e-12))
                        / medians[target],
                    ),
                )
                for target in range(3)
            ],
            dtype=float,
        )
    return result


def assignment_for_prediction(
    predicted: list[tuple[float, float]],
    obs_az: list[float],
    obs_el: list[float],
) -> tuple[tuple[int, ...], list[float]]:
    best = None
    for perm in PERMS:
        errors = [
            math.hypot(
                circ_abs(predicted[target][0], obs_az[perm[target]]),
                abs(predicted[target][1] - obs_el[perm[target]]),
            )
            for target in range(3)
        ]
        score = float(np.mean(errors))
        if best is None or score < best[0]:
            best = (score, perm, errors)
    assert best is not None
    return tuple(best[1]), list(best[2])


def candidate_node_cost(
    estimates: list[np.ndarray],
    measurements: dict[str, tuple[list[float], list[float]]],
    positions: dict[str, np.ndarray],
    strengths: dict[str, tuple[list[float], list[float]]],
    confidence_floor: float,
) -> tuple[float, dict[str, dict]]:
    """Robust, confidence-weighted angular reprojection cost and assignments."""
    confidence = strength_confidences(measurements, strengths, confidence_floor)
    total = 0.0
    assignments: dict[str, dict] = {}
    for node, (obs_az, obs_el) in measurements.items():
        pred = [predicted_angles(x, positions[node]) for x in estimates]
        perm, errors = assignment_for_prediction(pred, obs_az, obs_el)
        node_cost = 0.0
        az_s, el_s = strengths[node]
        for target in range(3):
            idx = perm[target]
            angular = math.hypot(
                circ_abs(pred[target][0], obs_az[idx]),
                abs(pred[target][1] - obs_el[idx]),
            )
            # A weak peak contributes less evidence but is retained in the
            # association search.  The robust loss prevents one bad node from
            # dominating all other nodes.
            node_cost += huber(angular, 18.0) / confidence[node][target]
        node_cost /= 3.0
        total += node_cost
        assignments[node] = {
            "perm": list(perm),
            "score_deg": float(np.mean(errors)),
            "errors_deg": [float(x) for x in errors],
            "confidence": [float(confidence[node][target]) for target in range(3)],
            "assigned_azimuth_peak_strength": [float(az_s[perm[target]]) for target in range(3)],
            "assigned_elevation_peak_strength": [float(el_s[perm[target]]) for target in range(3)],
        }
    # Put the angular loss on an O(1) scale before combining it with the
    # geometric and temporal terms.  Without this normalization, degree^2
    # residuals overwhelm the hold/update decision by several orders.
    return float(total / max(len(measurements) * 18.0 * 18.0, 1.0)), assignments


def refine_positions_weighted(
    initial: list[np.ndarray],
    measurements: dict[str, tuple[list[float], list[float]]],
    strengths: dict[str, tuple[list[float], list[float]]],
    positions: dict[str, np.ndarray],
    assignments: dict[str, dict],
    position_bound_m: float,
    confidence_floor: float,
) -> list[np.ndarray]:
    """Robustly refine positions with node/source peak-confidence weights."""
    confidence = strength_confidences(measurements, strengths, confidence_floor)
    refined = []
    for target, x0 in enumerate(initial):
        observations = []
        for node, (obs_az, obs_el) in measurements.items():
            if node not in assignments:
                continue
            index = int(assignments[node]["perm"][target])
            direction = unit_vector(obs_az[index], obs_el[index])
            observations.append((positions[node], direction, confidence[node][target]))
        if len(observations) < 3:
            refined.append(np.asarray(x0, dtype=float))
            continue

        def residual(x: np.ndarray) -> np.ndarray:
            values = []
            for node_position, observed_direction, weight in observations:
                delta = x - node_position
                norm = max(float(np.linalg.norm(delta)), 1.0)
                predicted_direction = delta / norm
                values.extend((math.sqrt(weight) * (predicted_direction - observed_direction)).tolist())
            return np.asarray(values, dtype=float)

        fit = least_squares(
            residual,
            np.asarray(x0, dtype=float),
            loss="soft_l1",
            f_scale=0.08,
            max_nfev=100,
            bounds=(-position_bound_m, position_bound_m),
        )
        refined.append(fit.x)
    return refined


def enumerate_candidates(
    measurements: dict[str, tuple[list[float], list[float]]],
    strengths: dict[str, tuple[list[float], list[float]]],
    positions: dict[str, np.ndarray],
    anchors: list[str],
    position_bound_m: float,
    prefilter_count: int,
    confidence_floor: float,
) -> list[dict]:
    """Enumerate and refine the best acoustic hypotheses for one frame."""
    available = [node for node in anchors if node in measurements]
    if len(available) < 3:
        return []
    base_records = []
    for anchor_nodes in itertools.combinations(available, 3):
        base = anchor_nodes[0]
        for perm_b in PERMS:
            for perm_c in PERMS:
                perms = {base: (0, 1, 2), anchor_nodes[1]: perm_b, anchor_nodes[2]: perm_c}
                estimates, conditions, residuals = [], [], []
                for target in range(3):
                    dirs = [
                        unit_vector(
                            measurements[node][0][perms[node][target]],
                            measurements[node][1][perms[node][target]],
                        )
                        for node in anchor_nodes
                    ]
                    estimate, condition, residual = line_intersection(
                        [positions[node] for node in anchor_nodes], dirs
                    )
                    estimates.append(estimate)
                    conditions.append(condition)
                    residuals.append(residual)
                if max(conditions) > 1e6 or max(np.linalg.norm(x) for x in estimates) > position_bound_m:
                    continue
                node_assignments = {}
                raw_score = 0.5 * float(np.mean(residuals))
                for node, (obs_az, obs_el) in measurements.items():
                    pred = [predicted_angles(x, positions[node]) for x in estimates]
                    perm, errors = assignment_for_prediction(pred, obs_az, obs_el)
                    node_assignments[node] = {"perm": list(perm), "score_deg": float(np.mean(errors)), "errors_deg": errors}
                    raw_score += float(np.mean(errors))
                # Reject backwards ray intersections before expensive refinement.
                backwards = 0
                for node in anchor_nodes:
                    for target in range(3):
                        idx = perms[node][target]
                        direction = unit_vector(measurements[node][0][idx], measurements[node][1][idx])
                        if float(np.dot(estimates[target] - positions[node], direction)) < 0:
                            backwards += 1
                raw_score += 25.0 * backwards
                base_records.append(
                    {
                        "prefilter_score": float(raw_score),
                        "positions": estimates,
                        "anchor_nodes": list(anchor_nodes),
                        "condition_numbers": [float(x) for x in conditions],
                        "line_residual_m": [float(x) for x in residuals],
                        "node_assignments": node_assignments,
                    }
                )
    base_records.sort(key=lambda row: row["prefilter_score"])
    refined = []
    for record in base_records[:prefilter_count]:
        estimates = refine_positions_weighted(
            record["positions"],
            measurements,
            strengths,
            positions,
            record["node_assignments"],
            position_bound_m,
            confidence_floor,
        )
        _, assignments = candidate_node_cost(
            estimates, measurements, positions, strengths, confidence_floor
        )
        estimates = refine_positions_weighted(
            estimates,
            measurements,
            strengths,
            positions,
            assignments,
            position_bound_m,
            confidence_floor,
        )
        weighted_cost, assignments = candidate_node_cost(
            estimates, measurements, positions, strengths, confidence_floor
        )
        record = dict(record)
        record["positions"] = [np.asarray(x, dtype=float) for x in estimates]
        record["measurement_cost"] = float(weighted_cost + np.mean(record["line_residual_m"]) / 50.0)
        record["node_assignments"] = assignments
        refined.append(record)
    refined.sort(key=lambda row: row["measurement_cost"])
    return refined


def align_positions(candidate: list[np.ndarray], reference: list[np.ndarray]) -> tuple[list[np.ndarray], tuple[int, ...], np.ndarray]:
    best = None
    for perm in PERMS:
        aligned = [np.asarray(candidate[perm[target]], dtype=float) for target in range(3)]
        distances = np.asarray([np.linalg.norm(aligned[target] - reference[target]) for target in range(3)])
        score = float(np.mean(distances))
        if best is None or score < best[0]:
            best = (score, aligned, perm, distances)
    assert best is not None
    return best[1], tuple(best[2]), best[3]


def reorder_assignments(assignments: dict[str, dict], permutation: tuple[int, ...]) -> dict[str, dict]:
    """Map candidate source indices into the persistent track labels."""
    reordered = {}
    for node, row in assignments.items():
        old = row["perm"]
        new = dict(row)
        new["perm"] = [int(old[permutation[target]]) for target in range(3)]
        for key in ("errors_deg", "confidence", "assigned_azimuth_peak_strength", "assigned_elevation_peak_strength"):
            if key in row:
                new[key] = [float(row[key][permutation[target]]) for target in range(3)]
        reordered[node] = new
    return reordered


def load_music(
    json_dir: Path,
    nodes: list[str],
    pattern: str,
    nod: dict[str, dict[str, float]],
    start_frame: int,
    end_frame: int,
):
    data = {}
    for node in nodes:
        payload = json.loads((json_dir / pattern.format(node=node)).read_text(encoding="utf-8"))
        frames = {int(row["frame"]): row for row in payload["frames"]}
        data[node] = {}
        for frame in range(start_frame, end_frame + 1):
            item = frames[frame]
            az, el = corrected_angles(item, nod[node])
            data[node][frame] = {
                "angles": (az, el),
                "strengths": (
                    [float(x) for x in item.get("azimuth_refined_peak", [1.0, 1.0, 1.0])],
                    [float(x) for x in item.get("elevation_refined_peak", [1.0, 1.0, 1.0])],
                ),
            }
    return data


def nearest_xyz(gps: tuple[np.ndarray, np.ndarray], t: float) -> np.ndarray:
    xyz, times = gps
    return xyz[int(np.argmin(np.abs(times - t)))]


def score_gps(positions: list[np.ndarray], gps: list[tuple[np.ndarray, np.ndarray]], origin: np.ndarray, time_s: float):
    truth = [nearest_xyz(track, time_s) - origin for track in gps]
    best = None
    for perm in PERMS:
        errors = [float(np.linalg.norm(positions[target] - truth[perm[target]])) for target in range(3)]
        score = float(np.mean(errors))
        if best is None or score < best[0]:
            best = (score, perm, errors)
    assert best is not None
    return truth, best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-dir", type=Path, required=True)
    parser.add_argument("--nod", type=Path, required=True)
    parser.add_argument("--gps-dir", type=Path, required=True)
    parser.add_argument("--nodes", nargs="+", required=True)
    parser.add_argument("--anchors", nargs="+", default=["5", "43", "47", "61"])
    parser.add_argument("--pattern", default="node{node}_k3_paper512x4_82.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=15)
    parser.add_argument("--end-frame", type=int, default=67)
    parser.add_argument("--start-time", type=float, default=132618.985)
    parser.add_argument("--fs", type=float, default=3050.0)
    parser.add_argument("--nfft", type=int, default=512)
    parser.add_argument("--fsnap", type=int, default=4)
    parser.add_argument("--position-bound-m", type=float, default=3000.0)
    parser.add_argument("--prefilter-count", type=int, default=32)
    parser.add_argument("--beam-size", type=int, default=12)
    parser.add_argument("--max-speed", type=float, default=120.0)
    parser.add_argument("--position-sigma-m", type=float, default=70.0)
    parser.add_argument("--acceleration-sigma-mps2", type=float, default=18.0)
    parser.add_argument("--temporal-weight", type=float, default=15.0)
    parser.add_argument("--confidence-floor", type=float, default=0.35)
    args = parser.parse_args()

    nod = load_nod(args.nod)
    origin = np.mean([[nod[node][key] for key in ("x", "y", "z")] for node in args.nodes], axis=0)
    positions = {
        node: np.asarray([nod[node][key] for key in ("x", "y", "z")], dtype=float) - origin
        for node in args.nodes
    }
    music = load_music(args.json_dir, args.nodes, args.pattern, nod, args.start_frame, args.end_frame)
    gps = [
        load_gps(args.gps_dir / "GPS1_plane1.gps"),
        load_gps(args.gps_dir / "GPS3_plane2.gps"),
        load_gps(args.gps_dir / "GPS4_plane2to3.gps"),
    ]
    frame_period = args.nfft * args.fsnap / args.fs
    frame_candidates = {}
    for frame in range(args.start_frame, args.end_frame + 1):
        measurements = {node: music[node][frame]["angles"] for node in args.nodes}
        strengths = {node: music[node][frame]["strengths"] for node in args.nodes}
        frame_candidates[frame] = enumerate_candidates(
            measurements,
            strengths,
            positions,
            args.anchors,
            args.position_bound_m,
            args.prefilter_count,
            args.confidence_floor,
        )
        if not frame_candidates[frame]:
            raise RuntimeError(f"no acoustic hypotheses for frame {frame}")

    # Beam state: cost, positions, velocity, path records.  Candidate source
    # labels are aligned to persistent track labels at every transition.
    first_frame = args.start_frame
    beam = []
    # Keep all prefiler candidates at the first frame.  A low measurement-cost
    # hypothesis can be a geometrically wrong branch, while a slightly higher
    # cost branch may be the only one that connects to the next frame.
    for candidate in frame_candidates[first_frame]:
        beam.append(
            {
                "cost": float(candidate["measurement_cost"]),
                "positions": [np.asarray(x) for x in candidate["positions"]],
                "velocity": np.zeros((3, 3), dtype=float),
                "path": [(candidate, (0, 1, 2), np.zeros(3))],
            }
        )
    dt = frame_period
    for frame in range(first_frame + 1, args.end_frame + 1):
        next_beam = []
        for previous in beam:
            predicted = [previous["positions"][target] + previous["velocity"][target] * dt for target in range(3)]
            for candidate in frame_candidates[frame]:
                aligned, permutation, distances = align_positions(candidate["positions"], predicted)
                velocity = np.asarray([(aligned[target] - previous["positions"][target]) / dt for target in range(3)])
                speed = np.linalg.norm(velocity, axis=1)
                if float(np.max(speed)) > args.max_speed:
                    continue
                acceleration = (velocity - previous["velocity"]) / dt
                temporal = float(
                    args.temporal_weight
                    * np.mean(np.square(distances / args.position_sigma_m))
                    + 0.5 * np.mean(np.square(acceleration / args.acceleration_sigma_mps2))
                )
                next_beam.append(
                    {
                        "cost": previous["cost"] + float(candidate["measurement_cost"]) + temporal,
                        "positions": aligned,
                        "velocity": velocity,
                        "path": previous["path"] + [(candidate, permutation, distances)],
                    }
                )
        if not next_beam:
            # If every acoustic hypothesis violates the physical speed gate,
            # carry the constant-velocity prediction forward without a
            # measurement update.  Accepting an arbitrary far-away hypothesis
            # would turn an association failure into a false 3-D jump.
            for previous in beam:
                predicted = [
                    previous["positions"][target] + previous["velocity"][target] * dt
                    for target in range(3)
                ]
                hold = {
                    "positions": [np.asarray(x, dtype=float) for x in predicted],
                    "measurement_cost": 0.0,
                    "anchor_nodes": [],
                    "condition_numbers": [],
                    "line_residual_m": [],
                    "node_assignments": {},
                    "held_prediction": True,
                }
                next_beam.append(
                    {
                        "cost": previous["cost"] + 0.25,
                        "positions": [np.asarray(x, dtype=float) for x in predicted],
                        "velocity": previous["velocity"].copy(),
                        "path": previous["path"] + [(hold, (0, 1, 2), np.zeros(3, dtype=float))],
                    }
                )
        next_beam.sort(key=lambda state: state["cost"])
        beam = next_beam[: args.beam_size]

    best = min(beam, key=lambda state: state["cost"])
    rows = []
    identity_rows = []
    for offset, (candidate, permutation, distances) in enumerate(best["path"]):
        frame = first_frame + offset
        held_prediction = bool(candidate.get("held_prediction", False))
        if offset == 0:
            aligned = [np.asarray(x) for x in candidate["positions"]]
            reordered = candidate["node_assignments"]
            velocity = np.zeros((3, 3), dtype=float)
        else:
            aligned = [np.asarray(candidate["positions"][permutation[target]]) for target in range(3)]
            reordered = reorder_assignments(candidate["node_assignments"], permutation)
            previous = rows[-1]
            velocity = np.asarray(
                [(aligned[target] - np.asarray(previous["positions"][target])) / dt for target in range(3)]
            )
        time_s = args.start_time + frame * dt
        truth, gps_score = score_gps(aligned, gps, origin, time_s)
        row = {
            "frame": frame,
            "time": time_s,
            "positions": [x.tolist() for x in aligned],
            "velocity_mps": velocity.tolist(),
            "speed_mps": np.linalg.norm(velocity, axis=1).tolist(),
            "measurement_cost": float(candidate["measurement_cost"]),
            "held_prediction": held_prediction,
            "anchor_nodes": candidate["anchor_nodes"],
            "condition_numbers": candidate["condition_numbers"],
            "line_residual_m": candidate["line_residual_m"],
            "node_assignments": reordered,
            "temporal_jump_m": [float(x) for x in distances],
            "offline_gps_assignment": list(gps_score[1]),
            "offline_gps_error_m": gps_score[2],
            "offline_gps_mean_error_m": gps_score[0],
        }
        rows.append(row)
        for node, assignment in reordered.items():
            for target in range(3):
                identity_rows.append(
                    {
                        "frame": frame,
                        "node": node,
                        "target": target + 1,
                        "candidate_index": assignment["perm"][target] + 1,
                        "confidence": assignment["confidence"][target],
                        "angular_error_deg": assignment["errors_deg"][target],
                    }
                )

    errors = np.asarray([row["offline_gps_mean_error_m"] for row in rows], dtype=float)
    jumps = np.asarray([distance for row in rows for distance in row["temporal_jump_m"]], dtype=float)
    payload = {
        "protocol": {
            "estimator": "GPS-free beam-selected raw 19-channel MUSIC hypotheses",
            "nodes": args.nodes,
            "anchors": args.anchors,
            "frame_period_s": frame_period,
            "gps_role": "offline scoring only",
            "temporal_prior": "constant velocity with bounded speed and acceleration penalty",
            "confidence_weight": "per-node/source geometric mean of azimuth/elevation MUSIC peak strengths, median-normalized",
            "parameters": {
                "prefilter_count": args.prefilter_count,
                "beam_size": args.beam_size,
                "max_speed_mps": args.max_speed,
                "position_sigma_m": args.position_sigma_m,
                "acceleration_sigma_mps2": args.acceleration_sigma_mps2,
                "temporal_weight": args.temporal_weight,
                "confidence_floor": args.confidence_floor,
            },
        },
        "summary": {
            "frames_requested": len(rows),
            "frames_solved": len(rows),
            "offline_gps_mean_error_m": float(np.mean(errors)),
            "offline_gps_median_error_m": float(np.median(errors)),
            "offline_gps_p90_error_m": float(np.quantile(errors, 0.90)),
            "offline_gps_within_100m_fraction": float(np.mean(errors <= 100.0)),
            "temporal_jump_p90_m": float(np.quantile(jumps, 0.90)),
            "temporal_jump_max_m": float(np.max(jumps)),
            "hold_fraction": float(np.mean([row["held_prediction"] for row in rows])),
            "condition_p90": float(
                np.quantile(
                    [max(row["condition_numbers"]) for row in rows if row["condition_numbers"]],
                    0.90,
                )
            ) if any(row["condition_numbers"] for row in rows) else None,
        },
        "rows": rows,
        "identity_diagnostics": identity_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=True))


if __name__ == "__main__":
    main()
