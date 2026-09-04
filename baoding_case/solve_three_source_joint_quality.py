"""GPS-free three-source association with joint azimuth/elevation pairing.

This is a diagnostic extension of the Baoding raw-MUSIC solver.  The released
front end extracts azimuth and zenith peaks independently; pairing equal
indices is therefore an additional assumption.  Here source assignments for
azimuth and zenith are searched independently, and the MUSIC peak-quality audit
supplies bounded angle uncertainties and node/source weights.  GPS is joined
only after the acoustic path has been selected for offline scoring.
"""

from __future__ import annotations

import argparse
import heapq
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
    unit_vector,
)
from summarize_three_source_raw_music_xyz import load_gps, load_nod


def circ_abs(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def huber(value: float, delta: float = 3.0) -> float:
    value = abs(float(value))
    return 0.5 * value * value if value <= delta else delta * (value - 0.5 * delta)


def quality_weight(q: dict, floor: float) -> float:
    """Map peak prominence and width to a bounded inverse variance weight."""
    prominence = max(float(q.get("prominence_db", 0.0)), 0.0)
    width = np.clip(float(q.get("width_deg", 45.0)), 1.0, 45.0)
    sigma = np.clip(width / max(1.0 + 0.35 * prominence, 1.0), 1.0, 45.0)
    return float(max(floor, min(4.0, 12.0 / sigma)))


def independent_assignment(
    predicted: list[tuple[float, float]],
    obs_az: list[float],
    obs_el: list[float],
    az_quality: list[dict],
    el_quality: list[dict],
    confidence_floor: float,
    pairing_penalty_deg: float,
) -> tuple[tuple[int, ...], tuple[int, ...], list[float], float]:
    """Match azimuth and zenith peaks independently to persistent sources."""
    best = None
    for az_perm in PERMS:
        for el_perm in PERMS:
            errors = []
            score = 0.0
            for target in range(3):
                ai = az_perm[target]
                ei = el_perm[target]
                az_error = circ_abs(predicted[target][0], obs_az[ai])
                el_error = abs(predicted[target][1] - obs_el[ei])
                wa = quality_weight(az_quality[ai], confidence_floor)
                we = quality_weight(el_quality[ei], confidence_floor)
                angular = math.hypot(az_error, el_error)
                errors.append(float(angular))
                score += huber(az_error) / wa + huber(el_error) / we
            # The released front end orders the two peak lists independently,
            # so equal-index pairing is a useful weak prior.  A swap is still
            # accepted when it improves the angular fit by more than this
            # penalty; otherwise low-SNR flat peaks can create arbitrary
            # azimuth/zenith combinations.
            score += pairing_penalty_deg * sum(az_perm[t] != el_perm[t] for t in range(3))
            score /= 3.0
            if best is None or score < best[0]:
                best = (score, tuple(az_perm), tuple(el_perm), errors)
    assert best is not None
    return best[1], best[2], best[3], float(best[0])


def same_index_score(
    predicted: list[tuple[float, float]],
    obs_az: list[float],
    obs_el: list[float],
) -> float:
    """Cheap prefilter score used before independent pairing is evaluated."""
    best = float("inf")
    for perm in PERMS:
        value = float(np.mean([
            math.hypot(circ_abs(predicted[t][0], obs_az[perm[t]]), abs(predicted[t][1] - obs_el[perm[t]]))
            for t in range(3)
        ]))
        best = min(best, value)
    return best


def confidence_for_assignment(
    az_quality: list[dict], el_quality: list[dict], assignment: dict, floor: float
) -> list[float]:
    result = []
    for target in range(3):
        ai = int(assignment["az_perm"][target])
        ei = int(assignment["el_perm"][target])
        result.append(float(max(floor, math.sqrt(quality_weight(az_quality[ai], floor) * quality_weight(el_quality[ei], floor)))))
    return result


def assignment_record(
    predicted: list[tuple[float, float]],
    obs_az: list[float],
    obs_el: list[float],
    az_quality: list[dict],
    el_quality: list[dict],
    floor: float,
    pairing_penalty_deg: float,
) -> tuple[dict, float]:
    az_perm, el_perm, errors, score = independent_assignment(
        predicted, obs_az, obs_el, az_quality, el_quality, floor, pairing_penalty_deg
    )
    record = {
        "az_perm": list(az_perm),
        "el_perm": list(el_perm),
        # ``perm`` is retained as the azimuth index for older readers.
        "perm": list(az_perm),
        "score_deg": float(np.mean(errors)),
        "errors_deg": [float(x) for x in errors],
        "robust_weight": [float(max(0.03, min(1.0, (25.0 / max(float(error), 25.0)) ** 2))) for error in errors],
        "confidence": confidence_for_assignment(
            az_quality, el_quality, {"az_perm": az_perm, "el_perm": el_perm}, floor
        ),
        "assigned_azimuth_peak_strength": [float(az_quality[i].get("peak_strength", 0.0)) for i in az_perm],
        "assigned_elevation_peak_strength": [float(el_quality[i].get("peak_strength", 0.0)) for i in el_perm],
        "azimuth_sigma_deg": [float(az_quality[i].get("sigma_deg", 45.0)) for i in az_perm],
        "zenith_sigma_deg": [float(el_quality[i].get("sigma_deg", 45.0)) for i in el_perm],
    }
    return record, score


def refine_positions(
    initial: list[np.ndarray],
    measurements: dict[str, dict],
    positions: dict[str, np.ndarray],
    assignments: dict[str, dict],
    bound: float,
    floor: float,
) -> list[np.ndarray]:
    refined = []
    for target, x0 in enumerate(initial):
        observations = []
        for node, measurement in measurements.items():
            a = assignments.get(node)
            if not a:
                continue
            ai = int(a["az_perm"][target])
            ei = int(a["el_perm"][target])
            direction = unit_vector(measurement["az"][ai], measurement["el"][ei])
            sigma = math.hypot(float(a.get("azimuth_sigma_deg", [30.0] * 3)[target]), float(a.get("zenith_sigma_deg", [30.0] * 3)[target]))
            robust_weight = float(a.get("robust_weight", [1.0] * 3)[target])
            observations.append((positions[node], direction, max(sigma, 1.0), math.sqrt(max(robust_weight, 0.03))))
        if len(observations) < 3:
            refined.append(np.asarray(x0, dtype=float))
            continue

        def residual(x: np.ndarray) -> np.ndarray:
            values = []
            for node_position, observed_direction, sigma, robust_scale in observations:
                delta = x - node_position
                norm = max(float(np.linalg.norm(delta)), 1.0)
                predicted_direction = delta / norm
                values.extend((robust_scale * (predicted_direction - observed_direction) / (sigma / 25.0)).tolist())
            return np.asarray(values, dtype=float)

        fit = least_squares(
            residual,
            np.asarray(x0, dtype=float),
            loss="soft_l1",
            f_scale=0.6,
            max_nfev=100,
            bounds=(-bound, bound),
        )
        refined.append(fit.x)
    return refined


def load_quality(path: Path, nodes: list[str], start: int, end: int) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for node in nodes:
        rows = {int(r["frame"]): r for r in payload["nodes"][node]["rows"]}
        result[node] = {}
        for frame in range(start, end + 1):
            source_rows = rows[frame]["sources"]
            # The audit stores one quality row per raw peak index.  Azimuth and
            # zenith quality are separated before joint matching.
            az = []
            el = []
            for source in source_rows:
                fq = source["full"]
                az.append({
                    "prominence_db": fq["azimuth"]["prominence_db"],
                    "width_deg": fq["azimuth"]["width_deg"],
                    "curvature": fq["azimuth"]["curvature"],
                    "peak_strength": fq["azimuth"].get("local_snr", 0.0),
                    "sigma_deg": source["azimuth_sigma_deg"],
                })
                el.append({
                    "prominence_db": fq["zenith"]["prominence_db"],
                    "width_deg": fq["zenith"]["width_deg"],
                    "curvature": fq["zenith"]["curvature"],
                    "peak_strength": fq["zenith"].get("local_snr", 0.0),
                    "sigma_deg": source["zenith_sigma_deg"],
                })
            result[node][frame] = {"az": az, "el": el}
    return result


def enumerate_candidates(
    measurements: dict[str, dict],
    positions: dict[str, np.ndarray],
    anchors: list[str],
    bound: float,
    prefilter_count: int,
    floor: float,
    pairing_penalty_deg: float,
    elevation_perms: list[tuple[int, ...]],
) -> list[dict]:
    """Search azimuth permutations plus independent zenith permutations.

    The base anchor keeps its azimuth and zenith order fixed to remove global
    source-label symmetry.  The other two anchors sweep the configured
    elevation pairing pool.  The default pool is identity plus the three
    single transpositions, which captures the likely peak-order ambiguity while
    keeping the complete 35-second audit tractable; ``--full-pairing`` restores
    all six permutations.
    """
    available = [n for n in anchors if n in measurements]
    if len(available) < 3:
        return []
    ranked: list[tuple[float, dict]] = []
    for anchor_nodes in itertools.combinations(available, 3):
        base, node_b, node_c = anchor_nodes
        for az_b in PERMS:
            for el_b in elevation_perms:
                for az_c in PERMS:
                    for el_c in elevation_perms:
                        az_perms = {base: (0, 1, 2), node_b: az_b, node_c: az_c}
                        el_perms = {base: (0, 1, 2), node_b: el_b, node_c: el_c}
                        estimates, conditions, residuals = [], [], []
                        for target in range(3):
                            dirs = [
                                unit_vector(
                                    measurements[node]["az"][az_perms[node][target]],
                                    measurements[node]["el"][el_perms[node][target]],
                                )
                                for node in anchor_nodes
                            ]
                            estimate, condition, residual = line_intersection(
                                [positions[node] for node in anchor_nodes], dirs
                            )
                            estimates.append(estimate)
                            conditions.append(condition)
                            residuals.append(residual)
                        if max(conditions) > 1e6 or max(np.linalg.norm(x) for x in estimates) > bound:
                            continue
                        backward = 0
                        for node in anchor_nodes:
                            for target in range(3):
                                d = estimates[target] - positions[node]
                                u = unit_vector(
                                    measurements[node]["az"][az_perms[node][target]],
                                    measurements[node]["el"][el_perms[node][target]],
                                )
                                backward += int(float(np.dot(d, u)) < 0.0)
                        # The prefilter only needs a cheap geometry score.  A
                        # previous version evaluated 36 full-node assignments
                        # for every hypothesis, multiplying runtime without
                        # changing the retained refined candidates.  Full
                        # joint pairing is evaluated below after prefiltering.
                        anchor_angle = 0.0
                        for node in anchor_nodes:
                            pred = [predicted_angles(x, positions[node]) for x in estimates]
                            ap = az_perms[node]
                            ep = el_perms[node]
                            anchor_angle += float(np.mean([
                                math.hypot(circ_abs(pred[t][0], measurements[node]["az"][ap[t]]), abs(pred[t][1] - measurements[node]["el"][ep[t]]))
                                for t in range(3)
                            ]))
                        score = 0.5 * float(np.mean(residuals)) + 0.02 * float(max(conditions)) + 0.25 * anchor_angle + 25.0 * backward
                        record = {
                            "prefilter_score": float(score),
                            "positions": estimates,
                            "anchor_nodes": list(anchor_nodes),
                            "condition_numbers": [float(x) for x in conditions],
                            "line_residual_m": [float(x) for x in residuals],
                            "anchor_az_perm": {k: list(v) for k, v in az_perms.items()},
                            "anchor_el_perm": {k: list(v) for k, v in el_perms.items()},
                        }
                        ranked.append((float(score), record))
    ranked.sort(key=lambda x: x[0])
    refined = []
    for _, record in ranked[:prefilter_count]:
        estimates = [np.asarray(x, dtype=float) for x in record["positions"]]
        assignments = {}
        for node, measurement in measurements.items():
            pred = [predicted_angles(x, positions[node]) for x in estimates]
            a, _ = assignment_record(
                pred, measurement["az"], measurement["el"], measurement["az_quality"], measurement["el_quality"], floor, pairing_penalty_deg
            )
            assignments[node] = a
        estimates = refine_positions(estimates, measurements, positions, assignments, bound, floor)
        for _ in range(2):
            for node, measurement in measurements.items():
                pred = [predicted_angles(x, positions[node]) for x in estimates]
                assignments[node], _ = assignment_record(
                    pred, measurement["az"], measurement["el"], measurement["az_quality"], measurement["el_quality"], floor, pairing_penalty_deg
                )
            estimates = refine_positions(estimates, measurements, positions, assignments, bound, floor)
        cost = 0.0
        for node, measurement in measurements.items():
            pred = [predicted_angles(x, positions[node]) for x in estimates]
            assignments[node], node_cost = assignment_record(
                pred, measurement["az"], measurement["el"], measurement["az_quality"], measurement["el_quality"], floor, pairing_penalty_deg
            )
            cost += node_cost
        item = dict(record)
        item["positions"] = estimates
        item["node_assignments"] = assignments
        item["measurement_cost"] = float(cost / max(len(measurements), 1) + np.mean(record["line_residual_m"]) / 50.0)
        refined.append(item)
    refined.sort(key=lambda x: x["measurement_cost"])
    return refined


def align_positions(candidate: list[np.ndarray], reference: list[np.ndarray]):
    best = None
    for perm in PERMS:
        aligned = [np.asarray(candidate[perm[t]], dtype=float) for t in range(3)]
        distances = np.asarray([np.linalg.norm(aligned[t] - reference[t]) for t in range(3)])
        score = float(np.mean(distances))
        if best is None or score < best[0]:
            best = (score, aligned, perm, distances)
    assert best is not None
    return best[1], tuple(best[2]), best[3]


def reorder_assignments(assignments: dict[str, dict], permutation: tuple[int, ...]) -> dict[str, dict]:
    out = {}
    for node, row in assignments.items():
        new = dict(row)
        for key in ("perm", "az_perm", "el_perm", "errors_deg", "robust_weight", "confidence", "assigned_azimuth_peak_strength", "assigned_elevation_peak_strength", "azimuth_sigma_deg", "zenith_sigma_deg"):
            if key in row:
                new[key] = [row[key][permutation[t]] for t in range(3)]
        out[node] = new
    return out


def nearest_xyz(gps, time_s: float) -> np.ndarray:
    xyz, times = gps
    return xyz[int(np.argmin(np.abs(times - time_s)))]


def score_gps(positions, gps, origin, time_s):
    truth = [nearest_xyz(g, time_s) - origin for g in gps]
    best = None
    for perm in PERMS:
        errors = [float(np.linalg.norm(positions[t] - truth[perm[t]])) for t in range(3)]
        score = float(np.mean(errors))
        if best is None or score < best[0]:
            best = (score, perm, errors)
    assert best is not None
    return best


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--json-dir", type=Path, required=True)
    p.add_argument("--quality-json", type=Path, required=True)
    p.add_argument("--nod", type=Path, required=True)
    p.add_argument("--gps-dir", type=Path, required=True)
    p.add_argument("--nodes", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pattern", default="node{node}_k3_paper512x4_82.json")
    p.add_argument("--start-frame", type=int, default=15)
    p.add_argument("--end-frame", type=int, default=67)
    p.add_argument("--start-time", type=float, default=132618.985)
    p.add_argument("--fs", type=float, default=3050.0)
    p.add_argument("--nfft", type=int, default=512)
    p.add_argument("--fsnap", type=int, default=4)
    p.add_argument("--position-bound-m", type=float, default=3000.0)
    p.add_argument("--prefilter-count", type=int, default=32)
    p.add_argument("--beam-size", type=int, default=12)
    p.add_argument("--max-speed", type=float, default=500.0)
    p.add_argument("--position-sigma-m", type=float, default=150.0)
    p.add_argument("--acceleration-sigma-mps2", type=float, default=45.0)
    p.add_argument("--temporal-weight", type=float, default=15.0)
    p.add_argument("--confidence-floor", type=float, default=0.35)
    p.add_argument("--pairing-penalty-deg", type=float, default=4.0)
    p.add_argument("--full-pairing", action="store_true", help="sweep all six elevation permutations per anchor")
    args = p.parse_args()
    nod = load_nod(args.nod)
    origin = np.mean([[nod[n][k] for k in ("x", "y", "z")] for n in args.nodes], axis=0)
    positions = {n: np.asarray([nod[n][k] for k in ("x", "y", "z")], dtype=float) - origin for n in args.nodes}
    quality = load_quality(args.quality_json, args.nodes, args.start_frame, args.end_frame)
    music = {}
    for node in args.nodes:
        payload = json.loads((args.json_dir / args.pattern.format(node=node)).read_text(encoding="utf-8"))
        music[node] = {int(row["frame"]): row for row in payload["frames"]}
    frame_period = args.nfft * args.fsnap / args.fs
    elevation_perms = list(PERMS) if args.full_pairing else [(0, 1, 2), (1, 0, 2), (0, 2, 1), (0, 1, 2)]
    # Remove the duplicate identity while preserving deterministic order.
    elevation_perms = list(dict.fromkeys(elevation_perms))
    candidates = {}
    for frame in range(args.start_frame, args.end_frame + 1):
        measurements = {}
        for node in args.nodes:
            item = music[node][frame]
            az, el = corrected_angles(item, nod[node])
            q = quality[node][frame]
            measurements[node] = {"az": az, "el": el, "az_quality": q["az"], "el_quality": q["el"]}
        candidates[frame] = enumerate_candidates(measurements, positions, args.nodes, args.position_bound_m, args.prefilter_count, args.confidence_floor, args.pairing_penalty_deg, elevation_perms)
        if not candidates[frame]:
            raise RuntimeError(f"no joint hypotheses for frame {frame}")
        print(json.dumps({"frame": frame, "candidates": len(candidates[frame])}, ensure_ascii=True), flush=True)

    first = args.start_frame
    beam = []
    for candidate in candidates[first]:
        beam.append({"cost": float(candidate["measurement_cost"]), "positions": [np.asarray(x) for x in candidate["positions"]], "velocity": np.zeros((3, 3)), "path": [(candidate, (0, 1, 2), np.zeros(3))]})
    dt = frame_period
    for frame in range(first + 1, args.end_frame + 1):
        next_beam = []
        for previous in beam:
            predicted = [previous["positions"][t] + previous["velocity"][t] * dt for t in range(3)]
            for candidate in candidates[frame]:
                aligned, perm, distances = align_positions(candidate["positions"], predicted)
                velocity = np.asarray([(aligned[t] - previous["positions"][t]) / dt for t in range(3)])
                if float(np.max(np.linalg.norm(velocity, axis=1))) > args.max_speed:
                    continue
                acceleration = (velocity - previous["velocity"]) / dt
                temporal = args.temporal_weight * float(np.mean(np.square(distances / args.position_sigma_m))) + 0.5 * float(np.mean(np.square(acceleration / args.acceleration_sigma_mps2)))
                next_beam.append({"cost": previous["cost"] + float(candidate["measurement_cost"]) + temporal, "positions": aligned, "velocity": velocity, "path": previous["path"] + [(candidate, perm, distances)]})
        if not next_beam:
            for previous in beam:
                predicted = [previous["positions"][t] + previous["velocity"][t] * dt for t in range(3)]
                hold = {"positions": predicted, "measurement_cost": 0.0, "anchor_nodes": [], "condition_numbers": [], "line_residual_m": [], "node_assignments": {}, "held_prediction": True}
                next_beam.append({"cost": previous["cost"] + 0.25, "positions": predicted, "velocity": previous["velocity"].copy(), "path": previous["path"] + [(hold, (0, 1, 2), np.zeros(3))]})
        next_beam.sort(key=lambda x: x["cost"])
        beam = next_beam[: args.beam_size]

    gps = [load_gps(args.gps_dir / name) for name in ("GPS1_plane1.gps", "GPS3_plane2.gps", "GPS4_plane2to3.gps")]
    best = min(beam, key=lambda x: x["cost"])
    rows = []
    for offset, (candidate, permutation, distances) in enumerate(best["path"]):
        frame = first + offset
        if offset == 0:
            aligned = [np.asarray(x) for x in candidate["positions"]]
            assignments = candidate["node_assignments"]
            velocity = np.zeros((3, 3))
        else:
            aligned = [np.asarray(candidate["positions"][permutation[t]]) for t in range(3)]
            assignments = reorder_assignments(candidate["node_assignments"], permutation)
            velocity = np.asarray([(aligned[t] - np.asarray(rows[-1]["positions"][t])) / dt for t in range(3)])
        gps_score = score_gps(aligned, gps, origin, args.start_time + frame * dt)
        rows.append({"frame": frame, "time": args.start_time + frame * dt, "positions": [x.tolist() for x in aligned], "velocity_mps": velocity.tolist(), "speed_mps": np.linalg.norm(velocity, axis=1).tolist(), "measurement_cost": float(candidate.get("measurement_cost", 0.0)), "held_prediction": bool(candidate.get("held_prediction", False)), "anchor_nodes": candidate.get("anchor_nodes", []), "condition_numbers": candidate.get("condition_numbers", []), "line_residual_m": candidate.get("line_residual_m", []), "node_assignments": assignments, "temporal_jump_m": [float(x) for x in distances], "offline_gps_assignment": list(gps_score[1]), "offline_gps_error_m": gps_score[2], "offline_gps_mean_error_m": gps_score[0]})
    errors = np.asarray([r["offline_gps_mean_error_m"] for r in rows])
    jumps = np.asarray([x for r in rows for x in r["temporal_jump_m"]])
    payload = {"protocol": {"estimator": "GPS-free joint azimuth/zenith pairing with MUSIC spectral-quality weighting", "nodes": args.nodes, "frame_period_s": dt, "gps_role": "offline scoring only", "quality_input": str(args.quality_json), "pairing": "independent azimuth and zenith permutations with equal-index weak prior; base-anchor order fixed for label symmetry", "elevation_permutation_pool": [list(x) for x in elevation_perms], "parameters": {"prefilter_count": args.prefilter_count, "beam_size": args.beam_size, "max_speed_mps": args.max_speed, "position_sigma_m": args.position_sigma_m, "acceleration_sigma_mps2": args.acceleration_sigma_mps2, "temporal_weight": args.temporal_weight, "confidence_floor": args.confidence_floor, "pairing_penalty_deg": args.pairing_penalty_deg}}, "summary": {"frames_requested": len(rows), "frames_solved": len(rows), "offline_gps_mean_error_m": float(np.mean(errors)), "offline_gps_median_error_m": float(np.median(errors)), "offline_gps_p90_error_m": float(np.quantile(errors, 0.90)), "offline_gps_within_100m_fraction": float(np.mean(errors <= 100.0)), "temporal_jump_p90_m": float(np.quantile(jumps, 0.90)), "temporal_jump_max_m": float(np.max(jumps)), "hold_fraction": float(np.mean([r["held_prediction"] for r in rows])), "condition_p90": float(np.quantile([max(r["condition_numbers"]) for r in rows if r["condition_numbers"]], 0.90))}, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=True))


if __name__ == "__main__":
    main()
