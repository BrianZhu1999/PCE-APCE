"""GPS-free three-source bearing triangulation from raw MUSIC outputs.

The estimator uses only node geometry and acoustic azimuth/elevation peaks.
GPS, when supplied, is used after estimation for an offline diagnostic score.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from summarize_three_source_raw_music_xyz import gps_bearings, load_gps, load_nod, transform


PERMS = list(itertools.permutations(range(3)))


def circ_abs(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def unit_vector(az_deg: float, el_from_vertical_deg: float) -> np.ndarray:
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_from_vertical_deg)
    return np.asarray([
        np.sin(el) * np.cos(az),
        np.sin(el) * np.sin(az),
        np.cos(el),
    ], dtype=float)


def corrected_angles(item: dict, correction: dict[str, float]) -> tuple[list[float], list[float]]:
    az = transform(item["azimuth_deg"], correction["h_offset"], correction["h_direction"])
    # Elevation is an angle from the positive vertical axis. Keep it in degrees
    # instead of applying a circular wrap, which would hide a sign error.
    sign = -1.0 if int(round(correction["v_direction"])) == 1 else 1.0
    el = [float(sign * (x + correction["v_offset"])) for x in item["elevation_deg"]]
    return az, el


def line_intersection(points: list[np.ndarray], directions: list[np.ndarray]) -> tuple[np.ndarray, float, float]:
    a = np.zeros((3, 3), dtype=float)
    b = np.zeros(3, dtype=float)
    for p, u in zip(points, directions):
        q = np.eye(3) - np.outer(u, u)
        a += q
        b += q @ p
    x = np.linalg.lstsq(a, b, rcond=None)[0]
    cond = float(np.linalg.cond(a))
    residual = float(np.mean([np.linalg.norm((np.eye(3) - np.outer(u, u)) @ (x - p)) for p, u in zip(points, directions)]))
    return x, cond, residual


def predicted_angles(x: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    d = x - p
    return float(np.degrees(np.arctan2(d[1], d[0])) % 360.0), float(np.degrees(np.arctan2(np.hypot(d[0], d[1]), d[2])))


def assignment_cost(pred: list[tuple[float, float]], obs_az: list[float], obs_el: list[float]) -> tuple[float, tuple[int, ...], list[float]]:
    best = None
    for perm in PERMS:
        errors = []
        for i in range(3):
            az_err = circ_abs(pred[i][0], obs_az[perm[i]])
            el_err = abs(pred[i][1] - obs_el[perm[i]])
            errors.append(float(np.hypot(az_err, el_err)))
        score = float(np.mean(errors))
        if best is None or score < best[0]:
            best = (score, perm, errors)
    assert best is not None
    return best


def refine_positions(
    initial: list[np.ndarray],
    measurements: dict[str, tuple[list[float], list[float]]],
    positions: dict[str, np.ndarray],
    assignments: dict[str, dict],
    position_bound_m: float,
) -> list[np.ndarray]:
    """Robustly refine each source using all assigned node bearings."""
    refined = []
    for source, x0 in enumerate(initial):
        observations = []
        for node, (az, el) in measurements.items():
            if node not in assignments:
                continue
            obs_idx = int(assignments[node]["perm"][source])
            observations.append((positions[node], unit_vector(az[obs_idx], el[obs_idx])))
        if len(observations) < 3:
            refined.append(x0)
            continue

        def residual(x: np.ndarray) -> np.ndarray:
            values = []
            for p, u_obs in observations:
                d = x - p
                norm = max(float(np.linalg.norm(d)), 1.0)
                u_pred = d / norm
                values.extend((u_pred - u_obs).tolist())
            return np.asarray(values, dtype=float)

        fit = least_squares(
            residual,
            np.asarray(x0, dtype=float),
            loss="soft_l1",
            f_scale=0.04,
            max_nfev=80,
            bounds=(-position_bound_m, position_bound_m),
        )
        refined.append(fit.x)
    return refined


def solve_frame(
    measurements: dict[str, tuple[list[float], list[float]]],
    positions: dict[str, np.ndarray],
    anchors: list[str],
    position_bound_m: float,
) -> dict:
    available = [n for n in anchors if n in measurements]
    if len(available) < 3:
        return {"status": "insufficient_anchor_nodes", "anchors_available": available}
    best = None
    # Use all 3-node subsets of the four fixed acoustic anchors. This keeps the
    # search bounded while allowing one anchor to be rejected per frame.
    for anchor_nodes in itertools.combinations(available, 3):
        base = anchor_nodes[0]
        for perm_b in PERMS:
            for perm_c in PERMS:
                perms = {base: (0, 1, 2), anchor_nodes[1]: perm_b, anchor_nodes[2]: perm_c}
                estimates = []
                conditions = []
                residuals = []
                for source in range(3):
                    pts = [positions[n] for n in anchor_nodes]
                    dirs = [unit_vector(measurements[n][0][perms[n][source]], measurements[n][1][perms[n][source]]) for n in anchor_nodes]
                    x, cond, res = line_intersection(pts, dirs)
                    estimates.append(x)
                    conditions.append(cond)
                    residuals.append(res)
                if max(conditions) > 1e6:
                    continue
                score = 0.5 * float(np.mean(residuals))
                node_assignments = {}
                for node, (az, el) in measurements.items():
                    pred = [predicted_angles(x, positions[node]) for x in estimates]
                    node_score, perm, errors = assignment_cost(pred, az, el)
                    score += node_score
                    node_assignments[node] = {"perm": list(perm), "score_deg": node_score, "errors_deg": errors}
                # Penalize backwards intersections; a physical source should be
                # in front of all anchor arrays along the measured ray.
                for node in anchor_nodes:
                    for x, source in zip(estimates, range(3)):
                        d = x - positions[node]
                        u = unit_vector(measurements[node][0][perms[node][source]], measurements[node][1][perms[node][source]])
                        if float(np.dot(d, u)) < 0:
                            score += 25.0
                if best is None or score < best[0]:
                    best = (score, estimates, anchor_nodes, conditions, residuals, node_assignments)
    if best is None:
        return {"status": "no_geometrically_valid_hypothesis"}
    score, estimates, anchor_nodes, conditions, residuals, node_assignments = best
    # Refit each source with all node bearings, then update associations once.
    estimates = refine_positions(estimates, measurements, positions, node_assignments, position_bound_m=position_bound_m)
    refined_assignments = {}
    refined_score = 0.0
    for node, (az, el) in measurements.items():
        pred = [predicted_angles(x, positions[node]) for x in estimates]
        node_score, perm, errors = assignment_cost(pred, az, el)
        refined_score += node_score
        refined_assignments[node] = {"perm": list(perm), "score_deg": node_score, "errors_deg": errors}
    # A second short refit prevents a single swapped node from dominating the
    # final position while preserving the bounded, auditable search.
    estimates = refine_positions(estimates, measurements, positions, refined_assignments, position_bound_m=position_bound_m)
    refined_assignments = {}
    refined_score = 0.0
    for node, (az, el) in measurements.items():
        pred = [predicted_angles(x, positions[node]) for x in estimates]
        node_score, perm, errors = assignment_cost(pred, az, el)
        refined_score += node_score
        refined_assignments[node] = {"perm": list(perm), "score_deg": node_score, "errors_deg": errors}
    node_assignments = refined_assignments
    return {
        "status": "ok",
        "score": float(refined_score),
        "positions": [x.tolist() for x in estimates],
        "anchor_nodes": list(anchor_nodes),
        "condition_numbers": conditions,
        "line_residual_m": residuals,
        "node_assignments": node_assignments,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--json-dir", type=Path, required=True)
    p.add_argument("--nod", type=Path, required=True)
    p.add_argument("--gps-dir", type=Path, required=True)
    p.add_argument("--nodes", nargs="+", required=True)
    p.add_argument("--anchors", nargs="+", default=["5", "43", "47", "61"])
    p.add_argument("--pattern", default="node{node}_k3_paper512x4_82.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--end-frame", type=int, default=81)
    p.add_argument("--start-time", type=float, default=132618.985)
    p.add_argument("--fs", type=float, default=3050.0)
    p.add_argument("--nfft", type=int, default=512)
    p.add_argument("--fsnap", type=int, default=4)
    p.add_argument("--position-bound-m", type=float, default=3000.0)
    args = p.parse_args()

    nod = load_nod(args.nod)
    origin = np.mean([[nod[n]["x"], nod[n]["y"], nod[n]["z"]] for n in args.nodes], axis=0)
    positions = {n: np.asarray([nod[n]["x"], nod[n]["y"], nod[n]["z"]], dtype=float) - origin for n in args.nodes}
    music = {}
    for n in args.nodes:
        path = args.json_dir / args.pattern.format(node=n)
        music[n] = {int(x["frame"]): x for x in json.loads(path.read_text(encoding="utf-8"))["frames"]}

    gps = [
        load_gps(args.gps_dir / "GPS1_plane1.gps"),
        load_gps(args.gps_dir / "GPS3_plane2.gps"),
        load_gps(args.gps_dir / "GPS4_plane2to3.gps"),
    ]
    rows = []
    for frame in range(args.start_frame, args.end_frame + 1):
        measurements = {}
        for n in args.nodes:
            if frame in music[n]:
                az, el = corrected_angles(music[n][frame], nod[n])
                measurements[n] = (az, el)
        row = solve_frame(measurements, positions, args.anchors, position_bound_m=args.position_bound_m)
        row["frame"] = frame
        row["time"] = args.start_time + frame * args.nfft * args.fsnap / args.fs
        if row.get("status") == "ok":
            # Restore the original coordinate origin for reader-facing output.
            row["positions_absolute"] = [[float(x + origin[i]) for i, x in enumerate(pos)] for pos in row["positions"]]
            true_positions = [
                [float(nearest[0]) for nearest in []]
            ]
            # GPS is intentionally not used above; this block is offline scoring.
            truth = []
            for xyz, times in gps:
                idx = int(np.argmin(np.abs(times - row["time"])))
                truth.append((xyz[idx] - origin).tolist())
            best_truth = None
            est = [np.asarray(x) for x in row["positions"]]
            for perm in PERMS:
                errors_3d = [float(np.linalg.norm(est[i] - np.asarray(truth[perm[i]]))) for i in range(3)]
                score = float(np.mean(errors_3d))
                if best_truth is None or score < best_truth[0]:
                    best_truth = (score, perm, errors_3d)
            row["offline_gps_assignment"] = list(best_truth[1])
            row["offline_gps_error_m"] = best_truth[2]
            row["offline_gps_mean_error_m"] = best_truth[0]
        rows.append(row)

    ok = [r for r in rows if r.get("status") == "ok"]
    errors = [r["offline_gps_mean_error_m"] for r in ok if "offline_gps_mean_error_m" in r]
    payload = {
        "protocol": {
            "estimator": "GPS-free line-intersection association over raw 19-channel MUSIC peaks",
            "nodes": args.nodes, "anchors": args.anchors,
            "pattern": args.pattern, "frame_period_s": args.nfft * args.fsnap / args.fs,
            "gps_role": "offline scoring only",
        },
        "summary": {
            "frames_requested": args.end_frame - args.start_frame + 1,
            "frames_solved": len(ok),
            "solve_fraction": float(len(ok) / max(len(rows), 1)),
            "offline_gps_mean_error_m": float(np.mean(errors)) if errors else None,
            "offline_gps_median_error_m": float(np.median(errors)) if errors else None,
            "offline_gps_p90_error_m": float(np.quantile(errors, 0.90)) if errors else None,
            "offline_gps_within_100m_fraction": float(np.mean(np.asarray(errors) <= 100.0)) if errors else None,
        },
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=True))


if __name__ == "__main__":
    main()
