"""Score raw three-source MUSIC peaks against GPS XYZ-derived bearings.

The bearing convention follows the supplied ``gpstodoa.m``: azimuth is
``atan2(dy, dx)`` and the elevation variable is the angle from the positive
vertical axis, ``atan2(horizontal, dz)``. GPS is read only for offline scoring.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def circ_abs(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def best_perm(est: list[float], truth: list[float]) -> tuple[float, tuple[int, ...], list[float]]:
    best = None
    for perm in itertools.permutations(range(3)):
        errors = [circ_abs(est[perm[i]], truth[i]) for i in range(3)]
        score = float(np.mean(errors))
        if best is None or score < best[0]:
            best = (score, perm, errors)
    assert best is not None
    return best


def load_nod(path: Path) -> dict[str, dict[str, float]]:
    result = {}
    for line in path.read_text(errors="replace").splitlines():
        f = line.split()
        if len(f) < 11:
            continue
        result[f[0].rsplit(".", 1)[-1]] = {
            "x": float(f[3]), "y": float(f[4]), "z": float(f[5]),
            "h_offset": float(f[7]), "v_offset": float(f[8]),
            "h_direction": float(f[9]), "v_direction": float(f[10]),
        }
    return result


def load_gps(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr[:, 4:7], arr[:, 7]


def nearest(values: np.ndarray, times: np.ndarray, t: float) -> np.ndarray:
    return values[int(np.argmin(np.abs(times - t)))]


def gps_bearings(gps: list[tuple[np.ndarray, np.ndarray]], node: dict[str, float], t: float) -> tuple[list[float], list[float]]:
    azimuth, elevation = [], []
    for xyz, times in gps:
        p = nearest(xyz, times, t)
        dx, dy, dz = p - np.asarray([node["x"], node["y"], node["z"]])
        azimuth.append(float(np.degrees(np.arctan2(dy, dx)) % 360.0))
        elevation.append(float(np.degrees(np.arctan2(np.hypot(dx, dy), dz))))
    return azimuth, elevation


def transform(values: list[float], offset: float, direction: float) -> list[float]:
    sign = -1.0 if int(round(direction)) == 1 else 1.0
    return [float((sign * (v + offset)) % 360.0) for v in values]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--json-dir", type=Path, required=True)
    p.add_argument("--nod", type=Path, required=True)
    p.add_argument("--gps-dir", type=Path, required=True)
    p.add_argument("--nodes", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--start-time", type=float, default=132618.985)
    p.add_argument("--frame-period-s", type=float, default=640.0 / 3050.0)
    args = p.parse_args()

    nod = load_nod(args.nod)
    gps = [
        load_gps(args.gps_dir / "GPS1_plane1.gps"),
        load_gps(args.gps_dir / "GPS3_plane2.gps"),
        load_gps(args.gps_dir / "GPS4_plane2to3.gps"),
    ]
    summaries = {}
    pooled_az, pooled_el = [], []
    for node_id in args.nodes:
        result_path = args.json_dir / f"node{node_id}_k3_frames20.json"
        if not result_path.exists():
            summaries[node_id] = {"status": "missing_music_result"}
            continue
        if node_id not in nod:
            summaries[node_id] = {"status": "missing_node_geometry"}
            continue
        correction = nod[node_id]
        rows = []
        music = json.loads(result_path.read_text(encoding="utf-8"))
        for frame in music["frames"]:
            t = args.start_time + frame["frame"] * args.frame_period_s
            true_az, true_el = gps_bearings(gps, correction, t)
            est_az = transform(frame["azimuth_deg"], correction["h_offset"], correction["h_direction"])
            est_el = transform(frame["elevation_deg"], correction["v_offset"], correction["v_direction"])
            az_mean, perm, az_errors = best_perm(est_az, true_az)
            # Keep the azimuth source identity for the elevation residual.
            el_errors = [abs(est_el[perm[i]] - true_el[i]) for i in range(3)]
            row = {
                "frame": int(frame["frame"]), "time": t,
                "estimated_azimuth_deg": est_az, "gps_azimuth_deg": true_az,
                "estimated_elevation_deg": est_el, "gps_elevation_deg": true_el,
                "assignment": list(perm), "azimuth_abs_error_deg": az_errors,
                "elevation_abs_error_deg": el_errors,
                "azimuth_mean_error_deg": az_mean,
                "elevation_mean_error_deg": float(np.mean(el_errors)),
            }
            rows.append(row)
            pooled_az.extend(az_errors)
            pooled_el.extend(el_errors)
        az = np.asarray([r["azimuth_mean_error_deg"] for r in rows])
        el = np.asarray([r["elevation_mean_error_deg"] for r in rows])
        summaries[node_id] = {
            "status": "ok", "frames": len(rows),
            "azimuth_mean_deg": float(np.mean(az)),
            "azimuth_median_deg": float(np.median(az)),
            "azimuth_p90_deg": float(np.quantile(az, 0.90)),
            "azimuth_within_10deg_fraction": float(np.mean(az <= 10.0)),
            "azimuth_within_20deg_fraction": float(np.mean(az <= 20.0)),
            "elevation_mean_deg": float(np.mean(el)),
            "elevation_median_deg": float(np.median(el)),
            "first_frame": rows[0],
        }

    payload = {
        "protocol": {
            "music_results": str(args.json_dir),
            "node_geometry": str(args.nod),
            "gps_source": str(args.gps_dir),
            "gps_role": "offline scoring only",
            "bearing_formula": "az=atan2(dy,dx); elevation=atan2(hypot(dx,dy),dz)",
            "frame_period_s": args.frame_period_s,
        },
        "node_summaries": summaries,
        "pooled": {
            "nodes_ok": int(sum(v.get("status") == "ok" for v in summaries.values())),
            "azimuth_mean_deg": float(np.mean(pooled_az)) if pooled_az else None,
            "azimuth_median_deg": float(np.median(pooled_az)) if pooled_az else None,
            "azimuth_p90_deg": float(np.quantile(pooled_az, 0.90)) if pooled_az else None,
            "azimuth_within_10deg_fraction": float(np.mean(np.asarray(pooled_az) <= 10.0)) if pooled_az else None,
            "azimuth_within_20deg_fraction": float(np.mean(np.asarray(pooled_az) <= 20.0)) if pooled_az else None,
            "elevation_mean_deg": float(np.mean(pooled_el)) if pooled_el else None,
            "elevation_median_deg": float(np.median(pooled_el)) if pooled_el else None,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(payload["pooled"], ensure_ascii=True))


if __name__ == "__main__":
    main()
