"""Summarize raw three-source MUSIC peaks against offline GPS DOA tables."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def circular_abs(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def best_assignment(est: list[float], truth: list[float]) -> tuple[float, list[float], tuple[int, ...]]:
    errors = []
    best = None
    for perm in itertools.permutations(range(len(est))):
        vals = [circular_abs(est[perm[i]], truth[i]) for i in range(len(truth))]
        score = float(np.mean(vals))
        if best is None or score < best[0]:
            best = (score, vals, perm)
    assert best is not None
    return best


def load_nod(path: Path) -> dict[str, dict[str, float]]:
    nodes = {}
    for line in path.read_text(errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 11:
            continue
        ip = fields[0]
        nodes[ip.rsplit(".", 1)[-1]] = {
            "h_offset_deg": float(fields[7]),
            "v_offset_deg": float(fields[8]),
            "h_direction": float(fields[9]),
            "v_direction": float(fields[10]),
        }
    return nodes


def load_gps_doa(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr[None, :]
    # gps_doa format: node, azi1, ele1, azi2, ele2, azi3, ele3,
    # distance1, distance2, distance3, timestamp.
    return arr[:, 1:7], arr[:, 10], arr[:, 0]


def transform_angles(angles: list[float], offset: float, direction: float) -> list[float]:
    sign = -1.0 if int(round(direction)) == 1 else 1.0
    return [float((sign * (a + offset)) % 360.0) for a in angles]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--json-dir", type=Path, required=True)
    p.add_argument("--nod", type=Path, required=True)
    p.add_argument("--gps-dir", type=Path, required=True)
    p.add_argument("--nodes", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gps-tolerance-s", type=float, default=0.55)
    args = p.parse_args()

    nod = load_nod(args.nod)
    node_summaries = {}
    all_az_errors = []
    all_el_errors = []
    all_frame_rows = []
    for node in args.nodes:
        result_path = args.json_dir / f"node{node}_k3_frames20.json"
        if not result_path.exists():
            node_summaries[node] = {"status": "missing_music_result"}
            continue
        music = json.loads(result_path.read_text(encoding="utf-8"))
        gps_path = args.gps_dir / f"gps_doa_{node}.txt"
        if not gps_path.exists():
            node_summaries[node] = {"status": "missing_gps_doa_table"}
            continue
        gps_values, gps_times, gps_nodes = load_gps_doa(gps_path)
        correction = nod.get(node, {"h_offset_deg": 0.0, "v_offset_deg": 0.0, "h_direction": 0.0, "v_direction": 0.0})
        rows = []
        for item in music["frames"]:
            # Raw WAVFM begins at the first sample; 640 samples is one update.
            # The source package's .gpstime starts at 13:26:18.985 for this set.
            elapsed = float(item["frame"]) * 128.0 * 5.0 / 3050.0
            target_time = 132618.985 + elapsed
            idx = int(np.argmin(np.abs(gps_times - target_time)))
            if abs(float(gps_times[idx] - target_time)) > args.gps_tolerance_s:
                continue
            est_az = transform_angles(item["azimuth_deg"], correction["h_offset_deg"], correction["h_direction"])
            est_el = transform_angles(item["elevation_deg"], correction["v_offset_deg"], correction["v_direction"])
            true_az = [float(gps_values[idx, 0]), float(gps_values[idx, 2]), float(gps_values[idx, 4])]
            true_el = [float(gps_values[idx, 1]), float(gps_values[idx, 3]), float(gps_values[idx, 5])]
            az_score, az_vals, az_perm = best_assignment(est_az, true_az)
            # Elevation is not circular; retain the azimuth assignment so that
            # the result tests a complete source pairing rather than resorting
            # elevations independently.
            el_vals = [abs(est_el[az_perm[i]] - true_el[i]) for i in range(3)]
            rows.append({
                "frame": int(item["frame"]),
                "target_time": target_time,
                "gps_time": float(gps_times[idx]),
                "estimated_azimuth_deg": est_az,
                "gps_azimuth_deg": true_az,
                "estimated_elevation_deg": est_el,
                "gps_elevation_deg": true_el,
                "azimuth_assignment": list(az_perm),
                "azimuth_abs_error_deg": az_vals,
                "elevation_abs_error_deg": el_vals,
                "azimuth_mean_error_deg": az_score,
                "elevation_mean_error_deg": float(np.mean(el_vals)),
            })
            all_az_errors.extend(az_vals)
            all_el_errors.extend(el_vals)
            all_frame_rows.append({"node": node, **rows[-1]})
        if rows:
            az_means = np.asarray([r["azimuth_mean_error_deg"] for r in rows])
            el_means = np.asarray([r["elevation_mean_error_deg"] for r in rows])
            node_summaries[node] = {
                "status": "ok",
                "frames_scored": len(rows),
                "azimuth_mean_error_deg": float(np.mean(az_means)),
                "azimuth_median_error_deg": float(np.median(az_means)),
                "azimuth_p90_error_deg": float(np.quantile(az_means, 0.90)),
                "azimuth_within_20deg_fraction": float(np.mean(az_means <= 20.0)),
                "elevation_mean_error_deg": float(np.mean(el_means)),
                "elevation_median_error_deg": float(np.median(el_means)),
                "first_frame": rows[0],
            }
        else:
            node_summaries[node] = {"status": "no_time_aligned_frames"}

    payload = {
        "protocol": {
            "estimator": "raw 19-channel WAVFM MUSIC translation with K=3",
            "gps_role": "offline scoring only",
            "gps_time_tolerance_s": args.gps_tolerance_s,
            "nodes_requested": args.nodes,
            "correction_source": str(args.nod),
            "gps_source_dir": str(args.gps_dir),
        },
        "node_summaries": node_summaries,
        "pooled": {
            "nodes_scored": int(sum(v.get("status") == "ok" for v in node_summaries.values())),
            "azimuth_error_mean_deg": float(np.mean(all_az_errors)) if all_az_errors else None,
            "azimuth_error_median_deg": float(np.median(all_az_errors)) if all_az_errors else None,
            "azimuth_error_p90_deg": float(np.quantile(all_az_errors, 0.90)) if all_az_errors else None,
            "azimuth_within_20deg_fraction": float(np.mean(np.asarray(all_az_errors) <= 20.0)) if all_az_errors else None,
            "elevation_error_mean_deg": float(np.mean(all_el_errors)) if all_el_errors else None,
            "elevation_error_median_deg": float(np.median(all_el_errors)) if all_el_errors else None,
        },
        "frame_rows": all_frame_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(payload["pooled"], ensure_ascii=True))


if __name__ == "__main__":
    main()
