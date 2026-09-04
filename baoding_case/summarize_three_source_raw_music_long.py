"""Summarize a continuous raw-MUSIC run against GPS XYZ bearings."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from summarize_three_source_raw_music_xyz import gps_bearings, load_gps, load_nod, transform


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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--json-dir", type=Path, required=True)
    p.add_argument("--nod", type=Path, required=True)
    p.add_argument("--gps-dir", type=Path, required=True)
    p.add_argument("--nodes", nargs="+", required=True)
    p.add_argument("--pattern", default="node{node}_k3_paper512x4_82.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--start-time", type=float, default=132618.985)
    p.add_argument("--fs", type=float, default=3050.0)
    p.add_argument("--nfft", type=int, default=512)
    p.add_argument("--fsnap", type=int, default=4)
    args = p.parse_args()

    nod = load_nod(args.nod)
    gps = [
        load_gps(args.gps_dir / "GPS1_plane1.gps"),
        load_gps(args.gps_dir / "GPS3_plane2.gps"),
        load_gps(args.gps_dir / "GPS4_plane2to3.gps"),
    ]
    dt = args.nfft * args.fsnap / args.fs
    summaries = {}
    all_rows = []
    for node_id in args.nodes:
        path = args.json_dir / args.pattern.format(node=node_id)
        if not path.exists():
            summaries[node_id] = {"status": "missing_result"}
            continue
        if node_id not in nod:
            summaries[node_id] = {"status": "missing_geometry"}
            continue
        corr = nod[node_id]
        music = json.loads(path.read_text(encoding="utf-8"))
        rows = []
        for item in music["frames"]:
            t = args.start_time + int(item["frame"]) * dt
            true_az, true_el = gps_bearings(gps, corr, t)
            est_az = transform(item["azimuth_deg"], corr["h_offset"], corr["h_direction"])
            est_el = transform(item["elevation_deg"], corr["v_offset"], corr["v_direction"])
            score, perm, az_err = best_perm(est_az, true_az)
            el_err = [abs(est_el[perm[i]] - true_el[i]) for i in range(3)]
            row = {
                "node": node_id, "frame": int(item["frame"]), "time": t,
                "estimated_azimuth_deg": est_az, "gps_azimuth_deg": true_az,
                "estimated_elevation_deg": est_el, "gps_elevation_deg": true_el,
                "assignment": list(perm), "azimuth_abs_error_deg": az_err,
                "elevation_abs_error_deg": el_err,
                "azimuth_mean_error_deg": score,
                "elevation_mean_error_deg": float(np.mean(el_err)),
                "azimuth_peak_strength": item.get("azimuth_refined_peak", []),
                "elevation_peak_strength": item.get("elevation_refined_peak", []),
            }
            rows.append(row)
            all_rows.append(row)
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

    out = {
        "protocol": {
            "result_pattern": args.pattern,
            "frame_period_s": dt,
            "gps_role": "offline scoring only",
            "nodes": args.nodes,
        },
        "node_summaries": summaries,
        "pooled": {
            "nodes_ok": int(sum(x.get("status") == "ok" for x in summaries.values())),
            "frames": len(all_rows),
            "azimuth_mean_deg": float(np.mean([r["azimuth_mean_error_deg"] for r in all_rows])) if all_rows else None,
            "azimuth_median_deg": float(np.median([r["azimuth_mean_error_deg"] for r in all_rows])) if all_rows else None,
            "azimuth_p90_deg": float(np.quantile([r["azimuth_mean_error_deg"] for r in all_rows], 0.90)) if all_rows else None,
            "azimuth_within_10deg_fraction": float(np.mean(np.asarray([r["azimuth_mean_error_deg"] for r in all_rows]) <= 10.0)) if all_rows else None,
            "azimuth_within_20deg_fraction": float(np.mean(np.asarray([r["azimuth_mean_error_deg"] for r in all_rows]) <= 20.0)) if all_rows else None,
            "elevation_mean_deg": float(np.mean([r["elevation_mean_error_deg"] for r in all_rows])) if all_rows else None,
            "elevation_median_deg": float(np.median([r["elevation_mean_error_deg"] for r in all_rows])) if all_rows else None,
        },
        "rows": all_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(out["pooled"], ensure_ascii=True))


if __name__ == "__main__":
    main()
