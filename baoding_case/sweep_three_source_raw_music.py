"""Small remote-only sweep for the raw three-source MUSIC front end."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_three_source_raw_music import estimate_doa, read_wavfm
from summarize_three_source_raw_music_xyz import (
    best_perm,
    gps_bearings,
    load_gps,
    load_nod,
    transform,
)


CONFIGS = [
    {"name": "orig_128x5_100_500", "nfft": 128, "fsnap": 5, "fl": 100.0, "fh": 500.0},
    {"name": "low_128x5_50_300", "nfft": 128, "fsnap": 5, "fl": 50.0, "fh": 300.0},
    {"name": "low_128x5_30_250", "nfft": 128, "fsnap": 5, "fl": 30.0, "fh": 250.0},
    {"name": "long_256x5_100_500", "nfft": 256, "fsnap": 5, "fl": 100.0, "fh": 500.0},
    {"name": "long_256x10_100_500", "nfft": 256, "fsnap": 10, "fl": 100.0, "fh": 500.0},
    {"name": "long_512x5_100_500", "nfft": 512, "fsnap": 5, "fl": 100.0, "fh": 500.0},
    {"name": "mid_256x5_50_300", "nfft": 256, "fsnap": 5, "fl": 50.0, "fh": 300.0},
    {"name": "paper_256x8_100_500", "nfft": 256, "fsnap": 8, "fl": 100.0, "fh": 500.0},
    {"name": "paper_512x4_100_500", "nfft": 512, "fsnap": 4, "fl": 100.0, "fh": 500.0},
    {"name": "paper_1024x2_100_500", "nfft": 1024, "fsnap": 2, "fl": 100.0, "fh": 500.0},
    {"name": "paper_256x8_50_300", "nfft": 256, "fsnap": 8, "fl": 50.0, "fh": 300.0},
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--nod", type=Path, required=True)
    p.add_argument("--gps-dir", type=Path, required=True)
    p.add_argument("--nodes", nargs="+", required=True)
    p.add_argument("--frames", type=int, default=20)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    nod = load_nod(args.nod)
    gps = [
        load_gps(args.gps_dir / "GPS1_plane1.gps"),
        load_gps(args.gps_dir / "GPS3_plane2.gps"),
        load_gps(args.gps_dir / "GPS4_plane2to3.gps"),
    ]
    result = {}
    max_samples = args.frames * max(c["nfft"] * c["fsnap"] for c in CONFIGS)
    for node_id in args.nodes:
        raw = args.raw_dir / f"20171107baoding_132614_{node_id}_19.wavfm"
        data = read_wavfm(raw, max_samples)
        correction = nod[node_id]
        result[node_id] = {}
        for cfg in CONFIGS:
            n_per = cfg["nfft"] * cfg["fsnap"]
            rows = []
            for frame in range(args.frames):
                lo, hi = frame * n_per, (frame + 1) * n_per
                doa = estimate_doa(
                    data[:, lo:hi], 3, 3050.0, cfg["nfft"], cfg["fsnap"],
                    cfg["fl"], cfg["fh"], 340.0, 1.0,
                )
                t = 132618.985 + frame * n_per / 3050.0
                true_az, true_el = gps_bearings(gps, correction, t)
                est_az = transform(doa["azimuth_deg"], correction["h_offset"], correction["h_direction"])
                est_el = transform(doa["elevation_deg"], correction["v_offset"], correction["v_direction"])
                score, perm, az_err = best_perm(est_az, true_az)
                el_err = [abs(est_el[perm[i]] - true_el[i]) for i in range(3)]
                rows.append({"az": score, "el": float(np.mean(el_err))})
            az = np.asarray([r["az"] for r in rows])
            el = np.asarray([r["el"] for r in rows])
            result[node_id][cfg["name"]] = {
                "nfft": cfg["nfft"], "fsnap": cfg["fsnap"],
                "fl_hz": cfg["fl"], "fh_hz": cfg["fh"],
                "frames": len(rows),
                "azimuth_mean_deg": float(np.mean(az)),
                "azimuth_median_deg": float(np.median(az)),
                "azimuth_p90_deg": float(np.quantile(az, 0.90)),
                "azimuth_within_20deg_fraction": float(np.mean(az <= 20.0)),
                "elevation_mean_deg": float(np.mean(el)),
                "elevation_median_deg": float(np.median(el)),
            }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"configs": CONFIGS, "nodes": result}, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps({node: {k: v["azimuth_mean_deg"] for k, v in values.items()} for node, values in result.items()}, ensure_ascii=True))


if __name__ == "__main__":
    main()
