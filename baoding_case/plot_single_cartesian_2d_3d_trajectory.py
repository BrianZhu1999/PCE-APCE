#!/usr/bin/env python3
"""Render the real-acoustic single-source Cartesian trajectory in 2-D and 3-D.

GPS is loaded only as an offline reference.  The acoustic and PCE/APCE tracks
are read from the frozen frontend and filter bundle; no data are selected by
GPS error at plotting time.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"truth": "#252525", "acoustic": "#145A96", "pce": "#B84A3D", "apce": "#23846D"}


def rows(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def xyz(rows_, names=("px", "py", "pz")):
    return np.asarray([[float(r[n]) for n in names] for r in rows_], dtype=float)


def configure():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "axes.linewidth": 0.8,
        "axes.spines.right": False, "axes.spines.top": False,
        "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontend", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    configure(); args.output.mkdir(parents=True, exist_ok=True)

    truth_rows = rows(args.frontend / "gps_truth.csv")
    truth_t = np.asarray([float(r["time_s"]) for r in truth_rows])
    truth = xyz(truth_rows)
    tri_rows = [r for r in rows(args.frontend / "observations_cartesian.csv")
                if r["segment"] == "danyuan_panxuan_3" and r["valid"].lower() == "true"]
    tri_t = np.asarray([float(r["time_s"]) for r in tri_rows])
    tri = xyz(tri_rows, ("y_E", "y_N", "y_U"))
    pce_payload = json.loads((args.results / "runs/pce_seed_2026082501.json").read_text(encoding="utf-8"))
    apce_payload = json.loads((args.results / "runs/apce_seed_2026082501.json").read_text(encoding="utf-8"))
    pce = xyz(pce_payload["records"]); apce = xyz(apce_payload["records"])
    pce_t = np.asarray([float(r["time_s"]) for r in pce_payload["records"]])
    apce_t = np.asarray([float(r["time_s"]) for r in apce_payload["records"]])
    if not (np.allclose(tri_t, pce_t) and np.allclose(tri_t, apce_t)):
        raise RuntimeError("acoustic/PCE/APCE timestamps do not match")

    all_xyz = np.concatenate([truth, tri, pce, apce])
    lo, hi = all_xyz.min(axis=0), all_xyz.max(axis=0)
    centre = (lo + hi) / 2; span = np.maximum(hi - lo, 1.0)
    h = max(span[0], span[1]) * 0.56
    low = np.array([centre[0] - h, centre[1] - h, lo[2] - max(2.0, .08 * span[2])])
    high = np.array([centre[0] + h, centre[1] + h, hi[2] + max(2.0, .08 * span[2])])

    fig = plt.figure(figsize=(11.0, 5.0), constrained_layout=False)
    ax2 = fig.add_subplot(1, 2, 1); ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    fig.subplots_adjust(left=.07, right=.98, bottom=.19, top=.84, wspace=.18)
    tracks = [("GPS truth (offline)", truth, truth_t, COLORS["truth"], "-", 1.65),
              ("Acoustic triangulation", tri, tri_t, COLORS["acoustic"], "--", 1.05),
              ("PCE", pce, pce_t, COLORS["pce"], "-.", 1.0),
              ("APCE", apce, apce_t, COLORS["apce"], ":", 1.35)]
    for label, track, time, color, style, width in tracks:
        mark = max(1, len(track) // 8)
        ax2.plot(track[:, 0], track[:, 1], color=color, ls=style, lw=width,
                 marker="o", ms=2.7, markevery=mark, label=label)
        ax3.plot(track[:, 0], track[:, 1], track[:, 2], color=color, ls=style,
                 lw=width, marker="o", ms=2.7, markevery=mark)
    ax2.set(xlabel="East (m)", ylabel="North (m)", xlim=(low[0], high[0]), ylim=(low[1], high[1]))
    ax2.set_aspect("equal", adjustable="box"); ax2.grid(color="#D8D8D8", lw=.55)
    ax2.set_title("a  Horizontal trajectory", loc="left", fontweight="bold")
    ax2.legend(loc="upper left", fontsize=7)
    ax3.set(xlabel="East (m)", ylabel="North (m)", zlabel="Up (m)",
            xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]))
    ax3.set_box_aspect((1, 1, max((high[2]-low[2])/(high[0]-low[0]), .22)))
    ax3.view_init(elev=22, azim=-58); ax3.set_title("b  Three-dimensional trajectory", loc="left", fontweight="bold")
    fig.suptitle("Baoding real-acoustic single source: 267 s, 250 admitted frames", fontsize=10.5, fontweight="bold", x=.02, ha="left")
    fig.text(.5, .025, "GPS is evaluation-only; it was not used in initialization, filtering, branch weights, or frame selection. "
             "PCE/APCE are the frozen Cartesian smoke runs (seed 2026082501).", ha="center", fontsize=7, color="#4D4D4D")
    stem = args.output / "single_source_cartesian_2d_3d_trajectory"
    fig.savefig(stem.with_suffix(".png"), dpi=350, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    source = []
    for label, track, time, *_ in tracks:
        for i, (t, p) in enumerate(zip(time, track)):
            source.append({"source": label, "sample_index": i, "time_s": float(t), "east_m": float(p[0]), "north_m": float(p[1]), "up_m": float(p[2]), "gps_used_at_runtime": False})
    with (args.output / "single_source_cartesian_2d_3d_trajectory_source.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(source[0])); w.writeheader(); w.writerows(source)
    registry = {"figure_contract": {"core_conclusion": "The admitted Cartesian acoustic track is coherent over the full 267 s record; the frozen PCE/APCE point estimates are shown for audit and are not claimed superior.", "panels": {"a": "East-North trajectory", "b": "East-North-Up trajectory"}, "backend": "Python/Matplotlib"}, "gps_role": "offline evaluation only", "sources": {"frontend": str(args.frontend), "pce": str(args.results / "runs/pce_seed_2026082501.json"), "apce": str(args.results / "runs/apce_seed_2026082501.json")}, "outputs": {"png": str(stem.with_suffix('.png')), "pdf": str(stem.with_suffix('.pdf')), "svg": str(stem.with_suffix('.svg'))}}
    (args.output / "single_source_cartesian_2d_3d_trajectory_registry.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(registry, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
