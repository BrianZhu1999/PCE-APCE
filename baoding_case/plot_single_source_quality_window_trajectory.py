#!/usr/bin/env python3
"""Plot the frozen 10-s single-source quality window in 3-D and 2-D."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_truth(path: Path):
    rows = read_csv(path)
    return {float(row["time_s"]): np.asarray([float(row[k]) for k in ("px", "py", "pz")], dtype=float) for row in rows}


def nearest(truth, time_s):
    key = min(truth, key=lambda value: abs(value - time_s))
    return truth[key] if abs(key - time_s) <= 2.0 else None


def run_rows(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {float(row["time_s"]): np.asarray([float(row[k]) for k in ("px", "py", "pz")], dtype=float) for row in payload["records"]}


def rmse(est, truth):
    return float(np.sqrt(np.mean(np.sum((est - truth) ** 2, axis=1))))


def hms(seconds: float) -> str:
    seconds = int(round(seconds)) % 86400
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend", type=Path, required=True)
    parser.add_argument("--pce-runs", type=Path, required=True)
    parser.add_argument("--apce-runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    obs_rows = [row for row in read_csv(args.frontend / "observations_cartesian.csv") if row.get("valid", "False").lower() == "true"]
    obs_rows.sort(key=lambda row: float(row["time_s"]))
    times = [float(row["time_s"]) for row in obs_rows]
    truth_map = read_truth(args.frontend / "gps_truth.csv")
    truth = np.asarray([nearest(truth_map, time_s) for time_s in times], dtype=float)
    acoustic = np.asarray([[float(row[f"y_{k}"]) for k in ("E", "N", "U")] for row in obs_rows], dtype=float)

    def collect(root):
        series = []
        for path in sorted(root.glob("apce_seed_*.json")) + sorted(root.glob("pce_seed_*.json")):
            rows = run_rows(path)
            series.append(np.asarray([rows[time_s] for time_s in times], dtype=float))
        return np.asarray(series, dtype=float)

    pce = collect(args.pce_runs)
    apce = collect(args.apce_runs)
    pce_med, apce_med = np.median(pce, axis=0), np.median(apce, axis=0)
    origin = truth.mean(axis=0)
    truth -= origin; acoustic -= origin; pce_med -= origin; apce_med -= origin
    pce_errors = np.linalg.norm(pce_med - truth, axis=1)
    apce_errors = np.linalg.norm(apce_med - truth, axis=1)

    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8, "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.8,
        "legend.frameon": False,
    })
    colors = {"truth": "#20262C", "acoustic": "#9AA5AE", "pce": "#1D6F8A", "apce": "#C75A3C"}
    fig = plt.figure(figsize=(8.4, 4.4), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.15, 0.85))
    ax3 = fig.add_subplot(grid[0, 0], projection="3d")
    ax2 = fig.add_subplot(grid[0, 1])
    for data, color, label, lw, alpha in ((truth, colors["truth"], "GPS truth", 2.2, 1.0), (acoustic, colors["acoustic"], "Acoustic triangulation", 1.0, 0.75), (pce_med, colors["pce"], f"PCE median (RMSE {rmse(pce_med, truth):.1f} m)", 1.7, 1.0), (apce_med, colors["apce"], f"APCE median (RMSE {rmse(apce_med, truth):.1f} m)", 1.7, 1.0)):
        ax3.plot(data[:, 0], data[:, 1], data[:, 2], color=color, lw=lw, alpha=alpha, label=label)
    ax3.scatter(truth[0, 0], truth[0, 1], truth[0, 2], color=colors["truth"], s=22, marker="o", zorder=5)
    ax3.scatter(truth[-1, 0], truth[-1, 1], truth[-1, 2], color=colors["truth"], s=26, marker="s", zorder=5)
    ax3.set_xlabel("East offset (m)"); ax3.set_ylabel("North offset (m)"); ax3.set_zlabel("Up offset (m)")
    ax3.set_title("A  Three-dimensional trajectory", loc="left", fontweight="bold")
    span = np.ptp(np.vstack((truth, acoustic, pce_med, apce_med)), axis=0)
    ax3.set_box_aspect(tuple(np.maximum(span, 1.0))); ax3.view_init(elev=23, azim=-58)
    ax3.legend(loc="upper left", fontsize=7)
    for data, color, label, lw, alpha in ((truth, colors["truth"], "GPS truth", 2.2, 1.0), (acoustic, colors["acoustic"], "Acoustic", 1.0, 0.75), (pce_med, colors["pce"], "PCE", 1.7, 1.0), (apce_med, colors["apce"], "APCE", 1.7, 1.0)):
        ax2.plot(data[:, 0], data[:, 1], color=color, lw=lw, alpha=alpha, label=label)
    ax2.scatter(truth[0, 0], truth[0, 1], color=colors["truth"], s=22, marker="o"); ax2.scatter(truth[-1, 0], truth[-1, 1], color=colors["truth"], s=26, marker="s")
    ax2.set_aspect("equal", adjustable="box"); ax2.set_xlabel("East offset (m)"); ax2.set_ylabel("North offset (m)")
    ax2.set_title("B  Horizontal projection", loc="left", fontweight="bold"); ax2.grid(color="#E2E7EB", lw=0.5); ax2.legend(fontsize=7, loc="best")
    fig.suptitle(f"Baoding single-source stable window | 10 frames | {hms(times[0])}--{hms(times[-1])} | median of 5 seeds", x=0.02, ha="left", fontsize=11, fontweight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    registry = {"figure": str(args.output), "backend": "Python/matplotlib", "window": {"frames": len(times), "start_time_s": times[0], "end_time_s": times[-1]}, "source_frontend": str(args.frontend), "source_pce_runs": str(args.pce_runs), "source_apce_runs": str(args.apce_runs), "gps_role": "offline scoring only", "aggregation": "median trajectory across five seeds", "metrics": {"acoustic_rmse_m": rmse(acoustic, truth), "pce_median_rmse_m": rmse(pce_med, truth), "apce_median_rmse_m": rmse(apce_med, truth)}}
    args.output.with_name(args.output.name + "_registry.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(registry, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
