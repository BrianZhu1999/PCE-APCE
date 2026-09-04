#!/usr/bin/env python3
"""Plot GPS truth and frozen Cartesian APCE on the selected 67-s orbit window."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


START_S = 46254.0
END_S = 46320.0


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_truth(path: Path):
    rows = read_csv(path)
    return {float(r["time_s"]): np.asarray([float(r[k]) for k in ("px", "py", "pz")], dtype=float) for r in rows}


def load_apce(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {float(r["time_s"]): np.asarray([float(r[k]) for k in ("px", "py", "pz")], dtype=float) for r in payload["records"]}


def nearest(mapping, time_s):
    key = min(mapping, key=lambda value: abs(value - time_s))
    return mapping[key] if abs(key - time_s) <= 2.0 else None


def hms(seconds: float) -> str:
    seconds = int(round(seconds)) % 86400
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def rmse(est, truth):
    return float(np.sqrt(np.mean(np.sum((est - truth) ** 2, axis=1))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend", type=Path, required=True)
    parser.add_argument("--apce-runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    truth_map = load_truth(args.frontend / "gps_truth.csv")
    apce_maps = [load_apce(path) for path in sorted(args.apce_runs.glob("apce_seed_*.json"))]
    if len(apce_maps) < 1:
        raise RuntimeError("no APCE seed files found")
    times = sorted(t for t in truth_map if START_S <= t <= END_S)
    if len(times) < 60:
        raise RuntimeError(f"selected interval has only {len(times)} GPS frames")
    truth = np.asarray([truth_map[t] for t in times], dtype=float)
    apce_series = []
    for mapping in apce_maps:
        series = np.asarray([nearest(mapping, t) for t in times], dtype=float)
        if not np.isfinite(series).all():
            raise RuntimeError("APCE does not cover the complete selected interval")
        apce_series.append(series)
    apce = np.median(np.asarray(apce_series), axis=0)
    origin = truth.mean(axis=0)
    truth_offset, apce_offset = truth - origin, apce - origin
    errors = np.linalg.norm(apce - truth, axis=1)

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8, "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.8,
        "legend.frameon": False,
    })
    colors = {"truth": "#20262C", "apce": "#C75A3C"}
    fig = plt.figure(figsize=(8.5, 4.5), constrained_layout=False)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.15, 0.85))
    ax3 = fig.add_subplot(grid[0, 0], projection="3d")
    ax2 = fig.add_subplot(grid[0, 1])
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.20, top=0.82, wspace=0.18)

    ax3.plot(truth_offset[:, 0], truth_offset[:, 1], truth_offset[:, 2], color=colors["truth"], lw=2.1, label="GPS truth")
    ax3.plot(apce_offset[:, 0], apce_offset[:, 1], apce_offset[:, 2], color=colors["apce"], lw=1.8, label=f"APCE median (RMSE {rmse(apce, truth):.1f} m)")
    ax3.scatter(*truth_offset[0], color=colors["truth"], s=24, marker="o", zorder=5)
    ax3.scatter(*truth_offset[-1], color=colors["truth"], s=28, marker="s", zorder=5)
    ax3.set_xlabel("East offset (m)"); ax3.set_ylabel("North offset (m)"); ax3.set_zlabel("Up offset (m)")
    ax3.set_title("A  Three-dimensional trajectory", loc="left", fontweight="bold")
    span = np.ptp(np.vstack((truth_offset, apce_offset)), axis=0)
    ax3.set_box_aspect(tuple(np.maximum(span, 1.0))); ax3.view_init(elev=23, azim=-58)
    ax3.legend(loc="upper left", fontsize=7)

    ax2.plot(truth_offset[:, 0], truth_offset[:, 1], color=colors["truth"], lw=2.1, label="GPS truth")
    ax2.plot(apce_offset[:, 0], apce_offset[:, 1], color=colors["apce"], lw=1.8, label="APCE median")
    ax2.scatter(truth_offset[0, 0], truth_offset[0, 1], color=colors["truth"], s=24, marker="o")
    ax2.scatter(truth_offset[-1, 0], truth_offset[-1, 1], color=colors["truth"], s=28, marker="s")
    ax2.set_aspect("equal", adjustable="box"); ax2.set_xlabel("East offset (m)"); ax2.set_ylabel("North offset (m)")
    ax2.set_title("B  Horizontal projection", loc="left", fontweight="bold")
    ax2.grid(color="#E2E7EB", lw=0.5); ax2.legend(fontsize=7, loc="best")
    fig.suptitle(f"Baoding single-source near-full-circle window | {hms(START_S)}--{hms(END_S)} | {len(times)} s | {len(apce_maps)} APCE seeds", x=0.02, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.5, 0.01, "GPS is used for offline evaluation only; APCE receives Cartesian acoustic observations and their covariance.", ha="center", fontsize=7, color="#4D4D4D")
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / "baoding_single_source_67s_gps_apce_trajectory"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    with (args.output / "baoding_single_source_67s_gps_apce_source.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["time_s", "gps_E", "gps_N", "gps_U", "apce_E", "apce_N", "apce_U", "apce_error_m"])
        writer.writeheader()
        for t, gt, est, error in zip(times, truth, apce, errors):
            writer.writerow({"time_s": t, "gps_E": gt[0], "gps_N": gt[1], "gps_U": gt[2], "apce_E": est[0], "apce_N": est[1], "apce_U": est[2], "apce_error_m": error})
    registry = {
        "figure_contract": {"core_conclusion": "APCE is compared with GPS over the preselected near-full-circle 67-s window.", "panels": {"A": "East-North-Up trajectory", "B": "East-North projection"}, "backend": "Python/matplotlib"},
        "window": {"start_time_s": START_S, "end_time_s": END_S, "duration_s": END_S - START_S + 1, "gps_frames": len(times), "selection_rule": "predeclared offline geometric audit: contiguous window with >=300 degree GPS sweep and lowest acoustic triangulation RMSE"},
        "gps_role": "offline scoring only",
        "sources": {"frontend": str(args.frontend), "apce_runs": [str(path) for path in sorted(args.apce_runs.glob("apce_seed_*.json"))]},
        "aggregation": "median Cartesian APCE trajectory across seeds",
        "metrics": {"apce_median_rmse_m": rmse(apce, truth), "apce_median_error_m": float(np.median(errors)), "apce_p90_error_m": float(np.percentile(errors, 90))},
        "outputs": {"png": str(stem.with_suffix(".png")), "pdf": str(stem.with_suffix(".pdf")), "svg": str(stem.with_suffix(".svg")), "source_csv": str(args.output / "baoding_single_source_67s_gps_apce_source.csv")},
    }
    (args.output / "baoding_single_source_67s_gps_apce_trajectory_registry.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(registry, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
