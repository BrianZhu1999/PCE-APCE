#!/usr/bin/env python3
"""Plot the public wideband DBN synthetic trajectory reproduction."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {1: "#176B87", 2: "#D97941"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.trajectory.open(encoding="utf-8", newline="")))
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    tracks = {target: [row for row in rows if int(row["target"]) == target] for target in (1, 2)}

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    fig, (ax_traj, ax_err) = plt.subplots(1, 2, figsize=(9.4, 4.0), constrained_layout=True)

    for target in (1, 2):
        track = tracks[target]
        color = COLORS[target]
        truth_x = np.asarray([float(row["truth_x"]) for row in track])
        truth_y = np.asarray([float(row["truth_y"]) for row in track])
        est_x = np.asarray([float(row["estimated_x"]) for row in track])
        est_y = np.asarray([float(row["estimated_y"]) for row in track])
        time_s = np.asarray([float(row["time_s"]) for row in track])
        error = np.asarray([float(row["position_error_m"]) for row in track])
        ax_traj.plot(truth_x, truth_y, color=color, lw=1.8, alpha=0.45, label=f"True T{target}")
        ax_traj.plot(est_x, est_y, color=color, lw=1.8, marker="o", markersize=2.5, markevery=4, label=f"DBN T{target}")
        ax_traj.scatter(truth_x[0], truth_y[0], s=32, marker="s", color="black", zorder=4)
        ax_traj.scatter(truth_x[-1], truth_y[-1], s=44, marker="*", color="black", zorder=4)
        ax_err.plot(time_s, error, color=color, lw=1.8, label=f"T{target}")

    ax_traj.set_title("Synthetic wideband DBN trajectories", loc="left", weight="bold")
    ax_traj.set_xlabel("x (m)")
    ax_traj.set_ylabel("y (m)")
    ax_traj.grid(True, ls="--", lw=0.5, alpha=0.45)
    ax_traj.set_aspect("equal", adjustable="datalim")
    ax_traj.legend(frameon=False, fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))

    ax_err.set_title("Position error over time", loc="left", weight="bold")
    ax_err.set_xlabel("time (s)")
    ax_err.set_ylabel("position error (m)")
    ax_err.grid(True, ls="--", lw=0.5, alpha=0.45)
    ax_err.legend(frameon=False, fontsize=7, loc="upper right")
    ax_err.text(0.02, 0.03, f"mean OSPA$_2$ = {metrics['mean_ospa_order2_m']:.2f} m", transform=ax_err.transAxes, fontsize=8, color="#444444")

    fig.suptitle("Public MTT_WB_DBN synthetic reproduction", x=0.02, ha="left", fontsize=12, weight="bold")
    fig.text(0.02, 0.005, "80 frames × 1.024 s = 81.92 s; fixed-seed deterministic reproduction; not the Baoding field experiment.", fontsize=7, color="#555555")

    args.output_root.mkdir(parents=True, exist_ok=True)
    stem = args.output_root / "wideband_dbn_synthetic_trajectory"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    registry = {
        "claim_status": "upstream_synthetic_reproduction_figure",
        "trajectory_source": str(args.trajectory),
        "trajectory_sha256": sha256(args.trajectory),
        "metrics_source": str(args.metrics),
        "outputs": {"png": str(stem.with_suffix(".png")), "pdf": str(stem.with_suffix(".pdf")), "svg": str(stem.with_suffix(".svg"))},
        "frame_count": 80,
        "frame_dt_s": 1.024,
        "warning": "Synthetic two-target/four-node reproduction only; not a Baoding field result.",
    }
    (args.output_root / "wideband_dbn_synthetic_trajectory_registry.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(registry, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
