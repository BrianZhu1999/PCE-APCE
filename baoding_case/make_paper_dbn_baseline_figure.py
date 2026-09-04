#!/usr/bin/env python3
"""Render the paper-aligned three-target DBN raw baseline as an a/b figure."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = ("#1D6F8A", "#C75A3C", "#6C5B9B")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--truth-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = {target: args.baseline_root / f"target{target}_dbn_lanm_baseline.csv" for target in (1, 2, 3)}
    rows = {target: list(csv.DictReader(path.open(encoding="utf-8"))) for target, path in files.items()}
    truth_files = {1: "GPS1_plane1.gps", 2: "GPS3_plane2.gps", 3: "GPS4_plane2to3.gps"}
    # Baseline CSV already contains GPS-aligned truth coordinates only for the
    # error audit; use estimated coordinates and re-centre each panel on GPS.
    fig = plt.figure(figsize=(9.2, 4.9), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.35, 1.0))
    ax3 = fig.add_subplot(grid[0, 0], projection="3d")
    ax2 = fig.add_subplot(grid[0, 1])
    error_values = []
    for target, color in zip((1, 2, 3), COLORS):
        x = np.asarray([float(row["estimated_x"]) for row in rows[target]])
        y = np.asarray([float(row["estimated_y"]) for row in rows[target]])
        z = np.zeros_like(x)
        tx = np.asarray([float(row["truth_x"]) for row in rows[target]])
        ty = np.asarray([float(row["truth_y"]) for row in rows[target]])
        origin = np.asarray([tx[0], ty[0]])
        xe, yn = (x - origin[0]) / 1000.0, (y - origin[1]) / 1000.0
        te, tn = (tx - origin[0]) / 1000.0, (ty - origin[1]) / 1000.0
        ax3.plot(te, tn, np.zeros_like(te), color=color, lw=2.0, ls="--", alpha=0.65, label=f"T{target} GPS")
        ax3.plot(xe, yn, z, color=color, lw=1.8, label=f"T{target} DBN")
        ax2.plot(tn, te, color=color, lw=1.2, ls="--", alpha=0.6, label=f"T{target} GPS")
        ax2.plot(yn, xe, color=color, lw=1.5, label=f"T{target} DBN")
        error_values.append(np.asarray([float(row["position_error_m"]) for row in rows[target]]))
    ax3.set_title("a  Paper-aligned three-target DBN baseline", loc="left", weight="bold")
    ax3.set_xlabel("East offset (km)"); ax3.set_ylabel("North offset (km)"); ax3.set_zlabel("Up offset (km)")
    ax3.legend(fontsize=6, ncol=2, loc="best")
    ax2.boxplot(error_values, positions=(1, 2, 3), widths=0.45, patch_artist=True,
                boxprops={"facecolor": "#DCE7ED", "edgecolor": "#37474F"},
                medianprops={"color": "#111111", "linewidth": 1.0},
                whiskerprops={"color": "#37474F"}, capprops={"color": "#37474F"})
    ax2.set_xticks((1, 2, 3), ("T1", "T2", "T3")); ax2.set_ylabel("2-D position error (m)")
    ax2.set_title("b  Error audit", loc="left", weight="bold"); ax2.grid(axis="y", color="#E2E7EB", lw=0.5)
    fig.suptitle("Baoding sanyuan_tongxinyuan_6 | 8-node paper protocol | inspection baseline only", x=0.01, ha="left", fontsize=10, weight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    payload = {"claim_status": "paper_aligned_dbn_raw_baseline", "baseline_root": str(args.baseline_root), "panel_labels": ["a", "b"], "target_count": 3, "warning": "No PCE/APCE result is shown; DBN adapter is inspection-only."}
    args.output.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
