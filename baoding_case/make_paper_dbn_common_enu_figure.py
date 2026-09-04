#!/usr/bin/env python3
"""Render the Baoding DBN baseline in one common array-centred ENU frame."""
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


PAPER_NODES = (1, 3, 5, 6, 7, 8, 11, 13)
COLORS = {1: "#176B87", 2: "#C45136", 3: "#6F5AA5"}
TARGET_FILES = {1: "GPS1_plane1.gps", 2: "GPS3_plane2.gps", 3: "GPS4_plane2to3.gps"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_nodes(path: Path) -> dict[int, dict[str, float]]:
    nodes: dict[int, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        try:
            node = int(fields[2])
            if node in PAPER_NODES:
                nodes[node] = {
                    "ip": int(fields[0].split(".")[-1]),
                    "x": float(fields[3]),
                    "y": float(fields[4]),
                    "z": float(fields[5]),
                }
        except (TypeError, ValueError):
            continue
    missing = sorted(set(PAPER_NODES) - set(nodes))
    if missing:
        raise RuntimeError(f"missing paper node coordinates: {missing}")
    return nodes


def read_track(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"frame_index", "time_hhmmss", "estimated_x", "estimated_y", "truth_x", "truth_y", "position_error_m"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"invalid baseline CSV: {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--nod-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    node_coords = read_nodes(args.nod_file)
    origin = np.mean(np.asarray([[node_coords[n]["x"], node_coords[n]["y"]] for n in PAPER_NODES]), axis=0)
    tracks = {target: read_track(args.baseline_root / f"target{target}_dbn_lanm_baseline.csv") for target in (1, 2, 3)}

    mpl.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
    })
    fig, (ax, ax_err) = plt.subplots(1, 2, figsize=(10.0, 4.6), gridspec_kw={"width_ratios": [1.35, 1.0]}, constrained_layout=True)

    for target in (1, 2, 3):
        rows = tracks[target]
        color = COLORS[target]
        est_x = np.asarray([float(row["estimated_x"]) for row in rows])
        est_y = np.asarray([float(row["estimated_y"]) for row in rows])
        truth_x = np.asarray([float(row["truth_x"]) for row in rows])
        truth_y = np.asarray([float(row["truth_y"]) for row in rows])
        # Common array-centred ENU frame, displayed in km.
        east_est = (est_x - origin[0]) / 1000.0
        north_est = (est_y - origin[1]) / 1000.0
        east_truth = (truth_x - origin[0]) / 1000.0
        north_truth = (truth_y - origin[1]) / 1000.0
        ax.plot(east_truth, north_truth, color=color, lw=1.8, ls="--", alpha=0.65, label=f"T{target} GPS")
        ax.plot(east_est, north_est, color=color, lw=2.0, label=f"T{target} DBN")
        ax.scatter(east_truth[0], north_truth[0], s=48, marker="o", color=color, edgecolor="white", linewidth=0.8, zorder=5)
        ax.scatter(east_truth[-1], north_truth[-1], s=50, marker="s", facecolor="white", edgecolor=color, linewidth=1.5, zorder=5)
        ax.text(east_truth[0] + 0.015, north_truth[0] + 0.015, f"T{target} start", color=color, fontsize=8)
        ax_err.plot(np.arange(len(rows)), np.asarray([float(row["position_error_m"]) for row in rows]), color=color, lw=1.6, label=f"T{target}")

    array_e = (np.asarray([node_coords[n]["x"] for n in PAPER_NODES]) - origin[0]) / 1000.0
    array_n = (np.asarray([node_coords[n]["y"] for n in PAPER_NODES]) - origin[1]) / 1000.0
    ax.scatter(array_e, array_n, marker="+", s=35, color="#333333", linewidth=1.0, zorder=4, label="8-node array")
    ax.scatter(0, 0, marker="*", s=95, color="#111111", edgecolor="white", linewidth=0.7, zorder=6, label="array ENU origin")
    ax.set_xlabel("East (km)")
    ax.set_ylabel("North (km)")
    ax.set_title("Common array-centred ENU trajectories", loc="left", weight="bold")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#E5E8EB", linewidth=0.6)
    ax.legend(frameon=False, ncol=2, loc="best")

    ax_err.set_title("2-D position error", loc="left", weight="bold")
    ax_err.set_xlabel("Frame")
    ax_err.set_ylabel("Error (m)")
    ax_err.grid(True, axis="y", color="#E5E8EB", linewidth=0.6)
    ax_err.legend(frameon=False, loc="upper left")
    ax_err.set_xlim(0, max(len(rows) - 1 for rows in tracks.values()))

    fig.suptitle("Baoding sanyuan_tongxinyuan_6 | 8-node DBN baseline", x=0.02, ha="left", fontsize=12, weight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    png = args.output.with_suffix(".png")
    pdf = args.output.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "claim_status": "paper_aligned_dbn_raw_baseline_common_enu",
        "coordinate_frame": "common local ENU; origin is arithmetic centroid of paper nodes 1,3,5,6,7,8,11,13",
        "paper_nodes": list(PAPER_NODES),
        "origin_xy_m": [float(origin[0]), float(origin[1])],
        "node_file": str(args.nod_file),
        "baseline_root": str(args.baseline_root),
        "target_mapping": {f"target{t}": TARGET_FILES[t] for t in (1, 2, 3)},
        "outputs": {"png": str(png), "pdf": str(pdf)},
        "source_sha256": {p.name: sha256(p) for p in [args.nod_file, *[args.baseline_root / f"target{t}_dbn_lanm_baseline.csv" for t in (1, 2, 3)]]},
        "plot_notes": [
            "GPS start markers are filled circles; GPS end markers are open squares.",
            "DBN and GPS are plotted in one common frame; no per-target recentering.",
            "Baseline CSV contains x/y only; no fabricated altitude is shown.",
        ],
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
