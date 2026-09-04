#!/usr/bin/env python3
"""Render the Baoding DBN smoke baseline in a Fig.15-like absolute coordinate frame."""
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
COLORS = {1: "#2E86B7", 2: "#E67E22", 3: "#4DAF4A"}
PAPER_X_OFFSET_M = 3.861e7


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
                }
        except (TypeError, ValueError):
            continue
    missing = sorted(set(PAPER_NODES) - set(nodes))
    if missing:
        raise RuntimeError(f"missing paper node coordinates: {missing}")
    return nodes


def read_track(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"frame_index", "time_hhmmss", "estimated_x", "estimated_y", "truth_x", "truth_y"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"invalid baseline CSV: {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--nod-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    nodes = read_nodes(args.nod_file)
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
    fig, ax = plt.subplots(figsize=(7.5, 5.3), constrained_layout=True)

    # Paper-style coordinates: x is displayed after subtracting the paper's
    # scientific-notation offset, while y remains in projected metres.
    for target in (1, 2, 3):
        rows = tracks[target]
        color = COLORS[target]
        tx = np.asarray([float(row["truth_x"]) for row in rows])
        ty = np.asarray([float(row["truth_y"]) for row in rows])
        ex = np.asarray([float(row["estimated_x"]) for row in rows])
        ey = np.asarray([float(row["estimated_y"]) for row in rows])
        xtruth = tx - PAPER_X_OFFSET_M
        xest = ex - PAPER_X_OFFSET_M
        ax.plot(xtruth, ty, color=color, lw=1.5, alpha=0.85, label=f"GPS T{target}")
        ax.plot(xest, ey, color=color, lw=1.7, marker="o", markersize=2.7, markevery=1, alpha=0.95, label=f"DBN T{target}")
        # Fig.15 convention: black square = beginning, black star = end.
        ax.scatter(xtruth[0], ty[0], s=38, marker="s", color="black", zorder=6)
        ax.scatter(xtruth[-1], ty[-1], s=48, marker="*", color="black", zorder=6)
        ax.text(xtruth[0] + 18, ty[0] + 18, f"T{target} start", color=color, fontsize=8)

    # Array nodes from the eight paper nodes, with their IP suffix labels.
    for node in PAPER_NODES:
        x = nodes[node]["x"] - PAPER_X_OFFSET_M
        y = nodes[node]["y"]
        ax.scatter(x, y, marker="^", s=32, color="#D62728", edgecolor="white", linewidth=0.5, zorder=5)
        ax.text(x + 13, y + 11, f"IP{nodes[node]['ip']}", color="#8B1A1A", fontsize=7)

    ax.set_xlim(4300, 5900)
    ax.set_ylim(4_336_300, 4_337_700)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.text(0.995, 1.012, "+3.861×10⁷", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    ax.set_title("Tracking trajectories", weight="bold")
    ax.text(0.012, 0.985, "25 frames (~5 s)", transform=ax.transAxes, ha="left", va="top", fontsize=8, color="#555555")
    ax.grid(True, linestyle="--", linewidth=0.55, color="#BFC5CB", alpha=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=True, framealpha=0.92, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.10), borderpad=0.45, columnspacing=1.0, handlelength=2.0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    png = args.output.with_suffix(".png")
    pdf = args.output.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "claim_status": "paper_aligned_dbn_raw_baseline_fig15_coordinate_style",
        "paper_style": "Fig.15-like absolute projected coordinate frame",
        "paper_x_offset_m": PAPER_X_OFFSET_M,
        "x_display_limits_m": [4300.0, 5900.0],
        "y_display_limits_m": [4336300.0, 4337700.0],
        "paper_nodes": list(PAPER_NODES),
        "node_ip_mapping": {str(node): nodes[node]["ip"] for node in PAPER_NODES},
        "baseline_root": str(args.baseline_root),
        "nod_file": str(args.nod_file),
        "frame_count": {str(target): len(tracks[target]) for target in (1, 2, 3)},
        "outputs": {"png": str(png), "pdf": str(pdf)},
        "source_sha256": {
            "nod": sha256(args.nod_file),
            **{f"target{target}": sha256(args.baseline_root / f"target{target}_dbn_lanm_baseline.csv") for target in (1, 2, 3)},
        },
        "warning": "This is a 25-frame (~5 s) smoke baseline; it is not the paper's complete ~55 s Fig.15 reproduction.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
