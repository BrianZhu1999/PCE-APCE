#!/usr/bin/env python3
"""Create a compact audit figure for the wideband DBN reproduction attempt."""
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


PAPER_XYV = {
    1: (38614853.4, 4337388.27, 25.85033, 23.00194),
    2: (38615012.2, 4336467.20, -41.09208, 6.67753),
    3: (38615647.2, 4337215.10, 3.20862, -41.49795),
}
COLORS = {1: "#176B87", 2: "#D97941", 3: "#4D8B59"}
LABELS = {1: "T1", 2: "T2", 3: "T3"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(root: Path) -> dict[int, list[dict[str, str]]]:
    tracks = {}
    for target in (1, 2, 3):
        with (root / f"target{target}_dbn_lanm_baseline.csv").open(encoding="utf-8", newline="") as stream:
            tracks[target] = list(csv.DictReader(stream))
    return tracks


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def panel_a(ax, tracks: dict[int, list[dict[str, str]]], title: str) -> None:
    all_x = []
    all_y = []
    for target in (1, 2, 3):
        rows = tracks[target]
        color = COLORS[target]
        gx = np.asarray([float(row["truth_x"]) for row in rows]) - 38615229.471995264
        gy = np.asarray([float(row["truth_y"]) for row in rows]) - 4337066.551922237
        ex = np.asarray([float(row["estimated_x"]) for row in rows]) - 38615229.471995264
        ey = np.asarray([float(row["estimated_y"]) for row in rows]) - 4337066.551922237
        all_x.extend(gx.tolist()); all_x.extend(ex.tolist())
        all_y.extend(gy.tolist()); all_y.extend(ey.tolist())
        ax.plot(gx, gy, color=color, lw=2.0, alpha=0.55, label=f"GPS {LABELS[target]}")
        ax.plot(ex, ey, color=color, lw=1.9, marker="o", markersize=2.7, label=f"DBN {LABELS[target]}")
        ax.scatter(gx[0], gy[0], marker="s", s=28, color="black", zorder=5)
        ax.scatter(gx[-1], gy[-1], marker="*", s=48, color="black", zorder=5)
    ax.set_title(title, loc="left", weight="bold")
    ax.set_xlabel("East coordinate relative to node centroid (m)")
    ax.set_ylabel("North coordinate relative to node centroid (m)")
    ax.grid(True, ls="--", lw=0.5, alpha=0.45)
    x_margin = max(35.0, (max(all_x) - min(all_x)) * 0.18)
    y_margin = max(35.0, (max(all_y) - min(all_y)) * 0.18)
    ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
    ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)


def panel_b(ax, tracks: dict[int, list[dict[str, str]]], sample_rate: float, snapshot: int) -> None:
    dt = snapshot / sample_rate
    for target in (1, 2, 3):
        rows = tracks[target]
        dbn = np.asarray([float(row["position_error_m"]) for row in rows])
        prior = []
        x0, y0, vx, vy = PAPER_XYV[target]
        for row in rows:
            k = int(row["frame_index"]) + 1
            px = x0 + vx * dt * k
            py = y0 + vy * dt * k
            prior.append(np.hypot(px - float(row["truth_x"]), py - float(row["truth_y"])))
        t = np.arange(len(rows)) * dt
        color = COLORS[target]
        ax.plot(t, dbn, color=color, lw=1.8, label=f"DBN {LABELS[target]}")
        ax.plot(t, prior, color=color, lw=1.0, ls="--", alpha=0.55, label=f"CV prior {LABELS[target]}")
    ax.axhline(100.0, color="#8B1E3F", lw=1.0, ls=":")
    ax.text(0.99, 0.03, "100 m diagnostic gate", transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#8B1E3F")
    ax.set_title("DBN error is not better than the motion prior", loc="left", weight="bold")
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Position error (m)")
    ax.grid(True, ls="--", lw=0.5, alpha=0.45)
    ax.legend(ncol=2, fontsize=7, frameon=False, loc="upper left")


def panel_c(ax, sweep: list[dict], field: list[dict]) -> None:
    labels = [item["label"] for item in sweep] + [item["label"] for item in field]
    values = [float(item["mean_ospa_m"]) for item in sweep] + [float(item["mean_ospa_m"]) for item in field]
    colors = ["#8FA9B3"] * len(sweep) + ["#C08A9B"] * len(field)
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, edgecolor="none", height=0.62)
    ax.set_yticks(y, labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Mean order-2 OSPA (m)")
    ax.set_title("Existing runs remain diagnostic", loc="left", weight="bold")
    ax.grid(True, axis="x", ls="--", lw=0.5, alpha=0.45)
    for yi, value in zip(y, values):
        ax.text(value + max(values) * 0.015, yi, f"{value:.1f}", va="center", fontsize=7)
    ax.text(0.99, 0.02, "No field configuration passed admission", transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#8B1E3F")


def panel_d(ax) -> None:
    ax.axis("off")
    ax.text(0.0, 0.98, "What is still missing", transform=ax.transAxes, va="top", fontsize=10, weight="bold")
    text = (
        "Private field tracker / scripts\n"
        "Exact .wavfm channel mapping\n"
        "Packet-to-time synchronization\n"
        "Eq. (44)/(45) field initialization\n"
        "Frame overlap and covariance safeguards\n\n"
        "Status: synthetic core reproduced;\n"
        "field reproduction not admitted."
    )
    ax.text(0.0, 0.84, text, transform=ax.transAxes, va="top", fontsize=9, linespacing=1.55, color="#444444")
    ax.text(0.0, 0.16, "Purpose: advisor communication\nand author-protocol request", transform=ax.transAxes, va="top", fontsize=9, color="#8B1E3F")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--sweep-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    tracks = read_rows(args.baseline_root)
    baseline_manifest = load_manifest(args.baseline_manifest)
    sweep = json.loads(args.sweep_json.read_text(encoding="utf-8"))
    field = [
        {"label": "Field public init (5 frames)", "mean_ospa_m": 3717.76},
        {"label": "Field Eq.44 init (5 frames)", "mean_ospa_m": 9894.79},
        {"label": "Field Eq.44+45 init (5 frames)", "mean_ospa_m": 332.13},
        {"label": "Constant-velocity prior", "mean_ospa_m": 18.35},
    ]

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(11.2, 8.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.9], hspace=0.32, wspace=0.30)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_d = fig.add_subplot(grid[0, 1])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])
    panel_a(ax_a, tracks, "A  Paper-aligned three-target raw DBN smoke")
    panel_d(ax_d)
    panel_b(ax_b, tracks, baseline_manifest["sample_rate_hz"], baseline_manifest["snapshot_len"])
    panel_c(ax_c, sweep, field)
    fig.suptitle("Wideband DBN reproduction audit | diagnostic package", x=0.02, ha="left", fontsize=13, weight="bold")

    args.output_root.mkdir(parents=True, exist_ok=True)
    stem = args.output_root / "wideband_dbn_reproduction_audit_overview"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    registry = {
        "claim_status": "diagnostic_reproduction_audit",
        "figure": str(stem),
        "panels": {
            "A": {"role": "trajectory comparison", "source_root": str(args.baseline_root), "source_sha256": {f"target{t}": sha256(args.baseline_root / f"target{t}_dbn_lanm_baseline.csv") for t in (1, 2, 3)}},
            "B": {"role": "DBN versus constant-velocity prior error", "source_root": str(args.baseline_root)},
            "C": {"role": "diagnostic configuration summary", "source_json": str(args.sweep_json)},
            "D": {"role": "missing-protocol summary", "source": "WIDEBAND_DBN_REPRODUCTION_STATUS_20260823.md"},
        },
        "warning": "This figure is for advisor communication and audit only; it is not a formal superiority or paper-reproduction figure.",
    }
    (args.output_root / "wideband_dbn_reproduction_audit_overview_registry.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(registry, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
