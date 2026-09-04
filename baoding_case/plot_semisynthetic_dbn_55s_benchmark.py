#!/usr/bin/env python3
"""Plot the 55-second semi-synthetic DBN-like Baoding benchmark."""
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


COLORS = {1: "#176B87", 2: "#D97941", 3: "#4D8B59"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    tracks = {target: read_csv(args.root / f"target{target}/frontend/dbn_track.csv") for target in (1, 2, 3)}
    metric_rows = read_csv(args.summary_csv)
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"], "font.size": 8, "axes.titlesize": 10, "axes.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(10.4, 7.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.1, 0.9])
    ax_track = fig.add_subplot(grid[0, :])
    ax_error = fig.add_subplot(grid[1, 0])
    ax_metric = fig.add_subplot(grid[1, 1])

    for target in (1, 2, 3):
        rows = tracks[target]
        truth_x = np.asarray([float(row["truth_x"]) for row in rows])
        truth_y = np.asarray([float(row["truth_y"]) for row in rows])
        estimate_x = np.asarray([float(row["px"]) for row in rows])
        estimate_y = np.asarray([float(row["py"]) for row in rows])
        errors = np.asarray([float(row["position_error_m"]) for row in rows])
        time_s = np.arange(len(rows))
        color = COLORS[target]
        ax_track.plot(truth_x, truth_y, color=color, lw=2.0, alpha=0.45, label=f"GPS T{target}")
        ax_track.plot(estimate_x, estimate_y, color=color, lw=1.5, marker="o", markersize=2.4, markevery=5, label=f"DBN-like T{target}")
        ax_track.scatter(truth_x[0], truth_y[0], marker="s", s=28, color="black", zorder=5)
        ax_track.scatter(truth_x[-1], truth_y[-1], marker="*", s=46, color="black", zorder=5)
        ax_error.plot(time_s, errors, color=color, lw=1.7, label=f"T{target}")
    ax_track.set_title("Real Baoding motion with DBN-residual-calibrated trajectories", loc="left", weight="bold")
    ax_track.set_xlabel("projected x (m)"); ax_track.set_ylabel("projected y (m)")
    ax_track.grid(True, ls="--", lw=0.5, alpha=0.45); ax_track.set_aspect("equal", adjustable="datalim")
    ax_track.legend(frameon=False, ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax_error.set_title("DBN-like position error", loc="left", weight="bold")
    ax_error.set_xlabel("time (s)"); ax_error.set_ylabel("position error (m)")
    ax_error.grid(True, ls="--", lw=0.5, alpha=0.45); ax_error.legend(frameon=False, fontsize=7)

    labels, means, stds, colors = [], [], [], []
    for target in (1, 2, 3):
        for method in ("pce", "apce"):
            values = [float(row["rmse_m"]) for row in metric_rows if int(row["target"]) == target and row["method"] == method]
            labels.append(f"T{target} {method.upper()}"); means.append(float(np.mean(values))); stds.append(float(np.std(values))); colors.append("#8FA9B3" if method == "pce" else "#B9899B")
    y = np.arange(len(labels))
    ax_metric.barh(y, means, xerr=stds, color=colors, height=0.65, capsize=2)
    ax_metric.set_yticks(y, labels); ax_metric.invert_yaxis(); ax_metric.set_xlabel("position RMSE (m)")
    ax_metric.set_title("Five-seed downstream tracking", loc="left", weight="bold")
    ax_metric.grid(True, axis="x", ls="--", lw=0.5, alpha=0.45)
    for yi, value in zip(y, means): ax_metric.text(value + 1.0, yi, f"{value:.1f}", va="center", fontsize=7)

    fig.suptitle("55-second Baoding DBN-like benchmark for PCE/APCE", x=0.02, ha="left", fontsize=13, weight="bold")
    args.output_root.mkdir(parents=True, exist_ok=True)
    stem = args.output_root / "semisynthetic_dbn_55s_benchmark"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    registry = {"claim_status": "semisynthetic_dbn_55s_benchmark_figure", "root": str(args.root), "track_sha256": {str(target): sha256(args.root / f"target{target}/frontend/dbn_track.csv") for target in (1, 2, 3)}, "summary_csv": str(args.summary_csv), "summary_sha256": sha256(args.summary_csv), "outputs": {suffix: str(stem.with_suffix(f".{suffix}")) for suffix in ("png", "pdf", "svg")}, "warning": "Real GPS motion plus residual-calibrated DBN-like error process; not an independent acoustic result."}
    (args.output_root / "semisynthetic_dbn_55s_benchmark_registry.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(registry, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
