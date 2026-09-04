#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    method_names = ["DEnKF", "BMA", "PCE", "APCE"]
    colors = {"DEnKF": "#4c566a", "BMA": "#1f9e89", "PCE": "#2878b5", "APCE": "#d9772a"}
    records = {name: json.loads((args.smoke / f"{name.lower()}.json").read_text(encoding="utf-8")) for name in method_names}
    baseline = json.loads((args.smoke / "trilinear_baseline.json").read_text(encoding="utf-8"))
    with np.load(args.smoke / "apce.npz") as payload:
        truth = payload["truth_snapshots"]
        mean = payload["mean_snapshots"]
        positions = payload["positions"]
        observed = payload["observed_indices"]
        weights = payload["final_weights"]
    snapshot = 1
    plane = 4
    truth_plane = truth[snapshot, plane]
    mean_plane = mean[snapshot, plane]
    error_plane = mean_plane - truth_plane
    vmax = float(np.quantile(np.abs(truth_plane), 0.99))
    error_max = float(np.quantile(np.abs(error_plane), 0.99))
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.linewidth": 0.7, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.2, 5.1))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.05, 1, 1, 1], wspace=0.43, hspace=0.48)
    axa = fig.add_subplot(gs[0, 0], projection="3d")
    axb = fig.add_subplot(gs[0, 1]); axc = fig.add_subplot(gs[0, 2]); axd = fig.add_subplot(gs[0, 3])
    axe = fig.add_subplot(gs[1, 0:2]); axf = fig.add_subplot(gs[1, 2]); axg = fig.add_subplot(gs[1, 3])
    axa.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=1.2, color="#c5cbd3", alpha=0.22)
    axa.scatter(positions[observed, 0], positions[observed, 1], positions[observed, 2], s=7, color="#d9772a")
    axa.set_xlabel("x (m)", labelpad=-1); axa.set_ylabel("y (m)", labelpad=-1); axa.set_zlabel("z (m)", labelpad=-1)
    axa.set_title("7×7×3 observations", pad=2)
    im = axb.imshow(truth_plane, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=[-0.5,0.5,-0.5,0.5])
    axb.set_title("measured, 40 ms", pad=2); axb.set_xlabel("x (m)"); axb.set_ylabel("y (m)")
    axc.imshow(mean_plane, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=[-0.5,0.5,-0.5,0.5])
    axc.set_title("APCE, 40 ms", pad=2); axc.set_xlabel("x (m)"); axc.set_ylabel("y (m)")
    axd.imshow(error_plane, origin="lower", cmap="RdBu_r", vmin=-error_max, vmax=error_max, extent=[-0.5,0.5,-0.5,0.5])
    axd.set_title(f"APCE error (±{error_max:.1e})", pad=2); axd.set_xlabel("x (m)"); axd.set_ylabel("y (m)")
    analysis_values = [baseline["analysis_nrmse"]] + [records[name]["analysis_nrmse"] for name in method_names]
    labels = ["trilinear"] + method_names
    axe.bar(np.arange(len(labels)), analysis_values, color=["#9ca3af"] + [colors[name] for name in method_names], width=0.62)
    axe.set_xticks(np.arange(len(labels))); axe.set_xticklabels(labels, rotation=18)
    axe.set_ylabel("analysis nRMSE")
    horizons = np.asarray([1, 2, 4])
    for name in method_names:
        values = [records[name][f"forecast_{ms}ms_nrmse"] for ms in horizons]
        axf.plot(horizons, values, marker="o", lw=0.9, color=colors[name], label=name)
    axf.set_xlabel("forecast horizon (ms)"); axf.set_ylabel("forecast nRMSE"); axf.set_xticks(horizons)
    axf.legend(fontsize=5.5)
    speeds = np.asarray(records["APCE"]["candidate_speed_m_s"])
    axg.bar(speeds, weights, width=2.8, color="#d9772a")
    axg.set_xlabel("sound speed (m s$^{-1}$)"); axg.set_ylabel("APCE final weight")
    axes = [axa, axb, axc, axd, axe, axf, axg]
    for ax, letter in zip(axes, "abcdefg"):
        label_x = -0.21 if letter == "d" else -0.13
        if hasattr(ax, "text2D"):
            ax.text2D(label_x, 1.04, letter, transform=ax.transAxes, fontsize=10, fontweight="bold")
        else:
            ax.text(label_x, 1.04, letter, transform=ax.transAxes, fontsize=10, fontweight="bold")
    stem = args.output / "s1_full_dynamics_smoke"
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)
    rows = [{"method": "trilinear", **baseline}] + [{"method": name, **records[name]} for name in method_names]
    keys = sorted({key for row in rows for key, value in row.items() if not isinstance(value, (list, dict))})
    with (args.output / "s1_full_dynamics_figure_source_data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)
    registry = {
        "figure": "s1_full_dynamics_smoke",
        "status": "single-seed full-grid smoke; not released to CLEAN_MANUSCRIPT",
        "backend": "Python/matplotlib",
        "remote_authoritative_bundle": "<HILDA_RESULTS_ROOT>/experiments/meshir_s1_full_dynamics_20260819/",
        "panels": {
            "a": {"source": "smoke_aligned/apce.npz", "role": "full 21x21x9 grid and 7x7x3 observations"},
            "b": {"source": "smoke_aligned/apce.npz", "role": "measured z=0 plane at 40 ms"},
            "c": {"source": "smoke_aligned/apce.npz", "role": "APCE z=0 plane at 40 ms"},
            "d": {"source": "smoke_aligned/apce.npz", "role": "APCE error plane"},
            "e": {"source": "smoke_aligned/*.json", "role": "analysis-window reconstruction comparison"},
            "f": {"source": "smoke_aligned/*.json", "role": "1/2/4-ms free forecast"},
            "g": {"source": "smoke_aligned/apce.json", "role": "final candidate-speed weights"}
        }
    }
    (args.output / "s1_full_dynamics_figure_registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(json.dumps({"figure": str(stem)}, indent=2))


if __name__ == "__main__":
    main()
