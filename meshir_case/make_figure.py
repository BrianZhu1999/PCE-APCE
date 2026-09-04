#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from meshir.geometry import geometric_localization


def grid_plane(values: np.ndarray, positions: np.ndarray, z_value: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_unique = np.unique(np.round(positions[:, 2], 6))
    z = z_unique[np.argmin(np.abs(z_unique - z_value))]
    indices = np.flatnonzero(np.isclose(positions[:, 2], z))
    x_unique = np.unique(np.round(positions[indices, 0], 6))
    y_unique = np.unique(np.round(positions[indices, 1], 6))
    image = np.empty((len(y_unique), len(x_unique)), dtype=float)
    for index in indices:
        ix = int(np.argmin(np.abs(x_unique - positions[index, 0])))
        iy = int(np.argmin(np.abs(y_unique - positions[index, 1])))
        image[iy, ix] = values[index]
    return x_unique, y_unique, image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.root / "cache" / "geometry.npz") as geometry:
        s1_positions = geometry["s1_positions"]
        s32_positions = geometry["s32_positions"]
        s32_sources = geometry["s32_sources"]
    with np.load(args.root / "models" / "s1_fold0.npz") as model_s1:
        observed_s1 = model_s1["observed_indices"]
        heldout_s1 = model_s1["heldout_indices"]
    with np.load(args.root / "models" / "s32_fold0.npz") as model_s32:
        observed_s32 = model_s32["observed_indices"]
        candidate_positions = model_s32["candidate_source_positions"]
        test_sources = model_s32["test_source_indices"]
    s1_path = args.root / "smoke" / "s1" / "fold0" / "standard" / "apce" / "seed_00.npz"
    with np.load(s1_path) as payload:
        truth_snapshot = payload["truth_snapshots"][2]
        apce_snapshot = payload["snapshots"][2]
    s32_apce_path = args.root / "smoke3" / "s32" / "fold0" / "standard" / "apce" / "seed_00.npz"
    with np.load(s32_apce_path) as payload:
        weights = payload["final_weights"]
    true_source_index = int(test_sources[0])
    true_source = s32_sources[true_source_index]
    apce_source = np.sum(weights[:, None] * candidate_positions, axis=0)
    s32_rir = np.load(args.root / "cache" / "s32_rir_16k.npy", mmap_mode="r")
    toa_source, _ = geometric_localization(
        np.asarray(s32_rir[true_source_index, :320, observed_s32]),
        s32_positions[observed_s32], s32_sources, 343.0, 16000.0,
    )
    s1_frame = pd.read_csv(args.root / "aggregate" / "s1_run_source_data.csv")
    s32_frame = pd.read_csv(args.root / "aggregate" / "s32_run_source_data.csv")
    methods = ["DEnKF", "BMA", "PCE", "APCE"]
    colors = {"DEnKF": "#4c566a", "BMA": "#1f9e89", "PCE": "#2878b5", "APCE": "#d9772a"}
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.linewidth": 0.7, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(2, 3, wspace=0.42, hspace=0.48)
    axa = fig.add_subplot(gs[0, 0], projection="3d")
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[0, 2])
    axd = fig.add_subplot(gs[1, 0])
    axe = fig.add_subplot(gs[1, 1])
    axf = fig.add_subplot(gs[1, 2])
    axa.scatter(s1_positions[heldout_s1, 0], s1_positions[heldout_s1, 1], s1_positions[heldout_s1, 2], s=1.3, color="#c5cbd3", alpha=0.35)
    axa.scatter(s1_positions[observed_s1, 0], s1_positions[observed_s1, 1], s1_positions[observed_s1, 2], s=7, color="#d9772a")
    axa.set_xlabel("x (m)", labelpad=-1); axa.set_ylabel("y (m)", labelpad=-1); axa.set_zlabel("z (m)", labelpad=-1)
    axa.set_title("S1 sparse 3D observations", pad=3)
    for ax, letter in zip([axa, axb, axc, axd, axe, axf], "abcdef"):
        ax.text2D(-0.12, 1.04, letter, transform=ax.transAxes, fontsize=10, fontweight="bold") if hasattr(ax, "text2D") else ax.text(-0.12, 1.04, letter, transform=ax.transAxes, fontsize=10, fontweight="bold")
    x, y, truth_image = grid_plane(truth_snapshot, s1_positions)
    _, _, apce_image = grid_plane(apce_snapshot, s1_positions)
    vmax = np.quantile(np.abs(truth_image), 0.98)
    image_b = axb.imshow(truth_image, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    axb.set_title("measured, 120 ms", pad=3); axb.set_xlabel("x (m)"); axb.set_ylabel("y (m)")
    axc.imshow(apce_image, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    axc.set_title("APCE, 120 ms", pad=3); axc.set_xlabel("x (m)"); axc.set_ylabel("y (m)")
    fig.colorbar(image_b, ax=[axb, axc], fraction=0.035, pad=0.03, label="pressure")
    s1_standard = s1_frame[s1_frame.condition == "standard"].set_index("method")
    x_pos = np.arange(len(methods))
    axd.plot(x_pos, [s1_standard.loc[m, "reconstruction_nrmse"] for m in methods], "o-", color="#2878b5", lw=0.8, label="analysis")
    axd.plot(x_pos, [s1_standard.loc[m, "prediction_nrmse"] for m in methods], "s--", color="#d9772a", lw=0.8, label="64–200 ms")
    axd.axhline(1.0, color="#9ca3af", lw=0.6, ls=":")
    axd.set_xticks(x_pos); axd.set_xticklabels(methods, rotation=20)
    axd.set_ylabel("S1 nRMSE"); axd.legend(fontsize=6)
    axe.scatter(s32_sources[:, 0], s32_sources[:, 1], s=9, color="#c5cbd3", label="source grid")
    axe.scatter(candidate_positions[:, 0], candidate_positions[:, 1], s=12, color="#2878b5", alpha=0.5, label="candidate")
    axe.scatter(true_source[0], true_source[1], s=35, marker="*", color="black", label="held-out truth")
    axe.scatter(apce_source[0], apce_source[1], s=28, color="#d9772a", label="APCE")
    axe.scatter(toa_source[0], toa_source[1], s=28, marker="x", color="#1f9e89", label="TOA")
    axe.set_aspect("equal"); axe.set_xlabel("x (m)"); axe.set_ylabel("y (m)"); axe.legend(fontsize=5.5, ncol=2)
    s32_standard = s32_frame[s32_frame.condition == "standard"].set_index("method")
    errors = [s32_standard.loc[m, "localization_error_m"] for m in methods]
    axf.bar(x_pos, errors, color=[colors[m] for m in methods], width=0.62)
    axf.axhline(float(s32_standard.iloc[0]["geometric_baseline_error_m"]), color="#1f9e89", lw=1.0, ls="--", label="geometric TOA")
    axf.set_xticks(x_pos); axf.set_xticklabels(methods, rotation=20)
    axf.set_ylabel("S32 localization error (m)"); axf.legend(fontsize=6)
    output = args.root / "assets"; output.mkdir(parents=True, exist_ok=True)
    stem = output / "meshir_pre_admission_diagnostic"
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)
    source_rows = []
    for _, row in s1_frame.iterrows(): source_rows.append({"panel": "d", **row.to_dict()})
    for _, row in s32_frame.iterrows(): source_rows.append({"panel": "f", **row.to_dict()})
    source_rows.extend([
        {"panel": "e", "type": "truth", "x": float(true_source[0]), "y": float(true_source[1])},
        {"panel": "e", "type": "APCE", "x": float(apce_source[0]), "y": float(apce_source[1])},
        {"panel": "e", "type": "TOA", "x": float(toa_source[0]), "y": float(toa_source[1])},
    ])
    keys = sorted({key for row in source_rows for key in row})
    with (output / "meshir_pre_admission_figure_source_data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(source_rows)
    registry = {
        "figure": "meshir_pre_admission_diagnostic",
        "status": "pre-admission diagnostic; not released to CLEAN_MANUSCRIPT",
        "backend": "Python/matplotlib",
        "remote_result_root": str(args.root),
        "panels": {
            "a": {"source": "models/s1_fold0.npz", "role": "3D observed and held-out layout"},
            "b": {"source": "smoke/s1/fold0/standard/apce/seed_00.npz", "role": "measured mid-plane snapshot"},
            "c": {"source": "smoke/s1/fold0/standard/apce/seed_00.npz", "role": "APCE mid-plane snapshot"},
            "d": {"source": "aggregate/s1_run_source_data.csv", "role": "analysis and forward prediction error"},
            "e": {"source": "models/s32_fold0.npz and smoke3 S32 results", "role": "2D source localization geometry"},
            "f": {"source": "aggregate/s32_run_source_data.csv", "role": "localization comparison"}
        }
    }
    (output / "meshir_pre_admission_figure_registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(json.dumps({"figure": str(stem), "source_data": str(output / "meshir_pre_admission_figure_source_data.csv")}, indent=2))


if __name__ == "__main__":
    main()
