from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


METHODS = ("denkf", "pce", "apce")
COLORS = {
    "truth": "#1F1F1F",
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.5,
            "font.weight": "regular",
            "axes.titleweight": "regular",
            "axes.labelweight": "regular",
            "mathtext.default": "it",
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "legend.frameon": False,
        }
    )


def clean_axes(ax: plt.Axes, *, show_x: bool, show_y: bool) -> None:
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(labelsize=7.0, pad=1.4, length=2.0)
    if not show_x:
        ax.set_xticklabels([])
        ax.set_xlabel("")
    if not show_y:
        ax.set_yticklabels([])
        ax.set_ylabel("")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wave-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    set_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.wave_npz, allow_pickle=True)

    states = data["truth_states"]
    nx = states.shape[1] // 2
    times = data["times"]
    field_items = [
        ("Truth", data["truth_states"][:, :nx]),
        ("DEnKF", data["denkf_mean_states"][:, :nx]),
        ("PCE", data["pce_mean_states"][:, :nx]),
        ("APCE", data["apce_mean_states"][:, :nx]),
    ]
    error_items = [
        ("DEnKF error", field_items[1][1] - field_items[0][1]),
        ("PCE error", field_items[2][1] - field_items[0][1]),
        ("APCE error", field_items[3][1] - field_items[0][1]),
        ("PCE - APCE", field_items[2][1] - field_items[3][1]),
    ]

    vmax = max(float(np.nanmax(np.abs(field))) for _, field in field_items)
    emax = max(float(np.nanmax(np.abs(err))) for _, err in error_items[:3])
    diff_emax = max(float(np.nanmax(np.abs(error_items[3][1]))), 1.0e-12)

    fig = plt.figure(figsize=(7.35, 3.05))
    grid = GridSpec(
        2,
        7,
        figure=fig,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 0.050, 0.095, 0.050],
        height_ratios=[1.0, 1.0],
        left=0.060,
        right=0.970,
        top=0.900,
        bottom=0.155,
        wspace=0.155,
        hspace=0.275,
    )

    extent = [float(times[0]), float(times[-1]), 0.0, 1.0]

    top_images = []
    for col, (title, field) in enumerate(field_items):
        ax = fig.add_subplot(grid[0, col])
        image = ax.imshow(
            field.T,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="bilinear",
        )
        top_images.append(image)
        ax.set_title(title, fontsize=7.5, pad=2.0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_xticks([0, 0.5, 1.0])
        if col == 0:
            ax.set_ylabel(r"$x$", fontsize=7.5)
        clean_axes(ax, show_x=False, show_y=col == 0)

    cax_top = fig.add_subplot(grid[0, 4])
    cbar_top = fig.colorbar(top_images[0], cax=cax_top)
    cax_top.set_title(r"$u$", fontsize=7.0, pad=3.0)
    cbar_top.ax.tick_params(labelsize=6.4, width=0.45, length=2.0, pad=1.5)

    bottom_images = []
    for col, (title, error) in enumerate(error_items):
        ax = fig.add_subplot(grid[1, col])
        local_emax = diff_emax if col == 3 else emax
        cmap = "coolwarm" if col == 3 else "PuOr_r"
        image = ax.imshow(
            error.T,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap,
            vmin=-local_emax,
            vmax=local_emax,
            interpolation="bilinear",
        )
        bottom_images.append(image)
        ax.set_title(title, fontsize=7.5, pad=2.0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_xlabel(r"$t$", fontsize=7.5)
        if col == 0:
            ax.set_ylabel(r"$x$", fontsize=7.5)
        clean_axes(ax, show_x=True, show_y=col == 0)

    cax_error = fig.add_subplot(grid[1, 4])
    cbar_error = fig.colorbar(bottom_images[2], cax=cax_error)
    cax_error.set_title("error", fontsize=7.0, pad=3.0)
    cbar_error.ax.tick_params(labelsize=6.4, width=0.45, length=2.0, pad=1.5)

    cax_diff = fig.add_subplot(grid[1, 6])
    cbar_diff = fig.colorbar(bottom_images[3], cax=cax_diff)
    cax_error.set_title("err.", fontsize=7.0, pad=3.0)
    cax_diff.set_title(r"$\Delta$", fontsize=7.0, pad=3.0)
    cbar_diff.ax.tick_params(labelsize=6.4, width=0.45, length=2.0, pad=1.5)

    base = args.output_dir / "supp_figure_wave_displacement"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(base)


if __name__ == "__main__":
    main()
