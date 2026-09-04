"""Plot the mask-aware full-field VIV-PIV sensor layout."""
from __future__ import annotations

import argparse
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .io import VIVCase, locate_case


def save_publication_figure(fig: plt.Figure, output: pathlib.Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", facecolor="white")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the adaptive VIV-PIV sensor layout.")
    parser.add_argument("--data-root", type=pathlib.Path, required=True)
    parser.add_argument("--sensor-archive", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--case", default="0432")
    args = parser.parse_args()

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })

    case = VIVCase.open(locate_case(args.data_root, args.case))
    sensor = np.load(args.sensor_archive, allow_pickle=False)
    actual = np.asarray(sensor["sensor_coordinates_mm"], dtype=float) / 50.0
    x = np.asarray(case.x_mm, dtype=float) / 50.0
    y = np.asarray(case.y_mm, dtype=float) / 50.0
    mask = np.asarray(case.mask[0] > 0.5, dtype=bool)

    target_x = np.linspace(float(x.min()), float(x.max()), 20)
    target_y = np.linspace(float(y.min()), float(y.max()), 40)
    all_ideal = np.asarray([(xx, yy) for xx in target_x for yy in target_y], dtype=float)
    ideal = np.asarray(sensor["ideal_sensor_coordinates_mm"], dtype=float) / 50.0
    nearest_retained = np.min(np.linalg.norm(all_ideal[:, None, :] - ideal[None, :, :], axis=2), axis=1)
    dropped = nearest_retained > 1e-4

    fig = plt.figure(figsize=(7.15, 2.85), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.75, 1.0))
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    blue = "#2E6F9E"
    light = "#C7CDD3"
    accent = "#C44E52"

    for ax in axes:
        ax.contourf(x, y, (~mask).astype(float), levels=[0.5, 1.5], colors=["#E3E5E8"], zorder=0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$x/D$")
        ax.set_ylabel(r"$y/D$")
        ax.tick_params(length=2.5, width=0.7)

    axes[0].scatter(actual[:, 0], actual[:, 1], s=5.0, color=blue, linewidths=0, rasterized=True)
    axes[0].set_xlim(float(x.min()) - 0.12, float(x.max()) + 0.12)
    axes[0].set_ylim(float(y.min()) - 0.12, float(y.max()) + 0.12)
    axes[0].text(0.02, 0.98, "a", transform=axes[0].transAxes, va="top", ha="left", fontweight="bold", fontsize=9)
    axes[0].text(0.98, 0.03, f"{actual.shape[0]} valid spatial points\n{2 * actual.shape[0]:,} scalar observations",
                 transform=axes[0].transAxes, va="bottom", ha="right", color="#333333")

    axes[1].scatter(all_ideal[:, 0], all_ideal[:, 1], s=8, facecolors="none", edgecolors=light,
                    linewidths=0.65, label="ideal lattice")
    axes[1].scatter(actual[:, 0], actual[:, 1], s=8, color=blue, linewidths=0,
                    label="mask-aware layout")
    axes[1].scatter(all_ideal[dropped, 0], all_ideal[dropped, 1], s=16, marker="x",
                    color=accent, linewidths=0.8, zorder=4, label="masked point removed")
    axes[1].set_xlim(-1.7, 1.3)
    axes[1].set_ylim(-1.55, 1.55)
    axes[1].text(0.02, 0.98, "b", transform=axes[1].transAxes, va="top", ha="left", fontweight="bold", fontsize=9)
    axes[1].text(0.98, 0.03, f"{int(dropped.sum())} masked points removed",
                 transform=axes[1].transAxes, va="bottom", ha="right", color=accent)
    axes[1].legend(loc="upper right", handletextpad=0.45, borderaxespad=0.4, labelspacing=0.35)

    save_publication_figure(fig, args.output)
    print(args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
