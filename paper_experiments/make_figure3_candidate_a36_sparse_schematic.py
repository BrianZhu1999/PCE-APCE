from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle


OUT_DIR = Path(r"figures")
OUT_STEM = "figure3_candidate_A36_sparse_observation_schematic_v1"

PANEL_LABEL_SIZE = 26
LEGEND_SIZE = 18
AXIS_LABEL_SIZE = 16

plt.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 600,
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def box(ax, xy, w, h, text, fc, ec):
    patch = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.018,rounding_size=0.028",
                           facecolor=fc, edgecolor=ec, linewidth=1.5)
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=LEGEND_SIZE, color="#111111", linespacing=1.05)


def arrow(ax, p0, p1, color="#222222", lw=1.8):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=18,
                                 linewidth=lw, color=color, shrinkA=6, shrinkB=6))


def main() -> None:
    fig = plt.figure(figsize=(15.56, 3.20), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.text(0.006, 0.875, "d", fontsize=PANEL_LABEL_SIZE, fontweight="bold", ha="left", va="top")

    # Left: sparse observations.
    xs = np.linspace(0.08, 0.32, 220)
    y = 0.63 + 0.08 * np.sin(22 * xs) + 0.035 * np.sin(67 * xs)
    ax.plot(xs, y, color="#303030", lw=2.2, alpha=0.88)
    obs_idx = np.linspace(15, 205, 7).astype(int)
    ax.scatter(xs[obs_idx], y[obs_idx], s=62, facecolor="#f5f0df", edgecolor="#303030", linewidth=1.5, zorder=5)
    for xi, yi in zip(xs[obs_idx], y[obs_idx]):
        ax.plot([xi, xi], [0.36, yi], color="#c9bfa8", lw=1.0, alpha=0.55)
    ax.text(0.20, 0.245, "Sparse observations", fontsize=LEGEND_SIZE, ha="center")

    box(ax, (0.405, 0.56), 0.165, 0.170, "State--parameter\ncovariance update", "#eef3fa", "#4e79a7")
    box(ax, (0.405, 0.265), 0.165, 0.170, "Path evidence\naccumulation", "#fff4e5", "#ff8c00")

    arrow(ax, (0.325, 0.61), (0.405, 0.645), "#4e79a7")
    arrow(ax, (0.325, 0.55), (0.405, 0.350), "#ff8c00")

    # Right: consequence, stylized but not data-bearing.
    ax.plot([0.655, 0.905], [0.69, 0.58], color="#4e79a7", lw=3.0, alpha=0.85)
    ax.plot([0.655, 0.905], [0.43, 0.31], color="#9b75b6", lw=3.0, alpha=0.65)
    ax.plot([0.655, 0.905], [0.67, 0.74], color="#55b7e8", lw=3.2, alpha=0.95)
    ax.plot([0.655, 0.905], [0.64, 0.75], color="#ff8c00", lw=3.2, alpha=0.95)
    for x in [0.655, 0.905]:
        ax.add_patch(Circle((x, 0.67 if x == 0.655 else 0.74), 0.012, fc="white", ec="#55b7e8", lw=1.5))
        ax.add_patch(Circle((x, 0.64 if x == 0.655 else 0.75), 0.012, fc="white", ec="#ff8c00", lw=1.5))
    ax.text(0.780, 0.815, "Sparse-observation\nadvantage", fontsize=LEGEND_SIZE, ha="center", linespacing=1.05)
    ax.text(0.780, 0.205, "Prediction evidence remains comparable\nwhen direct joint updates weaken",
            fontsize=AXIS_LABEL_SIZE, ha="center", linespacing=1.05, color="#333333")
    arrow(ax, (0.570, 0.645), (0.650, 0.690), "#4e79a7")
    arrow(ax, (0.570, 0.350), (0.650, 0.650), "#ff8c00")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in [("png", {"dpi": 600}), ("pdf", {}), ("svg", {}), ("tiff", {"dpi": 600})]:
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", facecolor="white", bbox_inches=None, pad_inches=0, **kwargs)
    plt.close(fig)
    print(OUT_DIR / f"{OUT_STEM}.png")


if __name__ == "__main__":
    main()
