"""Generate report-only audit figures for the VIV-PIV case."""
from __future__ import annotations

import pathlib

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


OUT = pathlib.Path(__file__).resolve().parent / "report_assets"
OUT.mkdir(parents=True, exist_ok=True)
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 7,
    "axes.linewidth": 0.7,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def export(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=500, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def setup() -> None:
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    ax.set_xlim(-0.15, 2.25)
    ax.set_ylim(-0.25, 1.20)
    ax.axis("off")
    # Water-canal side view. Dimensions are stated in the report from the
    # official metadata and associated experimental description.
    ax.add_patch(Rectangle((0.0, 0.0), 2.1, 0.85, facecolor="#CFE8F3", edgecolor="#397A9B", lw=0.9))
    ax.add_patch(Rectangle((0.0, 0.0), 2.1, 0.04, facecolor="#B7C2C9", edgecolor="#59666D", lw=0.7))
    ax.plot([0.0, 2.1], [0.85, 0.85], color="#397A9B", lw=0.8)
    ax.text(1.05, 0.91, "water canal: 2.1 m x 0.51 m x 0.51 m", ha="center", va="bottom")
    ax.add_patch(Rectangle((0.93, 0.04), 0.10, 0.80, facecolor="#56616A", edgecolor="#20252A", lw=0.8))
    ax.text(1.08, 0.76, "cylinder\nL=0.4 m", ha="left", va="center")
    ax.add_patch(Rectangle((0.74, 0.84), 0.48, 0.08, facecolor="#606A70", edgecolor="#20252A", lw=0.7))
    ax.add_patch(Rectangle((0.84, 0.92), 0.28, 0.16, facecolor="#8B9BA4", edgecolor="#20252A", lw=0.7))
    ax.text(1.23, 1.05, "elastic support / moving platform", ha="left", va="center")
    ax.add_patch(Circle((0.82, 0.58), 0.045, facecolor="#E77A55", edgecolor="#5C2D21", lw=0.6))
    ax.text(0.74, 0.68, "laser", ha="center", va="bottom", color="#8A2A1D")
    ax.plot([0.08, 1.75], [0.45, 0.45], color="#43A047", lw=2.0, alpha=0.9)
    ax.text(1.77, 0.45, "PIV laser sheet / mid-height plane", ha="left", va="center", color="#2E7D32")
    ax.add_patch(FancyArrowPatch((0.15, 0.68), (0.65, 0.68), arrowstyle="-|>", mutation_scale=10, lw=1.0, color="#1D4E89"))
    ax.text(0.16, 0.73, r"$U_\infty$", color="#1D4E89", ha="left")
    ax.add_patch(FancyArrowPatch((0.98, 0.98), (0.98, 1.13), arrowstyle="<->", mutation_scale=8, lw=0.8, color="#555555"))
    ax.text(1.03, 1.13, r"$\pm 2D$", va="center", color="#555555")
    ax.add_patch(Rectangle((0.92, 0.03), 0.12, 0.86, fill=False, edgecolor="#FFFFFF", lw=1.3, linestyle="--"))
    ax.text(1.55, 0.15, "CCD laser displacement sensor\n(Keyence LK-G507)", ha="center", va="center")
    ax.annotate("D = 0.05 m", xy=(0.98, 0.22), xytext=(0.48, 0.18), arrowprops={"arrowstyle": "-", "lw": 0.7}, ha="center")
    ax.text(0.02, -0.10, "PIV: two-component, two-dimensional, 10 Hz; two lasers + two CMOS cameras", ha="left", va="top")
    export(fig, "report_experimental_setup_schematic")


def sensor_layout() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    x = np.linspace(1.0, 8.0, 20)
    y = np.linspace(-2.0, 2.0, 40)
    xx, yy = np.meshgrid(x, y)
    ax.scatter(xx.ravel(), yy.ravel(), s=7, facecolors="white", edgecolors="#2C7FB8", linewidths=0.35, label="800 sparse PIV locations")
    ax.add_patch(Circle((0.0, 0.0), 0.5, facecolor="#444444", edgecolor="#111111", lw=0.8, zorder=3))
    ax.add_patch(Circle((0.0, 0.0), 0.5, facecolor="white", edgecolor="#111111", lw=0.8, alpha=0.75, zorder=4))
    ax.set_xlim(-1.7, 8.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x/D$")
    ax.set_ylabel(r"$y/D$")
    ax.set_title("Final 20 x 40 sparse observation layout", loc="left", pad=4)
    ax.text(0.02, 0.04, "D = 0.05 m; u and v at every point\n= 1,600 scalar observations", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8})
    ax.legend(loc="upper right", frameon=False, fontsize=6.5)
    ax.grid(color="#E0E0E0", lw=0.4)
    export(fig, "report_final_sensor_layout_20x40")


def provenance() -> None:
    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    boxes = [
        (0.2, 1.2, 1.6, 0.8, "17 NPZ cases\nDOI + SHA-256", "#D9EAF7"),
        (2.25, 1.2, 1.7, 0.8, "12 train / 5\nexternal tests", "#E6F2E1"),
        (4.45, 1.2, 1.6, 0.8, "POD r=256\n99.960% energy", "#FFF1CC"),
        (6.55, 1.2, 1.7, 0.8, "12 DMDc\ncandidates", "#FCE1D8"),
        (8.75, 1.2, 1.0, 0.8, "PCE /\nAPCE", "#E5DDF6"),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="#3C3C3C", lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7)
    for i in range(len(boxes) - 1):
        x, y, w, h, *_ = boxes[i]
        nx, ny, *_ = boxes[i + 1]
        ax.add_patch(FancyArrowPatch((x + w, y + h / 2), (nx, ny + h / 2), arrowstyle="-|>", mutation_scale=10, lw=0.9, color="#555555"))
    ax.text(5.0, 0.72, "held-out evaluation: sparse PIV evidence -> wake reconstruction, uncertainty metrics and known-input blackout forecast", ha="center", va="center", fontsize=7)
    ax.text(5.0, 2.45, "Read-only raw data; only manifests, reduced summaries, traces and figures are written locally", ha="center", va="center", fontsize=7, color="#444444")
    export(fig, "report_provenance_and_leakage_flow")


if __name__ == "__main__":
    setup()
    sensor_layout()
    provenance()
