"""Figure 1a: hybrid uncertainty under sparse sensing.

The panel is an illustrative schematic, not an experimental result.  The
left field shows a smooth latent wave-like field with sparse sensors.  The two
right-hand plots separate stochastic variation within one candidate path from
separation between candidate cognitive paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures"
STEM = OUT_DIR / "figure1_panel_a_sparse_uncertainty"

INK = "#1F2937"
MUTED = "#667085"
GRID = "#D8DEE8"
BLUE = "#2C6AA6"
BLUE_LIGHT = "#E7F0FB"
VIOLET = "#6858A6"
VIOLET_LIGHT = "#F0ECFA"
ORANGE = "#D97724"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.2,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def field_data(n: int = 220) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(x, y)
    z = (
        0.95 * np.sin(2.15 * np.pi * (yy + 0.13 * np.sin(2.0 * np.pi * xx)))
        + 0.20 * np.cos(2.7 * np.pi * xx - 1.2 * yy)
        + 0.08 * np.sin(5.0 * np.pi * xx + 2.5 * yy)
    )
    z = (z - z.mean()) / np.max(np.abs(z - z.mean()))
    return xx, yy, z


def rounded_box(ax: plt.Axes, x: float, y: float, width: float, height: float, *, face: str, edge: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.035",
            linewidth=0.9,
            edgecolor=edge,
            facecolor=face,
            transform=ax.transAxes,
            zorder=0,
        )
    )


def draw_field(ax: plt.Axes) -> None:
    _, _, z = field_data()
    ax.imshow(z, origin="lower", extent=(0, 1, 0, 1), cmap="RdYlBu_r", vmin=-1, vmax=1, interpolation="bilinear", zorder=1)
    levels = np.linspace(-0.8, 0.8, 9)
    ax.contour(z, levels=levels, extent=(0, 1, 0, 1), colors="white", linewidths=0.30, alpha=0.60, zorder=2)

    # Flowing curves make the field structure legible without implying data.
    t = np.linspace(0.02, 0.98, 260)
    for offset in np.linspace(0.12, 0.88, 6):
        path = offset + 0.085 * np.sin(2 * np.pi * (t + 0.22 * offset))
        ax.plot(t, path, color="white", lw=0.55, alpha=0.70, zorder=3)

    sensor_xy = np.array(
        [
            [0.14, 0.20],
            [0.30, 0.71],
            [0.47, 0.38],
            [0.63, 0.79],
            [0.78, 0.29],
            [0.86, 0.61],
            [0.53, 0.16],
        ]
    )
    ax.scatter(sensor_xy[:, 0], sensor_xy[:, 1], s=42, facecolor="white", edgecolor=INK, linewidth=0.75, zorder=5)
    ax.text(0.035, 0.955, "latent field", transform=ax.transAxes, color=INK, fontsize=7.4, fontweight="bold", va="top", ha="left", bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.82))
    ax.text(0.50, -0.065, "sparse observations", transform=ax.transAxes, color=INK, fontsize=7.0, ha="center", va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_within_path(ax: plt.Axes) -> None:
    rounded_box(ax, 0.0, 0.0, 1.0, 1.0, face=BLUE_LIGHT, edge=BLUE)
    ax.text(0.08, 0.86, "within-path stochasticity", transform=ax.transAxes, color=INK, fontsize=7.3, fontweight="bold", ha="left", va="top")
    inner = ax.inset_axes([0.10, 0.20, 0.82, 0.50])
    x = np.linspace(0.0, 1.0, 170)
    for member in range(7):
        phase = 0.10 * member
        y = 0.50 + 0.20 * np.sin(2 * np.pi * x + phase) + 0.038 * np.sin(7 * np.pi * x + 0.6 * member)
        inner.plot(x, y, color=BLUE, lw=0.85, alpha=0.38 + 0.07 * (member == 3))
    inner.plot(x, 0.50 + 0.20 * np.sin(2 * np.pi * x + 0.30), color=BLUE, lw=1.35, alpha=0.95)
    inner.text(0.02, 0.96, r"same candidate $\alpha_k$", transform=inner.transAxes, color=BLUE, fontsize=6.4, ha="left", va="top")
    inner.set_xlim(0, 1)
    inner.set_ylim(0.18, 0.82)
    inner.set_xticks([])
    inner.set_yticks([])
    for spine in inner.spines.values():
        spine.set_visible(False)
    ax.text(0.50, 0.08, r"stochastic forcing $W_t$", transform=ax.transAxes, color=MUTED, fontsize=6.4, ha="center", va="center")
    ax.set_axis_off()


def draw_between_path(ax: plt.Axes) -> None:
    rounded_box(ax, 0.0, 0.0, 1.0, 1.0, face=VIOLET_LIGHT, edge=VIOLET)
    ax.text(0.08, 0.86, "between-path cognitive uncertainty", transform=ax.transAxes, color=INK, fontsize=7.3, fontweight="bold", ha="left", va="top")
    inner = ax.inset_axes([0.10, 0.20, 0.82, 0.50])
    x = np.linspace(0.0, 1.0, 170)
    paths = [
        (0.36, VIOLET, r"$\alpha_1$"),
        (0.47, "#8D7CC2", r"$\alpha_2$"),
        (0.58, "#A99DD2", r"$\alpha_3$"),
        (0.69, ORANGE, r"$\alpha_4$"),
    ]
    for idx, (base, color, label) in enumerate(paths):
        y = base + 0.055 * np.sin(2 * np.pi * x + 0.20 * idx)
        inner.plot(x, y, color=color, lw=1.10 if idx in (0, 3) else 0.90, alpha=0.92)
        inner.text(1.01, y[-1], label, transform=inner.transData, color=color, fontsize=6.2, va="center", ha="left")
    inner.text(0.02, 0.96, r"candidate equations $F_{\alpha_k}$", transform=inner.transAxes, color=VIOLET, fontsize=6.4, ha="left", va="top")
    inner.set_xlim(0, 1.12)
    inner.set_ylim(0.22, 0.82)
    inner.set_xticks([])
    inner.set_yticks([])
    for spine in inner.spines.values():
        spine.set_visible(False)
    ax.text(0.50, 0.08, "distinct candidate paths", transform=ax.transAxes, color=MUTED, fontsize=6.4, ha="center", va="center")
    ax.set_axis_off()


def main() -> None:
    configure_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(5.25, 3.55), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 0.88], wspace=0.16, left=0.07, right=0.98, top=0.84, bottom=0.13)
    fig.text(0.07, 0.955, "a", fontsize=11.0, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.12, 0.955, "Hybrid uncertainty under sparse sensing", fontsize=10.2, fontweight="bold", color=INK, ha="left", va="top")

    field_ax = fig.add_subplot(gs[0, 0])
    draw_field(field_ax)

    right = gs[0, 1].subgridspec(2, 1, hspace=0.20, height_ratios=[1, 1])
    draw_within_path(fig.add_subplot(right[0, 0]))
    draw_between_path(fig.add_subplot(right[1, 0]))

    legend_handles = [
        Line2D([0], [0], color=BLUE, lw=1.8, label="within-path stochasticity"),
        Line2D([0], [0], color=VIOLET, lw=1.8, label="between-path cognitive uncertainty"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.57, 0.015), ncol=2, fontsize=6.5, handlelength=1.5, columnspacing=1.1, frameon=False)

    for suffix, kwargs in ((".svg", {}), (".pdf", {}), (".png", {"dpi": 600}), (".tiff", {"dpi": 600})):
        fig.savefig(str(STEM) + suffix, bbox_inches="tight", **kwargs)
    plt.close(fig)

    provenance = {
        "figure": "Figure 1a",
        "version": "panel_a_sparse_uncertainty_v1",
        "source_script": "make_figure1_panel_a.py",
        "backend": "Python / matplotlib",
        "archetype": "schematic-led composite",
        "scientific_data_status": "illustrative schematic; no experimental data are plotted",
        "core_conclusion": "Sparse sensing mixes within-path stochastic variability with between-path cognitive uncertainty.",
        "field_source": "deterministic analytic wave-like field generated in the source script",
        "sensor_source": "fixed illustrative coordinates generated in the source script",
        "seed_or_sample_selection": "not applicable; deterministic field and fixed coordinates",
        "outputs": [str(STEM.with_suffix(suffix)) for suffix in (".svg", ".pdf", ".png", ".tiff")],
    }
    (STEM.with_name(STEM.name + "_provenance.json")).write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    qa = {
        "text_audit": {
            "forbidden_placeholder_terms": [],
            "forbidden_generation_errors": [],
            "terminology": ["within-path stochasticity", "between-path cognitive uncertainty", "candidate path", "sparse observations"],
        },
        "layout": {
            "panel": "a",
            "main_field": "left",
            "uncertainty_explanations": "right stacked plots",
            "background": "white",
        },
        "integrity": "No exported vector or raster output was edited after source rendering.",
    }
    (STEM.with_name(STEM.name + "_qa.json")).write_text(json.dumps(qa, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
