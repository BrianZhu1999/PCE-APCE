#!/usr/bin/env python3
"""Nature-style schematic for the MeshRIR S1 3D reconstruction task.

The figure is a task definition, not an experimental result. It distinguishes
boundary measurements, interior assimilation measurements, and held-out
interior evaluation points, and shows the intended causal information flow.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch


HERE = Path(__file__).resolve().parent
OUT = HERE

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

COLORS = {
    "boundary": "#D87942",
    "interior": "#3C78A8",
    "heldout": "#C8CDD3",
    "heldout_edge": "#8C969F",
    "wave": "#5B6570",
    "pce": "#3C78A8",
    "apce": "#D87942",
    "ink": "#27313A",
    "box": "#F5F7F8",
    "soft_blue": "#EAF1F6",
    "soft_orange": "#FBEEE6",
}


def grid_points() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return full points and three scientifically distinct masks."""
    z = np.linspace(-0.2, 0.2, 9)
    y = np.linspace(-0.5, 0.5, 21)
    x = np.linspace(-0.5, 0.5, 21)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    sparse_x = np.rint(np.linspace(0, 20, 7)).astype(int)
    sparse_y = np.rint(np.linspace(0, 20, 7)).astype(int)
    sparse_z = np.rint(np.linspace(0, 8, 3)).astype(int)
    sparse = {(int(ix), int(iy), int(iz)) for iz in sparse_z for iy in sparse_y for ix in sparse_x}
    boundary = np.zeros(len(points), dtype=bool)
    interior_observed = np.zeros(len(points), dtype=bool)
    for i, (iz, iy, ix) in enumerate(
        ( (iz, iy, ix) for iz in range(9) for iy in range(21) for ix in range(21) )
    ):
        if (ix, iy, iz) in sparse:
            if iz in (0, 8) or iy in (0, 20) or ix in (0, 20):
                boundary[i] = True
            else:
                interior_observed[i] = True
    heldout_interior = ~(boundary | interior_observed)
    return points, boundary, interior_observed, heldout_interior


def set_equal_3d(ax: mpl.axes.Axes) -> None:
    ax.set_xlim(-0.58, 0.58)
    ax.set_ylim(-0.58, 0.58)
    ax.set_zlim(-0.25, 0.25)
    ax.set_box_aspect((1.0, 1.0, 0.42))
    ax.view_init(elev=19, azim=-58)
    ax.set_xlabel("x (m)", labelpad=-3)
    ax.set_ylabel("y (m)", labelpad=-3)
    ax.set_zlabel("z (m)", labelpad=-3)
    ax.tick_params(pad=-2, labelsize=6)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#D5DADF")


def draw_box(ax: mpl.axes.Axes) -> None:
    x0, x1 = -0.5, 0.5
    y0, y1 = -0.5, 0.5
    z0, z1 = -0.2, 0.2
    vertices = np.array(
        [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ]
    )
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for i, j in edges:
        ax.plot(*vertices[[i, j]].T, color="#AAB3BC", lw=0.65, alpha=0.9, zorder=1)


def box(ax: mpl.axes.Axes, xy: tuple[float, float], width: float, height: float,
        title: str, body: str, color: str, title_color: str | None = None) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.012,rounding_size=0.025",
        transform=ax.transAxes, facecolor=color, edgecolor="#B7C0C8", lw=0.7,
    )
    ax.add_patch(patch)
    ax.text(x + 0.03, y + height - 0.055, title, transform=ax.transAxes,
            color=title_color or COLORS["ink"], fontsize=7.2, fontweight="bold", va="top")
    ax.text(x + 0.03, y + height - 0.115, body, transform=ax.transAxes,
            color=COLORS["ink"], fontsize=5.8, linespacing=1.18, va="top")


def arrow(ax: mpl.axes.Axes, start: tuple[float, float], end: tuple[float, float],
          color: str = COLORS["wave"], lw: float = 1.0, connectionstyle: str = "arc3") -> None:
    ax.add_patch(FancyArrowPatch(start, end, transform=ax.transAxes,
                                 arrowstyle="-|>", mutation_scale=10,
                                 lw=lw, color=color, connectionstyle=connectionstyle))


def main() -> None:
    points, boundary, interior, heldout = grid_points()
    fig = plt.figure(figsize=(7.2, 4.55), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.02, 1.18], wspace=0.02,
                          left=0.035, right=0.985, top=0.93, bottom=0.08)
    ax3 = fig.add_subplot(gs[0, 0], projection="3d")
    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")

    # Hero 3D task panel.
    draw_box(ax3)
    ax3.scatter(points[heldout, 0], points[heldout, 1], points[heldout, 2],
                s=1.3, color=COLORS["heldout"], alpha=0.22, depthshade=False, zorder=1)
    ax3.scatter(points[boundary, 0], points[boundary, 1], points[boundary, 2],
                s=8.5, color=COLORS["boundary"], alpha=0.92, depthshade=False, zorder=4)
    ax3.scatter(points[interior, 0], points[interior, 1], points[interior, 2],
                s=19, color=COLORS["interior"], edgecolor="white", linewidth=0.25,
                alpha=0.98, depthshade=False, zorder=5)
    set_equal_3d(ax3)
    ax3.set_title("S1-M3969: sparse 3D reconstruction", loc="left", pad=4,
                  fontsize=9, fontweight="bold", color=COLORS["ink"])
    ax3.text2D(0.02, 0.93, "21 × 21 × 9 grid  |  3969 points", transform=ax3.transAxes,
                fontsize=6.7, color="#56616B")
    legend = [
        (COLORS["boundary"], "122 boundary measurements"),
        (COLORS["interior"], "25 interior assimilation points"),
        (COLORS["heldout"], "2502 held-out interior points"),
    ]
    for i, (c, label) in enumerate(legend):
        ax3.scatter([], [], [], s=22 if i == 1 else 12, color=c, alpha=0.95,
                    label=label, edgecolor="white" if i == 1 else "none", linewidth=0.25)
    ax3.legend(loc="lower left", bbox_to_anchor=(0.0, -0.02), frameon=False,
               fontsize=6.2, handletextpad=0.45, labelspacing=0.25, borderaxespad=0)

    # Causal information flow panel.
    ax.text(0.02, 0.965, "Information flow and evaluation", transform=ax.transAxes,
            fontsize=8.2, fontweight="bold", color=COLORS["ink"], va="top")
    ax.text(0.02, 0.91, "Boundary input; interior reconstruction test.",
            transform=ax.transAxes, fontsize=6.1, color="#56616B", va="top")

    box(ax, (0.04, 0.68), 0.38, 0.16, "1  Boundary completion",
        "122 boundary measurements\ncausal interpolation on six faces\nboundary field ĝ(r,t)", COLORS["soft_orange"])
    box(ax, (0.54, 0.68), 0.39, 0.16, "2  Full 3D wave dynamics",
        "21 × 21 × 9 finite-difference field\nfixed calibrated sound speed ĉ\nno POD / IDW surrogate", COLORS["soft_blue"])
    box(ax, (0.04, 0.41), 0.38, 0.16, "3  Interior assimilation",
        "25 interior measurements\nstate update at analysis times\nlocalised ensemble covariance", COLORS["soft_blue"])
    box(ax, (0.54, 0.41), 0.39, 0.16, "4  Shadow evidence",
        "candidate boundary closures\nuncorrected shadow forecasts\nPCE / APCE weight updates", COLORS["soft_orange"])
    box(ax, (0.18, 0.10), 0.64, 0.19, "5  Primary evaluation",
        "2502 held-out interior points\n3D pressure: nRMSE · correlation · LSD · coverage\nheld-out boundary points reported separately", "#F4F5F6")

    arrow(ax, (0.42, 0.76), (0.54, 0.76), color=COLORS["boundary"], lw=1.2)
    arrow(ax, (0.23, 0.68), (0.23, 0.57), color=COLORS["wave"], lw=1.0)
    arrow(ax, (0.73, 0.68), (0.73, 0.57), color=COLORS["wave"], lw=1.0)
    arrow(ax, (0.42, 0.49), (0.54, 0.49), color=COLORS["interior"], lw=1.2)
    arrow(ax, (0.73, 0.41), (0.65, 0.29), color=COLORS["apce"], lw=1.0)
    arrow(ax, (0.31, 0.41), (0.38, 0.29), color=COLORS["interior"], lw=1.0)
    arrow(ax, (0.73, 0.68), (0.86, 0.58), color=COLORS["pce"], lw=0.8, connectionstyle="arc3,rad=-0.25")
    ax.text(0.78, 0.61, "model state", transform=ax.transAxes, fontsize=6.1, color="#56616B")
    ax.text(0.43, 0.61, "causal input", transform=ax.transAxes, fontsize=6.1, color=COLORS["boundary"])

    fig.text(0.035, 0.965, "a", fontsize=11, fontweight="bold", color=COLORS["ink"], va="top")
    fig.text(0.505, 0.965, "b", fontsize=11, fontweight="bold", color=COLORS["ink"], va="top")
    fig.text(0.035, 0.018,
             "Task definition: measurements are restricted to the coloured points; held-out points are used only for final scoring.",
             fontsize=6.5, color="#56616B")

    stem = OUT / "s1_reconstruction_task_schematic"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    registry = {
        "figure": "s1_reconstruction_task_schematic",
        "type": "task_definition_schematic",
        "backend": "python/matplotlib",
        "claim": "Sparse boundary and interior measurements are routed through a full 3D wave model, while held-out interior points are reserved for evaluation.",
        "panels": {
            "a": {
                "source": "deterministic 21x21x9 Cartesian task grid",
                "counts": {"boundary_measurements": 122, "interior_assimilation": 25, "heldout_interior": 2502},
                "source_script": str(Path(__file__).resolve()),
            },
            "b": {
                "source": "task specification, no experimental result data",
                "information_flow": ["boundary completion", "full 3D wave dynamics", "interior assimilation", "shadow evidence", "held-out evaluation"],
                "source_script": str(Path(__file__).resolve()),
            },
        },
        "outputs": [str(stem.with_suffix(s)) for s in (".svg", ".pdf", ".png", ".tiff")],
        "scientific_boundary": "The schematic does not report reconstruction performance or imply that held-out points were used by the algorithm.",
    }
    (OUT / "s1_reconstruction_task_schematic.provenance.json").write_text(
        json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(registry, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
