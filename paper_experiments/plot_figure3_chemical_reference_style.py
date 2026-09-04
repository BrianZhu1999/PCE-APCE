from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


plt.rcParams.update(
    {
        "figure.dpi": 180,
        "savefig.dpi": 600,
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)

BG = "#f5f1e8"
PLANE = "#594541"
PLANE_EDGE = "#322623"

TRUTH = "#7fded8"
PCE = "#7dbcf0"
APCE = "#efb941"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACE_DIR = (
    PROJECT_ROOT
    / "source_data"
    / "figure3_selected6_caseprofile_formal_50seeds_20260812"
    / "representative_traces_v2"
)
OUT_BASE = (
    PROJECT_ROOT
    / "ncs_chinese_submission"
    / "figures"
    / "figure3_panel_a_chemical_reference_style_v04"
)


def load_trace(method: str) -> dict[str, np.ndarray]:
    path = TRACE_DIR / f"fig3_chemical_{method.lower()}_s2026081200.npz"
    with np.load(path, allow_pickle=True) as data:
        return {k: np.asarray(data[k]) for k in data.files}


def normalize_columns(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    stacked = np.vstack(arrays)
    lo = np.nanmin(stacked, axis=0)
    hi = np.nanmax(stacked, axis=0)
    span = np.where((hi - lo) > 1e-12, hi - lo, 1.0)
    return tuple((arr - lo) / span for arr in arrays)


def plot_glow_line(ax, xyz, color, lw=0.8, zorder=10):
    x, y, z = xyz.T
    ax.plot(x, y, z, color=color, lw=3.2, alpha=0.035, zorder=zorder - 0.4)
    ax.plot(x, y, z, color=color, lw=1.6, alpha=0.08, zorder=zorder - 0.2)
    ax.plot(x, y, z, color=color, lw=lw, alpha=0.96, zorder=zorder)


def glow_point(ax, xyz, color, zorder=26):
    x, y, z = xyz
    for size, alpha in [(540, 0.022), (310, 0.050), (170, 0.100), (92, 0.170)]:
        ax.scatter(
            [x],
            [y],
            [z],
            s=size,
            c=color,
            alpha=alpha,
            edgecolors="none",
            depthshade=False,
            zorder=zorder - 2,
        )
    ax.scatter([x], [y], [z], s=42, c=color, alpha=1.0, edgecolors="none", depthshade=False, zorder=zorder)
    ax.scatter([x], [y], [z], s=10, c="white", alpha=1.0, edgecolors="none", depthshade=False, zorder=zorder + 1)


def draw_room(ax):
    xmin, xmax = -1.25, 1.25
    ymin, ymax = -1.02, 1.02
    zmin, zmax = -0.82, 1.02

    floor = np.array(
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
        ]
    )
    wall_left = np.array(
        [
            [xmin, ymin, zmin],
            [xmin, ymax, zmin],
            [xmin, ymax, zmax],
            [xmin, ymin, zmax],
        ]
    )
    wall_back = np.array(
        [
            [xmin, ymax, zmin],
            [xmax, ymax, zmin],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ]
    )

    for verts, alpha in [(wall_left, 0.90), (wall_back, 0.90), (floor, 0.98)]:
        ax.add_collection3d(
            Poly3DCollection(
                [verts],
                facecolor=PLANE,
                edgecolor=PLANE_EDGE,
                linewidth=0.75,
                alpha=alpha,
                zorder=1,
            )
        )


def build_coords(states: np.ndarray) -> np.ndarray:
    t = np.linspace(0.0, 1.0, states.shape[0])
    # Reference-style visual encoding: time sweeps horizontally, while the two
    # chemical concentrations form the vertical/lateral state-space curve.
    coords = np.column_stack([t, states[:, 0], states[:, 1]])
    coords, = normalize_columns(coords)
    # Slight shaping so the line fills the room similarly to the reference.
    coords = coords.copy()
    coords[:, 0] = 2.34 * (coords[:, 0] - 0.50)
    coords[:, 1] = 1.46 * (coords[:, 1] - 0.50)
    coords[:, 2] = 1.64 * (coords[:, 2] - 0.50)
    return coords


def main() -> None:
    truth = load_trace("pce")["truth_states"]
    pce = load_trace("pce")["mean_states"]
    apce = load_trace("apce")["mean_states"]

    truth_c = build_coords(truth)
    pce_c = build_coords(pce)
    apce_c = build_coords(apce)

    fig = plt.figure(figsize=(4.35, 6.70), facecolor=BG)
    ax = fig.add_axes([0.025, 0.430, 0.950, 0.530], projection="3d", computed_zorder=False)
    ax.set_facecolor(BG)
    ax.set_axis_off()

    ax.view_init(elev=21.5, azim=-60.0)
    try:
        ax.set_proj_type("persp", focal_length=0.88)
    except TypeError:
        ax.set_proj_type("persp")

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.02, 1.02)
    ax.set_zlim(-0.88, 1.05)
    ax.set_box_aspect((1.45, 1.00, 0.92))
    draw_room(ax)

    plot_glow_line(ax, truth_c, TRUTH, lw=0.92, zorder=10)
    plot_glow_line(ax, pce_c, PCE, lw=0.76, zorder=11)
    plot_glow_line(ax, apce_c, APCE, lw=0.92, zorder=12)

    for arr, color in [(truth_c, TRUTH), (pce_c, PCE), (apce_c, APCE)]:
        glow_point(ax, arr[0], color)
        glow_point(ax, arr[-1], color)

    ax.quiver(0.0, 0.0, 0.0, -1.22, 0.0, 0.0, color="black", linewidth=1.65, arrow_length_ratio=0.075, zorder=20)
    ax.quiver(0.0, 0.0, 0.0, 0.0, 1.22, 0.0, color="black", linewidth=1.65, arrow_length_ratio=0.075, zorder=20)
    ax.quiver(0.0, 0.0, 0.0, 0.0, 0.0, 1.15, color="black", linewidth=1.65, arrow_length_ratio=0.075, zorder=20)

    ax.text(-1.36, 0.00, -0.05, r"$t$", fontsize=19, fontfamily="Arial", fontstyle="italic", color="black", zorder=30)
    ax.text(0.00, 1.35, -0.05, r"$x_1$", fontsize=19, fontfamily="Arial", fontstyle="italic", color="black", zorder=30)
    ax.text(0.00, 0.00, 1.28, r"$x_2$", fontsize=19, fontfamily="Arial", fontstyle="italic", color="black", zorder=30)

    legend_handles = [
        Line2D([0], [0], color=TRUTH, linewidth=0.90, marker="o", markersize=5.7, markerfacecolor="white", markeredgecolor=TRUTH, markeredgewidth=1.15, label="Truth"),
        Line2D([0], [0], color=PCE, linewidth=0.90, marker="o", markersize=5.7, markerfacecolor="white", markeredgecolor=PCE, markeredgewidth=1.15, label="PCE"),
        Line2D([0], [0], color=APCE, linewidth=0.90, marker="o", markersize=5.7, markerfacecolor="white", markeredgecolor=APCE, markeredgewidth=1.15, label="APCE"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.078, 1.045),
        frameon=False,
        prop={"family": "Arial", "size": 15.0},
        handlelength=2.0,
        handletextpad=0.55,
        borderpad=0.0,
        labelspacing=0.15,
    )

    fig.text(0.013, 0.965, "a", fontsize=25, fontweight="bold", fontfamily="Arial", ha="left", va="top", color="black")
    fig.text(0.50, 0.397, "Chemical reaction", ha="center", va="center", fontsize=19.0, fontfamily="Arial", color="black")

    eq_box = dict(boxstyle="square,pad=0.36", facecolor="#fbf8f0", edgecolor="black", linewidth=1.25)
    fig.text(
        0.50,
        0.274,
        r"$\dot a=-2k(\alpha)a^2,\qquad \dot b=k(\alpha)a^2$",
        ha="center",
        va="center",
        fontsize=15.7,
        family="Arial",
        bbox=eq_box,
    )
    fig.text(
        0.50,
        0.180,
        r"$k(\alpha)=k_0+k_1\Phi_{\mathrm{L}}^{-1}(\alpha)$",
        ha="center",
        va="center",
        fontsize=15.7,
        family="Arial",
        bbox=eq_box,
    )
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{OUT_BASE}.{ext}", facecolor=BG, bbox_inches="tight", pad_inches=0.018)

    plt.close(fig)
    print(OUT_BASE)


if __name__ == "__main__":
    main()
