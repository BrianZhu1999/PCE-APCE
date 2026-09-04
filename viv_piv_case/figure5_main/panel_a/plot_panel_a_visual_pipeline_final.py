"""Figure 5a: visual sparse-observation to POD-state pipeline.

The panel is intentionally image-led:
  paired sparse u/v pixel blocks + measured cylinder motion
      -> 256 POD-state nodes
      -> downward pointer to the full-field panels b-e.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import MaxNLocator


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREVIEW = ROOT.parent / "results_preview" / "x40y20_formal5" / "excel_source"
FIELD = PREVIEW / "field_reconstruction_frame_0200.csv"
LAYOUT = PREVIEW / "layout_points_0679.csv"
DISPLACEMENT = PREVIEW / "displacement_timeseries.csv"
BLACKOUT = HERE / "blackout_x40y20_source_0679.npz"
OUT = HERE / "outputs_visual_pipeline_final_x40y20"
OUT.mkdir(parents=True, exist_ok=True)
STEM = "figure5a_viv_piv_visual_pipeline_final_x40y20"

DPI = 650
FIG_W_PX, FIG_H_PX = 10553, 2450
FIG_W, FIG_H = FIG_W_PX / DPI, FIG_H_PX / DPI
BLACK = "#202020"
BLUE = "#4C78A8"
ORANGE = "#F28E2B"
GRID = "#D8E0E3"
PALE = "#F4F6F6"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.weight": "normal",
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_layout() -> tuple[np.ndarray, np.ndarray]:
    with LAYOUT.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return (
        np.asarray([float(row["x_over_d"]) for row in rows], dtype=float),
        np.asarray([float(row["y_over_d"]) for row in rows], dtype=float),
    )


def load_snapshot() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    with FIELD.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    frame_time = float(rows[0]["time_s"])
    fx = np.sort(np.unique([float(row["x_over_d"]) for row in rows]))
    fy = np.sort(np.unique([float(row["y_over_d"]) for row in rows]))
    field = np.full((fy.size, fx.size, 2), np.nan, dtype=float)
    for row in rows:
        ix = int(np.argmin(np.abs(fx - float(row["x_over_d"]))))
        iy = int(np.argmin(np.abs(fy - float(row["y_over_d"]))))
        if row["valid_fluid"].lower() == "true":
            field[iy, ix] = [
                float(row["reference_u_m_s"]),
                float(row["reference_v_m_s"]),
            ]
    sx, sy = load_layout()
    values = np.asarray(
        [
            field[
                int(np.argmin(np.abs(fy - y))),
                int(np.argmin(np.abs(fx - x))),
            ]
            for x, y in zip(sx, sy)
        ],
        dtype=float,
    )
    return sx, sy, values[:, 0], values[:, 1], frame_time


def load_displacement() -> tuple[np.ndarray, np.ndarray]:
    with DISPLACEMENT.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["case_id"] == "0679"]
    return (
        np.asarray([float(row["time_s"]) for row in rows], dtype=float),
        np.asarray([float(row["displacement_over_d"]) for row in rows], dtype=float),
    )


def grid_edges(values: np.ndarray) -> np.ndarray:
    values = np.sort(np.unique(np.asarray(values, dtype=float)))
    if values.size == 1:
        return np.asarray([values[0] - 0.1, values[0] + 0.1])
    mids = 0.5 * (values[:-1] + values[1:])
    return np.r_[values[0] - (mids[0] - values[0]), mids, values[-1] + (values[-1] - mids[-1])]


def sparse_grid(
    sx: np.ndarray, sy: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gx = np.sort(np.unique(sx))
    gy = np.sort(np.unique(sy))
    z = np.full((gy.size, gx.size), np.nan, dtype=float)
    for x, y, value in zip(sx, sy, values):
        ix = int(np.argmin(np.abs(gx - x)))
        iy = int(np.argmin(np.abs(gy - y)))
        z[iy, ix] = value
    return gx, gy, grid_edges(gx), grid_edges(gy), z


def add_pixel_map(
    ax: plt.Axes,
    sx: np.ndarray,
    sy: np.ndarray,
    values: np.ndarray,
    cylinder_y: float,
    norm: Normalize,
    *,
    show_ylabel: bool,
) -> mpl.collections.QuadMesh:
    gx, gy, x_edges, y_edges, z = sparse_grid(sx, sy, values)
    nominal = np.zeros_like(z)
    ax.pcolormesh(
        x_edges,
        y_edges,
        nominal,
        cmap=mpl.colors.ListedColormap([PALE]),
        shading="flat",
        edgecolors="white",
        linewidth=0.22,
        rasterized=True,
        zorder=0,
    )
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_invalid(z),
        cmap="RdBu_r",
        norm=norm,
        shading="flat",
        edgecolors="white",
        linewidth=0.22,
        rasterized=True,
        zorder=1,
    )
    ax.add_patch(
        Circle(
            (0.0, cylinder_y),
            0.5,
            facecolor="white",
            edgecolor=BLACK,
            linewidth=0.9,
            zorder=4,
        )
    )
    ax.set_xlim(float(x_edges[0]), float(x_edges[-1]))
    ax.set_ylim(float(y_edges[0]), float(y_edges[-1]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x/D$", fontsize=10, labelpad=1)
    if show_ylabel:
        ax.set_ylabel(r"$y/D$", fontsize=10, labelpad=1)
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    ax.set_xticks([-1, 2, 5, 8])
    ax.set_yticks([-2, 0, 2])
    ax.tick_params(labelsize=8, width=0.55, length=2.0, pad=1.0)
    return mesh


def fig_arrow(fig: plt.Figure, start: tuple[float, float], end: tuple[float, float], color: str, lw: float = 1.45) -> None:
    fig.add_artist(
        FancyArrowPatch(
            start,
            end,
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=lw,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def main() -> None:
    sx, sy, sparse_u, sparse_v, frame_time = load_snapshot()
    time_s, displacement = load_displacement()
    with np.load(BLACKOUT, allow_pickle=False) as blackout:
        blackout_time = float(blackout["origin_time_s"])
    cylinder_y = float(np.interp(frame_time, time_s, displacement))
    vmax = float(np.nanpercentile(np.abs(np.r_[sparse_u, sparse_v]), 99.5))
    norm = Normalize(vmin=-max(vmax, 1e-3), vmax=max(vmax, 1e-3))

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor="white")
    fig.text(0.018, 0.94, "a", fontsize=22, fontweight="bold", va="center", color=BLACK)

    u_ax = fig.add_axes([0.055, 0.405, 0.155, 0.40])
    v_ax = fig.add_axes([0.235, 0.405, 0.155, 0.40])
    disp_ax = fig.add_axes([0.055, 0.145, 0.335, 0.145])
    state_ax = fig.add_axes([0.565, 0.365, 0.19, 0.44])

    u_mesh = add_pixel_map(u_ax, sx, sy, sparse_u, cylinder_y, norm, show_ylabel=True)
    add_pixel_map(v_ax, sx, sy, sparse_v, cylinder_y, norm, show_ylabel=False)
    u_ax.text(0.03, 1.02, r"$u$", transform=u_ax.transAxes, fontsize=14, fontweight="bold", va="bottom")
    v_ax.text(0.03, 1.02, r"$v$", transform=v_ax.transAxes, fontsize=14, fontweight="bold", va="bottom")

    cax = fig.add_axes([0.055, 0.345, 0.335, 0.018])
    cb = fig.colorbar(u_mesh, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=7, length=1.6, pad=0.5)
    cb.set_label(r"$\mathrm{m\,s^{-1}}$", fontsize=8, labelpad=0.2)

    rel_t = time_s - blackout_time
    disp_ax.plot(rel_t, displacement, color=BLACK, lw=1.05)
    disp_ax.axvspan(0.0, 4.0, color=ORANGE, alpha=0.09, zorder=0)
    disp_ax.axvline(0.0, color=ORANGE, lw=0.9, ls="--")
    disp_ax.scatter(
        [0.0],
        [np.interp(blackout_time, time_s, displacement)],
        s=14,
        color=ORANGE,
        edgecolor="white",
        linewidth=0.45,
        zorder=3,
    )
    disp_ax.set_xlim(-8.0, 6.0)
    disp_ax.set_xlabel(r"$t-t_b$", fontsize=9, labelpad=0.5)
    disp_ax.set_ylabel(r"$y_c/D$", fontsize=9, labelpad=0.5)
    disp_ax.tick_params(labelsize=7, width=0.5, length=1.8, pad=0.5)
    disp_ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    disp_ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    state_ax.set_xlim(0, 1)
    state_ax.set_ylim(0, 1)
    state_ax.axis("off")
    state_x, state_y, state_w = 0.24, 0.24, 0.52
    nodes = np.linspace(state_x + 0.035, state_x + state_w - 0.035, 16)
    node_x, node_y = np.meshgrid(nodes, nodes)
    state_ax.scatter(
        node_x.ravel(),
        node_y.ravel(),
        s=15.5,
        marker="s",
        facecolor="#AFC6DC",
        edgecolor="white",
        linewidth=0.24,
        zorder=2,
    )
    state_ax.add_patch(
        FancyBboxPatch(
            (state_x, state_y),
            state_w,
            state_w,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            facecolor="none",
            edgecolor="#8EA9C2",
            linewidth=1.0,
            zorder=3,
        )
    )
    state_ax.text(
        state_x + state_w / 2,
        state_y - 0.08,
        r"$z_t\in\mathbb{R}^{256}$",
        ha="center",
        va="top",
        fontsize=12,
        color=BLACK,
    )

    # One visual junction: field observations and measured cylinder motion
    # converge on the POD state.  A single downward pointer continues into
    # the already assembled full-field panels b-e.
    fig_arrow(fig, (0.405, 0.60), (0.555, 0.60), BLUE, lw=1.5)
    fig_arrow(fig, (0.405, 0.215), (0.555, 0.47), ORANGE, lw=1.35)
    fig_arrow(fig, (0.66, 0.34), (0.66, 0.08), BLACK, lw=1.65)
    fig.text(0.66, 0.045, r"$b$–$e$", ha="center", va="top", fontsize=12, color=BLACK)

    stem = OUT / STEM
    for ext, kwargs in (
        (".png", {"dpi": DPI}),
        (".tiff", {"dpi": DPI}),
        (".pdf", {}),
        (".svg", {}),
    ):
        fig.savefig(stem.with_suffix(ext), facecolor="white", pad_inches=0, **kwargs)
    plt.close(fig)

    outputs = {ext: str(stem.with_suffix(ext)) for ext in (".png", ".tiff", ".pdf", ".svg")}
    metadata = {
        "figure": STEM,
        "panel": "a",
        "core_conclusion": "Paired sparse u/v observations and the measured cylinder motion are assimilated into a 256-dimensional POD state that feeds the full-field reconstructions in b-e.",
        "backend": "Python/matplotlib only",
        "canvas": {"width_px": FIG_W_PX, "height_px": FIG_H_PX, "dpi": DPI},
        "observation_definition": {
            "nominal_layout": "40 x 20",
            "effective_spatial_locations": 751,
            "scalar_observations": 1502,
            "components": "paired u and v at every retained location",
            "map_frame": 200,
            "map_time_s": frame_time,
        },
        "integrity": {
            "full_field_is_not_shown_as_observation": True,
            "cylinder_displacement_is_measured": True,
            "latent_state_glyph_is_schematic": True,
        },
        "visual_sources": {
            "field_snapshot": str(FIELD),
            "layout": str(LAYOUT),
            "displacement": str(DISPLACEMENT),
            "blackout_source": str(BLACKOUT),
            "remote_result_root": "<HILDA_RESULTS_ROOT>/results/viv_piv_pce_apce_adaptive_valid_x40y20_formal5",
        },
        "outputs": outputs,
    }
    metadata["output_sha256"] = {ext: sha256(stem.with_suffix(ext)) for ext in outputs}
    (OUT / f"{STEM}_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
