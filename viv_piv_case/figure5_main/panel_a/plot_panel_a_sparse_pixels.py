"""Figure 5a with paired sparse u/v observations shown as enlarged pixels."""
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
TRACE = HERE / "source_x40y20_apce_trace.npz"
BLACKOUT = HERE / "blackout_x40y20_source_0679.npz"
OUT = HERE / "outputs_sparse_pixels_x40y20"
OUT.mkdir(parents=True, exist_ok=True)
STEM = "figure5a_viv_piv_sparse_pixel_pipeline_x40y20"

DPI = 650
FIG_W_PX, FIG_H_PX = 10553, 2450
FIG_W, FIG_H = FIG_W_PX / DPI, FIG_H_PX / DPI
BLACK = "#202020"
BLUE = "#4C78A8"
ORANGE = "#F28E2B"
GRAY = "#879395"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.weight": "normal",
        "axes.linewidth": 0.75,
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


def pca2(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centred = np.asarray(values, dtype=float) - np.mean(values, axis=0, keepdims=True)
    _, _, vectors = np.linalg.svd(centred, full_matrices=False)
    return centred @ vectors[:2].T, vectors[0], vectors[1]


def grid_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.sort(np.unique(values))
    if values.size == 1:
        delta = 0.1
        return np.asarray([values[0] - delta, values[0] + delta])
    mids = 0.5 * (values[:-1] + values[1:])
    return np.r_[values[0] - (mids[0] - values[0]), mids, values[-1] + (values[-1] - mids[-1])]


def sparse_grid(
    sx: np.ndarray, sy: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gx = np.sort(np.unique(sx))
    gy = np.sort(np.unique(sy))
    z = np.full((gy.size, gx.size), np.nan, dtype=float)
    for x, y, value in zip(sx, sy, values):
        ix = int(np.argmin(np.abs(gx - x)))
        iy = int(np.argmin(np.abs(gy - y)))
        z[iy, ix] = value
    return gx, gy, grid_edges(gx), grid_edges(gy), z


def add_sparse_map(
    ax: plt.Axes,
    sx: np.ndarray,
    sy: np.ndarray,
    values: np.ndarray,
    cylinder_y: float,
    norm: Normalize,
) -> mpl.collections.QuadMesh:
    gx, gy, x_edges, y_edges, z = sparse_grid(sx, sy, values)
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        z,
        cmap="RdBu_r",
        norm=norm,
        shading="flat",
        edgecolors="white",
        linewidth=0.18,
        rasterized=True,
    )
    ax.add_patch(
        Circle(
            (0.0, cylinder_y),
            0.5,
            facecolor="white",
            edgecolor=BLACK,
            linewidth=0.85,
            zorder=5,
        )
    )
    ax.set_xlim(float(x_edges[0]), float(x_edges[-1]))
    ax.set_ylim(float(y_edges[0]), float(y_edges[-1]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x/D$", fontsize=13, labelpad=1)
    ax.set_ylabel(r"$y/D$", fontsize=13, labelpad=1)
    ax.tick_params(labelsize=11, width=0.65, length=2.5, pad=1.3)
    ax.set_xticks([-1, 2, 5, 8])
    ax.set_yticks([-2, 0, 2])
    return mesh


def flow_arrow(fig: plt.Figure, start: tuple[float, float], end: tuple[float, float], color: str) -> None:
    fig.add_artist(
        FancyArrowPatch(
            start,
            end,
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.25,
            color=color,
            shrinkA=5,
            shrinkB=5,
        )
    )


def main() -> None:
    sx, sy, sparse_u, sparse_v, frame_time = load_snapshot()
    time_s, displacement = load_displacement()
    with np.load(BLACKOUT, allow_pickle=False) as blackout:
        blackout_time = float(blackout["origin_time_s"])
    cylinder_y = float(np.interp(frame_time, time_s, displacement))
    vmax = float(np.nanpercentile(np.abs(np.r_[sparse_u, sparse_v]), 99.5))
    vmax = max(vmax, 1e-3)
    norm = Normalize(vmin=-vmax, vmax=vmax)

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor="white")
    fig.text(0.018, 0.94, "a", fontsize=22, fontweight="bold", va="center", color=BLACK)

    u_ax = fig.add_axes([0.055, 0.22, 0.145, 0.62])
    v_ax = fig.add_axes([0.225, 0.22, 0.145, 0.62])
    latent_ax = fig.add_axes([0.445, 0.22, 0.49, 0.62])
    # Keep the measured cylinder motion as an independent inset.  It must not
    # overlap the state-estimation schematic.
    disp_ax = fig.add_axes([0.735, 0.655, 0.17, 0.155], zorder=6)

    u_mesh = add_sparse_map(u_ax, sx, sy, sparse_u, cylinder_y, norm)
    add_sparse_map(v_ax, sx, sy, sparse_v, cylinder_y, norm)
    u_ax.text(0.03, 1.03, r"$u$", transform=u_ax.transAxes, fontsize=15, fontweight="bold", va="bottom")
    v_ax.text(0.03, 1.03, r"$v$", transform=v_ax.transAxes, fontsize=15, fontweight="bold", va="bottom")
    cax = fig.add_axes([0.055, 0.125, 0.315, 0.025])
    cb = fig.colorbar(u_mesh, cax=cax, orientation="horizontal")
    cb.set_label(r"$\mathrm{m\,s^{-1}}$", fontsize=12, labelpad=1)
    cb.ax.tick_params(labelsize=10, length=2, pad=1)

    rel_t = time_s - blackout_time
    disp_ax.set_facecolor("white")
    disp_ax.patch.set_alpha(0.96)
    disp_ax.plot(rel_t, displacement, color=BLACK, lw=1.0)
    disp_ax.axvspan(0.0, 4.0, color=ORANGE, alpha=0.10, zorder=0)
    disp_ax.axvline(0.0, color=ORANGE, lw=0.95, ls="--")
    disp_ax.scatter(
        [0.0],
        [np.interp(blackout_time, time_s, displacement)],
        s=14,
        color=ORANGE,
        edgecolor="white",
        linewidth=0.5,
        zorder=4,
    )
    disp_ax.set_xlim(-8.0, 6.0)
    disp_ax.set_xlabel(r"$t-t_b$", fontsize=9, labelpad=0)
    disp_ax.set_ylabel(r"$y_c/D$", fontsize=9, labelpad=0)
    disp_ax.tick_params(labelsize=7, width=0.55, length=2, pad=0.5)
    disp_ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    disp_ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    latent_ax.set_xlim(0.0, 1.0)
    latent_ax.set_ylim(0.0, 1.0)
    latent_ax.axis("off")
    # A compact visual encoding of the 256-dimensional latent state.  The
    # 16x16 node array is intentionally value-free: it communicates
    # dimensionality without fabricating a particular latent realization.
    state_x, state_y = 0.52, 0.39
    n_state = 16
    nodes = np.linspace(state_x + 0.018, state_x + 0.212, n_state)
    node_x, node_y = np.meshgrid(nodes, nodes)
    latent_ax.scatter(
        node_x.ravel(),
        node_y.ravel(),
        s=9.5,
        marker="s",
        facecolor="#AFC6DC",
        edgecolor="white",
        linewidth=0.18,
        zorder=2,
    )
    latent_ax.add_patch(
        FancyBboxPatch(
            (state_x, state_y),
            0.23,
            0.23,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            facecolor="none",
            edgecolor="#8EA9C2",
            linewidth=1.0,
            zorder=3,
        )
    )
    latent_ax.text(
        state_x + 0.115,
        state_y - 0.035,
        r"$z_t \in \mathbb{R}^{256}$",
        ha="center",
        va="top",
        fontsize=14,
        color=BLACK,
    )

    # Two clean incoming arrows: one from the paired field observations and
    # one from the measured cylinder motion inset.
    latent_ax.annotate(
        "",
        xy=(state_x, state_y + 0.15),
        xytext=(0.03, state_y + 0.15),
        arrowprops=dict(arrowstyle="-|>", lw=1.35, color=BLUE, shrinkA=0, shrinkB=0),
    )
    latent_ax.annotate(
        "",
        xy=(state_x + 0.11, state_y + 0.23),
        xytext=(0.79, 0.66),
        arrowprops=dict(arrowstyle="-|>", lw=1.25, color=ORANGE, shrinkA=0, shrinkB=0),
    )
    # The only outgoing arrow is vertical and intentionally reaches the lower
    # edge, where the assembled Figure 5 places the b–e full-field panels.
    latent_ax.annotate(
        "",
        xy=(state_x + 0.115, 0.04),
        xytext=(state_x + 0.115, state_y - 0.005),
        arrowprops=dict(arrowstyle="-|>", lw=1.65, color=BLACK, shrinkA=0, shrinkB=0),
    )
    latent_ax.text(
        state_x + 0.115,
        0.015,
        r"$b$–$e$",
        ha="center",
        va="top",
        fontsize=12,
        color=BLACK,
    )

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
        "backend": "Python/matplotlib only",
        "canvas": {"width_px": FIG_W_PX, "height_px": FIG_H_PX, "dpi": DPI},
        "observation_definition": {
            "nominal_layout": "40 x 20",
            "effective_spatial_locations": 751,
            "scalar_observations": 1502,
            "components": "paired u and v at every retained location",
            "map_frame": 200,
            "map_time_s": frame_time,
            "pixel_values": "reference experimental u/v sampled at the registered sparse locations",
        },
        "integrity": {
            "full_field_is_not_shown_as_observation": True,
            "cylinder_displacement_is_measured": True,
            "cylinder_displacement_source": str(DISPLACEMENT),
            "latent_state_glyph_is_schematic": True,
            "shadow_receives_analysis_update": False,
            "weights_are_not_bayesian_posteriors": True,
        },
        "visual_sources": {
            "field_snapshot": str(FIELD),
            "layout": str(LAYOUT),
            "displacement": str(DISPLACEMENT),
            "x40y20_apce_trace": str(TRACE),
            "x40y20_blackout_source": str(BLACKOUT),
            "remote_result_root": "<HILDA_RESULTS_ROOT>/results/viv_piv_pce_apce_adaptive_valid_x40y20_formal5",
        },
        "deduplication": {
            "candidate_identification_panel_removed": True,
            "reason": "candidate evidence is already presented in Figure 5j",
            "panel_a_role": "paired sparse u/v observations, measured cylinder input and POD state compression",
        },
        "outputs": outputs,
    }
    metadata["output_sha256"] = {ext: sha256(stem.with_suffix(ext)) for ext in outputs}
    (OUT / f"{STEM}_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
