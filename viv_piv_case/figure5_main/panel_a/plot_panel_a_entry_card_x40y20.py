from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


RAW_DEFAULT = Path("<PUBLIC_DATA_ROOT>/viv_piv_hpa87o/reduced_velocity_0679.npz")
LAYOUT_DEFAULT = Path(
    "<HILDA_RESULTS_ROOT>/results/viv_piv_pce_apce_adaptive_valid_x40y20_formal5/"
    "figures/excel_source/layout_points_0679.csv"
)
OUTDIR_DEFAULT = Path(
    "<HILDA_RESULTS_ROOT>/code/hybrid_uncertain_wave/viv_piv_case/"
    "figure5_main/panel_a/outputs_entry_card_x40y20"
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _cell_edges(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    if centers.size == 1:
        step = 1.0
        return np.asarray([centers[0] - step / 2, centers[0] + step / 2])
    mids = 0.5 * (centers[:-1] + centers[1:])
    first = centers[0] - (mids[0] - centers[0])
    last = centers[-1] + (centers[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def _make_sparse_component_grid(
    component_values: np.ndarray,
    layout: pd.DataFrame,
    scalar_column: str,
    nx_full: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_centers = np.asarray(sorted(layout["x_over_d"].unique()), dtype=float)
    y_centers = np.asarray(sorted(layout["y_over_d"].unique()), dtype=float)
    x_to_i = {float(x): i for i, x in enumerate(x_centers)}
    y_to_i = {float(y): i for i, y in enumerate(y_centers)}
    grid = np.full((y_centers.size, x_centers.size), np.nan, dtype=float)

    for row in layout.itertuples(index=False):
        scalar_index = int(getattr(row, scalar_column))
        flat_spatial = scalar_index // 2
        iy = flat_spatial // nx_full
        ix = flat_spatial % nx_full
        gx = x_to_i[float(row.x_over_d)]
        gy = y_to_i[float(row.y_over_d)]
        grid[gy, gx] = float(component_values[iy, ix])

    return grid, x_centers, y_centers


def _style_sparse_axis(ax: plt.Axes, title: str, grid: np.ndarray, x_centers: np.ndarray, y_centers: np.ndarray, cmap, vlim: float, yc_over_d: float) -> None:
    x_edges = _cell_edges(x_centers)
    y_edges = _cell_edges(y_centers)
    ax.pcolormesh(
        x_edges,
        y_edges,
        grid,
        cmap=cmap,
        vmin=-vlim,
        vmax=vlim,
        shading="flat",
        edgecolors=(1, 1, 1, 0.78),
        linewidth=0.18,
        rasterized=True,
    )
    ax.add_patch(Circle((0.0, yc_over_d), 0.50, facecolor="white", edgecolor="#52606d", lw=0.8, zorder=5))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])
    ax.set_title(title, fontsize=13, fontweight="bold", color="#202428", pad=3)
    ax.set_xlabel(r"$x/D$", fontsize=10, labelpad=1)
    ax.set_ylabel(r"$y/D$", fontsize=10, labelpad=1)
    ax.tick_params(labelsize=8, length=2.5, width=0.55, colors="#333333", pad=1)
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color("#6c737a")
    ax.set_facecolor("white")


def _draw_state_grid(ax: plt.Axes) -> None:
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(-0.5, 15.5)
    ax.set_aspect("equal")
    xs, ys = np.meshgrid(np.arange(16), np.arange(16))
    ax.scatter(
        xs.ravel(),
        ys.ravel(),
        s=18,
        facecolor="#b8c9d8",
        edgecolor="#7f97aa",
        linewidth=0.25,
        alpha=0.94,
    )
    ax.add_patch(
        FancyBboxPatch(
            (-0.72, -0.72),
            16.44,
            16.44,
            boxstyle="round,pad=0.18,rounding_size=0.55",
            facecolor="none",
            edgecolor="#8aa0b2",
            lw=0.85,
        )
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _arrow(fig: plt.Figure, xy_a: tuple[float, float], xy_b: tuple[float, float], color: str, lw: float = 1.15, rad: float = 0.0) -> None:
    fig.patches.append(
        FancyArrowPatch(
            xy_a,
            xy_b,
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=9.0,
            lw=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=2,
            shrinkB=2,
            zorder=20,
        )
    )


def draw_panel(raw_path: Path, layout_path: Path, outdir: Path, frame: int, dpi: int) -> dict:
    raw = np.load(raw_path, allow_pickle=True)
    layout = pd.read_csv(layout_path)
    velocities_norm = raw["velocities"]
    mask = raw["mask"]
    time_s = np.asarray(raw["time"], dtype=float)
    cyl_displ_m = np.asarray(raw["cyl_displ"], dtype=float)
    norm_values = np.asarray(raw["norm_values"], dtype=np.float32)

    if frame < 0 or frame >= velocities_norm.shape[0]:
        raise ValueError(f"frame {frame} outside available range 0..{velocities_norm.shape[0] - 1}")

    low = norm_values[0]
    high = norm_values[1]
    velocity_phys = np.asarray(velocities_norm[frame], dtype=np.float32) * (high - low)[None, None, :] + low[None, None, :]
    valid_mask = np.asarray(mask[frame] > 0.5, dtype=bool)
    velocity_phys = np.where(valid_mask[..., None], velocity_phys, np.nan)

    u_grid, x_centers, y_centers = _make_sparse_component_grid(velocity_phys[..., 0], layout, "u_scalar_index", velocity_phys.shape[1])
    v_grid, _, _ = _make_sparse_component_grid(velocity_phys[..., 1], layout, "v_scalar_index", velocity_phys.shape[1])

    finite = np.concatenate([u_grid[np.isfinite(u_grid)], v_grid[np.isfinite(v_grid)]])
    vlim = float(np.nanpercentile(np.abs(finite), 98.5))
    vlim = max(vlim, 1e-6)
    yc_over_d = float(cyl_displ_m[frame] / 0.05)
    displacement_over_d = cyl_displ_m / 0.05

    outdir.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "mathtext.fontset": "dejavusans",
        }
    )

    cmap = LinearSegmentedColormap.from_list("soft_blue_white_red", ["#2f6f91", "#f8f9f6", "#b5524b"], N=256)
    cmap.set_bad("white")

    fig = plt.figure(figsize=(10553 / dpi, 6000 / dpi), dpi=dpi, facecolor="white")
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.axis("off")

    # Very subtle card backgrounds: enough grouping, no PPT-like boxes.
    canvas.add_patch(
        FancyBboxPatch(
            (0.045, 0.122),
            0.610,
            0.780,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            transform=fig.transFigure,
            facecolor="#fbfcfc",
            edgecolor="#edf1f2",
            lw=0.7,
            zorder=-5,
        )
    )
    canvas.add_patch(
        FancyBboxPatch(
            (0.695, 0.255),
            0.255,
            0.560,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            transform=fig.transFigure,
            facecolor="#fbfcfd",
            edgecolor="#eef2f5",
            lw=0.7,
            zorder=-5,
        )
    )

    fig.text(0.026, 0.930, "a", fontsize=23, fontweight="bold", color="#101418", ha="left", va="top")
    fig.text(0.075, 0.875, "Sparse velocity observations", fontsize=15, fontweight="bold", color="#1f2933", ha="left")
    fig.text(
        0.075,
        0.836,
        "20 × 40 nominal layout  →  751 valid spatial locations  →  1502 scalar observations",
        fontsize=10.5,
        color="#5d6670",
        ha="left",
    )

    ax_u = fig.add_axes([0.075, 0.540, 0.255, 0.245])
    ax_v = fig.add_axes([0.370, 0.540, 0.255, 0.245])
    _style_sparse_axis(ax_u, r"$u$", u_grid, x_centers, y_centers, cmap, vlim, yc_over_d)
    _style_sparse_axis(ax_v, r"$v$", v_grid, x_centers, y_centers, cmap, vlim, yc_over_d)

    ax_disp = fig.add_axes([0.095, 0.190, 0.500, 0.205])
    ax_disp.plot(time_s, displacement_over_d, color="#17191c", lw=0.95)
    ax_disp.axvline(time_s[frame], color="#d98a36", lw=0.95, ls=(0, (3.2, 2.4)))
    ax_disp.set_title("Known cylinder displacement", fontsize=13, fontweight="bold", color="#202428", pad=4)
    ax_disp.set_xlabel(r"$t$", fontsize=10, labelpad=1)
    ax_disp.set_ylabel(r"$y_c/D$", fontsize=10, labelpad=1)
    ax_disp.tick_params(labelsize=8.5, length=2.5, width=0.55, colors="#333333", pad=1)
    ax_disp.spines["left"].set_color("#6c737a")
    ax_disp.spines["bottom"].set_color("#6c737a")
    ax_disp.spines["left"].set_linewidth(0.6)
    ax_disp.spines["bottom"].set_linewidth(0.6)
    ymin, ymax = float(np.nanmin(displacement_over_d)), float(np.nanmax(displacement_over_d))
    pad = 0.08 * (ymax - ymin)
    ax_disp.set_ylim(ymin - pad, ymax + pad)
    ax_disp.text(
        time_s[frame],
        ymax + 0.52 * pad,
        r"$t_b$",
        color="#a25f20",
        fontsize=9.5,
        ha="center",
        va="bottom",
    )

    ax_state = fig.add_axes([0.720, 0.485, 0.155, 0.272])
    _draw_state_grid(ax_state)
    fig.text(0.903, 0.628, r"$z_t \in \mathbb{R}^{256}$", fontsize=15.5, color="#1f2933", ha="left", va="center")
    fig.text(0.905, 0.585, "POD state", fontsize=11.5, color="#56616c", ha="left", va="center")

    # Sparse u/v merge to the state; routed above the pixel maps to avoid
    # visually editing or crossing the observed data blocks.
    merge = (0.655, 0.805)
    _arrow(fig, (0.330, 0.805), merge, "#6f8da7", lw=0.90, rad=0.0)
    _arrow(fig, (0.625, 0.805), merge, "#6f8da7", lw=0.90, rad=0.0)
    _arrow(fig, merge, (0.720, 0.665), "#6f8da7", lw=1.10, rad=-0.10)

    # Cylinder displacement to the same state.
    _arrow(fig, (0.595, 0.360), (0.720, 0.545), "#b88b60", lw=1.05, rad=0.06)

    # State to panels b-e.
    _arrow(fig, (0.797, 0.485), (0.797, 0.287), "#3d4248", lw=1.10, rad=0.0)
    fig.text(0.797, 0.228, "to b–e", fontsize=13, fontweight="bold", color="#24292f", ha="center")
    fig.text(0.797, 0.190, "full-field reconstruction / forecast", fontsize=10.8, color="#5b636b", ha="center")

    basename = "figure5a_entry_card_x40y20_0679"
    paths = {
        ".png": outdir / f"{basename}.png",
        ".tiff": outdir / f"{basename}.tiff",
        ".pdf": outdir / f"{basename}.pdf",
        ".svg": outdir / f"{basename}.svg",
    }
    fig.savefig(paths[".png"], dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(paths[".tiff"], dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(paths[".pdf"], bbox_inches="tight", pad_inches=0.02)
    fig.savefig(paths[".svg"], bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    metadata = {
        "figure": "figure5a_entry_card",
        "purpose": "Method-entry panel only: sparse u/v observations and known cylinder displacement constrain a 256-dimensional POD state, which links to panels b-e.",
        "not_included": [
            "candidate dynamics",
            "PCE/APCE weight maps",
            "full-field thumbnails",
            "latent trajectories",
            "blackout forecast timeline",
        ],
        "backend": "Python/matplotlib",
        "case_id": "0679",
        "frame": int(frame),
        "time_s": float(time_s[frame]),
        "cylinder_y_over_d": yc_over_d,
        "observation_layout": {
            "name": "adaptive_fullfield_valid_x40y20",
            "x_points": 40,
            "y_points": 20,
            "nominal_points": 800,
            "effective_points": int(layout.shape[0]),
            "scalar_observations": int(layout.shape[0] * 2),
            "mask_aware": True,
        },
        "data_sources": {
            "raw_truth_archive": str(raw_path),
            "layout_points": str(layout_path),
        },
        "integrity": {
            "sparse_uv_from_truth_velocity_frame": True,
            "velocity_denormalized_with_norm_values": True,
            "invalid_and_missing_nominal_layout_cells_left_white": True,
            "cylinder_displacement_from_measured_signal": True,
            "pod_state_grid_symbolic_only": True,
        },
        "canvas": {
            "width_px": 10553,
            "height_px": 6000,
            "dpi": dpi,
            "aspect_ratio": 10553 / 6000,
        },
        "outputs": {ext: str(path) for ext, path in paths.items()},
        "output_sha256": {ext: _sha256(path) for ext, path in paths.items()},
    }
    meta_path = outdir / f"{basename}_metadata.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata["metadata_path"] = str(meta_path)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw Figure 5a as a sparse-observation entry card for the x40-y20 / 751-point VIV-PIV setup.")
    parser.add_argument("--raw", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--layout", type=Path, default=LAYOUT_DEFAULT)
    parser.add_argument("--outdir", type=Path, default=OUTDIR_DEFAULT)
    parser.add_argument("--frame", type=int, default=940, help="Frame used for sparse u/v and the t_b marker.")
    parser.add_argument("--dpi", type=int, default=650)
    args = parser.parse_args()
    metadata = draw_panel(args.raw, args.layout, args.outdir, args.frame, args.dpi)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
