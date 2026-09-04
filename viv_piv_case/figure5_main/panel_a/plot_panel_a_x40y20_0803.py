from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrow, FancyArrowPatch, Polygon
from matplotlib.ticker import FuncFormatter, MaxNLocator


DPI = 650
WIDTH_PX = 10553
HEIGHT_PX = 3300
WIDTH_IN = WIDTH_PX / DPI
HEIGHT_IN = HEIGHT_PX / DPI
DIAMETER_M = 0.05
POD_RANK = 256
POD_COLOR_LIMIT = 3.0

RAW_DEFAULT = Path("<PUBLIC_DATA_ROOT>/viv_piv_hpa87o/reduced_velocity_0803.npz")
LAYOUT_DEFAULT = Path(
    "<HILDA_RESULTS_ROOT>/results/viv_piv_pce_apce_adaptive_valid_x40y20_formal5/"
    "models/rank256_stride1/sensor_layouts/adaptive_fullfield_valid_x40y20/case_0803.npz"
)
BC_FRAME_DEFAULT = Path(
    "<HILDA_RESULTS_ROOT>/results/viv_piv_pce_apce_adaptive_valid_x40y20_formal5/"
    "figures/figure5_sources_0803/best_reconstruction_frame_0554.npz"
)
TRACE_DEFAULT = Path(
    "<HILDA_RESULTS_ROOT>/results/viv_piv_pce_apce_adaptive_valid_x40y20_formal5/"
    "runs/rank256_stride1/traces/viv_0803_apce_seed000_layoutadaptive_fullfield_valid_x40y20_ens064_covfull_shr050.npz"
)
POD_DEFAULT = Path(
    "<HILDA_RESULTS_ROOT>/results/viv_piv_pce_apce_adaptive_valid_x40y20_formal5/"
    "models/rank256_stride1/pod_model.npz"
)
MODEL_MANIFEST_DEFAULT = Path(
    "<HILDA_RESULTS_ROOT>/results/viv_piv_pce_apce_adaptive_valid_x40y20_formal5/"
    "models/rank256_stride1/model_manifest.json"
)
COEFFICIENTS_DEFAULT = Path(
    "<HILDA_RESULTS_ROOT>/results/viv_piv_pce_apce_adaptive_valid_x40y20_formal5/"
    "models/rank256_stride1/coefficients"
)
OUTDIR_DEFAULT = Path(
    "<HILDA_RESULTS_ROOT>/code/hybrid_uncertain_wave/viv_piv_case/"
    "figure5_main/panel_a/outputs_x40y20_0803_spacetime"
)


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 11,
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "axes.linewidth": 0.75,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "mathtext.fontset": "dejavusans",
        }
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def file_size(path: Path) -> int:
    return int(path.stat().st_size)


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as img:
        return int(img.size[0]), int(img.size[1])


def cell_edges(centres: np.ndarray) -> np.ndarray:
    centres = np.asarray(centres, dtype=float)
    mids = 0.5 * (centres[:-1] + centres[1:])
    return np.concatenate([[centres[0] - (mids[0] - centres[0])], mids, [centres[-1] + (centres[-1] - mids[-1])]])


def bcde_cmap() -> mpl.colors.LinearSegmentedColormap:
    return mpl.colors.LinearSegmentedColormap.from_list(
        "blue_warmwhite_red",
        [
            (0.000, "#185685"),
            (0.125, "#6C93B0"),
            (0.250, "#B6CCD9"),
            (0.375, "#EBEFEE"),
            (0.500, "#F0ECE1"),
            (0.625, "#F8E9D6"),
            (0.750, "#E4B6A7"),
            (0.875, "#BF6B69"),
            (1.000, "#9A1D28"),
        ],
        N=256,
    )


def pod_cmap() -> mpl.colors.LinearSegmentedColormap:
    """
    Signed POD-coefficient palette, deliberately distinct from
    the physical u/v blue-white-red field palette.
    """
    return mpl.colors.LinearSegmentedColormap.from_list(
        "pod_indigo_ivory_ochre",
        [
            (0.00, "#3E3A78"),
            (0.18, "#665C9D"),
            (0.36, "#A99FC2"),
            (0.50, "#F3EFE7"),
            (0.64, "#E4C999"),
            (0.82, "#C58D45"),
            (1.00, "#8A541E"),
        ],
        N=256,
    )


def sparse_grid(
    values: np.ndarray,
    x_indices: np.ndarray,
    y_indices: np.ndarray,
    x_over_d: np.ndarray,
    y_over_d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.asarray(x_over_d[x_indices], dtype=float)
    ys = np.asarray(y_over_d[y_indices], dtype=float)
    x_unique = np.asarray(sorted(np.unique(xs)), dtype=float)
    y_unique = np.asarray(sorted(np.unique(ys)), dtype=float)
    x_to_j = {float(x): j for j, x in enumerate(x_unique)}
    y_to_i = {float(y): i for i, y in enumerate(y_unique)}
    grid = np.full((y_unique.size, x_unique.size), np.nan, dtype=float)
    for val, x, y in zip(values, xs, ys):
        grid[y_to_i[float(y)], x_to_j[float(x)]] = float(val)
    return grid, x_unique, y_unique


def asymmetric_norm(values: np.ndarray) -> Normalize | TwoSlopeNorm:
    finite = values[np.isfinite(values)]
    lo = float(np.nanpercentile(finite, 0.5))
    hi = max(1e-6, float(np.nanpercentile(finite, 99.5)))
    if lo >= 0.0:
        return Normalize(vmin=lo, vmax=hi)
    return TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)


def symmetric_norm(values: np.ndarray) -> TwoSlopeNorm:
    finite = values[np.isfinite(values)]
    lim = max(1e-6, float(np.nanpercentile(np.abs(finite), 99.5)))
    return TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)


def norm_metadata(norm: Normalize | TwoSlopeNorm) -> dict[str, float | None]:
    return {
        "vmin": float(norm.vmin),
        "vmax": float(norm.vmax),
        "vcenter": float(norm.vcenter) if hasattr(norm, "vcenter") else None,
    }


def scaled_tick_formatter(value: float, _position: float) -> str:
    scaled = value * 10.0
    if abs(scaled - round(scaled)) < 1e-9:
        return str(int(round(scaled)))
    return f"{scaled:.2f}".rstrip("0").rstrip(".")


def find_history_frames(raw_time_s: np.ndarray, frame: int, time_s: float) -> list[int]:
    raw_time_s = np.asarray(raw_time_s, dtype=float)
    indices = np.arange(raw_time_s.size)
    frames: list[int] = []
    for target_time in (time_s - 2.0, time_s - 1.0, time_s):
        distances = np.where(indices <= frame, np.abs(raw_time_s - target_time), np.inf)
        frames.append(int(np.argmin(distances)))
    assert all(v <= frame for v in frames), f"future frame leakage: {frames} > {frame}"
    assert max(frames) == frame, f"current frame not included as foreground plane: {frames}"
    return frames


def add_overlay_axes(fig: plt.Figure) -> plt.Axes:
    overlay = fig.add_axes([0.0, 0.0, 1.0, 1.0], zorder=1000)
    overlay.set_axis_off()
    overlay.patch.set_alpha(0.0)
    overlay.set_xlim(0.0, 1.0)
    overlay.set_ylim(0.0, 1.0)
    return overlay


def overlay_line(
    overlay: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#202225",
    lw: float = 0.75,
    alpha: float = 1.0,
    linestyle: str | tuple[float, tuple[float, ...]] = "-",
    zorder: int = 5,
) -> None:
    overlay.add_line(
        Line2D(
            [start[0], end[0]],
            [start[1], end[1]],
            transform=overlay.transAxes,
            color=color,
            lw=lw,
            alpha=alpha,
            linestyle=linestyle,
            solid_capstyle="round",
            zorder=zorder,
        )
    )


def overlay_arrow(
    overlay: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#202225",
    lw: float = 0.82,
    mutation_scale: float = 7.2,
    rad: float = 0.0,
    alpha: float = 1.0,
    zorder: int = 6,
) -> None:
    overlay.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=overlay.transAxes,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            lw=lw,
            color=color,
            alpha=alpha,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        )
    )


def overlay_block_arrow(
    overlay: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#6E91B8",
    alpha: float = 0.98,
    zorder: int = 40,
) -> None:
    x0, y0 = start
    x1, y1 = end

    body_h = 0.024
    head_h = 0.050
    head_len = 0.016

    # Build a symmetric arrow polygon.
    if x1 > x0:
        # right-pointing
        xb = x1 - head_len
        verts = [
            (x0, y0 - body_h / 2),
            (xb, y0 - body_h / 2),
            (xb, y0 - head_h / 2),
            (x1, y0),
            (xb, y0 + head_h / 2),
            (xb, y0 + body_h / 2),
            (x0, y0 + body_h / 2),
        ]
        c0 = "#D7E3EF"   # tail, pale mist blue
        c1 = "#8FAECD"   # mid, muted steel blue
        c2 = "#4F6F97"   # head, slate blue near POD
    else:
        # left-pointing
        xb = x1 + head_len
        verts = [
            (x0, y0 - body_h / 2),
            (xb, y0 - body_h / 2),
            (xb, y0 - head_h / 2),
            (x1, y0),
            (xb, y0 + head_h / 2),
            (xb, y0 + body_h / 2),
            (x0, y0 + body_h / 2),
        ]
        # keep "tail light -> head dark" relative to direction toward POD
        c0 = "#4F6F97"
        c1 = "#8FAECD"
        c2 = "#D7E3EF"

    clip = Polygon(
        verts,
        closed=True,
        facecolor="none",
        edgecolor="#5A6E86",
        linewidth=0.36,
        alpha=0.80,
        transform=overlay.transAxes,
        zorder=zorder + 0.2,
        joinstyle="miter",
    )
    overlay.add_patch(clip)

    rgba0 = np.asarray(mpl.colors.to_rgba(c0))
    rgba1 = np.asarray(mpl.colors.to_rgba(c1))
    rgba2 = np.asarray(mpl.colors.to_rgba(c2))

    n = 600
    t = np.linspace(0.0, 1.0, n)

    grad = np.zeros((32, n, 4), dtype=float)

    left = t <= 0.55
    right = ~left

    tl = np.zeros_like(t)
    tl[left] = t[left] / 0.55
    tr = np.zeros_like(t)
    tr[right] = (t[right] - 0.55) / 0.45

    grad[:, left, :] = (
        rgba0[None, None, :] * (1.0 - tl[left][None, :, None])
        + rgba1[None, None, :] * tl[left][None, :, None]
    )
    grad[:, right, :] = (
        rgba1[None, None, :] * (1.0 - tr[right][None, :, None])
        + rgba2[None, None, :] * tr[right][None, :, None]
    )

    grad[..., 3] *= alpha

    xmin = min(x0, x1)
    xmax = max(x0, x1)

    im = overlay.imshow(
        grad,
        extent=[xmin, xmax, y0 - head_h / 2, y0 + head_h / 2],
        origin="lower",
        aspect="auto",
        interpolation="bicubic",
        transform=overlay.transAxes,
        zorder=zorder,
    )
    im.set_clip_path(clip)


def rect_corners(rect: list[float]) -> list[tuple[float, float]]:
    x, y, w, h = rect
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def draw_coordinate_triad(
    overlay: plt.Axes,
    origin: tuple[float, float],
    layer_dx: float,
    layer_dy: float,
) -> None:
    ox, oy = origin

    # Intentionally compact coordinate triad.
    # x is substantially shorter than before.
    x_end = (ox + 0.020, oy)
    y_end = (ox, oy + 0.060)
    t_end = (ox + 0.022, oy + 0.048)

    arrow_kw = dict(
        arrowstyle="-|>",
        mutation_scale=6.6,
        lw=0.85,
        color="#111111",
        shrinkA=0,
        shrinkB=0,
    )

    overlay.add_patch(
        FancyArrowPatch(
            (ox, oy),
            x_end,
            transform=overlay.transAxes,
            zorder=30,
            **arrow_kw,
        )
    )

    overlay.add_patch(
        FancyArrowPatch(
            (ox, oy),
            y_end,
            transform=overlay.transAxes,
            zorder=30,
            **arrow_kw,
        )
    )

    overlay.add_patch(
        FancyArrowPatch(
            (ox, oy),
            t_end,
            transform=overlay.transAxes,
            zorder=30,
            **arrow_kw,
        )
    )

    overlay.text(
        x_end[0] + 0.004,
        x_end[1] - 0.006,
        r"$x$",
        transform=overlay.transAxes,
        fontsize=13,
        color="#111111",
        ha="left",
        va="center",
    )

    overlay.text(
        y_end[0] - 0.004,
        y_end[1] + 0.004,
        r"$y$",
        transform=overlay.transAxes,
        fontsize=13,
        color="#111111",
        ha="center",
        va="bottom",
    )

    overlay.text(
        t_end[0] + 0.004,
        t_end[1] + 0.002,
        r"$t$",
        transform=overlay.transAxes,
        fontsize=13,
        color="#111111",
        ha="left",
        va="bottom",
    )


def draw_sparse_plane(
    ax: plt.Axes,
    grid: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    cylinder_y_over_d: float,
    cmap: mpl.colors.Colormap,
    norm: Normalize | TwoSlopeNorm,
    *,
    alpha: float,
    edge_alpha: float,
    spine_alpha: float,
    spine_lw: float,
) -> None:
    cmap_local = cmap.copy()
    cmap_local.set_bad((1.0, 1.0, 1.0, 0.0))
    ax.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_invalid(grid),
        shading="flat",
        cmap=cmap_local,
        norm=norm,
        edgecolors=(1.0, 1.0, 1.0, edge_alpha),
        linewidth=0.10,
        alpha=alpha,
        rasterized=True,
        zorder=2,
    )
    ax.add_patch(
        Circle(
            (0.0, float(cylinder_y_over_d)),
            0.50,
            facecolor="white",
            edgecolor="#111111",
            linewidth=0.68,
            alpha=min(1.0, 0.20 + 0.80 * alpha),
            zorder=5,
        )
    )
    ax.set_xlim(float(x_edges.min()), float(x_edges.max()))
    ax.set_ylim(float(y_edges.min()), float(y_edges.max()))
    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color((0.07, 0.07, 0.07, spine_alpha))
        spine.set_linewidth(spine_lw)


def draw_horizontal_colorbar(
    fig: plt.Figure,
    bbox: list[float],
    cmap: mpl.colors.Colormap,
    norm: Normalize | TwoSlopeNorm,
    label: str,
    *,
    scaled_ticks: bool = False,
    nbins: int = 3,
) -> mpl.colorbar.Colorbar:
    cax = fig.add_axes(bbox)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=11, length=2.5, width=0.60, pad=2, colors="#111111")
    colorbar.ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins))
    if scaled_ticks:
        colorbar.ax.xaxis.set_major_formatter(FuncFormatter(scaled_tick_formatter))
    colorbar.set_label(label, fontsize=13, labelpad=2, color="#111111")
    for spine in colorbar.ax.spines.values():
        spine.set_visible(False)
    return colorbar


def draw_spatiotemporal_slab(
    fig: plt.Figure,
    overlay: plt.Axes,
    *,
    title: str,
    front_rect: list[float],
    layer_dx: float,
    layer_dy: float,
    grids: list[np.ndarray],
    x_centres: np.ndarray,
    y_centres: np.ndarray,
    cylinder_y_values: list[float],
    cmap: mpl.colors.Colormap,
    norm: Normalize | TwoSlopeNorm,
    colorbar_label: str,
) -> dict[str, object]:
    assert len(grids) == 3
    assert len(cylinder_y_values) == 3
    x_edges = cell_edges(x_centres)
    y_edges = cell_edges(y_centres)
    layer_specs = [
        ("back", [front_rect[0] + 2.0 * layer_dx, front_rect[1] + 2.0 * layer_dy, front_rect[2], front_rect[3]], 0.36, 0.14, 0.25, 0.46),
        ("middle", [front_rect[0] + layer_dx, front_rect[1] + layer_dy, front_rect[2], front_rect[3]], 0.64, 0.24, 0.38, 0.56),
        ("front", front_rect, 1.00, 0.42, 0.82, 0.78),
    ]

    for layer_name, rect, alpha, grid_edge_alpha, spine_alpha, spine_lw in layer_specs:
        layer_idx = {"back": 0, "middle": 1, "front": 2}[layer_name]
        ax = fig.add_axes(rect, zorder=10 + layer_idx)
        draw_sparse_plane(
            ax,
            grids[layer_idx],
            x_edges,
            y_edges,
            cylinder_y_values[layer_idx],
            cmap,
            norm,
            alpha=alpha,
            edge_alpha=grid_edge_alpha,
            spine_alpha=spine_alpha,
            spine_lw=spine_lw,
        )

    back_rect = layer_specs[0][1]
    middle_rect = layer_specs[1][1]
    front = layer_specs[2][1]
    for corner_idx in range(4):
        p_front = rect_corners(front)[corner_idx]
        p_mid = rect_corners(middle_rect)[corner_idx]
        p_back = rect_corners(back_rect)[corner_idx]
        linestyle = (0, (2.8, 2.5)) if corner_idx in (2, 3) else "-"
        overlay_line(overlay, p_front, p_mid, color="#202225", lw=0.52, alpha=0.34, linestyle=linestyle, zorder=20)
        overlay_line(overlay, p_mid, p_back, color="#202225", lw=0.52, alpha=0.26, linestyle=linestyle, zorder=20)

    title_x = front_rect[0] + 0.5 * (front_rect[2] + 2.0 * layer_dx)
    title_y = front_rect[1] + front_rect[3] + 2.0 * layer_dy + 0.016
    fig.text(title_x, title_y, title, ha="center", va="bottom", fontsize=14, fontweight="normal", color="#111111")

    cbar_bbox = [
        front_rect[0],
        front_rect[1] - 0.054,
        front_rect[2],
        0.016,
    ]
    draw_horizontal_colorbar(fig, cbar_bbox, cmap, norm, "", scaled_ticks=True, nbins=3)

    return {
        "front_rect": front_rect,
        "middle_rect": middle_rect,
        "back_rect": back_rect,
        "colorbar_bbox": cbar_bbox,
        "layer_dx_fig": layer_dx,
        "layer_dy_fig": layer_dy,
        "layer_dx_px": layer_dx * WIDTH_PX,
        "layer_dy_px": layer_dy * HEIGHT_PX,
    }


def load_pod_state_and_scaling(
    trace: np.lib.npyio.NpzFile,
    pod: np.lib.npyio.NpzFile,
    manifest: dict[str, object],
    coefficients_dir: Path,
    frame: int,
) -> dict[str, object]:
    latent_trace = np.asarray(trace["latent_estimate"], dtype=float)
    latent_t = np.asarray(latent_trace[frame], dtype=float)
    assert latent_t.shape == (POD_RANK,), f"expected 256-D APCE state; got {latent_t.shape}"

    singular_values = np.asarray(pod["singular_values"][:POD_RANK], dtype=float)
    sample_count = int(manifest["sample_count"])
    sqrt_lambda = singular_values / np.sqrt(float(sample_count - 1))
    z_tilde = latent_t / sqrt_lambda
    assert z_tilde.shape == (POD_RANK,)

    trace_z_tilde = latent_trace[:, :POD_RANK] / sqrt_lambda
    trace_abs = np.abs(trace_z_tilde[np.isfinite(trace_z_tilde)])
    trace_abs_percentiles = {str(p): float(np.nanpercentile(trace_abs, p)) for p in (95, 98, 99, 99.5, 99.9, 100)}

    train_abs_percentiles: dict[str, float] = {}
    coefficient_files = sorted(coefficients_dir.glob("case_*.npz"))
    if coefficient_files:
        standardized_chunks: list[np.ndarray] = []
        for path in coefficient_files:
            with np.load(path, allow_pickle=False) as coeff_raw:
                if "coefficients" not in coeff_raw.files:
                    continue
                coeff = np.asarray(coeff_raw["coefficients"][:, :POD_RANK], dtype=float)
                standardized_chunks.append(np.ravel(coeff / sqrt_lambda))
        if standardized_chunks:
            train_abs = np.abs(np.concatenate(standardized_chunks))
            train_abs = train_abs[np.isfinite(train_abs)]
            train_abs_percentiles = {str(p): float(np.nanpercentile(train_abs, p)) for p in (95, 98, 99, 99.5, 99.9, 100)}

    return {
        "latent_trace": latent_trace,
        "latent_t": latent_t,
        "singular_values": singular_values,
        "sqrt_lambda": sqrt_lambda,
        "z_tilde": z_tilde,
        "trace_abs_percentiles": trace_abs_percentiles,
        "train_abs_percentiles": train_abs_percentiles,
        "coefficient_file_count": len(coefficient_files),
        "sample_count": sample_count,
    }


def draw_pod_state(
    fig: plt.Figure,
    ax: plt.Axes,
    z_tilde: np.ndarray,
    cmap: mpl.colors.Colormap,
    norm: TwoSlopeNorm,
    *,
    title_center_x: float,
) -> None:
    indices = np.arange(POD_RANK)
    cols = indices % 16
    rows = indices // 16
    y_display = 15 - rows
    ax.scatter(
        cols,
        y_display,
        c=z_tilde,
        cmap=cmap,
        norm=norm,
        s=38.0,
        edgecolor="white",
        linewidth=0.28,
        alpha=0.98,
        zorder=3,
    )
    ax.set_xlim(-0.70, 15.70)
    ax.set_ylim(-0.70, 15.70)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(title_center_x, 0.805, "256-D POD state", ha="center", va="bottom", fontsize=14, fontweight="normal", color="#111111")
    fig.text(title_center_x, 0.737, r"$\hat{\mathbf{z}}_t \in \mathbb{R}^{256}$", ha="center", va="bottom", fontsize=13.5, color="#111111")


def draw_displacement_axis(
    ax: plt.Axes,
    raw_time_s: np.ndarray,
    cyl_y_over_d: np.ndarray,
    *,
    frame: int,
    time_s: float,
) -> float:
    current_y = float(cyl_y_over_d[frame])

    # Only a subtle zero reference; no current-time vertical line.
    ax.axhline(
        0.0,
        color="#AEB4BA",
        lw=0.58,
        alpha=0.62,
        zorder=0,
    )

    ax.plot(
        raw_time_s,
        cyl_y_over_d,
        color="#B8DDE4",
        lw=2.05,
        alpha=0.62,
        solid_capstyle="round",
        zorder=2,
    )
    ax.plot(
        raw_time_s,
        cyl_y_over_d,
        color="#247F91",
        lw=1.08,
        solid_capstyle="round",
        zorder=3,
    )

    ax.set_title(
        "Known cylinder displacement",
        fontsize=14,
        fontweight="normal",
        color="#111111",
        pad=6,
    )

    ax.set_xlabel(
        r"$t$",
        fontsize=13,
        color="#111111",
        labelpad=2,
    )

    ax.set_ylabel(
        r"$y_c/D$",
        fontsize=13,
        color="#111111",
        labelpad=3,
    )

    ax.set_xlim(
        float(raw_time_s.min()),
        float(raw_time_s.max()),
    )

    ymin = float(np.nanmin(cyl_y_over_d))
    ymax = float(np.nanmax(cyl_y_over_d))
    span = ymax - ymin
    pad = 0.09 * span if span > 0 else 0.1

    ax.set_ylim(
        ymin - pad,
        ymax + pad,
    )

    # Clean scientific ticks.
    xmin = float(raw_time_s.min())
    xmax = float(raw_time_s.max())

    preferred_xticks = np.asarray(
        [20.0, 40.0, 60.0, 80.0, 100.0],
        dtype=float,
    )
    preferred_xticks = preferred_xticks[
        (preferred_xticks >= xmin) &
        (preferred_xticks <= xmax)
    ]

    if preferred_xticks.size >= 3:
        ax.set_xticks(preferred_xticks)
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    ax.tick_params(
        axis="both",
        labelsize=11,
        length=2.8,
        width=0.65,
        pad=2,
        colors="#111111",
        direction="out",
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)
        spine.set_color("#222222")

    return current_y


def save_exact(fig: plt.Figure, base: Path) -> dict[str, Path]:
    paths = {
        ".png": base.with_suffix(".png"),
        ".tiff": base.with_suffix(".tiff"),
        ".pdf": base.with_suffix(".pdf"),
        ".svg": base.with_suffix(".svg"),
    }
    fig.savefig(paths[".png"], dpi=DPI, facecolor="white")
    fig.savefig(paths[".tiff"], dpi=DPI, facecolor="white")
    fig.savefig(paths[".pdf"], facecolor="white")
    fig.savefig(paths[".svg"], facecolor="white")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw Figure 5 panel a, x40-y20 / 751-point case 0803 spacetime entry.")
    parser.add_argument("--raw", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--layout", type=Path, default=LAYOUT_DEFAULT)
    parser.add_argument("--bc-frame", type=Path, default=BC_FRAME_DEFAULT)
    parser.add_argument("--trace", type=Path, default=TRACE_DEFAULT)
    parser.add_argument("--pod", type=Path, default=POD_DEFAULT)
    parser.add_argument("--model-manifest", type=Path, default=MODEL_MANIFEST_DEFAULT)
    parser.add_argument("--coefficients-dir", type=Path, default=COEFFICIENTS_DEFAULT)
    parser.add_argument("--outdir", type=Path, default=OUTDIR_DEFAULT)
    args = parser.parse_args()

    configure_matplotlib()

    raw = np.load(args.raw, allow_pickle=True)
    layout = np.load(args.layout, allow_pickle=True)
    bc = np.load(args.bc_frame, allow_pickle=True)
    trace = np.load(args.trace, allow_pickle=True)
    pod = np.load(args.pod, allow_pickle=True)
    manifest = json.loads(args.model_manifest.read_text(encoding="utf-8"))

    frame = int(np.asarray(bc["frame"]))
    time_s = float(np.asarray(bc["time_s"]))
    raw_time_s = np.asarray(raw["time"], dtype=float)
    cyl_y_over_d = np.asarray(raw["cyl_displ"], dtype=float) / DIAMETER_M
    x_over_d = np.asarray(raw["x"], dtype=float) / 50.0
    y_over_d = np.asarray(raw["y"], dtype=float) / 50.0

    assert frame == 554, f"unexpected display frame: {frame}"
    assert abs(time_s - 60.816) < 1e-6, f"unexpected display time: {time_s}"

    observations = np.asarray(layout["sensor_observations"], dtype=np.float32)
    sensor_uv = observations[frame].reshape(-1, 2)
    x_indices = np.asarray(layout["x_indices"], dtype=np.int64)
    y_indices = np.asarray(layout["y_indices"], dtype=np.int64)
    assert sensor_uv.shape == (751, 2), f"expected 751 u/v locations; got {sensor_uv.shape}"
    assert sensor_uv.size == 1502, f"expected 1502 scalar observations; got {sensor_uv.size}"
    full_field_spatial_locations = int(x_over_d.size * y_over_d.size)
    spatial_coverage_percent = 100.0 * float(sensor_uv.shape[0]) / float(full_field_spatial_locations)

    stack_frames = find_history_frames(raw_time_s, frame, time_s)
    u_grids: list[np.ndarray] = []
    v_grids: list[np.ndarray] = []
    stack_cylinder_y: list[float] = []
    gx = gy = None
    for stack_frame in stack_frames:
        stack_uv = observations[stack_frame].reshape(-1, 2)
        u_stack, gx, gy = sparse_grid(stack_uv[:, 0], x_indices, y_indices, x_over_d, y_over_d)
        v_stack, _, _ = sparse_grid(stack_uv[:, 1], x_indices, y_indices, x_over_d, y_over_d)
        assert int(np.isfinite(u_stack).sum()) == 751
        assert int(np.isfinite(v_stack).sum()) == 751
        u_grids.append(u_stack)
        v_grids.append(v_stack)
        stack_cylinder_y.append(float(cyl_y_over_d[stack_frame]))
    assert gx is not None and gy is not None
    u_grid = u_grids[-1]
    v_grid = v_grids[-1]

    # ------------------------------------------------------------
    # TRUE physical x/D:y/D aspect ratio.
    #
    # Width is frozen. Height ONLY is corrected.
    # Because normalized figure x and y units have different
    # pixel scales, WIDTH_PX / HEIGHT_PX must be included.
    # ------------------------------------------------------------
    x_edges_real = cell_edges(gx)
    y_edges_real = cell_edges(gy)

    x_span = float(
        x_edges_real[-1] - x_edges_real[0]
    )
    y_span = float(
        y_edges_real[-1] - y_edges_real[0]
    )

    data_aspect = x_span / y_span

    # Keep font sizes fixed, but use more of the available panel-a canvas by
    # expanding the data-bearing axes themselves.  This is the source-level
    # layout change requested for the final composite; the downstream
    # compositor must not scale panel a as a raster.
    front_w = 0.145

    front_h = (
        front_w
        * (WIDTH_PX / HEIGHT_PX)
        / data_aspect
    )

    rendered_aspect = (
        front_w * WIDTH_PX
    ) / (
        front_h * HEIGHT_PX
    )

    aspect_rel_error = abs(
        rendered_aspect / data_aspect - 1.0
    )

    assert aspect_rel_error < 0.01, (
        "sparse plane aspect mismatch: "
        f"data={data_aspect:.8f}, "
        f"rendered={rendered_aspect:.8f}, "
        f"relative_error={aspect_rel_error:.4%}"
    )

    u_norm = asymmetric_norm(u_grid)
    v_norm = symmetric_norm(v_grid)

    # u/v retain the approved Figure-5 field palette.
    uv_cmap = bcde_cmap()

    # POD has its own palette.
    podmap = pod_cmap()

    pod_state = load_pod_state_and_scaling(trace, pod, manifest, args.coefficients_dir, frame)
    latent_t = np.asarray(pod_state["latent_t"], dtype=float)
    z_tilde = np.asarray(pod_state["z_tilde"], dtype=float)
    pod_norm = TwoSlopeNorm(vmin=-POD_COLOR_LIMIT, vcenter=0.0, vmax=POD_COLOR_LIMIT)

    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN), dpi=DPI, facecolor="white")
    overlay = add_overlay_axes(fig)
    fig.text(0.0077, 0.885, "a", ha="left", va="center", fontsize=22, fontweight="bold", color="#111111")

    layer_dx = 0.0280
    layer_dy = 0.0640
    sparse_dx = 0.010
    u_front_x = 0.018 + sparse_dx
    v_front_x = 0.213 + sparse_dx

    u_slab = draw_spatiotemporal_slab(
        fig,
        overlay,
        title=r"$u$",
        front_rect=[u_front_x, 0.417, front_w, front_h],
        layer_dx=layer_dx,
        layer_dy=layer_dy,
        grids=u_grids,
        x_centres=gx,
        y_centres=gy,
        cylinder_y_values=stack_cylinder_y,
        cmap=uv_cmap,
        norm=u_norm,
        colorbar_label=r"$u$ ($\times 10^{-1}$ m s$^{-1}$)",
    )
    v_slab = draw_spatiotemporal_slab(
        fig,
        overlay,
        title=r"$v$",
        front_rect=[v_front_x, 0.417, front_w, front_h],
        layer_dx=layer_dx,
        layer_dy=layer_dy,
        grids=v_grids,
        x_centres=gx,
        y_centres=gy,
        cylinder_y_values=stack_cylinder_y,
        cmap=uv_cmap,
        norm=v_norm,
        colorbar_label=r"$v$ ($\times 10^{-1}$ m s$^{-1}$)",
    )

    # ------------------------------------------------------------
    # ONE shared external x-y-t triad.
    # It belongs to the u/v observation group, not to either plane.
    # ------------------------------------------------------------
    triad_origin = (0.008, 0.720)

    draw_coordinate_triad(
        overlay,
        triad_origin,
        layer_dx,
        layer_dy,
    )

    fig.text(0.231, 0.848, f"Sparse velocity observations ({spatial_coverage_percent:.1f}% spatial coverage)",
        ha="center", va="center",
        fontsize=14,
        fontweight="normal",
        color="#111111",
    )

    # Define the fixed input arrows once, then derive the POD position from
    # their inward tips.  The POD graphic and colourbar stay centered even if
    # a future arrow-length adjustment is required.
    input_arrow_y = 0.570
    input_arrow_length = 0.023
    left_arrow_start = 0.426
    left_arrow_end = left_arrow_start + input_arrow_length
    right_arrow_start = 0.590
    right_arrow_end = right_arrow_start - input_arrow_length

    pod_bbox_width = 0.130
    pod_expected_center_x = 0.5 * (left_arrow_end + right_arrow_end)
    pod_bbox_x = pod_expected_center_x - 0.5 * pod_bbox_width
    pod_bbox = [pod_bbox_x, 0.420, pod_bbox_width, 0.300]
    pod_cbar_bbox = [pod_bbox_x + 0.007, 0.361, 0.116, 0.014]
    pod_left_clearance = pod_bbox_x - left_arrow_end
    pod_right_clearance = right_arrow_end - (pod_bbox_x + pod_bbox_width)
    assert abs(pod_left_clearance - pod_right_clearance) < 1e-12, (
        "POD-to-arrow clearances must be symmetric: "
        f"left={pod_left_clearance:.12f}, right={pod_right_clearance:.12f}"
    )

    ax_state = fig.add_axes(pod_bbox, zorder=12)
    pod_center_x_actual = (
        ax_state.get_position().x0
        + 0.5 * ax_state.get_position().width
    )

    assert abs(pod_center_x_actual - pod_expected_center_x) < 0.002, (
        f"POD glyph is not centered: x={pod_center_x_actual:.6f}"
    )
    draw_pod_state(fig, ax_state, z_tilde, podmap, pod_norm, title_center_x=pod_expected_center_x)
    draw_horizontal_colorbar(
        fig,
        pod_cbar_bbox,
        podmap,
        pod_norm,
        r"$\hat z_k / \sqrt{\lambda_k}$",
        scaled_ticks=False,
        nbins=3,
    )

    displacement_bbox = [0.6440, 0.385, 0.350, 0.360]
    ax_disp = fig.add_axes(displacement_bbox, zorder=12)
    current_y = draw_displacement_axis(ax_disp, raw_time_s, cyl_y_over_d, frame=frame, time_s=time_s)

    # ------------------------------------------------------------
    # Two broad, symmetric input arrows.
    # POD -> b-e connector is intentionally omitted for now.
    # ------------------------------------------------------------
    assert abs((left_arrow_end - left_arrow_start) - (right_arrow_start - right_arrow_end)) < 1e-12

    # Sparse velocity observations -> POD
    overlay_block_arrow(
        overlay,
        (left_arrow_start, input_arrow_y),
        (left_arrow_end, input_arrow_y),
        color="#3DA4BA",
        alpha=0.98,
        zorder=35,
    )

    # Cylinder displacement -> POD
    overlay_block_arrow(
        overlay,
        (right_arrow_start, input_arrow_y),
        (right_arrow_end, input_arrow_y),
        color="#3DA4BA",
        alpha=0.98,
        zorder=35,
    )





    args.outdir.mkdir(parents=True, exist_ok=True)
    base = args.outdir / "figure5a_x40y20_0803_spacetime"
    paths = save_exact(fig, base)
    plt.close(fig)

    png_size = image_size(paths[".png"])
    tiff_size = image_size(paths[".tiff"])
    size_exact = png_size == (WIDTH_PX, HEIGHT_PX) and tiff_size == (WIDTH_PX, HEIGHT_PX)

    spatiotemporal_records = [
        {
            "role": role,
            "frame": int(stack_frame),
            "time_s": float(raw_time_s[stack_frame]),
            "temporal_separation_to_current_s": float(time_s - raw_time_s[stack_frame]),
            "cylinder_y_over_d": float(cyl_y_over_d[stack_frame]),
        }
        for role, stack_frame in zip(("back", "middle", "front/current"), stack_frames)
    ]
    future_frame_check = {
        "all_frames_lte_current": bool(all(v <= frame for v in stack_frames)),
        "max_frame_equals_current": bool(max(stack_frames) == frame),
    }

    metadata = {
        "figure": "figure5a_x40y20_0803_spacetime",
        "core_conclusion": "Case-0803 sparse u/v history and measured cylinder displacement constrain the 256-dimensional POD state feeding panels b-e.",
        "archetype": "schematic-led composite with data-native sparse observation and state glyphs",
        "backend": "Python/matplotlib",
        "design": "two pseudo-3D spatiotemporal sparse-observation slabs, centered true APCE POD state, boxed measured cylinder displacement axis",
        "case_id": "0803",
        "reduced_velocity": 8.03,
        "display_frame_source": str(args.bc_frame),
        "display_frame": frame,
        "time_s": time_s,
        "width_px": WIDTH_PX,
        "height_px": HEIGHT_PX,
        "dpi": DPI,
        "export_uses_tight_bbox": False,
        "final_pixel_size_changed": False,
        "spatiotemporal_frames": spatiotemporal_records,
        "spatiotemporal_future_frame_check": future_frame_check,
        "spatiotemporal_layer_offsets": {
            "dx_fig": layer_dx,
            "dy_fig": layer_dy,
            "dx_px": layer_dx * WIDTH_PX,
            "dy_px": layer_dy * HEIGHT_PX,
            "front_to_back_dx_px": 2.0 * layer_dx * WIDTH_PX,
            "front_to_back_dy_px": 2.0 * layer_dy * HEIGHT_PX,
        },
        "spatial_plane_aspect": {
            "x_span_over_d": x_span,
            "y_span_over_d": y_span,
            "data_aspect": data_aspect,
            "front_width_px": front_w * WIDTH_PX,
            "front_height_px": front_h * HEIGHT_PX,
            "rendered_pixel_aspect": rendered_aspect,
            "relative_error": aspect_rel_error,
        },
        "coordinate_triad": {
            "count": 1,
            "location": "upper-left outside sparse observation slabs",
        },
        "cylinder_displacement_source": str(args.raw),
        "cylinder_displacement_current_y_over_d": current_y,
        "cylinder_displacement_display_window_s": [float(raw_time_s.min()), float(raw_time_s.max())],
        "figure_text_policy": {
            "grey_explanatory_text_removed": True,
            "layout_counts_hidden_from_panel": True,
            "t_b_label_shown": False,
            "full_field_caption_text_shown": False,
        },
        "colorbars_shown": {
            "sparse_u": True,
            "sparse_v": True,
            "pod_state": True,
        },
        "pod_colormap": "pod_indigo_ivory_ochre",
        "pod_to_be_arrow_shown": False,
        "current_time_marker_shown_on_displacement": False,
        "input_arrow_style": "filled symmetric block arrows",
        "observation_layout_summary": {
            "layout": "adaptive_fullfield_valid_x40y20",
            "nominal": "20x40",
            "nominal_points": 800,
            "effective_spatial_locations": int(sensor_uv.shape[0]),
            "full_field_spatial_locations": full_field_spatial_locations,
            "spatial_coverage_percent": spatial_coverage_percent,
            "scalar_observations": int(sensor_uv.size),
            "valid_fraction": float(np.asarray(layout["valid_fraction"])),
            "source": str(args.layout),
        },
        "color_scaling": {
            "u": {
                "logic": "asymmetric robust 0.5-99.5 percentile bounds at display frame, inherited from b-e convention",
                **norm_metadata(u_norm),
            },
            "v": {
                "logic": "symmetric zero-centered robust 99.5 percentile bound at display frame, inherited from b-e convention",
                **norm_metadata(v_norm),
            },
            "colormap": "blue_warmwhite_red copied from current b-e script",
            "display_multiplier": "x10^-1",
        },
        "pod_state_glyph": {
            "source": str(args.trace),
            "key": "latent_estimate",
            "pod_model_source": str(args.pod),
            "model_manifest_source": str(args.model_manifest),
            "coefficient_training_source": str(args.coefficients_dir),
            "pod_state_definition": "APCE analysis-state estimate at frame 554 from latent_estimate",
            "pod_scaling_definition": "sqrt(lambda_k) = singular_values[:256] / sqrt(sample_count - 1); displayed value is z_hat_k / sqrt(lambda_k)",
            "frame": frame,
            "trace_time_s": float(np.asarray(trace["time_s"], dtype=float)[frame]),
            "vector_shape": list(latent_t.shape),
            "layout": "16x16 row-major colored circles; mode 1 top-left and mode 256 bottom-right",
            "color_norm": {
                "logic": "fixed standardized coefficient diverging scale; not single-frame min/max stretching",
                "vmin": -POD_COLOR_LIMIT,
                "vcenter": 0.0,
                "vmax": POD_COLOR_LIMIT,
            },
            "raw_z_stats": {
                "min": float(np.nanmin(latent_t)),
                "max": float(np.nanmax(latent_t)),
                "mean": float(np.nanmean(latent_t)),
            },
            "standardized_z_stats": {
                "min": float(np.nanmin(z_tilde)),
                "max": float(np.nanmax(z_tilde)),
                "mean": float(np.nanmean(z_tilde)),
            },
            "sqrt_lambda_stats": {
                "min": float(np.nanmin(np.asarray(pod_state["sqrt_lambda"], dtype=float))),
                "max": float(np.nanmax(np.asarray(pod_state["sqrt_lambda"], dtype=float))),
            },
            "sample_count": int(pod_state["sample_count"]),
            "training_standardized_abs_percentiles": pod_state["train_abs_percentiles"],
            "apce_trace_standardized_abs_percentiles": pod_state["trace_abs_percentiles"],
            "coefficient_file_count": int(pod_state["coefficient_file_count"]),
        },
        "layout_geometry": {
            "pod_glyph_center_x": float(pod_center_x_actual),
            "pod_group_shift_x": pod_bbox_x - 0.435,
            "pod_glyph_bbox": pod_bbox,
            "pod_colorbar_bbox": pod_cbar_bbox,
            "pod_arrow_clearance_left_fig": float(pod_left_clearance),
            "pod_arrow_clearance_right_fig": float(pod_right_clearance),
            "u_slab": u_slab,
            "v_slab": v_slab,
            "sparse_group_shift_x": sparse_dx,
            "coordinate_triad_origin": list(triad_origin),
            "input_arrows": {
                "length_fig": input_arrow_length,
                "left_start": [left_arrow_start, input_arrow_y],
                "left_end": [left_arrow_end, input_arrow_y],
                "right_start": [right_arrow_start, input_arrow_y],
                "right_end": [right_arrow_end, input_arrow_y],
                "equal_geometry_asserted": True,
            },
            "displacement_bbox": displacement_bbox,
            "displacement_right_edge_aligned_to_c_right_panel_px": displacement_bbox[0] + displacement_bbox[2],
        },
        "finite_counts": {
            "sparse_u_front": int(np.isfinite(u_grid).sum()),
            "sparse_v_front": int(np.isfinite(v_grid).sum()),
            "latent_state": int(np.isfinite(latent_t).sum()),
            "standardized_latent_state": int(np.isfinite(z_tilde).sum()),
        },
        "outputs": {ext: str(path) for ext, path in paths.items()},
        "exported_file_sizes_bytes": {ext: file_size(path) for ext, path in paths.items()},
        "raster_dimensions": {".png": list(png_size), ".tiff": list(tiff_size)},
        "output_size_exactly_matches_metadata": bool(size_exact),
        "output_sha256": {ext: sha256_file(path) for ext, path in paths.items()},
    }
    meta_path = base.with_name(base.name + "_metadata.json")
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    audit_lines = [
        f"case_id={metadata['case_id']}",
        f"reduced_velocity={metadata['reduced_velocity']}",
        f"display_frame={frame}",
        f"time_s={time_s:.6f}",
        f"width_px={WIDTH_PX}",
        f"height_px={HEIGHT_PX}",
        f"dpi={DPI}",
        f"effective_point_count={sensor_uv.shape[0]}",
        f"scalar_observations={sensor_uv.size}",
        "spatiotemporal_frames=" + ",".join(str(v) for v in stack_frames),
        "spatiotemporal_times_s=" + ",".join(f"{raw_time_s[v]:.6f}" for v in stack_frames),
        "temporal_separation_to_current_s=" + ",".join(f"{time_s - raw_time_s[v]:.6f}" for v in stack_frames),
        f"all_frames_lte_current={future_frame_check['all_frames_lte_current']}",
        f"max_frame_equals_current={future_frame_check['max_frame_equals_current']}",
        f"layer_dx_px={layer_dx * WIDTH_PX:.2f}",
        f"layer_dy_px={layer_dy * HEIGHT_PX:.2f}",
        f"plane_front_width_px={front_w * WIDTH_PX:.2f}",
        f"plane_front_height_px={front_h * HEIGHT_PX:.2f}",
        f"data_aspect={data_aspect:.8f}",
        f"rendered_pixel_aspect={rendered_aspect:.8f}",
        f"aspect_relative_error={aspect_rel_error:.8e}",
        "coordinate_triad_count=1",
        "coordinate_triad_location=upper-left outside sparse observation slabs",
        "pod_colormap=pod_indigo_ivory_ochre",
        "pod_to_be_arrow_shown=False",
        "current_time_marker_shown_on_displacement=False",
        "input_arrow_style=filled symmetric block arrows",
        f"finite_sparse_u_front={int(np.isfinite(u_grid).sum())}",
        f"finite_sparse_v_front={int(np.isfinite(v_grid).sum())}",
        f"pod_state_source={args.trace}",
        "pod_state_definition=APCE analysis-state estimate at frame 554 from latent_estimate",
        "pod_scaling_definition=sqrt(lambda_k)=singular_values[:256]/sqrt(sample_count-1); displayed z_hat_k/sqrt(lambda_k)",
        f"pod_state_shape={latent_t.shape}",
        f"pod_z_min={float(np.nanmin(latent_t)):.6f}",
        f"pod_z_max={float(np.nanmax(latent_t)):.6f}",
        f"pod_ztilde_min={float(np.nanmin(z_tilde)):.6f}",
        f"pod_ztilde_max={float(np.nanmax(z_tilde)):.6f}",
        f"pod_color_norm=[{-POD_COLOR_LIMIT:.1f},0,{POD_COLOR_LIMIT:.1f}]",
        f"training_standardized_abs_p99_5={metadata['pod_state_glyph']['training_standardized_abs_percentiles'].get('99.5', 'NA')}",
        f"grey_explanatory_text_removed=True",
        f"colorbars_shown=True",
        f"t_b_label_shown=False",
        f"png_size={png_size[0]}x{png_size[1]}",
        f"tiff_size={tiff_size[0]}x{tiff_size[1]}",
        f"output_size_exactly_matches_metadata={size_exact}",
    ]
    audit_path = base.with_name(base.name + "_audit.txt")
    audit_path.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "metadata": str(meta_path),
                "audit": str(audit_path),
                "size_exact": size_exact,
                "png_size": png_size,
                "spatiotemporal_frames": stack_frames,
                "spatiotemporal_times_s": [float(raw_time_s[v]) for v in stack_frames],
                "pod_state_source": str(args.trace),
                "pod_z_minmax": [float(np.nanmin(latent_t)), float(np.nanmax(latent_t))],
                "pod_ztilde_minmax": [float(np.nanmin(z_tilde)), float(np.nanmax(z_tilde))],
                "outputs": metadata["outputs"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
