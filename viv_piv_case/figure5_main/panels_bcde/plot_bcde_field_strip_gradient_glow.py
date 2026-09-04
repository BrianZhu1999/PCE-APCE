"""Render Figure 5 panels b-e for the corrected U_r=8.03 case.

Panel contract:
    b: upper-left 2 x 2 reference/APCE streamwise velocity fields
    c: upper-right 2 x 2 reference/APCE cross-stream velocity fields
    d: lower-left reference/APCE streamwise forecast fields
    e: lower-right reference/APCE cross-stream forecast fields

The displayed snapshots are selected by an explicit minimum-MSE rule recorded
in ``source_0803/best_snapshot_selection.json``. This script is independent
of the Illustrator-authored full-width panel a.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Circle, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(__file__).resolve().parent / "source_0803"
OUT_ROOT = Path(__file__).resolve().parent / "outputs"

FIG_W_PX = 12685
# Keep the audited 11,532 px width.  The 1.2x enlargement is applied to the
# vertical cloud-panel dimension; horizontal enlargement is geometrically
# impossible for four columns within this fixed motherboard.
FIG_H_PX = 6400
DPI = 650
FIG_W_IN = FIG_W_PX / DPI
FIG_H_IN = FIG_H_PX / DPI
CASE_ID = "0803"
SEED = 0


def _configure() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "font.weight": "normal",
        "axes.titleweight": "normal",
        "axes.labelweight": "normal",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.75,
        "axes.spines.top": True,
        "axes.spines.right": True,
    })


def _load_field(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as raw:
        reference = np.asarray(raw["reference"], dtype=float)
        apce = np.asarray(raw["apce"], dtype=float)
        return {
            "time_s": float(raw["time_s"]) if "time_s" in raw else float(raw["target_time_s"]),
            "target_time_s": float(raw["target_time_s"]) if "target_time_s" in raw else None,
            "origin_time_s": float(raw["origin_time_s"]) if "origin_time_s" in raw else None,
            "horizon_s": float(raw["horizon_s"]) if "horizon_s" in raw else None,
            "x": np.asarray(raw["x_over_d"], dtype=float),
            "y": np.asarray(raw["y_over_d"], dtype=float),
            "valid": np.asarray(raw["valid"], dtype=bool),
            "reference_u": reference[..., 0],
            "reference_v": reference[..., 1],
            "apce_u": apce[..., 0],
            "apce_v": apce[..., 1],
            "cylinder_y_over_d": float(raw["cylinder_y_over_d"]) if "cylinder_y_over_d" in raw else 0.0,
        }


def _asymmetric_norm(fields: list[np.ndarray], valid: np.ndarray) -> Normalize | TwoSlopeNorm:
    values = np.concatenate([field[valid & np.isfinite(field)] for field in fields])
    low = float(np.nanpercentile(values, 0.5))
    high = max(1e-6, float(np.nanpercentile(values, 99.5)))
    if low >= 0.0:
        return Normalize(vmin=low, vmax=high)
    return TwoSlopeNorm(vmin=low, vcenter=0.0, vmax=high)


def _symmetric_norm(fields: list[np.ndarray], valid: np.ndarray) -> TwoSlopeNorm:
    values = np.concatenate([field[valid & np.isfinite(field)] for field in fields])
    limit = max(1e-6, float(np.nanpercentile(np.abs(values), 99.5)))
    return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def _field_axis(
    axis: plt.Axes,
    field: np.ndarray,
    record: dict[str, object],
    norm: TwoSlopeNorm,
    cmap: str,
) -> mpl.cm.ScalarMappable:
    x = np.asarray(record["x"], dtype=float)
    y = np.asarray(record["y"], dtype=float)
    valid = np.asarray(record["valid"], dtype=bool)
    shown = np.ma.masked_where(~valid | ~np.isfinite(field), field)
    image = axis.pcolormesh(x, y, shown, shading="auto", cmap=cmap, norm=norm, rasterized=True)
    axis.add_patch(
        Circle(
            (0.0, float(record["cylinder_y_over_d"])),
            0.5,
            facecolor="#F7F7F7",
            edgecolor="#222222",
            linewidth=0.95,
            zorder=5,
        )
    )
    axis.set_xlim(float(x[0]), float(x[-1]))
    axis.set_ylim(float(y[0]), float(y[-1]))
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([-1, 2, 5, 8])
    axis.set_yticks([-2, 0, 2])
    axis.tick_params(axis="both", labelsize=11, length=2.8, width=0.65, pad=2)
    return image


def _add_group_colorbar(fig: plt.Figure, axes: list[plt.Axes], norm: Normalize | TwoSlopeNorm, cmap: str, label: str) -> None:
    fig.canvas.draw()
    left = min(axis.get_position().x0 for axis in axes)
    right = max(axis.get_position().x1 for axis in axes)
    bottom = min(axis.get_position().y0 for axis in axes)
    # Keep the colorbar at its natural position below the d/e panels.
    # Canvas height is extended during save instead of moving this bar upward.
    cbar_bottom = max(0.022, bottom - 0.085)
    cax = fig.add_axes(
        [left, cbar_bottom, right - left, 0.021]
    )
    cax.set_facecolor((1.0, 1.0, 1.0, 0.0))
    cax.patch.set_alpha(0.0)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    colorbar.ax.set_facecolor((1.0, 1.0, 1.0, 0.0))
    colorbar.ax.patch.set_alpha(0.0)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=11, length=2.5, width=0.6, pad=2)
    colorbar.ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    def _format_scaled(value: float, _position: float) -> str:
        scaled = value * 10.0
        if abs(scaled - round(scaled)) < 1e-9:
            return str(int(round(scaled)))
        return f"{scaled:.2f}".rstrip("0").rstrip(".")
    colorbar.ax.xaxis.set_major_formatter(FuncFormatter(_format_scaled))
    colorbar.set_label(label, fontsize=13, labelpad=2)
    for spine in colorbar.ax.spines.values():
        spine.set_visible(False)



def _add_background_gradient(
    fig: plt.Figure,
    y0: float,
    y1: float,
    color_bottom: str,
    color_top: str,
    zorder: float = -19.0,
) -> plt.Axes:
    """Add a subtle vertical gradient without changing layout geometry."""
    from matplotlib.colors import to_rgba

    if y1 <= y0:
        raise ValueError(
            f"Invalid gradient region: y0={y0}, y1={y1}"
        )

    n = 512

    rgba_bottom = np.asarray(
        to_rgba(color_bottom),
        dtype=float,
    )

    rgba_top = np.asarray(
        to_rgba(color_top),
        dtype=float,
    )

    t = np.linspace(
        0.0,
        1.0,
        n,
        dtype=float,
    )[:, None]

    rgba = (
        rgba_bottom[None, :]
        * (1.0 - t)
        + rgba_top[None, :]
        * t
    )

    image = np.repeat(
        rgba[:, None, :],
        8,
        axis=1,
    )

    gradient_ax = fig.add_axes(
        [0.0, y0, 1.0, y1 - y0],
        zorder=zorder,
    )

    gradient_ax.imshow(
        image,
        origin="lower",
        aspect="auto",
        interpolation="bicubic",
        extent=[0.0, 1.0, 0.0, 1.0],
    )

    gradient_ax.set_axis_off()

    return gradient_ax


def _tint_toward_white(color: str, scale: float) -> str:
    """Scale a decorative tint's distance from white without touching data colours."""
    from matplotlib.colors import to_hex, to_rgb

    rgb = np.asarray(to_rgb(color), dtype=float)
    return to_hex(1.0 - scale * (1.0 - rgb))


def _add_soft_vignette(
    fig: plt.Figure,
    y0: float,
    y1: float,
    color: str,
    edge_alpha: float = 0.035,
    zorder: float = -18.5,
) -> plt.Axes:
    """Add a subtle rectangular vignette near the edges only."""
    from matplotlib.colors import to_rgba

    if y1 <= y0:
        raise ValueError(
            f"Invalid vignette region: y0={y0}, y1={y1}"
        )

    ny = 512
    nx = 512

    y = np.linspace(-1.0, 1.0, ny, dtype=float)[:, None]
    x = np.linspace(-1.0, 1.0, nx, dtype=float)[None, :]

    # Rectangular edge-distance measure:
    # 0 at center, 1 near the outer boundary.
    edge = np.maximum(np.abs(x), np.abs(y))

    # Smoothstep-like ramp: no effect in the center,
    # gentle darkening only near edges.
    t = np.clip((edge - 0.58) / (1.0 - 0.58), 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)

    rgba = np.zeros((ny, nx, 4), dtype=float)
    base_rgba = np.asarray(to_rgba(color), dtype=float)

    rgba[..., 0] = base_rgba[0]
    rgba[..., 1] = base_rgba[1]
    rgba[..., 2] = base_rgba[2]
    rgba[..., 3] = edge_alpha * t

    vignette_ax = fig.add_axes(
        [0.0, y0, 1.0, y1 - y0],
        zorder=zorder,
    )

    vignette_ax.imshow(
        rgba,
        origin="lower",
        aspect="auto",
        interpolation="bicubic",
        extent=[0.0, 1.0, 0.0, 1.0],
    )

    vignette_ax.set_axis_off()
    return vignette_ax



def _add_local_soft_glow(
    fig: plt.Figure,
    y0: float,
    y1: float,
    title_y: float,
    center_x: float = 0.50,
    glow_color: str = "#FFFFFF",
    max_alpha: float = 0.050,
    sigma_x: float = 0.22,
    sigma_y: float = 0.10,
    zorder: float = -18.8,
) -> plt.Axes:
    """Add a restrained elliptical Gaussian glow behind a stage title."""
    from matplotlib.colors import to_rgba

    if y1 <= y0:
        raise ValueError(
            f"Invalid glow region: y0={y0}, y1={y1}"
        )

    nx = 768
    ny = 384

    x = np.linspace(
        0.0,
        1.0,
        nx,
        dtype=float,
    )[None, :]

    y = np.linspace(
        0.0,
        1.0,
        ny,
        dtype=float,
    )[:, None]

    # Convert the title's figure-coordinate y position into
    # the local [0, 1] coordinate of this background region.
    center_y = (
        (title_y - y0)
        / max(y1 - y0, 1e-12)
    )

    center_y = float(
        np.clip(center_y, 0.0, 1.0)
    )

    gaussian = np.exp(
        -0.5
        * (
            ((x - center_x) / sigma_x) ** 2
            + ((y - center_y) / sigma_y) ** 2
        )
    )

    base_rgba = np.asarray(
        to_rgba(glow_color),
        dtype=float,
    )

    rgba = np.zeros(
        (ny, nx, 4),
        dtype=float,
    )

    rgba[..., 0] = base_rgba[0]
    rgba[..., 1] = base_rgba[1]
    rgba[..., 2] = base_rgba[2]

    rgba[..., 3] = (
        max_alpha
        * gaussian
    )

    glow_ax = fig.add_axes(
        [0.0, y0, 1.0, y1 - y0],
        zorder=zorder,
    )

    glow_ax.imshow(
        rgba,
        origin="lower",
        aspect="auto",
        interpolation="bicubic",
        extent=[0.0, 1.0, 0.0, 1.0],
    )

    glow_ax.set_axis_off()

    return glow_ax


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT_ROOT)
    parser.add_argument(
        "--plain-background",
        action="store_true",
        help="Render the b-e board on a pure-white canvas without stage background tints, gradients, glow, or vignette.",
    )
    parser.add_argument(
        "--background-tint-scale",
        type=float,
        default=1.0,
        help="Scale decorative background tint strength relative to the original (0=white; 1=original).",
    )
    parser.add_argument(
        "--background-palette",
        choices=("semantic-blue-rose", "coordinated-mist"),
        default="semantic-blue-rose",
        help="Decorative stage palette; does not alter field colours or data scaling.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.background_tint_scale <= 1.0:
        raise ValueError("--background-tint-scale must lie in [0, 1]")
    tint = lambda color: _tint_toward_white(color, args.background_tint_scale)
    _configure()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.background_palette == "coordinated-mist":
        # A single low-chroma blue-to-sage sequence keeps the two stages
        # distinct without competing with the scientific blue-red field map.
        stage_colours = {
            "upper_base": "#F5F9FA",
            "lower_base": "#F8FAF8",
            "upper_underlay": "#F7FAFA",
            "lower_underlay": "#F9FAF9",
            "upper_bottom": "#F7FAFA",
            "upper_top": "#EAF3F5",
            "lower_bottom": "#EEF5F3",
            "lower_top": "#FAFBFA",
            "upper_vignette": "#A3B8BC",
            "lower_vignette": "#AEBDB8",
        }
    else:
        stage_colours = {
            "upper_base": "#F6FAFC",
            "lower_base": "#FDF8F6",
            "upper_underlay": "#F8FBFD",
            "lower_underlay": "#FEFAF8",
            "upper_bottom": "#FAFCFD",
            "upper_top": "#EAF3F8",
            "lower_bottom": "#F8EAE5",
            "lower_top": "#FEFAF8",
            "upper_vignette": "#9FB5C5",
            "lower_vignette": "#C7AEA6",
        }

    first = _load_field(SOURCE_ROOT / "best_reconstruction_frame_0554.npz")
    second = _load_field(SOURCE_ROOT / "best_reconstruction_frame_0997.npz")
    forecast = _load_field(SOURCE_ROOT / "best_forecast_4s.npz")
    top_records = [first, second]
    records = top_records + [forecast]
    valid = np.asarray(first["valid"], dtype=bool)
    u_norm = _asymmetric_norm(
        [np.asarray(record[key]) for record in records for key in ("reference_u", "apce_u")], valid
    )
    v_norm = _symmetric_norm(
        [np.asarray(record[key]) for record in records for key in ("reference_v", "apce_v")], valid
    )
    # Custom journal-style diverging palette:
    # low/negative = blue, zero/mid = warm white, high/positive = red.
    field_cmap = mpl.colors.LinearSegmentedColormap.from_list(
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
    cmap_u = field_cmap
    cmap_v = field_cmap

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI, facecolor="white")
    grid = fig.add_gridspec(
        nrows=3,
        ncols=4,
        height_ratios=[1.0, 1.0, 1.0],
        left=0.015,
        right=0.995,
        top=0.90,
        bottom=0.145,
        wspace=0.0,
        hspace=0.82,
    )

    # Optional stage backgrounds distinguish available and withheld observations.
    # The plain variant deliberately removes this decorative layer while leaving
    # all scientific field, label, and layout content unchanged.
    upper_background = Rectangle(
        (0.0, 0.335),
        1.0,
        0.665,
        transform=fig.transFigure,
        facecolor="#FFFFFF" if args.plain_background else tint(stage_colours["upper_base"]),
        edgecolor="none",
        zorder=-20,
    )
    lower_background = Rectangle(
        (0.0, 0.0),
        1.0,
        0.335,
        transform=fig.transFigure,
        facecolor="#FFFFFF" if args.plain_background else tint(stage_colours["lower_base"]),
        edgecolor="none",
        zorder=-20,
    )
    fig.add_artist(upper_background)
    fig.add_artist(lower_background)

    axes: dict[str, list[plt.Axes]] = {"b": [], "c": [], "d": [], "e": []}
    for row in range(2):
        axes["b"].extend([fig.add_subplot(grid[row, 0]), fig.add_subplot(grid[row, 1])])
        axes["c"].extend([fig.add_subplot(grid[row, 2]), fig.add_subplot(grid[row, 3])])
    axes["d"] = [fig.add_subplot(grid[2, 0]), fig.add_subplot(grid[2, 1])]
    axes["e"] = [fig.add_subplot(grid[2, 2]), fig.add_subplot(grid[2, 3])]

    for row, record in enumerate(top_records):
        for col, (key, title, norm) in enumerate((("reference_u", "Ref.", u_norm), ("apce_u", "APCE", u_norm))):
            axis = axes["b"][row * 2 + col]
            _field_axis(axis, np.asarray(record[key]), record, norm, cmap_u)
            axis.set_title(f"{title}  ($t={float(record['time_s']):.2f}$ s)", fontsize=14, pad=7, fontweight="normal")
            axis.set_ylabel(r"$y/D$", fontsize=13, labelpad=4)
            axis.set_xlabel(r"$x/D$", fontsize=13, labelpad=3)

        for col, (key, title, norm) in enumerate((("reference_v", "Ref.", v_norm), ("apce_v", "APCE", v_norm))):
            axis = axes["c"][row * 2 + col]
            _field_axis(axis, np.asarray(record[key]), record, norm, cmap_v)
            axis.set_title(f"{title}  ($t={float(record['time_s']):.2f}$ s)", fontsize=14, pad=7, fontweight="normal")
            axis.set_ylabel(r"$y/D$", fontsize=13, labelpad=4)
            axis.set_xlabel(r"$x/D$", fontsize=13, labelpad=3)

    for panel, keys, norm in (("d", ("reference_u", "apce_u"), u_norm), ("e", ("reference_v", "apce_v"), v_norm)):
        for col, (key, title) in enumerate(zip(keys, ("Ref.", "APCE"))):
            axis = axes[panel][col]
            _field_axis(axis, np.asarray(forecast[key]), forecast, norm, cmap_u if panel == "d" else cmap_v)
            axis.set_title(f"{title}  ($t={float(forecast['target_time_s']):.2f}$ s)", fontsize=14, pad=7, fontweight="normal")
            axis.set_ylabel(r"$y/D$", fontsize=13, labelpad=4)
            axis.set_xlabel(r"$x/D$", fontsize=13, labelpad=3)


    # BEGIN COLUMN COMPRESSION
    # GridSpec wspace is already zero.  Because each field axis preserves an
    # equal physical data aspect, unused horizontal room remains inside each
    # GridSpec cell.  Shift the four columns inward without distorting fields.
    fig.canvas.draw()

    column_shifts = {
        0: +0.018,
        1: +0.006,
        2: -0.006,
        3: -0.018,
    }

    column_axes = {
        0: [axes["b"][0], axes["b"][2], axes["d"][0]],
        1: [axes["b"][1], axes["b"][3], axes["d"][1]],
        2: [axes["c"][0], axes["c"][2], axes["e"][0]],
        3: [axes["c"][1], axes["c"][3], axes["e"][1]],
    }

    for col, col_axes in column_axes.items():
        dx = column_shifts[col]
        for axis in col_axes:
            pos = axis.get_position()
            axis.set_position(
                [pos.x0 + dx, pos.y0, pos.width, pos.height],
                which="both",
            )

    fig.canvas.draw()
    # END COLUMN COMPRESSION

    # BEGIN INTERNAL PAIR GAP
    fig.canvas.draw()

    pair_gap_delta = 0.035

    # b and d: move only Ref. panels right
    for axis in (
        axes["b"][0],
        axes["b"][2],
        axes["d"][0],
    ):
        pos = axis.get_position()
        axis.set_position(
            [pos.x0 + pair_gap_delta, pos.y0, pos.width, pos.height],
            which="both",
        )

    # c and e: move only APCE panels left
    for axis in (
        axes["c"][1],
        axes["c"][3],
        axes["e"][1],
    ):
        pos = axis.get_position()
        axis.set_position(
            [pos.x0 - pair_gap_delta, pos.y0, pos.width, pos.height],
            which="both",
        )

    fig.canvas.draw()
    # END INTERNAL PAIR GAP



    # BEGIN DE ROW DROP
    fig.canvas.draw()

    # Stable d/e vertical offset.
    de_row_drop = 0.050

    for axis in (
        axes["d"][0],
        axes["d"][1],
        axes["e"][0],
        axes["e"][1],
    ):
        pos = axis.get_position()
        axis.set_position(
            [pos.x0, pos.y0 - de_row_drop, pos.width, pos.height],
            which="both",
        )

    fig.canvas.draw()
    # END DE ROW DROP

    # Derive label positions from the actual axes tops.
    # b/c and d/e use identical title/label offsets.
    # The two tinted regions remain continuous.
    fig.canvas.draw()

    label_offset = 200 / FIG_H_PX
    title_label_clearance = 0.04

    top_axis_y = axes["b"][0].get_position().y1
    bottom_axis_y = axes["d"][0].get_position().y1

    top_label_y = top_axis_y + label_offset
    bottom_label_y = bottom_axis_y + label_offset

    # Identical large-title-to-panel-label spacing.
    top_title_y = top_label_y + title_label_clearance
    bottom_title_y = bottom_label_y + title_label_clearance

    # Identical title-to-background-top spacing.
    title_background_margin = 1.0 - top_title_y

    # Shared continuous blue/red boundary.
    background_split_y = (
        bottom_title_y
        + title_background_margin
    )

    background_split_y = min(
        max(background_split_y, 0.05),
        0.95,
    )

    # Upper blue block:
    # background_split_y -> 1.0
    upper_background.set_y(background_split_y)
    upper_background.set_height(
        1.0 - background_split_y
    )

    # Lower red block:
    # 0.0 -> background_split_y
    lower_background.set_y(0.0)
    lower_background.set_height(
        background_split_y
    )

    # Consistent horizontal clearance for panel labels b/c/d/e.
    # BEGIN GRADIENT BACKGROUND OVERLAY
    # Read the FINAL geometry of the existing background regions.
    upper_bg_y0 = float(upper_background.get_y())
    upper_bg_y1 = float(
        upper_background.get_y()
        + upper_background.get_height()
    )

    lower_bg_y0 = float(lower_background.get_y())
    lower_bg_y1 = float(
        lower_background.get_y()
        + lower_background.get_height()
    )

    if not args.plain_background:
        # Neutral underlay beneath the gradients.
        upper_background.set_facecolor(tint(stage_colours["upper_underlay"]))
        lower_background.set_facecolor(tint(stage_colours["lower_underlay"]))

        _add_background_gradient(
            fig, upper_bg_y0, upper_bg_y1,
            color_bottom=tint(stage_colours["upper_bottom"]), color_top=tint(stage_colours["upper_top"]), zorder=-19.0,
        )
        _add_background_gradient(
            fig, lower_bg_y0, lower_bg_y1,
            color_bottom=tint(stage_colours["lower_bottom"]), color_top=tint(stage_colours["lower_top"]), zorder=-19.0,
        )
        _add_local_soft_glow(
            fig, upper_bg_y0, upper_bg_y1, title_y=top_title_y, center_x=0.50,
            glow_color="#FFFFFF", max_alpha=0.050 * args.background_tint_scale, sigma_x=0.23, sigma_y=0.105, zorder=-18.8,
        )
        _add_local_soft_glow(
            fig, lower_bg_y0, lower_bg_y1, title_y=bottom_title_y, center_x=0.50,
            glow_color="#FFFFFF", max_alpha=0.046 * args.background_tint_scale, sigma_x=0.23, sigma_y=0.105, zorder=-18.8,
        )
        _add_soft_vignette(
            fig, upper_bg_y0, upper_bg_y1, color=tint(stage_colours["upper_vignette"]), edge_alpha=0.030 * args.background_tint_scale, zorder=-18.5,
        )
        _add_soft_vignette(
            fig, lower_bg_y0, lower_bg_y1, color=tint(stage_colours["lower_vignette"]), edge_alpha=0.028 * args.background_tint_scale, zorder=-18.5,
        )

    panel_label_gap = 0.022

    b_label_x = axes["b"][0].get_position().x0 - panel_label_gap
    c_label_x = axes["c"][0].get_position().x0 - panel_label_gap
    d_label_x = axes["d"][0].get_position().x0 - panel_label_gap
    e_label_x = axes["e"][0].get_position().x0 - panel_label_gap

    # ------------------------------------------------------------
    # Temporal-evolution ellipses for observation-conditioned panels
    #
    # upper ellipsis row:
    #   between the first and second displayed observation-conditioned rows
    #
    # lower ellipsis row:
    #   below the second observation-conditioned row
    #
    # The lower x/D-to-ellipsis distance is forced to equal the upper
    # x/D-to-ellipsis distance, using actual text bounding boxes.
    # ------------------------------------------------------------
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    ellipsis_axes = [
        axes["b"][0],
        axes["b"][1],
        axes["c"][0],
        axes["c"][1],
    ]
    ellipsis_x = [
        0.5 * (axis.get_position().x0 + axis.get_position().x1)
        for axis in ellipsis_axes
    ]

    # Upper ellipsis row stays between row 1 and row 2.
    upper_first_bottom = axes["b"][0].get_position().y0
    upper_second_top = axes["b"][2].get_position().y1
    ellipsis_y = 0.5 * (upper_first_bottom + upper_second_top)

    # Use the actual x/D label bounding boxes to mirror the spacing.
    upper_xlabel_box = axes["b"][0].xaxis.label.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
    lower_xlabel_box = axes["b"][2].xaxis.label.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())

    # Distance from the top-row x/D label bottom edge to the upper ellipsis center.
    ellipsis_gap_to_xlabel = upper_xlabel_box.y0 - ellipsis_y

    # Force the lower ellipsis row to have the SAME distance to the second-row x/D label.
    lower_ellipsis_y_nominal = (
        lower_xlabel_box.y0
        - ellipsis_gap_to_xlabel
    )

    # Keep the ellipsis inside the blue region.
    lower_ellipsis_y = max(
        lower_ellipsis_y_nominal,
        background_split_y + 0.018,
    )

    # Also keep a safe gap below the second-row x/D label.
    lower_ellipsis_y = min(
        lower_ellipsis_y,
        lower_xlabel_box.y0 - 0.012,
    )

    for x in ellipsis_x:
        fig.text(
            x,
            ellipsis_y,
            r"$\vdots$",
            ha="center",
            va="center",
            fontsize=24,
            fontweight="bold",
        )
        fig.text(
            x,
            lower_ellipsis_y,
            r"$\vdots$",
            ha="center",
            va="center",
            fontsize=24,
            fontweight="bold",
        )

    fig.text(b_label_x, top_label_y, "b", ha="left", va="center", fontsize=22, fontweight="bold")
    fig.text(c_label_x, top_label_y, "c", ha="left", va="center", fontsize=22, fontweight="bold")
    fig.text(d_label_x, bottom_label_y, "d", ha="left", va="center", fontsize=22, fontweight="bold")
    fig.text(e_label_x, bottom_label_y, "e", ha="left", va="center", fontsize=22, fontweight="bold")
    fig.text(0.50, top_title_y, "Observation-conditioned", ha="center", va="center", fontsize=19, fontweight="bold")
    fig.text(0.50, bottom_title_y, "Observation-free propagation", ha="center", va="center", fontsize=19, fontweight="bold")

    _add_group_colorbar(fig, axes["d"], u_norm, cmap_u, r"$u$ ($\times 10^{-1}$ m s$^{-1}$)")
    _add_group_colorbar(fig, axes["e"], v_norm, cmap_v, r"$v$ ($\times 10^{-1}$ m s$^{-1}$)")


    # BEGIN HORIZONTAL CANVAS CROP
    # Crop ONLY left/right whitespace.
    #
    # Vertically, DO NOT crop the figure at y=0.
    # Extend the saved canvas downward so colorbar tick labels
    # and physical-unit labels remain fully visible.
    crop_left = 0.078
    crop_right = 0.910

    # Extra physical canvas below the original motherboard.
    # Units: inches.
    #
    # Increase this if more bottom breathing room is wanted.
    # Decrease it if the bottom margin looks too large.
    bottom_canvas_extension_in = 0.28

    horizontal_crop_bbox = mpl.transforms.Bbox.from_extents(
        FIG_W_IN * crop_left,
        -bottom_canvas_extension_in,
        FIG_W_IN * crop_right,
        FIG_H_IN,
    )

    save_facecolor = "#FFFFFF" if args.plain_background else tint(stage_colours["lower_bottom"])
    # END HORIZONTAL CANVAS CROP

    base = args.output / "figure5_panels_bcde_x40y20_0803_best_mse_gradient_glow"
    fig.savefig(base.with_suffix(".png"), dpi=DPI, facecolor=save_facecolor, bbox_inches=horizontal_crop_bbox, pad_inches=0.0)
    fig.savefig(base.with_suffix(".tiff"), dpi=DPI, facecolor=save_facecolor, bbox_inches=horizontal_crop_bbox, pad_inches=0.0)
    fig.savefig(base.with_suffix(".pdf"), facecolor=save_facecolor, bbox_inches=horizontal_crop_bbox, pad_inches=0.0)
    fig.savefig(base.with_suffix(".svg"), facecolor=save_facecolor, bbox_inches=horizontal_crop_bbox, pad_inches=0.0)
    plt.close(fig)

    selection = json.loads((SOURCE_ROOT / "best_snapshot_selection.json").read_text(encoding="utf-8"))
    metadata = {
        "figure": "figure5_panels_bcde",
        "panel_roles": {
            "b": "upper-left four panels: Reference/APCE streamwise velocity at two minimum-MSE reconstruction times",
            "c": "upper-right four panels: Reference/APCE cross-stream velocity at the same times",
            "d": "lower-left two panels: Reference/APCE streamwise velocity at the minimum-MSE 4-s forecast endpoint",
            "e": "lower-right two panels: Reference/APCE cross-stream velocity at the minimum-MSE 4-s forecast endpoint",
        },
        "case_id": CASE_ID,
        "reduced_velocity": 8.03,
        "seed": SEED,
        "layout": "corrected x40-y20 adaptive full-field valid mask; 751 effective locations",
        "selection": selection,
        "source_root_local": str(SOURCE_ROOT),
        "source_root_remote": "<HILDA_RESULTS_ROOT>/results/viv_piv_pce_apce_adaptive_valid_x40y20_formal5/figures/figure5_sources_0803/",
        "color_scaling": {
            "u": "asymmetric robust 0.5-99.5 percentile bounds; custom blue-warm-white-red map",
            "v": "symmetric robust 99.5 percentile bound around zero; negative blue, zero warm white, positive red",
            "map": "custom blue-warm-white-red; minimum/negative blue, midpoint warm white, maximum/positive red",
            "display_multiplier": "x10^-1",
            "colorbar_outline": False,
        },
        "backgrounds": {
            "mode": "plain_white" if args.plain_background else "stage_tint_gradient_glow",
            "palette": args.background_palette,
            "tint_scale_relative_to_original": 0.0 if args.plain_background else args.background_tint_scale,
            "available": "#FFFFFF" if args.plain_background else tint(stage_colours["upper_base"]),
            "withheld": "#FFFFFF" if args.plain_background else tint(stage_colours["lower_base"]),
            "covers_stage_titles": True,
            "split_y": background_split_y,
            "title_background_margin": title_background_margin,
        },
        "fonts_pt": {"panel": 22, "panel_weight": "bold", "group_title": 19, "group_title_weight": "bold", "axis": 13, "tick": 11},
        "geometry": {
            "left_margin": 0.040,
            "right_margin": 0.010,
            "wspace": 0.035,
            "colorbar_height": 0.021,
            "vertical_cloud_scale_relative_to_previous": 1.1,
            "motherboard_height_px": FIG_H_PX,
            "de_row_drop": de_row_drop,
            "horizontal_cloud_scale_relative_to_previous": 1.0,
            "horizontal_scale_constraint": "four columns cannot be 1.2x within fixed 11532 px motherboard",
            "ellipsis_y": ellipsis_y,
            "lower_ellipsis_y": lower_ellipsis_y,
            "ellipsis_gap_to_xlabel": ellipsis_gap_to_xlabel,
            "ellipsis_fontsize": 24,
            "ellipsis_fontweight": "bold",
            "title_panel_label_clearance": title_label_clearance,
            "panel_label_offset_above_axes_px": 25.5,
            "panel_label_offset_is_equal_for_bcde": True,
        },
        "integrity": {
            "test_field_used_for_model_fit": False,
            "future_field_used_for_forecast_update": False,
            "cylinder_position_from_measured_displacement": True,
            "minimum_mse_snapshots_are_explicitly_labelled_in_metadata": True,
        },
        "outputs": {suffix: str(base.with_suffix(suffix)) for suffix in (".png", ".tiff", ".pdf", ".svg")},
    }
    (args.output / "figure5_panels_bcde_x40y20_0803_best_mse_gradient_glow_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
