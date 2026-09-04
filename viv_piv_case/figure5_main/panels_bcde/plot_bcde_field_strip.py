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
    # The colorbar is 1.31x the former height and sits above the bottom margin,
    # leaving the x/D labels unobstructed.
    cax = fig.add_axes([left, max(0.022, bottom - 0.085), right - left, 0.021])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    _configure()
    args.output.mkdir(parents=True, exist_ok=True)

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

    # Distinct but restrained backgrounds for available and withheld observations.
    upper_background = Rectangle(
        (0.0, 0.335),
        1.0,
        0.665,
        transform=fig.transFigure,
        facecolor="#F6FAFC",
        edgecolor="none",
        zorder=-20,
    )
    lower_background = Rectangle(
        (0.0, 0.0),
        1.0,
        0.335,
        transform=fig.transFigure,
        facecolor="#FDF8F6",
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

    label_offset = 100 / FIG_H_PX
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
    # Crop only the left/right motherboard whitespace.
    # Coordinates are fractions of the original figure width.
    crop_left = 0.078
    crop_right = 0.910

    horizontal_crop_bbox = mpl.transforms.Bbox.from_extents(
        FIG_W_IN * crop_left,
        0.0,
        FIG_W_IN * crop_right,
        FIG_H_IN,
    )
    # END HORIZONTAL CANVAS CROP

    base = args.output / "figure5_panels_bcde_x40y20_0803_best_mse"
    fig.savefig(base.with_suffix(".png"), dpi=DPI, facecolor="white", bbox_inches=horizontal_crop_bbox, pad_inches=0.0)
    fig.savefig(base.with_suffix(".tiff"), dpi=DPI, facecolor="white", bbox_inches=horizontal_crop_bbox, pad_inches=0.0)
    fig.savefig(base.with_suffix(".pdf"), facecolor="white", bbox_inches=horizontal_crop_bbox, pad_inches=0.0)
    fig.savefig(base.with_suffix(".svg"), facecolor="white", bbox_inches=horizontal_crop_bbox, pad_inches=0.0)
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
        "backgrounds": {"available": "#F6FAFC", "withheld": "#FDF8F6", "covers_stage_titles": True, "split_y": background_split_y, "title_background_margin": title_background_margin},
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
    (args.output / "figure5_panels_bcde_x40y20_0803_best_mse_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
