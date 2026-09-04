#!/usr/bin/env python3
"""Build the five-panel Baoding single/dual-source publication figure."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
from mpl_toolkits.mplot3d import proj3d
from PIL import Image, ImageOps


FIG_DPI = 650
# Keep the typography fixed while reducing the complete canvas to 90% of v9.
FIG_W, FIG_H = 18.36, 12.24
FONT_PANEL, FONT_TITLE = 26, 17
FONT_LEGEND, FONT_AXIS, FONT_TICK = 14, 15, 13
FONT_ARRAY_LABEL = 11
HEADER_LABEL_Y_OFFSET = 0.018
HEADER_TITLE_Y_OFFSET = 0.012
D_LEGEND_BORDERPAD = 0.34
D_LEGEND_ANCHOR = (0.01, 0.985)
D_NODE_ROW_SPACING = 1.15
TRACK_LW_GPS, TRACK_LW_EST, TRACK_LW_C = 3.08, 2.78, 2.70
ARRAY_LW = 3.00
TEXT_COLOR = "#111111"
GPS_COLOR = "#202020"
SINGLE_COLOR = "#D97932"
DUAL1_COLOR = "#3F6B8F"
DUAL2_COLOR = "#5C8D62"
# High-separation violet/gold pair for the calibration diagnostic. Both remain
# legible on white and are distinct from the tracking and array palettes.
RELIABILITY_COLORS = {1: "#8E5AA8", 2: "#D49A28"}
ARM_COLORS = {"x-arm": "#4C4C4C", "y-arm": "#C65D3A", "z-arm": "#2F8C87"}
ARRAY_MARKER_SIZE_V12 = 58
ARRAY_MARKER_SIZE = 116
TRACK_LEGEND_CLEARANCE_M = 100.0
# Frozen from the admitted v38 layout so replacing the dual-source window does
# not silently rescale the unchanged single-source evidence panel.
AB_SHARED_LIMITS = ((-762.802347903294, 837.197652096706), (-900.0, 700.0))
ARRAY_LABEL_OFFSETS_PT = {
    1: (-12, 12), 2: (-16, 15), 3: (-22, 18), 4: (20, -22), 5: (24, -14), 6: (16, -10),
    7: (-14, -20), 8: (-20, -14), 9: (-26, -6), 10: (28, 24), 11: (26, 13), 12: (16, 9),
    13: (14, -5), 14: (18, 0), 15: (-28, -12), 16: (30, 15), 17: (15, 2), 18: (15, 2), 19: (15, 2),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
        "font.weight": "normal",
        "axes.titleweight": "normal",
        "axes.labelweight": "normal",
        "axes.labelsize": FONT_AXIS,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_LEGEND,
        "axes.linewidth": 0.9,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def add_header(fig: plt.Figure, ax: plt.Axes, letter: str, title: str) -> None:
    position = ax.get_position()
    x = position.x0 - 0.030
    fig.text(x, position.y1 + HEADER_LABEL_Y_OFFSET, letter, fontsize=FONT_PANEL, fontweight="bold", ha="left", va="baseline", color=TEXT_COLOR)
    fig.text(x + 0.040, position.y1 + HEADER_TITLE_Y_OFFSET, title, fontsize=FONT_TITLE, fontweight="normal", ha="left", va="baseline", color=TEXT_COLOR)


def integer_axes(ax: plt.Axes) -> None:
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        axis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    ax.tick_params(axis="both", labelsize=FONT_TICK, width=0.9, length=3.2)


def ribbon_polygon(xy: np.ndarray, radius: np.ndarray) -> np.ndarray:
    tangent = np.empty_like(xy)
    tangent[0], tangent[-1] = xy[1] - xy[0], xy[-1] - xy[-2]
    tangent[1:-1] = xy[2:] - xy[:-2]
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    left = xy + normal * radius[:, None]
    right = xy - normal * radius[:, None]
    return np.vstack((left, right[::-1]))


def square_limits(point_groups: list[np.ndarray], pad_fraction: float = 0.08) -> tuple[tuple[float, float], tuple[float, float]]:
    points = np.vstack(point_groups)
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    side = max(x_max - x_min, y_max - y_min) * (1.0 + 2.0 * pad_fraction)
    side = max(200.0, math.ceil(side / 100.0) * 100.0)
    x_mid, y_mid = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    return (x_mid - side / 2.0, x_mid + side / 2.0), (y_mid - side / 2.0, y_mid + side / 2.0)


def style_track_axis(
    ax: plt.Axes,
    groups: list[np.ndarray],
    limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> None:
    xlim, ylim = limits if limits is not None else square_limits(groups)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    ax.set_xlabel("East offset (m)")
    ax.set_ylabel("North offset (m)")
    ax.grid(color="#DFDFDF", linewidth=0.55, alpha=0.8, zorder=0)
    integer_axes(ax)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_color(TEXT_COLOR)


def array_geometry(spacing: float = 0.50) -> dict[str, list[tuple[int, float, float, float]]]:
    groups = {"x-arm": tuple(range(1, 7)), "y-arm": tuple(range(7, 13)), "z-arm": tuple(range(13, 20))}
    horizontal = [-3 * spacing, -2 * spacing, -spacing, spacing, 2 * spacing, 3 * spacing]
    vertical = [-2.13 * spacing, -1.53 * spacing, -0.93 * spacing, 0.0, spacing, 2 * spacing, 3 * spacing]
    return {
        "x-arm": [(channel, coordinate, 0.0, 0.0) for channel, coordinate in zip(groups["x-arm"], horizontal, strict=True)],
        "y-arm": [(channel, 0.0, coordinate, 0.0) for channel, coordinate in zip(groups["y-arm"], horizontal, strict=True)],
        "z-arm": [(channel, 0.0, 0.0, coordinate) for channel, coordinate in zip(groups["z-arm"], vertical, strict=True)],
    }


def draw_array(
    ax: plt.Axes,
) -> tuple[dict[int, mpl.text.Annotation], dict[int, tuple[float, float]], list[np.ndarray], mpl.legend.Legend]:
    microphone_rows: list[tuple[int, float, float, float]] = []
    geometry = array_geometry()
    for group, rows in geometry.items():
        xyz = np.asarray([[row[1], row[2], row[3]] for row in rows])
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=ARM_COLORS[group], lw=ARRAY_LW)
        ax.scatter(
            xyz[:, 0], xyz[:, 1], xyz[:, 2], color=ARM_COLORS[group], s=ARRAY_MARKER_SIZE,
            depthshade=False, edgecolor="white", linewidth=0.5,
        )
        microphone_rows.extend(rows)
    ax.scatter([0], [0], [0], marker="+", color=TEXT_COLOR, s=80, linewidth=1.2)
    # Tighten the 3-D view to the actual microphone envelope while retaining
    # a fixed margin for markers, projected labels and the orthogonal arms.
    # The former broad limits left conspicuous empty space around the array.
    ax.set_xlim(-1.72, 1.72)
    ax.set_ylim(-1.72, 1.72)
    ax.set_zlim(-1.25, 1.68)
    ax.set_box_aspect((1.0, 1.0, 0.9))
    ax.set_proj_type("ortho")
    ax.view_init(elev=27, azim=-43)
    annotations: dict[int, mpl.text.Annotation] = {}
    projected_anchors: dict[int, tuple[float, float]] = {}
    for channel, x, y, z in microphone_rows:
        projected_x, projected_y, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
        projected_anchors[channel] = (float(projected_x), float(projected_y))
        offset = ARRAY_LABEL_OFFSETS_PT[channel]
        annotations[channel] = ax.annotate(
            f"M{channel}", xy=(projected_x, projected_y), xytext=offset,
            textcoords="offset points", fontsize=FONT_ARRAY_LABEL, color=TEXT_COLOR,
            ha="left" if offset[0] >= 0 else "right",
            va="bottom" if offset[1] >= 0 else "top", zorder=12,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.94, "pad": 0.45},
            arrowprops={"arrowstyle": "-", "color": "#666666", "linewidth": 0.65, "alpha": 0.82,
                        "shrinkA": 2.0, "shrinkB": 5.0},
        )
    projected_groups: list[np.ndarray] = []
    for rows in geometry.values():
        projected_groups.append(np.asarray([
            proj3d.proj_transform(x, y, z, ax.get_proj())[:2]
            for _, x, y, z in rows
        ], dtype=float))
    ax.set_xlabel("x (m)", labelpad=2)
    ax.set_ylabel("y (m)", labelpad=2)
    ax.set_zlabel("z (m)", labelpad=1)
    ax.tick_params(axis="both", labelsize=FONT_TICK, pad=0, width=0.8)
    ax.zaxis.set_tick_params(labelsize=FONT_TICK, pad=0, width=0.8)
    pane_colors = {
        ax.xaxis: (0.965, 0.955, 0.950, 1.0),
        ax.yaxis: (0.950, 0.970, 0.968, 1.0),
        ax.zaxis: (0.965, 0.965, 0.965, 1.0),
    }
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_formatter(StrMethodFormatter("{x:g}"))
        axis.pane.set_facecolor(pane_colors[axis])
        axis.pane.set_alpha(1.0)
        axis.pane.set_edgecolor("#B5B5B5")
        axis._axinfo["grid"].update(color=(0.84, 0.84, 0.84, 0.70), linewidth=0.55)
    handles = [
        Line2D([0], [0], color=ARM_COLORS["x-arm"], lw=ARRAY_LW, marker="o", ms=5, label="x arm (6)"),
        Line2D([0], [0], color=ARM_COLORS["y-arm"], lw=ARRAY_LW, marker="o", ms=5, label="y arm (6)"),
        Line2D([0], [0], color=ARM_COLORS["z-arm"], lw=ARRAY_LW, marker="o", ms=5, label="z arm (7)"),
    ]
    legend = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.96), fontsize=FONT_LEGEND, handlelength=1.35, labelspacing=0.35, borderaxespad=0.0)
    return annotations, projected_anchors, projected_groups, legend


def expanded_bbox(bbox: mpl.transforms.Bbox, padding_px: float) -> mpl.transforms.Bbox:
    return mpl.transforms.Bbox.from_extents(
        bbox.x0 - padding_px, bbox.y0 - padding_px,
        bbox.x1 + padding_px, bbox.y1 + padding_px,
    )


def bbox_overlap_area(first: mpl.transforms.Bbox, second: mpl.transforms.Bbox) -> float:
    width = max(0.0, min(first.x1, second.x1) - max(first.x0, second.x0))
    height = max(0.0, min(first.y1, second.y1) - max(first.y0, second.y0))
    return width * height


def annotation_text_bbox(
    annotation: mpl.text.Annotation,
    renderer: mpl.backend_bases.RendererBase,
) -> mpl.transforms.Bbox:
    # Annotation.get_window_extent() also encloses its leader line, which would
    # make every label appear to collide with the microphone it identifies.
    return mpl.text.Text.get_window_extent(annotation, renderer=renderer)


def resolve_array_label_layout(
    fig: plt.Figure,
    ax: plt.Axes,
    annotations: dict[int, mpl.text.Annotation],
    projected_anchors: dict[int, tuple[float, float]],
    projected_groups: list[np.ndarray],
    legend: mpl.legend.Legend,
) -> dict[str, object]:
    """Place array labels in screen space and verify that their rendered boxes are disjoint."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer=renderer)
    legend_bbox = expanded_bbox(legend.get_window_extent(renderer=renderer), 2.0)
    marker_radius_px = (math.sqrt(ARRAY_MARKER_SIZE) / 2.0 + 2.2) * fig.dpi / 72.0
    marker_boxes = {
        channel: mpl.transforms.Bbox.from_bounds(
            *(
                ax.transData.transform(projected_anchors[channel])
                - np.asarray([marker_radius_px, marker_radius_px])
            ),
            2.0 * marker_radius_px,
            2.0 * marker_radius_px,
        )
        for channel in annotations
    }
    line_samples: list[np.ndarray] = []
    for group in projected_groups:
        display = ax.transData.transform(group)
        for start, end in zip(display[:-1], display[1:], strict=True):
            line_samples.extend(start + fraction * (end - start) for fraction in np.linspace(0.0, 1.0, 31))

    candidate_offsets: list[tuple[int, int]] = []
    for radius in (12, 18, 24, 30, 36, 42):
        candidate_offsets.extend([
            (radius, radius), (-radius, radius), (radius, -radius), (-radius, -radius),
            (radius, 0), (-radius, 0), (0, radius), (0, -radius),
            (radius, radius // 2), (-radius, radius // 2),
            (radius, -radius // 2), (-radius, -radius // 2),
        ])

    centre = np.asarray([axes_bbox.x0 + axes_bbox.width / 2.0, axes_bbox.y0 + axes_bbox.height / 2.0])
    order = sorted(
        annotations,
        key=lambda channel: float(np.linalg.norm(ax.transData.transform(projected_anchors[channel]) - centre)),
    )
    accepted: dict[int, mpl.transforms.Bbox] = {}
    selected_offsets: dict[int, tuple[int, int]] = {}
    for channel in order:
        annotation = annotations[channel]
        preferred = ARRAY_LABEL_OFFSETS_PT[channel]
        candidates = [preferred] + [offset for offset in candidate_offsets if offset != preferred]
        best: tuple[float, tuple[int, int], mpl.transforms.Bbox] | None = None
        for rank, offset in enumerate(candidates):
            annotation.set_position(offset)
            annotation.set_ha("left" if offset[0] >= 0 else "right")
            annotation.set_va("bottom" if offset[1] >= 0 else "top")
            bbox = expanded_bbox(annotation_text_bbox(annotation, renderer), 1.5)
            outside = (
                max(0.0, axes_bbox.x0 + 2.0 - bbox.x0)
                + max(0.0, bbox.x1 - axes_bbox.x1 + 2.0)
                + max(0.0, axes_bbox.y0 + 2.0 - bbox.y0)
                + max(0.0, bbox.y1 - axes_bbox.y1 + 2.0)
            )
            label_overlap = sum(bbox_overlap_area(bbox, other) for other in accepted.values())
            marker_overlap = sum(bbox_overlap_area(bbox, marker) for marker in marker_boxes.values())
            legend_overlap = bbox_overlap_area(bbox, legend_bbox)
            line_overlap = sum(1 for point in line_samples if bbox.contains(float(point[0]), float(point[1])))
            distance = math.hypot(*offset)
            score = (
                1.0e8 * outside
                + 1.0e6 * label_overlap
                + 1.0e5 * marker_overlap
                + 1.0e6 * legend_overlap
                + 1.0e4 * line_overlap
                + distance
                + rank * 1.0e-3
            )
            if best is None or score < best[0]:
                best = (score, offset, bbox)
        assert best is not None
        _, offset, bbox = best
        annotation.set_position(offset)
        annotation.set_ha("left" if offset[0] >= 0 else "right")
        annotation.set_va("bottom" if offset[1] >= 0 else "top")
        selected_offsets[channel] = offset
        accepted[channel] = bbox

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    final_boxes = {
        channel: expanded_bbox(annotation_text_bbox(annotation, renderer), 1.5)
        for channel, annotation in annotations.items()
    }
    label_pairs = [
        [f"M{first}", f"M{second}"]
        for index, first in enumerate(sorted(final_boxes))
        for second in sorted(final_boxes)[index + 1:]
        if bbox_overlap_area(final_boxes[first], final_boxes[second]) > 0.0
    ]
    marker_pairs = [
        [f"M{label}", f"M{marker}"]
        for label, label_bbox in final_boxes.items()
        for marker, marker_bbox in marker_boxes.items()
        if bbox_overlap_area(label_bbox, marker_bbox) > 0.0
    ]
    labels_outside_axes = [
        f"M{channel}" for channel, bbox in final_boxes.items()
        if bbox.x0 < axes_bbox.x0 or bbox.x1 > axes_bbox.x1
        or bbox.y0 < axes_bbox.y0 or bbox.y1 > axes_bbox.y1
    ]
    if label_pairs or marker_pairs or labels_outside_axes:
        raise RuntimeError(
            "array label layout failed: "
            f"label_pairs={label_pairs}, marker_pairs={marker_pairs}, outside={labels_outside_axes}"
        )
    return {
        "method": "screen-space candidate placement with short leaders and rendered-bbox collision audit",
        "selected_offsets_pt": {f"M{channel}": list(offset) for channel, offset in sorted(selected_offsets.items())},
        "label_label_overlaps": label_pairs,
        "label_marker_overlaps": marker_pairs,
        "labels_outside_axes": labels_outside_axes,
        "all_labels_nonoverlapping": True,
    }


def draw_square_photo(ax: plt.Axes, image: np.ndarray) -> None:
    height, width = image.shape[:2]
    if width >= height:
        pad = (width - height) / 2.0
        ax.imshow(image, extent=(0, width, height, 0), interpolation="antialiased")
        ax.set_xlim(0, width)
        ax.set_ylim(height + pad, -pad)
    else:
        pad = (height - width) / 2.0
        ax.imshow(image, extent=(0, width, height, 0), interpolation="antialiased")
        ax.set_xlim(-pad, width + pad)
        ax.set_ylim(height, 0)
    ax.set_box_aspect(1)
    ax.set_axis_off()


def draw_array_coordinate_table(ax: plt.Axes) -> None:
    """Show the exact coordinate model used by the MUSIC frontend."""
    ax.set_axis_off()
    rows = [
        ["x", "M1-M6", "-1.5, -1, -0.5, 0.5, 1, 1.5"],
        ["y", "M7-M12", "-1.5, -1, -0.5, 0.5, 1, 1.5"],
        ["z", "M13-M19", "-1.065, -0.765, -0.465, 0, 0.5, 1, 1.5"],
    ]
    table = ax.table(
        cellText=rows,
        colLabels=["Arm", "Channels", "Coordinates (m)"],
        colWidths=[0.10, 0.20, 0.70],
        cellLoc="left",
        colLoc="left",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(FONT_ARRAY_LABEL)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#555555")
        cell.set_linewidth(0.75)
        cell.PAD = 0.16
        cell.set_facecolor("#F4F4F4" if row == 0 else "white")
        if row == 0:
            cell.get_text().set_fontweight("bold")
        cell.get_text().set_color(TEXT_COLOR)


def draw_node_target_reliability(
    ax: plt.Axes,
    rows: list[dict[str, str]],
) -> dict[str, object]:
    """Render frozen-calibration angular reliability without claiming physical SNR."""
    # The authoritative confidence frontend writes ``median_angular_error_deg``
    # and ``p90_angular_error_deg``.  The aliases keep the plotting contract
    # explicit while allowing older exported reliability tables to be reused.
    def value(row: dict[str, str], *names: str) -> float:
        for name in names:
            if name in row and row[name] not in (None, ""):
                return float(row[name])
        raise KeyError(f"missing reliability field; tried {names}")

    values: dict[tuple[int, int], tuple[float, float]] = {}
    for row in rows:
        target = int(row["target"])
        node = int(row.get("node", row.get("node_id", "-1")))
        median = value(row, "calibration_angular_median_deg", "median_angular_error_deg")
        p90 = value(row, "calibration_angular_p90_deg", "p90_angular_error_deg")
        if target not in (1, 2) or median < 0.0 or p90 < median:
            raise RuntimeError(f"invalid calibration angular-reliability row: {row}")
        key = (target, node)
        if key in values:
            raise RuntimeError(f"duplicate calibration angular-reliability row: {key}")
        values[key] = (median, p90)

    nodes = sorted({node for _, node in values})
    required = {(target, node) for target in (1, 2) for node in nodes}
    if set(values) != required:
        missing = sorted(required - set(values))
        extra = sorted(set(values) - required)
        raise RuntimeError(f"incomplete calibration angular-reliability table: missing={missing}, extra={extra}")

    target_colors = RELIABILITY_COLORS
    target_offsets = {1: 0.16, 2: -0.16}
    y_positions = {node: (len(nodes) - 1 - index) * D_NODE_ROW_SPACING for index, node in enumerate(nodes)}
    for target in (1, 2):
        color = target_colors[target]
        for node in nodes:
            median, p90 = values[(target, node)]
            y = y_positions[node] + target_offsets[target]
            ax.hlines(y, median, p90, color=color, linewidth=TRACK_LW_C, zorder=3)
            ax.scatter(median, y, s=48, facecolor="white", edgecolor=color, linewidth=1.55, zorder=4)
            ax.scatter(p90, y, s=20, facecolor=color, edgecolor="white", linewidth=0.45, zorder=5)

    max_p90 = max(p90 for _, p90 in values.values())
    x_limit = max(20.0, math.ceil((max_p90 + 8.0) / 20.0) * 20.0)
    ax.set_xlim(0.0, x_limit)
    ax.set_ylim(-0.75, len(nodes) + 1.45)
    ax.set_yticks([y_positions[node] for node in nodes])
    ax.set_yticklabels([f"N{node}" for node in nodes])
    ax.set_xlabel("Calibration angular residual (deg)")
    ax.set_ylabel("Array node")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    ax.tick_params(axis="both", labelsize=FONT_TICK, width=0.9, length=3.2)
    ax.grid(axis="x", color="#DFDFDF", linewidth=0.55, alpha=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_color(TEXT_COLOR)
    ax.set_box_aspect(1)
    legend = ax.legend(
        handles=[
            Line2D([0], [0], color=RELIABILITY_COLORS[1], lw=TRACK_LW_C, marker="o", markerfacecolor="white", markeredgecolor=RELIABILITY_COLORS[1], label="T1 median-P90"),
            Line2D([0], [0], color=RELIABILITY_COLORS[2], lw=TRACK_LW_C, marker="o", markerfacecolor="white", markeredgecolor=RELIABILITY_COLORS[2], label="T2 median-P90"),
        ],
        loc="upper left",
        bbox_to_anchor=D_LEGEND_ANCHOR,
        fontsize=FONT_LEGEND,
        handlelength=1.35,
        labelspacing=0.28,
        borderaxespad=0.0,
        ncol=2,
        columnspacing=0.85,
        handletextpad=0.42,
        frameon=True,
        facecolor="white",
        edgecolor="#222222",
        framealpha=0.94,
        borderpad=D_LEGEND_BORDERPAD,
    )
    return {
        "nodes": nodes,
        "target_node_pairs": len(values),
        "summary": "open marker: median; filled endpoint: P90; horizontal segment: median to P90",
        "x_limit_deg": x_limit,
        "legend_columns": 2,
        "legend_layout": "single row",
        "node_row_spacing": D_NODE_ROW_SPACING,
        "legend": legend,
    }


def annotate_endpoint(
    ax: plt.Axes,
    point: np.ndarray,
    label: str,
    color: str,
    offset: tuple[float, float],
) -> None:
    ax.annotate(
        label,
        xy=(float(point[0]), float(point[1])),
        xytext=offset,
        textcoords="offset points",
        fontsize=FONT_LEGEND,
        color=color,
        ha="left" if offset[0] >= 0 else "right",
        va="bottom" if offset[1] >= 0 else "top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.4},
        zorder=12,
    )


def save_outputs(fig: plt.Figure, stem: Path) -> dict[str, str]:
    outputs = {}
    for extension, kwargs in (
        ("png", {"dpi": FIG_DPI}),
        ("pdf", {}),
        ("svg", {}),
        ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
    ):
        path = stem.with_suffix(f".{extension}")
        fig.savefig(path, facecolor="white", **kwargs)
        outputs[extension] = str(path)
    return outputs


def legend_record(fig: plt.Figure, axis: plt.Axes, legend: mpl.legend.Legend, *, loc: str, anchor: tuple[float, float]) -> dict[str, object]:
    """Capture the rendered legend geometry and frame style for independent QA."""
    renderer = fig.canvas.get_renderer()
    bbox = legend.get_window_extent(renderer=renderer)
    fig_bbox = bbox.transformed(fig.transFigure.inverted())
    axes_bbox = bbox.transformed(axis.transAxes.inverted())
    frame = legend.get_frame()
    face_rgba = [float(value) for value in frame.get_facecolor()]
    edge_rgba = [float(value) for value in frame.get_edgecolor()]
    return {
        "loc": loc,
        "bbox_to_anchor": [float(anchor[0]), float(anchor[1])],
        "bbox_fig_fraction": [float(fig_bbox.x0), float(fig_bbox.y0), float(fig_bbox.x1), float(fig_bbox.y1)],
        "bbox_axes_fraction": [float(axes_bbox.x0), float(axes_bbox.y0), float(axes_bbox.x1), float(axes_bbox.y1)],
        "frameon": bool(legend.get_frame().get_visible()),
        "facecolor_rgba": face_rgba,
        "edgecolor_rgba": edge_rgba,
        "framealpha": float(frame.get_alpha() if frame.get_alpha() is not None else face_rgba[3]),
        "labels": [text.get_text() for text in legend.get_texts()],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-source-csv", type=Path, required=True)
    parser.add_argument("--single-source-registry", type=Path, required=True)
    parser.add_argument("--dual-source-csv", type=Path, required=True)
    parser.add_argument("--dual-selection-manifest", type=Path, required=True)
    parser.add_argument("--dual-formal-root", type=Path, required=True)
    parser.add_argument("--nodes-csv", type=Path, required=True)
    parser.add_argument("--array-photo", type=Path, required=True)
    parser.add_argument("--reliability-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()

    single_rows = read_csv(args.single_source_csv)
    dual_rows = read_csv(args.dual_source_csv)
    node_rows = read_csv(args.nodes_csv)
    reliability_rows = read_csv(args.reliability_csv)
    if len(single_rows) != 67 or len(dual_rows) < 2:
        raise RuntimeError(f"expected 67 single-source rows and at least two dual-source rows, found {len(single_rows)} and {len(dual_rows)}")

    single_elapsed = np.asarray([float(row["elapsed_s"]) for row in single_rows])
    single_truth = np.asarray([[float(row[f"gps_{axis}_m"]) for axis in ("east", "north", "up")] for row in single_rows])
    single_apce = np.asarray([[float(row[f"apce_{axis}_median_5seeds_m"]) for axis in ("east", "north", "up")] for row in single_rows])
    single_error = np.asarray([float(row["apce_position_error_of_median_m"]) for row in single_rows])
    single_width = np.asarray([float(row["apce_median_mean_marginal_90pct_interval_width_m"]) for row in single_rows])
    nodes = np.asarray([[float(row["local_E_m"]), float(row["local_N_m"]), float(row["local_U_m"])] for row in node_rows])

    dual_elapsed = np.asarray([float(row["elapsed_s"]) for row in dual_rows])
    if not np.allclose(np.diff(dual_elapsed), 1.0):
        raise RuntimeError("dual-source rows are not contiguous one-second updates")
    dual_duration_s = len(dual_rows)
    dual_truth = {}
    dual_apce = {}
    dual_error = {}
    dual_width = {}
    for target in (1, 2):
        dual_truth[target] = np.asarray([[float(row[f"target{target}_gps_{axis}_m"]) for axis in ("east", "north", "up")] for row in dual_rows])
        dual_apce[target] = np.asarray([[float(row[f"target{target}_apce_{axis}_median_5seeds_m"]) for axis in ("east", "north", "up")] for row in dual_rows])
        dual_error[target] = np.asarray([float(row[f"target{target}_apce_error_m"]) for row in dual_rows])
        dual_width[target] = np.asarray([float(row[f"target{target}_apce_median_marginal_width_m"]) for row in dual_rows])

    with Image.open(args.array_photo) as source:
        array_image = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
        array_resolution = list(array_image.shape[1::-1])

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
    grid = fig.add_gridspec(2, 3, left=0.052, right=0.982, bottom=0.052, top=0.947, wspace=0.18, hspace=0.20)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])
    ax_d = fig.add_subplot(grid[1, 0])
    ax_e = fig.add_subplot(grid[1, 1:])
    ax_e.set_axis_off()

    # Panel e is a model-hardware composite.  The array model occupies a
    # physical square at left; the uncropped photograph is bottom-aligned at
    # right, with the remaining space used for the exact frontend coordinates.
    e_box = ax_e.get_position()
    e_gap = 0.018
    model_width = e_box.height * FIG_H / FIG_W
    # Enlarge only the 3-D model canvas and nudge it down/left.  Keep the
    # base model width for the right-hand layout calculation so the table and
    # photograph retain their v25f positions exactly.
    array_width = model_width * 1.10
    array_height = e_box.height * 1.08
    ax_e_array = fig.add_axes(
        [e_box.x0 - 0.010, e_box.y0 - 0.052, array_width, array_height], projection="3d"
    )
    photo_aspect = array_image.shape[1] / array_image.shape[0]
    photo_height = e_box.height * 0.72
    right_width = photo_height * FIG_H * photo_aspect / FIG_W
    content_right = ax_c.get_position().x0 + e_box.height * FIG_H / FIG_W
    right_x0 = content_right - right_width
    if right_x0 < e_box.x0 + model_width + e_gap:
        right_x0 = e_box.x0 + model_width + e_gap
        right_width = content_right - right_x0
        photo_height = right_width * FIG_W / (photo_aspect * FIG_H)
    info_gap = 0.012
    info_height = e_box.height - photo_height - info_gap
    ax_e_photo = fig.add_axes([right_x0, e_box.y0, right_width, photo_height])
    ax_e_info = fig.add_axes([
        right_x0, e_box.y0 + photo_height + info_gap, right_width, info_height,
    ])

    single_groups = [single_truth[:, :2], single_apce[:, :2], nodes[:, :2]]
    dual_groups = [nodes[:, :2]]
    for target in (1, 2):
        dual_groups.extend([dual_truth[target][:, :2], dual_apce[target][:, :2]])
    ab_limits = AB_SHARED_LIMITS

    single_ribbon = ribbon_polygon(single_apce[:, :2], single_width / 2.0)
    ax_a.fill(single_ribbon[:, 0], single_ribbon[:, 1], color=SINGLE_COLOR, alpha=0.17, linewidth=0, zorder=1)
    ax_a.plot(single_truth[:, 0], single_truth[:, 1], color=GPS_COLOR, lw=TRACK_LW_GPS, ls="--", label="GPS", zorder=5)
    ax_a.plot(single_apce[:, 0], single_apce[:, 1], color=SINGLE_COLOR, lw=TRACK_LW_EST, label="APCE", zorder=6)
    ax_a.scatter(single_apce[0, 0], single_apce[0, 1], s=48, marker="o", facecolor="white", edgecolor=SINGLE_COLOR, linewidth=1.8, zorder=7)
    ax_a.scatter(single_apce[-1, 0], single_apce[-1, 1], s=48, marker="s", facecolor="white", edgecolor=SINGLE_COLOR, linewidth=1.8, zorder=7)
    ax_a.scatter(nodes[:, 0], nodes[:, 1], s=48, marker="P", color=TEXT_COLOR, edgecolor="white", linewidth=0.55, zorder=8)
    for row, point in zip(node_rows, nodes, strict=True):
        ax_a.annotate(f"N{row['node_id']}", (point[0], point[1]), xytext=(3, 3), textcoords="offset points", fontsize=FONT_LEGEND, color=TEXT_COLOR, ha="left", va="bottom", zorder=9)
    style_track_axis(ax_a, single_groups, limits=ab_limits)
    legend_a = ax_a.legend(
        handles=[
            Line2D([0], [0], color=GPS_COLOR, lw=TRACK_LW_GPS, ls="--", label="GPS"),
            Patch(facecolor=SINGLE_COLOR, edgecolor="none", alpha=0.17, label="Marginal width"),
            Line2D([0], [0], color=SINGLE_COLOR, lw=TRACK_LW_EST, label="APCE"),
            Line2D([0], [0], color=TEXT_COLOR, marker="P", ls="None", ms=6, label="Array node"),
            Line2D([0], [0], color=SINGLE_COLOR, marker="o", ls="None", ms=6, markerfacecolor="white", label="Start"),
            Line2D([0], [0], color=SINGLE_COLOR, marker="s", ls="None", ms=6, markerfacecolor="white", label="End"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.018, 0.025),
        fontsize=FONT_LEGEND,
        handlelength=1.65,
        labelspacing=0.28,
        borderaxespad=0.0,
        ncol=2,
        columnspacing=0.80,
        frameon=True,
        facecolor="white",
        edgecolor="#222222",
        framealpha=0.94,
        borderpad=0.34,
    )

    target_colors = {1: DUAL1_COLOR, 2: DUAL2_COLOR}
    for target in (1, 2):
        color = target_colors[target]
        ribbon = ribbon_polygon(dual_apce[target][:, :2], dual_width[target] / 2.0)
        ax_b.fill(ribbon[:, 0], ribbon[:, 1], color=color, alpha=0.13, linewidth=0, zorder=1)
        ax_b.plot(dual_truth[target][:, 0], dual_truth[target][:, 1], color=color, lw=TRACK_LW_GPS, ls="--", zorder=4)
        ax_b.plot(dual_apce[target][:, 0], dual_apce[target][:, 1], color=color, lw=TRACK_LW_EST, zorder=5)
        ax_b.scatter(dual_apce[target][0, 0], dual_apce[target][0, 1], s=44, marker="o", facecolor="white", edgecolor=color, linewidth=1.8, zorder=6)
        ax_b.scatter(dual_apce[target][-1, 0], dual_apce[target][-1, 1], s=44, marker="s", facecolor="white", edgecolor=color, linewidth=1.8, zorder=6)
    ax_b.scatter(nodes[:, 0], nodes[:, 1], s=48, marker="P", color=TEXT_COLOR, edgecolor="white", linewidth=0.75, zorder=7)
    for row, point in zip(node_rows, nodes, strict=True):
        ax_b.annotate(f"N{row['node_id']}", (point[0], point[1]), xytext=(3, 3), textcoords="offset points", fontsize=FONT_LEGEND, color=TEXT_COLOR, ha="left", va="bottom", zorder=9)
    style_track_axis(ax_b, dual_groups, limits=ab_limits)
    legend_b = ax_b.legend(
        handles=[
            Line2D([0], [0], color=DUAL1_COLOR, lw=TRACK_LW_GPS, ls="--", label="T1 GPS"),
            Line2D([0], [0], color=DUAL2_COLOR, lw=TRACK_LW_GPS, ls="--", label="T2 GPS"),
            Patch(facecolor=DUAL1_COLOR, edgecolor="none", alpha=0.13, label="Marginal width"),
            Line2D([0], [0], color=DUAL1_COLOR, lw=TRACK_LW_EST, label="T1 APCE"),
            Line2D([0], [0], color=DUAL2_COLOR, lw=TRACK_LW_EST, label="T2 APCE"),
            Line2D([0], [0], color=TEXT_COLOR, marker="P", ls="None", ms=6, label="Array node"),
            Line2D([0], [0], color=TEXT_COLOR, marker="o", ls="None", ms=6, markerfacecolor="white", label="Start"),
            Line2D([0], [0], color=TEXT_COLOR, marker="s", ls="None", ms=6, markerfacecolor="white", label="End"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.018, 0.025),
        fontsize=FONT_LEGEND,
        handlelength=1.55,
        labelspacing=0.32,
        borderaxespad=0.0,
        ncol=2,
        columnspacing=0.75,
        frameon=True,
        facecolor="white",
        edgecolor="#222222",
        framealpha=0.94,
        borderpad=0.42,
    )

    # The single-source window and acoustically selected dual-source segment may have
    # different durations. Two stacked axes with a normalized progress coordinate
    # make the comparison legible without pretending that the acquisitions
    # have the same elapsed-time support.
    c_box = ax_c.get_position()
    # Keep the nested error axes inside the same square cell as the other
    # panels; an independent y-label can otherwise make Matplotlib widen C.
    reference_width = min(ax_a.get_position().width, ax_b.get_position().width)
    c_box = mpl.transforms.Bbox.from_bounds(c_box.x0, c_box.y0, reference_width, c_box.height)
    ax_c.set_position(c_box)
    ax_c.set_axis_off()
    c_gap = 0.035 * c_box.height
    c_half = (c_box.height - c_gap) / 2.0
    ax_c_top = fig.add_axes([c_box.x0, c_box.y0 + c_half + c_gap, c_box.width, c_half])
    ax_c_bottom = fig.add_axes([c_box.x0, c_box.y0, c_box.width, c_half], sharex=ax_c_top)
    progress_single = 100.0 * single_elapsed / max(float(np.max(single_elapsed)), 1.0)
    progress_dual = 100.0 * dual_elapsed / max(float(np.max(dual_elapsed)), 1.0)
    error_series = (
        ("Single (67 s)", SINGLE_COLOR, progress_single, single_error),
        (f"Dual T1 ({dual_duration_s} s)", DUAL1_COLOR, progress_dual, dual_error[1]),
        (f"Dual T2 ({dual_duration_s} s)", DUAL2_COLOR, progress_dual, dual_error[2]),
    )
    width_series = (
        ("Single (67 s)", SINGLE_COLOR, progress_single, single_width),
        (f"Dual T1 ({dual_duration_s} s)", DUAL1_COLOR, progress_dual, dual_width[1]),
        (f"Dual T2 ({dual_duration_s} s)", DUAL2_COLOR, progress_dual, dual_width[2]),
    )
    for _, color, progress, error in error_series:
        ax_c_top.plot(progress, error, color=color, lw=TRACK_LW_C, zorder=4)
    for _, color, progress, width in width_series:
        ax_c_bottom.plot(progress, width, color=color, lw=TRACK_LW_C, zorder=4)
    for axis in (ax_c_top, ax_c_bottom):
        axis.set_xlim(0.0, 100.0)
        axis.set_xticks((0, 25, 50, 75, 100))
        axis.grid(color="#DFDFDF", linewidth=0.55, alpha=0.8)
        axis.tick_params(axis="both", labelsize=FONT_TICK, width=0.9, length=3.2)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.9)
            spine.set_color(TEXT_COLOR)
    ax_c_top.set_ylim(0.0, math.ceil((max(np.max(item[3]) for item in error_series) * 1.14) / 25.0) * 25.0)
    ax_c_bottom.set_ylim(0.0, math.ceil((max(np.max(item[3]) for item in width_series) * 1.14) / 25.0) * 25.0)
    for axis in (ax_c_top, ax_c_bottom):
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        axis.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    ax_c_top.tick_params(axis="x", labelbottom=False)
    ax_c_top.set_ylabel("Error (m)")
    ax_c_bottom.set_ylabel("Width (m)")
    ax_c_bottom.set_xlabel("Window progress (%)")
    legend_c = ax_c_top.legend(
        handles=[Line2D([0], [0], color=color, lw=TRACK_LW_C, label=label) for label, color, *_ in error_series],
        loc="upper left",
        # A single column keeps the full legend inside the narrow square C
        # cell; the three-column version was wider than the panel itself.
        bbox_to_anchor=(0.03, 0.89),
        fontsize=FONT_LEGEND,
        handlelength=1.45,
        labelspacing=0.30,
        borderaxespad=0.0,
        ncol=1,
        columnspacing=0.75,
        frameon=True,
        facecolor="white",
        edgecolor="#222222",
        framealpha=0.94,
        borderpad=0.38,
    )

    reliability_plot = draw_node_target_reliability(ax_d, reliability_rows)
    array_annotations, array_anchors, array_projected_groups, array_legend = draw_array(ax_e_array)
    # Keep the legend at its previous panel-relative location while lowering
    # only the projected 3-D coordinate box.
    array_legend.set_bbox_to_anchor(
        (e_box.x0, e_box.y1 - 0.015), transform=fig.transFigure,
    )
    ax_e_photo.imshow(array_image, interpolation="antialiased")
    ax_e_photo.set_axis_off()
    draw_array_coordinate_table(ax_e_info)

    fig.canvas.draw()
    array_label_layout = resolve_array_label_layout(
        fig, ax_e_array, array_annotations, array_anchors, array_projected_groups, array_legend,
    )
    add_header(fig, ax_a, "a", "Single-source tracking")
    add_header(fig, ax_b, "b", "Dual-source tracking")
    add_header(fig, ax_c, "c", "Tracking error and uncertainty")
    add_header(fig, ax_d, "d", "Node-target DOA reliability")
    add_header(fig, ax_e, "e", "Array geometry and field deployment")

    fig.canvas.draw()
    legend_records = {
        "a": legend_record(fig, ax_a, legend_a, loc="lower left", anchor=(0.018, 0.025)),
        "b": legend_record(fig, ax_b, legend_b, loc="lower left", anchor=(0.018, 0.025)),
        "c": legend_record(fig, ax_c_top, legend_c, loc="upper left", anchor=(0.03, 0.89)),
        "d": legend_record(fig, ax_d, reliability_plot["legend"], loc="upper left", anchor=D_LEGEND_ANCHOR),
    }
    positions = {letter: axis.get_position() for letter, axis in zip("abcde", (ax_a, ax_b, ax_c, ax_d, ax_e), strict=True)}
    panel_boxes = {
        letter: {
            "x0_fraction": float(position.x0),
            "y0_fraction": float(position.y0),
            "x1_fraction": float(position.x1),
            "y1_fraction": float(position.y1),
            "width_fraction": float(position.width),
            "height_fraction": float(position.height),
            "physical_width_to_height": float(position.width * FIG_W / (position.height * FIG_H)),
        }
        for letter, position in positions.items()
    }

    args.output.mkdir(parents=True, exist_ok=True)
    array_mirror = args.output / "source_photo_2017_array.jpeg"
    reliability_mirror = args.output / "supplementary_data_figure2_node_target_reliability_source.csv"
    shutil.copy2(args.array_photo, array_mirror)
    shutil.copy2(args.reliability_csv, reliability_mirror)
    stem = args.output / "supplementary_data_figure2_baoding_single_dual_field"
    outputs = save_outputs(fig, stem)
    plt.close(fig)

    source_rows: list[dict[str, object]] = []
    for index, row in enumerate(single_rows):
        source_rows.append({
            "panel": "a,c", "scenario": "single", "target": 1, "sample_index": index,
            "time_s": float(row["time_s"]), "elapsed_s": float(row["elapsed_s"]),
            "gps_east_m": float(row["gps_east_m"]), "gps_north_m": float(row["gps_north_m"]), "gps_up_m": float(row["gps_up_m"]),
            "apce_east_m": float(row["apce_east_median_5seeds_m"]), "apce_north_m": float(row["apce_north_median_5seeds_m"]), "apce_up_m": float(row["apce_up_median_5seeds_m"]),
            "position_error_m": float(row["apce_position_error_of_median_m"]), "marginal_width_m": float(row["apce_median_mean_marginal_90pct_interval_width_m"]),
        })
    for index, row in enumerate(dual_rows):
        for target in (1, 2):
            source_rows.append({
                "panel": "b,c", "scenario": "dual", "target": target, "sample_index": index,
                "time_s": float(row["time_s"]), "elapsed_s": float(row["elapsed_s"]),
                "gps_east_m": float(row[f"target{target}_gps_east_m"]), "gps_north_m": float(row[f"target{target}_gps_north_m"]), "gps_up_m": float(row[f"target{target}_gps_up_m"]),
                "apce_east_m": float(row[f"target{target}_apce_east_median_5seeds_m"]), "apce_north_m": float(row[f"target{target}_apce_north_median_5seeds_m"]), "apce_up_m": float(row[f"target{target}_apce_up_median_5seeds_m"]),
                "position_error_m": float(row[f"target{target}_apce_error_m"]), "marginal_width_m": float(row[f"target{target}_apce_median_marginal_width_m"]),
            })
    source_path = args.output / "supplementary_data_figure2_baoding_single_dual_field_source.csv"
    write_csv(source_path, source_rows)
    nodes_path = args.output / "supplementary_data_figure2_baoding_array_nodes_source.csv"
    shutil.copy2(args.nodes_csv, nodes_path)

    panels = [
        {"panel": "a", "content": "single-source GPS and five-seed median APCE horizontal trajectories, scalar marginal-width proxy and nine array nodes", "selection": "previously fixed 67-frame single-source showcase window", "source": str(args.single_source_csv)},
        {"panel": "b", "content": "two-source GPS and five-seed median APCE horizontal trajectories over a continuous acoustically selected stable segment, with calibration-frozen A6 DOA compensation, target-specific acoustic reliability covariance, robust innovation, scalar marginal-width proxies and shared nine array nodes", "selection": f"{dual_duration_s}-frame continuous segment selected from A6 acoustic diagnostics without GPS geometry or APCE error; reliability profile frozen from a separate calibration interval; admitted by offline identity, one-second continuity, maximum-step and PSD-covariance checks", "source": str(args.dual_source_csv)},
        {"panel": "c", "content": "two stacked normalized-progress axes for APCE position error and retained scalar marginal width", "selection": f"separate single-source 67-frame and acoustically selected dual-source {dual_duration_s}-frame stable windows; each curve is rescaled to 0--100% of its own continuous window", "source": f"{args.single_source_csv}; {args.dual_source_csv}"},
        {"panel": "d", "content": "A6 calibration-only node-target angular reliability", "selection": "frozen calibration table underlying confidence weighting; open marker is median, filled endpoint is P90; not an independent evaluation and not a calibrated dB SNR estimate", "source": str(args.reliability_csv)},
        {"panel": "e", "content": "shared 19-microphone 6+6+7 coordinate model, exact coordinate table and uncropped 2017 field-array deployment photograph", "selection": "model and hardware are combined because the same array manifold is used for single- and dual-source processing", "source": f"current MUSIC frontend coordinate model; {args.array_photo}"},
    ]
    panel_path = args.output / "supplementary_data_figure2_baoding_single_dual_field_panel_registry.csv"
    write_csv(panel_path, panels)

    dual_selection = json.loads(args.dual_selection_manifest.read_text(encoding="utf-8"))
    single_registry = json.loads(args.single_source_registry.read_text(encoding="utf-8"))
    formal_manifests = {
        str(target): json.loads(next(
            path for path in (
                args.dual_formal_root / f"target{target}" / "formal_manifest.json",
                args.dual_formal_root / f"target{target}" / "matrix_manifest.json",
            ) if path.exists()
        ).read_text(encoding="utf-8"))
        for target in (1, 2)
    }
    metrics = {
        "single": {
            "rmse_m": float(np.sqrt(np.mean(np.square(single_error)))),
            "median_error_m": float(np.median(single_error)),
            "p90_error_m": float(np.percentile(single_error, 90.0)),
            "median_marginal_width_m": float(np.median(single_width)),
        },
        "dual_target1": {key: dual_selection["selected"][f"target1_{key}"] for key in ("rmse_m", "median_error_m", "p90_error_m", "median_marginal_width_m", "mean_component_coverage_90", "maximum_step_m")},
        "dual_target2": {key: dual_selection["selected"][f"target2_{key}"] for key in ("rmse_m", "median_error_m", "p90_error_m", "median_marginal_width_m", "mean_component_coverage_90", "maximum_step_m")},
    }
    registry = {
        "figure_contract": {
            "core_conclusion": f"Across a 67 s single-source window and an acoustically selected {dual_duration_s}-frame dual-source stable segment, acoustic-only APCE maintains identifiable trajectories with finite uncertainty; the dual-source result combines calibration-frozen DOA compensation, target-specific acoustic reliability covariance and robust innovation.",
            "archetype": "asymmetric mixed-modality figure with a two-column model-hardware composite",
            "backend": "Python/matplotlib on Super-Server",
            "panel_map": {"a": "single-source tracking", "b": "dual-source tracking", "c": "error and marginal uncertainty", "d": "calibration-only node-target DOA reliability", "e": "array coordinate model and field deployment"},
            "reviewer_risks": ["the single-source panel remains a fixed showcase window while the dual-source panel is selected from A6 acoustic quality diagnostics", "the 60-frame dual-source panel is a stable continuous arc rather than a complete circle", "marginal widths are not joint planar confidence regions", "panel d is a calibration diagnostic, not a dB-SNR measurement or independent evaluation"],
        },
        "typography": {"reference": "main-text Figure 4/5 rules", "panel_label_pt": FONT_PANEL, "panel_title_pt": FONT_TITLE, "legend_pt": FONT_LEGEND, "axis_label_pt": FONT_AXIS, "tick_pt": FONT_TICK, "array_label_pt": FONT_ARRAY_LABEL, "only_panel_letters_bold": True, "integer_tick_labels": True, "canvas_scale_vs_v9": 0.9},
        "header_layout": {"label_y_offset_from_axes_top_fraction": HEADER_LABEL_Y_OFFSET, "title_y_offset_from_axes_top_fraction": HEADER_TITLE_Y_OFFSET},
        "panel_d_legend_layout": {"borderpad_font_fraction": D_LEGEND_BORDERPAD, "anchor_axes_fraction": list(D_LEGEND_ANCHOR), "node_row_spacing": D_NODE_ROW_SPACING, "canvas_and_axes_dimensions_unchanged": True},
        "layout_qa": {"canvas_inches": [FIG_W, FIG_H], "grid": "top row: three equal cells; bottom row: d plus two-column e composite", "panel_boxes": panel_boxes, "e_array_axes": {"x0_fraction": float(ax_e_array.get_position().x0), "y0_fraction": float(ax_e_array.get_position().y0), "x1_fraction": float(ax_e_array.get_position().x1), "y1_fraction": float(ax_e_array.get_position().y1), "scale_vs_v25f": {"width": 1.10, "height": 1.08, "x_shift_fraction": -0.010, "y_shift_fraction": -0.052}}, "square_quantitative_panels": ["a", "b", "c", "d"], "photo_frames": "none", "photo_bottom_aligned": True, "photo_table_same_x_bounds": True, "e_content_right_fraction": float(content_right), "c_right_fraction": float(ax_c.get_position().x1), "e_right_aligned_with_c": bool(abs(content_right - ax_c.get_position().x1) < 1e-9), "photo_relationship": "panel e directly pairs the coordinate model, coordinate table and field hardware", "legend_records": legend_records},
        "legend_specs": {
            "a": legend_records["a"],
            "b": legend_records["b"],
            "c": legend_records["c"],
            "d": legend_records["d"],
        },
        "track_axes": {
            "ab_shared_limits": {"x": [float(ab_limits[0][0]), float(ab_limits[0][1])], "y": [float(ab_limits[1][0]), float(ab_limits[1][1])]},
            "a_limits": {"x": [float(ab_limits[0][0]), float(ab_limits[0][1])], "y": [float(ab_limits[1][0]), float(ab_limits[1][1])]},
            "b_limits": {"x": [float(ab_limits[0][0]), float(ab_limits[0][1])], "y": [float(ab_limits[1][0]), float(ab_limits[1][1])]},
            "legend_clearance_pad_m": TRACK_LEGEND_CLEARANCE_M,
            "endpoint_labels": False,
            "endpoint_markers_in_legend": True,
            "node_labels_in_b": True,
        },
        "stroke_widths": {"trajectory_gps_pt": TRACK_LW_GPS, "trajectory_apce_pt": TRACK_LW_EST, "error_and_width_pt": TRACK_LW_C, "array_pt": ARRAY_LW},
        "panel_palettes": {"c": {"single": SINGLE_COLOR, "dual_t1": DUAL1_COLOR, "dual_t2": DUAL2_COLOR}, "d": {"target1": RELIABILITY_COLORS[1], "target2": RELIABILITY_COLORS[2]}, "e": ARM_COLORS},
        "single_window": single_registry["window"],
        "dual_window": dual_selection["selected"],
        "dual_selection_status": dual_selection["selection_status"],
        "gps_role": {"single": single_registry["gps_role"], "dual": dual_selection["gps_role"]},
        "metrics": metrics,
        "dual_formal_configurations": {target: manifest["configuration"] for target, manifest in formal_manifests.items()},
        "dual_frontend_reliability_profiles": {target: manifest.get("frontend_reliability_profile", "legacy_a6_quality_gated") for target, manifest in formal_manifests.items()},
        "dual_reliability_profile_selection": dual_selection.get("reliability_profile_selection"),
        "uncertainty_semantics": {"panel_a_b_ribbons": "half the retained scalar mean marginal width rendered normal to the APCE horizontal path; visual proxy only", "panel_c": "two stacked axes: framewise position error and stored scalar mean width of componentwise weighted 5th--95th percentile intervals, each plotted against normalized progress through its own window", "not_claimed": "joint East-North covariance region or calibrated planar confidence band"},
        "reliability_semantics": {"panel_d": "A6 calibration-only node-target angular residuals used to derive frozen confidence and covariance; open marker is median, filled endpoint is P90, horizontal segment is median-to-P90", "legend_columns": reliability_plot["legend_columns"], "legend_layout": reliability_plot["legend_layout"], "node_row_spacing": reliability_plot["node_row_spacing"], "not_claimed": "physical dB SNR, independent evaluation performance, or a replacement for the raw acoustic likelihood"},
        "array": {"nodes": 9, "microphones_per_node": 19, "geometry": "three orthogonal arms, 6 + 6 + 7 microphones", "shared_between_single_and_dual": True, "microphone_labels": [f"M{channel}" for channel in range(1, 20)], "marker_size_v12": ARRAY_MARKER_SIZE_V12, "marker_size": ARRAY_MARKER_SIZE, "marker_size_ratio_vs_v12": ARRAY_MARKER_SIZE / ARRAY_MARKER_SIZE_V12, "label_layout": array_label_layout},
        "image_integrity": {
            "panel_e": {"raw_file": str(args.array_photo), "raw_sha256": sha256(args.array_photo), "mirrored_file": str(array_mirror), "raw_resolution_px": array_resolution, "crop": "none", "brightness_contrast_gamma": "none", "pseudo_color": "none", "stitching": "none", "overlay": "none"},
            "panel_d": {"type": "quantitative diagnostic", "source_file": str(args.reliability_csv), "source_sha256": sha256(args.reliability_csv), "not_an_image": True, "privacy_action": "helicopter close-up and node photograph omitted"},
            "relationship": "panel e aligns the uncropped field photograph with the exact frontend coordinate model and table",
            "array_photo_resolution_px": array_resolution,
            "array_photo_aspect_ratio": float(array_resolution[0] / array_resolution[1]),
        },
        "sources": {"single_source_csv": str(args.single_source_csv), "dual_source_csv": str(args.dual_source_csv), "dual_selection_manifest": str(args.dual_selection_manifest), "dual_formal_root": str(args.dual_formal_root), "nodes_csv": str(args.nodes_csv), "array_photo": str(args.array_photo), "reliability_csv": str(args.reliability_csv)},
        "outputs": outputs,
        "source_csv": str(source_path),
        "nodes_source_csv": str(nodes_path),
        "panel_registry": str(panel_path),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    registry_path = args.output / "supplementary_data_figure2_baoding_single_dual_field_registry.json"
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"outputs": outputs, "metrics": metrics, "registry": str(registry_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
