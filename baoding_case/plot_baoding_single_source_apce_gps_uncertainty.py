#!/usr/bin/env python3
"""Publication plot for the selected Baoding single-source APCE window."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, StrMethodFormatter
from PIL import Image, ImageOps

START_S, END_S = 46254.0, 46320.0
FIG_DPI = 650
FIG_W, FIG_H = 13.6, 13.6
FONT_PANEL, FONT_TITLE = 22, 14
FONT_LEGEND, FONT_AXIS, FONT_TICK = 14, 13, 11
GPS_COLOR, APCE_COLOR = "#202020", "#D97932"
TEXT_COLOR = "#111111"
ARM_COLORS = {"x-arm": "#3F6B8F", "y-arm": "#5C8D62", "z-arm": "#B66A3C"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_truth(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = read_csv(path)
    times = np.asarray([float(row["time_s"]) for row in rows])
    xyz = np.asarray([[float(row[key]) for key in ("px", "py", "pz")] for row in rows])
    order = np.argsort(times)
    return times[order], xyz[order]


def load_apce_runs(root: Path) -> dict[str, object]:
    paths = sorted((root / "runs").glob("apce_seed_*.json"))
    if len(paths) != 5:
        raise RuntimeError(f"expected five APCE runs, found {len(paths)}")
    positions, widths, errors, coverages, seeds = [], [], [], [], []
    reference_times = None
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = [row for row in payload["records"] if START_S <= float(row["time_s"]) <= END_S]
        times = np.asarray([float(row["time_s"]) for row in records])
        if reference_times is None:
            reference_times = times
        elif times.shape != reference_times.shape or not np.allclose(times, reference_times):
            raise RuntimeError(f"inconsistent time grid in {path}")
        positions.append(np.asarray([[float(row[key]) for key in ("px", "py", "pz")] for row in records]))
        widths.append(np.asarray([float(row["interval_width_m"]) for row in records]))
        errors.append(np.asarray([float(row["position_error_m"]) for row in records]))
        coverages.append(np.asarray([float(row["coverage_90"]) for row in records]))
        seeds.append(int(payload["seed"]))
    assert reference_times is not None
    return {
        "times": reference_times,
        "positions": np.asarray(positions),
        "widths": np.asarray(widths),
        "errors": np.asarray(errors),
        "coverages": np.asarray(coverages),
        "seeds": seeds,
        "paths": paths,
    }


def load_nodes(path: Path) -> tuple[list[dict[str, str]], np.ndarray]:
    rows = read_csv(path)
    points = np.asarray(
        [[float(row["local_E_m"]), float(row["local_N_m"]), float(row["local_U_m"])] for row in rows],
        dtype=float,
    )
    return rows, points


def array_geometry(spacing: float = 0.50) -> dict[str, list[tuple[int, float, float, float]]]:
    groups = {"x-arm": tuple(range(1, 7)), "y-arm": tuple(range(7, 13)), "z-arm": tuple(range(13, 20))}
    horizontal = [-3 * spacing, -2 * spacing, -spacing, spacing, 2 * spacing, 3 * spacing]
    vertical = [-2.13 * spacing, -1.53 * spacing, -0.93 * spacing, 0.0, spacing, 2 * spacing, 3 * spacing]
    return {
        "x-arm": [(channel, coordinate, 0.0, 0.0) for channel, coordinate in zip(groups["x-arm"], horizontal, strict=True)],
        "y-arm": [(channel, 0.0, coordinate, 0.0) for channel, coordinate in zip(groups["y-arm"], horizontal, strict=True)],
        "z-arm": [(channel, 0.0, 0.0, coordinate) for channel, coordinate in zip(groups["z-arm"], vertical, strict=True)],
    }


def draw_array(ax: plt.Axes) -> None:
    positions = array_geometry()
    for group, rows in positions.items():
        xyz = np.asarray([[row[1], row[2], row[3]] for row in rows])
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=ARM_COLORS[group], lw=2.0)
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=ARM_COLORS[group], s=58, depthshade=False, edgecolor="white", linewidth=0.5)
    ax.scatter([0], [0], [0], marker="+", color=TEXT_COLOR, s=80, linewidth=1.2)
    ax.set_xlim(-1.95, 1.95); ax.set_ylim(-1.95, 1.95); ax.set_zlim(-1.3, 1.95)
    ax.set_box_aspect((1.0, 1.0, 0.9)); ax.set_proj_type("ortho"); ax.view_init(elev=25, azim=-52)
    ax.set_xlabel("x (m)", labelpad=2); ax.set_ylabel("y (m)", labelpad=2); ax.set_zlabel("z (m)", labelpad=1)
    ax.tick_params(axis="both", labelsize=FONT_TICK, pad=0, width=0.8); ax.zaxis.set_tick_params(labelsize=FONT_TICK, pad=0, width=0.8)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_formatter(FormatStrFormatter("%g"))
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0)); axis.pane.set_edgecolor("#B5B5B5")
        axis._axinfo["grid"].update(color=(0.84, 0.84, 0.84, 0.70), linewidth=0.55)
    handles = [Line2D([0], [0], color=ARM_COLORS["x-arm"], lw=2, marker="o", ms=5, label="x arm (6)"), Line2D([0], [0], color=ARM_COLORS["y-arm"], lw=2, marker="o", ms=5, label="y arm (6)"), Line2D([0], [0], color=ARM_COLORS["z-arm"], lw=2, marker="o", ms=5, label="z arm (7)")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.96), fontsize=FONT_LEGEND, handlelength=1.35, labelspacing=0.35, borderaxespad=0.0)


def draw_array_photo_with_topology_overlay(ax: plt.Axes, path: Path) -> None:
    with Image.open(path) as source:
        image = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
    height, width = image.shape[:2]
    pad = max(0.0, (width - height) / 2.0)
    # Native pixel coordinates are retained so the topology guide can be
    # registered directly to the three visible physical arms. The overlay is
    # a topology correspondence, not a recovery of metrological spacing.
    ax.imshow(image, extent=(0, width, height, 0), interpolation="antialiased")
    ax.set_xlim(0, width)
    ax.set_ylim(height + pad, -pad)
    ax.set_box_aspect(1)
    ax.set_axis_off()

    projected_arms = {
        "x-arm": np.asarray([(2, 329), (293, 334), (535, 322)], dtype=float),
        "y-arm": np.asarray([(221, 350), (293, 334), (430, 276)], dtype=float),
        "z-arm": np.asarray([(304, 463), (293, 334), (288, 8)], dtype=float),
    }
    labels = {
        "x-arm": (500, 350, "x arm (6)", "right", "top"),
        "y-arm": (432, 260, "y arm (6)", "left", "bottom"),
        "z-arm": (309, 30, "z arm (7)", "left", "top"),
    }
    for group, points in projected_arms.items():
        ax.plot(points[:, 0], points[:, 1], color="white", lw=6.0, alpha=0.82, solid_capstyle="round", zorder=6)
        ax.plot(points[:, 0], points[:, 1], color=ARM_COLORS[group], lw=3.2, alpha=0.95, solid_capstyle="round", zorder=7)
        x, y, label, horizontal, vertical = labels[group]
        text = ax.text(x, y, label, color=ARM_COLORS[group], fontsize=FONT_LEGEND, ha=horizontal, va=vertical, zorder=8)
        text.set_path_effects([path_effects.withStroke(linewidth=3.0, foreground="white")])
    ax.scatter([293], [334], s=44, facecolor="white", edgecolor=TEXT_COLOR, linewidth=1.0, zorder=9)


def interpolate_truth(times: np.ndarray, truth_times: np.ndarray, truth_xyz: np.ndarray) -> np.ndarray:
    if times[0] < truth_times[0] or times[-1] > truth_times[-1]:
        raise RuntimeError("GPS truth does not cover the APCE time grid")
    return np.column_stack([np.interp(times, truth_times, truth_xyz[:, dim]) for dim in range(3)])


def ribbon_polygon(xy: np.ndarray, radius: np.ndarray) -> np.ndarray:
    tangent = np.empty_like(xy)
    tangent[0], tangent[-1] = xy[1] - xy[0], xy[-1] - xy[-2]
    tangent[1:-1] = xy[2:] - xy[:-2]
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    left, right = xy + normal * radius[:, None], xy - normal * radius[:, None]
    return np.vstack((left, right[::-1]))


def integer_axes(ax: plt.Axes) -> None:
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        axis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    ax.tick_params(axis="both", labelsize=FONT_TICK, width=0.9, length=3.2)


def configure() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
        "font.weight": "normal", "axes.titleweight": "normal",
        "axes.labelweight": "normal", "axes.labelsize": FONT_AXIS,
        "xtick.labelsize": FONT_TICK, "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_LEGEND, "axes.linewidth": 0.9,
        "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
    })


def add_header(fig: plt.Figure, ax: plt.Axes, letter: str, title: str) -> None:
    position = ax.get_position()
    x = position.x0 - 0.035
    y = position.y1 + 0.010
    fig.text(x, y, letter, fontsize=FONT_PANEL, fontweight="bold", ha="left", va="baseline", color=TEXT_COLOR)
    fig.text(x + 0.047, y + 0.002, title, fontsize=FONT_TITLE, fontweight="normal", ha="left", va="baseline", color=TEXT_COLOR)


def save_outputs(fig: plt.Figure, stem: Path) -> dict[str, str]:
    outputs = {}
    for ext, kwargs in (("png", {"dpi": FIG_DPI}), ("pdf", {}), ("svg", {}), ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}})):
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, facecolor="white", **kwargs)
        outputs[ext] = str(path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend", type=Path, required=True)
    parser.add_argument("--apce-runs", type=Path, required=True)
    parser.add_argument("--nodes-csv", type=Path, required=True)
    parser.add_argument("--array-photo", type=Path, required=True)
    parser.add_argument("--array-photo-source-docx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()

    truth_times, truth_all = load_truth(args.frontend / "gps_truth.csv")
    node_rows, nodes = load_nodes(args.nodes_csv)
    run = load_apce_runs(args.apce_runs)
    times = np.asarray(run["times"])
    if len(times) != 67:
        raise RuntimeError(f"expected 67 one-second frames, found {len(times)}")
    truth = interpolate_truth(times, truth_times, truth_all)
    apce = np.median(np.asarray(run["positions"]), axis=0)
    width = np.median(np.asarray(run["widths"]), axis=0)
    errors = np.linalg.norm(apce - truth, axis=1)
    elapsed = times - times[0]
    # GPS, APCE and node coordinates already share the frontend's node-centred
    # local ENU frame. Do not recenter the trajectories independently.
    truth_xy, apce_xy = truth[:, :2], apce[:, :2]

    # Only a scalar mean marginal width is retained by the APCE run files.
    # The horizontal band is therefore a registered isotropic visual proxy.
    ribbon = ribbon_polygon(apce_xy, width / 2.0)

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
    grid = fig.add_gridspec(2, 2, left=0.07, right=0.97, bottom=0.05, top=0.95, wspace=0.20, hspace=0.20)
    ax_a, ax_b = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    ax_c_model = fig.add_subplot(grid[1, 0], projection="3d")
    ax_c_photo = fig.add_subplot(grid[1, 1])

    ax_a.fill(ribbon[:, 0], ribbon[:, 1], color=APCE_COLOR, alpha=0.17, linewidth=0, zorder=1)
    ax_a.plot(truth_xy[:, 0], truth_xy[:, 1], color=GPS_COLOR, lw=2.05, ls="--", label="GPS", zorder=5)
    ax_a.plot(apce_xy[:, 0], apce_xy[:, 1], color=APCE_COLOR, lw=1.85, label="APCE", zorder=6)
    ax_a.scatter(apce_xy[0, 0], apce_xy[0, 1], s=46, marker="o", facecolor="white", edgecolor=APCE_COLOR, linewidth=1.2, zorder=7)
    ax_a.scatter(apce_xy[-1, 0], apce_xy[-1, 1], s=46, marker="s", facecolor="white", edgecolor=APCE_COLOR, linewidth=1.2, zorder=7)
    ax_a.scatter(nodes[:, 0], nodes[:, 1], s=50, marker="P", color="#111111", edgecolor="white", linewidth=0.55, zorder=8)
    for row, point in zip(node_rows, nodes, strict=True):
        ax_a.annotate(f"N{row['node_id']}", (point[0], point[1]), xytext=(4, 4), textcoords="offset points", fontsize=FONT_LEGEND, color=TEXT_COLOR, ha="left", va="bottom", zorder=9)
    ax_a.set_xlabel("East offset (m)")
    ax_a.set_ylabel("North offset (m)")
    # Reserve a curve-free strip below the orbit for the in-panel legend.
    ax_a.set_xlim(-650, 650)
    ax_a.set_ylim(-650, 650)
    ax_a.set_aspect("equal", adjustable="box")
    ax_a.set_box_aspect(1)
    ax_a.grid(color="#DFDFDF", linewidth=0.55, alpha=0.8, zorder=0)
    integer_axes(ax_a)
    handles = [
        Line2D([0], [0], color=GPS_COLOR, lw=2.05, ls="--", label="GPS"),
        Line2D([0], [0], color=APCE_COLOR, lw=1.85, label="APCE"),
        Patch(facecolor=APCE_COLOR, edgecolor="none", alpha=0.17, label="APCE 90% width proxy"),
        Line2D([0], [0], color="#111111", marker="P", ls="None", ms=6, label="Array node"),
    ]
    ax_a.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.012), fontsize=FONT_LEGEND, handlelength=1.75, labelspacing=0.42, borderaxespad=0.0, ncol=2, columnspacing=0.9)

    ax_b.plot(elapsed, errors, color="#3F6B8F", lw=1.75, label="Position error")
    ax_b.plot(elapsed, width, color=APCE_COLOR, lw=1.75, label="90% marginal width")
    ax_b.set_xlabel("Elapsed time (s)")
    ax_b.set_ylabel("Distance (m)")
    ax_b.grid(color="#DFDFDF", linewidth=0.55, alpha=0.8)
    ax_b.set_box_aspect(1)
    integer_axes(ax_b)
    ax_b.legend(loc="upper left", bbox_to_anchor=(0.015, 0.985), fontsize=FONT_LEGEND, handlelength=1.75, labelspacing=0.45, borderaxespad=0.0)

    draw_array(ax_c_model)
    draw_array_photo_with_topology_overlay(ax_c_photo, args.array_photo)

    for ax in (ax_a, ax_b):
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.9)
            spine.set_color(TEXT_COLOR)

    fig.canvas.draw()
    add_header(fig, ax_a, "a", "Single-source trajectory and array nodes")
    add_header(fig, ax_b, "b", "APCE error and uncertainty proxy")
    add_header(fig, ax_c_model, "c", "Nineteen-microphone array: model-to-hardware topology")
    panel_positions = {letter: axis.get_position() for letter, axis in (("a", ax_a), ("b", ax_b), ("c_model", ax_c_model), ("c_photo_overlay", ax_c_photo))}
    panel_boxes = {
        letter: {"width_fraction": float(position.width), "height_fraction": float(position.height), "width_to_height": float(position.width / position.height)}
        for letter, position in panel_positions.items()
    }

    args.output.mkdir(parents=True, exist_ok=True)
    photo_mirror = args.output / "source_photo_2017_array.jpeg"
    shutil.copy2(args.array_photo, photo_mirror)
    stem = args.output / "supplementary_data_figure2_single_source_gps_apce_uncertainty"
    outputs = save_outputs(fig, stem)
    plt.close(fig)

    source_path = args.output / "supplementary_data_figure2_single_source_gps_apce_uncertainty_source.csv"
    rows = []
    for index, time_s in enumerate(times):
        rows.append({
            "sample_index": index, "time_s": float(time_s), "elapsed_s": float(elapsed[index]),
            "gps_east_m": float(truth[index, 0]), "gps_north_m": float(truth[index, 1]), "gps_up_m": float(truth[index, 2]),
            "apce_east_median_5seeds_m": float(apce[index, 0]), "apce_north_median_5seeds_m": float(apce[index, 1]),
            "apce_up_median_5seeds_m": float(apce[index, 2]), "apce_position_error_of_median_m": float(errors[index]),
            "apce_median_mean_marginal_90pct_interval_width_m": float(width[index]),
            "apce_median_proxy_radius_m": float(width[index] / 2.0),
        })
    with source_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    panel_path = args.output / "supplementary_data_figure2_single_source_gps_apce_uncertainty_panel_registry.csv"
    panels = [
        {"panel": "a", "content": "GPS and five-seed median APCE horizontal trajectories with nine array-node positions", "selection": "fixed 67-frame window 46254--46320 s; all nine frontend nodes", "uncertainty": "median scalar mean-marginal 90% interval half-width drawn normal to trajectory; isotropic visual proxy, not full EN covariance", "sources": f"{args.frontend}; {args.apce_runs}; {args.nodes_csv}"},
        {"panel": "b", "content": "position error of five-seed median trajectory and retained APCE interval width", "selection": "same fixed 67-frame window", "uncertainty": "mean width of six-dimensional state componentwise weighted 5th--95th percentile intervals, restricted to the first three position components", "sources": f"{args.frontend}; {args.apce_runs}"},
        {"panel": "c", "content": "shared 19-microphone three-arm coordinate model and unframed 2017 field-array photograph with the same arm colours projected directly onto the physical rods", "selection": "single integrated model--hardware panel shown once for the single- and dual-source cases; embedded image 4 selected before layout", "uncertainty": "not applicable", "sources": f"current MUSIC frontend legacy nonuniform-z 6 + 6 + 7 geometry; {args.array_photo_source_docx}; {args.array_photo}"},
    ]
    with panel_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(panels[0])); writer.writeheader(); writer.writerows(panels)

    rmse = float(np.sqrt(np.mean(np.square(errors))))
    pooled_rmse = float(np.sqrt(np.mean(np.square(np.asarray(run["errors"])))))
    with Image.open(args.array_photo) as source_photo:
        photo_resolution = list(ImageOps.exif_transpose(source_photo).size)
    registry = {
        "figure_contract": {
            "core_conclusion": "The fixed 67-second single-source segment relates GPS/APCE trajectory agreement to the nine-node field geometry while retaining an honest display of APCE uncertainty.",
            "evidence_chain": {"a": "horizontal trajectory, APCE width proxy and nine array-node positions", "b": "time-resolved error and retained uncertainty width", "c": "single shared 19-microphone coordinate model mapped directly onto the corresponding arms in the original 2017 field photograph"},
            "archetype": "image plate plus quantitative grid", "backend": "Python/matplotlib on Super-Server",
        },
        "typography": {"reference": "main-text Figure 4/5 rules", "panel_label_pt": FONT_PANEL, "panel_title_pt": FONT_TITLE, "legend_pt": FONT_LEGEND, "axis_label_pt": FONT_AXIS, "tick_pt": FONT_TICK, "only_panel_letters_bold": True, "integer_tick_labels": True},
        "window": {"start_time_s": START_S, "end_time_s": END_S, "frames": len(times), "update_interval_s": float(np.median(np.diff(times))), "selection_status": "fixed before this figure revision"},
        "configuration": {"q_min_accel_mps2": 2.0, "q_max_accel_mps2": 12.0, "observation_covariance_scale": 1.0, "seeds": run["seeds"]},
        "metrics": {"median_trajectory_rmse_m": rmse, "median_trajectory_median_error_m": float(np.median(errors)), "median_trajectory_p90_error_m": float(np.percentile(errors, 90)), "pooled_five_seed_rmse_m": pooled_rmse, "median_interval_width_m": float(np.median(width)), "pooled_coverage_90": float(np.mean(np.asarray(run["coverages"])))},
        "uncertainty_semantics": {"stored_quantity": "per-frame mean width of componentwise weighted 5th--95th percentile intervals across the three position state components", "panel_a_rendering": "half the stored width as an isotropic normal ribbon around the median horizontal trajectory", "limitation": "visual proxy only; not a joint East-North covariance ellipse or calibrated 2-D confidence tube"},
        "gps_role": "offline evaluation and display only; not an APCE assimilation input",
        "sources": {"frontend": str(args.frontend), "apce_run_files": [str(path) for path in run["paths"]], "nodes_csv": str(args.nodes_csv), "array_photo": str(args.array_photo), "array_photo_source_docx": str(args.array_photo_source_docx), "array_photo_mirror": str(photo_mirror)},
        "array": {"nodes": 9, "microphones": 19, "geometry": "three orthogonal arms, 6 + 6 + 7 microphones", "manifold_reused_for_single_and_dual": True},
        "image_integrity": {"panel": "c photo component", "raw_file": str(args.array_photo), "raw_sha256": sha256_file(args.array_photo), "mirrored_file": str(photo_mirror), "mirrored_sha256": sha256_file(photo_mirror), "crop": "none", "orientation": "EXIF transpose only", "brightness_contrast_gamma": "none", "pseudo_color": "none", "stitching": "none", "frame": "none", "scientific_overlay": "same-colour projected arm centre-lines and direct arm-count labels; illustrative topology registration, not metrological spacing recovery", "square_layout": "uncropped image centred with white padding", "resolution_px": photo_resolution, "resolution_note": "document-embedded image; sufficient for the present supporting panel but not a high-resolution source photograph"},
        "array_geometry_scope": "The photograph supports the three-arm topology but does not independently verify every model spacing.",
        "layout_qa": {"canvas_inches": [FIG_W, FIG_H], "grid": "2 x 2 equal cells; lower two cells jointly form panel c", "panel_boxes": panel_boxes, "square_target": True, "title_offset_fraction": 0.010, "model_photo_link": "no connector arrows; the model colours are projected directly over the matching physical rods in the unframed photograph, which has no separate panel letter"},
        "outputs": outputs, "source_csv": str(source_path), "panel_registry": str(panel_path), "script": str(Path(__file__).resolve()), "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    registry_path = args.output / "supplementary_data_figure2_single_source_gps_apce_uncertainty_registry.json"
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**registry, "registry": str(registry_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
