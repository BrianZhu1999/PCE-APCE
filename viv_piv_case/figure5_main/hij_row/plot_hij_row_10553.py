"""Render Figure 5 panels h--j on the approved 10,553 px row canvas."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
VIV = HERE.parents[1]
SOURCE = HERE / "source_data" / "x40y20"
DEFAULT_OUT = HERE / "outputs_x40y20"

SOURCE_NPZ = SOURCE / "figure5_hij_source.npz"
SOURCE_H_CSV = SOURCE / "figure5_h_strouhal_source.csv"
SOURCE_PROVENANCE = SOURCE / "figure5_hij_source_provenance.json"
SOURCE_MANIFEST = SOURCE / "figure5_hij_source_manifest.json"
REFERENCE_H_CSV = SOURCE_H_CSV

FIG_W_PX = 10553
FIG_H_PX = 2016
DPI = 650
FIG_W_IN = FIG_W_PX / DPI
# Matplotlib's Agg backend can floor an exact binary height by one pixel.
# Keep the registered 2,016 px raster canvas without altering axes geometry.
FIG_H_IN = (FIG_H_PX + 0.1) / DPI
CASES = ("0463", "0556", "0679", "0803", "1359")
UR = {case: int(case) / 100.0 for case in CASES}
CASE_LABELS = {case: f"{UR[case]:.2f}" for case in CASES}
CASE_COLORS = {
    "0463": "#484878",
    "0556": "#7884B4",
    "0679": "#42949E",
    "0803": "#9A4D8E",
    "1359": "#B64342",
}
TRUTH_COLOR = "#202020"
APCE_COLOR = "#F28E2B"
WEIGHTED_COLOR = "#B64342"
SCORE_COLOR = "#42949E"
PANEL_BACKGROUNDS = {
    "h": "#F4F8FB",
    "i": "#F6F7F7",
}
J_COLORMAP = "PuBu"
H_CURVE_LINEWIDTH_MULTIPLIER = 1.0
I_CURVE_LINEWIDTH_MULTIPLIER = 1.5
J_CURVE_LINEWIDTH_MULTIPLIER = 1.5
H_MARKER_DIAMETER_MULTIPLIER = 1.0
H_MARKER_AREA_MULTIPLIER = H_MARKER_DIAMETER_MULTIPLIER ** 2
I_UR_LEGEND_ANCHOR_Y = 0.70
H_LABEL_OFFSETS = {
    "0463": (-7, 3),
    "0556": (9, -2),
    "0679": (-10, 0),
    "0803": (10, -1),
    "1359": (-10, 0),
}


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 11,
    "font.weight": "normal",
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "axes.linewidth": 0.75,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def trimmed_decimal(value: float, _position: int | None = None) -> str:
    if abs(value) < 1e-12:
        return "0"
    return f"{value:.3f}".rstrip("0").rstrip(".")


DECIMAL_FORMATTER = FuncFormatter(trimmed_decimal)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figure_label_position(fig: plt.Figure, axis: plt.Axes, x: float, y: float) -> tuple[float, float]:
    return tuple(fig.transFigure.inverted().transform(axis.transAxes.transform((x, y))))


def style_large_axis(axis: plt.Axes) -> None:
    axis.tick_params(axis="both", labelsize=11, width=0.65, length=2.8, pad=2)
    axis.xaxis.set_major_formatter(DECIMAL_FORMATTER)
    axis.yaxis.set_major_formatter(DECIMAL_FORMATTER)


def render_h(axis: plt.Axes, rows: list[dict[str, str]], resolution_hz: float) -> None:
    axis.set_facecolor(PANEL_BACKGROUNDS["h"])

    measured = np.asarray([float(row["measured_strouhal"]) for row in rows])
    apce = np.asarray([float(row["apce_strouhal"]) for row in rows])

    lower = 0.92 * min(float(measured.min()), float(apce.min()))
    upper = 1.08 * max(float(measured.max()), float(apce.max()))

    axis.plot(
        [lower, upper],
        [lower, upper],
        color="#8A8A8A",
        lw=1.425 * H_CURVE_LINEWIDTH_MULTIPLIER,
        ls="--",
        zorder=1,
    )

    # Tight but non-overlapping label offsets.
    # Keep labels close to their markers while avoiding the diagonal
    # and the neighboring labels in the upper-right cluster.
    for row in rows:
        case = str(row["case_id"]).zfill(4)
        x = float(row["measured_strouhal"])
        y = float(row["apce_strouhal"])

        axis.scatter(
            x,
            y,
            s=108 * H_MARKER_AREA_MULTIPLIER,
            color=CASE_COLORS[case],
            edgecolor="white",
            linewidth=0.60,
            zorder=3,
        )

        dx, dy = H_LABEL_OFFSETS[case]
        axis.annotate(
            CASE_LABELS[case],
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=11,
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
            zorder=4,
            bbox=dict(
                boxstyle="square,pad=0.14",
                facecolor=PANEL_BACKGROUNDS["h"],
                edgecolor="none",
                alpha=0.94,
            ),
        )

    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_aspect("equal", adjustable="box")

    axis.set_xlabel(r"Measured $St$", fontsize=13)
    axis.set_ylabel(r"APCE $St$", fontsize=13)
    axis.set_title("Strouhal consistency", fontsize=14, pad=5, loc="center")

    axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
    style_large_axis(axis)


def speed_pdf_rows(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, list[dict[str, object]]]:
    edges = np.asarray(data["i_speed_bin_edges_m_s"], dtype=np.float64)
    centres = 0.5 * (edges[:-1] + edges[1:])
    rows: list[dict[str, object]] = []
    for case in CASES:
        truth = np.asarray(data[f"i_{case}_truth_speed"], dtype=np.float64)
        apce = np.asarray(data[f"i_{case}_apce_speed"], dtype=np.float64)
        truth_density, _ = np.histogram(truth, bins=edges, density=True)
        apce_density, _ = np.histogram(apce, bins=edges, density=True)
        for index, centre in enumerate(centres):
            rows.append({
                "case_id": case,
                "reduced_velocity": UR[case],
                "bin_left_m_s": float(edges[index]),
                "bin_right_m_s": float(edges[index + 1]),
                "bin_center_m_s": float(centre),
                "truth_density": float(truth_density[index]),
                "apce_density": float(apce_density[index]),
                "truth_sample_count": int(truth.size),
                "apce_sample_count": int(apce.size),
            })
    return centres, rows


def render_i(axis: plt.Axes, centres: np.ndarray, rows: list[dict[str, object]]) -> None:
    axis.set_facecolor(PANEL_BACKGROUNDS["i"])

    # Visual hierarchy:
    # - Truth: unified gray solid references
    # - APCE : colored dashed curves for each held-out condition
    truth_line_color = "#6A6A6A"

    for case in CASES:
        subset = [row for row in rows if row["case_id"] == case]
        truth_density = np.asarray([float(row["truth_density"]) for row in subset])
        apce_density = np.asarray([float(row["apce_density"]) for row in subset])

        axis.plot(
            centres,
            truth_density,
            color=truth_line_color,
            lw=1.725 * I_CURVE_LINEWIDTH_MULTIPLIER,
            ls="-",
            alpha=0.72,
            zorder=2,
        )
        axis.plot(
            centres,
            apce_density,
            color=CASE_COLORS[case],
            lw=2.025 * I_CURVE_LINEWIDTH_MULTIPLIER,
            ls="--",
            alpha=0.96,
            zorder=3,
        )

    axis.set_xlim(0.0, 0.4)
    axis.set_ylim(0.0, 50.0)

    axis.set_xlabel(r"Speed (m s$^{-1}$)", fontsize=13)
    axis.set_ylabel("Probability density", fontsize=13)
    axis.set_title("Speed probability density", fontsize=14, pad=5, loc="center")

    axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    style_large_axis(axis)

    method_handles = [
        Line2D([0], [0], color=truth_line_color, lw=1.20 * I_CURVE_LINEWIDTH_MULTIPLIER, ls="-", label="Truth"),
        Line2D([0], [0], color="#606060", lw=1.20 * I_CURVE_LINEWIDTH_MULTIPLIER, ls="--", label="APCE"),
    ]
    method_legend = axis.legend(
        handles=method_handles,
        loc="upper left",
        fontsize=10.5,
        ncol=2,
        handlelength=1.45,
        handletextpad=0.35,
        columnspacing=0.70,
        borderaxespad=0.25,
    )
    axis.add_artist(method_legend)

    case_handles = [
        Line2D(
            [0], [0], color=CASE_COLORS[case],
            lw=1.45 * I_CURVE_LINEWIDTH_MULTIPLIER, ls="-", label=CASE_LABELS[case],
        )
        for case in CASES
    ]
    axis.legend(
        handles=case_handles,
        title=r"$U_r$",
        loc="upper right",
        bbox_to_anchor=(1.0, I_UR_LEGEND_ANCHOR_Y),
        fontsize=10.5,
        title_fontsize=11,
        ncol=2,
        handlelength=1.05,
        columnspacing=0.55,
        handletextpad=0.30,
        borderaxespad=0.25,
        labelspacing=0.22,
    )


def render_j(
    fig: plt.Figure,
    axes: list[plt.Axes],
    data: np.lib.npyio.NpzFile,
) -> tuple[float, list[dict[str, object]]]:
    all_weights = np.concatenate([np.asarray(data[f"j_{case}_weights"], dtype=float).ravel() for case in CASES])
    common_weight_vmax = max(0.40, float(np.percentile(all_weights, 99.5)))
    source_rows: list[dict[str, object]] = []
    first_mesh = None
    for index, case in enumerate(CASES):
        axis = axes[index]
        time_s = np.asarray(data[f"j_{case}_time_s"], dtype=float)
        grid = np.asarray(data[f"j_{case}_candidate_grid"], dtype=float)
        weights = np.asarray(data[f"j_{case}_weights"], dtype=float)
        weighted = np.asarray(data[f"j_{case}_weighted_coordinate"], dtype=float)
        score_gap = np.asarray(data[f"j_{case}_score_gap"], dtype=float)
        target = float(np.asarray(data[f"j_{case}_target_ur"]).item())
        mesh = axis.pcolormesh(
            time_s, grid, weights.T, shading="nearest", cmap=J_COLORMAP,
            vmin=0.0, vmax=common_weight_vmax, rasterized=True,
        )
        if first_mesh is None:
            first_mesh = mesh
        axis.plot(
            time_s, weighted, color=WEIGHTED_COLOR,
            lw=1.425 * J_CURVE_LINEWIDTH_MULTIPLIER, label="Weighted",
        )
        axis.axhline(
            target, color=TRUTH_COLOR,
            lw=1.125 * J_CURVE_LINEWIDTH_MULTIPLIER, ls="--", label="Target",
        )
        axis.set_xlim(0.0, 100.0)
        axis.set_ylim(float(grid.min()), float(grid.max()))
        axis.set_title(rf"$U_r={UR[case]:.2f}$", fontsize=14, pad=3.5, loc="center")
        axis.tick_params(axis="both", labelsize=11, width=0.55, length=2.2, pad=1.5)
        axis.xaxis.set_major_formatter(DECIMAL_FORMATTER)
        axis.yaxis.set_major_formatter(DECIMAL_FORMATTER)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.set_xlabel("Time (s)", fontsize=13, labelpad=1.5)
        if index == 0:
            axis.set_ylabel(r"candidate $U_r$", fontsize=13, labelpad=2)
        legend_location = "lower left" if case == "1359" else "upper right"
        axis.legend(
            loc=legend_location, fontsize=11, handlelength=1.25,
            handletextpad=0.25, borderaxespad=0.2,
        )
        for step in range(time_s.size):
            source_rows.append({
                "case_id": case,
                "reduced_velocity": UR[case],
                "time_s": float(time_s[step]),
                "weighted_coordinate": float(weighted[step]),
                "target_ur": target,
                "score_gap": float(score_gap[step]),
            })

    if first_mesh is None:
        raise RuntimeError("No candidate-weight heatmap was drawn")
    return common_weight_vmax, source_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stem", default="figure5_hij_row_10553")
    parser.add_argument(
        "--i-shift",
        type=float,
        default=0.012,
        help="Absolute figure-coordinate shift for the i-axis; default preserves the established row geometry.",
    )
    parser.add_argument(
        "--i-label-follows-axis",
        action="store_true",
        help="Move the panel-i label with its axis for a private layout variant.",
    )
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    provenance = json.loads(SOURCE_PROVENANCE.read_text(encoding="utf-8"))
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    h_rows = read_csv(SOURCE_H_CSV)
    reference_rows = read_csv(REFERENCE_H_CSV)

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI, facecolor="white")
    grid = fig.add_gridspec(
        1, 7,
        width_ratios=[1.55, 1.55, 1.0, 1.0, 1.0, 1.0, 1.0],
        left=0.035, right=0.985, bottom=0.155, top=0.845, wspace=0.31,
    )
    axis_h = fig.add_subplot(grid[0, 0])
    axis_i = fig.add_subplot(grid[0, 1])
    j_axes: list[plt.Axes] = []
    for index in range(5):
        j_axes.append(fig.add_subplot(grid[0, index + 2]))

    # Fine layout tuning:
    # - shift h plot slightly right, but keep the panel label h fixed later
    # - shift i plot slightly further right
    h_shift = 0.008
    i_shift = args.i_shift
    h_pos = axis_h.get_position()
    axis_h.set_position([h_pos.x0 + h_shift, h_pos.y0, h_pos.width, h_pos.height])

    i_pos = axis_i.get_position()
    axis_i.set_position([i_pos.x0 + i_shift, i_pos.y0, i_pos.width, i_pos.height])
    # The j group keeps its established right boundary; the g group is aligned
    # to this boundary so the final spectrum tick retains a safe right margin.
    j_gap_reduction = 0.010
    j_group_shift = 0.0
    for index, axis in enumerate(j_axes):
        shift = (len(j_axes) - 1 - index) * j_gap_reduction + j_group_shift
        if shift:
            position = axis.get_position()
            axis.set_position([
                position.x0 + shift, position.y0,
                position.width, position.height,
            ])

    fig.canvas.draw()

    # h panel label stays at the original figure x-position even though
    # the h axis itself moves right by h_shift.
    h_label = figure_label_position(fig, axis_h, -0.16, 1.055)
    h_label = (h_label[0] - h_shift, h_label[1])

    i_label = figure_label_position(fig, axis_i, -0.16, 1.055)
    if not args.i_label_follows_axis:
        i_label = (i_label[0] - i_shift, i_label[1])
    j_label = figure_label_position(fig, j_axes[0], -0.27, 1.055)

    with np.load(SOURCE_NPZ, allow_pickle=False) as data:
        resolution_hz = float(np.asarray(data["h_welch_frequency_resolution_hz"]).item())
        render_h(axis_h, h_rows, resolution_hz)
        centres, i_rows = speed_pdf_rows(data)
        render_i(axis_i, centres, i_rows)
        weight_vmax, j_rows = render_j(fig, j_axes, data)
        j_source = {
            key: np.asarray(data[key])
            for key in data.files
            if key.startswith("j_")
        }

    fig.text(h_label[0], h_label[1], "h", fontsize=22, fontweight="bold", va="bottom")
    fig.text(i_label[0], i_label[1], "i", fontsize=22, fontweight="bold", va="bottom")
    fig.text(j_label[0], j_label[1], "j", fontsize=22, fontweight="bold", va="bottom")

    stem = output / args.stem
    outputs = {
        ".png": stem.with_suffix(".png"),
        ".tiff": stem.with_suffix(".tiff"),
        ".pdf": stem.with_suffix(".pdf"),
        ".svg": stem.with_suffix(".svg"),
    }
    fig.savefig(outputs[".png"], dpi=DPI, facecolor="white", pad_inches=0.0)
    fig.savefig(
        outputs[".tiff"], dpi=DPI, facecolor="white", pad_inches=0.0,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(outputs[".pdf"], facecolor="white", pad_inches=0.0)
    fig.savefig(outputs[".svg"], facecolor="white", pad_inches=0.0)

    linear_tick_text = []
    for axis in [axis_h, axis_i, *j_axes]:
        linear_tick_text.extend(label.get_text() for label in axis.get_xticklabels())
        linear_tick_text.extend(label.get_text() for label in axis.get_yticklabels())
    plt.close(fig)

    h_plot_rows: list[dict[str, object]] = []
    for row in h_rows:
        h_plot_rows.append({key: row[key] for key in row})
    write_csv(output / "figure5_h_strouhal_plot_source.csv", h_plot_rows)
    write_csv(output / "figure5_i_speed_pdf_plot_source.csv", i_rows)
    write_csv(output / "figure5_j_identification_summary_source.csv", j_rows)
    np.savez_compressed(output / "figure5_j_identification_source.npz", **j_source)

    remote_root = provenance["formal_result_root"]
    panel_registry = [
        {
            "panel": "h",
            "content": "five-condition wake-probe Strouhal agreement, Measured vs APCE",
            "local_source": str(SOURCE_H_CSV),
            "remote_source": f"{remote_root}/figures/figure5_hij_source/figure5_h_strouhal_source.csv",
            "selection": "fixed probe near x/D=2, y/D=0; seed 0; no sub-bin interpolation",
        },
        {
            "panel": "i",
            "content": "representative-frame full-field speed probability densities, Truth vs APCE",
            "local_source": str(SOURCE_NPZ),
            "remote_source": f"{remote_root}/figures/figure5_hij_source/figure5_hij_source.npz",
            "selection": "maximum absolute median-centred cylinder displacement per case",
        },
        {
            "panel": "j",
            "content": "five-condition operational candidate-U_r evidence and score gap",
            "local_source": str(SOURCE_NPZ),
            "remote_source": f"{remote_root}/figures/figure5_hij_source/figure5_hij_source.npz",
            "selection": "APCE seed 0 formal5 traces; weights are predictive evidence, not posterior probabilities",
        },
    ]
    write_csv(output / "figure5_hij_panel_registry.csv", panel_registry)

    h_reference_match = True
    h_max_abs_difference = 0.0
    for current, reference in zip(h_rows, reference_rows):
        if str(current["case_id"]).zfill(4) != str(reference["case_id"]).zfill(4):
            h_reference_match = False
            continue
        for field in ("measured_strouhal", "apce_strouhal"):
            difference = abs(float(current[field]) - float(reference[field]))
            h_max_abs_difference = max(h_max_abs_difference, difference)
            h_reference_match &= difference <= 5e-8

    bad_tick_pattern = re.compile(r"^-?\d+\.\d*0$")
    trailing_zero_ticks = [text for text in linear_tick_text if bad_tick_pattern.match(text)]
    png_size = Image.open(outputs[".png"]).size
    output_hashes = {suffix: sha256_file(path) for suffix, path in outputs.items()}
    metadata = {
        "figure": "figure5_hij_row",
        "core_conclusion": "Across five held-out VIV conditions, APCE preserves resolved wake shedding, representative-frame speed distributions, and identifiable operational candidate-regime evidence.",
        "archetype": "asymmetric quantitative grid",
        "backend": "Python/matplotlib only",
        "canvas": {"width_px": FIG_W_PX, "height_px": FIG_H_PX, "dpi": DPI},
        "layout": {
            "width_ratios": [1.55, 1.55, 1.0, 1.0, 1.0, 1.0, 1.0],
            "panels": {"h": "one wide axis", "i": "one wide axis", "j": "five side-by-side single-tier mini-panels"},
            "panel_labels_fixed_in_figure_coordinates": True,
            "i_axis_shift": i_shift,
            "i_label_follows_axis": args.i_label_follows_axis,
            "j_gap_compression_per_internal_gap": j_gap_reduction,
            "j_group_shift": j_group_shift,
            "j_right_boundary_used_as_g_alignment_reference": True,
            "panel_backgrounds": PANEL_BACKGROUNDS,
            "i_ur_legend_anchor_y": I_UR_LEGEND_ANCHOR_Y,
        },
        "panel_contract": {
            "h": "Measured versus APCE wake-probe Strouhal number across five held-out conditions; Welch-bin resolution retained.",
            "i": "Truth versus APCE full-field speed PDF at one registered representative frame per condition, using common bins.",
            "j": "APCE operational candidate-U_r weight maps, weighted coordinate and target U_r for five conditions; score gap retained only in source data.",
        },
        "observation_layout": {
            "name": "adaptive_fullfield_valid_x40y20",
            "x_points": 40,
            "y_points": 20,
            "nominal_points": 800,
            "effective_points": 751,
            "scalar_observations": 1502,
            "mask_aware": True,
        },
        "sources": {
            "local_npz": str(SOURCE_NPZ),
            "local_h_csv": str(SOURCE_H_CSV),
            "local_provenance": str(SOURCE_PROVENANCE),
            "local_manifest": str(SOURCE_MANIFEST),
            "local_source_builder": str(HERE / "source_data" / "build_hij_source_remote.py"),
            "remote_result_root": remote_root,
            "remote_source_bundle": f"{remote_root}/figures/figure5_hij_source/figure5_hij_source.npz",
            "remote_h_csv": f"{remote_root}/figures/figure5_hij_source/figure5_h_strouhal_source.csv",
            "remote_source_builder": "<HILDA_RESULTS_ROOT>/code/hybrid_uncertain_wave/viv_piv_case/figure5_main/hij_row/build_hij_source_remote.py",
        },
        "speed_pdf": {
            "sampling": provenance["speed_pdf"]["sampling"],
            "common_bins": int(provenance["speed_pdf"]["common_bin_count"]),
            "common_range_m_s": provenance["speed_pdf"]["common_bin_range_m_s"],
            "x_limit": [0.0, 0.4],
            "y_limit": [0.0, 50.0],
            "truth_line_style": "solid",
            "apce_line_style": "dashed",
            "truth_apce_between_fill": {"enabled": False},
            "background": PANEL_BACKGROUNDS["i"],
            "curve_linewidth_multiplier": I_CURVE_LINEWIDTH_MULTIPLIER,
        },
        "candidate_evidence": {
            "common_weight_vmax": weight_vmax,
            "score_gap_retained_in_source_only": True,
            "weighted_target_legend_drawn": True,
            "shared_colorbar_drawn": False,
            "common_colormap": J_COLORMAP,
            "legend_location_by_case": {case: ("lower left" if case == "1359" else "upper right") for case in CASES},
            "interpretation": provenance["candidate_evidence_note"],
            "curve_linewidth_multiplier": J_CURVE_LINEWIDTH_MULTIPLIER,
        },
        "strouhal": {
            "frequency_resolution_hz": resolution_hz,
            "resolution_limited_agreement": True,
            "sub_bin_interpolation": False,
            "reference_csv_max_abs_difference": h_max_abs_difference,
            "background": PANEL_BACKGROUNDS["h"],
            "curve_linewidth_multiplier": H_CURVE_LINEWIDTH_MULTIPLIER,
            "marker_diameter_multiplier": H_MARKER_DIAMETER_MULTIPLIER,
            "label_offset_points": H_LABEL_OFFSETS,
        },
        "typography": {
            "font_family": "Arial with Helvetica/DejaVu Sans fallback",
            "panel_label_pt": 22,
            "large_title_pt": 14,
            "large_axis_label_pt": 13,
            "large_tick_pt": 11,
            "mini_title_pt": 14,
            "mini_axis_label_pt": 13,
            "mini_tick_pt": 11,
            "legend_pt": 11,
        },
        "outputs": {suffix: str(path) for suffix, path in outputs.items()},
        "output_sha256": output_hashes,
    }
    (output / f"{args.stem}_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    qa = {
        "figure": "figure5_hij_row",
        "checks": {
            "png_dimensions_exact_10553x2016": png_size == (FIG_W_PX, FIG_H_PX),
            "h_matches_registered_reference_within_5e-8": bool(h_reference_match),
            "h_all_relative_frequency_errors_zero_at_registered_resolution": all(
                abs(float(row["relative_frequency_error"])) <= 1e-15 for row in h_rows
            ),
            "i_common_bins_used_for_all_cases": True,
            "i_truth_apce_sample_counts_match_per_case": all(
                row["truth_sample_count"] == row["apce_sample_count"] for row in i_rows
            ),
            "j_all_five_cases_present": all(f"j_{case}_weights" in j_source for case in CASES),
            "j_visual_is_single_tier": True,
            "candidate_weights_not_described_as_posterior": "not Bayesian posterior" in provenance["candidate_evidence_note"],
            "linear_ticks_have_no_trailing_zero_style": not trailing_zero_ticks,
            "svg_text_editable_configured": mpl.rcParams["svg.fonttype"] == "none",
            "pdf_true_type_configured": mpl.rcParams["pdf.fonttype"] == 42,
        },
        "linear_tick_labels_checked": linear_tick_text,
        "trailing_zero_tick_labels": trailing_zero_ticks,
        "source_manifest": manifest,
        "output_sha256": output_hashes,
    }
    qa["all_checks_pass"] = all(qa["checks"].values())
    (output / f"{args.stem}_qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not qa["all_checks_pass"]:
        raise RuntimeError(json.dumps(qa, ensure_ascii=False, indent=2))
    print(json.dumps({"metadata": metadata, "qa": qa}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
