from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
import numpy as np

import make_figure2_classical_systems_v12_fghi_xticklabels as base


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


OUTPUT_STEM = "figure2_classical_uncertain_systems_v24_i_legend_aligned"

METRIC_GROUPS = (
    ("nrmse", r"nRMSE (%)", 100.0, "e", "nRMSE"),
    ("crps", r"CRPS ($10^{-3}$)", 1000.0, "f", "CRPS"),
    ("coverage_90", "90% coverage", 1.0, "g", "90% coverage"),
    ("interval_width_90", r"Interval width ($10^{-2}$)", 100.0, "h", "Interval width"),
)

RADAR_DIMENSIONS = (
    "Acc.",
    "Dist.",
    "Cal.",
    "C.Sh.",
    "Win",
    "Cog.",
)

RADAR_GROUPS = (
    ("Error tests", 0, 1, "#D98DA7"),
    ("Uncertainty", 2, 3, "#E8B96B"),
    ("Reliability tests", 4, 5, "#72AAA6"),
)

RADAR_PLOT_METHODS = base.METHODS
RADAR_METHOD_LABELS = {
    "denkf": "DEnKF",
    "letkf": "LETKF",
    "iensf": "IEnSF",
    "pce": "PCE",
    "apce": "APCE",
}
RADAR_COLORS = {
    "denkf": base.COLORS["denkf"],
    "letkf": base.COLORS["letkf"],
    "iensf": base.COLORS["iensf"],
    "pce": base.COLORS["pce"],
    "apce": base.TOP_APCE_COLOR,
}
NONCOGNITIVE_BASELINES = ("denkf", "letkf", "iensf")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _score_higher_better(values: np.ndarray, mode: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if mode == "target_0.90":
        values = np.abs(values - 0.90)
    finite = np.isfinite(values)
    scores = np.full(values.shape, 0.5, dtype=float)
    if not np.any(finite):
        return scores
    v = values[finite]
    span = float(np.max(v) - np.min(v))
    if span <= 1.0e-14:
        scores[finite] = 1.0
    else:
        scores[finite] = 1.0 - (v - float(np.min(v))) / span
    return np.clip(scores, 0.0, 1.0)


def compute_radar_scores(
    summary: dict[tuple[str, str], dict[str, str]],
    run_rows: list[dict[str, str]],
    case: str,
) -> dict[str, np.ndarray]:
    """Return all-method normalized score profiles on one common [0, 1] scale.

    Scores are normalized within each system and spoke.  Lower-is-better
    metrics are inverted; coverage is scored by closeness to 0.90; paired-win
    score uses seed-wise nRMSE/CRPS wins; cognitive score is zero for methods
    without cognitive-parameter inference.
    """
    methods = base.METHODS

    def raw(field: str) -> np.ndarray:
        return np.asarray([float(summary[(case, method)][field]) for method in methods], dtype=float)

    def normalize_quality(quality: dict[str, float]) -> dict[str, float]:
        values = np.asarray([quality[m] for m in methods], dtype=float)
        finite = np.isfinite(values)
        if not np.any(finite):
            return {m: 0.5 for m in methods}
        vmin = float(np.min(values[finite]))
        vmax = float(np.max(values[finite]))
        if vmax - vmin <= 1.0e-14:
            return {m: 1.0 for m in methods}
        return {m: float(np.clip((quality[m] - vmin) / (vmax - vmin), 0.0, 1.0)) for m in methods}

    def paired_win_quality(method: str) -> float:
        rows = [
            r
            for r in run_rows
            if r["case"] == case and r["method"] in methods and r.get("valid", "True") == "True"
        ]
        by_key = {(r["method"], r["seed"]): r for r in rows}
        rates: list[float] = []
        opponents = [m for m in methods if m != method]
        for opponent in opponents:
            seeds = sorted({seed for m, seed in by_key if m == method} & {seed for m, seed in by_key if m == opponent})
            for metric in ("nrmse", "crps"):
                wins = [
                    float(by_key[(method, seed)][metric]) < float(by_key[(opponent, seed)][metric])
                    for seed in seeds
                    if by_key[(method, seed)].get(metric, "") and by_key[(opponent, seed)].get(metric, "")
                ]
                if wins:
                    rates.append(float(np.mean(wins)))
        return float(np.mean(rates)) if rates else 0.0

    nrmse_skill = _score_higher_better(raw("nrmse_mean"), "low")
    rmse_skill = _score_higher_better(raw("rmse_mean"), "low")
    crps_skill = _score_higher_better(raw("crps_mean"), "low")
    coverage_skill = _score_higher_better(raw("coverage_90_mean"), "target_0.90")
    width_skill = _score_higher_better(raw("interval_width_90_mean"), "low")
    alpha_skill = _score_higher_better(raw("alpha_absolute_error_mean"), "low")

    qualities_by_dim: dict[str, dict[str, float]] = {
        "Acc.": {method: float(0.5 * nrmse_skill[i] + 0.5 * rmse_skill[i]) for i, method in enumerate(methods)},
        "Dist.": {method: float(crps_skill[i]) for i, method in enumerate(methods)},
        "Cal.": {method: float(coverage_skill[i]) for i, method in enumerate(methods)},
        "C.Sh.": {
            method: float(np.sqrt(max(coverage_skill[i], 0.0) * max(width_skill[i], 0.0)))
            for i, method in enumerate(methods)
        },
        "Win": {method: paired_win_quality(method) for method in methods},
        "Cog.": {method: (float(alpha_skill[i]) if method in {"pce", "apce"} else 0.0) for i, method in enumerate(methods)},
    }

    score_by_dim = {dim: normalize_quality(quality) for dim, quality in qualities_by_dim.items()}
    return {
        method: np.asarray([score_by_dim[dim][method] for dim in RADAR_DIMENSIONS], dtype=float)
        for method in methods
    }


def draw_single_radar(
    ax: plt.Axes,
    scores: dict[str, np.ndarray],
    case_label: str,
) -> None:
    labels = list(RADAR_DIMENSIONS)
    n_metrics = len(labels)
    angles = np.linspace(0.0, 2.0 * np.pi, n_metrics, endpoint=False)
    angles_closed = np.r_[angles, angles[0]]
    step = 2.0 * np.pi / n_metrics

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0.0, 1.24)
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    ax.set_facecolor("#FFFFFF")
    ax.set_xticks(angles)
    ax.set_xticklabels([])
    ax.set_yticks([])

    ax.fill(angles_closed, np.full_like(angles_closed, 1.0), color="#DDE1E4", alpha=0.70, zorder=-3)

    for group_label, start, end, color in RADAR_GROUPS:
        theta_start = angles[start] - step * 0.48
        theta_end = angles[end] + step * 0.48
        theta_center = 0.5 * (theta_start + theta_end)
        width = theta_end - theta_start
        ax.bar(
            theta_center,
            0.125,
            width=width,
            bottom=1.035,
            color=color,
            edgecolor="none",
            alpha=0.92,
            align="center",
            zorder=-2,
            clip_on=False,
        )
        rotation = -np.degrees(theta_center)
        if rotation < -90:
            rotation += 180
        if rotation > 90:
            rotation -= 180
        ax.text(
            theta_center,
            1.085,
            group_label,
            fontsize=base.AXIS_TICK_FONT_SIZE - 0.7,
            rotation=rotation,
            rotation_mode="anchor",
            ha="center",
            va="center",
            color="#111111",
            clip_on=False,
        )

    for radius in (0.25, 0.50, 0.75, 1.0):
        ax.plot(
            angles_closed,
            np.full_like(angles_closed, radius),
            color="#B8C0C5" if radius < 1.0 else "#0F77A8",
            lw=0.55 if radius < 1.0 else 0.90,
            zorder=0,
        )
    for angle in angles:
        ax.plot([angle, angle], [0.0, 1.0], color="#9FA9AF", lw=0.55, zorder=0)

    for angle, label in zip(angles, labels, strict=True):
        deg = np.degrees(angle)
        ha = "center"
        if 8 < deg < 172:
            ha = "left"
        elif 188 < deg < 352:
            ha = "right"
        ax.text(
            angle,
            1.245,
            label,
            fontsize=base.AXIS_TICK_FONT_SIZE,
            ha=ha,
            va="center",
            color="#111111",
            clip_on=False,
        )

    for radius, label in zip((0.0, 0.5, 1.0), ("0", "0.5", "1"), strict=True):
        ax.text(
            np.deg2rad(0.0),
            radius,
            label,
            fontsize=base.AXIS_TICK_FONT_SIZE - 0.8,
            color="#111111",
            ha="center",
            va="bottom" if radius > 0 else "center",
            zorder=2,
        )

    polygon_area = {method: float(np.mean(scores[method])) for method in RADAR_PLOT_METHODS}
    fill_order = sorted(RADAR_PLOT_METHODS, key=lambda method: polygon_area[method], reverse=True)
    line_order = sorted(RADAR_PLOT_METHODS, key=lambda method: polygon_area[method])

    for method in fill_order:
        values = np.r_[scores[method], scores[method][0]]
        if method == "apce":
            ax.fill(angles_closed, values, color=base.TOP_APCE_COLOR, alpha=0.13, zorder=2)
        elif method == "pce":
            ax.fill(angles_closed, values, color=base.COLORS["pce"], alpha=0.095, zorder=2)
        else:
            ax.fill(angles_closed, values, color=RADAR_COLORS[method], alpha=0.080, zorder=2)

    for method in line_order:
        values = np.r_[scores[method], scores[method][0]]
        if method == "apce":
            ax.plot(angles_closed, values, color=base.APCE_FRAME, lw=1.65, alpha=0.98, zorder=9)
            ax.plot(angles_closed, values, color=base.TOP_APCE_COLOR, lw=0.95, alpha=1.0, zorder=10)
        elif method == "pce":
            ax.plot(angles_closed, values, color=base.COLORS["pce"], lw=1.05, alpha=0.92, zorder=8)
        else:
            ax.plot(angles_closed, values, color=RADAR_COLORS[method], lw=0.78, alpha=0.72, zorder=7)

    ax.text(
        0.5,
        -0.125,
        case_label,
        transform=ax.transAxes,
        fontsize=base.PANEL_TITLE_FONT_SIZE,
        ha="center",
        va="top",
        clip_on=False,
    )
def add_radar_method_legend(ax: plt.Axes) -> None:
    handles = []
    for method in RADAR_PLOT_METHODS:
        if method == "apce":
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    linestyle="None",
                    markerfacecolor=base.COLORS[method],
                    markeredgecolor=base.APCE_FRAME,
                    markeredgewidth=1.35,
                    markersize=7.2,
                    label=RADAR_METHOD_LABELS[method],
                )
            )
        elif method == "pce":
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    linestyle="None",
                    markerfacecolor="#FFFFFF",
                    markeredgecolor=base.COLORS["pce"],
                    markeredgewidth=1.25,
                    markersize=7.2,
                    label=RADAR_METHOD_LABELS[method],
                )
            )
        else:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    linestyle="None",
                    markerfacecolor="#FFFFFF",
                    markeredgecolor=RADAR_COLORS[method],
                    markeredgewidth=1.25,
                    markersize=6.8,
                    label=RADAR_METHOD_LABELS[method],
                )
            )
    ax.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.50, 0.42),
        ncol=len(handles),
        fontsize=base.MID_LEGEND_FONT_SIZE,
        handlelength=0.75,
        handletextpad=0.34,
        columnspacing=0.92,
        borderpad=0.02,
        labelspacing=0.03,
        frameon=False,
    )


def panel_radar_summary(
    fig: plt.Figure,
    subplot_spec,
    summary: dict[tuple[str, str], dict[str, str]],
    run_rows: list[dict[str, str]],
) -> None:
    panel_grid = GridSpecFromSubplotSpec(2, 1, subplot_spec=subplot_spec, height_ratios=[0.115, 1.00], hspace=-0.02)
    label_ax = fig.add_subplot(panel_grid[0, 0])
    label_ax.set_axis_off()
    label_ax.text(
        0.000,
        0.58,
        "i",
        transform=label_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=base.PANEL_LABEL_FONT_SIZE,
        fontweight="bold",
    )
    label_ax.text(
        0.022,
        0.58,
        "Normalized score profiles",
        transform=label_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=base.PANEL_TITLE_FONT_SIZE,
    )
    add_radar_method_legend(label_ax)

    radar_grid = GridSpecFromSubplotSpec(1, 3, subplot_spec=panel_grid[1, 0], wspace=-0.02)
    for index, case in enumerate(base.CASES):
        ax = fig.add_subplot(radar_grid[0, index], projection="polar")
        scores = compute_radar_scores(summary, run_rows, case)
        draw_single_radar(ax, scores, base.CASE_LABELS[case])


def write_radar_source_data(
    output_dir: Path,
    summary: dict[tuple[str, str], dict[str, str]],
    run_rows: list[dict[str, str]],
) -> None:
    rows: list[dict[str, str]] = []
    for case in base.CASES:
        scores = compute_radar_scores(summary, run_rows, case)
        for method in RADAR_PLOT_METHODS:
            for metric_label, score in zip(RADAR_DIMENSIONS, scores[method], strict=True):
                rows.append(
                    {
                        "case": case,
                        "method": method,
                        "metric": metric_label.replace("\n", " "),
                        "score_higher_is_better": f"{float(score):.10g}",
                        "score_definition": "within-system normalized score in [0, 1]; lower-is-better metrics are inverted, coverage is scored by closeness to 0.90, Win is paired-seed win quality, and Cog. is zero for methods without cognitive-parameter inference",
                    }
                )
    out = output_dir / f"{OUTPUT_STEM}_radar_scores.csv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "method", "metric", "score_higher_is_better", "score_definition"])
        writer.writeheader()
        writer.writerows(rows)


def _weights_on_common_alpha(
    alpha: np.ndarray,
    weights: np.ndarray,
    common_alpha: np.ndarray,
) -> np.ndarray:
    out = np.full(common_alpha.shape, np.nan, dtype=float)
    for i, a in enumerate(common_alpha):
        idx = int(np.argmin(np.abs(alpha - a)))
        if abs(float(alpha[idx]) - float(a)) < 1.0e-8:
            out[i] = float(weights[idx])
    return out


def _alpha_to_heatmap_x(alpha_value: float, common_alpha: np.ndarray) -> float:
    centers = np.arange(common_alpha.size, dtype=float) + 0.5
    if alpha_value <= common_alpha[0]:
        return float(centers[0])
    if alpha_value >= common_alpha[-1]:
        return float(centers[-1])
    return float(np.interp(alpha_value, common_alpha, centers))


def panel_weight_heatmap(
    ax: plt.Axes,
    wave: np.lib.npyio.NpzFile,
    spring: np.lib.npyio.NpzFile,
    heat: np.lib.npyio.NpzFile,
) -> None:
    base.add_panel(ax, "d", "Cognitive-weight map")

    wave_alpha = np.linspace(0.08, 0.92, len(wave["pce_final_weights"]))
    alpha = np.asarray(spring["alpha_grid"], dtype=float)
    common_alpha = alpha[alpha <= 0.5 + 1.0e-12]
    if common_alpha.size == 0:
        common_alpha = wave_alpha[wave_alpha <= 0.5 + 1.0e-12]

    series = [
        ("Wave", "PCE", wave_alpha, wave["pce_final_weights"]),
        ("Wave", "APCE", wave_alpha, wave["apce_final_weights"]),
        ("Spring", "PCE", spring["alpha_grid"], spring["pce_alpha_weight_history"][-1]),
        ("Spring", "APCE", spring["alpha_grid"], spring["apce_alpha_weight_history"][-1]),
        ("Heat", "PCE", heat["alpha_grid"], heat["pce_alpha_weight_history"][-1]),
        ("Heat", "APCE", heat["alpha_grid"], heat["apce_alpha_weight_history"][-1]),
    ]
    matrix = np.vstack(
        [
            _weights_on_common_alpha(np.asarray(a, dtype=float), np.asarray(w, dtype=float), common_alpha)
            for _, _, a, w in series
        ]
    )

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "apce_weight_map",
        ["#F4E6D2", "#EEC48E", "#DE8840", "#A84624", "#641A0D"],
    )
    cmap.set_bad("#F2F2F2")
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    x_edges = np.arange(common_alpha.size + 1)
    y_edges = np.arange(matrix.shape[0] + 1)
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        matrix,
        cmap=cmap,
        norm=norm,
        edgecolors="#FFFFFF",
        linewidth=0.9,
        antialiased=True,
    )

    ax.set_xlim(0, common_alpha.size)
    ax.set_ylim(matrix.shape[0], 0)
    ax.set_xticks(np.arange(common_alpha.size) + 0.5)
    ax.set_xticklabels([base.compact_tick(v) for v in common_alpha])
    ax.figure2_alpha_tick_labels = [base.compact_tick(v) for v in common_alpha]
    ax.set_yticks(np.arange(matrix.shape[0]) + 0.5)
    ax.set_yticklabels([f"{case} {method}" for case, method, _, _ in series])
    ax.set_xlabel(r"candidate $\alpha$")
    ax.set_ylabel("")
    ax.tick_params(axis="both", which="both", length=0, pad=2.0)

    for label, (_, method, _, _) in zip(ax.get_yticklabels(), series, strict=True):
        if method == "APCE":
            label.set_color("#C94B2C")

    for y in (2, 4):
        ax.axhline(y, color="#D6D6D6", lw=0.75, clip_on=False)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)

    cbar = ax.figure.colorbar(
        mesh,
        ax=ax,
        fraction=0.055,
        pad=0.025,
        ticks=[0, 1],
    )
    cbar.outline.set_visible(True)
    cbar.outline.set_edgecolor("#A8A8A8")
    cbar.outline.set_linewidth(0.55)
    cbar.ax.tick_params(labelsize=base.AXIS_TICK_FONT_SIZE, length=0, pad=1.5)
    cbar.ax.set_yticklabels(["0", "1"])
    cbar.set_label("weight", fontsize=base.AXIS_LABEL_FONT_SIZE, labelpad=1)

    base.polish_axis(ax)


def write_contract(output_dir: Path) -> None:
    text = """Core conclusion:
Across wave, spring and heat systems under one frozen paired protocol, APCE/PCE improve deterministic and probabilistic reconstruction metrics relative to valid training-free baselines.

Figure archetype:
Four-row mixed-modality evidence wall. Version 24 keeps the accepted Version 18 panels a-h and the Version 23 normalized-score radar, and aligns the panel-i method legend to the same horizontal centre as the e-h method legend.

Panel map:
a-c: representative dynamics.
d: final cognitive-weight map over candidate alpha values for PCE and APCE across Wave, Spring and Heat.
e: nRMSE seed-wise boxplots across Wave, Spring and Heat.
f: CRPS seed-wise boxplots across Wave, Spring and Heat.
g: 90% coverage mean bars across Wave, Spring and Heat.
h: Interval width mean bars across Wave, Spring and Heat.
i: normalized score profiles. Each radar shows DEnKF, LETKF, IEnSF, PCE and APCE. The six abbreviated spokes are Acc., Dist., Cal., C.Sh., Win and Cog.; they report within-system normalized scores in [0, 1]. Lower-is-better metrics are inverted, coverage is scored by closeness to 0.90, Win is paired-seed win quality, and Cog. is zero for methods without cognitive-parameter inference.

Statistics:
n=50 paired seeds per system and method. Panels e and f show seed-wise boxplots. Panels g and h show mean bars with 95% CI and overlaid seed-level dots. Panel i is a summary view, not an additional inferential test. APCE is highlighted by a red frame in statistical panels and by a framed curve and legend square in the normalized-score radar panel.
"""
    (output_dir / f"{OUTPUT_STEM}_contract.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--runs-csv", type=Path, required=True)
    parser.add_argument("--paired-csv", type=Path, required=True)
    parser.add_argument("--wave-npz", type=Path, required=True)
    parser.add_argument("--spring-npz", type=Path, required=True)
    parser.add_argument("--heat-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    base.set_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = read_csv(args.summary_csv)
    run_rows = read_csv(args.runs_csv)
    summary = {(r["case"], r["method"]): r for r in summary_rows}
    wave = np.load(args.wave_npz, allow_pickle=True)
    spring = np.load(args.spring_npz, allow_pickle=True)
    heat = np.load(args.heat_npz, allow_pickle=True)

    fig = plt.figure(figsize=(11.55, 12.35))
    outer = fig.add_gridspec(
        8,
        1,
        height_ratios=[1.03, 0.13, 0.11, 0.92, 0.28, 0.92, 0.20, 1.72],
        hspace=0.19,
        left=0.050,
        right=0.994,
        top=0.955,
        bottom=0.040,
    )

    top = GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[0, 0], wspace=0.29, width_ratios=[1, 1, 1, 0.92])
    base.panel_wave(fig.add_subplot(top[0, 0]), wave)
    base.panel_spring(fig.add_subplot(top[0, 1]), spring)
    base.panel_heat(fig.add_subplot(top[0, 2]), heat)
    ax_weight = fig.add_subplot(top[0, 3])
    panel_weight_heatmap(ax_weight, wave, spring, heat)

    base.add_mid_stat_legend(fig.add_subplot(outer[1, 0]))
    ax_pre_row2_spacer = fig.add_subplot(outer[2, 0])
    ax_pre_row2_spacer.set_axis_off()

    row2 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[3, 0], wspace=0.14)
    ax_spacer = fig.add_subplot(outer[4, 0])
    ax_spacer.set_axis_off()
    row3 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[5, 0], wspace=0.14)
    for row_index, row_metrics in enumerate((METRIC_GROUPS[:2], METRIC_GROUPS[2:])):
        row_spec = row2 if row_index == 0 else row3
        for metric_index, (metric, ylabel, scale, letter, title) in enumerate(row_metrics):
            metric_spec = GridSpecFromSubplotSpec(1, 3, subplot_spec=row_spec[0, metric_index], wspace=0.18)
            for case_index, case in enumerate(base.CASES):
                ax = fig.add_subplot(metric_spec[0, case_index])
                if metric in {"nrmse", "crps"}:
                    base.metric_case_box_panel(
                        ax,
                        run_rows,
                        case,
                        metric,
                        ylabel,
                        scale,
                        letter if case_index == 0 else "",
                        title,
                        show_ylabel=case_index == 0,
                    )
                else:
                    base.metric_case_panel(
                        ax,
                        summary,
                        run_rows,
                        case,
                        metric,
                        ylabel,
                        scale,
                        letter if case_index == 0 else "",
                        title,
                        show_ylabel=case_index == 0,
                    )

    ax_pre_row4_spacer = fig.add_subplot(outer[6, 0])
    ax_pre_row4_spacer.set_axis_off()
    panel_radar_summary(fig, outer[7, 0], summary, run_rows)

    base.freeze_all_compact_numeric_ticklabels(fig)
    alpha_tick_labels = getattr(ax_weight, "figure2_alpha_tick_labels", [])
    if alpha_tick_labels:
        ax_weight.set_xticks(np.arange(len(alpha_tick_labels)) + 0.5)
        ax_weight.set_xticklabels(alpha_tick_labels, fontsize=base.AXIS_TICK_FONT_SIZE)

    out_base = args.output_dir / OUTPUT_STEM
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    for source in (args.summary_csv, args.runs_csv, args.paired_csv):
        shutil.copy2(source, args.output_dir / f"source_data_{source.name}")
    write_radar_source_data(args.output_dir, summary, run_rows)
    write_contract(args.output_dir)
    qa = {
        "figure": OUTPUT_STEM,
        "layout": "version 24: preserves accepted v18 panels a-h and the v23 all-method normalized-score panel i; shifts the panel-i legend to the same horizontal centre as the e-h method legend; polygons are filled from larger to smaller areas following the reference style",
        "backend": "python/matplotlib",
        "font": "Arial with sans-serif fallback",
        "font_weight": "regular except panel letters",
        "formats": ["svg", "pdf", "png", "tiff"],
        "changed_from_v23": "aligns the panel-i legend horizontally with the e-h method legend only; keeps no true-alpha marker, no Oracle, continuous panel labels a-i, and the d-panel right-side vertical colorbar with ticks 0 and 1 only",
        "radar_methods": [RADAR_METHOD_LABELS[m] for m in RADAR_PLOT_METHODS],
        "radar_dimensions": [label.replace("\n", " ") for label in RADAR_DIMENSIONS],
        "radar_groups": [group[0] for group in RADAR_GROUPS],
        "score_definition": "within-system normalized score in [0, 1]; lower-is-better metrics are inverted, coverage is scored by closeness to 0.90, Win is paired-seed win quality, and Cog. is zero for methods without cognitive-parameter inference",
    }
    (args.output_dir / f"{OUTPUT_STEM}_qa.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(out_base)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
