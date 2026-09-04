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
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import numpy as np

import make_figure2_classical_systems_v12_fghi_xticklabels as base


FONT_PLUS = 1.0
base.BASE_FONT_SIZE += FONT_PLUS
base.AXIS_TICK_FONT_SIZE += FONT_PLUS
base.AXIS_LABEL_FONT_SIZE += FONT_PLUS
base.PANEL_TITLE_FONT_SIZE += FONT_PLUS
base.PANEL_LABEL_FONT_SIZE += FONT_PLUS
base.LOCAL_LEGEND_FONT_SIZE += FONT_PLUS
base.MID_LEGEND_FONT_SIZE += FONT_PLUS

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.size"] = base.BASE_FONT_SIZE


OUTPUT_STEM = "figure2_classical_uncertain_systems_v39_v2_formal"

METHODS = (
    "denkf",
    "letkf",
    "iensf",
    "aug_enkf",
    "bma_static",
    "pce",
    "apce",
)
ALPHA_METHODS = ("aug_enkf", "bma_static", "pce", "apce")
METHOD_LABELS = {
    "denkf": "DEnKF",
    "letkf": "LETKF",
    "iensf": "IEnSF",
    "aug_enkf": "Aug-EnKF",
    "bma_static": "BMA",
    "pce": "PCE",
    "apce": "APCE",
}
METHOD_COLORS = {
    "truth": "#1F1F1F",
    "denkf": "#5F83C5",
    "letkf": "#009E73",
    "iensf": "#8A6A4B",
    "aug_enkf": "#C44E52",
    "bma_static": "#56B4E9",
    "pce": "#7B4DA0",
    "apce": "#FF8A00",
}
METHOD_ALPHA = {
    "denkf": 0.26,
    "letkf": 0.28,
    "iensf": 0.30,
    "aug_enkf": 0.42,
    "bma_static": 0.52,
    "pce": 0.90,
    "apce": 1.00,
}
METHOD_LINE_WIDTH = {
    "truth": 1.72,
    "denkf": 0.82,
    "letkf": 0.84,
    "iensf": 0.84,
    "aug_enkf": 0.96,
    "bma_static": 1.00,
    "pce": 1.38,
    "apce": 1.82,
}

base.METHODS = METHODS
base.METHOD_LABELS = METHOD_LABELS
base.COLORS = METHOD_COLORS
base.TOP_COLORS = METHOD_COLORS
base.BASELINE_ALPHA = METHOD_ALPHA
base.LINE_WIDTH = METHOD_LINE_WIDTH

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
    r"$\alpha$ est.",
)

RADAR_GROUPS = (
    ("Error tests", 0, 1, "#D98DA7"),
    ("Uncertainty", 2, 3, "#E8B96B"),
    ("Reliability tests", 4, 5, "#72AAA6"),
)
RADAR_GROUP_PAD = 0.43

RADAR_PLOT_METHODS = base.METHODS
RADAR_METHOD_LABELS = METHOD_LABELS
RADAR_COLORS = METHOD_COLORS
NONCOGNITIVE_BASELINES = ("denkf", "letkf", "iensf")
REPRESENTATIVE_METHODS = ("letkf", "aug_enkf", "bma_static", "pce", "apce")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def representative_line_handles() -> list[Line2D]:
    handles = [Line2D([0], [0], color=METHOD_COLORS["truth"], lw=METHOD_LINE_WIDTH["truth"], label="Truth")]
    handles.extend(
        Line2D([0], [0], color=METHOD_COLORS[method], lw=METHOD_LINE_WIDTH[method], label=METHOD_LABELS[method])
        for method in REPRESENTATIVE_METHODS
    )
    return handles


def add_representative_legend(ax: plt.Axes, x_anchor: float = 0.55) -> None:
    ax.legend(
        handles=representative_line_handles(),
        loc="upper center",
        bbox_to_anchor=(x_anchor, 0.995),
        ncol=3,
        fontsize=base.LOCAL_LEGEND_FONT_SIZE,
        handlelength=0.90,
        handletextpad=0.30,
        columnspacing=0.55,
        borderpad=0.02,
        labelspacing=0.10,
    )


def _representative_phase_panel(
    ax: plt.Axes,
    data: np.lib.npyio.NpzFile,
    letter: str,
    title: str,
    u_idx: int,
    v_idx: int,
    legend_x: float = 0.56,
) -> None:
    base.add_panel(ax, letter, title)
    truth = data["truth_states"]
    ax.plot(truth[:, u_idx], truth[:, v_idx], color=METHOD_COLORS["truth"], lw=METHOD_LINE_WIDTH["truth"], zorder=5)
    for method in REPRESENTATIVE_METHODS:
        states = data[f"{method}_mean_states"]
        ax.plot(
            states[:, u_idx],
            states[:, v_idx],
            color=METHOD_COLORS[method],
            lw=METHOD_LINE_WIDTH[method],
            alpha=METHOD_ALPHA[method],
            zorder=7 if method == "apce" else 6,
        )
    focus = [truth] + [data[f"{method}_mean_states"] for method in REPRESENTATIVE_METHODS]
    xs = np.concatenate([arr[:, u_idx] for arr in focus])
    ys = np.concatenate([arr[:, v_idx] for arr in focus])
    xlo, xhi = np.percentile(xs, [0.5, 99.5])
    ylo, yhi = np.percentile(ys, [0.5, 99.5])
    ax.set_xlim(xlo - 0.08 * (xhi - xlo), xhi + 0.08 * (xhi - xlo))
    ax.set_ylim(ylo - 0.08 * (yhi - ylo), yhi + 0.34 * (yhi - ylo))
    ax.set_xlabel(r"$u(t)$")
    ax.set_ylabel(r"$v(t)$")
    base.compact_numeric_ticks(ax)
    add_representative_legend(ax, x_anchor=legend_x)
    base.polish_axis(ax)


def _seedband_curve_panel(
    ax: plt.Axes,
    data: np.lib.npyio.NpzFile,
    letter: str,
    title: str,
    x_key: str,
    y_label: str,
    x_label: str,
    legend_x: float = 0.56,
) -> None:
    base.add_panel(ax, letter, title)
    x = np.asarray(data[x_key], dtype=float)
    truth = np.asarray(data["truth_mean"], dtype=float)
    ax.plot(x, truth, color=METHOD_COLORS["truth"], lw=METHOD_LINE_WIDTH["truth"], zorder=6)
    for method in REPRESENTATIVE_METHODS:
        mean = np.asarray(data[f"{method}_mean"], dtype=float)
        low = np.asarray(data[f"{method}_low"], dtype=float)
        high = np.asarray(data[f"{method}_high"], dtype=float)
        ax.fill_between(
            x,
            low,
            high,
            color=METHOD_COLORS[method],
            alpha=0.12 if method != "apce" else 0.16,
            linewidth=0.0,
            zorder=2 if method != "apce" else 3,
        )
        ax.plot(
            x,
            mean,
            color=METHOD_COLORS[method],
            lw=METHOD_LINE_WIDTH[method] * (0.96 if method != "apce" else 1.02),
            alpha=0.96 if method in {"pce", "apce"} else 0.86,
            zorder=7 if method == "apce" else 5,
        )
    all_values = [truth] + [np.asarray(data[f"{method}_low"], dtype=float) for method in REPRESENTATIVE_METHODS] + [np.asarray(data[f"{method}_high"], dtype=float) for method in REPRESENTATIVE_METHODS]
    y_all = np.concatenate(all_values)
    ylo, yhi = np.percentile(y_all, [0.5, 99.5])
    yrng = max(float(yhi - ylo), 1.0e-12)
    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.set_ylim(float(ylo - 0.08 * yrng), float(yhi + 0.18 * yrng))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    base.compact_numeric_ticks(ax)
    add_representative_legend(ax, x_anchor=legend_x)
    base.polish_axis(ax)


def _band_alpha(method: str) -> float:
    if method == "apce":
        return 0.145
    if method == "pce":
        return 0.105
    if method == "bma_static":
        return 0.055
    return 0.040


def _smooth_series(y: np.ndarray, window: int = 11) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.size < window:
        return y
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(y, pad_width=pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _overlay_phase_seedband(
    ax: plt.Axes,
    band: np.lib.npyio.NpzFile | None,
    x_key: str = "times",
) -> None:
    if band is None or "truth_mean" not in band.files:
        return
    t = np.asarray(band[x_key], dtype=float)
    if t.size < 3:
        return
    for method in ("pce", "apce"):
        if f"{method}_low" not in band.files or f"{method}_high" not in band.files:
            continue
        u_low = _smooth_series(np.asarray(band[f"{method}_low"], dtype=float), 13)
        u_high = _smooth_series(np.asarray(band[f"{method}_high"], dtype=float), 13)
        v_low = _smooth_series(np.gradient(u_low, t), 13)
        v_high = _smooth_series(np.gradient(u_high, t), 13)
        xs = np.r_[u_low, u_high[::-1]]
        ys = np.r_[v_low, v_high[::-1]]
        ax.fill(
            xs,
            ys,
            facecolor=METHOD_COLORS[method],
            edgecolor="none",
            alpha=_band_alpha(method),
            zorder=1,
        )


def _overlay_heat_seedband(
    ax: plt.Axes,
    band: np.lib.npyio.NpzFile | None,
) -> None:
    if band is None or "truth_mean" not in band.files:
        return
    x = np.asarray(band["space"], dtype=float)
    for target_ax in [ax] + list(getattr(ax, "child_axes", [])):
        for method in REPRESENTATIVE_METHODS:
            if f"{method}_low" not in band.files or f"{method}_high" not in band.files:
                continue
            low = np.asarray(band[f"{method}_low"], dtype=float)
            high = np.asarray(band[f"{method}_high"], dtype=float)
            target_ax.fill_between(
                x,
                low,
                high,
                color=METHOD_COLORS[method],
                alpha=_band_alpha(method),
                linewidth=0.0,
                zorder=1,
            )


def panel_wave(ax: plt.Axes, data: np.lib.npyio.NpzFile) -> None:
    if "truth_states" not in data.files:
        _seedband_curve_panel(ax, data, "a", r"Wave trajectory at $x_c$", "times", r"$u(x_c, t)$", r"$t$", legend_x=0.56)
        return
    nx = data["truth_states"].shape[1] // 2
    node = nx // 2
    _representative_phase_panel(ax, data, "a", r"Wave phase at $x_c$", node, nx + node)


def panel_spring(ax: plt.Axes, data: np.lib.npyio.NpzFile) -> None:
    if "truth_states" not in data.files:
        _seedband_curve_panel(ax, data, "b", "Spring trajectory", "times", r"$u(t)$", r"$t$", legend_x=0.48)
        return
    _representative_phase_panel(ax, data, "b", "Spring phase", 0, 1, legend_x=0.48)


def panel_heat(ax: plt.Axes, data: np.lib.npyio.NpzFile) -> None:
    if "truth_mean" in data.files:
        base.add_panel(ax, "c", "Heat terminal profile")
        x = np.asarray(data["space"], dtype=float)
        truth = np.asarray(data["truth_mean"], dtype=float)
        ax.plot(x, truth, color=METHOD_COLORS["truth"], lw=METHOD_LINE_WIDTH["truth"], zorder=6)
        for method in REPRESENTATIVE_METHODS:
            mean = np.asarray(data[f"{method}_mean"], dtype=float)
            low = np.asarray(data[f"{method}_low"], dtype=float)
            high = np.asarray(data[f"{method}_high"], dtype=float)
            ax.fill_between(
                x,
                low,
                high,
                color=METHOD_COLORS[method],
                alpha=0.12 if method != "apce" else 0.16,
                linewidth=0.0,
                zorder=2 if method != "apce" else 3,
            )
            ax.plot(
                x,
                mean,
                color=METHOD_COLORS[method],
                lw=METHOD_LINE_WIDTH[method] * (0.96 if method != "apce" else 1.02),
                alpha=0.96 if method in {"pce", "apce"} else 0.86,
                zorder=7 if method == "apce" else 5,
            )
        obs = None
        y_all = np.concatenate([truth] + [np.asarray(data[f"{m}_low"], dtype=float) for m in REPRESENTATIVE_METHODS] + [np.asarray(data[f"{m}_high"], dtype=float) for m in REPRESENTATIVE_METHODS])
        ylo, yhi = np.percentile(y_all, [0.5, 99.5])
        yrng = max(float(yhi - ylo), 1.0e-12)
        ax.set_ylim(float(ylo - 0.08 * yrng), float(yhi + 0.18 * yrng))
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$u(x,t_f)$")
        ax.set_xlim(float(x[0]), float(x[-1]))
        base.compact_numeric_ticks(ax)
        add_representative_legend(ax, x_anchor=0.48)
        base.polish_axis(ax)
        return
    base.add_panel(ax, "c", "Heat terminal profile")
    x = data["space"]
    truth = data["truth_states"][-1]
    ax.plot(x, truth, color=METHOD_COLORS["truth"], lw=METHOD_LINE_WIDTH["truth"], zorder=5)
    for method in REPRESENTATIVE_METHODS:
        y = data[f"{method}_mean_states"][-1]
        ax.plot(
            x,
            y,
            color=METHOD_COLORS[method],
            lw=METHOD_LINE_WIDTH[method],
            alpha=METHOD_ALPHA[method],
            zorder=7 if method == "apce" else 6,
        )
    obs = data["observation_indices"]
    ax.scatter(x[obs], truth[obs], s=9.0, facecolor="white", edgecolor=METHOD_COLORS["truth"], linewidth=0.55, zorder=8)
    all_profiles = np.concatenate([truth] + [data[f"{m}_mean_states"][-1] for m in REPRESENTATIVE_METHODS])
    ylo = float(np.nanmin(all_profiles))
    yhi = float(np.nanmax(all_profiles))
    yrng = max(yhi - ylo, 1.0e-12)
    ax.set_ylim(ylo - 0.08 * yrng, yhi + 0.34 * yrng)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u(x,t_f)$")
    inset = ax.inset_axes([0.29, 0.065, 0.42, 0.29])
    lo = int(np.searchsorted(x, 0.31))
    hi = int(np.searchsorted(x, 0.53))
    inset.plot(x[lo:hi], truth[lo:hi], color=METHOD_COLORS["truth"], lw=METHOD_LINE_WIDTH["truth"] * 0.78, zorder=5)
    for method in REPRESENTATIVE_METHODS:
        y = data[f"{method}_mean_states"][-1]
        inset.plot(
            x[lo:hi],
            y[lo:hi],
            color=METHOD_COLORS[method],
            lw=METHOD_LINE_WIDTH[method] * 0.78,
            alpha=METHOD_ALPHA[method],
            zorder=7 if method == "apce" else 6,
        )
    y_slice = np.concatenate([truth[lo:hi]] + [data[f"{m}_mean_states"][-1][lo:hi] for m in REPRESENTATIVE_METHODS])
    y_pad = max(0.05 * (float(np.max(y_slice)) - float(np.min(y_slice))), 1.0e-12)
    inset.set_xlim(float(x[lo]), float(x[hi - 1]))
    inset.set_ylim(float(np.min(y_slice)) - y_pad, float(np.max(y_slice)) + y_pad)
    inset.set_xticks([])
    inset.set_yticks([])
    for side in ("left", "bottom", "right", "top"):
        inset.spines[side].set_visible(True)
        inset.spines[side].set_color("#B8B8B8")
        inset.spines[side].set_linewidth(0.55)
    mark_inset(ax, inset, loc1=1, loc2=2, fc="none", ec="#B8B8B8", linewidth=0.60)
    base.compact_numeric_ticks(ax)
    add_representative_legend(ax, x_anchor=0.48)
    base.polish_axis(ax)


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


def safe_float(value: object) -> float:
    try:
        output = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return np.nan
    return output if np.isfinite(output) else np.nan


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
        return np.asarray([safe_float(summary.get((case, method), {}).get(field, "")) for method in methods], dtype=float)

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
        r"$\alpha$ est.": {
            method: (
                float(alpha_skill[i])
                if method in ALPHA_METHODS and np.isfinite(raw("alpha_absolute_error_mean")[i])
                else 0.0
            )
            for i, method in enumerate(methods)
        },
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
    ax.set_ylim(0.0, 1.39)
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    ax.set_facecolor("#FFFFFF")
    ax.set_xticks(angles)
    ax.set_xticklabels([])
    ax.set_yticks([])

    ax.fill(angles_closed, np.full_like(angles_closed, 1.0), color="#DDE1E4", alpha=0.70, zorder=-3)

    for group_label, start, end, color in RADAR_GROUPS:
        theta_start = angles[start] - step * RADAR_GROUP_PAD
        theta_end = angles[end] + step * RADAR_GROUP_PAD
        theta_center = 0.5 * (theta_start + theta_end)
        width = theta_end - theta_start
        ax.bar(
            theta_center,
            0.190,
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
        if group_label == "Reliability tests":
            rotation += 180
        ax.text(
            theta_center,
            1.105,
            group_label,
            fontsize=base.AXIS_TICK_FONT_SIZE - 0.7,
            rotation=rotation,
            rotation_mode="anchor",
            ha="center",
            va="center",
            color="#111111",
            clip_on=False,
        )

    for radius in (0.20, 0.40, 0.60, 0.80, 1.0):
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
            1.315,
            label,
            fontsize=base.AXIS_TICK_FONT_SIZE,
            ha=ha,
            va="center",
            color="#111111",
            clip_on=False,
        )

    for radius, label in zip((0.0, 0.2, 0.4, 0.6, 0.8, 1.0), ("0", "0.2", "0.4", "0.6", "0.8", "1"), strict=True):
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
            lw = 1.02 if method == "letkf" else 0.90
            alpha = 0.95 if method == "letkf" else 0.82
            zorder = 8 if method == "letkf" else 7
            ax.plot(angles_closed, values, color=RADAR_COLORS[method], lw=lw, alpha=alpha, zorder=zorder)

    ax.text(
        0.5,
        -0.010,
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
        bbox_to_anchor=(0.50, 0.50),
        ncol=len(handles),
        fontsize=base.MID_LEGEND_FONT_SIZE,
        handlelength=1.05,
        handletextpad=0.40,
        columnspacing=1.25,
        borderpad=0.03,
        labelspacing=0.05,
        frameon=False,
    )


def panel_radar_summary(
    fig: plt.Figure,
    subplot_spec,
    summary: dict[tuple[str, str], dict[str, str]],
    run_rows: list[dict[str, str]],
) -> None:
    panel_grid = GridSpecFromSubplotSpec(2, 1, subplot_spec=subplot_spec, height_ratios=[0.082, 1.00], hspace=0.000)
    label_ax = fig.add_subplot(panel_grid[0, 0])
    label_ax.set_axis_off()
    label_ax.text(
        0.000,
        0.50,
        "i",
        transform=label_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=base.PANEL_LABEL_FONT_SIZE,
        fontweight="bold",
    )
    label_ax.text(
        0.022,
        0.50,
        "Normalized score profiles",
        transform=label_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=base.PANEL_TITLE_FONT_SIZE,
    )
    add_radar_method_legend(label_ax)

    radar_grid = GridSpecFromSubplotSpec(1, 3, subplot_spec=panel_grid[1, 0], wspace=0.02)
    for index, case in enumerate(base.CASES):
        ax = fig.add_subplot(radar_grid[0, index], projection="polar")
        if index in (0, 2):
            bbox = ax.get_position()
            dx = 0.014 if index == 2 else -0.014
            ax.set_position([bbox.x0 + dx, bbox.y0, bbox.width, bbox.height])
        bbox = ax.get_position()
        ax.set_position([bbox.x0 - 0.008, bbox.y0, bbox.width, bbox.height])
        scores = compute_radar_scores(summary, run_rows, case)
        draw_single_radar(ax, scores, base.CASE_LABELS[case])


def panel_calibration_sharpness(
    fig: plt.Figure,
    subplot_spec,
    run_rows: list[dict[str, str]],
) -> None:
    """Calibration-sharpness comparison for Aug-EnKF, BMA, PCE and APCE."""

    g_methods = (
        "aug_enkf",
        "bma_static",
        "pce",
        "apce",
    )

    # Three systems. Each system has:
    #   top    -> independent legend + case name
    #   bottom -> calibration-sharpness data
    grid = GridSpecFromSubplotSpec(
        2,
        3,
        subplot_spec=subplot_spec,
        height_ratios=[0.25, 1.00],
        hspace=0.025,
        wspace=0.20,
    )

    for case_index, case in enumerate(base.CASES):

        head_ax = fig.add_subplot(grid[0, case_index])
        ax = fig.add_subplot(grid[1, case_index])

        head_ax.set_axis_off()

        rows_case = [
            row
            for row in run_rows
            if row.get("case") == case
        ]

        stats = []

        for method in g_methods:

            vals = [
                row
                for row in rows_case
                if (
                    row.get("method") == method
                    and row.get("valid", "true").lower() != "false"
                )
            ]

            if not vals:
                continue

            coverage = np.asarray(
                [
                    safe_float(row.get("coverage_90"))
                    for row in vals
                ],
                dtype=float,
            )

            width = (
                np.asarray(
                    [
                        safe_float(row.get("interval_width_90"))
                        for row in vals
                    ],
                    dtype=float,
                )
                * 100.0
            )

            keep = (
                np.isfinite(coverage)
                & np.isfinite(width)
                & (width > 0.0)
            )

            coverage = coverage[keep]
            width = width[keep]

            if coverage.size == 0:
                continue

            x = float(np.mean(width))
            y = float(np.mean(coverage))

            if width.size > 1:
                xerr = (
                    1.96
                    * float(
                        np.std(width, ddof=1)
                        / np.sqrt(width.size)
                    )
                )
            else:
                xerr = 0.0

            if coverage.size > 1:
                yerr = (
                    1.96
                    * float(
                        np.std(coverage, ddof=1)
                        / np.sqrt(coverage.size)
                    )
                )
            else:
                yerr = 0.0

            stats.append(
                {
                    "method": method,
                    "x": x,
                    "y": y,
                    "xerr": xerr,
                    "yerr": yerr,
                }
            )

        # ------------------------------------------------------------
        # Panel-g title
        # ------------------------------------------------------------
        if case_index == 0:
            head_ax.text(
                -0.08,
                1.23,
                "g",
                transform=head_ax.transAxes,
                ha="left",
                va="top",
                fontsize=base.PANEL_LABEL_FONT_SIZE,
                fontweight="bold",
                clip_on=False,
            )

            head_ax.text(
                0.00,
                1.23,
                "Calibration-sharpness frontier",
                transform=head_ax.transAxes,
                ha="left",
                va="top",
                fontsize=base.PANEL_TITLE_FONT_SIZE,
                clip_on=False,
            )

        # ------------------------------------------------------------
        # Same-size legend markers for ALL four methods
        # ------------------------------------------------------------
        legend_handles = []

        for method in g_methods:

            is_apce = method == "apce"

            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="None",
                    markerfacecolor=METHOD_COLORS[method],
                    markeredgecolor=(
                        "#E83E8C"
                        if is_apce
                        else "white"
                    ),
                    markeredgewidth=(
                        1.35
                        if is_apce
                        else 0.75
                    ),
                    markersize=7.0,
                    label=METHOD_LABELS[method],
                )
            )

        # Independent legend at the TOP of every small panel
        head_ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.50, 0.88),
            ncol=4,
            fontsize=base.LOCAL_LEGEND_FONT_SIZE - 0.85,
            handlelength=0.52,
            handletextpad=0.20,
            columnspacing=0.52,
            borderpad=0.0,
            labelspacing=0.0,
            frameon=False,
        )

        # Wave / Spring / Heat below legend
        head_ax.text(
            0.50,
            0.06,
            base.CASE_LABELS[case],
            transform=head_ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=base.PANEL_TITLE_FONT_SIZE - 0.20,
            color="#111111",
        )

        # ------------------------------------------------------------
        # Darker 90% calibration band
        # ------------------------------------------------------------
        ax.axhspan(
            0.875,
            0.925,
            color="#C9D6E2",
            alpha=0.95,
            zorder=0,
        )

        ax.axhline(
            0.90,
            color="#666666",
            linestyle=(0, (2.0, 2.0)),
            linewidth=0.90,
            zorder=1,
        )

        # ------------------------------------------------------------
        # Four methods -- exactly the SAME marker size
        # ------------------------------------------------------------
        for item in stats:

            method = item["method"]
            is_apce = method == "apce"

            ax.errorbar(
                item["x"],
                item["y"],
                xerr=item["xerr"],
                yerr=item["yerr"],
                fmt="o",

                # All markers exactly same size
                ms=7.2,

                color=METHOD_COLORS[method],
                markerfacecolor=METHOD_COLORS[method],

                # APCE distinction comes only from the frame
                markeredgecolor=(
                    "#E83E8C"
                    if is_apce
                    else "white"
                ),

                markeredgewidth=(
                    1.40
                    if is_apce
                    else 0.78
                ),

                ecolor=METHOD_COLORS[method],
                elinewidth=0.88,
                capsize=2.1,
                capthick=0.80,
                alpha=0.97,
                zorder=7 if is_apce else 6,
            )

        # ------------------------------------------------------------
        # Tight x limits based only on the four retained methods
        # ------------------------------------------------------------
        if stats:

            xs = np.asarray(
                [item["x"] for item in stats],
                dtype=float,
            )

            xe = np.asarray(
                [item["xerr"] for item in stats],
                dtype=float,
            )

            x_low = float(np.min(xs - xe))
            x_high = float(np.max(xs + xe))

            x_span = max(
                x_high - x_low,
                0.05,
            )

            ax.set_xlim(
                max(
                    0.0,
                    x_low - 0.20 * x_span,
                ),
                x_high + 0.20 * x_span,
            )

        # ------------------------------------------------------------
        # IMPORTANT:
        # Same y range for Wave / Spring / Heat.
        # Every panel keeps its OWN visible y-axis tick labels.
        # ------------------------------------------------------------
        ax.set_ylim(
            0.865,
            1.005,
        )

        ax.set_yticks(
            [
                0.88,
                0.90,
                0.92,
                0.94,
                0.96,
                0.98,
                1.00,
            ]
        )

        ax.set_yticklabels(
            [
                "0.88",
                "0.90",
                "0.92",
                "0.94",
                "0.96",
                "0.98",
                "1",
            ],
            fontsize=base.AXIS_TICK_FONT_SIZE,
        )

        # Every small graph keeps its y-axis ticks and labels
        ax.tick_params(
            axis="y",
            labelleft=True,
            left=True,
            pad=1.5,
        )

        ax.set_xlabel(
            r"interval width ($10^{-2}$)",
            labelpad=1.0,
        )

        # Only left-most panel repeats the axis-name text;
        # all three still retain numerical y axes.
        if case_index == 0:
            ax.set_ylabel(
                "90% coverage",
                labelpad=1.5,
            )
        else:
            ax.set_ylabel("")

        ax.locator_params(
            axis="x",
            nbins=4,
        )

        ax.tick_params(
            axis="x",
            pad=1.4,
        )

        # Subtle horizontal guides
        ax.grid(
            axis="y",
            color="#E0E4E7",
            linewidth=0.48,
        )

        ax.grid(
            False,
            axis="x",
        )

        ax.set_axisbelow(True)

        base.polish_axis(ax)


def write_radar_source_data(
    output_dir: Path,
    summary: dict[tuple[str, str], dict[str, str]],
    run_rows: list[dict[str, str]],
    output_stem: str,
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
    out = output_dir / f"{output_stem}_radar_scores.csv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "method", "metric", "score_higher_is_better", "score_definition"])
        writer.writeheader()
        writer.writerows(rows)


def _alpha_values(run_rows: list[dict[str, str]], case: str, method: str, field: str = "alpha_estimate") -> np.ndarray:
    values = np.asarray(
        [
            safe_float(row.get(field, ""))
            for row in run_rows
            if row.get("case") == case and row.get("method") == method and row.get("valid", "True") == "True"
        ],
        dtype=float,
    )
    return values[np.isfinite(values)]


def _case_true_alpha(run_rows: list[dict[str, str]], case: str) -> float:
    values = np.asarray(
        [safe_float(row.get("alpha_true", "")) for row in run_rows if row.get("case") == case],
        dtype=float,
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.12
    return float(np.median(values))


def _mean_ci(values: np.ndarray, seed: int, resamples: int = 4000) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(values))
    if values.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, values.size, size=(resamples, values.size))].mean(axis=1)
    return mean, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def panel_continuous_alpha(ax: plt.Axes, run_rows: list[dict[str, str]]) -> None:
    base.add_panel(ax, "d", r"Continuous $\alpha$ estimates")
    case_x = np.arange(len(base.CASES), dtype=float) + 1.0
    offsets = np.linspace(-0.27, 0.27, len(ALPHA_METHODS))
    rng = np.random.default_rng(2026081039)
    all_values: list[float] = []
    for case_index, case in enumerate(base.CASES):
        true_alpha = _case_true_alpha(run_rows, case)
        ax.hlines(
            true_alpha,
            case_x[case_index] - 0.42,
            case_x[case_index] + 0.42,
            color="#505050",
            lw=0.78,
            ls=(0, (3, 2)),
            zorder=1,
        )
        all_values.append(true_alpha)
        for method_index, method in enumerate(ALPHA_METHODS):
            xpos = case_x[case_index] + offsets[method_index]
            values = _alpha_values(run_rows, case, method)
            if values.size == 0:
                continue
            all_values.extend(values.tolist())
            bp = ax.boxplot(
                [values],
                positions=[xpos],
                widths=0.115,
                patch_artist=True,
                showfliers=False,
                whis=1.5,
                manage_ticks=False,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(METHOD_COLORS[method])
                patch.set_alpha(0.74 if method != "apce" else 0.88)
                patch.set_edgecolor(base.APCE_FRAME if method == "apce" else "#303030")
                patch.set_linewidth(1.05 if method == "apce" else 0.64)
            for key in ("whiskers", "caps", "medians"):
                for artist in bp[key]:
                    artist.set_color("#303030")
                    artist.set_linewidth(0.64)
            jitter = rng.normal(0.0, 0.018, values.size)
            ax.scatter(
                np.full(values.size, xpos) + jitter,
                values,
                s=4.6,
                color=METHOD_COLORS[method],
                alpha=0.34 if method != "apce" else 0.52,
                edgecolors="none",
                zorder=2,
            )
            mean, low, high = _mean_ci(values, seed=1000 + case_index * 37 + method_index)
            ax.errorbar(
                [xpos],
                [mean],
                yerr=[[mean - low], [high - mean]],
                fmt="o",
                ms=3.4,
                mfc="#FFFFFF",
                mec=base.APCE_FRAME if method == "apce" else "#202020",
                mew=0.70,
                ecolor="#202020",
                elinewidth=0.68,
                capsize=1.7,
                capthick=0.68,
                zorder=4,
            )
    finite = np.asarray(all_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size:
        ymin, ymax = float(np.min(finite)), float(np.max(finite))
        pad = max(0.18 * (ymax - ymin), 0.018)
        ax.set_ylim(max(0.0, ymin - pad), min(1.0, ymax + 2.25 * pad))
    ax.set_xlim(0.42, len(base.CASES) + 0.58)
    ax.set_xticks(case_x)
    ax.set_ylabel(r"$\hat{\alpha}$", fontsize=base.AXIS_LABEL_FONT_SIZE)
    base.compact_numeric_yticks(ax)
    ax.set_xticks(case_x)
    ax.set_xticklabels([base.CASE_LABELS[c] for c in base.CASES], fontsize=base.AXIS_TICK_FONT_SIZE)
    base.polish_axis(ax)
    handles = [
        Line2D([0], [0], color="#505050", lw=0.78, ls=(0, (3, 2)), label=r"true $\alpha$"),
        *[
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="None",
                markerfacecolor=METHOD_COLORS[method],
                markeredgecolor=base.APCE_FRAME if method == "apce" else "#303030",
                markeredgewidth=1.05 if method == "apce" else 0.64,
                markersize=5.8,
                label=METHOD_LABELS[method],
            )
            for method in ALPHA_METHODS
        ],
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.99),
        ncol=2,
        fontsize=base.LOCAL_LEGEND_FONT_SIZE,
        handlelength=0.95,
        handletextpad=0.34,
        columnspacing=0.56,
        borderpad=0.02,
        labelspacing=0.10,
    )


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


def write_contract(output_dir: Path, output_stem: str) -> None:
    text = """Core conclusion:
Across wave, spring and heat systems under one frozen paired protocol, APCE/PCE improve deterministic and probabilistic reconstruction metrics relative to valid training-free baselines.

Figure archetype:
Four-row mixed-modality evidence wall. The top row shows representative phase portraits, a heat-profile reconstruction and continuous cognitive-coordinate estimates; the two compact middle rows provide seed-wise quantitative evidence; the bottom row summarizes the multi-metric trade-off.

Panel map:
a-b: representative phase portraits for Truth, LETKF, Aug-EnKF, BMA, PCE and APCE.
c: heat terminal-profile reconstruction with cross-seed bands where available.
d: continuous alpha estimates for Aug-EnKF, BMA, PCE and APCE across Wave, Spring and Heat.
e: nRMSE seed-wise boxplots across Wave, Spring and Heat.
f: CRPS seed-wise boxplots across Wave, Spring and Heat.
g: 90% coverage mean bars across Wave, Spring and Heat.
h: Interval width mean bars across Wave, Spring and Heat.
i: normalized score profiles. Each radar shows DEnKF, LETKF, IEnSF, Aug-EnKF, BMA, PCE and APCE. The six abbreviated spokes are Acc., Dist., Cal., C.Sh., Win and alpha est.; they report within-system normalized scores in [0, 1]. Lower-is-better metrics are inverted, coverage is scored by closeness to 0.90, Win is paired-seed win quality, and alpha est. scores only methods that infer the cognitive coordinate.

Statistics:
n=50 paired seeds per system and method. Panels e and f show seed-wise boxplots. Panels g and h show mean bars with 95% CI and overlaid seed-level dots. Panel i is a summary view, not an additional inferential test. APCE is highlighted by a red frame in statistical panels and by a framed curve and legend square in the normalized-score radar panel.
"""
    (output_dir / f"{output_stem}_contract.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--runs-csv", type=Path, required=True)
    parser.add_argument("--paired-csv", type=Path, required=True)
    parser.add_argument("--wave-npz", type=Path, required=True)
    parser.add_argument("--spring-npz", type=Path, required=True)
    parser.add_argument("--heat-npz", type=Path, required=True)
    parser.add_argument("--wave-band-npz", type=Path)
    parser.add_argument("--spring-band-npz", type=Path)
    parser.add_argument("--heat-band-npz", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-stem", type=str, default=OUTPUT_STEM)
    args = parser.parse_args()

    base.set_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = read_csv(args.summary_csv)
    run_rows = read_csv(args.runs_csv)
    summary = {(r["case"], r["method"]): r for r in summary_rows}
    wave = np.load(args.wave_npz, allow_pickle=True)
    spring = np.load(args.spring_npz, allow_pickle=True)
    heat = np.load(args.heat_npz, allow_pickle=True)
    wave_band = np.load(args.wave_band_npz, allow_pickle=True) if args.wave_band_npz is not None else None
    spring_band = np.load(args.spring_band_npz, allow_pickle=True) if args.spring_band_npz is not None else None
    heat_band = np.load(args.heat_band_npz, allow_pickle=True) if args.heat_band_npz is not None else None
    # Keep Wave/Spring top-row panels as representative phase portraits.
    # The optional seed-band files are intentionally not allowed to override
    # these two panels; seed-wise uncertainty is carried by panels e-i.

    fig = plt.figure(figsize=(11.55, 12.35))
    outer = fig.add_gridspec(
        8,
        1,
        height_ratios=[1.05, 0.15, 0.12, 0.94, 0.31, 0.94, 0.11, 1.90],
        hspace=0.205,
        left=0.050,
        right=0.994,
        top=0.955,
        bottom=0.040,
    )

    top = GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[0, 0], wspace=0.29, width_ratios=[1, 1, 1, 0.92])
    ax_wave = fig.add_subplot(top[0, 0])
    panel_wave(ax_wave, wave)
    _overlay_phase_seedband(ax_wave, wave_band)
    ax_spring = fig.add_subplot(top[0, 1])
    panel_spring(ax_spring, spring)
    _overlay_phase_seedband(ax_spring, spring_band)
    ax_heat = fig.add_subplot(top[0, 2])
    panel_heat(ax_heat, heat)
    _overlay_heat_seedband(ax_heat, heat_band)
    ax_weight = fig.add_subplot(top[0, 3])
    panel_continuous_alpha(ax_weight, run_rows)

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

    ax_pre_radar_spacer = fig.add_subplot(outer[6, 0])
    ax_pre_radar_spacer.set_axis_off()
    panel_radar_summary(fig, outer[7, 0], summary, run_rows)

    base.freeze_all_compact_numeric_ticklabels(fig)
    out_base = args.output_dir / args.output_stem
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    for source in (args.summary_csv, args.runs_csv, args.paired_csv):
        shutil.copy2(source, args.output_dir / f"{args.output_stem}_source_{source.name}")
    write_radar_source_data(args.output_dir, summary, run_rows, args.output_stem)
    write_contract(args.output_dir, args.output_stem)
    qa = {
        "figure": args.output_stem,
        "layout": "e-h quantitative rows are compressed by 8.5% while the top representative-dynamics/continuous-alpha row and the bottom radar row retain their physical dimensions",
        "backend": "python/matplotlib",
        "font": "Arial with sans-serif fallback",
        "font_weight": "regular except panel letters",
        "formats": ["svg", "pdf", "png", "tiff"],
        "changed_from_v23": "aligns the panel-i legend horizontally with the e-h method legend; restores Wave/Spring phase portraits; keeps no true-alpha marker, no Oracle, continuous panel labels a-i, and the d-panel right-side vertical colorbar with ticks 0 and 1 only",
        "radar_methods": [RADAR_METHOD_LABELS[m] for m in RADAR_PLOT_METHODS],
        "radar_dimensions": [label.replace("\n", " ") for label in RADAR_DIMENSIONS],
        "radar_groups": [group[0] for group in RADAR_GROUPS],
        "score_definition": "within-system normalized score in [0, 1]; lower-is-better metrics are inverted, coverage is scored by closeness to 0.90, Win is paired-seed win quality, and alpha est. is zero for methods without cognitive-parameter inference",
    }
    (args.output_dir / f"{args.output_stem}_qa.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(out_base)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
