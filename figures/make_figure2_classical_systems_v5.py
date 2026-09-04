from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory
import numpy as np


CASES = ("wave", "spring", "heat")
CASE_LABELS = {"wave": "Wave", "spring": "Spring", "heat": "Heat"}
METHODS = ("denkf", "letkf", "iensf", "pce", "apce")
REP_METHODS = ("denkf", "letkf", "iensf", "pce", "apce")
METHOD_LABELS = {
    "denkf": "DEnKF",
    "letkf": "LETKF",
    "iensf": "IEnSF",
    "pce": "PCE",
    "apce": "APCE",
}
COLORS = {
    "truth": "#202020",
    "denkf": "#747474",
    "letkf": "#B3B3B3",
    "iensf": "#7E84B8",
    "pce": "#3F6EA8",
    "apce": "#D99035",
}
REP_COLORS = {
    "truth": "#1F1F1F",
    "denkf": "#B8B8B8",
    "letkf": "#D7D7D7",
    "iensf": "#AEB6D8",
    "pce": "#356FA8",
    "apce": "#D8842F",
}
CASE_COLORS = {"wave": "#3F6EA8", "spring": "#8A6BAE", "heat": "#4F8A58"}
BASELINE_COLORS = {"denkf": "#7A7A7A", "letkf": "#A9A9A9", "iensf": "#7E84B8"}
PAIR_BASELINES = ("denkf", "letkf", "iensf", "pce")
PAIR_COLORS_NRMSE = {
    "denkf": "#6D6D6D",
    "letkf": "#A5A5A5",
    "iensf": "#7E84B8",
    "pce": "#4E79A7",
}
PAIR_COLORS_CRPS = {
    "denkf": "#7A6F67",
    "letkf": "#B8AFA8",
    "iensf": "#9C7DAE",
    "pce": "#5F9EA0",
}
REP_ALPHA = {
    "denkf": 0.34,
    "letkf": 0.30,
    "iensf": 0.40,
    "pce": 0.86,
    "apce": 1.00,
}
REP_LW = {
    "denkf": 0.50,
    "letkf": 0.48,
    "iensf": 0.50,
    "pce": 0.82,
    "apce": 1.10,
}
REP_Z = {
    "denkf": 2,
    "letkf": 2,
    "iensf": 2,
    "pce": 4,
    "apce": 6,
}

STAT_COLORS = {
    "truth": "#202020",
    "denkf": "#4E79A7",
    "letkf": "#2A9D8F",
    "iensf": "#8C6D4E",
    "pce": "#8E63A8",
    "apce": "#F2C94C",
}
STAT_EDGE = {
    "default": "#FFFFFF",
    "apce": "#D62728",
}
STAT_ORDER = ("denkf", "letkf", "iensf", "pce", "apce")
STAT_ROW_SPECS = (
    ("nrmse", r"nRMSE ($\%$)", 100.0),
    ("crps", r"CRPS ($10^{-3}$)", 1000.0),
)


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6.4,
            "font.weight": "regular",
            "axes.titleweight": "regular",
            "axes.labelweight": "regular",
            "mathtext.default": "it",
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.65,
            "legend.frameon": False,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.3,
            "ytick.major.size": 2.3,
            "lines.solid_capstyle": "round",
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def add_panel(ax: plt.Axes, letter: str, title: str | None = None) -> None:
    ax.text(-0.09, 1.04, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.0, fontweight="regular")
    if title:
        ax.text(0.0, 1.04, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=6.8)


def add_top_representative_legend(fig: plt.Figure) -> None:
    handles = [
        Line2D([0], [0], color=REP_COLORS["truth"], lw=1.15, label="Truth"),
        Line2D([0], [0], color=REP_COLORS["denkf"], lw=0.66, alpha=REP_ALPHA["denkf"], label="DEnKF"),
        Line2D([0], [0], color=REP_COLORS["letkf"], lw=0.66, alpha=REP_ALPHA["letkf"], label="LETKF"),
        Line2D([0], [0], color=REP_COLORS["iensf"], lw=0.68, alpha=REP_ALPHA["iensf"], label="IEnSF"),
        Line2D([0], [0], color=REP_COLORS["pce"], lw=0.90, label="PCE"),
        Line2D([0], [0], color=REP_COLORS["apce"], lw=1.16, label="APCE"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.972),
        ncol=6,
        fontsize=5.8,
        handlelength=1.10,
        columnspacing=0.70,
        borderpad=0.08,
        labelspacing=0.2,
    )


def compact_number(value: float, metric: str) -> str:
    if metric == "nrmse":
        return f"{value:.3f}"
    if metric == "crps":
        text = f"{value:.4f}"
        return text.rstrip("0").rstrip(".") if value >= 0.01 else text
    if abs(value) >= 0.01 and abs(value) < 100:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{value:.1e}".replace("e-0", "e-").replace("e+0", "e+")


def add_phase_inset(
    ax: plt.Axes,
    truth: np.ndarray,
    estimates: dict[str, np.ndarray],
    displacement_index: int,
    velocity_index: int,
) -> None:
    inset = ax.inset_axes([0.56, 0.08, 0.38, 0.36])
    start = int(0.66 * len(truth))
    all_curves = [truth[start:]]
    all_curves.extend(estimates[method][start:] for method in ("pce", "apce") if method in estimates)
    xs = np.concatenate([curve[:, displacement_index] for curve in all_curves])
    ys = np.concatenate([curve[:, velocity_index] for curve in all_curves])
    xlo, xhi = np.percentile(xs, [2, 98])
    ylo, yhi = np.percentile(ys, [2, 98])
    for states, color, lw, zorder in [(truth, COLORS["truth"], 0.78, 4)]:
        inset.plot(states[start:, displacement_index], states[start:, velocity_index], color=color, lw=lw, zorder=zorder)
    for method, states in estimates.items():
        inset.plot(
            states[start:, displacement_index],
            states[start:, velocity_index],
            color=COLORS[method],
            lw=REP_LW[method] * 0.78,
            alpha=REP_ALPHA[method],
            zorder=REP_Z[method],
        )
    inset.set_xlim(xlo, xhi)
    inset.set_ylim(ylo, yhi)
    inset.set_xticks([])
    inset.set_yticks([])
    for side in ("right", "top"):
        inset.spines[side].set_visible(True)
    for side in ("left", "bottom"):
        inset.spines[side].set_color("#AFAFAF")
        inset.spines[side].set_linewidth(0.45)
    for side in ("right", "top"):
        inset.spines[side].set_color("#AFAFAF")
        inset.spines[side].set_linewidth(0.45)


def _plot_phase(
    ax: plt.Axes,
    truth: np.ndarray,
    estimates: dict[str, np.ndarray],
    letter: str,
    title: str,
    displacement_index: int,
    velocity_index: int,
) -> None:
    add_panel(ax, letter, title)
    ax.plot(
        truth[:, displacement_index],
        truth[:, velocity_index],
        color=REP_COLORS["truth"],
        lw=1.05,
        zorder=3,
    )
    for method, states in estimates.items():
        ax.plot(
            states[:, displacement_index],
            states[:, velocity_index],
            color=REP_COLORS[method],
            lw=REP_LW[method],
            alpha=REP_ALPHA[method],
            zorder=REP_Z[method],
        )
    focus_curves = [truth]
    focus_curves.extend(estimates[method] for method in ("pce", "apce") if method in estimates)
    focus_x = np.concatenate([curve[:, displacement_index] for curve in focus_curves])
    focus_y = np.concatenate([curve[:, velocity_index] for curve in focus_curves])
    xlo, xhi = np.percentile(focus_x, [0.5, 99.5])
    ylo, yhi = np.percentile(focus_y, [0.5, 99.5])
    xpad = max(0.08 * (xhi - xlo), 1.0e-12)
    ypad = max(0.08 * (yhi - ylo), 1.0e-12)
    ax.set_xlim(xlo - xpad, xhi + xpad)
    ax.set_ylim(ylo - ypad, yhi + ypad)
    ax.set_xlabel(r"$u(t)$")
    ax.set_ylabel(r"$v(t)$")
    ax.tick_params(labelsize=5.6)
    ax.margins(0.05)


def panel_wave_phase(ax: plt.Axes, data: np.lib.npyio.NpzFile) -> None:
    states = data["truth_states"]
    nx = states.shape[1] // 2
    node = nx // 2
    estimates = {
        method: data[f"{method}_mean_states"] for method in REP_METHODS
    }
    _plot_phase(ax, states, estimates, "a", r"Wave phase trajectory at $x_c$", node, nx + node)


def panel_spring(ax: plt.Axes, data: np.lib.npyio.NpzFile) -> None:
    truth = data["truth_states"]
    estimates = {
        method: data[f"{method}_mean_states"] for method in REP_METHODS
    }
    _plot_phase(ax, truth, estimates, "b", "Spring phase trajectory", 0, 1)


def panel_heat(ax: plt.Axes, data: np.lib.npyio.NpzFile) -> None:
    add_panel(ax, "c", "Heat terminal profile")
    x = data["space"]
    truth = data["truth_states"][-1]
    ax.plot(x, truth, color=REP_COLORS["truth"], lw=1.05, zorder=3)
    for method in REP_METHODS:
        key = f"{method}_mean_states"
        ax.plot(x, data[key][-1], color=REP_COLORS[method], lw=REP_LW[method], alpha=REP_ALPHA[method], zorder=REP_Z[method])
    obs_x = x[data["observation_indices"]]
    ax.scatter(
        obs_x,
        truth[data["observation_indices"]],
        s=6.5,
        color="#202020",
        facecolor="white",
        linewidth=0.45,
        zorder=5,
    )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u(x,t_f)$")
    inset = ax.inset_axes([0.54, 0.16, 0.40, 0.36])
    lo = int(np.searchsorted(x, 0.31))
    hi = int(np.searchsorted(x, 0.53))
    inset.plot(x[lo:hi], truth[lo:hi], color=REP_COLORS["truth"], lw=0.78)
    for method in REP_METHODS:
        key = f"{method}_mean_states"
        inset.plot(x[lo:hi], data[key][-1][lo:hi], color=REP_COLORS[method], lw=REP_LW[method] * 0.78, alpha=REP_ALPHA[method])
    y_slice = np.concatenate([truth[lo:hi]] + [data[f"{method}_mean_states"][-1][lo:hi] for method in REP_METHODS])
    y_lo = float(np.min(y_slice))
    y_hi = float(np.max(y_slice))
    y_pad = max(0.05 * (y_hi - y_lo), 1.0e-12)
    inset.set_xlim(float(x[lo]), float(x[hi - 1]))
    inset.set_ylim(y_lo - y_pad, y_hi + y_pad)
    inset.set_xticks([])
    inset.set_yticks([])
    for side in ("right", "top"):
        inset.spines[side].set_visible(True)
    for side in ("left", "bottom"):
        inset.spines[side].set_color("#AFAFAF")
        inset.spines[side].set_linewidth(0.45)
    for side in ("right", "top"):
        inset.spines[side].set_color("#AFAFAF")
        inset.spines[side].set_linewidth(0.45)
    mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec="#AFAFAF", linewidth=0.55)
    ax.tick_params(labelsize=5.6)
    ax.margins(0.04)


def panel_wave_displacement_supplement(
    ax_container: plt.Axes,
    data: np.lib.npyio.NpzFile,
) -> None:
    """Detailed Wave displacement/error plate for Supplementary Information."""
    ax_container.set_axis_off()
    sub = GridSpecFromSubplotSpec(
        2,
        4,
        subplot_spec=ax_container.get_subplotspec(),
        hspace=0.24,
        wspace=0.12,
        width_ratios=[1, 1, 1, 1],
    )
    states = data["truth_states"]
    nx = states.shape[1] // 2
    times = data["times"]
    fields = [
        ("Truth", data["truth_states"][:, :nx]),
        ("DEnKF", data["denkf_mean_states"][:, :nx]),
        ("PCE", data["pce_mean_states"][:, :nx]),
        ("APCE", data["apce_mean_states"][:, :nx]),
    ]
    vmax = max(float(np.nanmax(np.abs(field))) for _, field in fields)
    errors = [
        ("DEnKF error", fields[1][1] - fields[0][1]),
        ("PCE error", fields[2][1] - fields[0][1]),
        ("APCE error", fields[3][1] - fields[0][1]),
        ("PCE - APCE", fields[2][1] - fields[3][1]),
    ]
    emax = max(float(np.nanmax(np.abs(error))) for _, error in errors[:3])
    diff_emax = max(float(np.nanmax(np.abs(errors[3][1]))), 1.0e-12)
    top_axes: list[plt.Axes] = []
    bottom_axes: list[plt.Axes] = []
    for col, (label, field) in enumerate(fields):
        ax = ax_container.figure.add_subplot(sub[0, col])
        top_axes.append(ax)
        im = ax.imshow(
            field.T,
            origin="lower",
            aspect="auto",
            extent=[float(times[0]), float(times[-1]), 0, 1],
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="bilinear",
        )
        ax.set_title(label, fontsize=6.0, pad=1.5, fontweight="regular")
        ax.set_xticklabels([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(r"$x$", fontsize=5.7)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cax_top = ax_container.figure.add_axes([0.900, 0.56, 0.010, 0.30])
    cbar_top = ax_container.figure.colorbar(im, cax=cax_top)
    cbar_top.set_label(r"$u(x,t)$", fontsize=5.7, labelpad=2)
    cbar_top.ax.tick_params(labelsize=5.1, width=0.4, length=2)

    for col, (label, error) in enumerate(errors):
        ax = ax_container.figure.add_subplot(sub[1, col])
        bottom_axes.append(ax)
        local_emax = diff_emax if col == 3 else emax
        local_cmap = "PuOr_r" if col < 3 else "coolwarm"
        ax.imshow(
            error.T,
            origin="lower",
            aspect="auto",
            extent=[float(times[0]), float(times[-1]), 0, 1],
            cmap=local_cmap,
            vmin=-local_emax,
            vmax=local_emax,
            interpolation="bilinear",
        )
        ax.set_title(label, fontsize=6.0, pad=1.5, fontweight="regular")
        ax.set_xlabel(r"$t$", fontsize=5.7)
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(r"$x$", fontsize=5.7)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cax_bottom = ax_container.figure.add_axes([0.900, 0.16, 0.010, 0.30])
    cbar_bottom = ax_container.figure.colorbar(bottom_axes[2].images[0], cax=cax_bottom)
    cbar_bottom.set_label("error", fontsize=5.7, labelpad=2)
    cbar_bottom.ax.tick_params(labelsize=5.1, width=0.4, length=2)
    cax_diff = ax_container.figure.add_axes([0.955, 0.16, 0.010, 0.30])
    cbar_diff = ax_container.figure.colorbar(bottom_axes[3].images[0], cax=cax_diff)
    cbar_diff.set_label("PCE - APCE", fontsize=5.3, labelpad=2)
    cbar_diff.ax.tick_params(labelsize=4.8, width=0.4, length=2)


def metric_heatmap(
    ax: plt.Axes,
    summary: dict[tuple[str, str], dict[str, str]],
    metric: str,
    letter: str,
    title: str,
    cbar_label: str,
    cmap: str,
) -> None:
    add_panel(ax, letter, title)
    values = np.array(
        [[float(summary[(case, method)][f"{metric}_mean"]) for method in METHODS] for case in CASES],
        dtype=float,
    )
    vmin = max(float(values.min()) * 0.82, 1.0e-8)
    vmax = float(values.max()) * 1.18
    image = ax.imshow(
        values,
        aspect="auto",
        cmap=cmap,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(len(METHODS)))
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(len(CASES)))
    ax.set_yticklabels([CASE_LABELS[c] for c in CASES])
    ax.tick_params(labelsize=5.1, length=1.8)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            scaled = (np.log(values[row, col]) - np.log(vmin)) / (np.log(vmax) - np.log(vmin))
            color = "white" if scaled > 0.58 else "#202020"
            ax.text(col, row, compact_number(values[row, col], metric), ha="center", va="center", fontsize=4.65, color=color)
    for col, method in ((METHODS.index("pce"), "pce"), (METHODS.index("apce"), "apce")):
        for row in range(values.shape[0]):
            ax.add_patch(
                Rectangle(
                    (col - 0.47, row - 0.47),
                    0.94,
                    0.94,
                    fill=False,
                    edgecolor=COLORS[method],
                    linewidth=0.9,
                )
            )
    # Cell values carry the quantitative scale; omitting colorbars prevents
    # label collisions in the compact 3 x 3 Figure 2 layout.


def calibration_sharpness(ax: plt.Axes, summary: dict[tuple[str, str], dict[str, str]]) -> None:
    add_panel(ax, "f", "Calibration-sharpness map")
    markers = {"wave": "o", "spring": "s", "heat": "^"}
    for case in CASES:
        for method in METHODS:
            row = summary[(case, method)]
            coverage = float(row["coverage_90_mean"])
            width = float(row["interval_width_90_mean"])
            ax.scatter(
                coverage,
                width,
                s=25 if method in {"pce", "apce"} else 17,
                marker=markers[case],
                color=COLORS[method],
                edgecolor="#202020" if method in {"pce", "apce"} else "none",
                linewidth=0.3,
                zorder=4,
            )
    ax.axvline(0.9, color="#666666", lw=0.6, ls=(0, (3, 2)))
    ax.set_yscale("log")
    ax.set_xlim(0.35, 1.02)
    ax.set_ylim(8.0e-3, 1.8e-1)
    ax.set_xlabel("90% coverage")
    ax.set_ylabel("interval width")
    ax.grid(False)
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 3.0), numticks=5))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10, labelOnlyBase=False))
    case_handles = [
        Line2D([0], [0], marker=markers[c], color="none", markerfacecolor="#777777", markersize=4.1, label=CASE_LABELS[c])
        for c in CASES
    ]
    leg1 = ax.legend(handles=case_handles, loc="upper left", fontsize=4.7, ncol=3, handletextpad=0.2, columnspacing=0.45, borderpad=0.12)
    ax.add_artist(leg1)
    ax.tick_params(labelsize=5.2)


def bootstrap_interval(values: np.ndarray, seed: int, n_bootstrap: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, values.size, size=(n_bootstrap, values.size))].mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def build_pairwise_entries(
    paired: list[dict[str, str]],
    runs: list[dict[str, str]],
    metric: str,
) -> dict[tuple[str, str], dict[str, float]]:
    entries: dict[tuple[str, str], dict[str, float]] = {}
    for row in paired:
        if row["metric"] != metric:
            continue
        entries[(row["case"], row["baseline"])] = {
            "mean": float(row["mean_difference_apce_minus_baseline"]),
            "low": float(row["ci95_low"]),
            "high": float(row["ci95_high"]),
        }
    by_key: dict[tuple[str, str, str], float] = {}
    for row in runs:
        case = row["case"]
        method = row["method"]
        seed = row["seed"]
        if case in CASES and method in METHODS and row.get(metric, ""):
            by_key[(case, method, seed)] = float(row[metric])
    for case_index, case in enumerate(CASES):
        for baseline in ("pce",):
            diffs = []
            seeds = sorted(
                seed for c, method, seed in by_key
                if c == case and method == "apce" and (case, baseline, seed) in by_key
            )
            for seed in seeds:
                diffs.append(by_key[(case, "apce", seed)] - by_key[(case, baseline, seed)])
            values = np.asarray(diffs, dtype=float)
            if values.size:
                low, high = bootstrap_interval(values, seed=20260807 + 101 * case_index + len(baseline))
                entries[(case, baseline)] = {
                    "mean": float(values.mean()),
                    "low": low,
                    "high": high,
                }
    return entries


def forest(
    ax: plt.Axes,
    paired: list[dict[str, str]],
    runs: list[dict[str, str]],
    metric: str,
    letter: str,
    title: str,
    xlabel: str,
) -> None:
    add_panel(ax, letter, title)
    entries_by_key = build_pairwise_entries(paired, runs, metric)
    x = np.arange(len(CASES), dtype=float)
    width = 0.145
    offsets = {
        "denkf": -2 * width,
        "letkf": -width,
        "iensf": 0.0,
        "pce": width,
    }
    palette = PAIR_COLORS_NRMSE if metric == "nrmse" else PAIR_COLORS_CRPS
    for baseline in PAIR_BASELINES:
        estimates = []
        low = []
        high = []
        for case in CASES:
            row = entries_by_key[(case, baseline)]
            est = float(row["mean"])
            estimates.append(est)
            low.append(est - float(row["low"]))
            high.append(float(row["high"]) - est)
        ax.bar(
            x + offsets[baseline],
            estimates,
            width=width * 0.88,
            color=palette[baseline],
            edgecolor="#202020",
            linewidth=0.32,
            label=METHOD_LABELS[baseline],
            zorder=3,
        )
        ax.errorbar(
            x + offsets[baseline],
            estimates,
            yerr=np.vstack([low, high]),
            fmt="none",
            ecolor="#202020",
            elinewidth=0.55,
            capsize=1.8,
            capthick=0.55,
            zorder=4,
        )
    ax.axhline(0.0, color="#606060", lw=0.62, ls=(0, (3, 2)))
    ax.set_xticks(x)
    ax.set_xticklabels([CASE_LABELS[c] for c in CASES], fontsize=5.0)
    ax.set_xlabel(xlabel, fontsize=5.6)
    ax.grid(False)
    if metric == "nrmse":
        ax.legend(loc="lower left", fontsize=5.2, ncol=3, handlelength=0.8, columnspacing=0.45, borderpad=0.08)
    ax.tick_params(labelsize=5.1)


def alpha_summary(ax: plt.Axes, wave: np.lib.npyio.NpzFile, spring: np.lib.npyio.NpzFile, heat: np.lib.npyio.NpzFile) -> None:
    add_panel(ax, "i", "Cognitive-orbit weights")
    series = [
        ("wave", wave["pce_final_weights"], wave["apce_final_weights"], np.linspace(0.08, 0.92, len(wave["pce_final_weights"]))),
        ("spring", spring["pce_alpha_weight_history"][-1], spring["apce_alpha_weight_history"][-1], spring["alpha_grid"]),
        ("heat", heat["pce_alpha_weight_history"][-1], heat["apce_alpha_weight_history"][-1], heat["alpha_grid"]),
    ]
    for case, pce, apce, alpha in series:
        mask = alpha <= 0.5
        ax.plot(alpha[mask], pce[mask], color=CASE_COLORS[case], lw=0.82, ls=(0, (3, 2)), alpha=0.52, label=f"{CASE_LABELS[case]} PCE")
        ax.plot(alpha[mask], apce[mask], color=CASE_COLORS[case], lw=1.32, ls="-", alpha=0.98, label=f"{CASE_LABELS[case]} APCE")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("final weight")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(bottom=0)
    ax.grid(False)
    ax.tick_params(labelsize=5.2)
    case_handles = [
        Line2D([0], [0], color=CASE_COLORS[c], lw=1.0, label=CASE_LABELS[c]) for c in CASES
    ]
    style_handles = [
        Line2D([0], [0], color="#555555", lw=0.82, ls=(0, (3, 2)), alpha=0.52, label="PCE"),
        Line2D([0], [0], color="#555555", lw=1.32, ls="-", label="APCE"),
    ]
    leg1 = ax.legend(handles=case_handles, loc="upper right", fontsize=4.7, ncol=1, handlelength=1.0, borderpad=0.1, labelspacing=0.15)
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, loc="upper center", fontsize=4.7, ncol=2, handlelength=1.3, borderpad=0.1, columnspacing=0.65)


def _scaled_metric_series(rows: list[dict[str, str]], case: str, metric: str, scale: float) -> dict[str, np.ndarray]:
    series: dict[str, np.ndarray] = {}
    for method in STAT_ORDER:
        values = [
            float(row[metric]) * scale
            for row in rows
            if row["case"] == case and row["method"] == method and row.get("valid", "True") == "True" and row.get(metric, "")
        ]
        series[method] = np.asarray(values, dtype=float)
    return series


def _metric_scale(metric: str) -> float:
    for key, _, scale in STAT_ROW_SPECS:
        if key == metric:
            return scale
    raise KeyError(metric)


def _metric_label(metric: str) -> str:
    for key, label, _ in STAT_ROW_SPECS:
        if key == metric:
            return label
    raise KeyError(metric)


def _metric_summary_value(row: dict[str, str], metric: str, scale: float) -> float:
    return float(row[f"{metric}_mean"]) * scale


def _metric_summary_bounds(row: dict[str, str], metric: str, scale: float) -> tuple[float, float]:
    return float(row[f"{metric}_ci95_low"]) * scale, float(row[f"{metric}_ci95_high"]) * scale


def _style_boxplot_boxes(bp: dict[str, list], colors: dict[str, str]) -> None:
    for patch, method in zip(bp["boxes"], STAT_ORDER, strict=True):
        patch.set_facecolor(colors[method])
        patch.set_alpha(0.92)
        patch.set_edgecolor(STAT_EDGE["apce"] if method == "apce" else "#454545")
        patch.set_linewidth(1.0 if method == "apce" else 0.72)
    for median in bp["medians"]:
        median.set_color("#1F1F1F")
        median.set_linewidth(0.82)
    for whisker in bp["whiskers"]:
        whisker.set_color("#444444")
        whisker.set_linewidth(0.65)
    for cap in bp["caps"]:
        cap.set_color("#444444")
        cap.set_linewidth(0.65)


def _draw_seed_boxpanel(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    case: str,
    metric: str,
    scale: float,
    show_xlabel: bool,
    show_ylabel: bool,
    title: str | None,
) -> None:
    series = _scaled_metric_series(rows, case, metric, scale)
    data = [series[method] for method in STAT_ORDER]
    positions = np.arange(1, len(STAT_ORDER) + 1)
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.56,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
        medianprops={"color": "#1F1F1F", "linewidth": 0.82},
        whiskerprops={"color": "#444444", "linewidth": 0.65},
        capprops={"color": "#444444", "linewidth": 0.65},
    )
    _style_boxplot_boxes(bp, STAT_COLORS)
    rng = np.random.default_rng(20260807 + 13 * CASES.index(case) + (0 if metric == "nrmse" else 31))
    for x_pos, method in zip(positions, STAT_ORDER, strict=True):
        values = series[method]
        jitter = rng.normal(0.0, 0.05, size=values.size)
        ax.scatter(
            np.full(values.size, x_pos, dtype=float) + jitter,
            values,
            s=3.2,
            color=STAT_COLORS[method],
            alpha=0.22,
            edgecolors="none",
            zorder=1,
        )
    ax.set_xlim(0.45, len(STAT_ORDER) + 0.55)
    row_vals = np.concatenate([v for v in data if v.size])
    if row_vals.size:
        y_min = 0.0
        y_max = float(np.nanmax(row_vals)) * 1.20
        ax.set_ylim(y_min, y_max)
    if title:
        ax.set_title(title, fontsize=5.4, pad=1.0, fontweight="regular")
    ax.set_xticks(positions)
    if show_xlabel:
        ax.set_xticklabels([METHOD_LABELS[m] for m in STAT_ORDER], rotation=28, ha="right", fontsize=4.2)
    else:
        ax.tick_params(axis="x", labelbottom=False)
    if show_ylabel:
        ax.set_ylabel(_metric_label(metric), fontsize=5.0, labelpad=1.0)
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.tick_params(labelleft=False)
    ax.tick_params(labelsize=4.1, length=1.5)
    ax.grid(False)
    ax.tick_params(axis="x", pad=0.6)


def _draw_summary_barpanel(
    ax: plt.Axes,
    summary: dict[tuple[str, str], dict[str, str]],
    case: str,
    metric: str,
    scale: float,
    show_xlabel: bool,
    show_ylabel: bool,
    title: str | None,
) -> None:
    x = np.arange(1, len(STAT_ORDER) + 1, dtype=float)
    means = []
    lower = []
    upper = []
    for method in STAT_ORDER:
        row = summary[(case, method)]
        est = _metric_summary_value(row, metric, scale)
        lo, hi = _metric_summary_bounds(row, metric, scale)
        means.append(est)
        lower.append(est - lo)
        upper.append(hi - est)
    means_arr = np.asarray(means, dtype=float)
    lower_arr = np.asarray(lower, dtype=float)
    upper_arr = np.asarray(upper, dtype=float)
    bars = ax.bar(
        x,
        means_arr,
        width=0.66,
        color=[STAT_COLORS[m] for m in STAT_ORDER],
        edgecolor=[STAT_EDGE["apce"] if m == "apce" else "#FFFFFF" for m in STAT_ORDER],
        linewidth=0.95,
        zorder=3,
    )
    for bar, method in zip(bars, STAT_ORDER, strict=True):
        if method == "apce":
            bar.set_edgecolor(STAT_EDGE["apce"])
            bar.set_linewidth(1.35)
    ax.errorbar(
        x,
        means_arr,
        yerr=np.vstack([lower_arr, upper_arr]),
        fmt="none",
        ecolor="#2E2E2E",
        elinewidth=0.62,
        capsize=1.9,
        capthick=0.62,
        zorder=5,
    )
    ax.set_xlim(0.45, len(STAT_ORDER) + 0.55)
    if means_arr.size:
        upper_bound = float(np.max(means_arr + upper_arr)) * 1.18
        ax.set_ylim(0.0, upper_bound)
    if title:
        ax.set_title(title, fontsize=5.4, pad=1.0, fontweight="regular")
    ax.set_xticks(x)
    if show_xlabel:
        ax.set_xticklabels([METHOD_LABELS[m] for m in STAT_ORDER], rotation=28, ha="right", fontsize=4.2)
    else:
        ax.tick_params(axis="x", labelbottom=False)
    if show_ylabel:
        ax.set_ylabel(_metric_label(metric), fontsize=5.0, labelpad=1.0)
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.tick_params(labelleft=False)
    ax.tick_params(labelsize=4.1, length=1.5)
    ax.grid(False)
    ax.tick_params(axis="x", pad=0.6)


def boxplot_wall(ax_container: plt.Axes, rows: list[dict[str, str]]) -> None:
    ax_container.set_axis_off()
    add_panel(ax_container, "d", "Seed-wise distributions (n=50)")
    sub = GridSpecFromSubplotSpec(2, 3, subplot_spec=ax_container.get_subplotspec(), hspace=0.34, wspace=0.18)
    for row_idx, (metric, _, scale) in enumerate(STAT_ROW_SPECS):
        for col_idx, case in enumerate(CASES):
            ax = ax_container.figure.add_subplot(sub[row_idx, col_idx])
            _draw_seed_boxpanel(
                ax,
                rows,
                case,
                metric,
                scale,
                show_xlabel=row_idx == 1,
                show_ylabel=col_idx == 0,
                title=CASE_LABELS[case] if row_idx == 0 else None,
            )


def bar_wall(ax_container: plt.Axes, summary: dict[tuple[str, str], dict[str, str]]) -> None:
    ax_container.set_axis_off()
    add_panel(ax_container, "g", "Seed-wise means with 95% CI (n=50)")
    sub = GridSpecFromSubplotSpec(2, 3, subplot_spec=ax_container.get_subplotspec(), hspace=0.34, wspace=0.18)
    for row_idx, (metric, _, scale) in enumerate(STAT_ROW_SPECS):
        for col_idx, case in enumerate(CASES):
            ax = ax_container.figure.add_subplot(sub[row_idx, col_idx])
            _draw_summary_barpanel(
                ax,
                summary,
                case,
                metric,
                scale,
                show_xlabel=row_idx == 1,
                show_ylabel=col_idx == 0,
                title=CASE_LABELS[case] if row_idx == 0 else None,
            )


def write_contract(output_dir: Path) -> None:
    text = """Core conclusion:
Across wave, spring and heat systems under one frozen paired protocol, APCE improves state error and probabilistic skill over all numerically valid training-free baselines while retaining identifiable cognitive-orbit evidence.

Figure archetype:
Asymmetric mixed-modality evidence wall with a compact top row and two 2x3 statistical walls.

Panel map:
a: Wave central-node phase trajectory.
b: Spring phase-space trajectory.
c: Heat terminal spatial profile.
d: 2x3 seed-wise distribution wall (nRMSE and CRPS).
f: calibration-sharpness map.
g: 2x3 mean-and-CI wall (nRMSE and CRPS).
i: PCE/APCE cognitive-orbit weights across cases.

Statistics:
n=50 paired seeds per system and method; boxplot walls show seed-level distributions scaled to percent / 10^-3 units; bar walls show mean and 95% CI; Holm-adjusted comparisons remain in the source data.

Supplementary:
The detailed Wave displacement-field and error plate is exported separately as a Supplementary Figure.
"""
    (output_dir / "figure2_classical_uncertain_systems_v7_contract.md").write_text(text, encoding="utf-8")


def export_wave_supplement(data: np.lib.npyio.NpzFile, output_dir: Path) -> None:
    fig = plt.figure(figsize=(7.6, 4.25))
    outer = fig.add_gridspec(1, 1, left=0.055, right=0.88, top=0.94, bottom=0.12)
    host = fig.add_subplot(outer[0, 0])
    host.set_axis_off()
    panel_wave_displacement_supplement(host, data)
    base = output_dir / "supp_figure_wave_displacement"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


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
    set_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = read_csv(args.summary_csv)
    run_rows = read_csv(args.runs_csv)
    paired_rows = read_csv(args.paired_csv)
    summary = {(r["case"], r["method"]): r for r in summary_rows}
    wave = np.load(args.wave_npz, allow_pickle=True)
    spring = np.load(args.spring_npz, allow_pickle=True)
    heat = np.load(args.heat_npz, allow_pickle=True)

    fig = plt.figure(figsize=(14.9, 11.4))
    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=[1.00, 1.25, 1.25],
        hspace=0.30,
        left=0.045,
        right=0.985,
        top=0.915,
        bottom=0.060,
    )
    add_top_representative_legend(fig)
    top = GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[0, 0], wspace=0.30, width_ratios=[1.0, 1.0, 1.0, 0.90, 1.05])
    panel_wave_phase(fig.add_subplot(top[0, 0]), wave)
    panel_spring(fig.add_subplot(top[0, 1]), spring)
    panel_heat(fig.add_subplot(top[0, 2]), heat)
    calibration_sharpness(fig.add_subplot(top[0, 3]), summary)
    alpha_summary(fig.add_subplot(top[0, 4]), wave, spring, heat)

    boxplot_wall(fig.add_subplot(outer[1, 0]), run_rows)
    bar_wall(fig.add_subplot(outer[2, 0]), summary)

    base = args.output_dir / "figure2_classical_uncertain_systems_v7"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    export_wave_supplement(wave, args.output_dir)
    for source in (args.summary_csv, args.runs_csv, args.paired_csv):
        shutil.copy2(source, args.output_dir / f"source_data_{source.name}")
    write_contract(args.output_dir)
    qa = {
        "figure": "figure2_classical_uncertain_systems_v7",
        "backend": "python/matplotlib",
        "font": "Arial with sans-serif fallback",
        "font_weight": "regular",
        "svg_text_editable": True,
        "statistics": "n=50 paired seeds; boxplot walls show seed-level distributions in scaled units; bar walls show mean and 95% CI; Holm-adjusted comparisons in source data",
        "formats": ["svg", "pdf", "png", "tiff"],
        "panels": ["a", "b", "c", "f", "i", "d", "g"],
        "supplementary_outputs": ["supp_figure_wave_displacement.svg", "supp_figure_wave_displacement.pdf", "supp_figure_wave_displacement.png", "supp_figure_wave_displacement.tiff"],
    }
    (args.output_dir / "figure2_classical_uncertain_systems_v7_qa.json").write_text(
        json.dumps(qa, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(base), "supplement": str(args.output_dir / "supp_figure_wave_displacement")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
