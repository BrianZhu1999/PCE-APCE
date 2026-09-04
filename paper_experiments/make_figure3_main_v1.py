"""Build Figure 3 main composite from the authoritative freq1--freq8 data.

Figure contract
---------------
Core conclusion:
    Under progressively sparse observations, PCE/APCE become more favourable
    because cognitive-path evidence retains parameter information while direct
    state--parameter covariance updating becomes less reliable.

Archetype:
    Asymmetric mixed-modality figure: one accepted hero ODE atlas plus six
    compact quantitative/mechanistic panels.

Panel map:
    a  Accepted five-case ODE atlas (reused as a raster asset).
    b  Final alpha-error heatmap across cases and observation intervals.
    c  Seed-wise nRMSE boxplots at a sparse representative interval.
    d  Seed-level nRMSE gain distribution relative to Aug-EnKF.
    e  Paired forest against all baselines.
    f  Multi-metric radar summary.
    g  Evidence entropy versus observation interval.

Data:
    Authoritative 14,000-record Figure 3 freq1--freq8 formal source data.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = (
    PROJECT_ROOT
    / "source_data"
    / "figure3_selected5_freq1to8_formal_50seeds_allmethods_20260813"
    / "combined"
)
SOURCE_DIR = DATA_DIR / "source_data"
RUN_CSV = SOURCE_DIR / "figure3_freq1to8_formal_authoritative_run_source_data.csv"
SUMMARY_CSV = SOURCE_DIR / "figure3_freq1to8_formal_method_summary.csv"
PAIRED_CSV = SOURCE_DIR / "figure3_freq1to8_formal_paired_comparisons.csv"
MANIFEST_JSON = DATA_DIR / "figure3_freq1to8_formal_authoritative_manifest.json"
A1_IMAGE = Path(
    r"figures\figure3a_selected5_ode_panels_row_no_sir_v17.png"
)

OUT_DIR = PROJECT_ROOT / "ncs_english_latex" / "figures"
OUT_STEM = "figure3_main_selected5_freq1to8_v8"

CASES = ["pk_infusion", "chemical", "pendulum", "fhn", "robertson"]
CASE_LABELS = {
    "pk_infusion": "PK infusion",
    "chemical": "Chemical",
    "pendulum": "Pendulum",
    "fhn": "FHN",
    "robertson": "Robertson",
}
CASE_SHORT = {
    "pk_infusion": "PK",
    "chemical": "Chem.",
    "pendulum": "Pend.",
    "fhn": "FHN",
    "robertson": "Rob.",
}
METHODS = ["denkf", "letkf", "iensf", "aug_enkf", "bma_static", "pce", "apce"]
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
    "denkf": "#8B8B8B",
    "letkf": "#5AA087",
    "iensf": "#9A7B55",
    "aug_enkf": "#4C78A8",
    "bma_static": "#7A5195",
    "pce": "#64A9DC",
    "apce": "#E68613",
}
CASE_COLORS = {
    "pk_infusion": "#4477AA",
    "chemical": "#66A61E",
    "pendulum": "#AA3377",
    "fhn": "#CCBB44",
    "robertson": "#228833",
}

SELECTED_FREQ = 6
FREQS = list(range(1, 9))
N_SEEDS = 50


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.8,
            "axes.labelsize": 7.8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.1,
            "ytick.labelsize": 7.1,
            "legend.fontsize": 7.4,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.72,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "legend.frameon": False,
            "lines.solid_capstyle": "round",
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fnum(value: str | float | int | None) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan


def compact_tick(value: float, _pos: int | None = None) -> str:
    if abs(value) < 1e-12:
        return "0"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    if abs(value) >= 1:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def style_ax(ax: plt.Axes, *, numeric_x: bool = True, numeric_y: bool = True) -> None:
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(length=2.2, pad=1.6, width=0.65)
    ax.grid(False)
    if numeric_x:
        ax.xaxis.set_major_formatter(FuncFormatter(compact_tick))
    if numeric_y:
        ax.yaxis.set_major_formatter(FuncFormatter(compact_tick))


def add_panel_label(ax: plt.Axes, letter: str, title: str, *, x: float = -0.08, y: float = 1.05) -> None:
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12.5,
        fontweight="bold",
        color="#111111",
    )
    ax.text(
        0.055,
        y + 0.01,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.6,
        color="#111111",
    )


def load_data() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], dict]:
    for path in [RUN_CSV, SUMMARY_CSV, PAIRED_CSV, MANIFEST_JSON, A1_IMAGE]:
        if not path.is_file():
            raise FileNotFoundError(path)
    runs = read_csv(RUN_CSV)
    summary = read_csv(SUMMARY_CSV)
    paired = read_csv(PAIRED_CSV)
    manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    return runs, summary, paired, manifest


def valid_runs(runs: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        r
        for r in runs
        if r.get("status") == "completed"
        and r.get("numerical_status") == "valid"
        and r.get("case") in CASES
        and r.get("method") in METHODS
    ]


def grouped_values(
    runs: list[dict[str, str]],
    *,
    freq: int | None = None,
    case: str | None = None,
    method: str | None = None,
    metric: str,
    scale: float = 1.0,
) -> list[float]:
    vals: list[float] = []
    for r in runs:
        if freq is not None and int(float(r["obs_interval_factor"])) != freq:
            continue
        if case is not None and r["case"] != case:
            continue
        if method is not None and r["method"] != method:
            continue
        v = fnum(r.get(metric))
        if math.isfinite(v):
            vals.append(v * scale)
    return vals


def sem_ci(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan, math.nan, math.nan
    mean = float(np.mean(arr))
    if arr.size <= 1:
        return mean, mean, mean
    sem = float(np.std(arr, ddof=1) / math.sqrt(arr.size))
    z = NormalDist().inv_cdf(0.975)
    return mean, mean - z * sem, mean + z * sem


def luminance_rgba(rgba) -> float:
    r, g, b = rgba[:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def draw_panel_a(ax: plt.Axes) -> None:
    img = mpimg.imread(A1_IMAGE)
    # The accepted panel-a raster contains a large blank bottom margin because
    # it was originally exported as a standalone one-row figure.  Crop only the
    # display copy in this composite; the source panel-a asset is not modified.
    if img.shape[0] > 3200:
        img = img[300:2955, :, :]
    ax.imshow(img)
    ax.set_axis_off()
    # Re-label within the composite. The embedded asset already includes 'a';
    # cover it softly and redraw to match the main figure typography.
    ax.add_patch(Rectangle((0.0, 0.74), 0.055, 0.24, transform=ax.transAxes, facecolor="#F8F3EA", edgecolor="none", zorder=5))
    ax.text(
        -0.005,
        0.94,
        "a",
        transform=ax.transAxes,
        fontsize=12.5,
        fontweight="bold",
        ha="left",
        va="top",
        color="#111111",
        zorder=6,
    )


def draw_alpha_error_heatmap(ax: plt.Axes, runs: list[dict[str, str]], plot_rows: list[dict[str, object]]) -> None:
    add_panel_label(ax, "b", r"Final $\alpha$ error")
    data = np.zeros((len(CASES), len(FREQS)), dtype=float)
    method_choice: dict[tuple[str, int], str] = {}
    for i, case in enumerate(CASES):
        for j, freq in enumerate(FREQS):
            vals_by_method = {
                m: np.mean(grouped_values(runs, freq=freq, case=case, method=m, metric="alpha_absolute_error"))
                for m in ("pce", "apce")
            }
            best_method = min(vals_by_method, key=lambda m: vals_by_method[m])
            data[i, j] = vals_by_method[best_method] * 100.0
            method_choice[(case, freq)] = best_method
            plot_rows.append(
                {
                    "panel": "b",
                    "case": case,
                    "freq": freq,
                    "method": METHOD_LABELS[best_method],
                    "metric": "best_pce_apce_alpha_mae_percent",
                    "value": data[i, j],
                }
            )

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "alpha_heat",
        ["#F7EAD4", "#F0C987", "#D8755F", "#6B315F"],
    )
    finite = data[np.isfinite(data)]
    vmax = float(np.nanpercentile(finite, 96)) if finite.size else 1.0
    vmax = max(vmax, 5.0)
    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            rgba = cmap(norm(val))
            color = "white" if luminance_rgba(rgba) < 0.48 else "#222222"
            ax.text(j, i, f"{val:.1f}".rstrip("0").rstrip("."), ha="center", va="center", fontsize=6.7, color=color)
            if method_choice[(CASES[i], FREQS[j])] == "apce":
                ax.add_patch(Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, lw=0.75, ec="#E68613"))
    ax.set_xticks(range(len(FREQS)), [str(f) for f in FREQS])
    ax.set_yticks(range(len(CASES)), [CASE_SHORT[c] for c in CASES])
    ax.set_xlabel("Observation interval")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cbar.ax.tick_params(labelsize=6.5, length=2, pad=1)
    cbar.set_label("%", fontsize=6.8, labelpad=1)


def draw_nrmse_boxplots(ax: plt.Axes, runs: list[dict[str, str]], plot_rows: list[dict[str, object]]) -> None:
    add_panel_label(ax, "c", f"nRMSE distributions, interval {SELECTED_FREQ}")
    positions: list[float] = []
    samples: list[list[float]] = []
    colors: list[str] = []
    labels: list[str] = []
    center_positions: list[float] = []
    width = 0.13
    offsets = np.linspace(-0.34, 0.34, len(METHODS))
    for ci, case in enumerate(CASES):
        base = float(ci)
        center_positions.append(base)
        labels.append(CASE_SHORT[case])
        for method, off in zip(METHODS, offsets, strict=True):
            vals = grouped_values(runs, freq=SELECTED_FREQ, case=case, method=method, metric="nrmse", scale=100.0)
            if not vals:
                continue
            positions.append(base + float(off))
            samples.append(vals)
            colors.append(METHOD_COLORS[method])
            for v in vals:
                plot_rows.append(
                    {
                        "panel": "c",
                        "case": case,
                        "freq": SELECTED_FREQ,
                        "method": METHOD_LABELS[method],
                        "metric": "nrmse_percent",
                        "value": v,
                    }
                )

    bp = ax.boxplot(samples, positions=positions, widths=width, patch_artist=True, showfliers=False, whis=1.5)
    for patch, color, pos in zip(bp["boxes"], colors, positions, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.70)
        patch.set_edgecolor("white")
        patch.set_linewidth(0.55)
    for key in ("whiskers", "caps", "medians"):
        for artist in bp[key]:
            artist.set_color("#333333")
            artist.set_linewidth(0.55)
    rng = np.random.default_rng(20260813)
    for vals, pos, color in zip(samples, positions, colors, strict=True):
        jitter = rng.normal(0, 0.012, size=len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals, s=4.8, color=color, edgecolors="white", linewidths=0.16, alpha=0.34, zorder=2)
    ax.set_ylabel("nRMSE (%)")
    ymax = np.nanpercentile(np.concatenate([np.asarray(v) for v in samples]), 99.2) * 1.12
    ax.set_ylim(0, max(ymax, 2.2))
    style_ax(ax, numeric_x=False)
    ax.set_xticks(center_positions, labels)


def draw_gain_swarm(ax: plt.Axes, runs: list[dict[str, str]], plot_rows: list[dict[str, object]]) -> None:
    add_panel_label(ax, "d", r"Seed-wise nRMSE gain")
    rng = np.random.default_rng(20260813 + 17)
    for ci, case in enumerate(CASES):
        base_by_seed: dict[str, float] = {}
        for r in runs:
            if int(float(r["obs_interval_factor"])) == SELECTED_FREQ and r["case"] == case and r["method"] == "aug_enkf":
                base_by_seed[r["seed"]] = fnum(r["nrmse"]) * 100.0
        vals_by_method = {}
        for method in ("pce", "apce"):
            vals: list[float] = []
            for r in runs:
                if int(float(r["obs_interval_factor"])) == SELECTED_FREQ and r["case"] == case and r["method"] == method:
                    if r["seed"] in base_by_seed:
                        gain = base_by_seed[r["seed"]] - fnum(r["nrmse"]) * 100.0
                        vals.append(gain)
                        plot_rows.append(
                            {
                                "panel": "d",
                                "case": case,
                                "freq": SELECTED_FREQ,
                                "method": METHOD_LABELS[method],
                                "baseline": "Aug-EnKF",
                                "metric": "nrmse_gain_percent",
                                "value": gain,
                            }
                        )
            vals_by_method[method] = np.asarray(vals, dtype=float)
        for off, method in [(-0.095, "pce"), (0.095, "apce")]:
            vals = vals_by_method[method]
            if vals.size == 0:
                continue
            y_grid = np.linspace(vals.min(), vals.max(), 160)
            if vals.size > 1:
                bw = max(np.std(vals, ddof=1) * 0.35, 0.015)
                density = np.zeros_like(y_grid)
                for v in vals:
                    density += np.exp(-0.5 * ((y_grid - v) / bw) ** 2)
                density /= max(float(density.max()), 1e-12)
            else:
                density = np.ones_like(y_grid)
            half = 0.070 * density
            x0 = ci + off
            if method == "pce":
                poly_x = np.r_[np.full_like(y_grid, x0), x0 - half[::-1]]
                poly_y = np.r_[y_grid, y_grid[::-1]]
            else:
                poly_x = np.r_[np.full_like(y_grid, x0), x0 + half[::-1]]
                poly_y = np.r_[y_grid, y_grid[::-1]]
            ax.fill(poly_x, poly_y, color=METHOD_COLORS[method], alpha=0.24, lw=0, zorder=1)
            ax.scatter(
                np.full(vals.size, x0) + rng.normal(0, 0.012, size=vals.size),
                vals,
                s=6.0,
                color=METHOD_COLORS[method],
                edgecolors="white",
                linewidths=0.2,
                alpha=0.58,
                zorder=3,
            )
            mean, lo, hi = sem_ci(vals.tolist())
            ax.plot([x0 - 0.06, x0 + 0.06], [mean, mean], color=METHOD_COLORS[method], lw=1.1, zorder=4)
            ax.vlines(x0, lo, hi, color=METHOD_COLORS[method], lw=1.0, zorder=4)
    ax.axhline(0, color="#6E6E6E", lw=0.75, ls=(0, (3, 2)))
    ax.set_ylabel("gain vs Aug-EnKF (%)")
    ax.set_xlim(-0.45, len(CASES) - 0.55)
    ymin, ymax = ax.get_ylim()
    ax.text(0.02, 0.05, "higher is better", transform=ax.transAxes, fontsize=6.8, color="#666666")
    style_ax(ax, numeric_x=False)
    ax.set_xticks(range(len(CASES)), [CASE_SHORT[c] for c in CASES])


def draw_grouped_forest(ax: plt.Axes, runs: list[dict[str, str]], plot_rows: list[dict[str, object]]) -> None:
    add_panel_label(ax, "e", "Paired forest across baselines", x=-0.08, y=1.04)
    targets = ("pce", "apce")
    baselines = ["denkf", "letkf", "iensf", "aug_enkf", "bma_static"]
    baseline_markers = {
        "denkf": "o",
        "letkf": "s",
        "iensf": "D",
        "aug_enkf": "^",
        "bma_static": "P",
    }
    records: list[tuple[str, str, str, float, float, float]] = []
    for case in CASES:
        for target in targets:
            target_by_seed: dict[str, float] = {}
            for r in runs:
                if int(float(r["obs_interval_factor"])) == SELECTED_FREQ and r["case"] == case and r["method"] == target:
                    target_by_seed[r["seed"]] = fnum(r["nrmse"]) * 100.0
            for baseline in baselines:
                diffs: list[float] = []
                for r in runs:
                    if int(float(r["obs_interval_factor"])) == SELECTED_FREQ and r["case"] == case and r["method"] == baseline and r["seed"] in target_by_seed:
                        gain = fnum(r["nrmse"]) * 100.0 - target_by_seed[r["seed"]]
                        diffs.append(gain)
                mean, lo, hi = sem_ci(diffs)
                records.append((case, target, baseline, mean, lo, hi))
                plot_rows.append(
                    {
                        "panel": "e",
                        "case": case,
                        "freq": SELECTED_FREQ,
                        "method": METHOD_LABELS[target],
                        "baseline": METHOD_LABELS[baseline],
                        "metric": "paired_nrmse_gain_percent_mean",
                        "value": mean,
                        "ci_low": lo,
                        "ci_high": hi,
                        "n": len(diffs),
                    }
                )

    baseline_offsets = {
        baseline: off for baseline, off in zip(baselines, np.linspace(-0.28, 0.28, len(baselines)), strict=True)
    }
    target_offsets = {"pce": -0.026, "apce": 0.026}
    all_vals = [v for _, _, _, _mean, lo, hi in records for v in (lo, hi) if math.isfinite(v)]
    lower = -1.25
    upper = 6.0
    if all_vals:
        # A single Robertson-vs-BMA gain is much larger than the remaining
        # paired effects; cap the display window and mark values beyond it.
        lower = min(lower, float(np.nanpercentile(all_vals, 2)) - 0.25)
        upper = max(upper, float(np.nanpercentile(all_vals, 95)) + 0.55)

    ax.axvline(0, color="#666666", lw=0.75, ls=(0, (3, 2)), zorder=1)
    for ci, case in enumerate(CASES):
        ax.axhspan(ci - 0.43, ci + 0.43, color="#F6F6F6" if ci % 2 == 0 else "#FFFFFF", zorder=0)
        for baseline in baselines:
            for target in targets:
                rec = next(r for r in records if r[0] == case and r[1] == target and r[2] == baseline)
                _, _, _, mean, lo, hi = rec
                yy = ci + baseline_offsets[baseline] + target_offsets[target]
                color = METHOD_COLORS[target]
                if math.isfinite(lo) and math.isfinite(hi):
                    ax.hlines(yy, max(lo, lower), min(hi, upper), color=color, lw=0.95, alpha=0.90, zorder=3)
                if math.isfinite(mean):
                    marker = baseline_markers[baseline]
                    if mean > upper:
                        ax.plot(upper, yy, marker=">", ms=4.3, color=color, mec="white", mew=0.45, clip_on=False, zorder=5)
                        ax.text(upper, yy, f"{mean:.0f}", ha="left", va="center", fontsize=5.9, color=color, clip_on=False)
                    elif mean < lower:
                        ax.plot(lower, yy, marker="<", ms=4.3, color=color, mec="white", mew=0.45, clip_on=False, zorder=5)
                    else:
                        ax.plot(
                            mean,
                            yy,
                            marker=marker,
                            ms=3.8,
                            color=color,
                            mec="white",
                            mew=0.45,
                            linestyle="None",
                            zorder=5,
                        )

    ax.set_yticks(range(len(CASES)), [CASE_SHORT[c] for c in CASES])
    ax.set_ylim(len(CASES) - 0.55, -0.55)
    ax.set_xlim(lower, upper)
    ax.set_xlabel("nRMSE gain (%)")

    marker_handles = [
        Line2D(
            [0],
            [0],
            marker=baseline_markers[b],
            linestyle="None",
            color="#555555",
            markerfacecolor="#555555",
            markeredgecolor="white",
            markeredgewidth=0.35,
            markersize=3.6,
            label=METHOD_LABELS[b],
        )
        for b in baselines
    ]
    leg = ax.legend(
        handles=marker_handles,
        loc="upper right",
        bbox_to_anchor=(1.01, 1.015),
        ncol=2,
        fontsize=5.55,
        handletextpad=0.25,
        columnspacing=0.65,
        borderpad=0.15,
    )
    leg.set_zorder(10)
    style_ax(ax, numeric_y=False)
    ax.tick_params(axis="y", labelsize=7.0, pad=1.4)

    # Record the compact display encoding for downstream audit/source-data users.
    for baseline, off in baseline_offsets.items():
        plot_rows.append(
            {
                "panel": "e",
                "baseline": METHOD_LABELS[baseline],
                "metric": "baseline_marker_offset",
                "value": float(off),
            }
        )
    return

    y = 0
    yticks: list[float] = []
    ylabels: list[str] = []
    sep_y: list[float] = []
    for ci, case in enumerate(CASES):
        if ci:
            sep_y.append(y - 0.45)
        for baseline in baselines:
            for target, dx in [("pce", -0.06), ("apce", 0.06)]:
                rec = next(r for r in records if r[0] == case and r[1] == target and r[2] == baseline)
                _, _, _, mean, lo, hi = rec
                color = METHOD_COLORS[target]
                ax.hlines(y + dx, lo, hi, color=color, lw=1.0, alpha=0.9)
                ax.plot(mean, y + dx, marker="o", ms=3.2, color=color, mec="white", mew=0.45)
            if ci == 0:
                yticks.append(y)
                ylabels.append(METHOD_LABELS[baseline])
            y += 1
        y += 0.55
    for sy in sep_y:
        ax.axhline(sy, color="#E3E3E3", lw=0.65)
    ax.axvline(0, color="#666666", lw=0.75, ls=(0, (3, 2)))
    ax.set_yticks(yticks, ylabels)
    ax.set_ylim(y - 0.65, -0.7)
    ax.set_xlabel("nRMSE gain (%)")
    for ci, case in enumerate(CASES):
        block_mid = ci * (len(baselines) + 0.55) + (len(baselines) - 1) / 2
        ax.text(
            -0.12,
            block_mid,
            CASE_SHORT[case],
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=7.0,
            color="#333333",
            clip_on=False,
        )
    all_vals = [v for _, _, _, mean, lo, hi in records for v in (lo, hi) if math.isfinite(v)]
    if all_vals:
        # A single Robertson-vs-BMA gain is much larger than the remaining
        # paired effects; cap the display window and mark values beyond it.
        lower = min(-1.25, float(np.nanpercentile(all_vals, 2)) - 0.25)
        upper = max(6.0, float(np.nanpercentile(all_vals, 95)) + 0.55)
        ax.set_xlim(lower, upper)
        for line in ax.lines:
            pass
        # Re-plot any off-scale means as right-edge triangles with their value.
        for case, target, baseline, mean, lo, hi in records:
            if mean <= upper:
                continue
            # Recover the y location.
            case_i = CASES.index(case)
            base_i = baselines.index(baseline)
            target_dx = -0.06 if target == "pce" else 0.06
            yy = case_i * (len(baselines) + 0.55) + base_i + target_dx
            ax.plot(upper, yy, marker=">", ms=4.0, color=METHOD_COLORS[target], mec="white", mew=0.45, clip_on=False, zorder=5)
            ax.text(upper, yy, f"{mean:.0f}", ha="left", va="center", fontsize=6.0, color=METHOD_COLORS[target], clip_on=False)
    style_ax(ax, numeric_y=False)
    ax.set_yticks(yticks, ylabels)
    ax.tick_params(axis="y", labelsize=6.6, pad=1.0)


RADAR_DIMENSIONS = ["Acc.", "Dist.", "Cal.", "Sharp.", "Win", r"$\alpha$"]
RADAR_GROUPS = [("Error", 0, 1, "#E6A4B7"), ("Reliability", 2, 3, "#ABD8B6"), ("Evidence", 4, 5, "#F0D48A")]


def score_low(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    out = np.full(values.shape, 0.5, dtype=float)
    if not finite.any():
        return out
    v = values[finite]
    span = float(v.max() - v.min())
    if span <= 1e-14:
        out[finite] = 1.0
    else:
        out[finite] = 1.0 - (v - v.min()) / span
    return np.clip(out, 0, 1)


def score_target(values: np.ndarray, target: float = 0.9) -> np.ndarray:
    return score_low(np.abs(values - target))


def compute_radar_scores(summary: list[dict[str, str]], runs: list[dict[str, str]]) -> dict[str, np.ndarray]:
    freq_summary = [r for r in summary if int(float(r["obs_interval_factor"])) == SELECTED_FREQ]
    lookup: dict[str, dict[str, float]] = {}
    for method in METHODS:
        sub = [r for r in freq_summary if r["method"] == method and r["case"] in CASES]
        lookup[method] = {
            "nrmse": float(np.mean([fnum(r["nrmse_mean"]) for r in sub])),
            "crps": float(np.mean([fnum(r["crps_mean"]) for r in sub])),
            "coverage": float(np.mean([fnum(r["coverage_90_mean"]) for r in sub])),
            "width": float(np.mean([fnum(r["interval_width_90_mean"]) for r in sub])),
            "alpha": float(np.mean([fnum(r["alpha_absolute_error_mean"]) for r in sub])),
        }

    # Seed-level win rate against all non-self methods at selected freq.
    for method in METHODS:
        wins = []
        for case in CASES:
            method_by_seed = {
                r["seed"]: fnum(r["nrmse"])
                for r in runs
                if int(float(r["obs_interval_factor"])) == SELECTED_FREQ and r["case"] == case and r["method"] == method
            }
            for baseline in METHODS:
                if baseline == method:
                    continue
                base_by_seed = {
                    r["seed"]: fnum(r["nrmse"])
                    for r in runs
                    if int(float(r["obs_interval_factor"])) == SELECTED_FREQ and r["case"] == case and r["method"] == baseline
                }
                paired = set(method_by_seed) & set(base_by_seed)
                if paired:
                    wins.extend([method_by_seed[s] < base_by_seed[s] for s in paired])
        lookup[method]["win"] = float(np.mean(wins)) if wins else 0.0

    matrix = np.asarray(
        [
            [lookup[m]["nrmse"] for m in METHODS],
            [lookup[m]["crps"] for m in METHODS],
            [lookup[m]["coverage"] for m in METHODS],
            [lookup[m]["width"] for m in METHODS],
            [1.0 - lookup[m]["win"] for m in METHODS],  # low means high win rate
            [lookup[m]["alpha"] for m in METHODS],
        ],
        dtype=float,
    )
    scores = np.zeros_like(matrix)
    scores[0] = score_low(matrix[0])
    scores[1] = score_low(matrix[1])
    scores[2] = score_target(matrix[2], 0.9)
    scores[3] = np.sqrt(np.maximum(scores[2], 0) * np.maximum(score_low(matrix[3]), 0))
    scores[4] = score_low(matrix[4])
    scores[5] = score_low(matrix[5])
    return {m: scores[:, i] for i, m in enumerate(METHODS)}


def draw_radar(ax: plt.Axes, scores: dict[str, np.ndarray], plot_rows: list[dict[str, object]]) -> None:
    add_panel_label(ax, "f", "Multi-metric summary", x=-0.14, y=1.12)
    n = len(RADAR_DIMENSIONS)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles_closed = np.r_[angles, angles[0]]
    step = 2 * np.pi / n
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.22)
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.fill(angles_closed, np.ones_like(angles_closed), color="#E6EAED", alpha=0.55, zorder=-4)
    for group_label, start, end, color in RADAR_GROUPS:
        theta_start = angles[start] - step * 0.48
        theta_end = angles[end] + step * 0.48
        theta_center = 0.5 * (theta_start + theta_end)
        ax.bar(theta_center, 0.105, width=theta_end - theta_start, bottom=1.035, color=color, edgecolor="none", alpha=0.88, zorder=-2, clip_on=False)
        rot = -np.degrees(theta_center)
        if rot < -90:
            rot += 180
        if rot > 90:
            rot -= 180
        ax.text(theta_center, 1.086, group_label, fontsize=6.2, rotation=rot, rotation_mode="anchor", ha="center", va="center", clip_on=False)
    for radius in [0.2, 0.4, 0.6, 0.8, 1.0]:
        ax.plot(angles_closed, np.full_like(angles_closed, radius), color="#B6C0C7" if radius < 1 else "#5B7C8B", lw=0.45 if radius < 1 else 0.7, zorder=0)
    for angle in angles:
        ax.plot([angle, angle], [0, 1], color="#B6C0C7", lw=0.45, zorder=0)
    for angle, label in zip(angles, RADAR_DIMENSIONS, strict=True):
        deg = np.degrees(angle)
        ha = "center"
        if 8 < deg < 172:
            ha = "left"
        elif 188 < deg < 352:
            ha = "right"
        ax.text(angle, 1.22, label, fontsize=6.7, ha=ha, va="center", clip_on=False)
    for radius, label in [(0, "0"), (0.5, "0.5"), (1.0, "1")]:
        ax.text(np.deg2rad(0), radius, label, fontsize=5.9, ha="center", va="bottom" if radius else "center")

    # Draw broad baselines first, then PCE/APCE last.
    plot_order = ["denkf", "letkf", "iensf", "aug_enkf", "bma_static", "pce", "apce"]
    for method in plot_order:
        vals = np.r_[scores[method], scores[method][0]]
        for dim, score in zip(RADAR_DIMENSIONS, scores[method], strict=True):
            plot_rows.append(
                {
                    "panel": "f",
                    "case": "all",
                    "freq": SELECTED_FREQ,
                    "method": METHOD_LABELS[method],
                    "metric": f"radar_{dim}",
                    "value": float(score),
                }
            )
        if method == "apce":
            ax.fill(angles_closed, vals, color=METHOD_COLORS[method], alpha=0.15, zorder=4)
            ax.plot(angles_closed, vals, color="#B54708", lw=1.55, zorder=8)
            ax.plot(angles_closed, vals, color=METHOD_COLORS[method], lw=0.85, zorder=9)
        elif method == "pce":
            ax.fill(angles_closed, vals, color=METHOD_COLORS[method], alpha=0.07, zorder=3)
            ax.plot(angles_closed, vals, color=METHOD_COLORS[method], lw=1.10, alpha=0.95, zorder=7)
        else:
            ax.plot(angles_closed, vals, color=METHOD_COLORS[method], lw=0.75, alpha=0.48, zorder=5)


def draw_entropy_trend(ax: plt.Axes, runs: list[dict[str, str]], plot_rows: list[dict[str, object]]) -> None:
    add_panel_label(ax, "g", "Evidence entropy")
    for method in ["pce", "apce"]:
        means: list[float] = []
        los: list[float] = []
        his: list[float] = []
        for freq in FREQS:
            vals = []
            for r in runs:
                if int(float(r["obs_interval_factor"])) != freq or r["method"] != method:
                    continue
                grid_points = fnum(r.get("alpha_grid_points"))
                ent = fnum(r.get("alpha_evidence_entropy"))
                if math.isfinite(ent) and math.isfinite(grid_points) and grid_points > 1:
                    vals.append(ent / math.log(grid_points))
            mean, lo, hi = sem_ci(vals)
            means.append(mean)
            los.append(lo)
            his.append(hi)
            plot_rows.append(
                {
                    "panel": "g",
                    "case": "all",
                    "freq": freq,
                    "method": METHOD_LABELS[method],
                    "metric": "normalized_evidence_entropy",
                    "value": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n": len(vals),
                }
            )
        x = np.asarray(FREQS, dtype=float)
        ax.plot(x, means, color=METHOD_COLORS[method], lw=1.55, marker="o", ms=3.3, label=METHOD_LABELS[method])
        ax.fill_between(x, los, his, color=METHOD_COLORS[method], alpha=0.13, lw=0)
    ax.set_xlim(0.8, 8.2)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(FREQS)
    ax.set_xlabel("Observation interval")
    ax.set_ylabel(r"$H(w)/\log K$")
    ax.text(0.98, 0.10, "lower = concentrated evidence", transform=ax.transAxes, ha="right", va="center", fontsize=6.7, color="#666666")
    style_ax(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(0.00, 1.00), ncol=2, handlelength=1.3, columnspacing=0.9)


def draw_shared_legend(ax: plt.Axes) -> None:
    ax.set_axis_off()
    handles = [
        Line2D([0], [0], marker="s", linestyle="None", ms=5.8, mfc=METHOD_COLORS[m], mec="white", mew=0.35, label=METHOD_LABELS[m])
        for m in METHODS
    ]
    ax.legend(
        handles=handles,
        loc="center",
        ncol=len(METHODS),
        frameon=False,
        fontsize=7.6,
        handlelength=0.8,
        handletextpad=0.35,
        columnspacing=0.85,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_plot_source(rows: list[dict[str, object]]) -> Path:
    path = OUT_DIR / f"{OUT_STEM}_plot_source_data.csv"
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_contract_and_qa(manifest: dict, plot_source: Path, outputs: list[Path]) -> None:
    contract = (__doc__ or "").strip() + "\n"
    (OUT_DIR / f"{OUT_STEM}_contract.md").write_text(contract, encoding="utf-8")
    qa = {
        "figure": OUT_STEM,
        "core_conclusion": "Under progressively sparse observations, PCE/APCE become more favourable because cognitive-path evidence retains parameter information while direct state--parameter covariance updating becomes less reliable.",
        "archetype": "asymmetric mixed-modality figure",
        "backend": "Python/matplotlib only",
        "source_data": {
            "run_csv": str(RUN_CSV),
            "summary_csv": str(SUMMARY_CSV),
            "paired_csv": str(PAIRED_CSV),
            "manifest": str(MANIFEST_JSON),
            "authoritative_run_source_sha256": sha256_file(RUN_CSV),
            "expected_authoritative_sha256": "b2bb121a219252476d9c6bb3eb7f4c9f5c062cdf605be1f71aa6d53e5d9c6684",
            "manifest_records": manifest.get("records") or manifest.get("run_records"),
            "manifest_valid_records": manifest.get("valid_records"),
        },
        "layout": {
            "a": "accepted five-case hero atlas reused as raster asset",
            "b": "best PCE/APCE alpha MAE heatmap across intervals",
            "c": f"nRMSE boxplots at interval {SELECTED_FREQ}",
            "d": f"seed-wise nRMSE gain versus Aug-EnKF at interval {SELECTED_FREQ}",
            "e": f"paired forest against all baselines at interval {SELECTED_FREQ}",
            "f": f"multi-metric radar summary at interval {SELECTED_FREQ}",
            "g": "normalized evidence entropy trend over intervals",
        },
        "qa_checks": {
            "uses_authoritative_14000_source": True,
            "does_not_overwrite_panel_a": True,
            "no_internal_version_suffixes_in_labels": True,
            "shared_method_colours": True,
            "svg_text_editable": True,
            "pdf_fonttype_42": True,
            "crps_not_main_standalone_panel": True,
        },
        "plot_source_data": str(plot_source),
        "outputs": [str(p) for p in outputs],
    }
    (OUT_DIR / f"{OUT_STEM}_qa.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    apply_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs_raw, summary, _paired, manifest = load_data()
    runs = valid_runs(runs_raw)
    expected = len(CASES) * len(METHODS) * len(FREQS) * N_SEEDS
    if len(runs) != expected:
        raise RuntimeError(f"Expected {expected} valid records, found {len(runs)}.")
    if sha256_file(RUN_CSV) != "b2bb121a219252476d9c6bb3eb7f4c9f5c062cdf605be1f71aa6d53e5d9c6684":
        raise RuntimeError("Authoritative run-source SHA256 mismatch.")

    plot_rows: list[dict[str, object]] = []
    fig = plt.figure(figsize=(7.35, 8.45), dpi=180, facecolor="white")
    outer = fig.add_gridspec(4, 1, height_ratios=[1.02, 0.10, 1.05, 1.23], hspace=0.31)
    ax_a = fig.add_subplot(outer[0, 0])
    ax_leg = fig.add_subplot(outer[1, 0])
    mid = outer[2, 0].subgridspec(1, 3, width_ratios=[1.05, 1.40, 1.25], wspace=0.42)
    bot = outer[3, 0].subgridspec(1, 3, width_ratios=[1.35, 1.05, 1.20], wspace=0.50)

    draw_panel_a(ax_a)
    draw_shared_legend(ax_leg)
    draw_alpha_error_heatmap(fig.add_subplot(mid[0, 0]), runs, plot_rows)
    draw_nrmse_boxplots(fig.add_subplot(mid[0, 1]), runs, plot_rows)
    draw_gain_swarm(fig.add_subplot(mid[0, 2]), runs, plot_rows)
    draw_grouped_forest(fig.add_subplot(bot[0, 0]), runs, plot_rows)
    draw_radar(fig.add_subplot(bot[0, 1], projection="polar"), compute_radar_scores(summary, runs), plot_rows)
    draw_entropy_trend(fig.add_subplot(bot[0, 2]), runs, plot_rows)

    fig.subplots_adjust(left=0.055, right=0.988, top=0.985, bottom=0.055)

    outputs: list[Path] = []
    for ext, kwargs in [
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 600}),
        ("tiff", {"dpi": 600}),
    ]:
        out = OUT_DIR / f"{OUT_STEM}.{ext}"
        fig.savefig(out, bbox_inches="tight", **kwargs)
        outputs.append(out)
    plt.close(fig)
    plot_source = write_plot_source(plot_rows)
    write_contract_and_qa(manifest, plot_source, outputs)
    print(json.dumps({"outputs": [str(p) for p in outputs], "plot_source": str(plot_source)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
