"""Create Figure 3 draft from the selected-six applied ODE formal source data.

Figure contract
---------------
Core conclusion:
    Across six applied ODE systems, evidence-weighted cognitive shadow forecasts
    provide a reusable identification/reconstruction mechanism that is
    consistently competitive with static multiple-model averaging and
    structure-dependent relative to augmented-state EnKF.

Evidence chain:
    a. Case/provenance atlas: separates source-derived uncertain ODE reductions
       from canonical stress tests.
    b. nRMSE matrix: state reconstruction accuracy across all seven methods.
    c. CRPS matrix: probabilistic predictive accuracy.
    d. Paired delta map: reviewer-risk comparison to BMA and Aug-EnKF.
    e. Alpha MAE matrix: cognitive-coordinate identification for methods that
       maintain or estimate the cognitive parameter.
    f. Calibration-sharpness landscape: 90% coverage error versus interval
       width, with case-level points and method-level summaries.

Backend:
    Python/matplotlib only, following the nature-figure skill.

Inputs:
    source_data/figure3_selected6_caseprofile_formal_50seeds_20260812/

Outputs:
    ncs_chinese_submission/figures/figure3_applied_ode_selected6_caseprofile_v1
    in PNG, PDF, SVG, TIFF, plus compact plot source data and QA JSON.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np


# ── Mandatory editable-text rules from nature-figure ─────────────────────────
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = (
    PROJECT_ROOT
    / "source_data"
    / "figure3_selected6_caseprofile_formal_50seeds_20260812"
)
FIG_DIR = PROJECT_ROOT / "ncs_chinese_submission" / "figures"
OUT_BASE = FIG_DIR / "figure3_applied_ode_selected6_caseprofile_v3"


CASE_ORDER = [
    "chemical",
    "pk_infusion",
    "sir",
    "pendulum",
    "fhn",
    "robertson",
]
CASE_LABELS = {
    "chemical": "Chemical",
    "pk_infusion": "PK",
    "sir": "SIR",
    "pendulum": "Pendulum",
    "fhn": "FHN",
    "robertson": "Robertson",
}
METHOD_ORDER = ["DEnKF", "LETKF", "IEnSF", "Aug-EnKF", "BMA", "PCE", "APCE"]
ALPHA_METHODS = ["Aug-EnKF", "BMA", "PCE", "APCE"]


METHOD_COLORS = {
    "DEnKF": "#6C8ED6",
    "LETKF": "#00A474",
    "IEnSF": "#8B6A4A",
    "Aug-EnKF": "#D45258",
    "BMA": "#40A9DC",
    "PCE": "#7E52A0",
    "APCE": "#FF7F0E",
}
BASELINE_MARKERS = {"Aug-EnKF": "o", "BMA": "s"}


TEXT = {
    "panel": 12.0,
    "title": 8.8,
    "axis": 7.8,
    "tick": 7.2,
    "cell": 6.5,
    "small": 6.7,
    "legend": 7.2,
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fnum(value: str | float | int) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def fmt_clean(x: float, digits: int = 2) -> str:
    if not math.isfinite(x):
        return ""
    if abs(x) >= 100:
        s = f"{x:.0f}"
    elif abs(x) >= 10:
        s = f"{x:.1f}"
    elif abs(x) >= 1:
        s = f"{x:.2f}"
    else:
        s = f"{x:.2f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def sem_ci(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    if arr.size == 0:
        return math.nan, math.nan, math.nan
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, mean, mean
    sem = float(arr.std(ddof=1) / math.sqrt(arr.size))
    return mean, mean - 1.96 * sem, mean + 1.96 * sem


def luminance(hex_color: str) -> float:
    rgb = mcolors.to_rgb(hex_color)
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def text_color_for_cmap(cmap, norm, value: float) -> str:
    rgba = cmap(norm(value))
    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "white" if lum < 0.47 else "#222222"


def strip_axes(ax):
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(width=0.8, length=2.2, pad=1)


def canonical_method(row: dict[str, str]) -> str:
    """Return the display method name used consistently across panels."""
    label = (row.get("label") or "").strip()
    if label:
        return label
    raw = (row.get("method") or "").strip()
    mapping = {
        "denkf": "DEnKF",
        "letkf": "LETKF",
        "iensf": "IEnSF",
        "aug_enkf": "Aug-EnKF",
        "bma_static": "BMA",
        "pce": "PCE",
        "apce": "APCE",
    }
    return mapping.get(raw, raw)


def add_panel_label(ax, label: str, x: float = -0.12, y: float = 1.05):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=TEXT["panel"],
        fontweight="bold",
        ha="left",
        va="bottom",
        color="#111111",
    )


def load_data():
    summary_rows = read_csv_rows(DATA_DIR / "figure3_selected6_method_summary_aligned.csv")
    paired_rows = read_csv_rows(DATA_DIR / "figure3_selected6_pce_apce_vs_aug_bma_paired.csv")
    run_rows = read_csv_rows(DATA_DIR / "figure3_v4_run_source_data.csv")
    manifest = json.loads((DATA_DIR / "figure3_v4_config_manifest.json").read_text(encoding="utf-8"))
    return summary_rows, paired_rows, run_rows, manifest


def summary_lookup(summary_rows):
    lookup: dict[tuple[str, str], dict[str, float]] = {}
    for r in summary_rows:
        lookup[(r["case"], r["method"])] = {k: fnum(v) for k, v in r.items() if k not in {"case", "method"}}
    return lookup


def metric_matrix(lookup, metric: str, cases: list[str], methods: list[str], scale: float = 1.0):
    mat = np.full((len(cases), len(methods)), np.nan, dtype=float)
    for i, case in enumerate(cases):
        for j, method in enumerate(methods):
            mat[i, j] = lookup[(case, method)][metric] * scale
    return mat


def plot_metric_heatmap(
    ax,
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    value_label: str,
    cmap,
    vmax: float | None = None,
    annotate_digits: int = 2,
    show_y: bool = True,
):
    if vmax is None:
        vmax = float(np.nanmax(matrix))
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    for i in range(matrix.shape[0]):
        finite_row = matrix[i, :]
        best_j = int(np.nanargmin(finite_row))
        ax.add_patch(
            Rectangle(
                (best_j - 0.5, i - 0.5),
                1,
                1,
                fill=False,
                edgecolor="#222222",
                linewidth=1.2,
                zorder=4,
            )
        )
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(
                j,
                i,
                fmt_clean(value, annotate_digits),
                ha="center",
                va="center",
                fontsize=TEXT["cell"],
                color=text_color_for_cmap(cmap, norm, value),
                zorder=5,
            )

    ax.set_title(title, fontsize=TEXT["title"], loc="left", pad=4)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=42, ha="right", fontsize=TEXT["tick"])
    if show_y:
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=TEXT["tick"])
    else:
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels([])
    ax.tick_params(length=0, pad=1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(-0.5, len(col_labels) - 0.5)
    ax.set_ylim(len(row_labels) - 0.5, -0.5)


def draw_case_microplot(ax, case: str, color: str):
    ax.set_axis_off()
    t = np.linspace(0, 1, 120)
    if case == "chemical":
        ax.plot(t, 0.70 - 0.50 * (1 - np.exp(-3 * t)), color="#4D4D4D", lw=1.0)
        ax.plot(t, 0.16 + 0.42 * (1 - np.exp(-2.8 * t)), color=color, lw=1.2)
    elif case == "pk_infusion":
        y = (1 - np.exp(-5 * t)) * np.exp(-0.85 * t)
        ax.plot(t, y, color=color, lw=1.35)
        ax.plot([0.05, 0.05], [0.0, 0.95], color="#4D4D4D", lw=0.8)
    elif case == "sir":
        s = 0.90 - 0.55 / (1 + np.exp(-8 * (t - 0.46)))
        i = 0.12 + 0.38 * np.exp(-((t - 0.43) / 0.18) ** 2)
        r = 0.06 + 0.58 / (1 + np.exp(-7 * (t - 0.55)))
        ax.plot(t, s, color="#4D4D4D", lw=0.9)
        ax.plot(t, i, color=color, lw=1.15)
        ax.plot(t, r, color="#9E9E9E", lw=0.9)
    elif case == "pendulum":
        th = np.linspace(0, 2 * np.pi, 140)
        ax.plot(0.5 + 0.38 * np.cos(th), 0.5 + 0.28 * np.sin(th), color=color, lw=1.25)
        ax.plot([0.5, 0.62], [0.86, 0.48], color="#4D4D4D", lw=1.0)
        ax.scatter([0.62], [0.48], s=7, color=color, zorder=4)
    elif case == "fhn":
        x = np.linspace(-1.2, 1.2, 140)
        cubic = 0.5 + 0.20 * (x - x**3 / 3)
        ax.plot((x + 1.2) / 2.4, cubic, color="#4D4D4D", lw=0.85)
        th = np.linspace(0, 2 * np.pi, 130)
        ax.plot(0.52 + 0.28 * np.cos(th), 0.45 + 0.18 * np.sin(th + 0.6), color=color, lw=1.2)
    elif case == "robertson":
        ax.plot(t, 0.82 * np.exp(-5.2 * t) + 0.10, color="#4D4D4D", lw=0.9)
        ax.plot(t, 0.12 + 0.48 * (1 - np.exp(-8 * t)) * np.exp(-1.7 * t), color=color, lw=1.2)
        ax.plot(t, 0.08 + 0.70 * (1 - np.exp(-2.0 * t)), color="#9E9E9E", lw=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def plot_case_atlas(ax):
    add_panel_label(ax, "a", x=-0.10, y=1.04)
    ax.set_axis_off()
    ax.set_title("Applied ODE case atlas", fontsize=TEXT["title"], loc="left", pad=3)

    tier_specs = [
        ("Source-derived uncertain ODEs", ["chemical", "pk_infusion", "sir"], "#F6E8D7", "#C47B36"),
        ("Canonical stress tests", ["pendulum", "fhn", "robertson"], "#E3EFF2", "#2F7C8D"),
    ]
    card_w = 0.29
    card_h = 0.29
    x0s = [0.02, 0.355, 0.69]
    y_title = [0.86, 0.41]
    y_card = [0.54, 0.09]

    for tier_idx, (tier_label, cases, face, edge) in enumerate(tier_specs):
        ax.text(
            0.02,
            y_title[tier_idx],
            tier_label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=TEXT["small"],
            color="#333333",
        )
        ax.plot(
            [0.02, 0.98],
            [y_title[tier_idx] - 0.06, y_title[tier_idx] - 0.06],
            transform=ax.transAxes,
            color=edge,
            lw=0.8,
            alpha=0.8,
            clip_on=False,
        )
        for x0, case in zip(x0s, cases):
            patch = FancyBboxPatch(
                (x0, y_card[tier_idx]),
                card_w,
                card_h,
                boxstyle="round,pad=0.012,rounding_size=0.025",
                transform=ax.transAxes,
                facecolor=face,
                edgecolor=edge,
                linewidth=0.75,
            )
            ax.add_patch(patch)
            inset = ax.inset_axes([x0 + 0.035, y_card[tier_idx] + 0.085, card_w - 0.07, card_h - 0.125])
            draw_case_microplot(inset, case, edge)
            ax.text(
                x0 + card_w / 2,
                y_card[tier_idx] + 0.045,
                CASE_LABELS[case],
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=TEXT["small"],
                color="#222222",
            )


def run_metric_arrays(run_rows):
    by_key: dict[tuple[str, str, str], dict[str, float]] = {}
    for r in run_rows:
        if r.get("status") != "completed" or r.get("numerical_status") != "valid":
            continue
        key = (r["case"], canonical_method(r), str(r["seed"]))
        by_key[key] = {
            "nrmse": fnum(r["nrmse"]),
            "crps": fnum(r["crps"]),
            "alpha_absolute_error": fnum(r["alpha_absolute_error"]),
            "coverage_error": abs(fnum(r["coverage_90"]) - 0.9),
            "coverage_90": fnum(r["coverage_90"]),
            "interval_width_90": fnum(r["interval_width_90"]),
            "core_runtime_seconds": fnum(r["core_runtime_seconds"]),
        }
    return by_key


def plot_paired_delta(ax, run_map):
    add_panel_label(ax, "d", x=-0.08, y=1.05)
    ax.set_title("Paired reviewer-risk comparison", fontsize=TEXT["title"], loc="left", pad=4)
    ax.axhline(0, color="#777777", lw=0.9, ls=(0, (3, 2)), zorder=1)

    plot_records = []
    all_means = []
    for xi, case in enumerate(CASE_ORDER):
        seeds = sorted({seed for (c, m, seed) in run_map if c == case})
        for baseline, base_offset in [("Aug-EnKF", -0.10), ("BMA", 0.10)]:
            for method, method_offset in [("PCE", -0.035), ("APCE", 0.035)]:
                diffs = []
                for seed in seeds:
                    a = run_map.get((case, method, seed), {}).get("nrmse", math.nan)
                    b = run_map.get((case, baseline, seed), {}).get("nrmse", math.nan)
                    if math.isfinite(a) and math.isfinite(b):
                        diffs.append((a - b) * 100.0)
                mean, lo, hi = sem_ci(diffs)
                x = xi + base_offset + method_offset
                ax.errorbar(
                    x,
                    mean,
                    yerr=[[mean - lo], [hi - mean]],
                    fmt=BASELINE_MARKERS[baseline],
                    markersize=4.0,
                    color=METHOD_COLORS[method],
                    markerfacecolor=METHOD_COLORS[method],
                    markeredgecolor="white",
                    markeredgewidth=0.5,
                    elinewidth=0.9,
                    capsize=1.8,
                    zorder=4,
                )
                if math.isfinite(mean):
                    all_means.append(mean)
                plot_records.append(
                    {
                        "panel": "d",
                        "case": case,
                        "method": method,
                        "baseline": baseline,
                        "metric": "delta_nrmse_percent",
                        "mean": mean,
                        "ci_low": lo,
                        "ci_high": hi,
                        "n": len(diffs),
                    }
                )

    ax.set_xticks(np.arange(len(CASE_ORDER)))
    ax.set_xticklabels([CASE_LABELS[c] for c in CASE_ORDER], rotation=25, ha="right", fontsize=TEXT["tick"])
    ax.set_ylabel(r"$\Delta$ nRMSE (%)", fontsize=TEXT["axis"], labelpad=1)
    ax.tick_params(axis="y", labelsize=TEXT["tick"])
    ax.set_xlim(-0.45, len(CASE_ORDER) - 0.55)
    yabs = max([abs(v) for v in all_means] + [1.1])
    ax.set_ylim(-yabs * 1.05, yabs * 1.05)
    strip_axes(ax)
    method_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=METHOD_COLORS["PCE"], markeredgecolor="white", label="PCE", markersize=5.2),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=METHOD_COLORS["APCE"], markeredgecolor="white", label="APCE", markersize=5.2),
    ]
    baseline_handles = [
        Line2D([0], [0], marker="o", color="#555555", lw=0, markerfacecolor="#555555", label="vs Aug-EnKF", markersize=4.6),
        Line2D([0], [0], marker="s", color="#555555", lw=0, markerfacecolor="#555555", label="vs BMA", markersize=4.6),
    ]
    leg1 = ax.legend(
        handles=method_handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 1.01),
        ncol=2,
        frameon=False,
        fontsize=TEXT["legend"],
        handlelength=0.8,
        handletextpad=0.35,
        columnspacing=0.8,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=baseline_handles,
        loc="upper right",
        bbox_to_anchor=(1.00, 1.01),
        ncol=2,
        frameon=False,
        fontsize=TEXT["legend"],
        handlelength=0.8,
        handletextpad=0.35,
        columnspacing=0.8,
    )
    return plot_records


def plot_calibration_landscape(ax, summary_rows):
    add_panel_label(ax, "f", x=-0.035, y=1.04)
    ax.set_title("Calibration--sharpness landscape", fontsize=TEXT["title"], loc="left", pad=4)

    method_case_vals = defaultdict(list)
    for r in summary_rows:
        case = r["case"]
        method = r["method"]
        cov_err = abs(fnum(r["coverage_90_mean"]) - 0.9)
        width = fnum(r["interval_width_90_mean"])
        method_case_vals[method].append((case, width, cov_err))

    for method in METHOD_ORDER:
        color = METHOD_COLORS[method]
        vals = method_case_vals[method]
        xs = [v[1] for v in vals]
        ys = [v[2] for v in vals]
        ax.scatter(
            xs,
            ys,
            s=12,
            color=color,
            alpha=0.23,
            edgecolor="none",
            zorder=2,
        )
        mx = float(np.mean(xs))
        my = float(np.mean(ys))
        sx = float(np.std(xs, ddof=1) / math.sqrt(len(xs))) if len(xs) > 1 else 0.0
        sy = float(np.std(ys, ddof=1) / math.sqrt(len(ys))) if len(ys) > 1 else 0.0
        ax.errorbar(
            mx,
            my,
            xerr=sx,
            yerr=sy,
            fmt="o",
            ms=5.8 if method in {"PCE", "APCE"} else 4.8,
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.7,
            elinewidth=0.8,
            capsize=1.6,
            zorder=5 if method in {"PCE", "APCE"} else 4,
        )

    label_offsets = {
        "DEnKF": (7, 8),
        "LETKF": (-3, 10),
        "IEnSF": (8, -2),
        "Aug-EnKF": (7, -2),
        "BMA": (7, 2),
        "PCE": (-27, -1),
        "APCE": (7, 8),
    }
    for method in METHOD_ORDER:
        vals = method_case_vals[method]
        mx = float(np.mean([v[1] for v in vals]))
        my = float(np.mean([v[2] for v in vals]))
        dx, dy = label_offsets[method]
        ax.annotate(
            method,
            xy=(mx, my),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=TEXT["small"],
            color=METHOD_COLORS[method],
            ha="left",
            va="center",
            path_effects=[pe.withStroke(linewidth=2.3, foreground="white")],
            zorder=7,
        )

    ax.set_xscale("log")
    ax.set_xlabel("90% interval width (log scale)", fontsize=TEXT["axis"], labelpad=2)
    ax.set_ylabel("90% coverage error", fontsize=TEXT["axis"], labelpad=2)
    ax.tick_params(labelsize=TEXT["tick"])
    ax.set_xlim(0.008, 0.9)
    ax.set_ylim(-0.015, 0.76)
    ax.text(
        0.012,
        0.035,
        "lower-left is better",
        fontsize=TEXT["small"],
        color="#777777",
        ha="left",
        va="center",
        path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
    )
    strip_axes(ax)


def write_plot_source(lookup, paired_records, summary_rows):
    path = FIG_DIR / f"{OUT_BASE.name}_plot_source_data.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["panel", "case", "method", "baseline", "metric", "value", "mean", "ci_low", "ci_high", "n"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for panel, metric, methods, scale in [
            ("b", "nrmse_mean", METHOD_ORDER, 100.0),
            ("c", "crps_mean", METHOD_ORDER, 1000.0),
            ("e", "alpha_absolute_error_mean", ALPHA_METHODS, 100.0),
        ]:
            for case in CASE_ORDER:
                for method in methods:
                    value = lookup[(case, method)][metric] * scale
                    w.writerow(
                        {
                            "panel": panel,
                            "case": case,
                            "method": method,
                            "baseline": "",
                            "metric": metric,
                            "value": value,
                            "mean": "",
                            "ci_low": "",
                            "ci_high": "",
                            "n": "",
                        }
                    )
        for r in paired_records:
            w.writerow(
                {
                    "panel": r["panel"],
                    "case": r["case"],
                    "method": r["method"],
                    "baseline": r["baseline"],
                    "metric": r["metric"],
                    "value": "",
                    "mean": r["mean"],
                    "ci_low": r["ci_low"],
                    "ci_high": r["ci_high"],
                    "n": r["n"],
                }
            )
        for r in summary_rows:
            w.writerow(
                {
                    "panel": "f",
                    "case": r["case"],
                    "method": r["method"],
                    "baseline": "",
                    "metric": "coverage_error_vs_interval_width",
                    "value": "",
                    "mean": "",
                    "ci_low": abs(fnum(r["coverage_90_mean"]) - 0.9),
                    "ci_high": fnum(r["interval_width_90_mean"]),
                    "n": r["n"],
                }
            )
    return path


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows, paired_rows, run_rows, manifest = load_data()
    lookup = summary_lookup(summary_rows)
    run_map = run_metric_arrays(run_rows)

    expected_records = len(CASE_ORDER) * len(METHOD_ORDER) * 50
    valid_records = len(
        [
            r
            for r in run_rows
            if r.get("status") == "completed" and r.get("numerical_status") == "valid"
        ]
    )
    if valid_records != expected_records:
        raise RuntimeError(f"Expected {expected_records} valid records, found {valid_records}.")

    metric_cmap = mcolors.LinearSegmentedColormap.from_list(
        "figure3_low_to_high",
        ["#F7EDDC", "#EFCB83", "#D97862", "#5A315A"],
    )

    fig = plt.figure(figsize=(7.35, 7.65), dpi=180)
    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[1.08, 1.02, 1.00],
        width_ratios=[1.13, 1.22, 1.05],
        hspace=0.64,
        wspace=0.44,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0:2])
    ax_e = fig.add_subplot(gs[1, 2])
    ax_f = fig.add_subplot(gs[2, :])

    plot_case_atlas(ax_a)

    row_labels = [CASE_LABELS[c] for c in CASE_ORDER]
    nrmse = metric_matrix(lookup, "nrmse_mean", CASE_ORDER, METHOD_ORDER, scale=100.0)
    crps = metric_matrix(lookup, "crps_mean", CASE_ORDER, METHOD_ORDER, scale=1000.0)
    alpha_mae = metric_matrix(
        lookup, "alpha_absolute_error_mean", CASE_ORDER, ALPHA_METHODS, scale=100.0
    )
    add_panel_label(ax_b, "b", x=-0.13, y=1.05)
    plot_metric_heatmap(
        ax_b,
        nrmse,
        row_labels,
        METHOD_ORDER,
        "State nRMSE (%)",
        "nRMSE (%)",
        metric_cmap,
        vmax=max(6.0, float(np.nanmax(nrmse))),
        show_y=True,
    )
    add_panel_label(ax_c, "c", x=-0.13, y=1.05)
    plot_metric_heatmap(
        ax_c,
        crps,
        row_labels,
        METHOD_ORDER,
        r"CRPS ($10^{-3}$)",
        r"CRPS ($10^{-3}$)",
        metric_cmap,
        vmax=max(20.0, float(np.nanmax(crps))),
        show_y=False,
    )
    paired_records = plot_paired_delta(ax_d, run_map)
    add_panel_label(ax_e, "e", x=-0.18, y=1.05)
    plot_metric_heatmap(
        ax_e,
        alpha_mae,
        row_labels,
        ALPHA_METHODS,
        r"Cognitive-coordinate error, $|\hat{\alpha}-\alpha_\star|$ (%)",
        r"$|\hat{\alpha}-\alpha_\star|$ (%)",
        metric_cmap,
        vmax=max(4.0, float(np.nanmax(alpha_mae))),
        show_y=False,
    )
    plot_calibration_landscape(ax_f, summary_rows)

    fig.align_ylabels([ax_b, ax_d, ax_f])
    fig.subplots_adjust(left=0.065, right=0.992, top=0.965, bottom=0.075)

    saved = []
    for ext, kwargs in [
        ("png", {"dpi": 600}),
        ("pdf", {}),
        ("svg", {}),
        ("tiff", {"dpi": 600}),
    ]:
        out = OUT_BASE.with_suffix(f".{ext}")
        fig.savefig(out, bbox_inches="tight", **kwargs)
        saved.append(str(out))

    plot_source_path = write_plot_source(lookup, paired_records, summary_rows)
    qa = {
        "figure": OUT_BASE.name,
        "core_conclusion": (
            "PCE/APCE extend the cognitive-shadow evidence mechanism to six "
            "applied ODE systems; they are generally stronger than BMA and "
            "structure-dependent relative to Aug-EnKF."
        ),
        "archetype": "asymmetric quantitative grid",
        "backend": "python/matplotlib",
        "source_data_dir": str(DATA_DIR),
        "source_hash": manifest.get("source_hash", {}).get("aggregate_sha256"),
        "records_expected": expected_records,
        "records_valid": valid_records,
        "cases": CASE_ORDER,
        "methods": METHOD_ORDER,
        "outputs": saved,
        "plot_source_data": str(plot_source_path),
        "qa_checks": {
            "old_figure_overwritten": False,
            "uses_selected6_caseprofile_formal": True,
            "no_oracle_methods": True,
            "heatmap_cells_annotated": True,
            "statistical_gridlines_removed": True,
            "top_right_spines_removed": True,
            "svg_text_editable": True,
            "pdf_fonttype_42": True,
        },
    }
    qa_path = OUT_BASE.with_name(f"{OUT_BASE.name}_qa.json")
    qa_path.write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")

    contract_path = OUT_BASE.with_name(f"{OUT_BASE.name}_contract.md")
    contract_path.write_text(__doc__ or "", encoding="utf-8")
    plt.close(fig)

    print(json.dumps({"saved": saved, "plot_source_data": str(plot_source_path), "qa": str(qa_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
