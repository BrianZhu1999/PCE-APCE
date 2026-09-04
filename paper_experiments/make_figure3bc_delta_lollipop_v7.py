from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = Path(r".\hybrid_uncertain_wave")
SOURCE_CSV = PROJECT_ROOT / "source_data" / "figure3_selected5_freq1to8_formal_50seeds_allmethods_20260813" / "combined" / "source_data" / "figure3_freq1to8_formal_authoritative_run_source_data.csv"
OUT_DIR = Path(r"figures")
OUT_STEM = "figure3bc_delta_lollipop_v19_height130"

PANEL_LABEL_SIZE = 22
CASE_TITLE_SIZE = 16
LEGEND_SIZE = 16
AXIS_LABEL_SIZE = 14
TICK_SIZE = 11

FIG_W = 15.56
FIG_H = 3.62
WHITE = "white"
TEXT = "#111111"

CASES = ["pendulum", "fhn", "robertson"]
CASE_LABELS = {
    "pendulum": "Forced",
    "fhn": "FHN",
    "robertson": "Robertson",
}
BASELINES = ["aug_enkf", "bma"]
METHOD_LABELS = {"aug_enkf": "Aug-EnKF", "bma": "BMA"}
METHOD_COLORS = {"aug_enkf": "#4e79a7", "bma": "#9b75b6"}
REF_METHODS = ["pce", "apce"]


plt.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 600,
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "axes.unicode_minus": False,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.9,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def method_key(raw: str) -> str:
    s = raw.strip().lower().replace("-", "_")
    if s in {"augenkf", "aug_enkf"}:
        return "aug_enkf"
    if s in {"bma_static", "static_bma"}:
        return "bma"
    return s


def load_metric_means(metric: str) -> dict[tuple[str, int, str], float]:
    keep_methods = set(BASELINES + REF_METHODS)
    values: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    with SOURCE_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("status", "").strip().lower() != "completed":
                continue
            case = row["case"].strip().lower()
            method = method_key(row["method"])
            if case not in CASES or method not in keep_methods:
                continue
            try:
                freq = int(float(row["obs_interval_factor"]))
                val = float(row[metric])
            except Exception:
                continue
            if metric == "nrmse":
                val *= 100.0
            if 1 <= freq <= 8 and np.isfinite(val):
                values[(case, freq, method)].append(val)
    return {k: float(np.mean(v)) for k, v in values.items() if v}


def baseline_gaps(means: dict[tuple[str, int, str], float]) -> dict[tuple[str, int, str], float]:
    out: dict[tuple[str, int, str], float] = {}
    for case in CASES:
        for freq in range(1, 9):
            refs = [means.get((case, freq, m), np.nan) for m in REF_METHODS]
            ref = np.nanmin(refs)
            if not np.isfinite(ref):
                continue
            for method in BASELINES:
                val = means.get((case, freq, method), np.nan)
                if np.isfinite(val):
                    out[(case, freq, method)] = val - ref
    return out


def nice_ylim(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return -0.05, 1.0
    lo = min(float(np.min(arr)), 0.0)
    hi = max(float(np.max(arr)), 0.0)
    span = max(hi - lo, max(abs(hi), 1.0) * 0.08)
    pad = span * 0.14
    return lo - pad, hi + pad


def zero_aligned_positive_ylim(vals: list[float], zero_fraction: float) -> tuple[float, float]:
    arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return -0.001, 0.004
    hi = max(float(np.max(arr)), 0.0)
    pad = max(hi * 0.16, 0.00018)
    ymax = hi + pad
    ymin = -zero_fraction * ymax / max(1.0 - zero_fraction, 1e-6)
    return ymin, ymax


def draw_lollipop_panel(
    ax: plt.Axes,
    case: str,
    gaps: dict[tuple[str, int, str], float],
    ylabel: str | None,
    metric: str,
) -> None:
    x = np.arange(1, 9, dtype=float)
    offsets = {"aug_enkf": -0.095, "bma": 0.095}
    ax.axhline(0, color="#151515", lw=1.05, alpha=0.88, zorder=1)

    for method in BASELINES:
        xs = x + offsets[method]
        ys = np.array([gaps.get((case, f, method), np.nan) for f in range(1, 9)], dtype=float)
        color = METHOD_COLORS[method]
        for xi, yi in zip(xs, ys):
            if np.isfinite(yi):
                ax.vlines(xi, 0, yi, color=color, lw=2.15, alpha=0.82, zorder=2)
        ax.scatter(xs, ys, s=31, facecolor=WHITE, edgecolor=color, linewidth=1.55, alpha=0.98, zorder=3)

    ax.set_title(CASE_LABELS[case], fontsize=CASE_TITLE_SIZE, pad=8)
    ax.set_xlim(0.55, 8.45)
    ax.set_xticks(x)
    ax.set_xlabel("Obs. interval", fontsize=AXIS_LABEL_SIZE, labelpad=5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE, labelpad=5)
    else:
        ax.tick_params(labelleft=False)
    ax.tick_params(labelsize=TICK_SIZE, width=0.9, length=3)
    if metric == "crps":
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * 1e3:g}"))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def main() -> None:
    nrmse_gaps = baseline_gaps(load_metric_means("nrmse"))
    crps_gaps = baseline_gaps(load_metric_means("crps"))

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=WHITE)
    left, right = 0.052, 0.988
    gap = 0.012
    group_gap = 0.070
    panel_w = (right - left - 4 * gap - group_gap) / 6
    bottom, panel_h = 0.130, 0.579
    x_positions = []
    x0 = left
    for i in range(6):
        x_positions.append(x0)
        x0 += panel_w + (group_gap if i == 2 else gap)
    axes = [fig.add_axes([x, bottom, panel_w, panel_h], facecolor=WHITE) for x in x_positions]

    for ci, case in enumerate(CASES):
        draw_lollipop_panel(axes[ci], case, nrmse_gaps, r"$\Delta$ nRMSE (%)" if ci == 0 else None, "nrmse")
    for ci, case in enumerate(CASES):
        draw_lollipop_panel(axes[ci + 3], case, crps_gaps, r"$\Delta$ CRPS ($10^{-3}$)" if ci == 0 else None, "crps")

    nrmse_ylim = nice_ylim([nrmse_gaps.get((case, f, m), np.nan) for case in CASES for f in range(1, 9) for m in BASELINES])
    zero_fraction = (0.0 - nrmse_ylim[0]) / (nrmse_ylim[1] - nrmse_ylim[0])
    crps_ylim = zero_aligned_positive_ylim(
        [crps_gaps.get((case, f, m), np.nan) for case in CASES for f in range(1, 9) for m in BASELINES],
        zero_fraction,
    )
    for ax in axes[:3]:
        ax.set_ylim(*nrmse_ylim)
    for ax in axes[3:]:
        ax.set_ylim(*crps_ylim)

    label_y = 0.815
    fig.text(0.006, label_y, "b", fontsize=PANEL_LABEL_SIZE, fontweight="bold", ha="left", va="top", color=TEXT)
    fig.text(x_positions[3] - 0.036, label_y, "c", fontsize=PANEL_LABEL_SIZE, fontweight="bold", ha="left", va="top", color=TEXT)

    handles = [
        Line2D(
            [0], [0],
            color=METHOD_COLORS[m],
            lw=2.15,
            marker="o",
            ms=5.6,
            mfc=WHITE,
            mec=METHOD_COLORS[m],
            mew=1.55,
            label=METHOD_LABELS[m],
            alpha=0.92,
        )
        for m in BASELINES
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.500, 0.975),
        ncol=2,
        frameon=False,
        prop={"family": "Arial", "size": LEGEND_SIZE},
        handlelength=1.25,
        handletextpad=0.42,
        columnspacing=1.15,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in [("png", {"dpi": 600}), ("pdf", {}), ("svg", {}), ("tiff", {"dpi": 600})]:
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", facecolor=WHITE, bbox_inches=None, pad_inches=0, **kwargs)
    plt.close(fig)
    print(OUT_DIR / f"{OUT_STEM}.png")


if __name__ == "__main__":
    main()
