from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(r".\hybrid_uncertain_wave")
SOURCE_CSV = PROJECT_ROOT / "source_data" / "figure3_selected5_freq1to8_formal_50seeds_allmethods_20260813" / "combined" / "source_data" / "figure3_freq1to8_formal_authoritative_run_source_data.csv"
OUT_DIR = Path(r"figures")
OUT_STEM = "figure3bc_metric_row_v1"


PANEL_LABEL_SIZE = 26
CASE_TITLE_SIZE = 18
LEGEND_SIZE = 18
AXIS_LABEL_SIZE = 16
TICK_SIZE = 11

FIG_W = 15.56
FIG_H = 6.20
WHITE = "white"
TEXT = "#111111"

CASES = ["pendulum", "fhn", "robertson"]
CASE_LABELS = {
    "pendulum": "Forced pendulum",
    "fhn": "FitzHugh--Nagumo",
    "robertson": "Robertson kinetics",
}
METHODS = ["denkf", "letkf", "aug_enkf", "bma", "pce", "apce"]
METHOD_LABELS = {
    "denkf": "DEnKF",
    "letkf": "LETKF",
    "aug_enkf": "Aug-EnKF",
    "bma": "BMA",
    "pce": "PCE",
    "apce": "APCE",
}
METHOD_COLORS = {
    "denkf": "#8b8b8b",
    "letkf": "#6ba88f",
    "aug_enkf": "#4e79a7",
    "bma": "#8f63a9",
    "pce": "#67b7e8",
    "apce": "#f28e1c",
}


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
    by_key: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    with SOURCE_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("status", "").strip().lower() != "completed":
                continue
            case = row["case"].strip().lower()
            method = method_key(row["method"])
            if case not in CASES or method not in METHODS:
                continue
            try:
                freq = int(float(row["obs_interval_factor"]))
                val = float(row[metric])
            except Exception:
                continue
            if metric == "nrmse":
                val *= 100.0
            if 1 <= freq <= 8 and np.isfinite(val):
                by_key[(case, freq, method)].append(val)
    return {k: float(np.mean(v)) for k, v in by_key.items() if v}


def draw_row(fig: plt.Figure, metric: str, row_label: str, y_label: str, means: dict[tuple[str, int, str], float], row_y0: float) -> None:
    fig.text(0.006, row_y0 + 0.285, row_label, fontsize=PANEL_LABEL_SIZE, fontweight="bold", ha="left", va="top", color=TEXT)
    left, right, gap = 0.070, 0.980, 0.018
    width = (right - left - gap * 2) / 3
    x = np.arange(1, 9)
    for ci, case in enumerate(CASES):
        ax = fig.add_axes([left + ci * (width + gap), row_y0, width, 0.235], facecolor=WHITE)
        ax.set_title(CASE_LABELS[case], fontsize=CASE_TITLE_SIZE, pad=7)
        for method in METHODS:
            vals = np.array([means.get((case, f, method), np.nan) for f in range(1, 9)], dtype=float)
            if method in {"pce", "apce"}:
                lw = 2.6
                alpha = 0.98
                ms = 5.0
                z = 4
            else:
                lw = 1.35
                alpha = 0.72
                ms = 4.0
                z = 2
            ax.plot(
                x,
                vals,
                color=METHOD_COLORS[method],
                lw=lw,
                marker="o",
                ms=ms,
                mfc=WHITE,
                mec=METHOD_COLORS[method],
                mew=1.0 if method in {"pce", "apce"} else 0.8,
                alpha=alpha,
                zorder=z,
            )
        ax.set_xlim(0.75, 8.25)
        ax.set_xticks(x)
        ax.tick_params(labelsize=TICK_SIZE, width=0.9, length=3)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)
        if ci == 0:
            ax.set_ylabel(y_label, fontsize=AXIS_LABEL_SIZE, labelpad=6)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Observation interval", fontsize=AXIS_LABEL_SIZE, labelpad=5)


def main() -> None:
    nrmse_means = load_metric_means("nrmse")
    alpha_means = load_metric_means("alpha_absolute_error")

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=WHITE)
    draw_row(fig, "nrmse", "b", r"nRMSE (%)", nrmse_means, row_y0=0.535)
    draw_row(fig, "alpha", "c", r"$\alpha$ MAE", alpha_means, row_y0=0.165)

    handles = [
        Line2D([0], [0], color=METHOD_COLORS[m], lw=2.6 if m in {"pce", "apce"} else 1.35,
               marker="o", ms=5.0 if m in {"pce", "apce"} else 4.0, mfc=WHITE, mec=METHOD_COLORS[m],
               mew=1.0 if m in {"pce", "apce"} else 0.8, label=METHOD_LABELS[m], alpha=0.98 if m in {"pce", "apce"} else 0.72)
        for m in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.535, 0.988),
        ncol=6,
        frameon=False,
        prop={"family": "Arial", "size": LEGEND_SIZE},
        handlelength=1.45,
        handletextpad=0.36,
        columnspacing=0.88,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in [("png", {"dpi": 600}), ("pdf", {}), ("svg", {}), ("tiff", {"dpi": 600})]:
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", facecolor=WHITE, bbox_inches=None, pad_inches=0, **kwargs)
    plt.close(fig)
    print(OUT_DIR / f"{OUT_STEM}.png")


if __name__ == "__main__":
    main()
