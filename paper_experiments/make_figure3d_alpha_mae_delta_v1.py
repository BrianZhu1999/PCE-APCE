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
OUT_STEM = "figure3d_alpha_mae_delta_v2_localylim"

PANEL_LABEL_SIZE = 26
CASE_TITLE_SIZE = 18
LEGEND_SIZE = 18
AXIS_LABEL_SIZE = 16
TICK_SIZE = 11

FIG_W = 15.56
FIG_H = 3.35
WHITE = "white"
TEXT = "#111111"

CASES = ["chemical", "pk_infusion", "pendulum", "fhn", "robertson"]
CASE_LABELS = {
    "chemical": "Chemical",
    "pk_infusion": "PK infusion",
    "pendulum": "Forced",
    "fhn": "FHN",
    "robertson": "Robertson",
}
BASELINES = ["aug_enkf", "bma"]
REF_METHODS = ["pce", "apce"]
METHOD_LABELS = {"aug_enkf": "Aug-EnKF", "bma": "BMA"}
METHOD_COLORS = {"aug_enkf": "#4e79a7", "bma": "#9b75b6"}


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


def load_metric_means() -> dict[tuple[str, int, str], float]:
    values: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    keep_methods = set(BASELINES + REF_METHODS)
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
                val = float(row["alpha_absolute_error"])
            except Exception:
                continue
            if 1 <= freq <= 8 and np.isfinite(val):
                values[(case, freq, method)].append(val)
    return {k: float(np.mean(v)) for k, v in values.items() if v}


def baseline_gaps(means: dict[tuple[str, int, str], float]) -> dict[tuple[str, int, str], float]:
    out: dict[tuple[str, int, str], float] = {}
    for case in CASES:
        for freq in range(1, 9):
            ref = np.nanmin([means.get((case, freq, m), np.nan) for m in REF_METHODS])
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
        return -0.005, 0.05
    lo = min(float(np.min(arr)), 0.0)
    hi = max(float(np.max(arr)), 0.0)
    span = max(hi - lo, 0.01)
    pad = span * 0.13
    return lo - pad, hi + pad


def main() -> None:
    gaps = baseline_gaps(load_metric_means())

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=WHITE)
    left, right = 0.052, 0.988
    gap = 0.025
    panel_w = (right - left - 4 * gap) / 5
    bottom, panel_h = 0.230, 0.495
    x = np.arange(1, 9, dtype=float)
    offsets = {"aug_enkf": -0.095, "bma": 0.095}

    for i, case in enumerate(CASES):
        ax = fig.add_axes([left + i * (panel_w + gap), bottom, panel_w, panel_h], facecolor=WHITE)
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
        vals = [gaps.get((case, f, m), np.nan) for f in range(1, 9) for m in BASELINES]
        ax.set_ylim(*nice_ylim(vals))
        ax.set_xticks(x)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        ax.set_xlabel("Obs. interval", fontsize=AXIS_LABEL_SIZE, labelpad=5)
        if i == 0:
            ax.set_ylabel(r"$\Delta \alpha$ MAE", fontsize=AXIS_LABEL_SIZE, labelpad=5)
        else:
            ax.tick_params(labelleft=False)
        ax.tick_params(labelsize=TICK_SIZE, width=0.9, length=3)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)

    fig.text(0.006, 0.815, "d", fontsize=PANEL_LABEL_SIZE, fontweight="bold", ha="left", va="top", color=TEXT)

    handles = [
        Line2D([0], [0], color=METHOD_COLORS[m], lw=2.15, marker="o", ms=5.6, mfc=WHITE,
               mec=METHOD_COLORS[m], mew=1.55, label=METHOD_LABELS[m])
        for m in BASELINES
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.535, 0.988),
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
