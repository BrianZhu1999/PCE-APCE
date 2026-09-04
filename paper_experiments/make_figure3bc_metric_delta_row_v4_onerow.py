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
OUT_STEM = "figure3bc_metric_delta_row_v6_onerow_aug_bma_only"

PANEL_LABEL_SIZE = 26
CASE_TITLE_SIZE = 18
LEGEND_SIZE = 18
AXIS_LABEL_SIZE = 16
TICK_SIZE = 11

FIG_W = 15.56
FIG_H = 3.35
WHITE = "white"
TEXT = "#111111"

CASES = ["pendulum", "fhn", "robertson"]
CASE_LABELS = {
    "pendulum": "Forced",
    "fhn": "FHN",
    "robertson": "Robertson",
}

# In a delta plot referenced to best(PCE, APCE), PCE/APCE themselves are the zero reference
# and are intentionally not drawn. DEnKF is also removed here to keep the story focused.
METHODS = ["aug_enkf", "bma"]
METHOD_LABELS = {
    "aug_enkf": "Aug-EnKF",
    "bma": "BMA",
}
METHOD_COLORS = {
    "aug_enkf": "#4e79a7",
    "bma": "#9b75b6",
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
            if case not in CASES or method not in set(METHODS + ["pce", "apce"]):
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


def delta_to_best_pce_apce(
    means: dict[tuple[str, int, str], float]
) -> dict[tuple[str, int, str], float]:
    out: dict[tuple[str, int, str], float] = {}
    for case in CASES:
        for freq in range(1, 9):
            pce = means.get((case, freq, "pce"), np.nan)
            apce = means.get((case, freq, "apce"), np.nan)
            ref = np.nanmin([pce, apce])
            if not np.isfinite(ref):
                continue
            for method in METHODS:
                val = means.get((case, freq, method), np.nan)
                if np.isfinite(val):
                    out[(case, freq, method)] = val - ref
    return out


def nice_limits(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return -1.0, 1.0
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    pad = max(0.08 * (hi - lo), 0.015 * max(abs(hi), 1.0))
    return min(lo - pad, -pad), hi + pad


def plot_delta_panel(
    ax: plt.Axes,
    case: str,
    deltas: dict[tuple[str, int, str], float],
    y_label: str | None,
) -> None:
    x = np.arange(1, 9)
    ax.axhline(0, color="#1b1b1b", lw=1.0, alpha=0.85, zorder=1)
    for method in METHODS:
        vals = np.array([deltas.get((case, f, method), np.nan) for f in range(1, 9)], dtype=float)
        if method in {"pce", "apce"}:
            lw, alpha, ms, z = 2.9, 0.98, 4.8, 5
        else:
            lw, alpha, ms, z = 1.35, 0.72, 3.8, 3
        ax.plot(
            x,
            vals,
            color=METHOD_COLORS[method],
            lw=lw,
            marker="o",
            ms=ms,
            mfc=WHITE,
            mec=METHOD_COLORS[method],
            mew=1.05 if method in {"pce", "apce"} else 0.75,
            alpha=alpha,
            zorder=z,
        )
    ax.set_title(CASE_LABELS[case], fontsize=CASE_TITLE_SIZE, pad=8)
    ax.set_xlim(0.75, 8.25)
    ax.set_xticks(x)
    ax.set_xlabel("Obs. interval", fontsize=AXIS_LABEL_SIZE, labelpad=5)
    if y_label:
        ax.set_ylabel(y_label, fontsize=AXIS_LABEL_SIZE, labelpad=5)
    else:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=TICK_SIZE, width=0.9, length=3)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def main() -> None:
    nrmse_delta = delta_to_best_pce_apce(load_metric_means("nrmse"))
    alpha_delta = delta_to_best_pce_apce(load_metric_means("alpha_absolute_error"))

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=WHITE)
    left, right = 0.050, 0.988
    gap = 0.018
    panel_w = (right - left - 5 * gap) / 6
    bottom, panel_h = 0.150, 0.500
    axes = [fig.add_axes([left + i * (panel_w + gap), bottom, panel_w, panel_h], facecolor=WHITE) for i in range(6)]

    for ci, case in enumerate(CASES):
        plot_delta_panel(axes[ci], case, nrmse_delta, r"$\Delta$ nRMSE (%)" if ci == 0 else None)
    for ci, case in enumerate(CASES):
        plot_delta_panel(axes[ci + 3], case, alpha_delta, r"$\Delta \alpha$ MAE" if ci == 0 else None)

    for ci, case in enumerate(CASES):
        vals = [nrmse_delta.get((case, f, method), np.nan) for f in range(1, 9) for method in METHODS]
        axes[ci].set_ylim(*nice_limits(vals))
    for ci, case in enumerate(CASES):
        vals = [alpha_delta.get((case, f, method), np.nan) for f in range(1, 9) for method in METHODS]
        axes[ci + 3].set_ylim(*nice_limits(vals))

    label_y = bottom + panel_h + 0.030
    fig.text(0.006, label_y, "b", fontsize=PANEL_LABEL_SIZE, fontweight="bold", ha="left", va="top", color=TEXT)
    fig.text(left + 3 * (panel_w + gap) - 0.030, label_y, "c", fontsize=PANEL_LABEL_SIZE, fontweight="bold", ha="left", va="top", color=TEXT)

    handles = [
        Line2D(
            [0], [0],
            color=METHOD_COLORS[m],
            lw=2.9 if m in {"pce", "apce"} else 1.35,
            marker="o",
            ms=4.8 if m in {"pce", "apce"} else 3.8,
            mfc=WHITE,
            mec=METHOD_COLORS[m],
            mew=1.05 if m in {"pce", "apce"} else 0.75,
            label=METHOD_LABELS[m],
            alpha=0.98 if m in {"pce", "apce"} else 0.72,
        )
        for m in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.535, 0.980),
        ncol=5,
        frameon=False,
        prop={"family": "Arial", "size": LEGEND_SIZE},
        handlelength=1.35,
        handletextpad=0.35,
        columnspacing=0.88,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in [("png", {"dpi": 600}), ("pdf", {}), ("svg", {}), ("tiff", {"dpi": 600})]:
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", facecolor=WHITE, bbox_inches=None, pad_inches=0, **kwargs)
    plt.close(fig)
    print(OUT_DIR / f"{OUT_STEM}.png")


if __name__ == "__main__":
    main()
