from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


PROJECT_ROOT = Path(r".\hybrid_uncertain_wave")
SOURCE_CSV = PROJECT_ROOT / "source_data" / "figure3_selected5_freq1to8_formal_50seeds_allmethods_20260813" / "combined" / "source_data" / "figure3_freq1to8_formal_authoritative_run_source_data.csv"
OUT_DIR = Path(r"figures")
OUT_STEM = "figure3bc_advantage_row_v1"


PANEL_LABEL_SIZE = 26
CASE_TITLE_SIZE = 18
LEGEND_SIZE = 18
AXIS_LABEL_SIZE = 16
TICK_SIZE = 11

FIG_W = 15.56
FIG_H = 3.45
TEXT = "#111111"
WHITE = "white"

CASES = ["pk_infusion", "chemical", "pendulum", "fhn", "robertson"]
CASE_LABELS = {
    "pk_infusion": "PK infusion",
    "chemical": "Chemical reaction",
    "pendulum": "Forced pendulum",
    "fhn": "FitzHugh--Nagumo",
    "robertson": "Robertson kinetics",
}
BASELINES = ["denkf", "letkf", "aug_enkf", "bma"]
BASELINE_LABELS = {
    "denkf": "DEnKF",
    "letkf": "LETKF",
    "aug_enkf": "Aug-EnKF",
    "bma": "BMA",
}
BASELINE_COLORS = {
    "aug_enkf": "#4e79a7",
    "bma": "#8f63a9",
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


def load_run_values() -> dict[tuple[str, int, str, str], float]:
    out: dict[tuple[str, int, str, str], float] = {}
    with SOURCE_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("status", "").strip().lower() != "completed":
                continue
            case = row["case"].strip().lower()
            method = method_key(row["method"])
            if case not in CASES:
                continue
            try:
                freq = int(float(row["obs_interval_factor"]))
                val = float(row["nrmse"]) * 100.0
            except Exception:
                continue
            if 1 <= freq <= 8 and np.isfinite(val):
                out[(case, freq, method, row["seed"])] = val
    return out


def mean_by_case_freq(values: dict[tuple[str, int, str, str], float], baseline: str) -> np.ndarray:
    mat = np.full((len(CASES), 8), np.nan)
    for ci, case in enumerate(CASES):
        for freq in range(1, 9):
            deltas = []
            seeds = sorted({
                seed for (c, f, _m, seed) in values
                if c == case and f == freq
            })
            for seed in seeds:
                pce = values.get((case, freq, "pce", seed))
                apce = values.get((case, freq, "apce", seed))
                base = values.get((case, freq, baseline, seed))
                if pce is None or apce is None or base is None:
                    continue
                deltas.append(base - min(pce, apce))
            if deltas:
                mat[ci, freq - 1] = float(np.mean(deltas))
    return mat


def draw_casewise_panel(fig: plt.Figure, values: dict[tuple[str, int, str, str], float]) -> None:
    fig.text(0.006, 0.925, "b", fontsize=PANEL_LABEL_SIZE, fontweight="bold", ha="left", va="top", color=TEXT)
    mats = {b: mean_by_case_freq(values, b) for b in ["aug_enkf", "bma"]}
    x = np.arange(1, 9)
    ymin = min(np.nanmin(mats["aug_enkf"]), np.nanmin(mats["bma"]))
    ymax = max(np.nanmax(mats["aug_enkf"]), np.nanmax(mats["bma"]))
    pad = 0.14 * (ymax - ymin)

    left, right, gap = 0.055, 0.665, 0.013
    width = (right - left - gap * 4) / 5
    for ci, case in enumerate(CASES):
        ax = fig.add_axes([left + ci * (width + gap), 0.195, width, 0.585], facecolor=WHITE)
        ax.axhline(0, color="#222222", lw=0.75, ls=(0, (3, 3)))
        ax.plot(x, mats["aug_enkf"][ci], color=BASELINE_COLORS["aug_enkf"], lw=1.8, marker="o", ms=4.0, mfc=WHITE, mec=BASELINE_COLORS["aug_enkf"], mew=1.0)
        ax.plot(x, mats["bma"][ci], color=BASELINE_COLORS["bma"], lw=1.8, marker="o", ms=4.0, mfc=WHITE, mec=BASELINE_COLORS["bma"], mew=1.0)
        ax.set_title(CASE_LABELS[case], fontsize=CASE_TITLE_SIZE, pad=7)
        ax.set_xlim(0.75, 8.25)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_xticks([1, 4, 8])
        ax.tick_params(labelsize=TICK_SIZE, width=0.9, length=3)
        ax.grid(False)
        if ci == 0:
            ax.set_ylabel(r"$\Delta$nRMSE (%)", fontsize=AXIS_LABEL_SIZE, labelpad=5)
        else:
            ax.set_yticklabels([])
        if ci == 2:
            ax.set_xlabel("Observation interval", fontsize=AXIS_LABEL_SIZE, labelpad=5)

    handles = [
        mpl.lines.Line2D([0], [0], color=BASELINE_COLORS["aug_enkf"], lw=1.9, marker="o", ms=5.0, mfc=WHITE, label="vs Aug-EnKF"),
        mpl.lines.Line2D([0], [0], color=BASELINE_COLORS["bma"], lw=1.9, marker="o", ms=5.0, mfc=WHITE, label="vs BMA"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.360, 0.985), ncol=2,
               prop={"family": "Arial", "size": LEGEND_SIZE}, handlelength=1.25, columnspacing=1.0)


def draw_win_matrix_panel(fig: plt.Figure, values: dict[tuple[str, int, str, str], float]) -> None:
    fig.text(0.690, 0.925, "c", fontsize=PANEL_LABEL_SIZE, fontweight="bold", ha="left", va="top", color=TEXT)
    win = np.zeros((len(BASELINES), 8), dtype=float)
    for bi, baseline in enumerate(BASELINES):
        mat = mean_by_case_freq(values, baseline)
        win[bi] = np.sum(mat > 0, axis=0)

    cmap = LinearSegmentedColormap.from_list("wins", ["#f7f7f7", "#d8ebdc", "#80c59e", "#1f8a5b"], N=256)
    ax = fig.add_axes([0.755, 0.270, 0.185, 0.470], facecolor=WHITE)
    im = ax.imshow(win, vmin=0, vmax=5, cmap=cmap, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(8))
    ax.set_xticklabels([str(i) for i in range(1, 9)], fontsize=TICK_SIZE)
    ax.set_yticks(np.arange(len(BASELINES)))
    ax.set_yticklabels([BASELINE_LABELS[b] for b in BASELINES], fontsize=TICK_SIZE)
    ax.set_xlabel("Observation interval", fontsize=AXIS_LABEL_SIZE, labelpad=5)
    ax.set_ylabel("Baseline", fontsize=AXIS_LABEL_SIZE, labelpad=5)
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(win.shape[0]):
        for j in range(win.shape[1]):
            ax.text(j, i, f"{int(win[i, j])}/5", ha="center", va="center", fontsize=11.5, color="#111111", fontweight="bold")

    cax = fig.add_axes([0.958, 0.305, 0.012, 0.400])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=TICK_SIZE, length=3, width=0.8)
    cb.set_ticks([0, 1, 2, 3, 4, 5])
    cb.set_label("Wins", fontsize=AXIS_LABEL_SIZE, labelpad=6)


def main() -> None:
    values = load_run_values()
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=WHITE)
    draw_casewise_panel(fig, values)
    draw_win_matrix_panel(fig, values)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in [("png", {"dpi": 600}), ("pdf", {}), ("svg", {}), ("tiff", {"dpi": 600})]:
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", facecolor=WHITE, bbox_inches=None, pad_inches=0, **kwargs)
    plt.close(fig)
    print(OUT_DIR / f"{OUT_STEM}.png")


if __name__ == "__main__":
    main()
