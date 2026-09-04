from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(r".\hybrid_uncertain_wave")
SOURCE_CSV = PROJECT_ROOT / "source_data" / "figure3_selected5_freq1to8_formal_50seeds_allmethods_20260813" / "combined" / "source_data" / "figure3_freq1to8_formal_authoritative_run_source_data.csv"
OUT_DIR = Path(r"figures")


PANEL_LABEL_SIZE = 26
CASE_TITLE_SIZE = 18
LEGEND_SIZE = 18
AXIS_LABEL_SIZE = 16
TICK_SIZE = 12

FIG_W = 15.56
WHITE = "white"
TEXT = "#111111"

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
    "denkf": "#8b8b8b",
    "letkf": "#6ba88f",
    "aug_enkf": "#4e79a7",
    "bma": "#8f63a9",
}
OURS_BLUE = "#67b7e8"
OURS_ORANGE = "#f28e1c"


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


def paired_deltas(values: dict[tuple[str, int, str, str], float], baseline: str) -> dict[tuple[str, int], list[float]]:
    deltas: dict[tuple[str, int], list[float]] = defaultdict(list)
    for case in CASES:
        for freq in range(1, 9):
            seeds = sorted({
                seed
                for (c, f, m, seed) in values
                if c == case and f == freq and m in {"pce", "apce", baseline}
            })
            for seed in seeds:
                pce = values.get((case, freq, "pce", seed))
                apce = values.get((case, freq, "apce", seed))
                base = values.get((case, freq, baseline, seed))
                if pce is None or apce is None or base is None:
                    continue
                deltas[(case, freq)].append(base - min(pce, apce))
    return deltas


def mean_by_case_freq(values: dict[tuple[str, int, str, str], float], baseline: str) -> np.ndarray:
    d = paired_deltas(values, baseline)
    mat = np.full((len(CASES), 8), np.nan)
    for ci, case in enumerate(CASES):
        for freq in range(1, 9):
            vals = d.get((case, freq), [])
            if vals:
                mat[ci, freq - 1] = float(np.mean(vals))
    return mat


def save_all(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in [("png", {"dpi": 600}), ("pdf", {}), ("svg", {}), ("tiff", {"dpi": 600})]:
        fig.savefig(OUT_DIR / f"{stem}.{ext}", facecolor=WHITE, bbox_inches=None, pad_inches=0, **kwargs)
    plt.close(fig)
    print(OUT_DIR / f"{stem}.png")


def option1_overall_trend(values: dict[tuple[str, int, str, str], float]) -> None:
    mats = {b: mean_by_case_freq(values, b) for b in ["aug_enkf", "bma"]}
    fig = plt.figure(figsize=(FIG_W, 3.25), facecolor=WHITE)
    ax = fig.add_axes([0.070, 0.205, 0.895, 0.630], facecolor=WHITE)
    fig.text(0.006, 0.940, "b", fontsize=PANEL_LABEL_SIZE, fontweight="bold", ha="left", va="top", color=TEXT)
    x = np.arange(1, 9)
    for baseline, color, label in [
        ("aug_enkf", BASELINE_COLORS["aug_enkf"], "vs Aug-EnKF"),
        ("bma", BASELINE_COLORS["bma"], "vs BMA"),
    ]:
        mat = mats[baseline]
        for row in mat:
            ax.plot(x, row, color=color, lw=0.85, alpha=0.22)
        mean = np.nanmean(mat, axis=0)
        se = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(mat.shape[0])
        ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.13, lw=0)
        ax.plot(x, mean, color=color, lw=2.4, marker="o", ms=6.0, mfc="white", mec=color, mew=1.2, label=label)
    ax.axhline(0, color="#222222", lw=0.9, ls=(0, (3, 3)))
    ax.set_xlim(0.75, 8.25)
    ax.set_xticks(x)
    ax.set_xlabel("Observation interval", fontsize=AXIS_LABEL_SIZE, labelpad=6)
    ax.set_ylabel(r"$\Delta$nRMSE (%)", fontsize=AXIS_LABEL_SIZE, labelpad=7)
    ax.tick_params(labelsize=TICK_SIZE, width=0.9, length=3)
    ax.grid(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.50, 1.18), ncol=2, prop={"family": "Arial", "size": LEGEND_SIZE}, handlelength=1.45, columnspacing=1.20)
    save_all(fig, "figure3b_option1_advantage_trend_v1")


def option2_case_multiples(values: dict[tuple[str, int, str, str], float]) -> None:
    mats = {b: mean_by_case_freq(values, b) for b in ["aug_enkf", "bma"]}
    fig = plt.figure(figsize=(FIG_W, 3.35), facecolor=WHITE)
    fig.text(0.006, 0.940, "b", fontsize=PANEL_LABEL_SIZE, fontweight="bold", ha="left", va="top", color=TEXT)
    left, right, gap = 0.050, 0.980, 0.018
    width = (right - left - gap * 4) / 5
    x = np.arange(1, 9)
    ymin = min(np.nanmin(mats["aug_enkf"]), np.nanmin(mats["bma"]))
    ymax = max(np.nanmax(mats["aug_enkf"]), np.nanmax(mats["bma"]))
    pad = 0.13 * (ymax - ymin)
    for ci, case in enumerate(CASES):
        ax = fig.add_axes([left + ci * (width + gap), 0.185, width, 0.610], facecolor=WHITE)
        ax.axhline(0, color="#222222", lw=0.75, ls=(0, (3, 3)))
        for baseline, color, label in [
            ("aug_enkf", BASELINE_COLORS["aug_enkf"], "Aug-EnKF"),
            ("bma", BASELINE_COLORS["bma"], "BMA"),
        ]:
            ax.plot(x, mats[baseline][ci], color=color, lw=1.8, marker="o", ms=4.4, mfc="white", mec=color, mew=1.0, label=label)
        ax.set_title(CASE_LABELS[case], fontsize=CASE_TITLE_SIZE, pad=7)
        ax.set_xlim(0.75, 8.25)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_xticks(x)
        ax.set_xlabel("Observation interval", fontsize=AXIS_LABEL_SIZE, labelpad=5)
        if ci == 0:
            ax.set_ylabel(r"$\Delta$nRMSE (%)", fontsize=AXIS_LABEL_SIZE, labelpad=6)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=TICK_SIZE, width=0.9, length=3)
        ax.grid(False)
        if ci == 2:
            ax.legend(loc="upper center", bbox_to_anchor=(0.50, 1.43), ncol=2, prop={"family": "Arial", "size": LEGEND_SIZE}, handlelength=1.25, columnspacing=1.0)
    save_all(fig, "figure3b_option2_casewise_advantage_v1")


def option3_win_matrix(values: dict[tuple[str, int, str, str], float]) -> None:
    win = np.zeros((len(BASELINES), 8), dtype=float)
    for bi, baseline in enumerate(BASELINES):
        mat = mean_by_case_freq(values, baseline)
        win[bi] = np.sum(mat > 0, axis=0)

    cmap = LinearSegmentedColormap.from_list("wins", ["#f7f7f7", "#d8ebdc", "#80c59e", "#1f8a5b"], N=256)
    fig = plt.figure(figsize=(FIG_W, 3.05), facecolor=WHITE)
    fig.text(0.006, 0.925, "b", fontsize=PANEL_LABEL_SIZE, fontweight="bold", ha="left", va="top", color=TEXT)
    ax = fig.add_axes([0.125, 0.205, 0.710, 0.610], facecolor=WHITE)
    im = ax.imshow(win, vmin=0, vmax=5, cmap=cmap, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(8))
    ax.set_xticklabels([str(i) for i in range(1, 9)], fontsize=TICK_SIZE)
    ax.set_yticks(np.arange(len(BASELINES)))
    ax.set_yticklabels([BASELINE_LABELS[b] for b in BASELINES], fontsize=TICK_SIZE)
    ax.set_xlabel("Observation interval", fontsize=AXIS_LABEL_SIZE, labelpad=6)
    ax.set_ylabel("Baseline", fontsize=AXIS_LABEL_SIZE, labelpad=7)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(win.shape[0]):
        for j in range(win.shape[1]):
            ax.text(j, i, f"{int(win[i, j])}/5", ha="center", va="center", fontsize=15, color="#111111", fontweight="bold")
    ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(BASELINES), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    cax = fig.add_axes([0.870, 0.245, 0.018, 0.520])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=TICK_SIZE, length=3, width=0.8)
    cb.set_label("Wins among five cases", fontsize=AXIS_LABEL_SIZE, labelpad=8)
    cb.set_ticks([0, 1, 2, 3, 4, 5])
    save_all(fig, "figure3b_option3_win_matrix_v1")


def main() -> None:
    values = load_run_values()
    option1_overall_trend(values)
    option2_case_multiples(values)
    option3_win_matrix(values)


if __name__ == "__main__":
    main()
