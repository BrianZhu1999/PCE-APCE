from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap


PROJECT_ROOT = Path(r".\hybrid_uncertain_wave")
SOURCE_CSV = PROJECT_ROOT / "source_data" / "figure3_selected5_freq1to8_formal_50seeds_allmethods_20260813" / "combined" / "source_data" / "figure3_freq1to8_formal_authoritative_run_source_data.csv"
OUT_DIR = Path(r"figures")
OUT_STEM = "figure3bc_nrmse_alpha_advantage_matrix_v1"


PANEL_LABEL_SIZE = 26
CASE_TITLE_SIZE = 18
LEGEND_SIZE = 18
AXIS_LABEL_SIZE = 16
TICK_SIZE = 11
CELL_TEXT_SIZE = 8.2

FIG_W = 15.56
FIG_H = 3.45
WHITE = "white"
TEXT = "#111111"

CASES = ["pendulum", "fhn", "robertson"]
CASE_LABELS = {
    "pendulum": "Forced pendulum",
    "fhn": "FitzHugh--Nagumo",
    "robertson": "Robertson kinetics",
}
METHODS = ["denkf", "letkf", "aug_enkf", "bma", "pce", "apce"]
METHOD_LABELS = ["DEnKF", "LETKF", "Aug", "BMA", "PCE", "APCE"]


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


CMAP = LinearSegmentedColormap.from_list(
    "signed_advantage",
    ["#8f4d5b", "#f2d4c2", "#f9f7f2", "#d8ebdc", "#238b5b"],
    N=256,
)


def method_key(raw: str) -> str:
    s = raw.strip().lower().replace("-", "_")
    if s in {"augenkf", "aug_enkf"}:
        return "aug_enkf"
    if s in {"bma_static", "static_bma"}:
        return "bma"
    return s


def load_values(metric: str) -> dict[tuple[str, int, str], float]:
    values: dict[tuple[str, int, str], list[float]] = defaultdict(list)
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
                values[(case, freq, method)].append(val)
    return {key: float(np.mean(vals)) for key, vals in values.items() if vals}


def advantage_matrix(means: dict[tuple[str, int, str], float], case: str) -> np.ndarray:
    mat = np.full((len(METHODS), 8), np.nan)
    for fi in range(8):
        freq = fi + 1
        aug = means.get((case, freq, "aug_enkf"), np.nan)
        for mi, method in enumerate(METHODS):
            val = means.get((case, freq, method), np.nan)
            if np.isfinite(aug) and np.isfinite(val):
                mat[mi, fi] = aug - val
    return mat


def fmt(v: float, metric: str) -> str:
    if not np.isfinite(v):
        return ""
    if metric == "nrmse":
        return f"{v:+.1f}" if abs(v) >= 0.95 else f"{v:+.2f}"
    return f"{v:+.3f}"


def draw_cell_panel(ax: plt.Axes, mat: np.ndarray, case: str, metric: str, norm: TwoSlopeNorm) -> mpl.image.AxesImage:
    im = ax.imshow(mat, cmap=CMAP, norm=norm, aspect="auto", interpolation="nearest")
    ax.set_title(CASE_LABELS[case], fontsize=CASE_TITLE_SIZE, pad=7)
    ax.set_xticks(np.arange(8))
    ax.set_xticklabels([str(i) for i in range(1, 9)], fontsize=TICK_SIZE)
    ax.set_yticks(np.arange(len(METHODS)))
    ax.set_yticklabels(METHOD_LABELS, fontsize=TICK_SIZE)
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
        spine.set_color("#c9c9c9")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isfinite(v):
                continue
            color = "white" if abs(norm(v) - 0.5) > 0.34 else "#161616"
            ax.text(j, i, fmt(v, metric), ha="center", va="center", fontsize=CELL_TEXT_SIZE, color=color, fontweight="bold" if METHODS[i] in {"pce", "apce"} else "normal")
    return im


def main() -> None:
    nrmse_means = load_values("nrmse")
    alpha_means = load_values("alpha_absolute_error")
    nrmse_mats = [advantage_matrix(nrmse_means, c) for c in CASES]
    alpha_mats = [advantage_matrix(alpha_means, c) for c in CASES]

    nabs = np.nanpercentile(np.abs(np.concatenate([m[np.isfinite(m)] for m in nrmse_mats])), 96)
    aabs = np.nanpercentile(np.abs(np.concatenate([m[np.isfinite(m)] for m in alpha_mats])), 96)
    n_norm = TwoSlopeNorm(vmin=-float(nabs), vcenter=0.0, vmax=float(nabs))
    a_norm = TwoSlopeNorm(vmin=-float(aabs), vcenter=0.0, vmax=float(aabs))

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=WHITE)
    fig.text(0.006, 0.928, "b", fontsize=PANEL_LABEL_SIZE, fontweight="bold", ha="left", va="top", color=TEXT)
    fig.text(0.512, 0.928, "c", fontsize=PANEL_LABEL_SIZE, fontweight="bold", ha="left", va="top", color=TEXT)

    left1, right1 = 0.060, 0.478
    left2, right2 = 0.566, 0.984
    gap = 0.016
    w1 = (right1 - left1 - 2 * gap) / 3
    w2 = (right2 - left2 - 2 * gap) / 3
    y, h = 0.205, 0.595

    last_n = last_a = None
    for idx, case in enumerate(CASES):
        ax = fig.add_axes([left1 + idx * (w1 + gap), y, w1, h], facecolor=WHITE)
        last_n = draw_cell_panel(ax, nrmse_mats[idx], case, "nrmse", n_norm)
        if idx == 0:
            ax.set_ylabel("Method", fontsize=AXIS_LABEL_SIZE, labelpad=5)
        else:
            ax.set_yticklabels([])
        if idx == 1:
            ax.set_xlabel("Observation interval", fontsize=AXIS_LABEL_SIZE, labelpad=5)

        ax2 = fig.add_axes([left2 + idx * (w2 + gap), y, w2, h], facecolor=WHITE)
        last_a = draw_cell_panel(ax2, alpha_mats[idx], case, "alpha", a_norm)
        if idx == 0:
            ax2.set_ylabel("Method", fontsize=AXIS_LABEL_SIZE, labelpad=5)
        else:
            ax2.set_yticklabels([])
        if idx == 1:
            ax2.set_xlabel("Observation interval", fontsize=AXIS_LABEL_SIZE, labelpad=5)

    cax1 = fig.add_axes([0.165, 0.080, 0.205, 0.030])
    cb1 = fig.colorbar(last_n, cax=cax1, orientation="horizontal")
    cb1.set_label(r"$\Delta$nRMSE (%)", fontsize=AXIS_LABEL_SIZE, labelpad=4)
    cb1.ax.tick_params(labelsize=TICK_SIZE, length=3)

    cax2 = fig.add_axes([0.674, 0.080, 0.205, 0.030])
    cb2 = fig.colorbar(last_a, cax=cax2, orientation="horizontal")
    cb2.set_label(r"$\Delta\alpha$ MAE", fontsize=AXIS_LABEL_SIZE, labelpad=4)
    cb2.ax.tick_params(labelsize=TICK_SIZE, length=3)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in [("png", {"dpi": 600}), ("pdf", {}), ("svg", {}), ("tiff", {"dpi": 600})]:
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", facecolor=WHITE, bbox_inches=None, pad_inches=0, **kwargs)
    plt.close(fig)
    print(OUT_DIR / f"{OUT_STEM}.png")


if __name__ == "__main__":
    main()
