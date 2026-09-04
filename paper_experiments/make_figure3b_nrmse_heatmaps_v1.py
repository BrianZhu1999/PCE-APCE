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
OUT_STEM = "figure3b_nrmse_heatmaps_v1"


PANEL_LABEL_SIZE = 26
CASE_TITLE_SIZE = 18
LEGEND_SIZE = 18
AXIS_LABEL_SIZE = 16
TICK_SIZE = 12
CELL_TEXT_SIZE = 8.2

BG = "#fbf8f1"
TEXT = "#111111"
GRID = "#f3eadb"

CASES = ["pk_infusion", "chemical", "pendulum", "fhn", "robertson"]
CASE_LABELS = {
    "pk_infusion": "PK infusion",
    "chemical": "Chemical reaction",
    "pendulum": "Forced pendulum",
    "fhn": "FitzHugh--Nagumo",
    "robertson": "Robertson kinetics",
}

METHODS = ["denkf", "letkf", "iensf", "aug_enkf", "bma", "pce", "apce"]
METHOD_LABELS = ["DEnKF", "LETKF", "IEnSF", "Aug", "BMA", "PCE", "APCE"]


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
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


CMAP = LinearSegmentedColormap.from_list(
    "nrmse_nature",
    ["#fff8e4", "#f3d394", "#e69a6a", "#b85b65", "#65395c"],
    N=256,
)


def method_key(raw: str) -> str:
    s = raw.strip().lower().replace("-", "_")
    if s in {"augenkf", "aug_enkf"}:
        return "aug_enkf"
    if s in {"bma_static", "static_bma"}:
        return "bma"
    return s


def load_mean_matrix() -> dict[str, np.ndarray]:
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
                nrmse = float(row["nrmse"]) * 100.0
            except Exception:
                continue
            if 1 <= freq <= 8 and np.isfinite(nrmse):
                values[(case, freq, method)].append(nrmse)

    mats: dict[str, np.ndarray] = {}
    for case in CASES:
        mat = np.full((len(METHODS), 8), np.nan)
        for mi, method in enumerate(METHODS):
            for freq in range(1, 9):
                vals = values.get((case, freq, method), [])
                if vals:
                    mat[mi, freq - 1] = float(np.mean(vals))
        mats[case] = mat
    return mats


def fmt_value(v: float) -> str:
    if not np.isfinite(v):
        return ""
    if v >= 10:
        return f"{v:.0f}"
    if v >= 1:
        return f"{v:.1f}"
    return f"{v:.2f}"


def draw_heatmap(ax: plt.Axes, case: str, mat: np.ndarray, norm: mpl.colors.Normalize) -> None:
    im = ax.imshow(mat, cmap=CMAP, norm=norm, aspect="auto", interpolation="nearest")
    ax.set_title(CASE_LABELS[case], fontsize=CASE_TITLE_SIZE, pad=8, color=TEXT)
    ax.set_xticks(np.arange(8))
    ax.set_xticklabels([str(i) for i in range(1, 9)], fontsize=TICK_SIZE)
    ax.set_yticks(np.arange(len(METHODS)))
    ax.set_yticklabels(METHOD_LABELS, fontsize=TICK_SIZE)
    ax.set_xlabel("Observation interval", fontsize=AXIS_LABEL_SIZE, labelpad=5)
    ax.tick_params(length=0)

    ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(METHODS), 1), minor=True)
    ax.grid(which="minor", color=GRID, linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#b8ad98")

    best_per_freq = np.nanmin(mat, axis=0)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isfinite(v):
                continue
            color = "white" if norm(v) > 0.58 else "#27221f"
            weight = "bold" if abs(v - best_per_freq[j]) < 1e-10 else "normal"
            ax.text(j, i, fmt_value(v), ha="center", va="center", fontsize=CELL_TEXT_SIZE, color=color, fontweight=weight)
    return im


def main() -> None:
    mats = load_mean_matrix()
    all_vals = np.concatenate([m[np.isfinite(m)] for m in mats.values()])
    vmax = float(np.nanpercentile(all_vals, 98))
    vmax = max(vmax, 1.0)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=vmax)

    fig = plt.figure(figsize=(14.8, 3.75), facecolor=BG)
    fig.text(0.008, 0.950, "b", fontsize=PANEL_LABEL_SIZE, fontweight="bold", ha="left", va="top", color=TEXT)

    left = 0.055
    width = 0.172
    gap = 0.014
    axes = []
    last_im = None
    for idx, case in enumerate(CASES):
        ax = fig.add_axes([left + idx * (width + gap), 0.205, width, 0.635], facecolor=BG)
        last_im = draw_heatmap(ax, case, mats[case], norm)
        if idx > 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Method", fontsize=AXIS_LABEL_SIZE, labelpad=5)
        axes.append(ax)

    cax = fig.add_axes([0.230, 0.075, 0.540, 0.035])
    cb = fig.colorbar(last_im, cax=cax, orientation="horizontal")
    cb.set_label("Mean nRMSE across 50 paired seeds (%)", fontsize=LEGEND_SIZE, labelpad=5)
    cb.ax.tick_params(labelsize=TICK_SIZE, length=3, width=0.8)
    cb.outline.set_linewidth(0.8)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in [("png", {"dpi": 600}), ("pdf", {}), ("svg", {}), ("tiff", {"dpi": 600})]:
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", facecolor=BG, bbox_inches="tight", pad_inches=0.01, **kwargs)
    plt.close(fig)
    print(OUT_DIR / f"{OUT_STEM}.png")


if __name__ == "__main__":
    main()
