from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


PROJECT_ROOT = Path(r".\hybrid_uncertain_wave")
SOURCE_CSV = PROJECT_ROOT / "source_data" / "figure3_selected5_freq1to8_formal_50seeds_allmethods_20260813" / "combined" / "source_data" / "figure3_freq1to8_formal_authoritative_run_source_data.csv"
OUT_DIR = Path(r"figures")
OUT_STEM = "figure3b_nrmse_3d_bars_v2"


PANEL_LABEL_SIZE = 26
CASE_TITLE_SIZE = 18
LEGEND_SIZE = 18
AXIS_LABEL_SIZE = 16
TICK_SIZE = 9

BG = "#fbf8f1"
PANE = "#f1eadb"
GRID = "#d9cfbd"
TEXT = "#111111"

CASES = ["pk_infusion", "chemical", "pendulum", "fhn", "robertson"]
CASE_LABELS = {
    "pk_infusion": "PK infusion",
    "chemical": "Chemical reaction",
    "pendulum": "Forced pendulum",
    "fhn": "FitzHugh--Nagumo",
    "robertson": "Robertson kinetics",
}

METHODS = ["denkf", "letkf", "iensf", "aug_enkf", "bma", "pce", "apce"]
METHOD_LABELS = {
    "denkf": "DEnKF",
    "letkf": "LETKF",
    "iensf": "IEnSF",
    "aug_enkf": "Aug-EnKF",
    "bma": "BMA",
    "pce": "PCE",
    "apce": "APCE",
}
METHOD_COLORS = {
    "denkf": "#8b8b8b",
    "letkf": "#6ba88f",
    "iensf": "#a68a60",
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
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def method_key(raw: str) -> str:
    s = raw.strip().lower().replace("-", "_")
    if s == "augenkf":
        return "aug_enkf"
    if s == "aug_enkf":
        return "aug_enkf"
    return s


def read_mean_nrmse() -> dict[tuple[str, int, str], float]:
    values: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    with SOURCE_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            case = row["case"].strip().lower()
            method = method_key(row["method"])
            if case not in CASES or method not in METHODS:
                continue
            if row.get("status", "").strip().lower() != "completed":
                continue
            try:
                freq = int(float(row["obs_interval_factor"]))
                nrmse = float(row["nrmse"]) * 100.0
            except Exception:
                continue
            if 1 <= freq <= 8 and np.isfinite(nrmse):
                values[(case, freq, method)].append(nrmse)
    return {key: float(np.mean(vals)) for key, vals in values.items() if vals}


def set_3d_style(ax):
    ax.set_facecolor(BG)
    ax.xaxis.pane.set_facecolor(PANE)
    ax.yaxis.pane.set_facecolor(PANE)
    ax.zaxis.pane.set_facecolor(PANE)
    ax.xaxis.pane.set_alpha(1.0)
    ax.yaxis.pane.set_alpha(1.0)
    ax.zaxis.pane.set_alpha(1.0)
    ax.xaxis._axinfo["grid"]["color"] = GRID
    ax.yaxis._axinfo["grid"]["color"] = GRID
    ax.zaxis._axinfo["grid"]["color"] = GRID
    ax.xaxis._axinfo["grid"]["linewidth"] = 0.55
    ax.yaxis._axinfo["grid"]["linewidth"] = 0.55
    ax.zaxis._axinfo["grid"]["linewidth"] = 0.55
    ax.tick_params(axis="both", which="major", pad=0, labelsize=TICK_SIZE)
    ax.tick_params(axis="z", which="major", pad=1, labelsize=TICK_SIZE)


def draw_case(ax, case: str, means: dict[tuple[str, int, str], float], zmax: float):
    xpos = []
    ypos = []
    zpos = []
    dx = []
    dy = []
    dz = []
    colors = []
    for yi, method in enumerate(METHODS):
        for freq in range(1, 9):
            value = means.get((case, freq, method), np.nan)
            if not np.isfinite(value):
                continue
            xpos.append(freq - 0.34)
            ypos.append(yi - 0.34)
            zpos.append(0.0)
            dx.append(0.58)
            dy.append(0.58)
            dz.append(value)
            colors.append(METHOD_COLORS[method])

    ax.bar3d(
        xpos,
        ypos,
        zpos,
        dx,
        dy,
        dz,
        color=colors,
        edgecolor=(0, 0, 0, 0.23),
        linewidth=0.18,
        shade=True,
        alpha=0.96,
        zsort="average",
    )
    ax.view_init(elev=24, azim=-57)
    try:
        ax.set_proj_type("persp", focal_length=1.0)
    except TypeError:
        ax.set_proj_type("persp")
    ax.set_box_aspect((1.45, 1.15, 0.70))
    ax.set_xlim(0.35, 8.55)
    ax.set_ylim(-0.65, len(METHODS) - 0.10)
    ax.set_zlim(0, zmax)
    ax.set_xticks(range(1, 9))
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels(["DEnKF", "LETKF", "IEnSF", "Aug", "BMA", "PCE", "APCE"], fontsize=8.5)
    ax.set_xlabel("Observation interval", fontsize=AXIS_LABEL_SIZE, labelpad=5)
    ax.set_zlabel("nRMSE (%)", fontsize=AXIS_LABEL_SIZE, labelpad=5)
    ax.set_ylabel("")
    ax.set_title(CASE_LABELS[case], fontsize=CASE_TITLE_SIZE, pad=2, color=TEXT)
    set_3d_style(ax)


def main():
    means = read_mean_nrmse()
    if not means:
        raise RuntimeError("No nRMSE records were loaded.")
    zmax = max(means.values())
    zmax = float(np.ceil(zmax / 5.0) * 5.0)

    fig = plt.figure(figsize=(15.8, 4.25), facecolor=BG)
    fig.text(0.010, 0.935, "b", fontsize=PANEL_LABEL_SIZE, fontweight="bold", ha="left", va="top", color=TEXT)

    left = 0.032
    width = 0.188
    gap = 0.006
    axes = []
    for i, case in enumerate(CASES):
        ax = fig.add_axes([left + i * (width + gap), 0.095, width, 0.815], projection="3d")
        draw_case(ax, case, means, zmax)
        axes.append(ax)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in [("png", {"dpi": 600}), ("pdf", {}), ("svg", {}), ("tiff", {"dpi": 600})]:
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", facecolor=BG, bbox_inches="tight", pad_inches=0.01, **kwargs)
    plt.close(fig)
    print(OUT_DIR / f"{OUT_STEM}.png")


if __name__ == "__main__":
    main()
