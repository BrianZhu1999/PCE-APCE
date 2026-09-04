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
OUT_STEM = "figure3b_nrmse_boxplots_avgfreq_v1"


PANEL_LABEL_SIZE = 26
CASE_TITLE_SIZE = 18
LEGEND_SIZE = 18
AXIS_LABEL_SIZE = 16
TICK_SIZE = 12

BG = "#fbf8f1"
TEXT = "#111111"
GRID = "#ddd3c0"

CASES = ["pk_infusion", "chemical", "pendulum", "fhn", "robertson"]
CASE_LABELS = {
    "pk_infusion": "PK infusion",
    "chemical": "Chemical reaction",
    "pendulum": "Forced pendulum",
    "fhn": "FitzHugh--Nagumo",
    "robertson": "Robertson kinetics",
}

# IEnSF is intentionally excluded from this averaged-frequency summary panel.
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


def load_seed_averages() -> dict[str, dict[str, list[float]]]:
    by_seed: dict[tuple[str, str, str], list[float]] = defaultdict(list)
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
                by_seed[(case, method, row["seed"])].append(nrmse)

    out: dict[str, dict[str, list[float]]] = {case: {m: [] for m in METHODS} for case in CASES}
    for (case, method, _seed), vals in by_seed.items():
        if vals:
            out[case][method].append(float(np.mean(vals)))
    return out


def draw_case_boxplot(ax: plt.Axes, case: str, data: dict[str, list[float]]) -> None:
    positions = np.arange(len(METHODS)) + 1
    series = [data[m] for m in METHODS]

    bp = ax.boxplot(
        series,
        positions=positions,
        widths=0.54,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#1b1b1b", "linewidth": 1.35},
        whiskerprops={"color": "#686056", "linewidth": 0.9},
        capprops={"color": "#686056", "linewidth": 0.9},
    )
    for box, method in zip(bp["boxes"], METHODS):
        box.set_facecolor(METHOD_COLORS[method])
        box.set_edgecolor("#3e3832")
        box.set_alpha(0.58)
        box.set_linewidth(0.9)

    rng = np.random.default_rng(20260813)
    for pos, method in zip(positions, METHODS):
        vals = np.asarray(data[method], dtype=float)
        jitter = rng.normal(0, 0.045, size=len(vals))
        ax.scatter(
            np.full_like(vals, pos, dtype=float) + jitter,
            vals,
            s=9,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.22,
            alpha=0.42,
            zorder=3,
        )

    ax.set_title(CASE_LABELS[case], fontsize=CASE_TITLE_SIZE, pad=7, color=TEXT)
    ax.set_xticks(positions)
    ax.set_xticklabels([""] * len(METHODS))
    ax.tick_params(axis="y", labelsize=TICK_SIZE, width=0.8, length=3)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.78)
    ax.set_axisbelow(True)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)


def main() -> None:
    data = load_seed_averages()
    all_vals = np.concatenate([np.asarray(data[c][m], dtype=float) for c in CASES for m in METHODS])
    ymax = float(np.ceil(np.nanmax(all_vals) / 2.0) * 2.0)

    fig = plt.figure(figsize=(14.8, 3.55), facecolor=BG)
    fig.text(0.008, 0.950, "b", fontsize=PANEL_LABEL_SIZE, fontweight="bold", ha="left", va="top", color=TEXT)

    left = 0.060
    width = 0.170
    gap = 0.018
    for idx, case in enumerate(CASES):
        ax = fig.add_axes([left + idx * (width + gap), 0.225, width, 0.590], facecolor=BG)
        draw_case_boxplot(ax, case, data[case])
        ax.set_ylim(0, ymax)
        if idx == 0:
            ax.set_ylabel("Mean nRMSE (%)", fontsize=AXIS_LABEL_SIZE, labelpad=6)
        else:
            ax.set_yticklabels([])

    handles = [
        Line2D([0], [0], marker="s", linestyle="none", markersize=11,
               markerfacecolor=METHOD_COLORS[m], markeredgecolor="none", alpha=0.75,
               label=METHOD_LABELS[m])
        for m in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.535, 0.995),
        ncol=len(METHODS),
        frameon=False,
        prop={"family": "Arial", "size": LEGEND_SIZE},
        handlelength=0.80,
        handletextpad=0.35,
        columnspacing=0.90,
    )
    fig.text(
        0.515,
        0.075,
        "Each box summarizes 50 paired seeds after averaging nRMSE across observation intervals 1--8.",
        ha="center",
        va="center",
        fontsize=11.5,
        color="#6f675d",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in [("png", {"dpi": 600}), ("pdf", {}), ("svg", {}), ("tiff", {"dpi": 600})]:
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", facecolor=BG, bbox_inches="tight", pad_inches=0.01, **kwargs)
    plt.close(fig)
    print(OUT_DIR / f"{OUT_STEM}.png")


if __name__ == "__main__":
    main()
