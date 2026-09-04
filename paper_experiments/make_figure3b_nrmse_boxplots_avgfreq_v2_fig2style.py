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
OUT_STEM = "figure3b_nrmse_boxplots_avgfreq_v3_fig2style"


PANEL_LABEL_SIZE = 26
CASE_TITLE_SIZE = 18
LEGEND_SIZE = 18
AXIS_LABEL_SIZE = 16
TICK_SIZE = 12

CASES = ["pk_infusion", "chemical", "pendulum", "fhn", "robertson"]
CASE_LABELS = {
    "pk_infusion": "PK infusion",
    "chemical": "Chemical reaction",
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


def draw_box_panel(ax: plt.Axes, case: str, data: dict[str, list[float]], ymax: float) -> None:
    positions = np.arange(1, len(METHODS) + 1, dtype=float)
    samples = [data[m] for m in METHODS]
    colors = [METHOD_COLORS[m] for m in METHODS]

    bp = ax.boxplot(
        samples,
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
    )
    for patch, color in zip(bp["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.70)
        patch.set_edgecolor("white")
        patch.set_linewidth(0.75)
    for key in ("whiskers", "caps", "medians"):
        for artist in bp[key]:
            artist.set_color("#333333")
            artist.set_linewidth(0.75)

    rng = np.random.default_rng(20260813)
    for vals, pos, color in zip(samples, positions, colors, strict=True):
        vals_arr = np.asarray(vals, dtype=float)
        jitter = rng.normal(0, 0.022, size=len(vals_arr))
        ax.scatter(
            np.full(len(vals_arr), pos) + jitter,
            vals_arr,
            s=8.0,
            color=color,
            edgecolors="white",
            linewidths=0.18,
            alpha=0.32,
            zorder=2,
        )

    ax.set_title(CASE_LABELS[case], fontsize=CASE_TITLE_SIZE, pad=7)
    ax.set_xlim(0.45, len(METHODS) + 0.55)
    ax.set_ylim(0, ymax)
    ax.set_xticks(positions)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=TICK_SIZE, width=0.9, length=3)
    ax.grid(False)


def main() -> None:
    data = load_seed_averages()
    all_vals = np.concatenate([np.asarray(data[c][m], dtype=float) for c in CASES for m in METHODS])
    ymax = float(np.ceil(np.nanpercentile(all_vals, 99.0) / 2.0) * 2.0)
    ymax = max(ymax, 4.0)

    # Width is matched to the approved Figure 3a raster at 600 dpi
    # (9335 px / 600 dpi ~= 15.56 in). Do not use bbox_inches="tight",
    # otherwise Matplotlib changes the final panel width.
    fig = plt.figure(figsize=(15.56, 3.35), facecolor="white")
    fig.text(0.006, 0.940, "b", fontsize=PANEL_LABEL_SIZE, fontweight="bold", ha="left", va="top")

    left = 0.048
    right = 0.980
    gap = 0.018
    width = (right - left - gap * (len(CASES) - 1)) / len(CASES)
    axes = []
    for idx, case in enumerate(CASES):
        ax = fig.add_axes([left + idx * (width + gap), 0.155, width, 0.560], facecolor="white")
        draw_box_panel(ax, case, data[case], ymax)
        if idx == 0:
            ax.set_ylabel("Mean nRMSE (%)", fontsize=AXIS_LABEL_SIZE, labelpad=6)
        else:
            ax.set_yticklabels([])
        axes.append(ax)

    handles = [
        Line2D([0], [0], marker="s", linestyle="none", markersize=10.5,
               markerfacecolor=METHOD_COLORS[m], markeredgecolor="white",
               markeredgewidth=0.5, label=METHOD_LABELS[m])
        for m in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.535, 0.965),
        ncol=len(METHODS),
        frameon=False,
        prop={"family": "Arial", "size": LEGEND_SIZE},
        handlelength=0.75,
        handletextpad=0.32,
        columnspacing=0.82,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in [("png", {"dpi": 600}), ("pdf", {}), ("svg", {}), ("tiff", {"dpi": 600})]:
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", facecolor="white", bbox_inches=None, pad_inches=0, **kwargs)
    plt.close(fig)
    print(OUT_DIR / f"{OUT_STEM}.png")


if __name__ == "__main__":
    main()
