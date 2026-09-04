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
OUT_STEM = "figure3d_evidence_concentration_v1"

PANEL_LABEL_SIZE = 26
CASE_TITLE_SIZE = 18
LEGEND_SIZE = 18
AXIS_LABEL_SIZE = 16
TICK_SIZE = 11

FIG_W = 15.56
FIG_H = 2.85
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
METHODS = ["pce", "apce"]
METHOD_LABELS = {"pce": "PCE", "apce": "APCE"}
METHOD_COLORS = {"pce": "#55b7e8", "apce": "#ff8c00"}


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
    return raw.strip().lower().replace("-", "_")


def load_concentration() -> dict[tuple[str, int, str], float]:
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
                entropy = float(row["alpha_evidence_entropy"])
                k = float(row["alpha_grid_points"])
            except Exception:
                continue
            if not (1 <= freq <= 8 and np.isfinite(entropy) and np.isfinite(k) and k > 1):
                continue
            concentration = 1.0 - entropy / np.log(k)
            concentration = float(np.clip(concentration, 0.0, 1.0))
            values[(case, freq, method)].append(concentration)
    return {k: float(np.mean(v)) for k, v in values.items() if v}


def main() -> None:
    means = load_concentration()
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=WHITE)

    left, right = 0.052, 0.988
    gap = 0.025
    panel_w = (right - left - 4 * gap) / 5
    bottom, panel_h = 0.230, 0.500
    x = np.arange(1, 9)

    axes = []
    for i, case in enumerate(CASES):
        ax = fig.add_axes([left + i * (panel_w + gap), bottom, panel_w, panel_h], facecolor=WHITE)
        axes.append(ax)
        for method in METHODS:
            y = np.array([means.get((case, f, method), np.nan) for f in range(1, 9)], dtype=float)
            ax.plot(
                x, y,
                color=METHOD_COLORS[method],
                lw=2.8,
                marker="o",
                ms=5.2,
                mfc=WHITE,
                mec=METHOD_COLORS[method],
                mew=1.25,
                alpha=0.98,
                zorder=3,
            )
        ax.set_title(CASE_LABELS[case], fontsize=CASE_TITLE_SIZE, pad=8)
        ax.set_xlim(0.75, 8.25)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks(x)
        ax.set_yticks([0, 0.5, 1])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        ax.set_xlabel("Obs. interval", fontsize=AXIS_LABEL_SIZE, labelpad=5)
        if i == 0:
            ax.set_ylabel("Evidence conc.", fontsize=AXIS_LABEL_SIZE, labelpad=5)
        else:
            ax.tick_params(labelleft=False)
        ax.tick_params(labelsize=TICK_SIZE, width=0.9, length=3)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)

    fig.text(0.006, 0.815, "d", fontsize=PANEL_LABEL_SIZE, fontweight="bold", ha="left", va="top", color=TEXT)

    handles = [
        Line2D(
            [0], [0],
            color=METHOD_COLORS[m],
            lw=2.8,
            marker="o",
            ms=5.2,
            mfc=WHITE,
            mec=METHOD_COLORS[m],
            mew=1.25,
            label=METHOD_LABELS[m],
        )
        for m in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.535, 0.965),
        ncol=2,
        frameon=False,
        prop={"family": "Arial", "size": LEGEND_SIZE},
        handlelength=1.35,
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
