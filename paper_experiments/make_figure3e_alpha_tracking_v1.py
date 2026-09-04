from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = Path(r".\hybrid_uncertain_wave")
TRACE_DIR = PROJECT_ROOT / "source_data" / "figure3_selected5_freq1to8_formal_50seeds_allmethods_20260813" / "representative_traces_freq6"
OUT_DIR = Path(r"figures")
OUT_STEM = "figure3e_alpha_tracking_v1_freq6"

PANEL_LABEL_SIZE = 26
CASE_TITLE_SIZE = 18
LEGEND_SIZE = 18
AXIS_LABEL_SIZE = 16
TICK_SIZE = 11

FIG_W = 15.56
FIG_H = 3.10
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
TRUE_COLOR = "#222222"


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


def load_trace(case: str, method: str) -> tuple[np.ndarray, float]:
    path = TRACE_DIR / f"{case}_{method}_s2026081200.npz"
    z = np.load(path, allow_pickle=True)
    alpha = np.asarray(z["alpha_estimate_history"], dtype=float)
    cfg = json.loads(str(z["config_json"]))
    alpha_true = float(cfg.get("alpha_true", np.nan))
    return alpha, alpha_true


def nice_ylim(series: list[np.ndarray], alpha_true: float) -> tuple[float, float]:
    vals = [alpha_true]
    for s in series:
        vals.extend(s[np.isfinite(s)].tolist())
    arr = np.asarray(vals, dtype=float)
    lo, hi = float(np.min(arr)), float(np.max(arr))
    span = max(hi - lo, 0.04)
    pad = span * 0.16
    return lo - pad, hi + pad


def main() -> None:
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=WHITE)
    left, right = 0.052, 0.988
    gap = 0.025
    panel_w = (right - left - 4 * gap) / 5
    bottom, panel_h = 0.220, 0.500

    for i, case in enumerate(CASES):
        ax = fig.add_axes([left + i * (panel_w + gap), bottom, panel_w, panel_h], facecolor=WHITE)
        series = {}
        alpha_true = np.nan
        for method in METHODS:
            alpha, at = load_trace(case, method)
            series[method] = alpha
            if np.isfinite(at):
                alpha_true = at
        ylim = nice_ylim(list(series.values()), alpha_true)
        t = np.linspace(0.0, 1.0, len(next(iter(series.values()))))
        ax.axhline(alpha_true, color=TRUE_COLOR, lw=1.35, ls=(0, (4, 3)), alpha=0.85, zorder=1)
        for method in METHODS:
            ax.plot(t, series[method], color=METHOD_COLORS[method], lw=2.65, alpha=0.98, zorder=3)
        ax.set_title(CASE_LABELS[case], fontsize=CASE_TITLE_SIZE, pad=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(*ylim)
        ax.set_xticks([0, 0.5, 1])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        ax.set_xlabel("Normalized time", fontsize=AXIS_LABEL_SIZE, labelpad=5)
        if i == 0:
            ax.set_ylabel(r"$\hat{\alpha}$", fontsize=AXIS_LABEL_SIZE, labelpad=5)
        else:
            ax.tick_params(labelleft=False)
        ax.tick_params(labelsize=TICK_SIZE, width=0.9, length=3)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)

    fig.text(0.006, 0.810, "e", fontsize=PANEL_LABEL_SIZE, fontweight="bold", ha="left", va="top", color=TEXT)
    handles = [
        Line2D([0], [0], color=METHOD_COLORS["pce"], lw=2.65, label="PCE"),
        Line2D([0], [0], color=METHOD_COLORS["apce"], lw=2.65, label="APCE"),
        Line2D([0], [0], color=TRUE_COLOR, lw=1.35, ls=(0, (4, 3)), label="Truth"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.535, 0.965),
        ncol=3,
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
