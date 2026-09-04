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
OUT_STEM = "figure3_candidate_A22_calibration_sharpness_v1"

PANEL_LABEL_SIZE = 26
CASE_TITLE_SIZE = 18
LEGEND_SIZE = 18
AXIS_LABEL_SIZE = 16
TICK_SIZE = 11

CASES = ["chemical", "pk_infusion", "pendulum", "fhn", "robertson"]
CASE_LABELS = {
    "chemical": "Chemical",
    "pk_infusion": "PK infusion",
    "pendulum": "Forced",
    "fhn": "FHN",
    "robertson": "Robertson",
}
METHODS = ["aug_enkf", "bma", "pce", "apce"]
METHOD_LABELS = {"aug_enkf": "Aug-EnKF", "bma": "BMA", "pce": "PCE", "apce": "APCE"}
COLORS = {"aug_enkf": "#4e79a7", "bma": "#9b75b6", "pce": "#55b7e8", "apce": "#ff8c00"}
MARKERS = {"aug_enkf": "o", "bma": "s", "pce": "o", "apce": "s"}

plt.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 600,
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.9,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.unicode_minus": False,
})


def method_key(raw: str) -> str:
    s = raw.strip().lower().replace("-", "_")
    if s in {"augenkf", "aug_enkf"}:
        return "aug_enkf"
    if s in {"bma_static", "static_bma"}:
        return "bma"
    return s


def load_points() -> dict[tuple[str, str, int], tuple[float, float]]:
    vals: dict[tuple[str, str, int], list[tuple[float, float]]] = defaultdict(list)
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
                width = float(row["interval_width_90"])
                coverage = float(row["coverage_90"])
            except Exception:
                continue
            if 1 <= freq <= 8 and np.isfinite(width) and np.isfinite(coverage):
                vals[(case, method, freq)].append((width, abs(coverage - 0.90)))
    out = {}
    for key, pairs in vals.items():
        arr = np.asarray(pairs, dtype=float)
        out[key] = (float(np.mean(arr[:, 0])), float(np.mean(arr[:, 1])))
    return out


def main() -> None:
    points = load_points()
    fig = plt.figure(figsize=(15.56, 3.35), facecolor="white")
    left, right = 0.052, 0.988
    gap = 0.025
    panel_w = (right - left - 4 * gap) / 5
    bottom, panel_h = 0.230, 0.500

    for i, case in enumerate(CASES):
        ax = fig.add_axes([left + i * (panel_w + gap), bottom, panel_w, panel_h], facecolor="white")
        for method in METHODS:
            xs = np.array([points.get((case, method, f), (np.nan, np.nan))[0] for f in range(1, 9)])
            ys = np.array([points.get((case, method, f), (np.nan, np.nan))[1] for f in range(1, 9)])
            ax.plot(xs, ys, color=COLORS[method], lw=1.4 if method in {"aug_enkf", "bma"} else 2.4,
                    alpha=0.70 if method in {"aug_enkf", "bma"} else 0.95, zorder=2)
            ax.scatter(xs, ys, s=28 if method in {"pce", "apce"} else 20, marker=MARKERS[method],
                       facecolor="white", edgecolor=COLORS[method], linewidth=1.3, zorder=3)
        ax.set_title(CASE_LABELS[case], fontsize=CASE_TITLE_SIZE, pad=8)
        ax.set_xlabel("Interval width", fontsize=AXIS_LABEL_SIZE, labelpad=5)
        if i == 0:
            ax.set_ylabel("Coverage error", fontsize=AXIS_LABEL_SIZE, labelpad=5)
        else:
            ax.tick_params(labelleft=False)
        ax.tick_params(labelsize=TICK_SIZE, width=0.9, length=3)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        ax.grid(False)
    fig.text(0.006, 0.815, "d", fontsize=PANEL_LABEL_SIZE, fontweight="bold", ha="left", va="top")
    handles = [Line2D([0], [0], color=COLORS[m], lw=2.4 if m in {"pce", "apce"} else 1.4,
                      marker=MARKERS[m], ms=5.2, mfc="white", mec=COLORS[m], mew=1.3, label=METHOD_LABELS[m])
               for m in METHODS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.535, 0.975), ncol=4,
               frameon=False, prop={"family": "Arial", "size": LEGEND_SIZE},
               handlelength=1.25, handletextpad=0.42, columnspacing=0.9)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in [("png", {"dpi": 600}), ("pdf", {}), ("svg", {}), ("tiff", {"dpi": 600})]:
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", facecolor="white", bbox_inches=None, pad_inches=0, **kwargs)
    plt.close(fig)
    print(OUT_DIR / f"{OUT_STEM}.png")


if __name__ == "__main__":
    main()
