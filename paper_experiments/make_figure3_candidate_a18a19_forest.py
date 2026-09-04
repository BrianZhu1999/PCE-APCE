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
OUT_STEM = "figure3_candidate_A18A19_paired_forest_v1"

PANEL_LABEL_SIZE = 26
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
BASELINES = ["aug_enkf", "bma"]
REF_METHODS = ["pce", "apce"]
COLORS = {"aug_enkf": "#4e79a7", "bma": "#9b75b6"}
LABELS = {"aug_enkf": "vs Aug-EnKF", "bma": "vs BMA"}

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


def bootstrap_ci(vals: np.ndarray, n_boot: int = 4000, seed: int = 123) -> tuple[float, float]:
    if vals.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    means = vals[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_diffs(metric: str) -> dict[tuple[str, str], np.ndarray]:
    vals: dict[tuple[str, int, int, str], float] = {}
    with SOURCE_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("status", "").strip().lower() != "completed":
                continue
            case = row["case"].strip().lower()
            method = method_key(row["method"])
            if case not in CASES or method not in set(BASELINES + REF_METHODS):
                continue
            try:
                freq = int(float(row["obs_interval_factor"]))
                seed = int(row["seed"])
                val = float(row[metric])
            except Exception:
                continue
            if metric == "nrmse":
                val *= 100.0
            if 1 <= freq <= 8 and np.isfinite(val):
                vals[(case, freq, seed, method)] = val
    out: dict[tuple[str, str], list[float]] = defaultdict(list)
    for case in CASES:
        keys = {(freq, seed) for (c, freq, seed, m) in vals if c == case}
        for freq, seed in sorted(keys):
            p = vals.get((case, freq, seed, "pce"), np.nan)
            a = vals.get((case, freq, seed, "apce"), np.nan)
            ref = np.nanmin([p, a])
            if not np.isfinite(ref):
                continue
            for base in BASELINES:
                b = vals.get((case, freq, seed, base), np.nan)
                if np.isfinite(b):
                    out[(case, base)].append(b - ref)
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def main() -> None:
    diffs = load_diffs("nrmse")
    fig = plt.figure(figsize=(15.56, 3.35), facecolor="white")
    ax = fig.add_axes([0.120, 0.185, 0.835, 0.610], facecolor="white")
    y_positions = []
    labels = []
    y = 0
    for case in CASES:
        for base in BASELINES:
            arr = diffs.get((case, base), np.array([]))
            mean = float(np.mean(arr))
            lo, hi = bootstrap_ci(arr, seed=100 + len(y_positions))
            ax.plot([lo, hi], [y, y], color=COLORS[base], lw=2.2, alpha=0.85)
            ax.scatter([mean], [y], s=42, facecolor="white", edgecolor=COLORS[base], linewidth=1.7, zorder=3)
            y_positions.append(y)
            labels.append(CASE_LABELS[case] if base == "aug_enkf" else "")
            y += 1
        y += 0.45
    ax.axvline(0, color="#151515", lw=1.0, alpha=0.85)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=TICK_SIZE + 1)
    ax.invert_yaxis()
    ax.set_xlabel(r"Baseline minus best(PCE, APCE), $\Delta$ nRMSE (%)", fontsize=AXIS_LABEL_SIZE, labelpad=7)
    ax.tick_params(axis="x", labelsize=TICK_SIZE, width=0.9, length=3)
    ax.tick_params(axis="y", width=0, length=0)
    ax.grid(False)
    fig.text(0.006, 0.890, "d", fontsize=PANEL_LABEL_SIZE, fontweight="bold", ha="left", va="top")
    handles = [Line2D([0], [0], color=COLORS[b], lw=2.2, marker="o", ms=6, mfc="white",
                      mec=COLORS[b], mew=1.7, label=LABELS[b]) for b in BASELINES]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.535, 0.965), ncol=2,
               frameon=False, prop={"family": "Arial", "size": LEGEND_SIZE})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in [("png", {"dpi": 600}), ("pdf", {}), ("svg", {}), ("tiff", {"dpi": 600})]:
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", facecolor="white", bbox_inches=None, pad_inches=0, **kwargs)
    plt.close(fig)
    print(OUT_DIR / f"{OUT_STEM}.png")


if __name__ == "__main__":
    main()
