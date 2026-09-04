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
OUT_STEM = "figure3d_sparsity_advantage_bubble_v2_median_iqr_clipped"

PANEL_LABEL_SIZE = 26
LEGEND_SIZE = 18
AXIS_LABEL_SIZE = 16
TICK_SIZE = 11

CASES = ["chemical", "pk_infusion", "pendulum", "fhn", "robertson"]
BASELINES = ["aug_enkf", "bma"]
OURS = ["pce", "apce"]

METRICS = {
    "nrmse": {"label": "nRMSE", "color": "#4e79a7", "scale": 100.0},
    "crps": {"label": "CRPS", "color": "#ff8c00", "scale": 1.0},
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


def load_means(metric: str) -> dict[tuple[str, int, str], float]:
    scale = METRICS[metric]["scale"]
    vals: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    with SOURCE_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("status", "").strip().lower() != "completed":
                continue
            case = row["case"].strip().lower()
            method = method_key(row["method"])
            if case not in CASES or method not in set(BASELINES + OURS):
                continue
            try:
                freq = int(float(row["obs_interval_factor"]))
                val = float(row[metric]) * scale
            except Exception:
                continue
            if 1 <= freq <= 8 and np.isfinite(val):
                vals[(case, freq, method)].append(val)
    return {k: float(np.mean(v)) for k, v in vals.items() if v}


def advantage_records(metric: str) -> list[dict[str, float | str]]:
    means = load_means(metric)
    records = []
    for case in CASES:
        for freq in range(1, 9):
            best_ours = np.nanmin([means.get((case, freq, m), np.nan) for m in OURS])
            best_base = np.nanmin([means.get((case, freq, m), np.nan) for m in BASELINES])
            if not (np.isfinite(best_ours) and np.isfinite(best_base) and best_base > 0):
                continue
            # Relative advantage: positive means best(PCE/APCE) is lower-error than best baseline.
            adv = 100.0 * (best_base - best_ours) / best_base
            records.append({
                "case": case,
                "freq": float(freq),
                "adv": float(adv),
                "difficulty": float(best_base),
            })
    return records


def median_iqr(vals: np.ndarray) -> tuple[float, float, float]:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.median(vals)), float(np.percentile(vals, 25)), float(np.percentile(vals, 75))


def size_map(difficulty: np.ndarray) -> np.ndarray:
    d = np.asarray(difficulty, dtype=float)
    lo, hi = np.nanmin(d), np.nanmax(d)
    if not np.isfinite(lo) or hi <= lo:
        return np.full_like(d, 55.0)
    z = (d - lo) / (hi - lo)
    return 28.0 + 92.0 * np.sqrt(z)


def main() -> None:
    all_records = {metric: advantage_records(metric) for metric in METRICS}
    fig = plt.figure(figsize=(15.56, 3.55), facecolor="white")
    ax = fig.add_axes([0.070, 0.210, 0.875, 0.570], facecolor="white")
    ymin, ymax = -25.0, 25.0

    jitter = {"nrmse": -0.075, "crps": 0.075}
    for metric, spec in METRICS.items():
        recs = all_records[metric]
        freqs = np.array([r["freq"] for r in recs], dtype=float)
        adv = np.array([r["adv"] for r in recs], dtype=float)
        diff = np.array([r["difficulty"] for r in recs], dtype=float)
        sizes = size_map(diff)
        adv_clip = np.clip(adv, ymin + 1.0, ymax - 1.0)
        inside = (adv >= ymin) & (adv <= ymax)
        low = adv < ymin
        high = adv > ymax
        ax.scatter(
            freqs[inside] + jitter[metric],
            adv_clip[inside],
            s=sizes[inside],
            facecolor="white",
            edgecolor=spec["color"],
            linewidth=1.45,
            alpha=0.55,
            zorder=2,
        )
        if np.any(low):
            ax.scatter(
                freqs[low] + jitter[metric],
                adv_clip[low],
                s=size_map(diff[low]),
                marker="v",
                facecolor="white",
                edgecolor=spec["color"],
                linewidth=1.45,
                alpha=0.70,
                zorder=3,
            )
        if np.any(high):
            ax.scatter(
                freqs[high] + jitter[metric],
                adv_clip[high],
                s=size_map(diff[high]),
                marker="^",
                facecolor="white",
                edgecolor=spec["color"],
                linewidth=1.45,
                alpha=0.70,
                zorder=3,
            )
        xs, mean, lo, hi = [], [], [], []
        for freq in range(1, 9):
            vals = np.array([r["adv"] for r in recs if int(r["freq"]) == freq], dtype=float)
            m, l, h = median_iqr(vals)
            xs.append(freq + jitter[metric])
            mean.append(m)
            lo.append(l)
            hi.append(h)
        xs = np.asarray(xs, dtype=float)
        mean = np.asarray(mean, dtype=float)
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        ax.fill_between(xs, np.clip(lo, ymin, ymax), np.clip(hi, ymin, ymax),
                        color=spec["color"], alpha=0.13, linewidth=0, zorder=1)
        ax.plot(xs, mean, color=spec["color"], lw=3.0, marker="o", ms=6.0,
                mfc="white", mec=spec["color"], mew=1.5, zorder=4, label=spec["label"])

    ax.axhline(0, color="#151515", lw=1.05, alpha=0.88, zorder=0)
    ax.set_xlim(0.55, 8.45)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(np.arange(1, 9))
    ax.set_xlabel("Observation interval", fontsize=AXIS_LABEL_SIZE, labelpad=7)
    ax.set_ylabel("Relative advantage (%)", fontsize=AXIS_LABEL_SIZE, labelpad=8)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.tick_params(labelsize=TICK_SIZE, width=0.9, length=3)
    ax.grid(False)

    fig.text(0.006, 0.880, "d", fontsize=PANEL_LABEL_SIZE, fontweight="bold", ha="left", va="top")
    handles = [
        Line2D([0], [0], color=METRICS["nrmse"]["color"], lw=3.0, marker="o", ms=6, mfc="white",
               mec=METRICS["nrmse"]["color"], mew=1.5, label="nRMSE"),
        Line2D([0], [0], color=METRICS["crps"]["color"], lw=3.0, marker="o", ms=6, mfc="white",
               mec=METRICS["crps"]["color"], mew=1.5, label="CRPS"),
        Line2D([0], [0], marker="o", color="#666666", lw=0, ms=8, mfc="white",
               mec="#666666", mew=1.4, alpha=0.65, label="case"),
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
        columnspacing=1.05,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in [("png", {"dpi": 600}), ("pdf", {}), ("svg", {}), ("tiff", {"dpi": 600})]:
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", facecolor="white", bbox_inches=None, pad_inches=0, **kwargs)
    plt.close(fig)
    print(OUT_DIR / f"{OUT_STEM}.png")


if __name__ == "__main__":
    main()
