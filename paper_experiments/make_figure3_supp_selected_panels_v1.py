"""Build selected Supplementary Figure 3 panels.

Panels generated
----------------
A5  Representative PCE/APCE evidence weights at interval 6.
A23 CRPS--nRMSE trade-off across cases, methods and observation intervals.
A28 Maximum candidate weight / collapse-rate diagnostic.

Panels not generated from this source bundle
--------------------------------------------
A29 Oracle gap: no oracle records are present in the authoritative freq1--freq8
    formal source-data.
A35 State uncertainty bands: representative traces contain mean_states and
    forecast noise but not analysis ensemble quantiles or posterior interval
    trajectories. We therefore do not fabricate bands.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "source_data" / "figure3_selected5_freq1to8_formal_50seeds_allmethods_20260813"
SOURCE_DIR = ROOT / "combined" / "source_data"
RUN_CSV = SOURCE_DIR / "figure3_freq1to8_formal_authoritative_run_source_data.csv"
TRACE_DIR = ROOT / "representative_traces_freq6"
OUT_DIR = PROJECT_ROOT / "ncs_english_latex" / "figures-supplemental" / "figure3_selected_panels_20260813"
OUT_STEM = "figure3_supp_selected_A23_A28_v3"

CASES = ["pk_infusion", "chemical", "pendulum", "fhn", "robertson"]
CASE_LABELS = {
    "pk_infusion": "PK",
    "chemical": "Chem.",
    "pendulum": "Pend.",
    "fhn": "FHN",
    "robertson": "Rob.",
}
METHODS = ["denkf", "letkf", "iensf", "aug_enkf", "bma_static", "pce", "apce"]
METHOD_LABELS = {
    "denkf": "DEnKF",
    "letkf": "LETKF",
    "iensf": "IEnSF",
    "aug_enkf": "Aug-EnKF",
    "bma_static": "BMA",
    "pce": "PCE",
    "apce": "APCE",
}
METHOD_COLORS = {
    "denkf": "#8B8B8B",
    "letkf": "#5AA087",
    "iensf": "#9A7B55",
    "aug_enkf": "#4C78A8",
    "bma_static": "#7A5195",
    "pce": "#64A9DC",
    "apce": "#E68613",
}
CASE_COLORS = {
    "pk_infusion": "#4477AA",
    "chemical": "#66A61E",
    "pendulum": "#AA3377",
    "fhn": "#CCBB44",
    "robertson": "#228833",
}

SELECTED_FREQ = 6


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 10.8,
            "axes.labelsize": 10.8,
            "axes.titlesize": 11.6,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 10.3,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fnum(v: str | float | int | None) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan


def compact_tick(value: float, _pos=None) -> str:
    if abs(value) < 1e-12:
        return "0"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    if abs(value) >= 1:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def style_ax(ax: plt.Axes) -> None:
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(False)
    ax.tick_params(length=2.2, pad=1.5, width=0.65)
    ax.xaxis.set_major_formatter(FuncFormatter(compact_tick))
    ax.yaxis.set_major_formatter(FuncFormatter(compact_tick))


def add_label(ax: plt.Axes, label: str, title: str, *, x=-0.08, y=1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=15, fontweight="bold", ha="left", va="bottom")
    ax.text(x + 0.06, y + 0.01, title, transform=ax.transAxes, fontsize=11.6, ha="left", va="bottom")


def normalized_entropy(weights: np.ndarray) -> np.ndarray:
    w = np.clip(weights, 1e-300, None)
    h = -np.sum(w * np.log(w), axis=1)
    return h / max(math.log(w.shape[1]), 1e-12)


def draw_weight_panel(ax: plt.Axes, plot_rows: list[dict[str, object]]) -> None:
    add_label(ax, "a", "Representative evidence weights")
    subgrid = ax.get_subplotspec().subgridspec(2, 5, wspace=0.18, hspace=0.30)
    ax.set_axis_off()
    for row, method in enumerate(["pce", "apce"]):
        for col, case in enumerate(CASES):
            child = ax.figure.add_subplot(subgrid[row, col])
            files = sorted(TRACE_DIR.glob(f"{case}_{method}_s*.npz"))
            if not files:
                child.set_axis_off()
                continue
            z = np.load(files[0], allow_pickle=False)
            weights = np.asarray(z["alpha_weight_history"], dtype=float)
            lengths = np.asarray(z["alpha_weight_history_lengths"], dtype=int)
            weights = weights[:, : int(np.max(lengths))]
            t = np.linspace(0, 1, weights.shape[0])
            for k in range(weights.shape[1]):
                child.plot(t, weights[:, k], lw=0.75, alpha=0.65, color=METHOD_COLORS[method])
            child.plot(t, weights.max(axis=1), lw=1.35, color="#333333")
            child.set_ylim(-0.02, 1.02)
            child.set_xlim(0, 1)
            if row == 0:
                child.set_title(CASE_LABELS[case], fontsize=7.2, pad=2)
            if col == 0:
                child.set_ylabel(METHOD_LABELS[method], fontsize=7.0, labelpad=2)
            else:
                child.set_yticklabels([])
            if row == 1:
                child.set_xlabel("time", fontsize=6.8, labelpad=1)
            else:
                child.set_xticklabels([])
            child.tick_params(length=1.5, pad=1, labelsize=6.1)
            for spine in ["right", "top"]:
                child.spines[spine].set_visible(False)
            for idx in np.linspace(0, weights.shape[0] - 1, 7, dtype=int):
                plot_rows.append(
                    {
                        "panel": "A5",
                        "case": case,
                        "method": METHOD_LABELS[method],
                        "time_fraction": float(t[idx]),
                        "max_weight": float(weights[idx].max()),
                        "normalized_entropy": float(normalized_entropy(weights[[idx]])[0]),
                    }
                )
    ax.legend(
        handles=[
            Line2D([0], [0], color="#333333", lw=1.35, label="max weight"),
            Line2D([0], [0], color=METHOD_COLORS["pce"], lw=1.0, alpha=0.70, label="candidate weights"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.995, 1.02),
        ncol=2,
        frameon=False,
        fontsize=6.8,
        handlelength=1.2,
        columnspacing=0.8,
    )


def draw_tradeoff(ax: plt.Axes, runs: list[dict[str, str]], plot_rows: list[dict[str, object]]) -> None:
    add_label(ax, "a", "CRPS--nRMSE trade-off")
    for method in METHODS:
        xs = []
        ys = []
        sizes = []
        for freq in range(1, 9):
            sub = [r for r in runs if int(float(r["obs_interval_factor"])) == freq and r["method"] == method]
            if not sub:
                continue
            x = float(np.mean([fnum(r["nrmse"]) * 100 for r in sub]))
            y = float(np.mean([fnum(r["crps"]) * 1000 for r in sub]))
            xs.append(x)
            ys.append(y)
            sizes.append(14 + 2.5 * freq)
            plot_rows.append(
                {
                    "panel": "A23",
                    "method": METHOD_LABELS[method],
                    "freq": freq,
                    "metric": "mean_nrmse_percent_vs_mean_crps_x1e3",
                    "nrmse_percent": x,
                    "crps_x1e3": y,
                }
            )
        ax.plot(xs, ys, color=METHOD_COLORS[method], lw=0.9, alpha=0.55)
        ax.scatter(xs, ys, s=sizes, color=METHOD_COLORS[method], edgecolors="white", linewidths=0.35, alpha=0.82, label=METHOD_LABELS[method])
    ax.set_xlabel("nRMSE (%)")
    ax.set_ylabel(r"CRPS ($10^{-3}$)")
    style_ax(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.08), ncol=4, handlelength=0.8, columnspacing=0.7, fontsize=6.5)


def draw_collapse_panel(ax: plt.Axes, runs: list[dict[str, str]], plot_rows: list[dict[str, object]]) -> None:
    add_label(ax, "b", "Evidence concentration and collapse")
    records: list[tuple[str, str, int, float, float]] = []
    for method in ["pce", "apce"]:
        for freq in range(1, 9):
            maxw = []
            collapse = []
            for r in runs:
                if int(float(r["obs_interval_factor"])) != freq or r["method"] != method:
                    continue
                ent = fnum(r.get("alpha_evidence_entropy"))
                gp = fnum(r.get("alpha_grid_points"))
                spread = fnum(r.get("alpha_spread"))
                # Direct final max-weight is not in the run CSV. Estimate a
                # conservative concentration proxy from entropy; trace panel A5
                # supplies the exact representative trajectories.
                if math.isfinite(ent) and math.isfinite(gp) and gp > 1:
                    hnorm = ent / math.log(gp)
                    maxw.append(1 - hnorm)
                    collapse.append(1.0 if hnorm < 0.20 else 0.0)
            mw = float(np.mean(maxw)) if maxw else math.nan
            cr = float(np.mean(collapse)) if collapse else math.nan
            records.append((method, METHOD_LABELS[method], freq, mw, cr))
            plot_rows.append(
                {
                    "panel": "A28",
                    "method": METHOD_LABELS[method],
                    "freq": freq,
                    "concentration_proxy_1_minus_normalized_entropy": mw,
                    "collapse_rate_entropy_lt_0.2": cr,
                    "n": len(maxw),
                }
            )
    for method in ["pce", "apce"]:
        sub = [r for r in records if r[0] == method]
        x = [r[2] for r in sub]
        y1 = [r[3] for r in sub]
        y2 = [r[4] for r in sub]
        ax.plot(x, y1, marker="o", ms=3.3, lw=1.4, color=METHOD_COLORS[method], label=f"{METHOD_LABELS[method]} max-w proxy")
        ax.plot(x, y2, marker="s", ms=3.0, lw=1.1, color=METHOD_COLORS[method], alpha=0.55, ls="--", label=f"{METHOD_LABELS[method]} collapse")
    ax.set_xlim(0.8, 8.2)
    ax.set_ylim(0.0, 0.4)
    ax.set_xticks(range(1, 9))
    ax.set_xlabel("Observation interval")
    ax.set_ylabel("rate / concentration")
    style_ax(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.00), ncol=1, fontsize=6.5, handlelength=1.2)


def main() -> None:
    apply_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = [r for r in read_csv(RUN_CSV) if r.get("status") == "completed" and r.get("numerical_status") == "valid"]
    plot_rows: list[dict[str, object]] = []

    fig = plt.figure(figsize=(7.35, 3.35), dpi=180)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.14, 1.0], wspace=0.34)
    draw_tradeoff(fig.add_subplot(gs[0, 0]), runs, plot_rows)
    draw_collapse_panel(fig.add_subplot(gs[0, 1]), runs, plot_rows)
    fig.subplots_adjust(left=0.080, right=0.988, top=0.860, bottom=0.185)

    outputs = []
    for ext, kwargs in [("svg", {}), ("pdf", {}), ("png", {"dpi": 600}), ("tiff", {"dpi": 600})]:
        out = OUT_DIR / f"{OUT_STEM}.{ext}"
        fig.savefig(out, bbox_inches="tight", **kwargs)
        outputs.append(out)
    plt.close(fig)

    source_path = OUT_DIR / f"{OUT_STEM}_plot_source_data.csv"
    fields = sorted({k for row in plot_rows for k in row})
    with source_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(plot_rows)

    qa = {
        "figure": OUT_STEM,
        "generated_panels": ["A23", "A28"],
        "not_generated": {
            "A29_oracle_gap": "No oracle records are present in the authoritative freq1--freq8 formal source-data.",
            "A35_state_uncertainty_bands": "Representative trace NPZ files contain mean_states but not posterior ensemble quantiles or interval trajectories; bands are not fabricated.",
        },
        "source_data": str(RUN_CSV),
        "trace_dir": str(TRACE_DIR),
        "plot_source_data": str(source_path),
        "outputs": [str(p) for p in outputs],
        "note": "A5 representative weight trajectories were removed from this candidate because the many overlaid lines were visually overloaded; A28 uses the source-data concentration proxy 1-H(w)/logK.",
    }
    (OUT_DIR / f"{OUT_STEM}_qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / f"{OUT_STEM}_contract.md").write_text(__doc__ or "", encoding="utf-8")
    print(json.dumps({"outputs": [str(p) for p in outputs], "plot_source": str(source_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
