from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


CASES = ("wave", "spring", "heat")
METHODS = ("denkf", "letkf", "iensf", "aug_enkf", "bma_static", "pce", "apce")
ALPHA_METHODS = ("aug_enkf", "bma_static", "pce", "apce")
SCALES = (0.50, 0.75, 1.00, 1.25, 1.50)
LABELS = {
    "denkf": "DEnKF",
    "letkf": "LETKF",
    "iensf": "IEnSF",
    "aug_enkf": "Aug-EnKF",
    "bma_static": "BMA",
    "pce": "PCE",
    "apce": "APCE",
}
COLORS = {
    "denkf": "#7C8797",
    "letkf": "#B3BCC7",
    "iensf": "#505A6B",
    "aug_enkf": "#8E6BBE",
    "bma_static": "#17807E",
    "pce": "#2468A2",
    "apce": "#E4572E",
}
LINE_WIDTH = {
    "denkf": 1.0,
    "letkf": 1.0,
    "iensf": 1.0,
    "aug_enkf": 1.1,
    "bma_static": 1.1,
    "pce": 1.8,
    "apce": 1.8,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def bootstrap_mean_ci(values: Iterable[float], rng: np.random.Generator, draws: int = 2500) -> tuple[float, float, float]:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, mean, mean
    sample = rng.choice(arr, size=(draws, arr.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(sample, [0.025, 0.975])
    return mean, float(lo), float(hi)


def build_rows(raw: list[dict[str, str]]) -> list[dict[str, float | str]]:
    nominal: dict[tuple[str, str, int], dict[str, float]] = {}
    for row in raw:
        key = (row["case"], row["method"], int(row["seed"]))
        if abs(f(row, "sensitivity_scale") - 1.0) < 1e-9:
            nominal[key] = {
                "nrmse": f(row, "nrmse"),
                "crps": f(row, "crps"),
            }
    rows: list[dict[str, float | str]] = []
    for row in raw:
        case = row["case"]
        method = row["method"]
        seed = int(row["seed"])
        scale = f(row, "sensitivity_scale")
        nom = nominal.get((case, method, seed), {})
        nrmse = f(row, "nrmse")
        crps = f(row, "crps")
        rows.append(
            {
                "case": case,
                "method": method,
                "seed": seed,
                "scale": scale,
                "nrmse_rel": nrmse / nom["nrmse"] if math.isfinite(nrmse) and nom.get("nrmse", 0) else float("nan"),
                "crps_rel": crps / nom["crps"] if math.isfinite(crps) and nom.get("crps", 0) else float("nan"),
                "coverage": f(row, "coverage_90"),
                "alpha_error": f(row, "alpha_absolute_error") if method in ALPHA_METHODS else float("nan"),
            }
        )
    return rows


def values(rows: list[dict[str, float | str]], case: str, method: str, scale: float, metric: str) -> list[float]:
    out = []
    for row in rows:
        if row["case"] == case and row["method"] == method and abs(float(row["scale"]) - scale) < 1e-9:
            value = float(row[metric])
            if math.isfinite(value):
                out.append(value)
    return out


def write_summary_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["case", "scale", "method", "metric", "mean", "ci_low", "ci_high", "n"]
    rng = np.random.default_rng(20260811)
    out: list[dict[str, object]] = []
    for case in CASES:
        for method in METHODS:
            for scale in SCALES:
                for metric in ("nrmse_rel", "crps_rel", "coverage", "alpha_error"):
                    vals = values(rows, case, method, scale, metric)
                    mean, lo, hi = bootstrap_mean_ci(vals, rng)
                    if math.isfinite(mean):
                        out.append(
                            {
                                "case": case,
                                "scale": scale,
                                "method": method,
                                "metric": metric,
                                "mean": mean,
                                "ci_low": lo,
                                "ci_high": hi,
                                "n": len(vals),
                            }
                        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out)


def fmt(value: float, digits: int = 3) -> str:
    if not math.isfinite(value):
        return "--"
    if value == 0:
        return "0"
    return f"{value:.{digits}g}"


def write_latex_table(summary_rows: list[dict[str, str]], path: Path) -> None:
    wanted_methods = ("bma_static", "pce", "apce")
    by_key = {(r["case"], float(r["scale"]), r["method"]): r for r in summary_rows}
    # The printed table focuses on the evidence methods and the strongest
    # reviewer-risk comparator; the full 105-row source table remains in CSV.
    lines = [
        r"{\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{lllrrrrrr}",
        r"\caption{\textbf{Sensitivity of the key methods to the cognitive scale.} Values are means across 50 paired seeds; the complete seven-method source table is provided as machine-readable source data.}\label{tab:stheta-sensitivity}\\",
        r"\toprule",
        r"System & Scale & Method & nRMSE & CRPS & 90\% coverage & 90\% width & \(|\hat{\alpha}-\alpha^\star|\) & Runtime (s) \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"System & Scale & Method & nRMSE & CRPS & 90\% coverage & 90\% width & \(|\hat{\alpha}-\alpha^\star|\) & Runtime (s) \\",
        r"\midrule",
        r"\endhead",
    ]
    # Runtime and width are read from the original summary CSV, which is
    # joined into this table by the plotting driver.
    for case in CASES:
        for scale in SCALES:
            for method in wanted_methods:
                r = by_key.get((case, scale, method))
                if r is None:
                    continue
                name = LABELS[method]
                if method == "pce":
                    name = r"\textbf{\PCE{}}"
                elif method == "apce":
                    name = r"\textbf{\APCE{}}"
                lines.append(
                    f"{case.title()} & {scale:g} & {name} & {r['nrmse']} & {r['crps']} & {r['coverage']} & {r['width']} & {r['alpha']} & {r['runtime']} \\\\"
                )
            if scale != SCALES[-1]:
                lines.append(r"\addlinespace[1pt]")
        if case != CASES[-1]:
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{longtable}",
        r"}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def make_figure(rows: list[dict[str, float | str]], output: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
        }
    )
    fig, axes = plt.subplots(
        3,
        4,
        figsize=(7.20, 5.35),
        sharex=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.095, top=0.875, wspace=0.30, hspace=0.44)
    columns = (
        ("nrmse_rel", r"Relative nRMSE"),
        ("crps_rel", r"Relative CRPS"),
        ("coverage", r"90% coverage"),
        ("alpha_error", r"$|\hat{\alpha}-\alpha^\star|$"),
    )
    rng = np.random.default_rng(20260811)
    letters = iter("abcdefghijkl")
    for i, case in enumerate(CASES):
        for j, (metric, title) in enumerate(columns):
            ax = axes[i, j]
            ax.text(-0.22, 1.06, next(letters), transform=ax.transAxes, ha="left", va="bottom", fontweight="bold", fontsize=8)
            if i == 0:
                ax.set_title(title, pad=4)
            if j == 0:
                ax.text(-0.35, 0.50, case.title(), transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=7.5)
            for method in METHODS:
                if metric == "alpha_error" and method not in ALPHA_METHODS:
                    continue
                means, lows, highs = [], [], []
                for scale in SCALES:
                    m, lo, hi = bootstrap_mean_ci(values(rows, case, method, scale, metric), rng)
                    means.append(m)
                    lows.append(lo)
                    highs.append(hi)
                if not any(math.isfinite(v) for v in means):
                    continue
                color = COLORS[method]
                lw = LINE_WIDTH[method]
                alpha = 0.19 if method in ("pce", "apce") else 0.10
                ax.fill_between(SCALES, lows, highs, color=color, alpha=alpha, linewidth=0, zorder=1)
                ax.plot(
                    SCALES,
                    means,
                    color=color,
                    linewidth=lw,
                    marker="o" if method in ("pce", "apce") else None,
                    markersize=2.4 if method in ("pce", "apce") else 0,
                    solid_capstyle="round",
                    zorder=3 if method in ("pce", "apce") else 2,
                    label=LABELS[method],
                )
            if metric in ("nrmse_rel", "crps_rel"):
                ax.axhline(1.0, color="#B8BDC4", linewidth=0.55, linestyle=(0, (2, 2)), zorder=0)
                ax.set_ylim(bottom=0)
            elif metric == "coverage":
                ax.axhline(0.90, color="#B8BDC4", linewidth=0.55, linestyle=(0, (2, 2)), zorder=0)
                ax.set_ylim(0.0, 1.02)
                ax.set_yticks([0.0, 0.5, 0.9, 1.0])
            else:
                ax.set_ylim(bottom=0)
            ax.set_xlim(0.47, 1.53)
            ax.set_xticks(SCALES)
            ax.set_xticklabels(["0.5", "0.75", "1", "1.25", "1.5"])
            if i == len(CASES) - 1:
                ax.set_xlabel(r"$s_\theta/s_{\theta,0}$")
            else:
                ax.tick_params(labelbottom=False)
            ax.grid(False)
    handles = []
    labels = []
    for method in METHODS:
        line = mpl.lines.Line2D([], [], color=COLORS[method], lw=LINE_WIDTH[method], marker="o" if method in ("pce", "apce") else None, markersize=2.6)
        handles.append(line)
        labels.append(LABELS[method])
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.955),
        ncol=7,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.0,
        handletextpad=0.35,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the formal s_theta sensitivity figure.")
    parser.add_argument("--input", type=Path, required=True, help="Run-level source-data CSV.")
    parser.add_argument("--summary-input", type=Path, required=True, help="Aggregate summary CSV.")
    parser.add_argument("--output", type=Path, required=True, help="Output basename without extension.")
    parser.add_argument("--summary-output", type=Path, required=True, help="Long-form figure source-data CSV.")
    parser.add_argument("--table-output", type=Path, required=True, help="Supplementary LaTeX table fragment.")
    args = parser.parse_args()

    raw = read_csv(args.input)
    rows = build_rows(raw)
    write_summary_csv(rows, args.summary_output)
    summary_raw = read_csv(args.summary_input)
    # Join the key absolute metrics from the aggregate summary to make the
    # compact supplementary table auditable without recomputing from plots.
    lookup = {(r["case"], float(r["sensitivity_scale"]), r["method"]): r for r in summary_raw}
    table_rows = []
    for case in CASES:
        for scale in SCALES:
            for method in ("bma_static", "pce", "apce"):
                r = lookup[(case, scale, method)]
                table_rows.append(
                    {
                        "case": case,
                        "scale": scale,
                        "method": method,
                        "nrmse": fmt(float(r["nrmse_mean"])),
                        "crps": fmt(float(r["crps_mean"])),
                        "coverage": fmt(float(r["coverage_90_mean"])),
                        "width": fmt(float(r["interval_width_90_mean"])),
                        "alpha": fmt(float(r["alpha_absolute_error_mean"])),
                        "runtime": fmt(float(r["runtime_seconds_mean"])),
                    }
                )
    write_latex_table(table_rows, args.table_output)
    make_figure(rows, args.output)


if __name__ == "__main__":
    main()
