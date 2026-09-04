from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
import numpy as np


CASES = ("wave", "spring", "heat")
CASE_LABELS = {"wave": "Wave", "spring": "Spring", "heat": "Heat"}
METHODS = ("denkf", "letkf", "iensf", "pce", "apce", "oracle_alpha")
METHOD_LABELS = {
    "denkf": "DEnKF",
    "letkf": "LETKF",
    "iensf": "IEnSF",
    "pce": "PCE",
    "apce": "APCE",
    "oracle_alpha": "Oracle-alpha",
}
COLORS = {
    "truth": "#202020",
    "denkf": "#777777",
    "letkf": "#B5B5B5",
    "iensf": "#7E84B8",
    "pce": "#3F6EA8",
    "apce": "#D99035",
    "oracle_alpha": "#4F8A58",
}
CASE_COLORS = {"wave": "#3F6EA8", "spring": "#8A6BAE", "heat": "#4F8A58"}

METRICS = (
    "nrmse",
    "crps",
    "coverage_error",
    "interval_width",
    "alpha_error",
)
METRIC_LABELS = {
    "nrmse": "nRMSE",
    "crps": "CRPS",
    "coverage_error": r"$|$coverage - 0.90$|$",
    "interval_width": "90% width",
    "alpha_error": r"$|\hat{\alpha}-\alpha^\ast|$",
}
LOG_METRICS = {"nrmse", "crps", "interval_width"}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6.0,
            "font.weight": "regular",
            "axes.titleweight": "regular",
            "axes.labelweight": "regular",
            "mathtext.default": "it",
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.60,
            "legend.frameon": False,
            "xtick.major.width": 0.50,
            "ytick.major.width": 0.50,
            "xtick.major.size": 1.8,
            "ytick.major.size": 1.8,
            "lines.solid_capstyle": "round",
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def add_panel(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(-0.055, 1.035, letter, transform=ax.transAxes, fontsize=8.0, va="bottom", ha="left")
    ax.text(0.0, 1.035, title, transform=ax.transAxes, fontsize=6.8, va="bottom", ha="left")


def _phase(ax: plt.Axes, truth: np.ndarray, estimates: dict[str, np.ndarray], title: str, u_idx: int, v_idx: int) -> None:
    ax.plot(truth[:, u_idx], truth[:, v_idx], color=COLORS["truth"], lw=1.0)
    for method, states in estimates.items():
        ax.plot(states[:, u_idx], states[:, v_idx], color=COLORS[method], lw=0.78)
    ax.set_title(title, fontsize=5.8, pad=1.2, fontweight="regular")
    ax.set_xlabel(r"$u$", fontsize=5.1)
    ax.set_ylabel(r"$v$", fontsize=5.1)
    ax.grid(color="#E9E9E9", lw=0.35)
    ax.tick_params(labelsize=4.9)
    ax.margins(0.05)


def representative_wall(ax_container: plt.Axes, wave: np.lib.npyio.NpzFile, spring: np.lib.npyio.NpzFile, heat: np.lib.npyio.NpzFile) -> None:
    ax_container.set_axis_off()
    add_panel(ax_container, "a", "Representative dynamics and residuals")
    sub = GridSpecFromSubplotSpec(2, 3, subplot_spec=ax_container.get_subplotspec(), hspace=0.38, wspace=0.30)

    wave_truth = wave["truth_states"]
    wave_nx = wave_truth.shape[1] // 2
    wave_node = wave_nx // 2
    wave_est = {
        "denkf": wave["denkf_mean_states"],
        "pce": wave["pce_mean_states"],
        "apce": wave["apce_mean_states"],
    }
    ax = ax_container.figure.add_subplot(sub[0, 0])
    _phase(ax, wave_truth, wave_est, r"Wave phase, $x_c$", wave_node, wave_nx + wave_node)
    ax = ax_container.figure.add_subplot(sub[1, 0])
    times = wave["times"]
    ax.plot(times, wave_truth[:, wave_node], color=COLORS["truth"], lw=1.0)
    for method, states in wave_est.items():
        ax.plot(times, states[:, wave_node], color=COLORS[method], lw=0.78)
    ax.set_title(r"Wave centre displacement, $u(x_c,t)$", fontsize=5.8, pad=1.2, fontweight="regular")
    ax.set_xlabel(r"$t$", fontsize=5.1)
    ax.set_ylabel(r"$u$", fontsize=5.1)
    ax.grid(color="#E9E9E9", lw=0.35)
    ax.tick_params(labelsize=4.9)

    spring_truth = spring["truth_states"]
    spring_est = {
        "denkf": spring["denkf_mean_states"],
        "pce": spring["pce_mean_states"],
        "apce": spring["apce_mean_states"],
    }
    ax = ax_container.figure.add_subplot(sub[0, 1])
    _phase(ax, spring_truth, spring_est, r"Spring phase", 0, 1)
    ax = ax_container.figure.add_subplot(sub[1, 1])
    spring_times = spring["times"]
    ax.plot(spring_times, spring_truth[:, 0], color=COLORS["truth"], lw=1.0)
    for method, states in spring_est.items():
        ax.plot(spring_times, states[:, 0], color=COLORS[method], lw=0.78)
    ax.set_title(r"Spring displacement, $x(t)$", fontsize=5.8, pad=1.2, fontweight="regular")
    ax.set_xlabel(r"$t$", fontsize=5.1)
    ax.set_ylabel(r"$x$", fontsize=5.1)
    ax.grid(color="#E9E9E9", lw=0.35)
    ax.tick_params(labelsize=4.9)

    heat_x = heat["space"]
    heat_truth = heat["truth_states"][-1]
    heat_est = {
        "denkf": heat["denkf_mean_states"][-1],
        "pce": heat["pce_mean_states"][-1],
        "apce": heat["apce_mean_states"][-1],
    }
    ax = ax_container.figure.add_subplot(sub[0, 2])
    ax.plot(heat_x, heat_truth, color=COLORS["truth"], lw=1.0)
    for method, values in heat_est.items():
        ax.plot(heat_x, values, color=COLORS[method], lw=0.78)
    obs_x = heat_x[heat["observation_indices"]]
    ax.scatter(obs_x, heat_truth[heat["observation_indices"]], s=5, facecolor="white", edgecolor="#202020", lw=0.35, zorder=4)
    ax.set_title(r"Heat terminal profile, $u(x,t_f)$", fontsize=5.8, pad=1.2, fontweight="regular")
    ax.set_xlabel(r"$x$", fontsize=5.1)
    ax.set_ylabel(r"$u$", fontsize=5.1)
    ax.grid(color="#E9E9E9", lw=0.35)
    ax.tick_params(labelsize=4.9)

    ax = ax_container.figure.add_subplot(sub[1, 2])
    ax.axhline(0.0, color="#555555", lw=0.55, ls=(0, (2, 2)))
    for method, values in heat_est.items():
        ax.plot(heat_x, values - heat_truth, color=COLORS[method], lw=0.78)
    ax.set_title(r"Heat terminal residual, $\hat{u}-u^\ast$", fontsize=5.8, pad=1.2, fontweight="regular")
    ax.set_xlabel(r"$x$", fontsize=5.1)
    ax.set_ylabel("error", fontsize=5.1)
    ax.grid(color="#E9E9E9", lw=0.35)
    ax.tick_params(labelsize=4.9)


def metric_value(row: dict[str, str], metric: str) -> float:
    if metric == "nrmse":
        return float(row["nrmse_mean"])
    if metric == "crps":
        return float(row["crps_mean"])
    if metric == "coverage_error":
        return abs(float(row["coverage_90_mean"]) - 0.90)
    if metric == "interval_width":
        return float(row["interval_width_90_mean"])
    if metric == "alpha_error":
        return float(row["alpha_absolute_error_mean"])
    raise KeyError(metric)


def metric_error(row: dict[str, str], metric: str) -> tuple[float, float]:
    if metric == "coverage_error":
        mean = float(row["coverage_90_mean"])
        low = float(row["coverage_90_ci95_low"])
        high = float(row["coverage_90_ci95_high"])
        val = abs(mean - 0.90)
        bounds = [abs(low - 0.90), abs(high - 0.90)]
        return max(0.0, val - min(bounds)), max(0.0, max(bounds) - val)
    base = "nrmse" if metric == "nrmse" else metric
    if metric == "interval_width":
        base = "interval_width_90"
    if metric == "alpha_error":
        base = "alpha_absolute_error"
    return (
        max(0.0, metric_value(row, metric) - float(row[f"{base}_ci95_low"])),
        max(0.0, float(row[f"{base}_ci95_high"]) - metric_value(row, metric)),
    )


def metric_wall(ax_container: plt.Axes, summary: dict[tuple[str, str], dict[str, str]]) -> None:
    ax_container.set_axis_off()
    add_panel(ax_container, "b", "Cross-system metric wall; 50 paired seeds per cell")
    sub = GridSpecFromSubplotSpec(3, len(METRICS), subplot_spec=ax_container.get_subplotspec(), hspace=0.32, wspace=0.14)
    column_limits: dict[str, tuple[float, float]] = {}
    for metric in METRICS:
        vals = [metric_value(summary[(case, method)], metric) for case in CASES for method in METHODS]
        lo = min(vals)
        hi = max(vals)
        if metric in LOG_METRICS:
            column_limits[metric] = (max(lo * 0.75, 1.0e-8), hi * 1.30)
        else:
            column_limits[metric] = (0.0, hi * 1.25 if hi > 0 else 1.0)

    for row_idx, case in enumerate(CASES):
        for col_idx, metric in enumerate(METRICS):
            ax = ax_container.figure.add_subplot(sub[row_idx, col_idx])
            means = [metric_value(summary[(case, method)], metric) for method in METHODS]
            lows, highs = [], []
            for method in METHODS:
                lo, hi = metric_error(summary[(case, method)], metric)
                lows.append(lo)
                highs.append(hi)
            x = np.arange(len(METHODS))
            bars = ax.bar(
                x,
                means,
                width=0.72,
                color=[COLORS[m] for m in METHODS],
                edgecolor="#202020",
                linewidth=0.20,
                yerr=np.vstack([lows, highs]),
                error_kw={"elinewidth": 0.42, "capthick": 0.42, "capsize": 1.0},
                zorder=3,
            )
            for bar, method in zip(bars, METHODS):
                if method == "apce":
                    bar.set_edgecolor(COLORS["apce"])
                    bar.set_linewidth(0.85)
                elif method == "pce":
                    bar.set_edgecolor(COLORS["pce"])
                    bar.set_linewidth(0.75)
            if row_idx == 0:
                ax.set_title(METRIC_LABELS[metric], fontsize=5.3, pad=1.2, fontweight="regular")
            if col_idx == 0:
                ax.set_ylabel(CASE_LABELS[case], fontsize=5.4, labelpad=2)
            else:
                ax.set_yticklabels([])
            if row_idx == len(CASES) - 1:
                ax.set_xticks(x)
                ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=60, ha="right", fontsize=3.8)
            else:
                ax.set_xticks([])
            if metric in LOG_METRICS:
                ax.set_yscale("log")
                ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=3))
                ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10, labelOnlyBase=False))
            ax.set_ylim(column_limits[metric])
            ax.grid(axis="y", color="#E9E9E9", lw=0.30, zorder=0)
            ax.tick_params(labelsize=4.0, length=1.4)


def paired_mini(ax: plt.Axes, rows: list[dict[str, str]], case: str, metric: str, show_y: bool, show_x: bool) -> None:
    selected = [r for r in rows if r["case"] == case and r["metric"] == metric and r["baseline"] in {"denkf", "letkf", "iensf"}]
    order = ("denkf", "letkf", "iensf")
    by_method = {r["baseline"]: r for r in selected}
    y = np.arange(len(order))[::-1]
    for yi, baseline in zip(y, order):
        r = by_method[baseline]
        est = float(r["mean_difference_apce_minus_baseline"])
        lo = float(r["ci95_low"])
        hi = float(r["ci95_high"])
        ax.plot([lo, hi], [yi, yi], color=CASE_COLORS[case], lw=0.85)
        ax.scatter(est, yi, color=CASE_COLORS[case], edgecolor="#202020", linewidth=0.25, s=11, zorder=3)
        if float(r["p_holm"]) < 0.05:
            ax.text(hi, yi + 0.12, "*", fontsize=5.0, ha="center", va="bottom")
    ax.axvline(0.0, color="#555555", lw=0.50, ls=(0, (2, 2)))
    ax.set_xlim(-0.50 if metric == "nrmse" else -0.034, 0.002)
    ax.set_ylim(-0.5, 2.5)
    if show_y:
        ax.set_yticks(y)
        ax.set_yticklabels(["DEnKF", "LETKF", "IEnSF"], fontsize=4.2)
    else:
        ax.set_yticks([])
    if show_x:
        ax.set_xlabel(r"$\Delta$" + (" nRMSE" if metric == "nrmse" else " CRPS"), fontsize=4.7)
        ax.tick_params(axis="x", labelsize=4.0)
    else:
        ax.set_xticklabels([])
    ax.grid(axis="x", color="#E9E9E9", lw=0.3)


def paired_wall(ax_container: plt.Axes, paired: list[dict[str, str]]) -> None:
    ax_container.set_axis_off()
    add_panel(ax_container, "c", "Paired APCE advantage; stars denote Holm-adjusted $p<0.05$")
    sub = GridSpecFromSubplotSpec(2, 3, subplot_spec=ax_container.get_subplotspec(), hspace=0.35, wspace=0.16)
    for row_idx, metric in enumerate(("nrmse", "crps")):
        for col_idx, case in enumerate(CASES):
            ax = ax_container.figure.add_subplot(sub[row_idx, col_idx])
            paired_mini(ax, paired, case, metric, show_y=col_idx == 0, show_x=row_idx == 1)
            if row_idx == 0:
                ax.set_title(CASE_LABELS[case], fontsize=5.2, pad=1.0, fontweight="regular")
            if col_idx == 0:
                ax.text(-0.26, 0.5, "nRMSE" if metric == "nrmse" else "CRPS", transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=5.0)


def alpha_calibration_wall(ax_container: plt.Axes, summary: dict[tuple[str, str], dict[str, str]], wave: np.lib.npyio.NpzFile, spring: np.lib.npyio.NpzFile, heat: np.lib.npyio.NpzFile) -> None:
    ax_container.set_axis_off()
    add_panel(ax_container, "d", "Cognitive-orbit identification and calibration")
    sub = GridSpecFromSubplotSpec(2, 3, subplot_spec=ax_container.get_subplotspec(), hspace=0.38, wspace=0.24)
    weight_data = [
        ("Wave", wave["pce_final_weights"], wave["apce_final_weights"], np.linspace(0.08, 0.92, len(wave["pce_final_weights"]))),
        ("Spring", spring["pce_alpha_weight_history"][-1], spring["apce_alpha_weight_history"][-1], spring["alpha_grid"]),
        ("Heat", heat["pce_alpha_weight_history"][-1], heat["apce_alpha_weight_history"][-1], heat["alpha_grid"]),
    ]
    for col_idx, (label, pce, apce, alpha) in enumerate(weight_data):
        ax = ax_container.figure.add_subplot(sub[0, col_idx])
        ax.plot(alpha, pce, color=COLORS["pce"], lw=0.85, ls="--", marker="o", ms=2.0, label="PCE")
        ax.plot(alpha, apce, color=COLORS["apce"], lw=0.95, ls="-", marker="o", ms=2.0, label="APCE")
        ax.set_title(f"{label} weights", fontsize=5.2, pad=1.0, fontweight="regular")
        ax.set_xlabel(r"$\alpha$", fontsize=4.8)
        if col_idx == 0:
            ax.set_ylabel("weight", fontsize=4.8)
        else:
            ax.set_yticklabels([])
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=4.2)
        ax.grid(axis="y", color="#E9E9E9", lw=0.3)
        if col_idx == 2:
            ax.legend(fontsize=4.0, loc="upper right", handlelength=0.8, borderpad=0.1)

    for col_idx, case in enumerate(CASES):
        ax = ax_container.figure.add_subplot(sub[1, col_idx])
        for method in METHODS:
            row = summary[(case, method)]
            x = float(row["coverage_90_mean"])
            y = float(row["interval_width_90_mean"])
            ax.scatter(x, y, color=COLORS[method], s=18 if method in {"pce", "apce"} else 13, edgecolor="#202020" if method in {"pce", "apce"} else "none", linewidth=0.25)
        ax.axvline(0.90, color="#555555", lw=0.50, ls=(0, (2, 2)))
        ax.set_title(f"{CASE_LABELS[case]} calibration", fontsize=5.2, pad=1.0, fontweight="regular")
        ax.set_xlim(0.35, 1.02)
        ax.set_yscale("log")
        ax.set_xlabel("coverage", fontsize=4.8)
        if col_idx == 0:
            ax.set_ylabel("width", fontsize=4.8)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=4.2)
        ax.grid(color="#E9E9E9", lw=0.3)


def write_contract(output_dir: Path) -> None:
    text = """Core conclusion:
APCE improves state error, probabilistic skill, interval quality and cognitive-parameter identification across Wave, Spring and Heat under a frozen 50-seed paired protocol.

Figure archetype:
Asymmetric mixed-modality evidence wall with nested small multiples.

Panel map:
a: Six representative dynamics/residual micro-panels.
b: 3 x 5 cross-system metric wall; each micro-panel compares six methods.
c: 2 x 3 paired APCE advantage wall for nRMSE and CRPS.
d: 2 x 3 cognitive-orbit and calibration wall.

Statistics:
Metric bars are means with bootstrap 95% confidence intervals; n=50 paired seeds. Paired mini-forests show APCE-minus-baseline differences with paired bootstrap intervals and Holm-adjusted significance.

Main figure exclusions:
Runtime, memory and Wave displacement fields remain in source data or Supplementary Information.
"""
    (output_dir / "figure2_classical_uncertain_systems_v4_contract.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--paired-csv", type=Path, required=True)
    parser.add_argument("--wave-npz", type=Path, required=True)
    parser.add_argument("--spring-npz", type=Path, required=True)
    parser.add_argument("--heat-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    set_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = read_csv(args.summary_csv)
    paired_rows = read_csv(args.paired_csv)
    summary = {(r["case"], r["method"]): r for r in summary_rows}
    wave = np.load(args.wave_npz, allow_pickle=True)
    spring = np.load(args.spring_npz, allow_pickle=True)
    heat = np.load(args.heat_npz, allow_pickle=True)

    fig = plt.figure(figsize=(8.6, 10.8))
    outer = fig.add_gridspec(
        4,
        1,
        height_ratios=[2.0, 4.15, 2.0, 2.05],
        hspace=0.48,
        left=0.055,
        right=0.985,
        top=0.955,
        bottom=0.085,
    )
    host_a = fig.add_subplot(outer[0, 0])
    representative_wall(host_a, wave, spring, heat)
    host_b = fig.add_subplot(outer[1, 0])
    metric_wall(host_b, summary)
    host_c = fig.add_subplot(outer[2, 0])
    paired_wall(host_c, paired_rows)
    host_d = fig.add_subplot(outer[3, 0])
    alpha_calibration_wall(host_d, summary, wave, spring, heat)

    handles = [Line2D([0], [0], color=COLORS[m], lw=1.6, label=METHOD_LABELS[m]) for m in METHODS]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.988),
        ncol=6,
        fontsize=5.3,
        handlelength=1.2,
        columnspacing=0.72,
    )
    base = args.output_dir / "figure2_classical_uncertain_systems_v4"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    for source in (args.summary_csv, args.paired_csv):
        shutil.copy2(source, args.output_dir / f"source_data_{source.name}")
    write_contract(args.output_dir)
    qa = {
        "figure": "figure2_classical_uncertain_systems_v4",
        "backend": "python/matplotlib",
        "font": "Arial with sans-serif fallback",
        "font_weight": "regular",
        "svg_text_editable": True,
        "statistics": "n=50 paired seeds; bootstrap 95% CI; paired APCE-minus-baseline forest plots; Holm-adjusted comparisons",
        "formats": ["svg", "pdf", "png", "tiff"],
        "panels": ["a", "b", "c", "d"],
        "micro_panels": {"a": 6, "b": 15, "c": 6, "d": 6},
    }
    (args.output_dir / "figure2_classical_uncertain_systems_v4_qa.json").write_text(
        json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"output": str(base), "micro_panels": qa["micro_panels"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
