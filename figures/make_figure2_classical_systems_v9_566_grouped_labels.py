from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import numpy as np


CASES = ("wave", "spring", "heat")
CASE_LABELS = {"wave": "Wave", "spring": "Spring", "heat": "Heat"}
METHODS = ("denkf", "letkf", "iensf", "pce", "apce")
METHOD_LABELS = {
    "denkf": "DEnKF",
    "letkf": "LETKF",
    "iensf": "IEnSF",
    "pce": "PCE",
    "apce": "APCE",
}
METRICS = (
    ("nrmse", r"nRMSE (%)", 100.0),
    ("crps", r"CRPS ($10^{-3}$)", 1000.0),
)

# Reference-image-style statistical palette:
# blue / teal / brown / purple / yellow, with APCE highlighted by a red frame.
COLORS = {
    "truth": "#1F1F1F",
    "denkf": "#1F77B4",
    "letkf": "#008B8B",
    "iensf": "#8A6A4B",
    "pce": "#7B4DA0",
    "apce": "#F2C94C",
}
APCE_FRAME = "#E83E8C"
BASELINE_ALPHA = {"denkf": 0.20, "letkf": 0.20, "iensf": 0.26, "pce": 0.92, "apce": 1.00}
LINE_WIDTH = {"truth": 1.35, "denkf": 0.65, "letkf": 0.65, "iensf": 0.70, "pce": 1.05, "apce": 1.35}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.6,
            "font.weight": "regular",
            "axes.titleweight": "regular",
            "axes.labelweight": "regular",
            "mathtext.default": "it",
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "legend.frameon": False,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "lines.solid_capstyle": "round",
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def add_panel(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(-0.085, 1.055, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=9.0)
    ax.text(0.0, 1.055, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.0)


def add_small_title(ax: plt.Axes, title: str) -> None:
    ax.text(0.0, 1.055, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.0)


def polish_axis(ax: plt.Axes, tick_size: float = 7.0, label_size: float = 8.0) -> None:
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(labelsize=tick_size, length=2.2, pad=1.8)
    ax.xaxis.label.set_size(label_size)
    ax.yaxis.label.set_size(label_size)
    ax.grid(False)


def add_top_legend(fig: plt.Figure) -> None:
    handles = [Line2D([0], [0], color=COLORS["truth"], lw=1.35, label="Truth")]
    handles += [
        Line2D([0], [0], color=COLORS[m], lw=1.1 if m in {"pce", "apce"} else 0.85, alpha=BASELINE_ALPHA[m], label=METHOD_LABELS[m])
        for m in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.995),
        ncol=6,
        fontsize=7.6,
        handlelength=1.25,
        columnspacing=0.75,
        borderpad=0.08,
        labelspacing=0.12,
    )


def _phase_panel(
    ax: plt.Axes,
    data: np.lib.npyio.NpzFile,
    letter: str,
    title: str,
    u_idx: int,
    v_idx: int,
) -> None:
    add_panel(ax, letter, title)
    truth = data["truth_states"]
    ax.plot(truth[:, u_idx], truth[:, v_idx], color=COLORS["truth"], lw=LINE_WIDTH["truth"], zorder=4)
    for method in METHODS:
        states = data[f"{method}_mean_states"]
        ax.plot(
            states[:, u_idx],
            states[:, v_idx],
            color=COLORS[method],
            lw=LINE_WIDTH[method],
            alpha=BASELINE_ALPHA[method],
            zorder=6 if method == "apce" else (5 if method == "pce" else 2),
        )
    focus = [truth, data["pce_mean_states"], data["apce_mean_states"]]
    xs = np.concatenate([arr[:, u_idx] for arr in focus])
    ys = np.concatenate([arr[:, v_idx] for arr in focus])
    xlo, xhi = np.percentile(xs, [0.5, 99.5])
    ylo, yhi = np.percentile(ys, [0.5, 99.5])
    ax.set_xlim(xlo - 0.08 * (xhi - xlo), xhi + 0.08 * (xhi - xlo))
    ax.set_ylim(ylo - 0.08 * (yhi - ylo), yhi + 0.08 * (yhi - ylo))
    ax.set_xlabel(r"$u(t)$")
    ax.set_ylabel(r"$v(t)$")
    polish_axis(ax)


def panel_wave(ax: plt.Axes, data: np.lib.npyio.NpzFile) -> None:
    nx = data["truth_states"].shape[1] // 2
    node = nx // 2
    _phase_panel(ax, data, "a", r"Wave phase at $x_c$", node, nx + node)


def panel_spring(ax: plt.Axes, data: np.lib.npyio.NpzFile) -> None:
    _phase_panel(ax, data, "b", "Spring phase", 0, 1)


def panel_heat(ax: plt.Axes, data: np.lib.npyio.NpzFile) -> None:
    add_panel(ax, "c", "Heat terminal profile")
    x = data["space"]
    truth = data["truth_states"][-1]
    ax.plot(x, truth, color=COLORS["truth"], lw=LINE_WIDTH["truth"], zorder=4)
    for method in METHODS:
        y = data[f"{method}_mean_states"][-1]
        ax.plot(x, y, color=COLORS[method], lw=LINE_WIDTH[method], alpha=BASELINE_ALPHA[method], zorder=6 if method == "apce" else 3)
    obs = data["observation_indices"]
    ax.scatter(x[obs], truth[obs], s=9.0, facecolor="white", edgecolor=COLORS["truth"], linewidth=0.55, zorder=7)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u(x,t_f)$")
    inset = ax.inset_axes([0.29, 0.065, 0.42, 0.29])
    lo = int(np.searchsorted(x, 0.31))
    hi = int(np.searchsorted(x, 0.53))
    inset.plot(x[lo:hi], truth[lo:hi], color=COLORS["truth"], lw=0.95, zorder=4)
    for method in METHODS:
        y = data[f"{method}_mean_states"][-1]
        inset.plot(x[lo:hi], y[lo:hi], color=COLORS[method], lw=LINE_WIDTH[method] * 0.78, alpha=BASELINE_ALPHA[method], zorder=6 if method == "apce" else 3)
    y_slice = np.concatenate([truth[lo:hi]] + [data[f"{m}_mean_states"][-1][lo:hi] for m in METHODS])
    y_pad = max(0.05 * (float(np.max(y_slice)) - float(np.min(y_slice))), 1.0e-12)
    inset.set_xlim(float(x[lo]), float(x[hi - 1]))
    inset.set_ylim(float(np.min(y_slice)) - y_pad, float(np.max(y_slice)) + y_pad)
    inset.set_xticks([])
    inset.set_yticks([])
    for side in ("left", "bottom", "right", "top"):
        inset.spines[side].set_visible(True)
        inset.spines[side].set_color("#B8B8B8")
        inset.spines[side].set_linewidth(0.55)
    mark_inset(ax, inset, loc1=1, loc2=2, fc="none", ec="#B8B8B8", linewidth=0.60)
    polish_axis(ax)


def panel_calibration(ax: plt.Axes, summary: dict[tuple[str, str], dict[str, str]]) -> None:
    add_panel(ax, "d", "Calibration")
    markers = {"wave": "o", "spring": "s", "heat": "^"}
    for case in CASES:
        for method in METHODS:
            row = summary[(case, method)]
            ax.scatter(
                float(row["coverage_90_mean"]),
                float(row["interval_width_90_mean"]),
                s=24 if method in {"pce", "apce"} else 18,
                marker=markers[case],
                color=COLORS[method],
                alpha=0.88 if method in {"pce", "apce"} else 0.62,
                edgecolor=APCE_FRAME if method == "apce" else ("#222222" if method == "pce" else "none"),
                linewidth=0.65 if method == "apce" else 0.35,
                zorder=5,
            )
    ax.axvline(0.90, color="#606060", lw=0.75, ls=(0, (3, 2)))
    ax.set_yscale("log")
    ax.set_xlim(0.35, 1.02)
    ax.set_xlabel("90% coverage")
    ax.set_ylabel("interval width")
    handles = [Line2D([0], [0], marker=markers[c], color="none", markerfacecolor="#6B6B6B", markersize=5.2, label=CASE_LABELS[c]) for c in CASES]
    ax.legend(handles=handles, loc="upper left", fontsize=6.3, ncol=3, handlelength=0.75, handletextpad=0.25, columnspacing=0.55, borderpad=0.05)
    polish_axis(ax, tick_size=6.8, label_size=7.6)


def panel_weights(ax: plt.Axes, wave: np.lib.npyio.NpzFile, spring: np.lib.npyio.NpzFile, heat: np.lib.npyio.NpzFile) -> None:
    add_panel(ax, "e", "Cognitive weights")
    case_colors = {"wave": "#1F77B4", "spring": "#7B4DA0", "heat": "#4F8A58"}
    series = [
        ("wave", wave["pce_final_weights"], wave["apce_final_weights"], np.linspace(0.08, 0.92, len(wave["pce_final_weights"]))),
        ("spring", spring["pce_alpha_weight_history"][-1], spring["apce_alpha_weight_history"][-1], spring["alpha_grid"]),
        ("heat", heat["pce_alpha_weight_history"][-1], heat["apce_alpha_weight_history"][-1], heat["alpha_grid"]),
    ]
    for case, pce, apce, alpha in series:
        mask = alpha <= 0.5
        ax.plot(alpha[mask], pce[mask], color=case_colors[case], lw=0.95, ls=(0, (3, 2)), alpha=0.42)
        ax.plot(alpha[mask], apce[mask], color=case_colors[case], lw=1.55, alpha=0.95)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1.06)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("final weight")
    handles = [
        Line2D([0], [0], color="#606060", lw=0.95, ls=(0, (3, 2)), label="PCE"),
        Line2D([0], [0], color="#606060", lw=1.55, label="APCE"),
        Line2D([0], [0], color=case_colors["wave"], lw=1.2, label="Wave"),
        Line2D([0], [0], color=case_colors["spring"], lw=1.2, label="Spring"),
        Line2D([0], [0], color=case_colors["heat"], lw=1.2, label="Heat"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=6.1, ncol=1, handlelength=1.0, borderpad=0.04, labelspacing=0.12)
    polish_axis(ax, tick_size=6.8, label_size=7.6)


def scaled_seed_values(rows: list[dict[str, str]], case: str, method: str, metric: str, scale: float) -> np.ndarray:
    return np.asarray(
        [
            float(row[metric]) * scale
            for row in rows
            if row["case"] == case and row["method"] == method and row.get("valid", "True") == "True" and row.get(metric, "")
        ],
        dtype=float,
    )


def summary_mean_ci(summary: dict[tuple[str, str], dict[str, str]], case: str, method: str, metric: str, scale: float) -> tuple[float, float, float]:
    row = summary[(case, method)]
    mean = float(row[f"{metric}_mean"]) * scale
    low = float(row[f"{metric}_ci95_low"]) * scale
    high = float(row[f"{metric}_ci95_high"]) * scale
    return mean, mean - low, high - mean


def apce_highlight(ax: plt.Axes, x_center: float, y_min: float, y_max: float, width: float = 0.72) -> None:
    ax.add_patch(
        Rectangle(
            (x_center - width / 2.0, y_min),
            width,
            y_max - y_min,
            fill=False,
            edgecolor=APCE_FRAME,
            linewidth=1.05,
            linestyle=(0, (3, 2)),
            zorder=10,
            clip_on=False,
        )
    )


def box_panel(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    case: str,
    metric: str,
    ylabel: str,
    scale: float,
    letter: str,
    title: str,
) -> None:
    if letter:
        add_panel(ax, letter, title)
    else:
        add_small_title(ax, title)
    data = [scaled_seed_values(rows, case, method, metric, scale) for method in METHODS]
    positions = np.arange(1, len(METHODS) + 1)
    bp = ax.boxplot(data, positions=positions, widths=0.56, patch_artist=True, showfliers=False, whis=1.5)
    for patch, method in zip(bp["boxes"], METHODS, strict=True):
        patch.set_facecolor(COLORS[method])
        patch.set_alpha(0.90)
        patch.set_edgecolor(APCE_FRAME if method == "apce" else "#303030")
        patch.set_linewidth(1.15 if method == "apce" else 0.70)
    for key in ("whiskers", "caps", "medians"):
        for artist in bp[key]:
            artist.set_color("#303030")
            artist.set_linewidth(0.70)
    rng = np.random.default_rng(101 + CASES.index(case) * 19 + (0 if metric == "nrmse" else 7))
    for xpos, method, values in zip(positions, METHODS, data, strict=True):
        ax.scatter(xpos + rng.normal(0.0, 0.045, values.size), values, s=3.8, color=COLORS[method], alpha=0.23, edgecolors="none", zorder=1)
    y_max = max(float(np.nanmax(v)) for v in data if v.size) * 1.20
    ax.set_ylim(0, y_max)
    apce_highlight(ax, positions[-1], 0, y_max * 0.98)
    ax.set_xticks(positions)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=55, ha="right", rotation_mode="anchor", fontsize=6.6)
    ax.set_ylabel(ylabel, fontsize=7.2)
    ax.set_facecolor("#FAFAFA")
    polish_axis(ax, tick_size=6.6, label_size=7.2)


def bar_panel(
    ax: plt.Axes,
    summary: dict[tuple[str, str], dict[str, str]],
    rows: list[dict[str, str]],
    case: str,
    metric: str,
    ylabel: str,
    scale: float,
    letter: str,
    title: str,
) -> None:
    if letter:
        add_panel(ax, letter, title)
    else:
        add_small_title(ax, title)
    x = np.arange(1, len(METHODS) + 1)
    means, lows, highs = [], [], []
    for method in METHODS:
        mean, low, high = summary_mean_ci(summary, case, method, metric, scale)
        means.append(mean)
        lows.append(low)
        highs.append(high)
    means = np.asarray(means)
    lows = np.asarray(lows)
    highs = np.asarray(highs)
    bars = ax.bar(
        x,
        means,
        width=0.62,
        color=[COLORS[m] for m in METHODS],
        edgecolor=[APCE_FRAME if m == "apce" else "#FFFFFF" for m in METHODS],
        linewidth=1.15,
        alpha=0.92,
        zorder=3,
    )
    for bar, method in zip(bars, METHODS, strict=True):
        if method == "apce":
            bar.set_linewidth(1.45)
    ax.errorbar(x, means, yerr=np.vstack([lows, highs]), fmt="none", ecolor="#222222", elinewidth=0.75, capsize=2.2, capthick=0.75, zorder=5)
    rng = np.random.default_rng(220 + CASES.index(case) * 23 + (0 if metric == "nrmse" else 11))
    for xpos, method in zip(x, METHODS, strict=True):
        values = scaled_seed_values(rows, case, method, metric, scale)
        if values.size == 0:
            continue
        jitter = rng.normal(0.0, 0.055, values.size)
        ax.scatter(
            xpos + jitter,
            values,
            s=4.2,
            color="#FFFFFF" if method == "apce" else "#202020",
            alpha=0.50 if method == "apce" else 0.32,
            edgecolors=COLORS[method],
            linewidths=0.35 if method == "apce" else 0.20,
            zorder=6,
        )
    y_max = float(np.max(means + highs)) * 1.18
    ax.set_ylim(0, y_max)
    apce_highlight(ax, x[-1], 0, y_max * 0.98)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=55, ha="right", rotation_mode="anchor", fontsize=6.6)
    ax.set_ylabel(ylabel, fontsize=7.2)
    ax.set_facecolor("#FAFAFA")
    polish_axis(ax, tick_size=6.6, label_size=7.2)


def write_contract(output_dir: Path) -> None:
    text = """Core conclusion:
Across wave, spring and heat systems under one frozen paired protocol, APCE/PCE improve deterministic and probabilistic reconstruction metrics relative to valid training-free baselines.

Figure archetype:
Strict three-row 5/6/6 evidence wall.

Panel map:
a-e: representative dynamics, calibration and cognitive-weight evidence.
f: six seed-wise boxplots for nRMSE and CRPS across Wave, Spring and Heat.
g: six mean-and-95% CI bar charts for the same metrics and systems.

Statistics:
n=50 paired seeds per system and method. Boxplots show seed-level distributions. Bars show mean with 95% CI and overlaid seed-level dots. APCE is highlighted by a red frame in statistical panels.
"""
    (output_dir / "figure2_classical_uncertain_systems_v9_566_grouped_labels_contract.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--runs-csv", type=Path, required=True)
    parser.add_argument("--paired-csv", type=Path, required=True)
    parser.add_argument("--wave-npz", type=Path, required=True)
    parser.add_argument("--spring-npz", type=Path, required=True)
    parser.add_argument("--heat-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    set_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = read_csv(args.summary_csv)
    run_rows = read_csv(args.runs_csv)
    summary = {(r["case"], r["method"]): r for r in summary_rows}
    wave = np.load(args.wave_npz, allow_pickle=True)
    spring = np.load(args.spring_npz, allow_pickle=True)
    heat = np.load(args.heat_npz, allow_pickle=True)

    fig = plt.figure(figsize=(15.0, 7.25))
    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=[1.03, 0.92, 0.92],
        hspace=0.44,
        left=0.045,
        right=0.992,
        top=0.905,
        bottom=0.085,
    )
    add_top_legend(fig)

    top = GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[0, 0], wspace=0.34, width_ratios=[1.05, 1.05, 1.05, 0.88, 1.02])
    panel_wave(fig.add_subplot(top[0, 0]), wave)
    panel_spring(fig.add_subplot(top[0, 1]), spring)
    panel_heat(fig.add_subplot(top[0, 2]), heat)
    panel_calibration(fig.add_subplot(top[0, 3]), summary)
    panel_weights(fig.add_subplot(top[0, 4]), wave, spring, heat)

    row2 = GridSpecFromSubplotSpec(1, 6, subplot_spec=outer[1, 0], wspace=0.35)
    row3 = GridSpecFromSubplotSpec(1, 6, subplot_spec=outer[2, 0], wspace=0.35)
    slot = 0
    for case in CASES:
        for metric, ylabel, scale in METRICS:
            box_panel(
                fig.add_subplot(row2[0, slot]),
                run_rows,
                case,
                metric,
                ylabel,
                scale,
                "f" if slot == 0 else "",
                f"{CASE_LABELS[case]} {ylabel.split()[0]}",
            )
            bar_panel(
                fig.add_subplot(row3[0, slot]),
                summary,
                run_rows,
                case,
                metric,
                ylabel,
                scale,
                "g" if slot == 0 else "",
                f"{CASE_LABELS[case]} {ylabel.split()[0]}",
            )
            slot += 1

    base = args.output_dir / "figure2_classical_uncertain_systems_v9_566_grouped_labels"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    for source in (args.summary_csv, args.runs_csv, args.paired_csv):
        shutil.copy2(source, args.output_dir / f"source_data_{source.name}")
    write_contract(args.output_dir)
    qa = {
        "figure": "figure2_classical_uncertain_systems_v9_566_grouped_labels",
        "layout": "strict three visual rows: first row a-e; second row one grouped panel f with six small multiples; third row one grouped panel g with six small multiples",
        "backend": "python/matplotlib",
        "font": "Arial with sans-serif fallback",
        "font_weight": "regular",
        "formats": ["svg", "pdf", "png", "tiff"],
        "statistics": "n=50 paired seeds; boxplots show seed distributions; bars show mean, 95% CI and overlaid seed dots; APCE red frame in box and bar panels",
        "panels": ["a", "b", "c", "d", "e", "f", "g"],
    }
    (args.output_dir / "figure2_classical_uncertain_systems_v9_566_grouped_labels_qa.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(base)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
