from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np


CASES = ("chemical", "pk_infusion", "sir")
CASE_LABELS = {
    "chemical": "Chemical reaction",
    "pk_infusion": "Multifactor PK",
    "sir": "SIR rumour",
}
CASE_SHORT = {"chemical": "Chemical", "pk_infusion": "PK", "sir": "SIR"}
STATE_LABELS = {"chemical": r"$x$", "pk_infusion": r"$x$", "sir": r"$i$"}
STATE_INDEX = {"chemical": 0, "pk_infusion": 0, "sir": 1}
METHODS = ("denkf", "letkf", "iensf", "pce", "apce")
METHOD_LABELS = {
    "denkf": "DEnKF",
    "letkf": "LETKF",
    "iensf": "IEnSF",
    "pce": "PCE",
    "apce": "APCE",
}
COLORS = {
    "truth": "#202020",
    "denkf": "#667DAA",
    "letkf": "#94A9CC",
    "iensf": "#C2C9D6",
    "pce": "#B38AC6",
    "apce": "#E9B44C",
}
CASE_COLORS = {"chemical": "#5B8DB8", "pk_infusion": "#8E6AAD", "sir": "#65A977"}
APCE_FRAME = "#D64255"
WIDTH_MM = 183.0
HEIGHT_MM = 152.0
MM_PER_INCH = 25.4


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def compact_tick(value: float, _pos: int | None = None) -> str:
    if abs(value) < 1.0e-12:
        return "0"
    if abs(value) < 1.0e-2 or abs(value) >= 1.0e3:
        return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e")
    return f"{value:.2f}".rstrip("0").rstrip(".")


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
            "axes.linewidth": 0.70,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "xtick.major.width": 0.60,
            "ytick.major.width": 0.60,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "lines.solid_capstyle": "round",
        }
    )


def polish(ax: plt.Axes) -> None:
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(False)
    ax.tick_params(labelsize=7.4, pad=1.5, length=2.0)
    ax.xaxis.set_major_formatter(FuncFormatter(compact_tick))
    ax.yaxis.set_major_formatter(FuncFormatter(compact_tick))


def add_panel_label(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(-0.10, 1.08, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=10.0, fontweight="bold")
    ax.text(0.00, 1.08, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.6)


def save_pub(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def case_dt(run_rows: list[dict[str, str]], case: str) -> float:
    for row in run_rows:
        if row["case"] == case:
            return float(row["dt"])
    raise KeyError(case)


def values_by_case_method(rows: list[dict[str, str]], metric: str) -> dict[tuple[str, str], list[float]]:
    output: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        if row.get("numerical_status") != "valid":
            continue
        key = (row["case"], row["method"])
        output.setdefault(key, []).append(float(row[metric]))
    return output


def draw_family_map(ax: plt.Axes) -> None:
    add_panel_label(ax, "a", "Applied ODE families")
    ax.set_axis_off()
    cards = [
        ("Chemical reaction", r"$\dot{x}=-2(\mu+\sigma F_\alpha)x^2$", "nonlinear kinetics"),
        ("Multifactor PK", r"$\dot{x}=k_0-k_1x+(\sigma_1x+\sigma_2)F_\alpha$", "positive concentration"),
        ("SIR rumour", r"$\dot{s}=\beta is-\lambda sr-\eta s-\sigma srF_\alpha$", "coupled compartments"),
    ]
    y_values = (0.695, 0.400, 0.105)
    card_h = 0.225
    for (title, equation, tag), y, case in zip(cards, y_values, CASES):
        box = FancyBboxPatch(
            (0.02, y),
            0.94,
            card_h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            transform=ax.transAxes,
            facecolor="#F7F8FA",
            edgecolor=CASE_COLORS[case],
            linewidth=0.9,
        )
        ax.add_patch(box)
        ax.add_patch(
            Rectangle(
                (0.02, y),
                0.020,
                card_h,
                transform=ax.transAxes,
                facecolor=CASE_COLORS[case],
                edgecolor="none",
                alpha=0.95,
            )
        )
        ax.text(0.065, y + 0.168, title, transform=ax.transAxes, ha="left", va="center", fontsize=7.6, clip_on=True)
        ax.text(0.065, y + 0.102, equation, transform=ax.transAxes, ha="left", va="center", fontsize=6.7, clip_on=True)
        ax.text(0.065, y + 0.040, tag, transform=ax.transAxes, ha="left", va="center", fontsize=7.1, color="#555555", clip_on=True)


def draw_trajectory(ax: plt.Axes, traces: np.lib.npyio.NpzFile, run_rows: list[dict[str, str]], case: str, letter: str) -> None:
    add_panel_label(ax, letter, CASE_LABELS[case])
    state_index = STATE_INDEX[case]
    truth_key = f"{case}_apce_truth_states"
    truth = traces[truth_key]
    time = np.arange(truth.shape[0]) * case_dt(run_rows, case)
    ax.plot(time, truth[:, state_index], color=COLORS["truth"], lw=1.35, zorder=8)
    for method in ("denkf", "letkf", "iensf", "pce", "apce"):
        key = f"{case}_{method}_mean_states"
        if key not in traces:
            continue
        alpha = 0.36 if method in {"denkf", "letkf"} else (0.55 if method == "iensf" else 0.95)
        lw = 0.8 if method in {"denkf", "letkf", "iensf"} else (1.05 if method == "pce" else 1.25)
        ax.plot(time, traces[key][:, state_index], color=COLORS[method], lw=lw, alpha=alpha, zorder=7 if method == "apce" else 5)
    obs_steps_key = f"{case}_apce_observation_steps"
    obs_values_key = f"{case}_apce_observation_values"
    if obs_steps_key in traces and obs_values_key in traces:
        obs_steps = traces[obs_steps_key]
        obs_values = traces[obs_values_key][:, 0]
        ax.scatter(obs_steps * case_dt(run_rows, case), obs_values, s=10, facecolors="white", edgecolors="#333333", linewidths=0.45, zorder=9)
    ax.set_xlabel(r"$t$", fontsize=7.6)
    ax.set_ylabel(STATE_LABELS[case], fontsize=7.6)
    polish(ax)

def grouped_boxplot(ax: plt.Axes, run_rows: list[dict[str, str]], metric: str, scale: float, letter: str, title: str, ylabel: str) -> None:
    add_panel_label(ax, letter, title)
    data = values_by_case_method(run_rows, metric)
    positions: list[float] = []
    samples: list[list[float]] = []
    colors: list[str] = []
    sample_methods: list[str] = []
    centers: list[float] = []
    labels: list[str] = []
    width = 0.12
    offsets = np.linspace(-0.26, 0.26, len(METHODS))
    for case_index, case in enumerate(CASES):
        base = float(case_index)
        centers.append(base)
        labels.append(CASE_SHORT[case])
        for method, offset in zip(METHODS, offsets):
            values = [v * scale for v in data.get((case, method), [])]
            if not values:
                continue
            positions.append(base + float(offset))
            samples.append(values)
            colors.append(COLORS[method])
            sample_methods.append(method)
    box = ax.boxplot(samples, positions=positions, widths=width, patch_artist=True, showfliers=False)
    for patch, color, method in zip(box["boxes"], colors, sample_methods):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
        patch.set_linewidth(1.15 if method == "apce" else 0.55)
        patch.set_edgecolor(APCE_FRAME if method == "apce" else "white")
    for key in ("whiskers", "caps", "medians"):
        for artist in box[key]:
            artist.set_color("#444444")
            artist.set_linewidth(0.55)
    rng = np.random.default_rng(20260808)
    for values, position, color in zip(samples, positions, colors):
        jitter = rng.normal(0.0, width * 0.12, size=len(values))
        ax.scatter(np.full(len(values), position) + jitter, values, s=7, color=color, edgecolors="white", linewidths=0.25, alpha=0.82, zorder=5)
    ax.set_ylabel(ylabel, fontsize=7.6)
    polish(ax)
    ax.set_xticks(centers)
    ax.set_xticklabels(labels, fontsize=7.4)


def draw_calibration(ax: plt.Axes, summary_rows: list[dict[str, str]]) -> None:
    add_panel_label(ax, "g", "Calibration")
    for case in CASES:
        for method in METHODS:
            row = next((r for r in summary_rows if r["case"] == case and r["method"] == method and r["valid"] == "True"), None)
            if row is None:
                continue
            ax.scatter(
                float(row["coverage_90_mean"]),
                float(row["interval_width_90_mean"]),
                s=24 if method == "apce" else 17,
                marker={"chemical": "o", "pk_infusion": "s", "sir": "^"}[case],
                color=COLORS[method],
                edgecolors=APCE_FRAME if method == "apce" else "white",
                linewidths=0.8 if method == "apce" else 0.35,
                alpha=0.9,
            )
    ax.axvline(0.9, color="#888888", lw=0.65, ls="--")
    ax.set_xlabel("90% coverage", fontsize=7.6)
    ax.set_ylabel("interval width", fontsize=7.6)
    polish(ax)


def draw_method_legend(ax: plt.Axes) -> None:
    ax.set_axis_off()
    handles = [Line2D([0], [0], color=COLORS["truth"], lw=1.6, label="Truth")]
    handles.extend(Line2D([0], [0], color=COLORS[m], lw=5.0, label=METHOD_LABELS[m]) for m in METHODS)
    handles.append(Line2D([0], [0], marker="o", color="#333333", markerfacecolor="white", lw=0, markersize=4.0, label="observed"))
    ax.legend(handles=handles, loc="center", ncol=7, fontsize=7.6, handlelength=1.0, columnspacing=0.75)


def main() -> None:
    parser = argparse.ArgumentParser(description="Make Figure 3 applied uncertain ODE figure.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    set_style()
    source = args.root / "source_data"
    run_rows = read_csv(source / "figure3_run_source_data_20260808.csv")
    summary_rows = read_csv(source / "figure3_method_summary_20260808.csv")
    traces = np.load(args.trace_root / "source_data" / "figure3_representative_traces_20260808.npz", allow_pickle=False)

    fig = plt.figure(figsize=(WIDTH_MM / MM_PER_INCH, HEIGHT_MM / MM_PER_INCH), constrained_layout=False)
    outer = fig.add_gridspec(4, 1, height_ratios=[1.06, 0.13, 1.00, 0.92], hspace=0.52)
    top = outer[0, 0].subgridspec(1, 4, width_ratios=[1.38, 1.0, 1.0, 1.0], wspace=0.46)
    middle = outer[2, 0].subgridspec(1, 2, wspace=0.24)
    bottom = outer[3, 0].subgridspec(1, 3, wspace=0.36)
    draw_family_map(fig.add_subplot(top[0, 0]))
    draw_trajectory(fig.add_subplot(top[0, 1]), traces, run_rows, "chemical", "b")
    draw_trajectory(fig.add_subplot(top[0, 2]), traces, run_rows, "pk_infusion", "c")
    draw_trajectory(fig.add_subplot(top[0, 3]), traces, run_rows, "sir", "d")
    draw_method_legend(fig.add_subplot(outer[1, 0]))
    grouped_boxplot(fig.add_subplot(middle[0, 0]), run_rows, "nrmse", 100.0, "e", "State error", "nRMSE (%)")
    grouped_boxplot(fig.add_subplot(middle[0, 1]), run_rows, "crps", 1000.0, "f", "Probabilistic score", r"CRPS ($10^{-3}$)")
    draw_calibration(fig.add_subplot(bottom[0, 0]), summary_rows)
    grouped_boxplot(fig.add_subplot(bottom[0, 1]), run_rows, "physical_validity_error", 1.0, "h", "Physical validity", "constraint error")
    grouped_boxplot(fig.add_subplot(bottom[0, 2]), run_rows, "alpha_absolute_error", 1.0, "i", r"Cognitive-path error", r"$|\hat{\alpha}-\alpha|$")
    save_pub(fig, args.output)


if __name__ == "__main__":
    main()
