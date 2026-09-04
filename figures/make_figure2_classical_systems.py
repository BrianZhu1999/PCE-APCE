from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle
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
    "denkf": "#7A7A7A",
    "letkf": "#A6A6A6",
    "iensf": "#7C83B6",
    "pce": "#3F6EA8",
    "apce": "#D99035",
    "oracle_alpha": "#4F8A58",
}


def set_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 6.7
    plt.rcParams["font.weight"] = "regular"
    plt.rcParams["axes.titleweight"] = "regular"
    plt.rcParams["axes.labelweight"] = "regular"
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 0.65
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["xtick.major.width"] = 0.55
    plt.rcParams["ytick.major.width"] = 0.55
    plt.rcParams["xtick.major.size"] = 2.4
    plt.rcParams["ytick.major.size"] = 2.4


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def grouped(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["case"], row["method"]): row for row in rows}


def add_panel(ax: plt.Axes, letter: str, title: str | None = None) -> None:
    ax.text(-0.075, 1.035, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.2)
    if title:
        ax.set_title(title, fontsize=7.2, pad=2.0)


def panel_a(ax: plt.Axes) -> None:
    ax.set_axis_off()
    add_panel(ax, "a", None)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    specs = [
        ("Wave", "hyperbolic PDE", "state 82; obs. 6/41", r"$\Delta t=2.5\times10^{-3}$; 20 cycles"),
        ("Spring", "second-order ODE", "state 2; obs. 1/2", r"$\Delta t=1.0\times10^{-2}$; 52 cycles"),
        ("Heat", "parabolic PDE", "state 64; obs. 8/64", r"$\Delta t=7.5\times10^{-4}$; 26 cycles"),
    ]
    for i, (name, family, dim, schedule) in enumerate(specs):
        x0 = 0.02 + i * 0.318
        ax.add_patch(Rectangle((x0, 0.18), 0.29, 0.62, facecolor="#F8F8F8", edgecolor="#D0D0D0", linewidth=0.6))
        ax.text(x0 + 0.018, 0.68, name, fontsize=7.4, ha="left", va="center")
        ax.text(x0 + 0.018, 0.50, family, fontsize=6.6, color="#555555", ha="left", va="center")
        ax.text(x0 + 0.018, 0.35, dim, fontsize=6.3, color="#555555", ha="left", va="center")
        ax.text(x0 + 0.018, 0.22, schedule, fontsize=6.1, color="#555555", ha="left", va="center")
    ax.text(0.02, 0.91, "Frozen paired protocol: 50 seeds, shared truth, observations, initial ensembles and forecast noise", fontsize=6.8, ha="left")
    x = 0.02
    y = 0.045
    for method in METHODS:
        ax.plot([x, x + 0.018], [y, y], lw=2.2, color=COLORS[method], solid_capstyle="round")
        ax.text(x + 0.023, y, METHOD_LABELS[method], fontsize=6.2, va="center", ha="left")
        x += 0.145 if method != "oracle_alpha" else 0.0


def panel_wave(ax_container: plt.Axes, data: np.lib.npyio.NpzFile) -> None:
    ax_container.set_axis_off()
    add_panel(ax_container, "b", "Wave displacement fields")
    sub = GridSpecFromSubplotSpec(4, 2, subplot_spec=ax_container.get_subplotspec(), width_ratios=[24, 1], hspace=0.08, wspace=0.04)
    times = data["times"]
    nx = data["observation_indices"].max() + 5
    fields = [
        ("Truth", data["truth_states"][:, :nx]),
        ("DEnKF", data["denkf_mean_states"][:, :nx]),
        ("PCE", data["pce_mean_states"][:, :nx]),
        ("APCE", data["apce_mean_states"][:, :nx]),
    ]
    vmax = max(float(np.nanmax(np.abs(field))) for _, field in fields)
    axes = []
    im = None
    for row, (label, field) in enumerate(fields):
        ax = ax_container.figure.add_subplot(sub[row, 0])
        axes.append(ax)
        im = ax.imshow(
            field.T,
            origin="lower",
            aspect="auto",
            extent=[float(times[0]), float(times[-1]), 0, 1],
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="bilinear",
        )
        ax.text(0.012, 0.80, label, transform=ax.transAxes, fontsize=6.2, ha="left", va="center", color="#202020")
        ax.set_yticks([])
        if row < len(fields) - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(r"$t$")
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[-1].set_ylabel(r"$x$", labelpad=6)
    cax = ax_container.figure.add_subplot(sub[:, 1])
    cbar = ax_container.figure.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=5.8, width=0.45, length=2)


def panel_spring(ax: plt.Axes, data: np.lib.npyio.NpzFile) -> None:
    add_panel(ax, "c", "Spring phase trajectory")
    truth = data["truth_states"]
    ax.plot(truth[:, 0], truth[:, 1], color=COLORS["truth"], lw=1.15, label="Truth")
    for method, key in (("denkf", "denkf_mean_states"), ("pce", "pce_mean_states"), ("apce", "apce_mean_states")):
        arr = data[key]
        ax.plot(arr[:, 0], arr[:, 1], color=COLORS[method], lw=0.95, label=METHOD_LABELS[method])
    ax.set_xlabel(r"$x(t)$")
    ax.set_ylabel(r"$v(t)$")
    ax.legend(loc="upper right", fontsize=5.8, handlelength=1.3, borderpad=0.1, labelspacing=0.25)


def panel_heat(ax: plt.Axes, data: np.lib.npyio.NpzFile) -> None:
    add_panel(ax, "d", "Heat terminal profile")
    x = data["space"]
    truth = data["truth_states"][-1]
    ax.plot(x, truth, color=COLORS["truth"], lw=1.25, label="Truth")
    for method, key in (("denkf", "denkf_mean_states"), ("pce", "pce_mean_states"), ("apce", "apce_mean_states")):
        arr = data[key][-1]
        ax.plot(x, arr, color=COLORS[method], lw=1.0, label=METHOD_LABELS[method])
    obs_x = x[data["observation_indices"]]
    obs_y = truth[data["observation_indices"]]
    ax.scatter(obs_x, obs_y, s=8, color="#202020", facecolor="white", linewidth=0.55, zorder=5)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u(x,t_f)$")
    ax.legend(loc="upper right", fontsize=5.8, handlelength=1.3, borderpad=0.1, labelspacing=0.25)


def plot_metric_panel(
    ax: plt.Axes,
    summary: dict[tuple[str, str], dict[str, str]],
    run_rows: list[dict[str, str]],
    metric: str,
    letter: str,
    title: str,
    ylabel: str,
    log_scale: bool,
    target_line: float | None = None,
) -> None:
    add_panel(ax, letter, title)
    x = np.arange(len(CASES), dtype=float)
    width = 0.118
    offsets = np.linspace(-0.31, 0.31, len(METHODS))
    rng = np.random.default_rng(20260807 + sum(ord(c) for c in metric))
    for j, method in enumerate(METHODS):
        means = []
        lower = []
        upper = []
        for case in CASES:
            row = summary[(case, method)]
            mean = float(row[f"{metric}_mean"])
            lo = float(row[f"{metric}_ci95_low"])
            hi = float(row[f"{metric}_ci95_high"])
            means.append(mean)
            lower.append(max(mean - lo, 0.0))
            upper.append(max(hi - mean, 0.0))
        pos = x + offsets[j]
        ax.bar(
            pos,
            means,
            width=width,
            color=COLORS[method],
            edgecolor="#202020",
            linewidth=0.35,
            yerr=np.vstack([lower, upper]),
            error_kw={"elinewidth": 0.65, "capthick": 0.65, "capsize": 2.4},
            zorder=3,
        )
        for i, case in enumerate(CASES):
            values = [float(row[metric]) for row in run_rows if row["case"] == case and row["method"] == method and row.get(metric)]
            if values:
                jitter = rng.normal(0.0, width * 0.18, size=len(values))
                ax.scatter(
                    np.full(len(values), pos[i]) + jitter,
                    values,
                    s=2.6,
                    color="#202020",
                    alpha=0.16,
                    linewidth=0,
                    zorder=4,
                )
    ax.set_xticks(x)
    ax.set_xticklabels([CASE_LABELS[c] for c in CASES])
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
    else:
        ax.set_ylim(0.32, 1.03)
    if target_line is not None:
        ax.axhline(target_line, color="#606060", lw=0.65, ls=(0, (3, 2)), zorder=2)
    ax.grid(axis="y", color="#E6E6E6", lw=0.45, zorder=0)
    ax.tick_params(labelsize=6.1)


def write_contract(output_dir: Path) -> None:
    text = """Core conclusion:
Across three classical uncertain physical equations, APCE consistently reduces state error and CRPS relative to all numerically valid training-free baselines under a frozen 50-seed paired protocol.

Figure archetype:
Asymmetric mixed-modality figure.

Target journal/output:
Nature Computational Science main-text double-column figure; SVG, PDF, PNG and 600 dpi TIFF.

Backend:
Python/matplotlib only.

Final size:
7.2 in x 8.7 in.

Panel map:
a: equation-suite and frozen-pairing protocol.
b: representative wave time-space reconstruction.
c: representative spring phase trajectory.
d: representative heat terminal profile.
e: 50-seed nRMSE.
f: 50-seed CRPS.
g: 50-seed 90% coverage.
h: 50-seed interval width.

Statistics needed:
n = 50 paired seeds per case and method; bars are means; error bars are bootstrap 95% confidence intervals; dots are seed-level runs.

Source data needed:
figure2_method_summary_20260807.csv, figure2_run_source_data_20260807.csv, and representative npz files for Wave, Spring and Heat.

Image-integrity notes:
All panels are generated from numeric source data with global color limits within the wave plate.

Reviewer risk:
Oracle-alpha is a diagnostic upper bound, not a deployable baseline; IEnSF is included because it passed numerical smoke, despite weak accuracy in Wave.
"""
    (output_dir / "figure2_classical_uncertain_systems_contract.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Make formal Figure 2 for classical uncertain systems.")
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--runs-csv", type=Path, required=True)
    parser.add_argument("--wave-npz", type=Path, required=True)
    parser.add_argument("--spring-npz", type=Path, required=True)
    parser.add_argument("--heat-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    set_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = read_csv(args.summary_csv)
    run_rows = read_csv(args.runs_csv)
    summary = grouped(summary_rows)
    wave = np.load(args.wave_npz, allow_pickle=True)
    spring = np.load(args.spring_npz, allow_pickle=True)
    heat = np.load(args.heat_npz, allow_pickle=True)

    fig = plt.figure(figsize=(7.2, 8.7), constrained_layout=False)
    gs = fig.add_gridspec(
        4,
        4,
        height_ratios=[0.82, 1.82, 1.22, 1.22],
        hspace=0.52,
        wspace=0.45,
        left=0.065,
        right=0.985,
        top=0.985,
        bottom=0.045,
    )
    ax_a = fig.add_subplot(gs[0, :])
    panel_a(ax_a)
    ax_b = fig.add_subplot(gs[1, :2])
    panel_wave(ax_b, wave)
    ax_c = fig.add_subplot(gs[1, 2])
    panel_spring(ax_c, spring)
    ax_d = fig.add_subplot(gs[1, 3])
    panel_heat(ax_d, heat)
    ax_e = fig.add_subplot(gs[2, :2])
    ax_f = fig.add_subplot(gs[2, 2:])
    ax_g = fig.add_subplot(gs[3, :2])
    ax_h = fig.add_subplot(gs[3, 2:])
    plot_metric_panel(ax_e, summary, run_rows, "nrmse", "e", "State error", "nRMSE", True)
    plot_metric_panel(ax_f, summary, run_rows, "crps", "f", "Probabilistic score", "CRPS", True)
    plot_metric_panel(ax_g, summary, run_rows, "coverage_90", "g", "Interval calibration", "90% coverage", False, target_line=0.90)
    plot_metric_panel(ax_h, summary, run_rows, "interval_width_90", "h", "Interval concentration", "interval width", True)
    base = args.output_dir / "figure2_classical_uncertain_systems"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    shutil.copy2(args.summary_csv, args.output_dir / "source_data_figure2_method_summary.csv")
    shutil.copy2(args.runs_csv, args.output_dir / "source_data_figure2_seed_runs.csv")
    write_contract(args.output_dir)
    qa = {
        "figure": "figure2_classical_uncertain_systems",
        "backend": "python/matplotlib",
        "font": "Arial with sans-serif fallback",
        "font_weight": "regular",
        "svg_text_editable": True,
        "statistics": "n=50 paired seeds; bars mean; error bars bootstrap 95% CI; dots seed-level runs",
        "formats": ["svg", "pdf", "png", "tiff"],
        "source_data": [
            "source_data_figure2_method_summary.csv",
            "source_data_figure2_seed_runs.csv",
            args.wave_npz.name,
            args.spring_npz.name,
            args.heat_npz.name,
        ],
    }
    (args.output_dir / "figure2_classical_uncertain_systems_qa.json").write_text(
        json.dumps(qa, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(base), "formats": qa["formats"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
