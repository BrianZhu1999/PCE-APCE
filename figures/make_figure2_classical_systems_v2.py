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
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np


CASES = ("wave", "spring", "heat")
CASE_LABELS = {"wave": "Wave", "spring": "Spring", "heat": "Heat"}
METHODS = ("denkf", "letkf", "iensf", "pce", "apce", "oracle_alpha")
REP_METHODS = ("denkf", "pce", "apce")
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
    "denkf": "#747474",
    "letkf": "#B3B3B3",
    "iensf": "#7E84B8",
    "pce": "#3F6EA8",
    "apce": "#D99035",
    "oracle_alpha": "#4F8A58",
}
CASE_COLORS = {"wave": "#3F6EA8", "spring": "#8A6BAE", "heat": "#4F8A58"}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6.4,
            "font.weight": "regular",
            "axes.titleweight": "regular",
            "axes.labelweight": "regular",
            "mathtext.default": "it",
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.65,
            "legend.frameon": False,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.3,
            "ytick.major.size": 2.3,
            "lines.solid_capstyle": "round",
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def add_panel(ax: plt.Axes, letter: str, title: str | None = None) -> None:
    ax.text(-0.09, 1.04, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.0)
    if title:
        ax.text(0.0, 1.04, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=6.8)


def _plot_phase(
    ax: plt.Axes,
    truth: np.ndarray,
    estimates: dict[str, np.ndarray],
    letter: str,
    title: str,
    displacement_index: int,
    velocity_index: int,
) -> None:
    add_panel(ax, letter, title)
    ax.plot(
        truth[:, displacement_index],
        truth[:, velocity_index],
        color=COLORS["truth"],
        lw=1.1,
        zorder=3,
    )
    for method, states in estimates.items():
        ax.plot(
            states[:, displacement_index],
            states[:, velocity_index],
            color=COLORS[method],
            lw=0.85,
            zorder=2,
        )
    ax.set_xlabel(r"$u(t)$")
    ax.set_ylabel(r"$v(t)$")
    ax.grid(color="#E9E9E9", lw=0.38, zorder=0)
    ax.tick_params(labelsize=5.6)
    ax.margins(0.05)


def panel_wave_phase(ax: plt.Axes, data: np.lib.npyio.NpzFile) -> None:
    states = data["truth_states"]
    nx = states.shape[1] // 2
    node = nx // 2
    estimates = {
        "denkf": data["denkf_mean_states"],
        "pce": data["pce_mean_states"],
        "apce": data["apce_mean_states"],
    }
    _plot_phase(ax, states, estimates, "a", r"Wave phase trajectory at $x_c$", node, nx + node)


def panel_spring(ax: plt.Axes, data: np.lib.npyio.NpzFile) -> None:
    truth = data["truth_states"]
    estimates = {
        "denkf": data["denkf_mean_states"],
        "pce": data["pce_mean_states"],
        "apce": data["apce_mean_states"],
    }
    _plot_phase(ax, truth, estimates, "b", "Spring phase trajectory", 0, 1)


def panel_heat(ax: plt.Axes, data: np.lib.npyio.NpzFile) -> None:
    add_panel(ax, "c", "Heat terminal profile")
    x = data["space"]
    truth = data["truth_states"][-1]
    ax.plot(x, truth, color=COLORS["truth"], lw=1.1, zorder=3)
    for method, key in (
        ("denkf", "denkf_mean_states"),
        ("pce", "pce_mean_states"),
        ("apce", "apce_mean_states"),
    ):
        ax.plot(x, data[key][-1], color=COLORS[method], lw=0.85, zorder=2)
    obs_x = x[data["observation_indices"]]
    ax.scatter(
        obs_x,
        truth[data["observation_indices"]],
        s=6.5,
        color="#202020",
        facecolor="white",
        linewidth=0.45,
        zorder=5,
    )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u(x,t_f)$")
    ax.grid(color="#E9E9E9", lw=0.38, zorder=0)
    ax.tick_params(labelsize=5.6)
    ax.margins(0.04)


def panel_wave_displacement_supplement(
    ax_container: plt.Axes,
    data: np.lib.npyio.NpzFile,
) -> None:
    """Detailed Wave displacement/error plate for Supplementary Information."""
    ax_container.set_axis_off()
    sub = GridSpecFromSubplotSpec(
        2,
        4,
        subplot_spec=ax_container.get_subplotspec(),
        hspace=0.24,
        wspace=0.12,
        width_ratios=[1, 1, 1, 1],
    )
    states = data["truth_states"]
    nx = states.shape[1] // 2
    times = data["times"]
    fields = [
        ("Truth", data["truth_states"][:, :nx]),
        ("DEnKF", data["denkf_mean_states"][:, :nx]),
        ("PCE", data["pce_mean_states"][:, :nx]),
        ("APCE", data["apce_mean_states"][:, :nx]),
    ]
    vmax = max(float(np.nanmax(np.abs(field))) for _, field in fields)
    errors = [
        ("DEnKF error", fields[1][1] - fields[0][1]),
        ("PCE error", fields[2][1] - fields[0][1]),
        ("APCE error", fields[3][1] - fields[0][1]),
        ("PCE - APCE", fields[2][1] - fields[3][1]),
    ]
    emax = max(float(np.nanmax(np.abs(error))) for _, error in errors[:3])
    diff_emax = max(float(np.nanmax(np.abs(errors[3][1]))), 1.0e-12)
    top_axes: list[plt.Axes] = []
    bottom_axes: list[plt.Axes] = []
    for col, (label, field) in enumerate(fields):
        ax = ax_container.figure.add_subplot(sub[0, col])
        top_axes.append(ax)
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
        ax.set_title(label, fontsize=6.0, pad=1.5, fontweight="regular")
        ax.set_xticklabels([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(r"$x$", fontsize=5.7)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cax_top = ax_container.figure.add_axes([0.900, 0.56, 0.010, 0.30])
    cbar_top = ax_container.figure.colorbar(im, cax=cax_top)
    cbar_top.set_label(r"$u(x,t)$", fontsize=5.7, labelpad=2)
    cbar_top.ax.tick_params(labelsize=5.1, width=0.4, length=2)

    for col, (label, error) in enumerate(errors):
        ax = ax_container.figure.add_subplot(sub[1, col])
        bottom_axes.append(ax)
        local_emax = diff_emax if col == 3 else emax
        local_cmap = "PuOr_r" if col < 3 else "coolwarm"
        ax.imshow(
            error.T,
            origin="lower",
            aspect="auto",
            extent=[float(times[0]), float(times[-1]), 0, 1],
            cmap=local_cmap,
            vmin=-local_emax,
            vmax=local_emax,
            interpolation="bilinear",
        )
        ax.set_title(label, fontsize=6.0, pad=1.5, fontweight="regular")
        ax.set_xlabel(r"$t$", fontsize=5.7)
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(r"$x$", fontsize=5.7)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cax_bottom = ax_container.figure.add_axes([0.900, 0.16, 0.010, 0.30])
    cbar_bottom = ax_container.figure.colorbar(bottom_axes[2].images[0], cax=cax_bottom)
    cbar_bottom.set_label("error", fontsize=5.7, labelpad=2)
    cbar_bottom.ax.tick_params(labelsize=5.1, width=0.4, length=2)
    cax_diff = ax_container.figure.add_axes([0.955, 0.16, 0.010, 0.30])
    cbar_diff = ax_container.figure.colorbar(bottom_axes[3].images[0], cax=cax_diff)
    cbar_diff.set_label("PCE - APCE", fontsize=5.3, labelpad=2)
    cbar_diff.ax.tick_params(labelsize=4.8, width=0.4, length=2)


def metric_heatmap(
    ax: plt.Axes,
    summary: dict[tuple[str, str], dict[str, str]],
    metric: str,
    letter: str,
    title: str,
    cbar_label: str,
) -> None:
    add_panel(ax, letter, title)
    values = np.array(
        [[float(summary[(case, method)][f"{metric}_mean"]) for method in METHODS] for case in CASES],
        dtype=float,
    )
    vmin = max(float(values.min()) * 0.82, 1.0e-8)
    vmax = float(values.max()) * 1.18
    image = ax.imshow(
        values,
        aspect="auto",
        cmap="Blues",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(len(METHODS)))
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(len(CASES)))
    ax.set_yticklabels([CASE_LABELS[c] for c in CASES])
    ax.tick_params(labelsize=5.1, length=1.8)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            scaled = (np.log(values[row, col]) - np.log(vmin)) / (np.log(vmax) - np.log(vmin))
            color = "white" if scaled > 0.58 else "#202020"
            ax.text(col, row, f"{values[row, col]:.1e}", ha="center", va="center", fontsize=4.25, color=color)
    for col, method in ((METHODS.index("pce"), "pce"), (METHODS.index("apce"), "apce")):
        for row in range(values.shape[0]):
            ax.add_patch(
                Rectangle(
                    (col - 0.47, row - 0.47),
                    0.94,
                    0.94,
                    fill=False,
                    edgecolor=COLORS[method],
                    linewidth=0.9,
                )
            )
    cbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.025, aspect=13)
    cbar.set_label(cbar_label, fontsize=5.3, labelpad=2)
    cbar.ax.tick_params(labelsize=4.8, width=0.4, length=1.8)
    cbar.ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10, labelOnlyBase=False))


def calibration_sharpness(ax: plt.Axes, summary: dict[tuple[str, str], dict[str, str]]) -> None:
    add_panel(ax, "f", "Calibration-sharpness map")
    markers = {"wave": "o", "spring": "s", "heat": "^"}
    for case in CASES:
        for method in METHODS:
            row = summary[(case, method)]
            coverage = float(row["coverage_90_mean"])
            width = float(row["interval_width_90_mean"])
            ax.scatter(
                coverage,
                width,
                s=25 if method in {"pce", "apce"} else 17,
                marker=markers[case],
                color=COLORS[method],
                edgecolor="#202020" if method in {"pce", "apce"} else "none",
                linewidth=0.3,
                zorder=4,
            )
            if method == "apce":
                ax.text(coverage + 0.008, width * 1.07, CASE_LABELS[case], fontsize=5.2, color=COLORS[method])
    ax.axvline(0.9, color="#666666", lw=0.6, ls=(0, (3, 2)))
    ax.set_yscale("log")
    ax.set_xlim(0.35, 1.02)
    ax.set_xlabel("90% coverage")
    ax.set_ylabel("interval width")
    ax.grid(color="#E8E8E8", lw=0.4)
    case_handles = [
        Line2D([0], [0], marker=markers[c], color="none", markerfacecolor="#777777", markersize=4.1, label=CASE_LABELS[c])
        for c in CASES
    ]
    leg1 = ax.legend(handles=case_handles, loc="upper left", fontsize=4.7, ncol=3, handletextpad=0.2, columnspacing=0.45, borderpad=0.12)
    ax.add_artist(leg1)
    ax.tick_params(labelsize=5.2)


def forest(
    ax: plt.Axes,
    paired: list[dict[str, str]],
    metric: str,
    letter: str,
    title: str,
    xlabel: str,
) -> None:
    add_panel(ax, letter, title)
    entries = [row for row in paired if row["metric"] == metric and row["baseline"] in {"denkf", "letkf", "iensf"}]
    order = [(case, baseline) for case in CASES for baseline in ("denkf", "letkf", "iensf")]
    entries_by_key = {(row["case"], row["baseline"]): row for row in entries}
    y = np.arange(len(order), dtype=float)[::-1]
    for yi, (case, baseline) in zip(y, order):
        row = entries_by_key[(case, baseline)]
        est = float(row["mean_difference_apce_minus_baseline"])
        lo = float(row["ci95_low"])
        hi = float(row["ci95_high"])
        ax.plot([lo, hi], [yi, yi], color=CASE_COLORS[case], lw=1.1)
        ax.scatter(est, yi, s=17, color=CASE_COLORS[case], edgecolor="#202020", linewidth=0.25, zorder=3)
    ax.axvline(0.0, color="#606060", lw=0.6, ls=(0, (3, 2)))
    ax.set_yticks(y)
    ax.set_yticklabels([f"{CASE_LABELS[case]} / {METHOD_LABELS[baseline]}" for case, baseline in order], fontsize=4.9)
    ax.set_xlabel(xlabel, fontsize=5.6)
    ax.grid(axis="x", color="#E8E8E8", lw=0.4)
    ax.tick_params(labelsize=5.1)


def alpha_summary(ax: plt.Axes, wave: np.lib.npyio.NpzFile, spring: np.lib.npyio.NpzFile, heat: np.lib.npyio.NpzFile) -> None:
    add_panel(ax, "i", "Cognitive-orbit weights")
    series = [
        ("wave", wave["pce_final_weights"], wave["apce_final_weights"], np.linspace(0.08, 0.92, len(wave["pce_final_weights"]))),
        ("spring", spring["pce_alpha_weight_history"][-1], spring["apce_alpha_weight_history"][-1], spring["alpha_grid"]),
        ("heat", heat["pce_alpha_weight_history"][-1], heat["apce_alpha_weight_history"][-1], heat["alpha_grid"]),
    ]
    for case, pce, apce, alpha in series:
        ax.plot(alpha, pce, color=CASE_COLORS[case], lw=0.9, ls="--", marker="o", ms=2.2, label=f"{CASE_LABELS[case]} PCE")
        ax.plot(alpha, apce, color=CASE_COLORS[case], lw=1.05, ls="-", marker="o", ms=2.2, label=f"{CASE_LABELS[case]} APCE")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("final weight")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#E8E8E8", lw=0.4)
    ax.tick_params(labelsize=5.2)
    case_handles = [
        Line2D([0], [0], color=CASE_COLORS[c], lw=1.0, label=CASE_LABELS[c]) for c in CASES
    ]
    style_handles = [
        Line2D([0], [0], color="#555555", lw=1.0, ls="--", label="PCE"),
        Line2D([0], [0], color="#555555", lw=1.0, ls="-", label="APCE"),
    ]
    leg1 = ax.legend(handles=case_handles, loc="upper right", fontsize=4.7, ncol=3, handlelength=1.0, borderpad=0.1, columnspacing=0.5)
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, loc="center right", fontsize=4.7, handlelength=1.0, borderpad=0.1)


def write_contract(output_dir: Path) -> None:
    text = """Core conclusion:
Across wave, spring and heat systems under one frozen paired protocol, APCE improves state error and probabilistic skill over all numerically valid training-free baselines while retaining identifiable cognitive-orbit evidence.

Figure archetype:
Compact 3x3 quantitative-grid with representative trajectory panels.

Panel map:
a: Wave central-node phase trajectory.
b: Spring phase-space trajectory.
c: Heat terminal spatial profile.
d: nRMSE heatmap across 50 paired seeds.
e: CRPS heatmap across 50 paired seeds.
f: calibration-sharpness map.
g: paired APCE nRMSE differences with bootstrap confidence intervals.
h: paired APCE CRPS differences with bootstrap confidence intervals.
i: PCE/APCE cognitive-orbit weights across cases.

Statistics:
n=50 paired seeds per system and method; heatmap cells are means; forest intervals are paired bootstrap 95% confidence intervals; Holm-adjusted comparisons remain in the source data.

Supplementary:
The detailed Wave displacement-field and error plate is exported separately as a Supplementary Figure.
"""
    (output_dir / "figure2_classical_uncertain_systems_v3_contract.md").write_text(text, encoding="utf-8")


def export_wave_supplement(data: np.lib.npyio.NpzFile, output_dir: Path) -> None:
    fig = plt.figure(figsize=(7.6, 4.25))
    outer = fig.add_gridspec(1, 1, left=0.055, right=0.88, top=0.94, bottom=0.12)
    host = fig.add_subplot(outer[0, 0])
    host.set_axis_off()
    panel_wave_displacement_supplement(host, data)
    base = output_dir / "supp_figure_wave_displacement"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


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
    paired_rows = read_csv(args.paired_csv)
    summary = {(r["case"], r["method"]): r for r in summary_rows}
    wave = np.load(args.wave_npz, allow_pickle=True)
    spring = np.load(args.spring_npz, allow_pickle=True)
    heat = np.load(args.heat_npz, allow_pickle=True)

    fig = plt.figure(figsize=(7.2, 7.1))
    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[1.15, 1.0, 1.15],
        hspace=0.54,
        wspace=0.42,
        left=0.075,
        right=0.985,
        top=0.965,
        bottom=0.105,
    )
    panel_wave_phase(fig.add_subplot(gs[0, 0]), wave)
    panel_spring(fig.add_subplot(gs[0, 1]), spring)
    panel_heat(fig.add_subplot(gs[0, 2]), heat)
    metric_heatmap(fig.add_subplot(gs[1, 0]), summary, "nrmse", "d", "State error", "nRMSE")
    metric_heatmap(fig.add_subplot(gs[1, 1]), summary, "crps", "e", "Probabilistic score", "CRPS")
    calibration_sharpness(fig.add_subplot(gs[1, 2]), summary)
    forest(fig.add_subplot(gs[2, 0]), paired_rows, "nrmse", "g", "Paired nRMSE difference", r"$\Delta$ nRMSE (APCE - baseline)")
    forest(fig.add_subplot(gs[2, 1]), paired_rows, "crps", "h", "Paired CRPS difference", r"$\Delta$ CRPS (APCE - baseline)")
    alpha_summary(fig.add_subplot(gs[2, 2]), wave, spring, heat)

    handles = [Line2D([0], [0], color=COLORS["truth"], lw=1.5, label="Truth")]
    handles.extend([Line2D([0], [0], color=COLORS[m], lw=1.7, label=METHOD_LABELS[m]) for m in METHODS])
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.018),
        ncol=7,
        fontsize=5.3,
        handlelength=1.25,
        columnspacing=0.55,
    )
    base = args.output_dir / "figure2_classical_uncertain_systems_v3"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    export_wave_supplement(wave, args.output_dir)
    for source in (args.summary_csv, args.runs_csv, args.paired_csv):
        shutil.copy2(source, args.output_dir / f"source_data_{source.name}")
    write_contract(args.output_dir)
    qa = {
        "figure": "figure2_classical_uncertain_systems_v3",
        "backend": "python/matplotlib",
        "font": "Arial with sans-serif fallback",
        "font_weight": "regular",
        "svg_text_editable": True,
        "statistics": "n=50 paired seeds; heatmap means; paired bootstrap 95% CI forest plots; Holm-adjusted comparisons in source data",
        "formats": ["svg", "pdf", "png", "tiff"],
        "panels": ["a", "b", "c", "d", "e", "f", "g", "h", "i"],
        "supplementary_outputs": ["supp_figure_wave_displacement.svg", "supp_figure_wave_displacement.pdf", "supp_figure_wave_displacement.png", "supp_figure_wave_displacement.tiff"],
    }
    (args.output_dir / "figure2_classical_uncertain_systems_v3_qa.json").write_text(
        json.dumps(qa, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(base), "supplement": str(args.output_dir / "supp_figure_wave_displacement")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
