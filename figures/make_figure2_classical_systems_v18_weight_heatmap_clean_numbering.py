from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
import numpy as np

import make_figure2_classical_systems_v12_fghi_xticklabels as base


OUTPUT_STEM = "figure2_classical_uncertain_systems_v18_weight_heatmap_clean_numbering"

METRIC_GROUPS = (
    ("nrmse", r"nRMSE (%)", 100.0, "e", "nRMSE"),
    ("crps", r"CRPS ($10^{-3}$)", 1000.0, "f", "CRPS"),
    ("coverage_90", "90% coverage", 1.0, "g", "90% coverage"),
    ("interval_width_90", r"Interval width ($10^{-2}$)", 100.0, "h", "Interval width"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _weights_on_common_alpha(
    alpha: np.ndarray,
    weights: np.ndarray,
    common_alpha: np.ndarray,
) -> np.ndarray:
    out = np.full(common_alpha.shape, np.nan, dtype=float)
    for i, a in enumerate(common_alpha):
        idx = int(np.argmin(np.abs(alpha - a)))
        if abs(float(alpha[idx]) - float(a)) < 1.0e-8:
            out[i] = float(weights[idx])
    return out


def _alpha_to_heatmap_x(alpha_value: float, common_alpha: np.ndarray) -> float:
    centers = np.arange(common_alpha.size, dtype=float) + 0.5
    if alpha_value <= common_alpha[0]:
        return float(centers[0])
    if alpha_value >= common_alpha[-1]:
        return float(centers[-1])
    return float(np.interp(alpha_value, common_alpha, centers))


def panel_weight_heatmap(
    ax: plt.Axes,
    wave: np.lib.npyio.NpzFile,
    spring: np.lib.npyio.NpzFile,
    heat: np.lib.npyio.NpzFile,
) -> None:
    base.add_panel(ax, "d", "Cognitive-weight map")

    wave_alpha = np.linspace(0.08, 0.92, len(wave["pce_final_weights"]))
    alpha = np.asarray(spring["alpha_grid"], dtype=float)
    common_alpha = alpha[alpha <= 0.5 + 1.0e-12]
    if common_alpha.size == 0:
        common_alpha = wave_alpha[wave_alpha <= 0.5 + 1.0e-12]

    series = [
        ("Wave", "PCE", wave_alpha, wave["pce_final_weights"]),
        ("Wave", "APCE", wave_alpha, wave["apce_final_weights"]),
        ("Spring", "PCE", spring["alpha_grid"], spring["pce_alpha_weight_history"][-1]),
        ("Spring", "APCE", spring["alpha_grid"], spring["apce_alpha_weight_history"][-1]),
        ("Heat", "PCE", heat["alpha_grid"], heat["pce_alpha_weight_history"][-1]),
        ("Heat", "APCE", heat["alpha_grid"], heat["apce_alpha_weight_history"][-1]),
    ]
    matrix = np.vstack(
        [
            _weights_on_common_alpha(np.asarray(a, dtype=float), np.asarray(w, dtype=float), common_alpha)
            for _, _, a, w in series
        ]
    )

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "apce_weight_map",
        ["#F4E6D2", "#EEC48E", "#DE8840", "#A84624", "#641A0D"],
    )
    cmap.set_bad("#F2F2F2")
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    x_edges = np.arange(common_alpha.size + 1)
    y_edges = np.arange(matrix.shape[0] + 1)
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        matrix,
        cmap=cmap,
        norm=norm,
        edgecolors="#FFFFFF",
        linewidth=0.9,
        antialiased=True,
    )

    ax.set_xlim(0, common_alpha.size)
    ax.set_ylim(matrix.shape[0], 0)
    ax.set_xticks(np.arange(common_alpha.size) + 0.5)
    ax.set_xticklabels([base.compact_tick(v) for v in common_alpha])
    ax.figure2_alpha_tick_labels = [base.compact_tick(v) for v in common_alpha]
    ax.set_yticks(np.arange(matrix.shape[0]) + 0.5)
    ax.set_yticklabels([f"{case} {method}" for case, method, _, _ in series])
    ax.set_xlabel(r"candidate $\alpha$")
    ax.set_ylabel("")
    ax.tick_params(axis="both", which="both", length=0, pad=2.0)

    for label, (_, method, _, _) in zip(ax.get_yticklabels(), series, strict=True):
        if method == "APCE":
            label.set_color("#C94B2C")

    for y in (2, 4):
        ax.axhline(y, color="#D6D6D6", lw=0.75, clip_on=False)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)

    cbar = ax.figure.colorbar(
        mesh,
        ax=ax,
        fraction=0.055,
        pad=0.025,
        ticks=[0, 1],
    )
    cbar.outline.set_visible(True)
    cbar.outline.set_edgecolor("#A8A8A8")
    cbar.outline.set_linewidth(0.55)
    cbar.ax.tick_params(labelsize=base.AXIS_TICK_FONT_SIZE, length=0, pad=1.5)
    cbar.ax.set_yticklabels(["0", "1"])
    cbar.set_label("weight", fontsize=base.AXIS_LABEL_FONT_SIZE, labelpad=1)

    base.polish_axis(ax)


def write_contract(output_dir: Path) -> None:
    text = """Core conclusion:
Across wave, spring and heat systems under one frozen paired protocol, APCE/PCE improve deterministic and probabilistic reconstruction metrics relative to valid training-free baselines.

Figure archetype:
Strict three-row mixed-modality evidence wall. Version 18 keeps the Version 15 geometry, removes the true-alpha marker, and relabels panels consecutively.

Panel map:
a-c: representative dynamics.
d: final cognitive-weight map over candidate alpha values for PCE and APCE across Wave, Spring and Heat.
e: nRMSE seed-wise boxplots across Wave, Spring and Heat.
f: CRPS seed-wise boxplots across Wave, Spring and Heat.
g: 90% coverage mean bars across Wave, Spring and Heat.
h: Interval width mean bars across Wave, Spring and Heat.

Statistics:
n=50 paired seeds per system and method. Panels e and f show seed-wise boxplots. Panels g and h show mean bars with 95% CI and overlaid seed-level dots. APCE is highlighted by a red frame in statistical panels.
"""
    (output_dir / f"{OUTPUT_STEM}_contract.md").write_text(text, encoding="utf-8")


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

    base.set_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = read_csv(args.summary_csv)
    run_rows = read_csv(args.runs_csv)
    summary = {(r["case"], r["method"]): r for r in summary_rows}
    wave = np.load(args.wave_npz, allow_pickle=True)
    spring = np.load(args.spring_npz, allow_pickle=True)
    heat = np.load(args.heat_npz, allow_pickle=True)

    fig = plt.figure(figsize=(11.55, 8.75))
    outer = fig.add_gridspec(
        6,
        1,
        height_ratios=[1.03, 0.13, 0.11, 0.92, 0.28, 0.92],
        hspace=0.22,
        left=0.050,
        right=0.994,
        top=0.945,
        bottom=0.105,
    )

    top = GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[0, 0], wspace=0.29, width_ratios=[1, 1, 1, 0.92])
    base.panel_wave(fig.add_subplot(top[0, 0]), wave)
    base.panel_spring(fig.add_subplot(top[0, 1]), spring)
    base.panel_heat(fig.add_subplot(top[0, 2]), heat)
    ax_weight = fig.add_subplot(top[0, 3])
    panel_weight_heatmap(ax_weight, wave, spring, heat)

    base.add_mid_stat_legend(fig.add_subplot(outer[1, 0]))
    ax_pre_row2_spacer = fig.add_subplot(outer[2, 0])
    ax_pre_row2_spacer.set_axis_off()

    row2 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[3, 0], wspace=0.14)
    ax_spacer = fig.add_subplot(outer[4, 0])
    ax_spacer.set_axis_off()
    row3 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[5, 0], wspace=0.14)
    for row_index, row_metrics in enumerate((METRIC_GROUPS[:2], METRIC_GROUPS[2:])):
        row_spec = row2 if row_index == 0 else row3
        for metric_index, (metric, ylabel, scale, letter, title) in enumerate(row_metrics):
            metric_spec = GridSpecFromSubplotSpec(1, 3, subplot_spec=row_spec[0, metric_index], wspace=0.18)
            for case_index, case in enumerate(base.CASES):
                ax = fig.add_subplot(metric_spec[0, case_index])
                if metric in {"nrmse", "crps"}:
                    base.metric_case_box_panel(
                        ax,
                        run_rows,
                        case,
                        metric,
                        ylabel,
                        scale,
                        letter if case_index == 0 else "",
                        title,
                        show_ylabel=case_index == 0,
                    )
                else:
                    base.metric_case_panel(
                        ax,
                        summary,
                        run_rows,
                        case,
                        metric,
                        ylabel,
                        scale,
                        letter if case_index == 0 else "",
                        title,
                        show_ylabel=case_index == 0,
                    )

    base.freeze_all_compact_numeric_ticklabels(fig)
    alpha_tick_labels = getattr(ax_weight, "figure2_alpha_tick_labels", [])
    if alpha_tick_labels:
        ax_weight.set_xticks(np.arange(len(alpha_tick_labels)) + 0.5)
        ax_weight.set_xticklabels(alpha_tick_labels, fontsize=base.AXIS_TICK_FONT_SIZE)

    out_base = args.output_dir / OUTPUT_STEM
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    for source in (args.summary_csv, args.runs_csv, args.paired_csv):
        shutil.copy2(source, args.output_dir / f"source_data_{source.name}")
    write_contract(args.output_dir)
    qa = {
        "figure": OUTPUT_STEM,
        "layout": "version 18: same global geometry as v15; first row a-c plus d; d keeps the v15 heatmap and right-side vertical colorbar, but removes true-alpha marker and the 0.5 colorbar tick; second row e/f; third row g/h",
        "backend": "python/matplotlib",
        "font": "Arial with sans-serif fallback",
        "font_weight": "regular except panel letters",
        "formats": ["svg", "pdf", "png", "tiff"],
        "changed_from_v15": "removes true-alpha marker, removes the 0.5 tick on the d-panel colorbar, moves the weight label closer to the colorbar, and relabels panels consecutively",
    }
    (args.output_dir / f"{OUTPUT_STEM}_qa.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(out_base)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
