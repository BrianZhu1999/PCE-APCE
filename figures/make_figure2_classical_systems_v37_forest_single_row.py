from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpecFromSubplotSpec
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import make_figure2_classical_systems_v35_panel_i_legend_realigned as v35  # noqa: E402
import make_figure2_classical_systems_v36_paired_forest as v36  # noqa: E402


OUTPUT_STEM = "figure2_classical_uncertain_systems_v37_forest_single_row"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def add_forest_label(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.text(
        -0.02,
        0.62,
        "i",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=v35.base.PANEL_LABEL_FONT_SIZE,
        fontweight="bold",
    )
    ax.text(
        0.035,
        0.62,
        "APCE gain over baselines",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=v35.base.PANEL_TITLE_FONT_SIZE,
    )
    ax.text(
        0.035,
        0.16,
        "paired differences; positive values mean APCE is better",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=v35.base.AXIS_TICK_FONT_SIZE,
        color="#3A3A3A",
    )


def compact_xlim(values: np.ndarray, lows: list[float], highs: list[float]) -> tuple[float, float]:
    finite = np.concatenate([values, np.asarray(lows, dtype=float), np.asarray(highs, dtype=float)])
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (-1.0, 1.0)
    lo = min(0.0, float(np.min(finite)))
    hi = max(0.0, float(np.max(finite)))
    span = max(hi - lo, 1.0e-6)
    return (lo - 0.08 * span, hi + 0.10 * span)


def forest_axis(
    ax: plt.Axes,
    gain_summary: dict[tuple[str, str, str], dict[str, str]],
    pair_values: dict[tuple[str, str], np.ndarray],
    metric: str,
    case: str,
    show_y: bool,
    metric_label: str,
) -> None:
    method_colors = v35.base.COLORS
    y_positions = np.arange(len(v36.BASELINES))[::-1]
    rng = np.random.default_rng(20260809 + len(metric) * 29 + len(case) * 13)
    lows: list[float] = []
    highs: list[float] = []
    all_values: list[np.ndarray] = []

    for baseline in v36.BASELINES:
        row = gain_summary[(metric, case, baseline)]
        lows.append(float(row["improvement_ci95_low"]))
        highs.append(float(row["improvement_ci95_high"]))
        all_values.append(pair_values[(case, baseline)])

    xlim = compact_xlim(np.concatenate(all_values) if all_values else np.asarray([], dtype=float), lows, highs)

    for y, baseline in zip(y_positions, v36.BASELINES, strict=True):
        values = pair_values[(case, baseline)]
        row = gain_summary[(metric, case, baseline)]
        low = float(row["improvement_ci95_low"])
        high = float(row["improvement_ci95_high"])
        mean = float(row["improvement_mean"])
        color = method_colors[baseline]

        if values.size:
            jitter = rng.normal(0.0, 0.055, size=values.size)
            ax.scatter(
                values,
                np.full(values.size, y, dtype=float) + jitter,
                s=5.0,
                color=color,
                alpha=0.15,
                linewidths=0,
                zorder=1,
            )
        ax.hlines(y, low, high, color=color, lw=1.35, zorder=2)
        ax.plot(
            mean,
            y,
            marker="o",
            ms=4.0,
            mfc=color,
            mec="white",
            mew=0.45,
            color=color,
            zorder=3,
        )

    ax.axvline(0.0, color="#6E6E6E", lw=0.70, ls=(0, (3, 2)), zorder=0)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.55, len(v36.BASELINES) - 0.45)
    ax.set_title(f"{metric_label} · {v35.base.CASE_LABELS[case]}", fontsize=v35.base.PANEL_TITLE_FONT_SIZE, pad=2.0)
    ax.set_yticks(y_positions)
    if show_y:
        ax.set_yticklabels([v36.BASELINE_LABELS[m] for m in v36.BASELINES], fontsize=v35.base.AXIS_TICK_FONT_SIZE)
        for tick, baseline in zip(ax.get_yticklabels(), v36.BASELINES, strict=True):
            tick.set_color(method_colors[baseline])
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    ax.set_xlabel("gain", fontsize=v35.base.AXIS_LABEL_FONT_SIZE, labelpad=2.5)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=3))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{x:g}"))
    ax.tick_params(axis="both", labelsize=v35.base.AXIS_TICK_FONT_SIZE, length=2.0, pad=1.4)
    for side in ("right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_linewidth(0.60)
    ax.spines["bottom"].set_linewidth(0.60)
    ax.grid(False)


def panel_forest_single_row(fig: plt.Figure, subplot_spec, run_rows: list[dict[str, str]]) -> None:
    panel_grid = GridSpecFromSubplotSpec(2, 1, subplot_spec=subplot_spec, height_ratios=[0.18, 1.0], hspace=0.06)
    add_forest_label(fig.add_subplot(panel_grid[0, 0]))
    row_grid = GridSpecFromSubplotSpec(
        1,
        7,
        subplot_spec=panel_grid[1, 0],
        width_ratios=[1, 1, 1, 0.10, 1, 1, 1],
        wspace=0.28,
    )
    gain_rows = v36.build_gain_summary(run_rows)
    gain_summary = {(row["metric"], row["case"], row["baseline"]): row for row in gain_rows}
    nrmse_values = v36.pair_seed_improvements(run_rows, "nrmse", 100.0)
    crps_values = v36.pair_seed_improvements(run_rows, "crps", 1000.0)

    slots = [
        ("nrmse", "nRMSE", "wave", 0, True, nrmse_values),
        ("nrmse", "nRMSE", "spring", 1, False, nrmse_values),
        ("nrmse", "nRMSE", "heat", 2, False, nrmse_values),
        ("crps", "CRPS", "wave", 4, True, crps_values),
        ("crps", "CRPS", "spring", 5, False, crps_values),
        ("crps", "CRPS", "heat", 6, False, crps_values),
    ]
    spacer = fig.add_subplot(row_grid[0, 3])
    spacer.set_axis_off()
    for metric, metric_label, case, slot, show_y, values in slots:
        forest_axis(
            fig.add_subplot(row_grid[0, slot]),
            gain_summary,
            values,
            metric,
            case,
            show_y,
            metric_label,
        )


def write_gain_source(output_dir: Path, run_rows: list[dict[str, str]]) -> None:
    rows = v36.build_gain_summary(run_rows)
    out = output_dir / f"{OUTPUT_STEM}_gain_summary.csv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case",
                "metric",
                "baseline",
                "baseline_label",
                "improvement_mean",
                "improvement_ci95_low",
                "improvement_ci95_high",
                "n",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_contract(output_dir: Path) -> None:
    text = """Core conclusion:
Across three classical uncertain physical equations, APCE consistently reduces state error and predictive score relative to every numerically valid training-free baseline under a frozen 50-seed paired protocol.

Figure archetype:
Asymmetric mixed-modality figure.

Backend:
Python/matplotlib only.

Panel map:
a-c: representative dynamics for Wave, Spring and Heat.
d: final cognitive-weight map over candidate alpha values.
e-f: seed-wise nRMSE and CRPS boxplots.
g-h: 90% coverage and interval-width summaries.
i: one-row paired-difference forest summary. The first three small plots show nRMSE gain and the last three show CRPS gain for Wave, Spring and Heat. Positive values mean APCE is better than the named baseline.

Statistics:
n=50 paired seeds per system and method. Panel i shows paired seed differences with bootstrap 95% confidence intervals.
"""
    (output_dir / f"{OUTPUT_STEM}_contract.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=ROOT.parent / "ncs_chinese_submission" / "figures" / "source_data_figure2_method_summary.csv",
    )
    parser.add_argument(
        "--runs-csv",
        type=Path,
        default=ROOT.parent / "ncs_chinese_submission" / "figures" / "source_data_figure2_run_source_data_20260807.csv",
    )
    parser.add_argument(
        "--paired-csv",
        type=Path,
        default=ROOT.parent / "ncs_chinese_submission" / "figures" / "source_data_figure2_paired_comparisons.csv",
    )
    parser.add_argument(
        "--wave-npz",
        type=Path,
        default=ROOT / "figure2_full_representative_source" / "wave_full_representative_seed_2026080700.npz",
    )
    parser.add_argument(
        "--spring-npz",
        type=Path,
        default=ROOT / "figure2_full_representative_source" / "spring_full_representative_seed_2026080700.npz",
    )
    parser.add_argument(
        "--heat-npz",
        type=Path,
        default=ROOT / "figure2_full_representative_source" / "heat_full_representative_seed_2026080700.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT.parent / "ncs_chinese_submission" / "figures",
    )
    args = parser.parse_args()

    v35.base.set_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = read_csv(args.summary_csv)
    run_rows = read_csv(args.runs_csv)
    summary = {(row["case"], row["method"]): row for row in summary_rows}
    wave = np.load(args.wave_npz, allow_pickle=True)
    spring = np.load(args.spring_npz, allow_pickle=True)
    heat = np.load(args.heat_npz, allow_pickle=True)

    fig = plt.figure(figsize=(11.55, 11.05))
    outer = fig.add_gridspec(
        8,
        1,
        height_ratios=[1.05, 0.15, 0.12, 0.94, 0.31, 0.94, 0.09, 1.20],
        hspace=0.205,
        left=0.050,
        right=0.994,
        top=0.955,
        bottom=0.050,
    )

    top = GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[0, 0], wspace=0.29, width_ratios=[1, 1, 1, 0.92])
    v35.base.panel_wave(fig.add_subplot(top[0, 0]), wave)
    v35.base.panel_spring(fig.add_subplot(top[0, 1]), spring)
    v35.base.panel_heat(fig.add_subplot(top[0, 2]), heat)
    ax_weight = fig.add_subplot(top[0, 3])
    v35.panel_weight_heatmap(ax_weight, wave, spring, heat)

    v35.base.add_mid_stat_legend(fig.add_subplot(outer[1, 0]))
    fig.add_subplot(outer[2, 0]).set_axis_off()

    row2 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[3, 0], wspace=0.14)
    fig.add_subplot(outer[4, 0]).set_axis_off()
    row3 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[5, 0], wspace=0.14)
    for row_index, row_metrics in enumerate((v35.METRIC_GROUPS[:2], v35.METRIC_GROUPS[2:])):
        row_spec = row2 if row_index == 0 else row3
        for metric_index, (metric, ylabel, scale, letter, title) in enumerate(row_metrics):
            metric_spec = GridSpecFromSubplotSpec(1, 3, subplot_spec=row_spec[0, metric_index], wspace=0.18)
            for case_index, case in enumerate(v35.base.CASES):
                ax = fig.add_subplot(metric_spec[0, case_index])
                if metric in {"nrmse", "crps"}:
                    v35.base.metric_case_box_panel(
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
                    v35.base.metric_case_panel(
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

    fig.add_subplot(outer[6, 0]).set_axis_off()
    panel_forest_single_row(fig, outer[7, 0], run_rows)

    v35.base.freeze_all_compact_numeric_ticklabels(fig)
    alpha_tick_labels = getattr(ax_weight, "figure2_alpha_tick_labels", [])
    if alpha_tick_labels:
        ax_weight.set_xticks(np.arange(len(alpha_tick_labels)) + 0.5)
        ax_weight.set_xticklabels(alpha_tick_labels, fontsize=v35.base.AXIS_TICK_FONT_SIZE)

    out_base = args.output_dir / OUTPUT_STEM
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    write_gain_source(args.output_dir, run_rows)
    write_contract(args.output_dir)
    qa = {
        "figure": OUTPUT_STEM,
        "layout": "version 37: starts from v36; keeps panels a-h unchanged and changes panel i from a two-row three-column forest summary to a single-row six-panel forest summary to reduce empty space",
        "backend": "python/matplotlib",
        "font": "Arial with sans-serif fallback",
        "formats": ["svg", "pdf", "png", "tiff"],
        "panel_i": "six small forest plots in one row: nRMSE Wave/Spring/Heat followed by CRPS Wave/Spring/Heat",
        "n": "50 paired seeds per case and method",
    }
    (args.output_dir / f"{OUTPUT_STEM}_qa.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(out_base)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
