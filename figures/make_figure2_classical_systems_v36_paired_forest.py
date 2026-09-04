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


OUTPUT_STEM = "figure2_classical_uncertain_systems_v36_paired_forest"

BASELINES = ("denkf", "letkf", "iensf", "pce")
BASELINE_LABELS = {
    "denkf": "DEnKF",
    "letkf": "LETKF",
    "iensf": "IEnSF",
    "pce": "PCE",
}
FOREST_METRICS = (
    ("nrmse", "nRMSE gain (%)", 100.0),
    ("crps", r"CRPS gain ($10^{-3}$)", 1000.0),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def group_summary(summary_rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["case"], row["method"]): row for row in summary_rows}


def pair_seed_improvements(
    run_rows: list[dict[str, str]],
    metric: str,
    scale: float,
) -> dict[tuple[str, str], np.ndarray]:
    lookup: dict[tuple[str, str, str], float] = {}
    for row in run_rows:
        if row.get("valid", "True") != "True":
            continue
        case = row["case"]
        seed = row["seed"]
        method = row["method"]
        value = row.get(metric, "")
        if value == "":
            continue
        lookup[(case, seed, method)] = float(value) * scale

    improvements: dict[tuple[str, str], np.ndarray] = {}
    for case in v35.base.CASES:
        for baseline in BASELINES:
            baseline_vals = []
            apce_vals = []
            for key, value in lookup.items():
                c, seed, method = key
                if c != case:
                    continue
                if method == "apce":
                    apce_vals.append((seed, value))
                elif method == baseline:
                    baseline_vals.append((seed, value))
            baseline_map = {seed: value for seed, value in baseline_vals}
            apce_map = {seed: value for seed, value in apce_vals}
            common = sorted(set(baseline_map) & set(apce_map))
            improvements[(case, baseline)] = np.asarray(
                [baseline_map[seed] - apce_map[seed] for seed in common],
                dtype=float,
            )
    return improvements


def paired_summary_rows(paired_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in paired_rows:
        metric = row["metric"]
        if metric not in {"nrmse", "crps"}:
            continue
        scale = 100.0 if metric == "nrmse" else 1000.0
        rows.append(
            {
                "case": row["case"],
                "metric": metric,
                "baseline": row["baseline"],
                "baseline_label": BASELINE_LABELS[row["baseline"]],
                "improvement_mean": f"{(-float(row['mean_difference_apce_minus_baseline']) * scale):.10g}",
                "improvement_ci95_low": f"{(-float(row['ci95_high']) * scale):.10g}",
                "improvement_ci95_high": f"{(-float(row['ci95_low']) * scale):.10g}",
                "p_holm": row["p_holm"],
                "n": row["n"],
            }
        )
    return rows


def bootstrap_mean_ci(values: np.ndarray, seed: int, draws: int = 2000) -> tuple[float, float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(np.mean(values))
    if values.size == 1:
        return (mean, mean, mean)
    rng = np.random.default_rng(seed)
    boot = rng.choice(values, size=(draws, values.size), replace=True).mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return (mean, float(low), float(high))


def build_gain_summary(run_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for metric, _, scale in FOREST_METRICS:
        pair_improvements = pair_seed_improvements(run_rows, metric, scale)
        for case in v35.base.CASES:
            for baseline in BASELINES:
                values = pair_improvements[(case, baseline)]
                mean, low, high = bootstrap_mean_ci(values, seed=20260807 + 101 * (0 if metric == "nrmse" else 1) + 7 * len(case) + len(baseline))
                rows.append(
                    {
                        "case": case,
                        "metric": metric,
                        "baseline": baseline,
                        "baseline_label": BASELINE_LABELS[baseline],
                        "improvement_mean": f"{mean:.10g}",
                        "improvement_ci95_low": f"{low:.10g}",
                        "improvement_ci95_high": f"{high:.10g}",
                        "n": str(values.size),
                    }
                )
    return rows


def plot_forest_axis(
    ax: plt.Axes,
    paired_summary: dict[tuple[str, str], dict[str, str]],
    pair_improvements: dict[tuple[str, str], np.ndarray],
    case: str,
    metric: str,
    scale: float,
    row_index: int,
    col_index: int,
    xlim: tuple[float, float],
) -> None:
    metric_title = "nRMSE" if metric == "nrmse" else "CRPS"
    method_colors = v35.base.COLORS
    y_positions = np.arange(len(BASELINES))[::-1]
    rng = np.random.default_rng(20260807 + 37 * row_index + 11 * col_index + (0 if metric == "nrmse" else 19))

    for y, baseline in zip(y_positions, BASELINES, strict=True):
        pair_vals = pair_improvements[(case, baseline)]
        if pair_vals.size:
            jitter = rng.normal(0.0, 0.06, size=pair_vals.size)
            ax.scatter(
                pair_vals,
                np.full(pair_vals.size, y, dtype=float) + jitter,
                s=5.5,
                color=method_colors[baseline],
                alpha=0.16,
                linewidths=0,
                zorder=1,
            )

        row = paired_summary[(case, baseline)]
        mean = float(row["improvement_mean"])
        low = float(row["improvement_ci95_low"])
        high = float(row["improvement_ci95_high"])

        ax.hlines(y, low, high, color=method_colors[baseline], lw=1.4, zorder=2)
        ax.plot(
            mean,
            y,
            marker="o",
            ms=4.0 if baseline != "pce" else 4.4,
            mfc=method_colors[baseline],
            mec="white",
            mew=0.45,
            color=method_colors[baseline],
            zorder=3,
        )

    ax.axvline(0.0, color="#606060", lw=0.7, ls=(0, (3, 2)), zorder=0)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.55, len(BASELINES) - 0.45)
    ax.set_yticks(y_positions)
    if col_index == 0:
        ax.set_yticklabels([BASELINE_LABELS[m] for m in BASELINES], fontsize=v35.base.AXIS_TICK_FONT_SIZE)
        for tick, baseline in zip(ax.get_yticklabels(), BASELINES, strict=True):
            tick.set_color(method_colors[baseline])
        ax.set_ylabel(f"{metric_title} gain", fontsize=v35.base.AXIS_LABEL_FONT_SIZE, labelpad=7.5)
    else:
        ax.set_yticklabels([])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{x:g}"))
    ax.tick_params(axis="both", labelsize=v35.base.AXIS_TICK_FONT_SIZE, length=2.0, pad=1.4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_linewidth(0.60)
    ax.spines["bottom"].set_linewidth(0.60)
    ax.set_title(v35.base.CASE_LABELS[case], fontsize=v35.base.PANEL_TITLE_FONT_SIZE, pad=2.0)
    if row_index == 1:
        ax.set_xlabel("APCE gain", fontsize=v35.base.AXIS_LABEL_FONT_SIZE, labelpad=3.0)
    else:
        ax.tick_params(axis="x", labelbottom=False)


def panel_forest_summary(
    fig: plt.Figure,
    subplot_spec,
    paired_rows: list[dict[str, str]],
    run_rows: list[dict[str, str]],
) -> None:
    panel_grid = GridSpecFromSubplotSpec(2, 1, subplot_spec=subplot_spec, height_ratios=[0.12, 1.0], hspace=0.05)
    label_ax = fig.add_subplot(panel_grid[0, 0])
    label_ax.set_axis_off()
    label_ax.text(-0.02, 0.62, "i", transform=label_ax.transAxes, ha="left", va="bottom", fontsize=v35.base.PANEL_LABEL_FONT_SIZE)
    label_ax.text(0.04, 0.62, "APCE gain over baselines", transform=label_ax.transAxes, ha="left", va="bottom", fontsize=v35.base.PANEL_TITLE_FONT_SIZE)
    label_ax.text(0.04, 0.16, "paired-difference forest summary; positive values mean APCE is better", transform=label_ax.transAxes, ha="left", va="bottom", fontsize=v35.base.AXIS_TICK_FONT_SIZE)

    metric_grid = GridSpecFromSubplotSpec(2, 3, subplot_spec=panel_grid[1, 0], wspace=0.18, hspace=0.28)
    processed_rows = build_gain_summary(run_rows)
    for row_index, (metric, _, scale) in enumerate(FOREST_METRICS):
        metric_rows = [row for row in processed_rows if row["metric"] == metric]
        metric_summary = {(row["case"], row["baseline"]): row for row in metric_rows}
        pair_improvements = pair_seed_improvements(run_rows, metric, scale)
        all_low = min(float(row["improvement_ci95_low"]) for row in metric_rows)
        all_high = max(float(row["improvement_ci95_high"]) for row in metric_rows)
        span = max(all_high - all_low, 1.0e-6)
        xlim = (min(-0.06 * span, all_low - 0.10 * span), all_high + 0.06 * span)

        for col_index, case in enumerate(v35.base.CASES):
            ax = fig.add_subplot(metric_grid[row_index, col_index])
            plot_forest_axis(
                ax,
                metric_summary,
                pair_improvements,
                case,
                metric,
                scale,
                row_index,
                col_index,
                xlim,
            )
            if row_index == 0:
                ax.tick_params(axis="x", labelbottom=False)


def write_forest_source_data(output_dir: Path, paired_rows: list[dict[str, str]], run_rows: list[dict[str, str]]) -> None:
    rows = build_gain_summary(run_rows)
    out = output_dir / f"{OUTPUT_STEM}_gain_summary.csv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case", "metric", "baseline", "baseline_label", "improvement_mean", "improvement_ci95_low", "improvement_ci95_high", "n"],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Keep the paired seed and run-level sources alongside the summary table.
    shutil.copy2(output_dir / "source_data_figure2_paired_comparisons.csv", output_dir / "source_data_figure2_paired_comparisons_copy.csv")
    shutil.copy2(output_dir / "source_data_figure2_run_source_data_20260807.csv", output_dir / "source_data_figure2_run_source_data_20260807_copy.csv")


def write_contract(output_dir: Path) -> None:
    text = """Core conclusion:
Across three classical uncertain physical equations, APCE consistently reduces state error and predictive score relative to every numerically valid training-free baseline under a frozen 50-seed paired protocol.

Figure archetype:
Asymmetric mixed-modality figure.

Target journal/output:
Nature Computational Science main-text double-column figure; SVG, PDF, PNG and 600 dpi TIFF.

Backend:
Python/matplotlib only.

Panel map:
a: equation-suite and frozen pairing protocol.
b: representative wave time-space reconstruction.
c: spring phase trajectory.
d: heat terminal profile with candidate-alpha cognitive weights.
e-f: seed-wise nRMSE and CRPS boxplots.
g-h: 90% coverage and interval-width bars.
i: paired-difference forest summary showing APCE gain over DEnKF, LETKF, IEnSF and PCE in each classical system.

Statistics:
n=50 paired seeds per system and method. Panels e and f show seed-wise boxplots. Panels g and h show mean bars with 95% CI and overlaid seed-level dots. Panel i summarizes paired differences; positive values mean APCE is better. APCE is highlighted by the same warm accent used throughout the statistical panels.
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
    paired_rows = read_csv(args.paired_csv)
    summary = group_summary(summary_rows)
    wave = np.load(args.wave_npz, allow_pickle=True)
    spring = np.load(args.spring_npz, allow_pickle=True)
    heat = np.load(args.heat_npz, allow_pickle=True)

    fig = plt.figure(figsize=(11.55, 12.20))
    outer = fig.add_gridspec(
        8,
        1,
        height_ratios=[1.05, 0.15, 0.12, 0.94, 0.31, 0.94, 0.11, 1.92],
        hspace=0.205,
        left=0.050,
        right=0.994,
        top=0.955,
        bottom=0.040,
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
    panel_forest_summary(fig, outer[7, 0], paired_rows, run_rows)

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

    for source in (args.summary_csv, args.runs_csv, args.paired_csv):
        shutil.copy2(source, args.output_dir / f"source_data_{source.name}")
    write_forest_source_data(args.output_dir, paired_rows, run_rows)
    write_contract(args.output_dir)
    qa = {
        "figure": OUTPUT_STEM,
        "layout": "version 36: keeps the accepted a-h panels from v35 but replaces the radar-style panel i with a paired-difference forest summary that directly shows APCE gain over each baseline in each classical system",
        "backend": "python/matplotlib",
        "font": "Arial with sans-serif fallback",
        "font_weight": "regular except panel letters",
        "formats": ["svg", "pdf", "png", "tiff"],
        "paired_summary": "positive values mean APCE is better; rows show DEnKF, LETKF, IEnSF and PCE; columns show Wave, Spring and Heat",
        "metrics": ["nRMSE gain (%)", r"CRPS gain ($10^{-3}$)"],
    }
    (args.output_dir / f"{OUTPUT_STEM}_qa.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(out_base)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
