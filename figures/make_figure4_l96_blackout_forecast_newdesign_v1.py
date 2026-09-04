from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib import patheffects as pe


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42


PANEL_LABEL_SIZE = 22
TITLE_SIZE = 14
LEGEND_SIZE = 14
AXIS_LABEL_SIZE = 13
TICK_SIZE = 11

METHOD_ORDER = ["Aug-EnKF", "BMA", "PCE", "APCE"]
METHOD_KEY_BY_LABEL = {
    "Aug-EnKF": "aug_enkf",
    "BMA": "bma_static",
    "PCE": "pce",
    "APCE": "apce",
}
METHOD_COLORS = {
    "Aug-EnKF": "#757575",
    "BMA": "#4E79A7",
    "PCE": "#D95F59",
    "APCE": "#2B9E8F",
    "Truth": "#202020",
}
METHOD_MARKERS = {
    "Aug-EnKF": "o",
    "BMA": "s",
    "PCE": "D",
    "APCE": "^",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except Exception:
        return np.nan


def add_panel_label(fig: plt.Figure, ax: plt.Axes, label: str, dx: float = -0.015, dy: float = 0.010) -> None:
    box = ax.get_position()
    fig.text(
        box.x0 + dx,
        box.y1 + dy,
        label,
        ha="left",
        va="bottom",
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        color="#111111",
    )


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=TICK_SIZE, width=1.0, length=3.5)


def style_image_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("#F8F8F8")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#D0D0D0")
    ax.tick_params(axis="both", labelsize=TICK_SIZE, width=0.8, length=3)


def format_tick_labels(ax: plt.Axes, *, x: bool = True, y: bool = True) -> None:
    def fmt(value: float, _pos: int | None = None) -> str:
        if abs(value) >= 10:
            return f"{value:.0f}"
        if abs(value - round(value)) < 1.0e-8:
            return f"{int(round(value))}"
        if abs(value) < 1:
            return f"{value:.2f}".rstrip("0").rstrip(".")
        return f"{value:.1f}".rstrip("0").rstrip(".")

    if x:
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
    if y:
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))


def draw_protocol(ax: plt.Axes, *, blackout_time: float, final_time: float) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    left, right = 0.06, 0.94
    y = 0.53
    x_blackout = left + (right - left) * blackout_time / final_time
    ax.plot([left, right], [y, y], lw=3.0, color="#1F2933", solid_capstyle="round")
    ax.add_patch(
        FancyArrowPatch(
            (right - 0.015, y),
            (right + 0.005, y),
            arrowstyle="-|>",
            mutation_scale=18,
            lw=2.2,
            color="#1F2933",
        )
    )
    ax.axvline(x_blackout, ymin=0.18, ymax=0.86, color="#D95F59", lw=2.0, ls="--")
    ax.text(left, 0.82, "assimilation", fontsize=TITLE_SIZE, color="#1F2933", ha="left", va="center")
    ax.text(x_blackout + 0.015, 0.82, "blackout forecast", fontsize=TITLE_SIZE, color="#1F2933", ha="left", va="center")
    ax.text(x_blackout, 0.11, f"t={blackout_time:.1f}", fontsize=AXIS_LABEL_SIZE, color="#D95F59", ha="center")
    ax.text(left, 0.11, "0", fontsize=AXIS_LABEL_SIZE, color="#333333", ha="center")
    ax.text(right, 0.11, f"{final_time:.1f}", fontsize=AXIS_LABEL_SIZE, color="#333333", ha="center")

    obs_x = np.linspace(left + 0.015, x_blackout - 0.018, 24)
    obs_y = y + 0.08 * np.sin(np.linspace(0, 5 * np.pi, obs_x.size))
    ax.scatter(obs_x, obs_y, s=16, facecolors="#4E79A7", edgecolors="white", linewidths=0.4, zorder=4)
    ax.add_patch(
        Rectangle(
            (x_blackout + 0.035, y - 0.095),
            right - x_blackout - 0.055,
            0.19,
            facecolor="#F6E8E6",
            edgecolor="#D95F59",
            linewidth=1.0,
            alpha=0.75,
        )
    )
    ax.text(x_blackout + 0.052, y, "no observations", fontsize=AXIS_LABEL_SIZE, color="#7E2C2A", va="center")

    legend_x = 0.69
    for i, label in enumerate(["Aug-EnKF", "BMA", "PCE", "APCE"]):
        x = legend_x + i * 0.075
        ax.plot([x, x + 0.035], [0.925, 0.925], lw=3.0, color=METHOD_COLORS[label], solid_capstyle="round")
        ax.text(x + 0.0175, 0.975, label, fontsize=LEGEND_SIZE - 3, ha="center", va="bottom", color="#333333")


def method_series(rows: list[dict[str, str]], metric: str) -> dict[str, dict[float, list[float]]]:
    grouped: dict[str, dict[float, list[float]]] = {label: defaultdict(list) for label in METHOD_ORDER}
    for row in rows:
        label = row["label"]
        if label not in grouped:
            continue
        lead = f(row, "lead_time")
        value = f(row, metric)
        if np.isfinite(lead) and np.isfinite(value):
            grouped[label][lead].append(value)
    return grouped


def plot_lead_metric(
    ax: plt.Axes,
    lead_rows: list[dict[str, str]],
    metric: str,
    ylabel: str,
    threshold: float | None = None,
    *,
    show_xlabel: bool = True,
) -> None:
    grouped = method_series(lead_rows, metric)
    for label in METHOD_ORDER:
        by_lead = grouped[label]
        leads = np.asarray(sorted(by_lead), dtype=float)
        values = [np.asarray(by_lead[lead], dtype=float) for lead in leads]
        mean = np.asarray([item.mean() for item in values])
        sem = np.asarray([item.std(ddof=1) / np.sqrt(max(item.size, 1)) if item.size > 1 else 0.0 for item in values])
        color = METHOD_COLORS[label]
        ax.plot(
            leads,
            mean,
            lw=2.5 if label in {"PCE", "APCE"} else 2.0,
            color=color,
            label=label,
            marker=METHOD_MARKERS[label],
            markevery=15,
            ms=5.0,
        )
        ax.fill_between(leads, mean - 1.96 * sem, mean + 1.96 * sem, color=color, alpha=0.12, linewidth=0)
    if threshold is not None:
        ax.axhline(threshold, color="#202020", lw=1.0, ls=(0, (3, 3)), alpha=0.65)
        ax.text(0.99, threshold + 0.006, f"{threshold:.2f}", ha="right", va="bottom", fontsize=TICK_SIZE, color="#333333")
    if show_xlabel:
        ax.set_xlabel("forecast lead time", fontsize=AXIS_LABEL_SIZE)
    else:
        ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE, labelpad=3)
    ax.set_xlim(0, 1.0)
    clean_axis(ax)
    format_tick_labels(ax)


def draw_metric_boxes(ax: plt.Axes, run_rows: list[dict[str, str]], metric: str, ylabel: str) -> None:
    data = []
    for label in METHOD_ORDER:
        values = [f(row, metric) for row in run_rows if row["label"] == label]
        data.append(np.asarray(values, dtype=float))
    positions = np.arange(len(METHOD_ORDER))
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.54,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#202020", "linewidth": 1.4},
        whiskerprops={"color": "#666666", "linewidth": 1.0},
        capprops={"color": "#666666", "linewidth": 1.0},
    )
    for patch, label in zip(bp["boxes"], METHOD_ORDER):
        patch.set_facecolor(METHOD_COLORS[label])
        patch.set_alpha(0.26 if label in {"Aug-EnKF", "BMA"} else 0.36)
        patch.set_edgecolor(METHOD_COLORS[label])
        patch.set_linewidth(1.3)
    for pos, label, values in zip(positions, METHOD_ORDER, data):
        jitter = np.linspace(-0.11, 0.11, values.size) if values.size > 1 else np.zeros_like(values)
        ax.scatter(
            np.full(values.shape, pos, dtype=float) + jitter,
            values,
            s=34,
            color=METHOD_COLORS[label],
            edgecolor="white",
            linewidth=0.65,
            zorder=5,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(["Aug", "BMA", "PCE", "APCE"], rotation=25, ha="right", fontsize=TICK_SIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
    clean_axis(ax)
    format_tick_labels(ax, x=False, y=True)


def draw_gain_heatmap(ax: plt.Axes, paired_rows: list[dict[str, str]]) -> None:
    row_order = [
        ("PCE", "Aug-EnKF"),
        ("APCE", "Aug-EnKF"),
        ("PCE", "BMA"),
        ("APCE", "BMA"),
    ]
    col_defs = [
        ("forecast_nrmse", "nRMSE"),
        ("forecast_crps", "CRPS"),
        ("skill_horizon_time_020", "skill@0.20"),
    ]
    by_pair = {(row["method_label"], row["baseline_label"]): row for row in paired_rows}
    matrix = np.full((len(row_order), len(col_defs)), np.nan)
    annotations = [["" for _ in col_defs] for _ in row_order]
    for r_i, pair in enumerate(row_order):
        row = by_pair.get(pair)
        if row is None:
            continue
        total = int(float(row["paired_seed_count"]))
        for c_i, (metric, _label) in enumerate(col_defs):
            wins = int(float(row[f"{metric}_win_count"]))
            matrix[r_i, c_i] = wins / max(total, 1)
            if metric == "skill_horizon_time_020":
                gain = f(row, f"{metric}_gain_mean")
                annotations[r_i][c_i] = f"{wins}/{total}\n+{gain:.3f}"
            else:
                gain = f(row, f"{metric}_gain_mean")
                annotations[r_i][c_i] = f"{wins}/{total}\n+{gain:.3f}"
    cmap = LinearSegmentedColormap.from_list("advantage", ["#F3F4F6", "#DDEEDB", "#6BBF73", "#247A3D"])
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(len(col_defs)))
    ax.set_xticklabels([label for _metric, label in col_defs], fontsize=TICK_SIZE)
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels([f"{m}/{b}" for m, b in row_order], fontsize=TICK_SIZE)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, annotations[i][j], ha="center", va="center", fontsize=TICK_SIZE, color="#162312")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("paired advantage", fontsize=TITLE_SIZE, pad=8)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("win fraction", fontsize=AXIS_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_SIZE, length=2)
    cbar.ax.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(
            lambda value, pos: f"{int(round(value))}" if abs(value - round(value)) < 1.0e-8 else f"{value:.1f}".rstrip("0").rstrip(".")
        )
    )


def make_figure(root: Path, out_base: Path, representative_seed: int) -> dict[str, str]:
    source = root / "source_data"
    run_rows = read_csv(source / "lorenz96_1024_blackout_run_source_data.csv")
    lead_rows = read_csv(source / "lorenz96_1024_blackout_lead_time_source_data.csv")
    paired_rows = read_csv(source / "lorenz96_1024_blackout_paired_gains.csv")

    shared = root / "shared_assets" / f"lorenz96_1024_shared_seed_{representative_seed}.npz"
    apce_trace = root / "artifacts" / "method_traces" / "lorenz96_1024" / "time8" / "apce" / f"seed_{representative_seed}.npz"
    with np.load(shared, allow_pickle=False) as data:
        truth = np.asarray(data["truth"], dtype=float)
        obs_indices = np.asarray(data["observation_indices"], dtype=int)
        config = json.loads(str(data["config_json"].item()))
    with np.load(apce_trace, allow_pickle=False) as data:
        apce = np.asarray(data["mean_states"], dtype=float)
        times = np.asarray(data["times"], dtype=float)
        blackout_step = int(np.asarray(data["blackout_start_step"]).item())

    final_time = float(times[-1])
    blackout_time = float(times[blackout_step])
    forecast = slice(blackout_step, None)
    truth_f = truth[forecast]
    apce_f = apce[forecast]
    truth_plot = truth_f - truth_f.mean(axis=1, keepdims=True)
    apce_plot = apce_f - apce_f.mean(axis=1, keepdims=True)
    time_f = times[forecast]
    vlim = float(np.nanpercentile(np.abs(np.concatenate([truth_plot.ravel(), apce_plot.ravel()])), 99.2))
    err = np.abs(apce_f - truth_f) / max(float(np.nanstd(truth_f)), 1.0e-12)
    elim = float(np.nanpercentile(err, 98.0))

    fig = plt.figure(figsize=(13.4, 8.5), facecolor="white")
    outer = fig.add_gridspec(
        3,
        12,
        height_ratios=[0.82, 3.65, 2.30],
        hspace=0.52,
        wspace=0.82,
        left=0.055,
        right=0.985,
        bottom=0.075,
        top=0.955,
    )

    ax_a = fig.add_subplot(outer[0, :])
    draw_protocol(ax_a, blackout_time=blackout_time, final_time=final_time)

    image_grid = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[1, :6], hspace=0.08)
    ax_b1 = fig.add_subplot(image_grid[0, 0])
    ax_b2 = fig.add_subplot(image_grid[1, 0], sharex=ax_b1, sharey=ax_b1)
    ax_b3 = fig.add_subplot(image_grid[2, 0], sharex=ax_b1, sharey=ax_b1)
    image_axes = [ax_b1, ax_b2, ax_b3]
    extent = [time_f[0], time_f[-1], 0, truth.shape[1]]
    im1 = ax_b1.imshow(truth_plot.T, origin="lower", aspect="auto", extent=extent, cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    im2 = ax_b2.imshow(apce_plot.T, origin="lower", aspect="auto", extent=extent, cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    im3 = ax_b3.imshow(err.T, origin="lower", aspect="auto", extent=extent, cmap="magma", vmin=0, vmax=elim)
    for ax, title in zip(image_axes, ["Truth anomaly", "APCE forecast anomaly", r"$|$forecast error$|/\sigma_{\mathrm{truth}}$"]):
        style_image_axis(ax)
        text_color = "white" if ax is ax_b3 else "#111111"
        txt = ax.text(0.012, 0.82, title, transform=ax.transAxes, fontsize=TITLE_SIZE, fontweight="bold", color=text_color)
        txt.set_path_effects([pe.withStroke(linewidth=2.2, foreground="black" if text_color == "white" else "white", alpha=0.72)])
        ax.axvline(blackout_time, color="#111111", lw=1.0, ls=(0, (3, 3)), alpha=0.75)
        ax.set_ylabel("state index", fontsize=AXIS_LABEL_SIZE)
        format_tick_labels(ax)
    ax_b1.tick_params(labelbottom=False)
    ax_b2.tick_params(labelbottom=False)
    ax_b3.set_xlabel("time", fontsize=AXIS_LABEL_SIZE)
    ax_b1.set_title("representative blackout forecast field", fontsize=TITLE_SIZE, pad=10)
    cbar1 = fig.colorbar(im1, ax=[ax_b1, ax_b2], fraction=0.032, pad=0.012)
    cbar1.set_label("state anomaly", fontsize=AXIS_LABEL_SIZE, labelpad=2)
    cbar1.ax.tick_params(labelsize=TICK_SIZE, length=2)
    cbar1.ax.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(
            lambda value, pos: f"{value:.0f}" if abs(value) >= 1 else f"{value:.2f}".rstrip("0").rstrip(".")
        )
    )
    cbar2 = fig.colorbar(im3, ax=ax_b3, fraction=0.032, pad=0.012)
    cbar2.set_label("error", fontsize=AXIS_LABEL_SIZE, labelpad=2)
    cbar2.ax.tick_params(labelsize=TICK_SIZE, length=2)
    cbar2.ax.yaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(
            lambda value, pos: f"{int(round(value))}" if abs(value - round(value)) < 1.0e-8 else f"{value:.2f}".rstrip("0").rstrip(".")
        )
    )

    right_grid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1, 7:], hspace=0.50)
    ax_c = fig.add_subplot(right_grid[0, 0])
    ax_d = fig.add_subplot(right_grid[1, 0], sharex=ax_c)
    plot_lead_metric(ax_c, lead_rows, "lead_nrmse", "lead nRMSE", threshold=0.20, show_xlabel=False)
    ax_c.set_title("forecast error over lead time", fontsize=TITLE_SIZE, pad=8)
    ax_c.legend(loc="upper left", fontsize=LEGEND_SIZE - 3, ncol=2, handlelength=1.5, columnspacing=1.0)
    ax_c.tick_params(labelbottom=False)
    plot_lead_metric(ax_d, lead_rows, "lead_crps", "lead CRPS")
    ax_d.set_title("probabilistic forecast error", fontsize=TITLE_SIZE, pad=8)

    bottom_left = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2, :6], wspace=0.46)
    ax_e1 = fig.add_subplot(bottom_left[0, 0])
    ax_e2 = fig.add_subplot(bottom_left[0, 1])
    draw_metric_boxes(ax_e1, run_rows, "forecast_nrmse", "forecast nRMSE")
    draw_metric_boxes(ax_e2, run_rows, "forecast_crps", "forecast CRPS")
    ax_e1.set_title("5-seed forecast distribution", fontsize=TITLE_SIZE, pad=8)
    ax_e2.set_title("probabilistic distribution", fontsize=TITLE_SIZE, pad=8)

    ax_f = fig.add_subplot(outer[2, 7:])
    draw_gain_heatmap(ax_f, paired_rows)

    add_panel_label(fig, ax_a, "a", dx=-0.030, dy=0.000)
    add_panel_label(fig, ax_b1, "b", dx=-0.030, dy=0.008)
    add_panel_label(fig, ax_c, "c", dx=-0.030, dy=0.008)
    add_panel_label(fig, ax_d, "d", dx=-0.030, dy=0.008)
    add_panel_label(fig, ax_e1, "e", dx=-0.030, dy=0.008)
    add_panel_label(fig, ax_f, "f", dx=-0.030, dy=0.008)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    for ext, kwargs in {
        "png": {"dpi": 450},
        "pdf": {},
        "svg": {},
        "tiff": {"dpi": 600},
    }.items():
        path = out_base.with_suffix(f".{ext}")
        fig.savefig(path, bbox_inches="tight", **kwargs)
        saved[ext] = str(path)
    plt.close(fig)

    qa = {
        "core_conclusion": "After the sensing blackout, PCE/APCE preserve lower forecast error and longer usable skill than Aug-EnKF/BMA in the 5-seed Lorenz-96-1024 smoke test.",
        "archetype": "asymmetric mixed-modality figure",
        "backend": "Python/matplotlib",
        "representative_seed": representative_seed,
        "quantitative_n": 5,
        "source_root": str(root),
        "source_files": {
            "run_source_data": str(source / "lorenz96_1024_blackout_run_source_data.csv"),
            "lead_time_source_data": str(source / "lorenz96_1024_blackout_lead_time_source_data.csv"),
            "paired_gains": str(source / "lorenz96_1024_blackout_paired_gains.csv"),
        },
        "exports": saved,
    }
    qa_path = out_base.with_suffix(".qa.json")
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    saved["qa"] = str(qa_path)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="New-design Lorenz-96 blackout forecast figure.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("<HILDA_RESULTS_ROOT>/results/figure4_lorenz96_1024_obs128_t8_blackout200_smoke_5seeds_20260815_gpu23"),
    )
    parser.add_argument("--representative-seed", type=int, default=2026080600)
    parser.add_argument("--out-base", type=Path, default=None)
    args = parser.parse_args()
    out_base = args.out_base or (args.root / "figures" / "figure4_l96_blackout_forecast_newdesign_v1")
    saved = make_figure(args.root, out_base, args.representative_seed)
    print(json.dumps(saved, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
