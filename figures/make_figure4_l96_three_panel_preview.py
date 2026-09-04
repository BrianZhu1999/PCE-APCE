from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
TRACE_SOURCE = ROOT / "figures" / "source_data" / "figure4_lorenz96_1024_obs128_t8_seed2026080601"
METRIC_SOURCE = ROOT / "figures" / "source_data" / "figure4_lorenz96_1024_obs128_t8_final5" / "run_source_data.csv"
OUTPUT_DIR = ROOT / "figures" / "figure4_l96_three_panel_preview"

FONT_PANEL = 22
FONT_TITLE = 14
FONT_LEGEND = 14
FONT_AXIS = 13
FONT_TICK = 11
FONT_ROW = 12

METHODS = ["apce", "pce", "bma_static", "aug_enkf"]
LABELS = {
    "truth": "GT",
    "apce": "APCE",
    "pce": "PCE",
    "bma_static": "BMA",
    "aug_enkf": "Aug-EnKF",
}
COLORS = {
    "truth": "#E84A3A",
    "apce": "#2ECC71",
    "pce": "#3775BA",
    "bma_static": "#F39C12",
    "aug_enkf": "#9B59B6",
}
TRACE_FILES = {
    "apce": "apce_trace.npz",
    "pce": "pce_trace.npz",
    "bma_static": "bma_trace.npz",
    "aug_enkf": "aug_enkf_trace.npz",
}
STATE_INDICES = np.asarray([0, 256, 512, 768], dtype=int)


# Figure contract
# Core conclusion: under the D=1024, 128-sensor, time-8 protocol, PCE/APCE
# reconstruct representative state trajectories, APCE preserves the global
# trajectory geometry, and the five paired seeds support lower state and
# probabilistic errors than the strong controls.
# Archetype: asymmetric mixed-modality figure.
# Panel roles: a representative time-series evidence; b dynamical geometry;
# c five-seed quantitative comparison.


def clean_number(value: float, _position: int | None = None) -> str:
    return f"{value:g}"


def configure_matplotlib() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "font.size": FONT_AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_traces() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with np.load(TRACE_SOURCE / "shared_asset.npz", allow_pickle=False) as data:
        truth = np.asarray(data["truth"], dtype=float)
    traces["truth"] = (np.arange(truth.shape[0], dtype=float) * 0.01, truth)
    for method, filename in TRACE_FILES.items():
        with np.load(TRACE_SOURCE / filename, allow_pickle=False) as data:
            traces[method] = (
                np.asarray(data["times"], dtype=float),
                np.asarray(data["mean_states"], dtype=float),
            )
    return traces


def load_metric_rows() -> list[dict[str, str]]:
    with METRIC_SOURCE.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def pca_project(truth: np.ndarray, apce: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    combined = np.vstack([truth, apce])
    mean = combined.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(combined - mean, full_matrices=False)
    basis = vt[:3].T
    return (truth - mean) @ basis, (apce - mean) @ basis


def style_timeseries_ax(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#C9CCCE")
        spine.set_linewidth(0.65)


def draw_panel_a(fig: plt.Figure, spec, traces: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    subgrid = spec.subgridspec(2, 2, wspace=0.24, hspace=0.34)
    ordered = ["truth", "apce", "pce", "bma_static", "aug_enkf"]
    line_styles = {
        "truth": ("--", 2.10, 0.92, 10),
        "apce": ("-", 1.85, 0.95, 8),
        "pce": ("-.", 1.50, 0.90, 6),
        "bma_static": ((0, (5, 2)), 1.42, 0.88, 5),
        "aug_enkf": ((0, (1.2, 1.6)), 1.45, 0.88, 4),
    }
    axes: list[plt.Axes] = []
    for index, state_index in enumerate(STATE_INDICES):
        row, col = divmod(index, 2)
        ax = fig.add_subplot(subgrid[row, col])
        axes.append(ax)
        values = [traces[key][1][:, state_index] for key in ordered]
        low = min(float(np.min(value)) for value in values)
        high = max(float(np.max(value)) for value in values)
        padding = max(0.08, 0.075 * (high - low))
        for key in ordered:
            times, states = traces[key]
            linestyle, linewidth, alpha, zorder = line_styles[key]
            ax.plot(
                times,
                states[:, state_index],
                color=COLORS[key],
                linestyle=linestyle,
                lw=linewidth,
                alpha=alpha,
                label="Truth" if key == "truth" else LABELS[key],
                solid_capstyle="round",
                zorder=zorder,
            )
        ax.set_xlim(traces["truth"][0][0], traces["truth"][0][-1])
        ax.set_ylim(low - padding, high + padding)
        ax.set_title(f"$x_{{{state_index}}}$", fontsize=FONT_TITLE, pad=6)
        ax.grid(False)
        ax.xaxis.set_major_formatter(FuncFormatter(clean_number))
        ax.yaxis.set_major_formatter(FuncFormatter(clean_number))
        ax.tick_params(labelsize=FONT_TICK)
        if row == 1:
            ax.set_xlabel("$t$", fontsize=FONT_AXIS, labelpad=2)
        if col == 0:
            ax.set_ylabel("$x_i$", fontsize=FONT_AXIS, labelpad=2)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#AEB3B6")
            spine.set_linewidth(0.75)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.00, 1.25),
        ncol=5,
        fontsize=FONT_LEGEND,
        handlelength=2.0,
        columnspacing=0.95,
        handletextpad=0.45,
        borderaxespad=0,
    )
    axes[0].text(
        -0.17,
        1.24,
        "a",
        transform=axes[0].transAxes,
        fontsize=FONT_PANEL,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def style_3d(ax: plt.Axes) -> None:
    ax.view_init(elev=25, azim=-54)
    ax.set_box_aspect((1.0, 1.0, 0.82))
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("#F4F6F7")
        axis.pane.set_edgecolor("#D4D9DC")
        axis.line.set_color("#A8AEB2")
        axis.line.set_linewidth(0.7)
        axis.set_major_formatter(FuncFormatter(clean_number))
    ax.tick_params(labelsize=FONT_TICK, pad=0)


def draw_panel_b(fig: plt.Figure, spec, traces: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    ax = fig.add_subplot(spec, projection="3d")
    truth_phase, apce_phase = pca_project(traces["truth"][1], traces["apce"][1])
    style_3d(ax)
    ax.plot(
        truth_phase[:, 0],
        truth_phase[:, 1],
        truth_phase[:, 2],
        color=COLORS["truth"],
        lw=2.05,
        alpha=0.80,
        label="Truth",
    )
    ax.plot(
        apce_phase[:, 0],
        apce_phase[:, 1],
        apce_phase[:, 2],
        color=COLORS["apce"],
        lw=1.65,
        alpha=0.96,
        label="APCE",
    )
    ax.scatter(truth_phase[0, 0], truth_phase[0, 1], truth_phase[0, 2], s=27, color=COLORS["truth"], depthshade=False)
    ax.scatter(apce_phase[-1, 0], apce_phase[-1, 1], apce_phase[-1, 2], s=31, color=COLORS["apce"], marker=">", depthshade=False)
    ax.set_xlabel("PC1", fontsize=FONT_AXIS, labelpad=2)
    ax.set_ylabel("PC2", fontsize=FONT_AXIS, labelpad=2)
    ax.set_zlabel("PC3", fontsize=FONT_AXIS, labelpad=0)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.00, 1.03),
        fontsize=FONT_LEGEND,
        handlelength=1.8,
        labelspacing=0.30,
        borderaxespad=0,
    )
    ax.text2D(
        -0.04,
        1.05,
        "b",
        transform=ax.transAxes,
        fontsize=FONT_PANEL,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def draw_boxplot(ax: plt.Axes, rows: list[dict[str, str]], metric: str, ylabel: str, *, show_labels: bool) -> None:
    values = [
        np.asarray([float(row[metric]) for row in rows if row["method"] == method], dtype=float)
        for method in METHODS
    ]
    bp = ax.boxplot(
        values,
        patch_artist=True,
        widths=0.58,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.45},
        whiskerprops={"color": "#555555", "linewidth": 1.0},
        capprops={"color": "#555555", "linewidth": 1.0},
    )
    for patch, method in zip(bp["boxes"], METHODS, strict=True):
        patch.set_facecolor(COLORS[method])
        patch.set_edgecolor(COLORS[method])
        patch.set_alpha(0.83)
        patch.set_linewidth(1.25)
    rng = np.random.default_rng(20260815)
    for index, (method, array) in enumerate(zip(METHODS, values, strict=True), start=1):
        jitter = index + rng.uniform(-0.10, 0.10, len(array))
        ax.scatter(
            jitter,
            array,
            s=23,
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.45,
            alpha=0.95,
            zorder=4,
        )
    ax.set_ylabel(ylabel, fontsize=FONT_AXIS)
    ax.set_xticks(
        np.arange(1, len(METHODS) + 1),
        [LABELS[method] for method in METHODS] if show_labels else [""] * len(METHODS),
        rotation=34,
        ha="right",
        fontsize=FONT_TICK,
    )
    ax.yaxis.set_major_formatter(FuncFormatter(clean_number))
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_panel_c(fig: plt.Figure, spec, rows: list[dict[str, str]]) -> None:
    subgrid = spec.subgridspec(2, 1, hspace=0.47)
    ax_nrmse = fig.add_subplot(subgrid[0, 0])
    ax_crps = fig.add_subplot(subgrid[1, 0])
    draw_boxplot(ax_nrmse, rows, "nrmse", "nRMSE", show_labels=False)
    draw_boxplot(ax_crps, rows, "crps", "CRPS", show_labels=True)
    ax_nrmse.set_ylim(0.085, 0.165)
    ax_crps.set_ylim(0.175, 0.270)
    ax_nrmse.text(
        -0.20,
        1.10,
        "c",
        transform=ax_nrmse.transAxes,
        fontsize=FONT_PANEL,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def save_all(fig: plt.Figure, base: Path) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix, kwargs in {
        ".svg": {},
        ".pdf": {},
        ".png": {"dpi": 650},
        ".tiff": {"dpi": 650},
    }.items():
        path = base.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", pad_inches=0.035, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> None:
    configure_matplotlib()
    traces = load_traces()
    rows = load_metric_rows()
    fig = plt.figure(figsize=(17.2, 5.6))
    grid = fig.add_gridspec(
        1,
        3,
        left=0.035,
        right=0.992,
        bottom=0.105,
        top=0.925,
        wspace=0.12,
        width_ratios=[10.2, 4.05, 3.45],
    )
    draw_panel_a(fig, grid[0, 0], traces)
    draw_panel_b(fig, grid[0, 1], traces)
    draw_panel_c(fig, grid[0, 2], rows)
    outputs = save_all(fig, OUTPUT_DIR / "figure4_l96_three_panel_preview_v1")
    manifest = {
        "layout": "a: 2x2 overlaid time-series plots; b: PCA phase projection; c: nRMSE and CRPS boxplots",
        "case": "Lorenz-96 D=1024, observed states=128, observation interval=8",
        "representative_seed": 2026080601,
        "boxplot_seeds": [2026080600, 2026080601, 2026080602, 2026080603, 2026080604],
        "state_indices": [int(value) for value in STATE_INDICES],
        "trace_source": str(TRACE_SOURCE),
        "metric_source": str(METRIC_SOURCE),
        "outputs": [str(path) for path in outputs],
    }
    (OUTPUT_DIR / "qa_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
