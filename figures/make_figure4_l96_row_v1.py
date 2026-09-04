from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE / "source_data" / "figure4_lorenz96_1024_obs128_t8_blackout200_smoke_5seeds"
OUTPUT_BASE = HERE / "figure4_l96_row_v1"

REMOTE_ROOT = (
    "<HILDA_RESULTS_ROOT>/results/"
    "figure4_lorenz96_1024_obs128_t8_blackout200_smoke_5seeds_20260815_gpu23"
)

METHODS = ["Aug-EnKF", "BMA", "PCE", "APCE"]
METHOD_KEY = {
    "Aug-EnKF": "aug_enkf",
    "BMA": "bma_static",
    "PCE": "pce",
    "APCE": "apce",
}
METHOD_COLORS = {
    "Truth": "#202020",
    "Aug-EnKF": "#7A7F85",
    "BMA": "#D6A542",
    "PCE": "#2F74B5",
    "APCE": "#27A36C",
}
METHOD_LW = {
    "Truth": 2.25,
    "Aug-EnKF": 1.65,
    "BMA": 1.75,
    "PCE": 2.25,
    "APCE": 2.45,
}
METHOD_ALPHA = {
    "Truth": 0.98,
    "Aug-EnKF": 0.78,
    "BMA": 0.82,
    "PCE": 0.96,
    "APCE": 0.98,
}
METHOD_MARKERS = {
    "Aug-EnKF": "o",
    "BMA": "s",
    "PCE": "D",
    "APCE": "^",
}

# Figure 4 hard rules.
FONT_PANEL = 22
FONT_TITLE = 14
FONT_LEGEND = 14
FONT_AXIS = 13
FONT_TICK = 11

# Copied from the accepted KSE two-row Figure 4 script.
FIG_W = 17.98
FIG_H = 3.28
PANEL_W = 1.80
C_PANEL_W = 2.20
LEFT_MARGIN = 0.28
RIGHT_MARGIN = 0.22
GAP_ARROW = 0.74
GAP_SMALL = 0.24
GAP_MAJOR = 0.74
GAP_C = 0.65
QUANT_SHIFT = 0.42

PANEL_LEFTS = [
    LEFT_MARGIN,
    LEFT_MARGIN + PANEL_W + GAP_ARROW,
    LEFT_MARGIN + PANEL_W + GAP_ARROW + PANEL_W + GAP_SMALL,
    LEFT_MARGIN + PANEL_W + GAP_ARROW + 2 * (PANEL_W + GAP_SMALL),
    LEFT_MARGIN + PANEL_W + GAP_ARROW + 3 * PANEL_W + 2 * GAP_SMALL + GAP_MAJOR,
    LEFT_MARGIN + PANEL_W + GAP_ARROW + 3 * PANEL_W + 2 * GAP_SMALL + GAP_MAJOR + C_PANEL_W + GAP_C,
    LEFT_MARGIN + PANEL_W + GAP_ARROW + 3 * PANEL_W + 2 * GAP_SMALL + GAP_MAJOR + 2 * C_PANEL_W + 2 * GAP_C,
]
QUANT_LEFTS = [PANEL_LEFTS[4] + QUANT_SHIFT, PANEL_LEFTS[5] + QUANT_SHIFT, PANEL_LEFTS[6] + QUANT_SHIFT]

ROW_BOTTOM = 0.57
ROW_TOP = 2.67
ROW_H = ROW_TOP - ROW_BOTTOM


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.9,
            "axes.grid": False,
            "legend.frameon": False,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
        }
    )


def clean_number(x: float, _pos: int | None = None) -> str:
    return f"{x:g}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def local_trace_path(method_key: str, seed: int) -> Path:
    return DATA_ROOT / "artifacts" / "method_traces" / "lorenz96_1024" / "time8" / method_key / f"seed_{seed}.npz"


def add_axes_inches(fig: plt.Figure, left: float, bottom: float, width: float, height: float, **kwargs) -> plt.Axes:
    return fig.add_axes([left / FIG_W, bottom / FIG_H, width / FIG_W, height / FIG_H], **kwargs)


def add_panel_label(fig: plt.Figure, left: float, top: float, label: str) -> None:
    fig.text(
        (left - 0.14) / FIG_W,
        (top + 0.32) / FIG_H,
        label,
        ha="left",
        va="top",
        fontsize=FONT_PANEL,
        fontweight="bold",
        color="#111111",
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=FONT_TICK, length=3.2, width=0.9, pad=2)
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(clean_number))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(clean_number))


def load_all(seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray, int, dict[str, object]]:
    shared = load_npz(DATA_ROOT / "shared_assets" / f"lorenz96_1024_shared_seed_{seed}.npz")
    truth = np.asarray(shared["truth"], dtype=float)
    obs_indices = np.asarray(shared["observation_indices"], dtype=int)
    config = json.loads(str(shared["config_json"].item()))
    traces: dict[str, np.ndarray] = {}
    times: np.ndarray | None = None
    blackout_step: int | None = None
    for label in METHODS:
        trace = load_npz(local_trace_path(METHOD_KEY[label], seed))
        traces[label] = np.asarray(trace["mean_states"], dtype=float)
        if times is None:
            times = np.asarray(trace["times"], dtype=float)
        if blackout_step is None:
            blackout_step = int(np.asarray(trace["blackout_start_step"]).item())
    if times is None or blackout_step is None:
        raise RuntimeError("Trace metadata is incomplete.")
    return truth, times, traces, obs_indices, blackout_step, config


def draw_state_traces(
    fig: plt.Figure,
    left: float,
    bottom: float,
    width: float,
    height: float,
    truth: np.ndarray,
    times: np.ndarray,
    traces: dict[str, np.ndarray],
    blackout_step: int,
) -> None:
    state_indices = [0, 256, 512, 768]
    gap = 0.06
    sub_h = (height - gap * (len(state_indices) - 1)) / len(state_indices)
    blackout_time = float(times[blackout_step])
    for row, idx in enumerate(state_indices):
        y = bottom + (len(state_indices) - 1 - row) * (sub_h + gap)
        ax = add_axes_inches(fig, left, y, width, sub_h)
        ax.plot(times, truth[:, idx], color=METHOD_COLORS["Truth"], lw=METHOD_LW["Truth"], alpha=METHOD_ALPHA["Truth"], label="Truth")
        for label in METHODS:
            ax.plot(
                times,
                traces[label][:, idx],
                color=METHOD_COLORS[label],
                lw=METHOD_LW[label],
                alpha=METHOD_ALPHA[label],
                label=label,
            )
        ax.axvline(blackout_time, color="#202020", lw=1.0, ls=(0, (3, 3)), alpha=0.65)
        ax.set_xlim(float(times[0]), float(times[-1]))
        y_all = np.concatenate([truth[:, idx]] + [traces[label][:, idx] for label in METHODS])
        lo, hi = np.nanpercentile(y_all, [1, 99])
        pad = 0.16 * max(float(hi - lo), 1.0)
        ax.set_ylim(float(lo - pad), float(hi + pad))
        ax.text(
            0.012,
            0.74,
            rf"$x_{{{idx}}}$",
            transform=ax.transAxes,
            fontsize=FONT_AXIS,
            ha="left",
            va="center",
            color="#111111",
        )
        style_axis(ax)
        if row < len(state_indices) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("time", fontsize=FONT_AXIS, labelpad=1)
        if row == 0:
            ax.set_title("state traces", fontsize=FONT_TITLE, pad=7)
        if row == 1:
            ax.set_ylabel("state value", fontsize=FONT_AXIS, labelpad=5)


def pca_basis(traces: list[np.ndarray], start_step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forecast_states = [trace[start_step:] for trace in traces]
    stacked = np.concatenate(forecast_states, axis=0)
    mean = stacked.mean(axis=0)
    centered = stacked - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:3].T
    explained = (singular_values[:3] ** 2) / max(float((singular_values**2).sum()), 1.0e-30)
    return mean, components, explained


def project(trace: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (trace - mean) @ components


def clean_3d_axes(ax: plt.Axes) -> None:
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor("#D8D8D8")
        axis._axinfo["grid"]["linewidth"] = 0.0
        axis._axinfo["tick"]["inward_factor"] = 0.0
        axis._axinfo["tick"]["outward_factor"] = 0.22
    ax.tick_params(axis="both", labelsize=FONT_TICK, pad=0)
    ax.zaxis.set_tick_params(labelsize=FONT_TICK, pad=0)
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(clean_number))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(clean_number))
    ax.zaxis.set_major_formatter(mpl.ticker.FuncFormatter(clean_number))


def set_equal_limits(ax: plt.Axes, coords: np.ndarray) -> None:
    mins = np.nanmin(coords, axis=0)
    maxs = np.nanmax(coords, axis=0)
    centers = 0.5 * (mins + maxs)
    ranges = np.maximum(maxs - mins, 1.0e-6)
    radius = 0.47 * ranges.max()
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def draw_pca(
    fig: plt.Figure,
    left: float,
    bottom: float,
    width: float,
    height: float,
    truth: np.ndarray,
    traces: dict[str, np.ndarray],
    blackout_step: int,
) -> None:
    ax = add_axes_inches(fig, left, bottom - 0.03, width, height + 0.06, projection="3d")
    mean, components, explained = pca_basis([truth] + [traces[label] for label in METHODS], blackout_step)
    truth_coords = project(truth[blackout_step:], mean, components)
    coords = {label: project(traces[label][blackout_step:], mean, components) for label in METHODS}
    stacked = np.concatenate([truth_coords] + [coords[label] for label in METHODS], axis=0)
    clean_3d_axes(ax)
    set_equal_limits(ax, stacked)
    ax.view_init(elev=27.0, azim=-55.0)
    try:
        ax.set_box_aspect((1.0, 0.92, 0.85))
    except Exception:
        pass
    ax.plot(
        truth_coords[:, 0],
        truth_coords[:, 1],
        truth_coords[:, 2],
        color=METHOD_COLORS["Truth"],
        lw=METHOD_LW["Truth"],
        ls=(0, (4, 3)),
        alpha=0.86,
        solid_capstyle="round",
        label="Truth",
    )
    for label in METHODS:
        zorder = 6 if label in {"PCE", "APCE"} else 3
        ax.plot(
            coords[label][:, 0],
            coords[label][:, 1],
            coords[label][:, 2],
            color=METHOD_COLORS[label],
            lw=METHOD_LW[label],
            alpha=METHOD_ALPHA[label],
            solid_capstyle="round",
            zorder=zorder,
            label=label,
        )
        ax.scatter(
            coords[label][0, 0],
            coords[label][0, 1],
            coords[label][0, 2],
            s=18,
            facecolors="white",
            edgecolors=METHOD_COLORS[label],
            linewidths=0.9,
            depthshade=False,
        )
    ax.scatter(
        truth_coords[0, 0],
        truth_coords[0, 1],
        truth_coords[0, 2],
        s=20,
        facecolors="white",
        edgecolors=METHOD_COLORS["Truth"],
        linewidths=0.9,
        depthshade=False,
    )
    ax.set_title("PCA forecast", fontsize=FONT_TITLE, pad=6)
    ax.set_xlabel("PC1", fontsize=FONT_AXIS, labelpad=3)
    ax.set_ylabel("PC2", fontsize=FONT_AXIS, labelpad=3)
    ax.set_zlabel("")
    ax.set_zticklabels([])


def grouped_lead_series(rows: list[dict[str, str]], metric: str) -> dict[str, dict[float, list[float]]]:
    grouped: dict[str, dict[float, list[float]]] = {label: defaultdict(list) for label in METHODS}
    for row in rows:
        label = row["label"]
        if label not in grouped:
            continue
        lead = f(row, "lead_time")
        value = f(row, metric)
        if np.isfinite(lead) and np.isfinite(value):
            grouped[label][lead].append(value)
    return grouped


def draw_lead_metric(
    ax: plt.Axes,
    lead_rows: list[dict[str, str]],
    metric: str,
    title: str,
    ylabel: str,
    *,
    threshold: float | None = None,
) -> None:
    grouped = grouped_lead_series(lead_rows, metric)
    for label in METHODS:
        by_lead = grouped[label]
        leads = np.asarray(sorted(by_lead), dtype=float)
        values = [np.asarray(by_lead[lead], dtype=float) for lead in leads]
        mean = np.asarray([item.mean() for item in values])
        sem = np.asarray([item.std(ddof=1) / np.sqrt(item.size) if item.size > 1 else 0.0 for item in values])
        ax.plot(
            leads,
            mean,
            color=METHOD_COLORS[label],
            lw=2.30 if label in {"PCE", "APCE"} else 1.85,
            alpha=METHOD_ALPHA[label],
            marker=METHOD_MARKERS[label],
            markevery=18,
            ms=4.0,
            label=label,
        )
        ax.fill_between(leads, mean - 1.96 * sem, mean + 1.96 * sem, color=METHOD_COLORS[label], alpha=0.12, linewidth=0)
    if threshold is not None:
        ax.axhline(threshold, color="#202020", lw=1.0, ls=(0, (3, 3)), alpha=0.62)
    ax.set_xlim(0, 1)
    ax.set_title(title, fontsize=FONT_TITLE, pad=7)
    ax.set_xlabel("lead time", fontsize=FONT_AXIS, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=FONT_AXIS, labelpad=3)
    style_axis(ax)


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
        ("skill_horizon_time_020", "skill"),
    ]
    by_pair = {(row["method_label"], row["baseline_label"]): row for row in paired_rows}
    matrix = np.full((len(row_order), len(col_defs)), np.nan)
    annotations = [["" for _ in col_defs] for _ in row_order]
    for i, pair in enumerate(row_order):
        row = by_pair.get(pair)
        if row is None:
            continue
        total = int(float(row["paired_seed_count"]))
        for j, (metric, _label) in enumerate(col_defs):
            wins = int(float(row[f"{metric}_win_count"]))
            gain = f(row, f"{metric}_gain_mean")
            matrix[i, j] = wins / max(total, 1)
            sign = "+" if gain >= 0 else ""
            annotations[i][j] = f"{wins}/{total}\n{sign}{gain:.2g}"
    cmap = mpl.colors.LinearSegmentedColormap.from_list("l96_win_fraction", ["#F5F5F3", "#D7EDD9", "#6CBD74", "#23783E"])
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    ax.set_title("paired gain", fontsize=FONT_TITLE, pad=7)
    ax.set_xticks(np.arange(len(col_defs)))
    ax.set_xticklabels([label for _metric, label in col_defs], fontsize=FONT_TICK)
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels([f"{m}/{b}" for m, b in row_order], fontsize=FONT_TICK)
    ax.tick_params(length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, annotations[i][j], ha="center", va="center", fontsize=FONT_TICK, color="#152017")


def save_all(fig: plt.Figure, output_base: Path) -> dict[str, str]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    for ext, kwargs in {
        "png": {"dpi": 650},
        "pdf": {},
        "svg": {},
        "tiff": {"dpi": 600},
    }.items():
        path = output_base.with_suffix(f".{ext}")
        fig.savefig(path, facecolor="white", **kwargs)
        saved[ext] = str(path)
    return saved


def main() -> None:
    configure_matplotlib()
    seed = 2026080600
    truth, times, traces, obs_indices, blackout_step, config = load_all(seed)
    run_rows = read_csv(DATA_ROOT / "source_data" / "lorenz96_1024_blackout_run_source_data.csv")
    lead_rows = read_csv(DATA_ROOT / "source_data" / "lorenz96_1024_blackout_lead_time_source_data.csv")
    paired_rows = read_csv(DATA_ROOT / "source_data" / "lorenz96_1024_blackout_paired_gains.csv")

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")

    # g: four fixed ring-state traces.
    trace_left = PANEL_LEFTS[0]
    trace_width = PANEL_LEFTS[2] + PANEL_W - PANEL_LEFTS[0] - 0.24
    draw_state_traces(fig, trace_left, ROW_BOTTOM, trace_width, ROW_H, truth, times, traces, blackout_step)

    # h: 3D PCA phase portrait for the blackout forecast window.
    pca_left = PANEL_LEFTS[3] - 0.42
    pca_width = 3.08
    draw_pca(fig, pca_left, ROW_BOTTOM, pca_width, ROW_H, truth, traces, blackout_step)

    # i-k: quantitative panels keep the accepted KSE right-column geometry.
    ax_nrmse = add_axes_inches(fig, QUANT_LEFTS[0], ROW_BOTTOM, C_PANEL_W, ROW_H)
    ax_crps = add_axes_inches(fig, QUANT_LEFTS[1], ROW_BOTTOM, C_PANEL_W, ROW_H)
    ax_gain = add_axes_inches(fig, QUANT_LEFTS[2], ROW_BOTTOM, C_PANEL_W, ROW_H)
    draw_lead_metric(ax_nrmse, lead_rows, "lead_nrmse", "nRMSE", "nRMSE", threshold=0.20)
    draw_lead_metric(ax_crps, lead_rows, "lead_crps", "CRPS", "CRPS")
    draw_gain_heatmap(ax_gain, paired_rows)

    handles = [Line2D([0], [0], color=METHOD_COLORS["Truth"], lw=METHOD_LW["Truth"], ls=(0, (4, 3)), label="Truth")]
    handles.extend(
        [
            Line2D([0], [0], color=METHOD_COLORS[label], lw=METHOD_LW[label], marker=METHOD_MARKERS[label], markersize=4.5, label=label)
            for label in METHODS
        ]
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.995),
        ncol=5,
        fontsize=FONT_LEGEND,
        handlelength=1.65,
        columnspacing=1.15,
        borderaxespad=0.0,
    )

    add_panel_label(fig, trace_left, ROW_TOP, "g")
    add_panel_label(fig, pca_left, ROW_TOP, "h")
    add_panel_label(fig, QUANT_LEFTS[0], ROW_TOP, "i")
    add_panel_label(fig, QUANT_LEFTS[1], ROW_TOP, "j")
    add_panel_label(fig, QUANT_LEFTS[2], ROW_TOP, "k")

    saved = save_all(fig, OUTPUT_BASE)
    plt.close(fig)

    method_metrics = {}
    for row in run_rows:
        if int(float(row["seed"])) != seed:
            continue
        method_metrics[row["label"]] = {
            "forecast_nrmse": f(row, "forecast_nrmse"),
            "forecast_crps": f(row, "forecast_crps"),
            "blackout_alpha_absolute_error": f(row, "blackout_alpha_absolute_error"),
            "skill_horizon_time_020": f(row, "skill_horizon_time_020"),
        }
    qa = {
        "core_conclusion": "For the Lorenz-96-1024 blackout forecast smoke test, PCE/APCE track the leading phase-space trajectory and maintain lower forecast nRMSE/CRPS than Aug-EnKF and BMA.",
        "backend": "Python/matplotlib",
        "figure_archetype": "asymmetric mixed-modality row",
        "figure4_width_inches": FIG_W,
        "figure_height_inches": FIG_H,
        "font_rules": {
            "panel_label": FONT_PANEL,
            "case_title": FONT_TITLE,
            "legend": FONT_LEGEND,
            "axis_label": FONT_AXIS,
            "tick": FONT_TICK,
        },
        "source_remote_root": REMOTE_ROOT,
        "source_local_root": str(DATA_ROOT),
        "representative_seed": seed,
        "state_dim": int(config["state_dim"]),
        "observed_points": int(config["observed_points"]),
        "obs_interval": int(config["obs_interval"]),
        "blackout_start_step": int(blackout_step),
        "blackout_time": float(times[blackout_step]),
        "final_time": float(times[-1]),
        "observation_index_count": int(obs_indices.size),
        "method_metrics_representative_seed": method_metrics,
        "exports": saved,
    }
    qa_path = OUTPUT_BASE.with_suffix(".qa.json")
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**saved, "qa": str(qa_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
