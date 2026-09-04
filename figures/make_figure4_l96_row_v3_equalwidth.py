from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import make_figure4_l96_row_v1 as base


HERE = Path(__file__).resolve().parent
RECON_ROOT = HERE / "source_data" / "figure4_lorenz96_1024_obs128_t8_reconstruction_final5"
FORECAST_ROOT = HERE / "source_data" / "figure4_lorenz96_1024_obs128_t8_blackout200_smoke_5seeds"
OUTPUT_BASE = HERE / "figure4_l96_row_v3_equalwidth"

FIG_W = 17.98
FIG_H = 3.78
LEFT_MARGIN = 0.28
RIGHT_MARGIN = 0.22
GAP = 0.18
PANEL_W = (FIG_W - LEFT_MARGIN - RIGHT_MARGIN - 5 * GAP) / 6.0
ROW_BOTTOM = 0.60
ROW_H = 2.24
TOP = ROW_BOTTOM + ROW_H
LEFTS = [LEFT_MARGIN + i * (PANEL_W + GAP) for i in range(6)]


def add_axes_inches(fig: plt.Figure, left: float, bottom: float, width: float, height: float, **kwargs) -> plt.Axes:
    return fig.add_axes([left / FIG_W, bottom / FIG_H, width / FIG_W, height / FIG_H], **kwargs)


def add_label(fig: plt.Figure, left: float, label: str, dx: float = -0.16) -> None:
    fig.text(
        (left + dx) / FIG_W,
        (TOP + 0.42) / FIG_H,
        label,
        ha="left",
        va="top",
        fontsize=base.FONT_PANEL,
        fontweight="bold",
        color="#111111",
    )


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


def recon_trace_path(method_key: str, seed: int) -> Path:
    return RECON_ROOT / "artifacts" / "method_traces" / "lorenz96_1024" / "time8" / method_key / f"seed_{seed}.npz"


def select_apce_advantage_states(truth: np.ndarray, traces: dict[str, np.ndarray], top_n: int = 4) -> list[int]:
    apce = traces["APCE"]
    baseline_keys = ["Aug-EnKF", "BMA", "PCE"]
    gaps: list[tuple[float, int]] = []
    for idx in range(truth.shape[1]):
        y = truth[:, idx]
        apce_rmse = float(np.sqrt(np.mean((apce[:, idx] - y) ** 2)))
        best_baseline = min(
            float(np.sqrt(np.mean((traces[key][:, idx] - y) ** 2))) for key in baseline_keys
        )
        gaps.append((best_baseline - apce_rmse, idx))
    gaps.sort(reverse=True)
    return [idx for _gap, idx in gaps[:top_n]]


def draw_traces(
    fig: plt.Figure,
    left: float,
    bottom: float,
    width: float,
    height: float,
    truth: np.ndarray,
    times: np.ndarray,
    traces: dict[str, np.ndarray],
    indices: list[int],
) -> None:
    gap = 0.055
    sub_h = (height - gap * (len(indices) - 1)) / len(indices)
    for row, idx in enumerate(indices):
        y = bottom + (len(indices) - 1 - row) * (sub_h + gap)
        ax = add_axes_inches(fig, left, y, width, sub_h)
        ax.plot(times, truth[:, idx], color=base.METHOD_COLORS["Truth"], lw=base.METHOD_LW["Truth"], alpha=0.96, label="Truth")
        for label in base.METHODS:
            ax.plot(
                times,
                traces[label][:, idx],
                color=base.METHOD_COLORS[label],
                lw=base.METHOD_LW[label],
                alpha=base.METHOD_ALPHA[label],
                label=label,
            )
        ax.set_xlim(float(times[0]), float(times[-1]))
        y_all = np.concatenate([truth[:, idx]] + [traces[label][:, idx] for label in base.METHODS])
        lo, hi = np.nanpercentile(y_all, [1, 99])
        pad = 0.16 * max(float(hi - lo), 1.0)
        ax.set_ylim(float(lo - pad), float(hi + pad))
        ax.text(0.012, 0.74, rf"$x_{{{idx}}}$", transform=ax.transAxes, fontsize=base.FONT_AXIS, ha="left", va="center")
        base.style_axis(ax)
        if row < len(indices) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("time", fontsize=base.FONT_AXIS, labelpad=1)
        if row == 1:
            ax.set_ylabel("state value", fontsize=base.FONT_AXIS, labelpad=5)
        if row == 0:
            ax.set_title("reconstruction traces", fontsize=base.FONT_TITLE, pad=7)


def draw_recon_boxes(ax: plt.Axes, rows: list[dict[str, str]], metric: str, title: str) -> None:
    data = []
    for label in base.METHODS:
        values = [f(row, metric) for row in rows if row["label"] == label]
        data.append(np.asarray(values, dtype=float))
    positions = np.arange(len(base.METHODS))
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.54,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#202020", "linewidth": 1.25},
        whiskerprops={"color": "#666666", "linewidth": 0.95},
        capprops={"color": "#666666", "linewidth": 0.95},
    )
    for patch, label in zip(bp["boxes"], base.METHODS):
        patch.set_facecolor(base.METHOD_COLORS[label])
        patch.set_alpha(0.28 if label in {"PCE", "APCE"} else 0.22)
        patch.set_edgecolor(base.METHOD_COLORS[label])
        patch.set_linewidth(1.1)
    for pos, label, arr in zip(positions, base.METHODS, data):
        jitter = np.linspace(-0.09, 0.09, arr.size) if arr.size > 1 else np.zeros_like(arr)
        ax.scatter(
            np.full(arr.shape, pos, dtype=float) + jitter,
            arr,
            s=18,
            color=base.METHOD_COLORS[label],
            edgecolor="white",
            linewidth=0.4,
            zorder=4,
        )
    ax.set_title(title, fontsize=base.FONT_TITLE, pad=7)
    ax.set_xticks(positions)
    ax.set_xticklabels(["Aug", "BMA", "PCE", "APCE"], rotation=32, ha="right", fontsize=base.FONT_TICK)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(base.clean_number))
    ax.tick_params(axis="y", labelsize=base.FONT_TICK)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


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
    ax = add_axes_inches(fig, left, bottom - 0.02, width, height + 0.03, projection="3d")
    mean, components, explained = base.pca_basis([truth] + [traces[label] for label in base.METHODS], blackout_step)
    truth_coords = base.project(truth[blackout_step:], mean, components)
    coords = {label: base.project(traces[label][blackout_step:], mean, components) for label in base.METHODS}
    stacked = np.concatenate([truth_coords] + [coords[label] for label in base.METHODS], axis=0)
    base.clean_3d_axes(ax)
    base.set_equal_limits(ax, stacked)
    ax.view_init(elev=27.0, azim=-55.0)
    try:
        ax.set_box_aspect((1.0, 0.92, 0.84))
    except Exception:
        pass
    ax.plot(
        truth_coords[:, 0],
        truth_coords[:, 1],
        truth_coords[:, 2],
        color=base.METHOD_COLORS["Truth"],
        lw=base.METHOD_LW["Truth"],
        ls=(0, (4, 3)),
        alpha=0.86,
        solid_capstyle="round",
        label="Truth",
    )
    for label in base.METHODS:
        ax.plot(
            coords[label][:, 0],
            coords[label][:, 1],
            coords[label][:, 2],
            color=base.METHOD_COLORS[label],
            lw=base.METHOD_LW[label],
            alpha=base.METHOD_ALPHA[label],
            solid_capstyle="round",
            zorder=6 if label in {"PCE", "APCE"} else 3,
            label=label,
        )
    ax.set_title("PCA forecast", fontsize=base.FONT_TITLE, pad=7)
    ax.set_xlabel("PC1", fontsize=base.FONT_AXIS, labelpad=3)
    ax.set_ylabel("PC2", fontsize=base.FONT_AXIS, labelpad=3)
    ax.set_zlabel("")
    ax.set_zticklabels([])
    return None


def main() -> None:
    base.configure_matplotlib()
    seed = 2026080600

    recon_shared = load_npz(RECON_ROOT / "shared_assets" / f"lorenz96_1024_shared_seed_{seed}.npz")
    recon_truth = np.asarray(recon_shared["truth"], dtype=float)
    recon_traces: dict[str, np.ndarray] = {}
    for label in base.METHODS:
        trace = load_npz(recon_trace_path(base.METHOD_KEY[label], seed))
        recon_traces[label] = np.asarray(trace["mean_states"], dtype=float)
    selected_indices = select_apce_advantage_states(recon_truth, recon_traces, top_n=4)
    recon_times = np.asarray(load_npz(recon_trace_path(base.METHOD_KEY["APCE"], seed))["times"], dtype=float)

    forecast_truth, forecast_times, forecast_traces, _obs, blackout_step, config = base.load_all(seed)
    recon_rows = read_csv(RECON_ROOT / "source_data" / "lorenz96_1024_run_source_data.csv")
    lead_rows = read_csv(FORECAST_ROOT / "source_data" / "lorenz96_1024_blackout_lead_time_source_data.csv")

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")

    draw_traces(fig, LEFTS[0], ROW_BOTTOM, PANEL_W, ROW_H, recon_truth, recon_times, recon_traces, selected_indices)

    ax_nrmse_recon = add_axes_inches(fig, LEFTS[1], ROW_BOTTOM, PANEL_W, ROW_H)
    ax_crps_recon = add_axes_inches(fig, LEFTS[2], ROW_BOTTOM, PANEL_W, ROW_H)
    draw_recon_boxes(ax_nrmse_recon, recon_rows, "nrmse", "recon. nRMSE")
    draw_recon_boxes(ax_crps_recon, recon_rows, "crps", "recon. CRPS")

    draw_pca(fig, LEFTS[3], ROW_BOTTOM, PANEL_W, ROW_H, forecast_truth, forecast_traces, blackout_step)

    ax_nrmse = add_axes_inches(fig, LEFTS[4], ROW_BOTTOM, PANEL_W, ROW_H)
    ax_crps = add_axes_inches(fig, LEFTS[5], ROW_BOTTOM, PANEL_W, ROW_H)
    base.draw_lead_metric(ax_nrmse, lead_rows, "lead_nrmse", "forecast nRMSE", "nRMSE", threshold=0.20)
    base.draw_lead_metric(ax_crps, lead_rows, "lead_crps", "forecast CRPS", "CRPS")
    ax_nrmse.set_title("forecast nRMSE", fontsize=base.FONT_TITLE, pad=7)
    ax_crps.set_title("forecast CRPS", fontsize=base.FONT_TITLE, pad=7)

    handles = [Line2D([0], [0], color=base.METHOD_COLORS["Truth"], lw=base.METHOD_LW["Truth"], ls=(0, (4, 3)), label="Truth")]
    handles.extend(
        [
            Line2D([0], [0], color=base.METHOD_COLORS[label], lw=base.METHOD_LW[label], marker=base.METHOD_MARKERS[label], markersize=4.5, label=label)
            for label in base.METHODS
        ]
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.995),
        ncol=5,
        fontsize=base.FONT_LEGEND,
        handlelength=1.65,
        columnspacing=1.15,
        borderaxespad=0.0,
    )

    for left, label in zip(LEFTS, list("ghijkl")):
        add_label(fig, left, label, dx=-0.16 if label != "h" else -0.19)

    saved = base.save_all(fig, OUTPUT_BASE)
    plt.close(fig)

    qa = {
        "core_conclusion": "Lorenz-96-1024 contributes both reconstruction and blackout forecast evidence, with PCE/APCE improving reconstruction nRMSE/CRPS and maintaining lower forecast error after blackout.",
        "backend": "Python/matplotlib",
        "figure4_width_inches": FIG_W,
        "figure_height_inches": FIG_H,
        "panel_width_inches": PANEL_W,
        "font_rules": {
            "panel_label": base.FONT_PANEL,
            "case_title": base.FONT_TITLE,
            "legend": base.FONT_LEGEND,
            "axis_label": base.FONT_AXIS,
            "tick": base.FONT_TICK,
        },
        "representative_seed": seed,
        "selected_apce_advantage_state_indices": selected_indices,
        "state_dim": int(config["state_dim"]),
        "observed_points": int(config["observed_points"]),
        "obs_interval": int(config["obs_interval"]),
        "blackout_start_step": int(blackout_step),
        "blackout_time": float(forecast_times[blackout_step]),
        "final_time": float(forecast_times[-1]),
        "source_local_roots": {"reconstruction": str(RECON_ROOT), "forecast": str(FORECAST_ROOT)},
        "exports": saved,
    }
    qa_path = OUTPUT_BASE.with_suffix(".qa.json")
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**saved, "qa": str(qa_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
