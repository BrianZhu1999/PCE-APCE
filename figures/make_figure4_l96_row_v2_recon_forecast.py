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
OUTPUT_BASE = HERE / "figure4_l96_row_v2_recon_forecast"
FIG_H2 = 3.72
ROW_BOTTOM2 = 0.62
ROW_TOP2 = 2.84
ROW_H2 = ROW_TOP2 - ROW_BOTTOM2


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


def load_recon(seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    shared = load_npz(RECON_ROOT / "shared_assets" / f"lorenz96_1024_shared_seed_{seed}.npz")
    truth = np.asarray(shared["truth"], dtype=float)
    traces: dict[str, np.ndarray] = {}
    times: np.ndarray | None = None
    for label in base.METHODS:
        trace = load_npz(recon_trace_path(base.METHOD_KEY[label], seed))
        traces[label] = np.asarray(trace["mean_states"], dtype=float)
        if times is None:
            times = np.asarray(trace["times"], dtype=float)
    if times is None:
        raise RuntimeError("Missing reconstruction times.")
    return truth, times, traces


def draw_reconstruction_traces(
    fig: plt.Figure,
    left: float,
    bottom: float,
    width: float,
    height: float,
    truth: np.ndarray,
    times: np.ndarray,
    traces: dict[str, np.ndarray],
) -> None:
    state_indices = [0, 256, 512, 768]
    gap = 0.06
    sub_h = (height - gap * (len(state_indices) - 1)) / len(state_indices)
    for row, idx in enumerate(state_indices):
        y = bottom + (len(state_indices) - 1 - row) * (sub_h + gap)
        ax = base.add_axes_inches(fig, left, y, width, sub_h)
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
        if row < len(state_indices) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("time", fontsize=base.FONT_AXIS, labelpad=1)
        if row == 0:
            ax.set_title("reconstruction traces", fontsize=base.FONT_TITLE, pad=7)
        if row == 1:
            ax.set_ylabel("state value", fontsize=base.FONT_AXIS, labelpad=5)


def draw_recon_boxes(fig: plt.Figure, left: float, bottom: float, width: float, height: float, rows: list[dict[str, str]]) -> None:
    gap = 0.18
    sub_w = (width - gap) / 2
    ax_n = base.add_axes_inches(fig, left, bottom, sub_w, height)
    ax_c = base.add_axes_inches(fig, left + sub_w + gap, bottom, sub_w, height)
    for ax, metric, title, ylim in [
        (ax_n, "nrmse", "recon. nRMSE", (0.08, 0.16)),
        (ax_c, "crps", "recon. CRPS", (0.16, 0.27)),
    ]:
        data = []
        for label in base.METHODS:
            values = [f(row, metric) for row in rows if row["method"] == base.METHOD_KEY[label]]
            data.append(np.asarray(values, dtype=float))
        pos = np.arange(len(base.METHODS))
        bp = ax.boxplot(
            data,
            positions=pos,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#202020", "linewidth": 1.25},
            whiskerprops={"color": "#666666", "linewidth": 0.9},
            capprops={"color": "#666666", "linewidth": 0.9},
        )
        for patch, label in zip(bp["boxes"], base.METHODS):
            patch.set_facecolor(base.METHOD_COLORS[label])
            patch.set_alpha(0.32 if label in {"PCE", "APCE"} else 0.24)
            patch.set_edgecolor(base.METHOD_COLORS[label])
            patch.set_linewidth(1.1)
        for x0, label, arr in zip(pos, base.METHODS, data):
            jitter = np.linspace(-0.09, 0.09, arr.size) if arr.size > 1 else np.zeros_like(arr)
            ax.scatter(
                np.full(arr.shape, x0) + jitter,
                arr,
                s=16,
                color=base.METHOD_COLORS[label],
                edgecolor="white",
                linewidth=0.4,
                zorder=4,
            )
        ax.set_title(title, fontsize=base.FONT_TITLE, pad=7)
        ax.set_xticks(pos)
        ax.set_xticklabels(["Aug", "BMA", "PCE", "APCE"], rotation=38, ha="right", fontsize=base.FONT_TICK)
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(base.clean_number))
        ax.tick_params(axis="y", labelsize=base.FONT_TICK)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def add_label(fig: plt.Figure, left: float, top: float, label: str, dx: float = -0.16) -> None:
    fig.text(
        (left + dx) / base.FIG_W,
        (top + 0.43) / base.FIG_H,
        label,
        ha="left",
        va="top",
        fontsize=base.FONT_PANEL,
        fontweight="bold",
        color="#111111",
    )


def main() -> None:
    base.configure_matplotlib()
    base.FIG_H = FIG_H2
    seed = 2026080600

    recon_truth, recon_times, recon_traces = load_recon(seed)
    forecast_truth, forecast_times, forecast_traces, _obs, blackout_step, config = base.load_all(seed)
    recon_rows = read_csv(RECON_ROOT / "source_data" / "lorenz96_1024_run_source_data.csv")
    lead_rows = read_csv(FORECAST_ROOT / "source_data" / "lorenz96_1024_blackout_lead_time_source_data.csv")

    fig = plt.figure(figsize=(base.FIG_W, FIG_H2), facecolor="white")

    # g: reconstruction traces.
    g_left = 0.28
    g_width = 4.02
    draw_reconstruction_traces(fig, g_left, ROW_BOTTOM2, g_width, ROW_H2, recon_truth, recon_times, recon_traces)

    # h: reconstruction quantitative distribution.
    h_left = 4.72
    h_width = 2.48
    draw_recon_boxes(fig, h_left, ROW_BOTTOM2, h_width, ROW_H2, recon_rows)

    # i: forecast phase portrait.
    i_left = 7.58
    i_width = 2.46
    base.draw_pca(fig, i_left, ROW_BOTTOM2, i_width, ROW_H2, forecast_truth, forecast_traces, blackout_step)

    # j/k: forecast curves.
    ax_nrmse = base.add_axes_inches(fig, 10.66, ROW_BOTTOM2, 2.55, ROW_H2)
    ax_crps = base.add_axes_inches(fig, 13.80, ROW_BOTTOM2, 2.55, ROW_H2)
    base.draw_lead_metric(ax_nrmse, lead_rows, "lead_nrmse", "forecast nRMSE", "nRMSE", threshold=0.20)
    base.draw_lead_metric(ax_crps, lead_rows, "lead_crps", "forecast CRPS", "CRPS")

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

    top = ROW_TOP2
    add_label(fig, g_left, top, "g", dx=-0.16)
    add_label(fig, h_left, top, "h", dx=-0.34)
    add_label(fig, i_left, top, "i", dx=-0.22)
    add_label(fig, 10.66, top, "j", dx=-0.22)
    add_label(fig, 13.80, top, "k", dx=-0.22)

    saved = base.save_all(fig, OUTPUT_BASE)
    plt.close(fig)

    qa = {
        "core_conclusion": "Lorenz-96-1024 contributes both reconstruction evidence and blackout forecast evidence: PCE/APCE improve reconstruction nRMSE/CRPS and preserve lower forecast error over the no-observation window.",
        "backend": "Python/matplotlib",
        "figure4_width_inches": base.FIG_W,
        "figure_height_inches": FIG_H2,
        "font_rules": {
            "panel_label": base.FONT_PANEL,
            "case_title": base.FONT_TITLE,
            "legend": base.FONT_LEGEND,
            "axis_label": base.FONT_AXIS,
            "tick": base.FONT_TICK,
        },
        "representative_seed": seed,
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
