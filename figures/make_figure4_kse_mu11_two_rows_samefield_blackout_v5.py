from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import PowerNorm
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import FuncFormatter

import make_figure4_kse_mu11_firstrow_v1 as first


DEFAULT_FIG4_ROOT = Path(
    "<HILDA_RESULTS_ROOT>/results/figure4_kse_nmi32_t2_core4_formal_50seeds_20260814_4gpu"
)
DEFAULT_BLACKOUT_ROOT = Path(
    "<HILDA_RESULTS_ROOT>/results/figure5_kse_blackout32_t2_step40_100seeds_20260814_4gpu"
)
DEFAULT_OUTPUT = Path(
    "<HILDA_RESULTS_ROOT>/results/figure5_kse_blackout32_t2_step40_100seeds_20260814_4gpu/plots/"
    "figure4_kse_mu11_two_rows_samefield_blackout_v5"
)

METHODS = ["aug_enkf", "bma_static", "pce", "apce"]
LABELS = first.LABELS
COLORS = first.COLORS
BOX_COLORS = first.BOX_COLORS

FONT_PANEL = 22
FONT_TITLE = 14
FONT_AXIS = 13
FONT_TICK = 11


def clean_number(x: float, _pos: int | None = None) -> str:
    return f"{x:g}"


def save_all(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=650, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_base.with_suffix(".tiff"), dpi=650, bbox_inches="tight", pad_inches=0.03)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    arr = np.load(path, allow_pickle=True)
    return {key: np.asarray(arr[key]) for key in arr.files}


def load_run_json(result_root: Path, method: str, seed_index: int, sample_index: int) -> dict:
    run_id = f"kse_nmi32x_t2_blackout40_{method}_seed{seed_index:02d}_sample{sample_index:02d}"
    return json.loads((result_root / "runs" / f"{run_id}.json").read_text(encoding="utf-8"))


def periodic_sparse_interpolation(sensor_indices: np.ndarray, values: np.ndarray, nx: int) -> np.ndarray:
    indices = np.asarray(sensor_indices, dtype=float)
    values = np.asarray(values, dtype=float)
    order = np.argsort(indices)
    indices = indices[order]
    values = values[order]
    x_ext = np.concatenate([indices, [indices[0] + nx]])
    y_ext = np.concatenate([values, [values[0]]])
    return np.interp(np.arange(nx, dtype=float), x_ext, y_ext)


def make_initial_frame_strip(trace: dict[str, np.ndarray], *, strip_fraction: float = 0.090) -> np.ndarray:
    truth = np.asarray(trace["truth"], dtype=float)
    total_steps = truth.shape[0]
    nx = truth.shape[1]
    steps = np.asarray(trace["assimilation_observation_steps"], dtype=int)
    obs = np.asarray(trace["assimilation_observations"], dtype=float)
    sensor_indices = np.asarray(trace["sensor_indices"], dtype=int)
    blackout_step = int(np.asarray(trace["blackout_start_step"]).item())
    window_steps = np.arange(blackout_step + 1, dtype=float)
    interpolated_obs = np.vstack(
        [periodic_sparse_interpolation(sensor_indices, obs_i, nx) for obs_i in obs]
    )
    order = np.argsort(steps)
    steps_sorted = steps[order].astype(float)
    interpolated_sorted = interpolated_obs[order]
    full_window = np.empty((blackout_step + 1, nx), dtype=float)
    for ix in range(nx):
        full_window[:, ix] = np.interp(window_steps, steps_sorted, interpolated_sorted[:, ix])
    canvas = np.full((nx, total_steps), np.nan, dtype=float)
    canvas[:, : blackout_step + 1] = full_window.T
    return canvas


def relative_error(reconstruction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    scale = float(np.sqrt(np.nanmean(np.square(truth))))
    return np.abs(reconstruction - truth) / max(scale, 1.0e-12)


def add_horizontal_colorbar(
    fig: plt.Figure,
    ax_list: list[plt.Axes],
    image: mpl.image.AxesImage,
    *,
    label: str,
    ticks: list[float],
    ticklabels: list[str],
    fig_h: float,
    offset: float = 0.115,
    reverse: bool = True,
) -> None:
    left = min(ax.get_position().x0 for ax in ax_list)
    right = max(ax.get_position().x1 for ax in ax_list)
    width = right - left
    bottom = min(ax.get_position().y0 for ax in ax_list) - 0.058
    cax = fig.add_axes([left, bottom, width, 0.026])
    cb = fig.colorbar(image, cax=cax, orientation="horizontal")
    cb.set_ticks(ticks)
    cb.set_ticklabels(ticklabels)
    if reverse:
        cb.ax.invert_xaxis()
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=FONT_TICK, length=0, pad=6.0)
    label_y = -0.72 if "hat u" in label else -1.85
    cax.text(0.5, label_y, label, transform=cax.transAxes, ha="center", va="top", fontsize=FONT_TICK, fontstyle="italic")


def draw_image_panel(
    ax: plt.Axes,
    image: np.ndarray,
    title: str,
    cmap,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    norm=None,
) -> mpl.image.AxesImage:
    local_cmap = cmap.copy() if hasattr(cmap, "copy") else cmap
    if hasattr(local_cmap, "set_bad"):
        local_cmap.set_bad("#EEEEEE")
    masked = np.ma.masked_invalid(image)
    if norm is None:
        im = ax.imshow(masked, origin="lower", aspect="auto", interpolation="nearest", cmap=local_cmap, vmin=vmin, vmax=vmax)
    else:
        im = ax.imshow(masked, origin="lower", aspect="auto", interpolation="nearest", cmap=local_cmap, norm=norm)
    ax.set_title(title, fontsize=FONT_TITLE, pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def draw_first_row(
    fig: plt.Figure,
    *,
    fig_w: float,
    fig_h: float,
    row_bottom: float,
    panel_lefts: list[float],
    quant_lefts: list[float],
    panel_w: float,
    panel_h: float,
    c_panel_w: float,
    c_panel_h: float,
    result_root: Path,
    blackout_root: Path,
    target_mu: float,
    seed_index: int,
    sample_index: int,
) -> dict:
    run = load_run_json(blackout_root, "apce", seed_index, sample_index)
    trace = load_npz(Path(run["trace_npz"]))
    summary = pd.read_csv(result_root / "source_data" / "run_source_data.csv", encoding="utf-8-sig")
    truth = np.asarray(trace["truth"], dtype=float)
    apce = np.asarray(trace["mean_states"], dtype=float)
    sparse = make_initial_frame_strip(trace)
    relerr = first.relative_error(apce, truth)
    same_seed_rows = summary[summary["seed_index"] == int(seed_index)]
    method_fields: dict[str, np.ndarray] = {}
    blackout_runs = {method: load_run_json(blackout_root, method, seed_index, sample_index) for method in METHODS}
    for method in METHODS:
        method_trace = load_npz(Path(blackout_runs[method]["trace_npz"]))
        method_fields[method] = np.asarray(method_trace["mean_states"], dtype=float)

    vmax = 3.0
    rel_vmax = 0.4
    cmap_field = first.field_cmap()
    cmap_error = first.error_cmap()

    panel_bottom = row_bottom + (c_panel_h - panel_h)

    def add_panel(left_in: float) -> plt.Axes:
        return fig.add_axes([left_in / fig_w, panel_bottom / fig_h, panel_w / fig_w, panel_h / fig_h])

    def add_c_panel(left_in: float) -> plt.Axes:
        return fig.add_axes([left_in / fig_w, row_bottom / fig_h, c_panel_w / fig_w, c_panel_h / fig_h])

    ax_sparse = add_panel(panel_lefts[0])
    ax_truth = add_panel(panel_lefts[1])
    ax_apce = add_panel(panel_lefts[2])
    ax_error = add_panel(panel_lefts[3])
    ax_nrmse = add_c_panel(quant_lefts[0])
    ax_crps = add_c_panel(quant_lefts[1])
    ax_stat = add_c_panel(quant_lefts[2])

    image_handles = {}
    image_handles["sparse"] = draw_image_panel(ax_sparse, sparse, "Observations", cmap_field, vmin=-vmax, vmax=vmax)
    image_handles["truth"] = draw_image_panel(ax_truth, truth.T, "Ref. ($\\mu=1.1$)", cmap_field, vmin=-vmax, vmax=vmax)
    image_handles["apce"] = draw_image_panel(ax_apce, apce.T, "APCE", cmap_field, vmin=-vmax, vmax=vmax)
    image_handles["error"] = draw_image_panel(
        ax_error,
        relerr.T,
        "Relative error",
        cmap_error,
        norm=PowerNorm(gamma=0.58, vmin=0.0, vmax=rel_vmax),
    )
    ax_sparse.set_xlabel("$t$", fontsize=FONT_AXIS, labelpad=2)
    ax_sparse.set_ylabel("$x$", fontsize=FONT_AXIS, labelpad=2, rotation=0)
    ax_sparse.yaxis.set_label_coords(-0.08, 0.5)

    arrow = FancyArrowPatch(
        ((panel_lefts[0] + panel_w + 0.08) / fig_w, (panel_bottom + panel_h * 0.50) / fig_h),
        ((panel_lefts[1] - 0.12) / fig_w, (panel_bottom + panel_h * 0.50) / fig_h),
        transform=fig.transFigure,
        arrowstyle="simple",
        mutation_scale=47,
        fc="#5DB8C4",
        ec="#5DB8C4",
        lw=0.0,
        alpha=0.95,
    )
    fig.add_artist(arrow)

    first.draw_metric_boxplot(ax_nrmse, summary, target_mu, "nrmse", "nRMSE", ymax=0.5)
    first.draw_metric_boxplot(ax_crps, summary, target_mu, "crps", "CRPS", ymax=0.4)
    first.draw_statistical_feature(ax_stat, truth, method_fields, downsampling_factor=32)
    ax_stat.set_title("Statistical feature", fontsize=FONT_TITLE, pad=8)
    ax_stat.set_ylim(-0.6, 1.02)
    add_horizontal_colorbar(fig, [ax_truth, ax_apce], image_handles["truth"], label="$u$", ticks=[-vmax, vmax], ticklabels=["-3", "3"], fig_h=fig_h)
    add_horizontal_colorbar(
        fig,
        [ax_error],
        image_handles["error"],
        label="$|u-\\hat u|/\\mathrm{rms}(u)$",
        ticks=[0.0, rel_vmax],
        ticklabels=["0", "0.4"],
        fig_h=fig_h,
    )
    fig.text((panel_lefts[0] - 0.14) / fig_w, (row_bottom + c_panel_h + 0.47) / fig_h, "a", fontsize=FONT_PANEL, fontweight="bold", ha="left", va="top")
    fig.text((panel_lefts[1] - 0.14) / fig_w, (row_bottom + c_panel_h + 0.47) / fig_h, "b", fontsize=FONT_PANEL, fontweight="bold", ha="left", va="top")
    fig.text((panel_lefts[4] - 0.14) / fig_w, (row_bottom + c_panel_h + 0.47) / fig_h, "c", fontsize=FONT_PANEL, fontweight="bold", ha="left", va="top")
    return {
        "first_row_run_id": str(run["run_id"]),
        "first_row_trace": str(run["trace_npz"]),
        "first_row_seed": int(seed_index),
        "first_row_sample": int(sample_index),
        "first_row_source": "same blackout trace as second-row field; used to align reconstruction/forecast on the same KSE field",
        "first_row_apce_nrmse": float(run.get("assimilation_nrmse", run.get("forecast_nrmse", np.nan))),
    }


def draw_metric_curve(ax: plt.Axes, runs: dict[str, dict], key: str, title: str, ylabel: str) -> None:
    styles = {
        "aug_enkf": ("Aug-EnKF", COLORS["aug_enkf"], 1.80, 0.88, 2),
        "bma_static": ("BMA", COLORS["bma_static"], 1.85, 0.95, 3),
        "pce": ("PCE", COLORS["pce"], 2.35, 1.00, 4),
        "apce": ("APCE", COLORS["apce"], 2.55, 1.00, 5),
    }
    for method in ["aug_enkf", "bma_static", "pce", "apce"]:
        run = runs[method]
        x = np.asarray(run["forecast_steps"], dtype=float)
        y = np.asarray(run[key], dtype=float)
        label, color, lw, alpha, zorder = styles[method]
        ax.plot(x, y, color=color, linewidth=lw, alpha=alpha, label=label, zorder=zorder)
    ax.set_xlim(40, 99)
    ax.set_xlabel("Time step", fontsize=FONT_AXIS, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=FONT_AXIS, labelpad=3)
    ax.set_title(title, fontsize=FONT_TITLE, pad=8)
    ax.xaxis.set_major_formatter(FuncFormatter(clean_number))
    ax.yaxis.set_major_formatter(FuncFormatter(clean_number))
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def draw_second_row(
    fig: plt.Figure,
    *,
    fig_w: float,
    fig_h: float,
    row_bottom: float,
    panel_lefts: list[float],
    quant_lefts: list[float],
    panel_w: float,
    panel_h: float,
    c_panel_w: float,
    c_panel_h: float,
    result_root: Path,
    seed_index: int,
    sample_index: int,
) -> dict:
    runs = {method: load_run_json(result_root, method, seed_index, sample_index) for method in METHODS}
    trace = load_npz(Path(runs["apce"]["trace_npz"]))
    truth = np.asarray(trace["truth"], dtype=float)
    apce = np.asarray(trace["mean_states"], dtype=float)
    blackout_step = int(np.asarray(trace["blackout_start_step"]).item())
    forecast_slice = slice(blackout_step, None)
    truth_f = truth[forecast_slice]
    truth_display = truth
    apce_display = apce
    relerr = relative_error(apce_display, truth_display)
    initial_strip = make_initial_frame_strip(trace)

    vmax = 3.0
    rel_vmax = 0.4
    cmap_field = first.field_cmap()
    cmap_error = first.error_cmap()

    panel_bottom = row_bottom + (c_panel_h - panel_h)

    def add_panel(left_in: float) -> plt.Axes:
        return fig.add_axes([left_in / fig_w, panel_bottom / fig_h, panel_w / fig_w, panel_h / fig_h])

    def add_c_panel(left_in: float) -> plt.Axes:
        return fig.add_axes([left_in / fig_w, row_bottom / fig_h, c_panel_w / fig_w, c_panel_h / fig_h])

    ax_init = add_panel(panel_lefts[0])
    ax_truth = add_panel(panel_lefts[1])
    ax_apce = add_panel(panel_lefts[2])
    ax_error = add_panel(panel_lefts[3])
    ax_nrmse = add_c_panel(quant_lefts[0])
    ax_crps = add_c_panel(quant_lefts[1])
    ax_stat = add_c_panel(quant_lefts[2])

    image_handles = {}
    image_handles["initial"] = draw_image_panel(ax_init, initial_strip, "Initial frame", cmap_field, vmin=-vmax, vmax=vmax)
    image_handles["truth"] = draw_image_panel(ax_truth, truth_display.T, "Ref.", cmap_field, vmin=-vmax, vmax=vmax)
    image_handles["apce"] = draw_image_panel(ax_apce, apce_display.T, "APCE", cmap_field, vmin=-vmax, vmax=vmax)
    image_handles["error"] = draw_image_panel(
        ax_error,
        relerr.T,
        "Relative error",
        cmap_error,
        norm=PowerNorm(gamma=0.58, vmin=0.0, vmax=rel_vmax),
    )
    ax_init.set_xlabel("$t$", fontsize=FONT_AXIS, labelpad=2)
    ax_init.set_ylabel("$x$", fontsize=FONT_AXIS, labelpad=2, rotation=0)
    ax_init.yaxis.set_label_coords(-0.08, 0.5)
    arrow = FancyArrowPatch(
        ((panel_lefts[0] + panel_w + 0.08) / fig_w, (panel_bottom + panel_h * 0.50) / fig_h),
        ((panel_lefts[1] - 0.12) / fig_w, (panel_bottom + panel_h * 0.50) / fig_h),
        transform=fig.transFigure,
        arrowstyle="simple",
        mutation_scale=47,
        fc="#5DB8C4",
        ec="#5DB8C4",
        lw=0.0,
        alpha=0.95,
    )
    fig.add_artist(arrow)

    draw_metric_curve(ax_nrmse, runs, "lead_nrmse", "nRMSE", "nRMSE")
    draw_metric_curve(ax_crps, runs, "lead_crps", "CRPS", "CRPS")
    for ax in [ax_nrmse, ax_crps]:
        ax.set_xlim(40, 100)
        ax.set_xticks([40, 60, 80, 100])
        ax.legend(
            loc="upper left",
            fontsize=FONT_TICK,
            ncol=1,
            handlelength=1.15,
            labelspacing=0.20,
            borderaxespad=0.25,
        )
    ax_nrmse.set_ylim(bottom=0)
    ax_crps.set_ylim(bottom=0)
    method_fields: dict[str, np.ndarray] = {}
    for method in METHODS:
        method_trace = load_npz(Path(runs[method]["trace_npz"]))
        method_fields[method] = np.asarray(method_trace["mean_states"], dtype=float)[forecast_slice]
    first.draw_statistical_feature(ax_stat, truth_f, method_fields, downsampling_factor=32)
    ax_stat.set_title("Statistical feature", fontsize=FONT_TITLE, pad=8)
    ax_stat.set_ylim(-0.7, 1.02)

    add_horizontal_colorbar(fig, [ax_truth, ax_apce], image_handles["truth"], label="$u$", ticks=[-vmax, vmax], ticklabels=["-3", "3"], fig_h=fig_h)
    add_horizontal_colorbar(
        fig,
        [ax_error],
        image_handles["error"],
        label="$|u-\\hat u|/\\mathrm{rms}(u)$",
        ticks=[0.0, rel_vmax],
        ticklabels=["0", "0.4"],
        fig_h=fig_h,
    )
    fig.text((panel_lefts[0] - 0.14) / fig_w, (row_bottom + c_panel_h + 0.47) / fig_h, "d", fontsize=FONT_PANEL, fontweight="bold", ha="left", va="top")
    fig.text((panel_lefts[1] - 0.14) / fig_w, (row_bottom + c_panel_h + 0.47) / fig_h, "e", fontsize=FONT_PANEL, fontweight="bold", ha="left", va="top")
    fig.text((panel_lefts[4] - 0.14) / fig_w, (row_bottom + c_panel_h + 0.47) / fig_h, "f", fontsize=FONT_PANEL, fontweight="bold", ha="left", va="top")
    return {
        "second_row_seed": int(seed_index),
        "second_row_sample": int(sample_index),
        "second_row_trace": str(runs["apce"]["trace_npz"]),
        "second_row_apce_forecast_nrmse": float(runs["apce"]["forecast_nrmse"]),
        "second_row_apce_forecast_crps": float(runs["apce"]["forecast_crps"]),
        "second_row_apce_skill_horizon_time_020": float(runs["apce"]["skill_horizon_time_020"]),
        "same_seed_metrics": {
            method: {
                "forecast_nrmse": float(runs[method]["forecast_nrmse"]),
                "forecast_crps": float(runs[method]["forecast_crps"]),
                "skill_horizon_time_020": float(runs[method]["skill_horizon_time_020"]),
                "mu_absolute_error_at_blackout": float(runs[method]["mu_absolute_error_at_blackout"]),
            }
            for method in METHODS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw Figure 4 two-row KSE sparse-field plus blackout forecast panel.")
    parser.add_argument("--figure4-root", type=Path, default=DEFAULT_FIG4_ROOT)
    parser.add_argument("--blackout-root", type=Path, default=DEFAULT_BLACKOUT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--first-row-seed", type=int, default=90)
    parser.add_argument("--second-row-seed", type=int, default=90)
    parser.add_argument("--second-row-sample", type=int, default=0)
    args = parser.parse_args()

    first.configure_matplotlib()
    panel_w = 1.80
    panel_h = 1.742
    c_panel_w = 2.20
    c_panel_h = 2.00
    left_margin = 0.28
    right_margin = 0.22
    gap_small = 0.24
    gap_c = 0.65
    quant_shift = 0.42
    gap_arrow = 0.74
    gap_major = 0.74
    panel_lefts = [
        left_margin,
        left_margin + panel_w + gap_arrow,
        left_margin + panel_w + gap_arrow + panel_w + gap_small,
        left_margin + panel_w + gap_arrow + 2 * (panel_w + gap_small),
        left_margin + panel_w + gap_arrow + 3 * panel_w + 2 * gap_small + gap_major,
        left_margin + panel_w + gap_arrow + 3 * panel_w + 2 * gap_small + gap_major + c_panel_w + gap_c,
        left_margin + panel_w + gap_arrow + 3 * panel_w + 2 * gap_small + gap_major + 2 * c_panel_w + 2 * gap_c,
    ]
    fig_w = panel_lefts[-1] + c_panel_w + right_margin
    quant_lefts = [panel_lefts[4] + quant_shift, panel_lefts[5] + quant_shift, panel_lefts[6] + quant_shift]
    fig_w = max(fig_w, quant_lefts[-1] + c_panel_w + right_margin)
    top_margin = 0.48
    row_gap = 1.25
    bottom_margin = 0.72
    row1_bottom = bottom_margin + c_panel_h + row_gap
    row2_bottom = bottom_margin
    fig_h = top_margin + 2 * c_panel_h + row_gap + bottom_margin
    fig = plt.figure(figsize=(fig_w, fig_h))

    qa = {
        "output_base": str(args.output),
        "figure4_root": str(args.figure4_root),
        "blackout_root": str(args.blackout_root),
        "panel_geometry_inches": {
            "panel_w": panel_w,
            "panel_h": panel_h,
            "curve_panel_h": c_panel_h,
            "figure_w": fig_w,
            "figure_h": fig_h,
            "row_gap": row_gap,
            "row_alignment": "within each row, image panels and quantitative panels share the same top edge; row positions are translated copies of the approved first-row layout",
        },
        "font_rules": {"panel": FONT_PANEL, "title": FONT_TITLE, "axis": FONT_AXIS, "tick": FONT_TICK},
        "second_row_protocol": "same KSE field for reconstruction and forecast: seed90/sample00; row 1 uses the same blackout trace for reconstruction/state estimate, row 2 shows the full 100-frame blackout forecast output",
    }
    qa.update(
        draw_first_row(
            fig,
            fig_w=fig_w,
            fig_h=fig_h,
            row_bottom=row1_bottom,
            panel_lefts=panel_lefts,
            quant_lefts=quant_lefts,
            panel_w=panel_w,
            panel_h=panel_h,
            c_panel_w=c_panel_w,
            c_panel_h=c_panel_h,
            result_root=args.figure4_root,
            blackout_root=args.blackout_root,
            target_mu=1.1,
            seed_index=args.first_row_seed,
            sample_index=args.second_row_sample,
        )
    )
    qa.update(
        draw_second_row(
            fig,
            fig_w=fig_w,
            fig_h=fig_h,
            row_bottom=row2_bottom,
            panel_lefts=panel_lefts,
            quant_lefts=quant_lefts,
            panel_w=panel_w,
            panel_h=panel_h,
            c_panel_w=c_panel_w,
            c_panel_h=c_panel_h,
            result_root=args.blackout_root,
            seed_index=args.second_row_seed,
            sample_index=args.second_row_sample,
        )
    )
    save_all(fig, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
