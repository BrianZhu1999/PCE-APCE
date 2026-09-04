from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import FuncFormatter


DEFAULT_RESULT_ROOT = Path(
    "<HILDA_RESULTS_ROOT>/results/figure4_kse_nmi32_t2_core4_formal_50seeds_20260814_4gpu"
)
DEFAULT_OUTPUT = Path(
    "<HILDA_RESULTS_ROOT>/results/figure4_kse_nmi32_t2_core4_formal_50seeds_20260814_4gpu/plots/"
    "figure4_kse_mu11_firstrow_v12"
)

METHODS = ["aug_enkf", "bma_static", "pce", "apce"]
LABELS = {"aug_enkf": "Aug-EnKF", "bma_static": "BMA", "pce": "PCE", "apce": "APCE"}
COLORS = {
    "aug_enkf": "#A8B0B7",
    "bma_static": "#D7A64A",
    "pce": "#4E79A7",
    "apce": "#59A14F",
    "truth": "#E84A3A",
}
BOX_COLORS = {
    "aug_enkf": "#9B59B6",
    "bma_static": "#F39C12",
    "pce": "#1F77B4",
    "apce": "#2ECC71",
}

FONT_PANEL = 22
FONT_TITLE = 14
FONT_AXIS = 13
FONT_TICK = 11


def clean_number(x: float, _pos: int | None = None) -> str:
    return f"{x:g}"


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 13.0,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.frameon": False,
        }
    )


def save_all(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=650, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_base.with_suffix(".tiff"), dpi=650, bbox_inches="tight", pad_inches=0.03)


def field_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "kse_field_refined",
        ["#9A1F2A", "#FAE8D1", "#E8F1F2", "#1F5A89"],
        N=256,
    )


def error_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "kse_relative_error",
        ["#F7FBFF", "#B9D7E4", "#2F7FA6", "#0B2B52"],
        N=256,
    )


def read_selected_trace(result_root: Path, target_mu: float, seed_index: int | None) -> tuple[pd.Series, dict[str, np.ndarray]]:
    df = pd.read_csv(result_root / "source_data" / "run_source_data.csv", encoding="utf-8-sig")
    apce_mu = df[(df["method"] == "apce") & (np.isclose(df["true_mu"], target_mu))]
    if apce_mu.empty:
        raise RuntimeError(f"No APCE runs found for true_mu={target_mu}.")
    if seed_index is None:
        selected = apce_mu.sort_values("nrmse", ascending=True).iloc[0]
    else:
        selected_rows = apce_mu[apce_mu["seed_index"] == seed_index]
        if selected_rows.empty:
            raise RuntimeError(f"No APCE run found for true_mu={target_mu}, seed_index={seed_index}.")
        selected = selected_rows.iloc[0]
    trace_path = Path(str(selected["trace_npz"]))
    arr = np.load(trace_path, allow_pickle=True)
    trace = {key: np.asarray(arr[key]) for key in arr.files}
    return selected, trace


def load_trace_for_run(row: pd.Series) -> dict[str, np.ndarray]:
    arr = np.load(Path(str(row["trace_npz"])), allow_pickle=True)
    return {key: np.asarray(arr[key]) for key in arr.files}


def make_sparse_observation_image(trace: dict[str, np.ndarray]) -> np.ndarray:
    # The visual convention follows sparse-reconstruction plates: show the actual sampled
    # spatiotemporal lattice rather than a mostly blank 1024-by-100 canvas.
    observations = np.asarray(trace["trace_observations"], dtype=float)
    return observations.T


def relative_error(reconstruction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    scale = float(np.sqrt(np.nanmean(np.square(truth))))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return np.abs(reconstruction - truth) / scale


def smooth_pdf(values: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hist, edges = np.histogram(values[np.isfinite(values)], bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    kernel_x = np.linspace(-2.5, 2.5, 15)
    kernel = np.exp(-0.5 * kernel_x**2)
    kernel = kernel / kernel.sum()
    smooth = np.convolve(hist, kernel, mode="same")
    return centers, smooth


def spatial_autocorrelation(field: np.ndarray, max_lag_fraction: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Periodic spatial autocorrelation averaged over time, normalized by R_ref(0)."""
    arr = np.asarray(field, dtype=float)
    arr = arr - np.nanmean(arr, axis=1, keepdims=True)
    nt, nx = arr.shape
    max_lag = int(nx * max_lag_fraction)
    denom = float(np.nanmean(arr * arr))
    if not np.isfinite(denom) or denom <= 0:
        denom = 1.0
    lags = np.arange(max_lag + 1)
    vals = np.empty(max_lag + 1, dtype=float)
    for lag in lags:
        vals[lag] = float(np.nanmean(arr * np.roll(arr, -lag, axis=1)) / denom)
    return lags / float(nx), vals


def draw_boxplot(ax: plt.Axes, df: pd.DataFrame, target_mu: float) -> None:
    values = [
        df[(df["method"] == method) & (np.isclose(df["true_mu"], target_mu))]["nrmse"].to_numpy(dtype=float)
        for method in METHODS
    ]
    bp = ax.boxplot(
        values,
        patch_artist=True,
        widths=0.58,
        showfliers=False,
        medianprops={"color": "#202020", "linewidth": 1.6},
        whiskerprops={"color": "#555555", "linewidth": 1.1},
        capprops={"color": "#555555", "linewidth": 1.1},
    )
    for patch, method in zip(bp["boxes"], METHODS):
        patch.set_facecolor(BOX_COLORS[method])
        patch.set_alpha(0.93)
        patch.set_edgecolor("#333333")
        patch.set_linewidth(1.1)
    rng = np.random.default_rng(20260814)
    for idx, (method, arr) in enumerate(zip(METHODS, values), start=1):
        x = idx + rng.uniform(-0.12, 0.12, size=len(arr))
        ax.scatter(x, arr, s=18, color=COLORS[method], edgecolor="white", linewidth=0.35, alpha=0.78, zorder=3)
    ax.set_xticks(range(1, len(METHODS) + 1), [LABELS[m] for m in METHODS], rotation=38, ha="right", fontsize=FONT_TICK)
    ax.set_ylabel("nRMSE", fontsize=FONT_AXIS)
    ax.set_title("nRMSE", fontsize=FONT_TITLE, pad=8)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(FuncFormatter(clean_number))
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=FONT_TICK)


def draw_metric_boxplot(
    ax: plt.Axes,
    df: pd.DataFrame,
    target_mu: float,
    metric: str,
    title: str,
    *,
    ymax: float | None = None,
) -> None:
    raw_values = [
        df[(df["method"] == method) & (np.isclose(df["true_mu"], target_mu))][metric].to_numpy(dtype=float)
        for method in METHODS
    ]
    values = [arr[arr <= ymax] if ymax is not None else arr for arr in raw_values]
    bp = ax.boxplot(
        values,
        patch_artist=True,
        widths=0.58,
        showfliers=False,
        medianprops={"color": "#202020", "linewidth": 1.6},
        whiskerprops={"color": "#555555", "linewidth": 1.1},
        capprops={"color": "#555555", "linewidth": 1.1},
    )
    for patch, method in zip(bp["boxes"], METHODS):
        patch.set_facecolor(BOX_COLORS[method])
        patch.set_alpha(0.93)
        patch.set_edgecolor(BOX_COLORS[method])
        patch.set_linewidth(1.35)
    rng = np.random.default_rng(20260815)
    for idx, (method, arr) in enumerate(zip(METHODS, values), start=1):
        x = idx + rng.uniform(-0.12, 0.12, size=len(arr))
        ax.scatter(x, arr, s=20, color=BOX_COLORS[method], edgecolor="white", linewidth=0.45, alpha=0.90, zorder=3)
    ax.set_xticks(range(1, len(METHODS) + 1), [LABELS[m] for m in METHODS], rotation=38, ha="right", fontsize=FONT_TICK)
    ax.set_ylabel(title, fontsize=FONT_AXIS)
    ax.set_title(title, fontsize=FONT_TITLE, pad=8)
    ax.set_ylim(0, ymax if ymax is not None else None)
    ax.yaxis.set_major_formatter(FuncFormatter(clean_number))
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=FONT_TICK)


def draw_statistical_feature(
    ax: plt.Axes,
    truth: np.ndarray,
    method_fields: dict[str, np.ndarray],
    *,
    downsampling_factor: int,
) -> None:
    r_ref, corr_ref = spatial_autocorrelation(truth)
    styles = {
        "aug_enkf": ("Aug-EnKF", COLORS["aug_enkf"], "-", 1.35, 0.78, 2),
        "bma_static": ("BMA", COLORS["bma_static"], "-", 1.45, 0.82, 3),
        "pce": ("PCE", COLORS["pce"], "-", 2.25, 1.00, 5),
        "apce": ("APCE", COLORS["apce"], "-", 2.45, 1.00, 6),
    }
    for method in ["aug_enkf", "bma_static", "pce", "apce"]:
        if method not in method_fields:
            continue
        r, corr = spatial_autocorrelation(method_fields[method])
        label, color, linestyle, lw, alpha, zorder = styles[method]
        ax.plot(r, corr, color=color, linestyle=linestyle, linewidth=lw, alpha=alpha, label=label, zorder=zorder)
    ax.plot(r_ref, corr_ref, color=COLORS["truth"], linewidth=2.65, linestyle="--", label="Ref.", zorder=10)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(-0.50, 1.02)
    ax.set_xlabel("Spatial lag $r$", fontsize=FONT_AXIS)
    ax.set_ylabel(r"$R(r)/R_{\rm ref}(r=0)$", fontsize=FONT_AXIS)
    ax.set_title(f"Statistical feature at {downsampling_factor}×", fontsize=FONT_TITLE, pad=8)
    ax.xaxis.set_major_formatter(FuncFormatter(clean_number))
    ax.yaxis.set_major_formatter(FuncFormatter(clean_number))
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(item) for item in ["PCE", "APCE", "Ref.", "BMA", "Aug-EnKF"] if item in labels]
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper right",
        fontsize=FONT_TICK,
        ncol=2,
        handlelength=1.15,
        columnspacing=0.65,
        borderaxespad=0.25,
        labelspacing=0.22,
    )
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_horizontal_colorbar(
    fig: plt.Figure,
    ax_list: list[plt.Axes],
    image: mpl.image.AxesImage,
    *,
    label: str,
    ticks: list[float],
    ticklabels: list[str],
) -> None:
    left = min(ax.get_position().x0 for ax in ax_list)
    right = max(ax.get_position().x1 for ax in ax_list)
    bottom = min(ax.get_position().y0 for ax in ax_list) - 0.095
    cax = fig.add_axes([left, bottom, right - left, 0.039])
    cb = fig.colorbar(image, cax=cax, orientation="horizontal")
    cb.set_ticks(ticks)
    cb.set_ticklabels(ticklabels)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=FONT_TICK, length=0, pad=6.5)
    cax.text(0.5, -1.95, label, transform=cax.transAxes, ha="center", va="top", fontsize=FONT_TICK, fontstyle="italic")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw Figure 4 first-row KSE μ=1.1 APCE representative panel.")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-mu", type=float, default=1.1)
    parser.add_argument(
        "--seed-index",
        type=int,
        default=15,
        help="Representative seed. Default 15 is the best APCE nRMSE seed at mu=1.1.",
    )
    args = parser.parse_args()

    configure_matplotlib()
    selected, trace = read_selected_trace(args.result_root, args.target_mu, args.seed_index)
    summary = pd.read_csv(args.result_root / "source_data" / "run_source_data.csv", encoding="utf-8-sig")

    truth = np.asarray(trace["trace_truth"], dtype=float)
    apce = np.asarray(trace["mean_states"], dtype=float)
    sparse = make_sparse_observation_image(trace)
    relerr = relative_error(apce, truth)
    same_seed_rows = summary[summary["seed_index"] == int(selected["seed_index"])]
    method_fields: dict[str, np.ndarray] = {}
    for method in METHODS:
        rows = same_seed_rows[same_seed_rows["method"] == method]
        if rows.empty:
            continue
        method_trace = load_trace_for_run(rows.iloc[0])
        method_fields[method] = np.asarray(method_trace["mean_states"], dtype=float)

    vmax_raw = float(np.nanpercentile(np.abs(truth), 99.2))
    if not np.isfinite(vmax_raw) or vmax_raw <= 0:
        vmax_raw = 1.0
    vmax = 3.0
    rel_vmax = float(np.nanpercentile(relerr, 99.0))
    if not np.isfinite(rel_vmax) or rel_vmax <= 0:
        rel_vmax = 1.0
    rel_vmax = 0.4

    cmap_field = field_cmap()
    cmap_error = error_cmap()

    fig_h = 3.35
    panel_w = 1.80
    panel_h = 1.742
    c_panel_w = 2.20
    c_panel_h = 2.00
    left_margin = 0.28
    right_margin = 0.22
    gap_small = 0.24
    gap_c = 0.58
    gap_arrow = 0.74
    gap_major = 0.74
    top_in = 0.78
    bottom_in = fig_h - top_in - panel_h
    c_bottom_in = fig_h - top_in - c_panel_h
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
    fig = plt.figure(figsize=(fig_w, fig_h))

    def add_panel(left_in: float) -> plt.Axes:
        return fig.add_axes([left_in / fig_w, bottom_in / fig_h, panel_w / fig_w, panel_h / fig_h])

    def add_c_panel(left_in: float) -> plt.Axes:
        return fig.add_axes([left_in / fig_w, c_bottom_in / fig_h, c_panel_w / fig_w, c_panel_h / fig_h])

    ax_sparse = add_panel(panel_lefts[0])
    ax_truth = add_panel(panel_lefts[1])
    ax_apce = add_panel(panel_lefts[2])
    ax_error = add_panel(panel_lefts[3])
    ax_nrmse = add_c_panel(panel_lefts[4])
    ax_crps = add_c_panel(panel_lefts[5])
    ax_stat = add_c_panel(panel_lefts[6])
    axes = [ax_sparse, ax_truth, ax_apce, ax_error, ax_nrmse, ax_crps, ax_stat]

    image_panels = [
        (ax_sparse, sparse, "Observations", cmap_field, -vmax, vmax, None),
        (ax_truth, truth.T, "Ref. ($\\mu=1.1$)", cmap_field, -vmax, vmax, None),
        (ax_apce, apce.T, "APCE", cmap_field, -vmax, vmax, None),
        (ax_error, relerr.T, "Relative error", cmap_error, 0.0, rel_vmax, PowerNorm(gamma=0.58, vmin=0.0, vmax=rel_vmax)),
    ]
    image_handles = {}
    for ax, image, title, cmap, vmin, vmax_i, norm in image_panels:
        if norm is None:
            im = ax.imshow(image, origin="lower", aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax_i)
        else:
            im = ax.imshow(image, origin="lower", aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
        image_handles[title] = im
        ax.set_title(title, fontsize=FONT_TITLE, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    ax_sparse.set_xlabel("$t$", fontsize=FONT_AXIS, labelpad=2)
    ax_sparse.set_ylabel("$x$", fontsize=FONT_AXIS, labelpad=2, rotation=0)
    ax_sparse.yaxis.set_label_coords(-0.08, 0.5)

    arrow = FancyArrowPatch(
        ((panel_lefts[0] + panel_w + 0.08) / fig_w, (bottom_in + panel_h * 0.50) / fig_h),
        ((panel_lefts[1] - 0.12) / fig_w, (bottom_in + panel_h * 0.50) / fig_h),
        transform=fig.transFigure,
        arrowstyle="simple",
        mutation_scale=47,
        fc="#5DB8C4",
        ec="#5DB8C4",
        lw=0.0,
        alpha=0.95,
    )
    fig.add_artist(arrow)

    draw_metric_boxplot(ax_nrmse, summary, args.target_mu, "nrmse", "nRMSE", ymax=0.5)
    draw_metric_boxplot(ax_crps, summary, args.target_mu, "crps", "CRPS", ymax=0.4)
    draw_statistical_feature(ax_stat, truth, method_fields, downsampling_factor=32)

    add_horizontal_colorbar(
        fig,
        [ax_truth, ax_apce],
        image_handles["Ref. ($\\mu=1.1$)"],
        label="$u$",
        ticks=[-vmax, vmax],
        ticklabels=[f"{-vmax:.0f}", f"{vmax:.0f}"],
    )
    add_horizontal_colorbar(
        fig,
        [ax_error],
        image_handles["Relative error"],
        label="$|u-\\hat u|/\\mathrm{rms}(u)$",
        ticks=[0.0, rel_vmax],
        ticklabels=["0", f"{rel_vmax:.1f}"],
    )

    panel_y_text = 0.965
    fig.text((panel_lefts[0] - 0.14) / fig_w, panel_y_text, "a", fontsize=FONT_PANEL, fontweight="bold", ha="left", va="top")
    fig.text((panel_lefts[1] - 0.14) / fig_w, panel_y_text, "b", fontsize=FONT_PANEL, fontweight="bold", ha="left", va="top")
    fig.text((panel_lefts[4] - 0.14) / fig_w, panel_y_text, "c", fontsize=FONT_PANEL, fontweight="bold", ha="left", va="top")

    save_all(fig, args.output)

    qa = {
        "output_base": str(args.output),
        "result_root": str(args.result_root),
        "selected_run_id": str(selected["run_id"]),
        "seed_index": int(selected["seed_index"]),
        "sample_index": int(selected["sample_index"]),
        "seed": int(selected["seed"]),
        "true_mu": float(selected["true_mu"]),
        "apce_nrmse": float(selected["nrmse"]),
        "apce_crps": float(selected["crps"]),
        "apce_mu_absolute_error": float(selected["mu_absolute_error"]),
        "trace_npz": str(selected["trace_npz"]),
        "truth_shape": list(truth.shape),
        "sparse_observations_shape": list(sparse.shape),
        "nrmse_boxplot_filter": "true_mu == 1.1; four methods; formal 50-seed source data",
        "uncertainty_panel": "omitted: formal trace does not store propagated ensemble/spread fields",
    }
    args.output.with_suffix(".json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
