"""Create a Figure 5-style VIV-PIV evidence composite.

The figure follows the approved Figure 4 visual contract: a wide six-column
layout, three compact rows, small readable axis text, lowercase panel labels,
and short in-figure titles.  The figure is deliberately assembled from the
sealed 20 x 40 formal summaries and the stored representative-field source
data; no test-field data are used to fit or tune the model here.

Rows:
    1. A representative transition-case reconstruction and blackout forecast.
    2. Shadow-evidence diagnostics for the same transition case.
    3. Paired five-case external-test metrics.

This is a plotting-only artifact. It does not modify the clean manuscript.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Circle
from PIL import Image


HERE = Path(__file__).resolve().parent
VIV = HERE.parent / "viv_piv_case"
OUT_DIR = HERE / "figure5_viv_piv"
OUT_BASE = OUT_DIR / "figure5_viv_piv_three_rows_v1"

FIELD_SOURCE = VIV / "figures" / "fig07_five_cases_uv_synthetic_speed_source.npz"
FIELD_META = VIV / "figures" / "fig07_five_cases_uv_synthetic_speed_metadata.json"
SUMMARY = VIV / "figures" / "fig08_five_cases_quantitative_metrics.csv"
BLACKOUT = VIV / "report_assets" / "blackout_metrics.csv"
ENERGY_SOURCE = VIV / "results_preview" / "kinetic_energy_spectra" / "kinetic_energy_spectra_source.npz"
EVIDENCE = VIV / "report_assets" / "fig04_weights_diagnostics_0679.png"
WEIGHT_MAP = VIV / "report_assets" / "fig04b_weight_maps_0679.png"

CASES = ["0463", "0556", "0679", "0803", "1359"]
CASE_UR = np.asarray([4.63, 5.56, 6.79, 8.03, 13.59])
PCE = "#3975a8"
APCE = "#c75b3f"
TRUTH = "#262626"
GREY = "#8b9399"
GRID = "#dedede"
FIELD_CMAP = "magma"
ERR_CMAP = "viridis"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7.0,
    "axes.titlesize": 8.0,
    "axes.labelsize": 7.0,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
    "legend.fontsize": 6.0,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
})


def short_title(ax: plt.Axes, text: str) -> None:
    ax.set_title(text, loc="left", pad=3.0, fontweight="normal")


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.14, 1.04, label, transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", va="bottom", ha="left", color="#111111")


def style_axis(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", which="major", pad=2, width=0.6, length=2.5)
    ax.grid(False)


def mask_image(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    out[~valid] = np.nan
    return out


def cylinder_center(x: np.ndarray, y: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    xx, yy = np.meshgrid(x, y)
    cylinder = (~valid) & (np.abs(xx) <= 0.65) & (np.abs(yy) <= 2.4)
    if not np.any(cylinder):
        return 0.0, 0.0
    return float(np.mean(xx[cylinder])), float(np.mean(yy[cylinder]))


def add_cylinder(ax: plt.Axes, center: tuple[float, float]) -> None:
    ax.add_patch(Circle(center, 0.5, facecolor="white", edgecolor="#333333", linewidth=0.55, zorder=5))


def load_field() -> dict[str, np.ndarray]:
    with np.load(FIELD_SOURCE, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def plot_sparse(ax: plt.Axes, x: np.ndarray, y: np.ndarray, truth_speed: np.ndarray,
                valid: np.ndarray, center: tuple[float, float]) -> None:
    # The 20 x 40 lattice is fixed before the external test is read.  Only
    # the valid fluid locations are displayed as sparse PIV measurements.
    ax.imshow(valid.astype(float), origin="lower", extent=[x.min(), x.max(), y.min(), y.max()],
              cmap="Greys", vmin=0, vmax=1, alpha=0.12, aspect="equal", interpolation="nearest")
    sx = np.linspace(1.0, 8.0, 20)
    sy = np.linspace(-2.0, 2.0, 40)
    xx, yy = np.meshgrid(sx, sy)
    ix = np.asarray([int(np.argmin(np.abs(x - value))) for value in sx])
    iy = np.asarray([int(np.argmin(np.abs(y - value))) for value in sy])
    sampled_valid = valid[np.ix_(iy, ix)]
    sampled_speed = truth_speed[np.ix_(iy, ix)]
    keep = sampled_valid.ravel()
    scatter = ax.scatter(xx.ravel()[keep], yy.ravel()[keep], c=sampled_speed.ravel()[keep],
                         s=7, cmap=FIELD_CMAP, norm=Normalize(vmin=0, vmax=float(np.nanpercentile(truth_speed[valid], 99))),
                         edgecolors="white", linewidths=0.12, zorder=3)
    add_cylinder(ax, center)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(float(y.min()), float(y.max()))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x/D$")
    ax.set_ylabel(r"$y/D$")
    short_title(ax, "Sparse PIV")
    style_axis(ax)
    return scatter


def plot_speed(ax: plt.Axes, image: np.ndarray, x: np.ndarray, y: np.ndarray,
               valid: np.ndarray, vmin: float, vmax: float, title: str,
               center: tuple[float, float]) -> mpl.image.AxesImage:
    shown = np.ma.masked_invalid(mask_image(image, valid))
    im = ax.imshow(shown, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()],
                   cmap=FIELD_CMAP, vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest")
    add_cylinder(ax, center)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(float(y.min()), float(y.max()))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x/D$")
    ax.set_ylabel(r"$y/D$")
    short_title(ax, title)
    style_axis(ax)
    return im


def crop_png(path: Path, box: tuple[int, int, int, int]) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").crop(box))


def evidence_crops() -> list[np.ndarray]:
    # Crop the plot areas only; titles are replaced by short Figure-4-style
    # panel titles in the outer figure.
    return [
        crop_png(EVIDENCE, (90, 330, 1810, 1325)),
        crop_png(EVIDENCE, (1710, 330, 3500, 1325)),
        crop_png(EVIDENCE, (90, 1640, 1810, 2515)),
        crop_png(EVIDENCE, (1710, 1640, 3500, 2515)),
        crop_png(WEIGHT_MAP, (1940, 460, 3700, 1320)),
        crop_png(WEIGHT_MAP, (80, 1640, 1880, 2510)),
    ]


def plot_crop(ax: plt.Axes, image: np.ndarray, title: str) -> None:
    ax.imshow(image, interpolation="nearest")
    ax.set_axis_off()
    short_title(ax, title)


def aggregate_blackout() -> pd.DataFrame:
    data = pd.read_csv(BLACKOUT)
    data = data[data["case_id"].astype(str).str.zfill(4) == "0679"].copy()
    data["method"] = data["method"].str.lower()
    data = data[data["method"].isin(["pce", "apce"])]
    return (data.groupby(["method", "horizon_s"], as_index=False)["evaluation_nrmse"]
            .agg(mean="mean", sd="std"))


def plot_energy(ax: plt.Axes) -> None:
    with np.load(ENERGY_SOURCE, allow_pickle=False) as data:
        truth = np.asarray(data["0679_truth"], dtype=float)
        pce = np.asarray(data["0679_PCE, full R"], dtype=float)
        apce = np.asarray(data["0679_APCE, full R"], dtype=float)
    time = np.arange(truth.size, dtype=float) * 0.1
    ax.plot(time, truth, color=TRUTH, lw=0.85, label="truth")
    ax.plot(time, pce, color=PCE, lw=0.8, label="PCE")
    ax.plot(time, apce, color=APCE, lw=0.8, label="APCE")
    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"$E(t)$")
    short_title(ax, "Kinetic energy")
    ax.legend(loc="lower right", ncol=3, handlelength=1.2, columnspacing=0.8, borderaxespad=0.2)
    style_axis(ax)


def plot_blackout(ax: plt.Axes) -> None:
    data = aggregate_blackout()
    for method, color, label in [("pce", PCE, "PCE"), ("apce", APCE, "APCE")]:
        row = data[data["method"] == method].sort_values("horizon_s")
        ax.errorbar(row["horizon_s"], row["mean"], yerr=row["sd"].fillna(0), color=color,
                    marker="o", ms=3.0, lw=1.0, capsize=1.8, label=label)
    ax.set_xlabel("horizon (s)")
    ax.set_ylabel("nRMSE")
    short_title(ax, "Blackout forecast")
    ax.set_xlim(0.35, 4.15)
    ax.legend(loc="upper left", ncol=2, handlelength=1.2, columnspacing=0.8, borderaxespad=0.2)
    style_axis(ax)


def plot_metric(ax: plt.Axes, summary: pd.DataFrame, field: str, title: str,
                ylabel: str, ylim: tuple[float, float] | None = None, target: float | None = None,
                percent: bool = False) -> None:
    for method, color, label, marker in [("pce", PCE, "PCE", "o"), ("apce", APCE, "APCE", "s")]:
        values = summary[summary["method"] == method]
        if values.empty:
            # The external summary has one row per case for the APCE branch;
            # PCE values for comparison are supplied by the paired report.
            continue
        values = values.set_index("case_id").reindex(CASES)
        y = values[field].to_numpy(dtype=float)
        ax.plot(CASE_UR, y, color=color, marker=marker, ms=3.4, lw=1.0, label=label)
    ax.set_xticks(CASE_UR, ["4.63", "5.56", "6.79", "8.03", "13.59"], rotation=0)
    ax.set_xlabel(r"$U_r$")
    ax.set_ylabel(ylabel)
    short_title(ax, title)
    if target is not None:
        ax.axhline(target, color="#777777", lw=0.7, ls=":")
    if ylim is not None:
        ax.set_ylim(*ylim)
    style_axis(ax)


def make_metric_table(summary: pd.DataFrame) -> pd.DataFrame:
    # fig08 is APCE-only.  The paired PCE rows are the corresponding values
    # in report_assets/summary_metrics.csv, which are deterministic across the
    # three probabilistic seeds after aggregation.
    paired = pd.read_csv(VIV / "report_assets" / "summary_metrics.csv")
    paired["case_id"] = paired["case_id"].astype(str).str.zfill(4)
    paired = paired[paired["case_id"].isin(CASES)].copy()
    paired = paired[paired["method"].isin(["pce", "apce"])]
    grouped = paired.groupby(["case_id", "method"], as_index=False).agg(
        field_nrmse=("full_field_physical_nrmse", "mean"),
        crps=("normalized_crps", "mean"),
        coverage=("coverage_90", "mean"),
        blackout=("blackout_mean_nrmse", "mean"),
        energy_nrmse=("kinetic_energy_nrmse", "mean"),
        energy_corr=("kinetic_energy_correlation", "mean"),
    )
    grouped["case_id"] = grouped["case_id"].astype(str).str.zfill(4)
    return grouped


def save_figure(fig: plt.Figure) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_BASE.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white", pad_inches=0.03)
    fig.savefig(OUT_BASE.with_suffix(".pdf"), bbox_inches="tight", facecolor="white", pad_inches=0.03)
    fig.savefig(OUT_BASE.with_suffix(".svg"), bbox_inches="tight", facecolor="white", pad_inches=0.03)
    fig.savefig(OUT_BASE.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white", pad_inches=0.03,
                pil_kwargs={"compression": "tiff_lzw"})


def main() -> None:
    fields = load_field()
    case = "0679"
    x = fields[f"{case}_x_over_d"]
    y = fields[f"{case}_y_over_d"]
    valid = fields[f"{case}_valid"].astype(bool)
    truth_speed = fields[f"{case}_truth_speed"]
    apce_speed = fields[f"{case}_apce_speed"]
    error = np.abs(apce_speed - truth_speed)
    speed_vmin = float(np.nanpercentile(truth_speed[valid], 1))
    speed_vmax = float(np.nanpercentile(truth_speed[valid], 99))
    error_vmax = float(np.nanpercentile(error[valid], 99.5))
    center = cylinder_center(x, y, valid)

    summary = make_metric_table(pd.read_csv(SUMMARY))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_DIR / "figure5_viv_piv_source_metrics.csv", index=False)
    aggregate_blackout().to_csv(OUT_DIR / "figure5_viv_piv_source_blackout_0679.csv", index=False)

    fig = plt.figure(figsize=(17.7, 9.25), facecolor="white")
    gs = fig.add_gridspec(3, 6, left=0.035, right=0.995, bottom=0.055, top=0.925,
                          wspace=0.28, hspace=0.44, height_ratios=[1.16, 0.72, 0.80])

    # Row 1: representative field and conditional forecast.
    axes = [fig.add_subplot(gs[0, i]) for i in range(6)]
    sparse_im = plot_sparse(axes[0], x, y, truth_speed, valid, center)
    ref_im = plot_speed(axes[1], truth_speed, x, y, valid, speed_vmin, speed_vmax,
                        r"Reference $|\mathbf{v}|$", center)
    pred_im = plot_speed(axes[2], apce_speed, x, y, valid, speed_vmin, speed_vmax, "APCE", center)
    err = np.ma.masked_invalid(mask_image(error, valid))
    axes[3].imshow(err, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()], cmap=ERR_CMAP,
                   vmin=0, vmax=error_vmax, aspect="equal", interpolation="nearest")
    add_cylinder(axes[3], center)
    axes[3].set_xlim(float(x.min()), float(x.max())); axes[3].set_ylim(float(y.min()), float(y.max()))
    axes[3].set_aspect("equal", adjustable="box"); axes[3].set_xlabel(r"$x/D$"); axes[3].set_ylabel(r"$y/D$")
    short_title(axes[3], "Absolute error"); style_axis(axes[3])
    plot_energy(axes[4])
    plot_blackout(axes[5])
    for label, ax in zip("abcdef", axes):
        panel_label(ax, label)
    cb1 = fig.colorbar(ref_im, ax=[axes[1], axes[2]], orientation="horizontal", fraction=0.055, pad=0.16, aspect=35)
    cb1.set_label(r"speed (m s$^{-1}$)", fontsize=6.5, labelpad=1)
    cb1.ax.tick_params(labelsize=5.5, length=2, pad=1)
    cb2 = fig.colorbar(axes[3].images[0], ax=axes[3], orientation="horizontal", fraction=0.055, pad=0.16, aspect=25)
    cb2.set_label("error", fontsize=6.5, labelpad=1)
    cb2.ax.tick_params(labelsize=5.5, length=2, pad=1)

    # Row 2: shadow evidence.  These source panels are exact crops of the
    # stored formal diagnostic figure; the surrounding labels remain editable.
    axes2 = [fig.add_subplot(gs[1, i]) for i in range(6)]
    crops = evidence_crops()
    for ax, image, title in zip(axes2, crops,
                               ["max weight", "shadow gap", "entropy", "effective count", "APCE weight map", "candidate coordinate"]):
        plot_crop(ax, image, title)
    for label, ax in zip("ghijkl", axes2):
        panel_label(ax, label)

    # Row 3: all five external test conditions, paired PCE/APCE.
    axes3 = [fig.add_subplot(gs[2, i]) for i in range(6)]
    plot_metric(axes3[0], summary, "field_nrmse", "Field nRMSE", "nRMSE", ylim=(0.0, 0.28))
    plot_metric(axes3[1], summary, "crps", "nCRPS", "CRPS", ylim=(0.0, 0.23))
    plot_metric(axes3[2], summary, "coverage", "90% coverage", "coverage", ylim=(0.88, 1.01), target=0.90)
    plot_metric(axes3[3], summary, "blackout", "Blackout nRMSE", "nRMSE", ylim=(0.18, 0.38))
    plot_metric(axes3[4], summary, "energy_nrmse", "Energy nRMSE", "nRMSE", ylim=(0.0, 0.06))
    plot_metric(axes3[5], summary, "energy_corr", "Energy correlation", "Pearson $r$", ylim=(0.80, 1.0), target=0.90)
    for label, ax in zip("mnopqr", axes3):
        panel_label(ax, label)
    axes3[0].legend(loc="upper left", ncol=2, handlelength=1.15, columnspacing=0.8, borderaxespad=0.2)

    fig.canvas.draw()
    fig.text(0.035, axes[0].get_position().y1 + 0.038, r"VIV--PIV transition case ($U_r=6.79$)",
             fontsize=11.0, fontweight="bold", ha="left", va="bottom")
    fig.text(0.035, axes2[0].get_position().y1 + 0.036, "Shadow evidence",
             fontsize=11.0, fontweight="bold", ha="left", va="bottom")
    fig.text(0.035, axes3[0].get_position().y1 + 0.036, r"External tests ($20\times40$)",
             fontsize=11.0, fontweight="bold", ha="left", va="bottom")

    save_figure(fig)
    plt.close(fig)

    qa = {
        "output_base": str(OUT_BASE),
        "backend": "Python/matplotlib",
        "figure_contract": {
            "claim": "Shadow-anchored predictive evidence supports auditable sparse-field reconstruction and short conditional forecasts in a real VIV-PIV experiment.",
            "archetype": "quantitative grid + image plate",
            "protocol": "20x40 formal external tests; 12 training cases; five held-out cases; three paired seeds",
        },
        "figure4_rules_applied": {
            "columns": 6,
            "rows": 3,
            "body_font_pt": 7.0,
            "axis_font_pt": 7.0,
            "tick_font_pt": 6.0,
            "panel_label_font_pt": 9.5,
            "row_title_font_pt": 11.0,
            "short_in_figure_titles": True,
            "long_caption_in_figure": False,
            "editable_outer_text_exports": ["svg", "pdf"],
            "raster_data_panels": "field rasters and stored evidence-diagnostic crops",
        },
        "sources": {
            "field_source": str(FIELD_SOURCE),
            "field_metadata": str(FIELD_META),
            "formal_summary": str(SUMMARY),
            "blackout_summary": str(BLACKOUT),
            "energy_source": str(ENERGY_SOURCE),
            "evidence_diagnostics": str(EVIDENCE),
            "weight_map": str(WEIGHT_MAP),
        },
        "representative_case": {"case_id": "0679", "reduced_velocity": 6.79, "frame_rule": "maximum absolute centred-cylinder displacement after warm-up"},
        "field_limits": {"speed_vmin": speed_vmin, "speed_vmax": speed_vmax, "error_vmax": error_vmax},
        "cylinder_center_x_over_d_y_over_d": [center[0], center[1]],
        "outputs": [str(OUT_BASE.with_suffix(ext)) for ext in [".png", ".pdf", ".svg", ".tiff"]],
    }
    (OUT_BASE.with_suffix(".json")).write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
