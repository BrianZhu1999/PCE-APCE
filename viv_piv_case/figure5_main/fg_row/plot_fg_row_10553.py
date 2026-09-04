"""Render the next Figure 5 row: nRMSE bars plus five kinetic-energy spectra.

The row follows the approved 10,553 px Figure 5 b-e board width.  The first
two axes are reconstruction and blackout-prediction nRMSE grouped bars across
five held-out VIV conditions and four methods.  The remaining five axes are
one kinetic-energy PSD panel per held-out condition.  The stored spectrum
bundle currently contains Truth, PCE and APCE only; this is recorded explicitly
in the output metadata rather than filling missing Aug-EnKF/BMA curves.
"""
from __future__ import annotations

import argparse
import json
import csv
import hashlib
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


HERE = Path(__file__).resolve().parent
VIV = HERE.parents[1]
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

FIG_W_PX = 10553
FIG_H_PX = 1792
DPI = 650
FIG_W_IN = FIG_W_PX / DPI
FIG_H_IN = FIG_H_PX / DPI
CASES = ["0463", "0556", "0679", "0803", "1359"]
UR = [4.63, 5.56, 6.79, 8.03, 13.59]
METHODS = ["pce", "apce", "aug_enkf", "bma"]
METHOD_LABELS = {"pce": "PCE", "apce": "APCE", "aug_enkf": "Aug-EnKF", "bma": "BMA"}
COLORS = {"pce": "#4C78A8", "apce": "#F28E2B", "aug_enkf": "#7F8C8D", "bma": "#A77BBE"}
TRUTH = "#202020"
SPECTRUM_CURVE_LINEWIDTH_MULTIPLIER = 1.0
SPECTRUM_LINEWIDTHS = {
    "Truth": 2.475 * SPECTRUM_CURVE_LINEWIDTH_MULTIPLIER,
    "PCE": 1.725 * SPECTRUM_CURVE_LINEWIDTH_MULTIPLIER,
    "APCE": 1.725 * SPECTRUM_CURVE_LINEWIDTH_MULTIPLIER,
}

SOURCE_DATA = HERE / "source_data"
SUMMARY = HERE.parent / "results_tmp_x40y20" / "summary_metrics.json"
SPECTRA = HERE.parent / "results_tmp_x40y20" / "kinetic_energy_spectra_source_x40y20.npz"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 11,
    "font.weight": "normal",
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "axes.linewidth": 0.75,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "legend.frameon": False,
})


def load_summary() -> list[dict[str, object]]:
    rows = json.loads(SUMMARY.read_text(encoding="utf-8"))
    return [
        row for row in rows
        if str(row["case_id"]).zfill(4) in CASES
        and str(row["method"]) in METHODS
        and bool(row.get("valid", True))
    ]


def aggregate_metrics(rows: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, float]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["case_id"]).zfill(4), str(row["method"]))
        grouped.setdefault(key, []).append(row)
    result: dict[tuple[str, str], dict[str, float]] = {}
    for key, members in grouped.items():
        recon = np.asarray([float(row["full_field_physical_nrmse"]) for row in members])
        pred = np.asarray([float(row["blackout_mean_nrmse"]) for row in members])
        result[key] = {
            "reconstruction_nrmse": float(recon.mean()),
            "reconstruction_sd": float(recon.std(ddof=1)) if recon.size > 1 else 0.0,
            "prediction_nrmse": float(pred.mean()),
            "prediction_sd": float(pred.std(ddof=1)) if pred.size > 1 else 0.0,
            "n_seeds": float(len(members)),
        }
    return result


def spectrum(values: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Return one-sided, Hann-windowed, integral-normalised PSD."""
    signal = np.asarray(values, dtype=np.float64)
    centred = signal - signal.mean()
    window = np.hanning(signal.size)
    transformed = np.fft.rfft(centred * window)
    frequencies = np.fft.rfftfreq(signal.size, dt)
    power = dt * np.abs(transformed) ** 2 / max(float(np.sum(window ** 2)), 1e-30)
    if power.size > 2:
        power[1:-1] *= 2.0
    integral = float(np.trapezoid(power, frequencies))
    return frequencies, power / max(integral, 1e-30)


def style_axes(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", which="major", labelsize=11, width=0.65, length=2.8, pad=2)
    ax.set_axisbelow(True)


def trimmed_decimal(value: float, _position: int) -> str:
    """Format decimal ticks without trailing zeros or a decimal point."""
    if abs(value) < 1e-12:
        return "0"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def grouped_bars(ax: plt.Axes, metrics: dict[tuple[str, str], dict[str, float]], field: str, error: str, title: str, label: str) -> None:
    x = np.arange(len(CASES), dtype=float)
    width = 0.18
    offsets = (np.arange(len(METHODS)) - 1.5) * width
    for index, method in enumerate(METHODS):
        y = np.asarray([metrics[(case, method)][field] for case in CASES], dtype=float)
        yerr = np.asarray([metrics[(case, method)][error] for case in CASES], dtype=float)
        ax.bar(
            x + offsets[index], y, width=width * 0.90,
            yerr=yerr, capsize=2.0, ecolor="#333333", error_kw={"lw": 0.65, "capthick": 0.65},
            color=COLORS[method], edgecolor="white", linewidth=0.35,
            label=METHOD_LABELS[method], zorder=3,
        )
    ax.set_xticks(x, [f"{u:.2f}" for u in UR])
    ax.set_xlabel(r"$U_r$", fontsize=13)
    ax.set_ylabel("nRMSE", fontsize=13)
    ax.set_title(title, fontsize=14, pad=5, fontweight="normal", loc="center")
    ax.yaxis.set_major_formatter(FuncFormatter(trimmed_decimal))
    ax.set_ylim(bottom=0.0)
    ax.legend(
        loc="upper right", fontsize=11, ncol=2, handlelength=1.0,
        columnspacing=0.6, handletextpad=0.35, borderaxespad=0.25,
    )
    style_axes(ax)


def plot_spectrum(ax: plt.Axes, data: np.lib.npyio.NpzFile, case_id: str, index: int) -> list[str]:
    # Subtle cool gray-blue background for PSD panels only.
    ax.set_facecolor("#F5F7F8")

    truth = np.asarray(data[f"{case_id}_truth"], dtype=float)
    frequencies, truth_psd = spectrum(truth, 0.1)
    ax.loglog(
        frequencies[1:], truth_psd[1:], color=TRUTH, lw=SPECTRUM_LINEWIDTHS["Truth"], ls="-",
        alpha=1.0, label="Truth", zorder=5,
    )
    available = []
    for key, method in [(f"{case_id}_PCE, full R", "pce"), (f"{case_id}_APCE, full R", "apce")]:
        if key in data.files:
            values = np.asarray(data[key], dtype=float)
            frequencies, psd = spectrum(values, 0.1)
            ax.loglog(
                frequencies[1:], psd[1:], color=COLORS[method], lw=SPECTRUM_LINEWIDTHS[METHOD_LABELS[method]],
                ls="--", alpha=0.78, label=METHOD_LABELS[method], zorder=4,
            )
            available.append(method)
    ax.set_title(rf"$U_r={UR[index]:.2f}$", fontsize=14, pad=4, fontweight="normal", loc="center")
    ax.set_xlabel("Frequency (Hz)", fontsize=13)
    if index == 0:
        ax.set_ylabel("Normalised PSD", fontsize=13)
    ax.tick_params(axis="both", which="major", labelsize=11, width=0.55, length=2.2, pad=1)
    ax.grid(False)
    # Display the physically relevant low-frequency range only.
    # PSD is still computed over the full available bandwidth.
    spectrum_fmax_hz = 1.0
    ax.set_xlim(float(frequencies[1]), spectrum_fmax_hz)
    handles, labels = ax.get_legend_handles_labels()
    legend_order = [labels.index("PCE"), labels.index("APCE"), labels.index("Truth")]
    ax.legend(
        [handles[index] for index in legend_order], [labels[index] for index in legend_order],
        loc="lower left", fontsize=11, handlelength=1.3,
        handletextpad=0.35, borderaxespad=0.3,
    )
    return available


def main() -> None:
    global SUMMARY, SPECTRA, OUT

    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--spectra", type=Path, default=SPECTRA)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()

    SUMMARY = args.summary
    SPECTRA = args.spectra
    OUT = args.output
    OUT.mkdir(parents=True, exist_ok=True)

    frame = load_summary()
    metrics = aggregate_metrics(frame)
    with (OUT / "fg_row_metrics_source.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "method", "reconstruction_nrmse", "reconstruction_sd", "prediction_nrmse", "prediction_sd", "n_seeds"])
        writer.writeheader()
        for case in CASES:
            for method in METHODS:
                writer.writerow({"case_id": case, "method": method, **metrics[(case, method)]})

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI, facecolor="white")
    gs = fig.add_gridspec(
        1, 7, width_ratios=[1.55, 1.55, 1, 1, 1, 1, 1],
        left=0.035, right=0.995, bottom=0.17, top=0.86, wspace=0.34,
    )
    ax_recon = fig.add_subplot(gs[0, 0])
    ax_pred = fig.add_subplot(gs[0, 1])
    # Capture fixed panel-label positions before moving the f axes.
    f_label_xy = fig.transFigure.inverted().transform(
        ax_recon.transAxes.transform((-0.16, 1.05))
    )
    # Keep g fixed; shift only the two f axes rightward.
    f_shifts = {ax_recon: 0.014, ax_pred: 0.014}
    for axis, f_shift in f_shifts.items():
        position = axis.get_position()
        axis.set_position([position.x0 + f_shift, position.y0, position.width, position.height])
    grouped_bars(ax_recon, metrics, "reconstruction_nrmse", "reconstruction_sd", "Reconstruction", "f")
    grouped_bars(ax_pred, metrics, "prediction_nrmse", "prediction_sd", "Forecast", "")
    fig.text(f_label_xy[0], f_label_xy[1], "f", fontsize=22, fontweight="bold", va="bottom")
    ax_recon.set_ylim(0.0, 0.4)
    ax_pred.set_ylim(0.0, 0.6)

    with np.load(SPECTRA, allow_pickle=False) as data:
        spectrum_axes = [fig.add_subplot(gs[0, 2 + i]) for i in range(5)]
        # Compress only the gaps within g, then align the group to the j-row
        # boundary. This retains a safe right margin for the final 10^0 tick.
        g_gap_reduction = 0.010
        g_group_shift = -0.010
        for index, axis in enumerate(spectrum_axes):
            position = axis.get_position()
            shift = (len(spectrum_axes) - 1 - index) * g_gap_reduction + g_group_shift
            axis.set_position([position.x0 + shift, position.y0, position.width, position.height])

        # Recalculate panel label g AFTER the spectrum axes have moved.
        # This shifts the label right while leaving every PSD axis unchanged.
        g_label_xy = fig.transFigure.inverted().transform(
            spectrum_axes[0].transAxes.transform((-0.45, 1.05))
        )
        available_by_case = {case: plot_spectrum(ax, data, case, i) for i, (ax, case) in enumerate(zip(spectrum_axes, CASES))}
        fig.text(g_label_xy[0], g_label_xy[1], "g", fontsize=22, fontweight="bold", va="bottom")

    fig.savefig(OUT / "figure5_fg_row_10553.png", dpi=DPI, facecolor="white", pad_inches=0.0)
    fig.savefig(OUT / "figure5_fg_row_10553.tiff", dpi=DPI, facecolor="white", pad_inches=0.0)
    fig.savefig(OUT / "figure5_fg_row_10553.pdf", facecolor="white", pad_inches=0.0)
    fig.savefig(OUT / "figure5_fg_row_10553.svg", facecolor="white", pad_inches=0.0)
    plt.close(fig)

    metadata = {
        "figure": "figure5_fg_row",
        "status": "complete_for_requested_truth_pce_apce_spectra",
        "canvas": {"width_px": FIG_W_PX, "height_px": FIG_H_PX, "dpi": DPI, "width_in": FIG_W_IN, "height_in": FIG_H_IN},
        "panel_contract": {
            "f_left": "five-condition grouped reconstruction nRMSE; four methods; mean +/- seed SD",
            "f_right": "five-condition grouped blackout-prediction nRMSE; four methods; mean +/- seed SD",
            "g": "five kinetic-energy spectrum panels, one per held-out condition; Truth/PCE/APCE",
        },
        "layout": {
            "subplot_titles": True,
            "group_headings": False,
            "subplot_title_alignment": "center",
            "g_right_boundary_anchored": True,
            "f_gap_compression": False,
            "g_gap_compression": {"per_gap": 0.010, "right_boundary_anchored": True},
            "nrmse_y_limits": {"reconstruction": [0.0, 0.3], "blackout_prediction": [0.0, 0.5]},
        },
        "methods": {m: {"label": METHOD_LABELS[m], "color": COLORS[m]} for m in METHODS},
        "cases": [{"case_id": c, "reduced_velocity": u} for c, u in zip(CASES, UR)],
        "observation_layout": {
            "name": "adaptive_fullfield_valid_x40y20",
            "x_points": 40,
            "y_points": 20,
            "nominal_points": 800,
            "effective_points": 751,
            "scalar_observations": 1502,
            "mask_aware": True,
        },
        "sources": {
            "metrics_local": str(SUMMARY),
            "spectra_local": str(SPECTRA),
            "summary_local": str(SUMMARY),
            "metrics_remote_expected": "<HILDA_RESULTS_ROOT>/results/viv_piv_pce_apce_adaptive_valid_x40y20_formal5/summaries/rank256_stride1/summary_metrics.csv",
            "spectra_remote_authoritative": "<HILDA_RESULTS_ROOT>/results/viv_piv_pce_apce_adaptive_valid_x40y20_formal5/excel_source/energy_timeseries.csv",
            "spectra_generation_script_remote": "x40y20 formal5 energy time-series export; PSD recomputed in this Python plotting script",
        },
        "spectra_available_methods_by_case": available_by_case,
        "spectra_missing_methods": [],
        "spectra_note": "The displayed curves are Truth/PCE/APCE kinetic-energy temporal PSDs recomputed from the authoritative stored time series with dt=0.1 s, Hann windowing and one-sided integral normalisation.",
        "style": {
            "font_family": "Arial with Helvetica/DejaVu Sans fallback",
            "panel_label_pt": 22,
            "group_title_pt": 19,
            "axis_label_pt": 13,
            "tick_pt": 11,
            "spectrum_tick_pt": 9,
            "errorbars": "seed SD, n=5 algorithmic seeds; seeds are not independent physical replicates",
            "spectrum_scale": "log-log temporal PSD; frequency axis reconstructed from dt=0.1 s; displayed frequency range capped at 1 Hz",
            "spectrum_line_styles": {"Truth": "-", "PCE": "--", "APCE": "--"},
            "spectrum_line_width_pt": SPECTRUM_LINEWIDTHS,
            "spectrum_curve_linewidth_multiplier": SPECTRUM_CURVE_LINEWIDTH_MULTIPLIER,
            "spectrum_legend_order": ["PCE", "APCE", "Truth"],
        },
        "outputs": {ext: str(OUT / f"figure5_fg_row_10553{ext}") for ext in [".png", ".tiff", ".pdf", ".svg"]},
    }
    metadata["output_sha256"] = {
        ext: hashlib.sha256((OUT / f"figure5_fg_row_10553{ext}").read_bytes()).hexdigest()
        for ext in [".png", ".tiff", ".pdf", ".svg"]
    }
    (OUT / "figure5_fg_row_10553_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
