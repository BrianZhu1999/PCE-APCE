"""Nature-style composite figure for the 14.82 pseudo-held-out VIV-PIV case.

The figure is an image-plate-plus-quantification composite.  It uses the
strict leave-one-training-case-out fold and the APCE trace produced by
``calibration_worker``.  The complete PIV field is used only for sealed
post-hoc visualization and metric calculation.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import matplotlib as mpl
import numpy as np
import torch
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import welch

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Required editable-text settings for Nature-style SVG/PDF output.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update({
    "font.size": 7.0,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.7,
    "axes.labelsize": 7.0,
    "axes.titlesize": 7.4,
    "xtick.labelsize": 6.2,
    "ytick.labelsize": 6.2,
    "legend.frameon": False,
})

from .common import load_config, write_json  # noqa: E402
from .io import VIVCase, list_cases  # noqa: E402
from .rom import PODModel  # noqa: E402


def add_label(ax: plt.Axes, label: str, *, x: float = -0.12) -> None:
    ax.text(x, 1.03, label, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom", ha="left")


def _energy_spectrum(values: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(values, dtype=np.float64)
    frequency, power = welch(
        signal - signal.mean(), fs=1.0 / dt, window="hann",
        nperseg=min(256, signal.size), noverlap=min(128, signal.size // 2),
        detrend="constant", scaling="density",
    )
    area = np.trapezoid(power, frequency)
    return frequency, power / max(float(area), 1e-30)


def _field_panel(
    ax: plt.Axes,
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    norm,
    cmap: str,
    title: str,
    displacement_over_d: float,
    *,
    add_sensors: bool = False,
    sensor_xy: np.ndarray | None = None,
):
    masked = np.where(valid, field, np.nan)
    mesh = ax.pcolormesh(x, y, masked, shading="auto", cmap=cmap, norm=norm, rasterized=True)
    ax.add_patch(Circle((0.0, displacement_over_d), 0.5, fill=False,
                        ec="#262626", lw=0.65, zorder=5))
    if add_sensors and sensor_xy is not None:
        ax.scatter(sensor_xy[:, 0], sensor_xy[:, 1], s=1.2, c="white",
                   edgecolors="#202020", linewidths=0.15, alpha=0.75, zorder=6,
                   rasterized=True)
        ax.text(0.98, 0.03, "20 x 40 sensors", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=5.8, color="#202020",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5})
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(float(y.min()), float(y.max()))
    ax.set_xlabel(r"$x/D$")
    ax.set_ylabel(r"$y/D$")
    ax.set_title(title, loc="left", pad=2.0)
    ax.tick_params(length=2.2, width=0.6)
    return mesh


def _row_norm(truth: np.ndarray, prediction: np.ndarray, valid: np.ndarray) -> TwoSlopeNorm:
    values = np.concatenate([truth[valid], prediction[valid]])
    scale = max(abs(float(np.nanpercentile(values, 1))),
                abs(float(np.nanpercentile(values, 99))), 1e-8)
    return TwoSlopeNorm(vcenter=0.0, vmin=-scale, vmax=scale)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the 14.82 VIV-PIV APCE Nature-style case figure.")
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--trace", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = load_config(args.config)
    case_id = "1482"
    model_root = pathlib.Path(config["output_root"]) / "models" / args.variant
    case = VIVCase.open(list_cases(pathlib.Path(config["data_root"]))[case_id])
    pod = PODModel.load(model_root / "pod_model.npz")
    sensor_archive = np.load(model_root / "sensor_layouts" / "20x40" / f"case_{case_id}.npz", allow_pickle=False)
    sensor_xy = np.asarray(sensor_archive["sensor_coordinates_mm"], dtype=np.float64)
    diameter_mm = float(config["cylinder_diameter_m"]) * 1000.0
    sensor_xy = sensor_xy / diameter_mm

    with np.load(args.trace, allow_pickle=False) as trace:
        latent = np.asarray(trace["latent_estimate"], dtype=np.float64)
        if "normalized_crps" not in trace.files or "crps_step" not in trace.files:
            raise ValueError("Trace must contain per-time normalized_crps and crps_step arrays")
        normalized_crps = np.asarray(trace["normalized_crps"], dtype=np.float64)
        crps_step = np.asarray(trace["crps_step"], dtype=np.int64)
        crps_evaluation_dimensions = int(trace["evaluation_truth"].shape[1])
    if latent.shape[0] != case.time_s.size:
        raise ValueError(f"Trace length {latent.shape[0]} does not match case length {case.time_s.size}")
    if normalized_crps.shape != crps_step.shape or np.any(crps_step < 0) or np.any(crps_step >= case.time_s.size):
        raise ValueError("Invalid per-time CRPS trace")

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    basis = torch.as_tensor(pod.basis, dtype=torch.float32, device=device)
    mean = torch.as_tensor(pod.mean, dtype=torch.float32, device=device)
    truth_energy = np.empty(case.time_s.size, dtype=np.float64)
    predicted_energy = np.empty(case.time_s.size, dtype=np.float64)
    frame_nrmse = np.empty(case.time_s.size, dtype=np.float64)
    truth_latent = np.empty_like(latent)
    error_square = 0.0
    truth_square = 0.0

    # Decode blocks once for sealed kinetic-energy diagnostics.  The full field
    # never enters the APCE update or candidate scoring.
    with torch.inference_mode():
        for start, values, valid in case.iter_physical(block=16):
            stop = start + values.shape[0]
            truth_t = torch.as_tensor(values, dtype=torch.float32, device=device)
            valid_t = torch.as_tensor(valid, dtype=torch.bool, device=device)
            latent_t = torch.as_tensor(latent[start:stop], dtype=torch.float32, device=device)
            prediction_t = mean[None, :] + latent_t @ basis.mT
            truth_latent[start:stop] = (
                torch.where(valid_t, truth_t - mean[None, :], torch.zeros_like(truth_t)) @ basis
            ).cpu().numpy()
            truth_field = truth_t.reshape(truth_t.shape[0], -1, 2)
            prediction_field = prediction_t.reshape(prediction_t.shape[0], -1, 2)
            valid_pixel = valid_t.reshape(valid_t.shape[0], -1, 2)[..., 0]
            truth_energy[start:stop] = (
                0.5 * torch.sum(torch.sum(truth_field.square(), dim=2) * valid_pixel, dim=1)
                / valid_pixel.sum(dim=1).clamp_min(1)
            ).cpu().numpy()
            predicted_energy[start:stop] = (
                0.5 * torch.sum(torch.sum(prediction_field.square(), dim=2) * valid_pixel, dim=1)
                / valid_pixel.sum(dim=1).clamp_min(1)
            ).cpu().numpy()
            residual = prediction_t - truth_t
            frame_error = torch.sum(residual.square() * valid_t, dim=1)
            frame_truth = torch.sum(truth_t.square() * valid_t, dim=1)
            frame_nrmse[start:stop] = torch.sqrt(
                frame_error / frame_truth.clamp_min(1e-30)
            ).cpu().numpy()
            error_square += float(torch.sum(residual[valid_t] ** 2))
            truth_square += float(torch.sum(truth_t[valid_t] ** 2))

    warmup = int(round(float(config["warmup_seconds"]) / float(config["time_step_s"])))
    centred_displacement = case.cyl_displ_m - np.median(case.cyl_displ_m[warmup:])
    frame = warmup + int(np.argmax(np.abs(centred_displacement[warmup:])))
    actual_time = float(case.time_s[frame])
    truth, valid = case.physical_frames(frame, frame + 1)
    truth = np.asarray(truth[0], dtype=np.float64)
    valid = np.asarray(valid[0], dtype=bool)
    with torch.inference_mode():
        state = torch.as_tensor(latent[frame], dtype=torch.float32, device=device)
        prediction = (mean + state @ basis.mT).cpu().numpy().reshape(truth.shape).astype(np.float64)
    u_truth, v_truth = truth[..., 0], truth[..., 1]
    u_prediction, v_prediction = prediction[..., 0], prediction[..., 1]
    speed_truth = np.sqrt(u_truth**2 + v_truth**2)
    speed_prediction = np.sqrt(u_prediction**2 + v_prediction**2)
    u_error = np.abs(u_prediction - u_truth)
    v_error = np.abs(v_prediction - v_truth)
    speed_error = np.abs(speed_prediction - speed_truth)
    x = case.x_mm / diameter_mm
    y = case.y_mm / diameter_mm
    displacement_over_d = float(case.cyl_displ_m[frame] / float(config["cylinder_diameter_m"]))

    row_data = [
        (u_truth, u_prediction, u_error, "u", "RdBu_r"),
        (v_truth, v_prediction, v_error, "v", "RdBu_r"),
        (speed_truth, speed_prediction, speed_error, r"$|V|$", "viridis"),
    ]
    fig = plt.figure(figsize=(7.25, 5.2), facecolor="white")
    outer = fig.add_gridspec(2, 1, height_ratios=[3.0, 0.9], hspace=0.24)
    field_gs = outer[0].subgridspec(3, 3, hspace=0.0, wspace=0.32)
    metric_gs = outer[1].subgridspec(1, 4, wspace=0.70)
    axes = np.asarray([[fig.add_subplot(field_gs[row, col]) for col in range(3)] for row in range(3)])
    for row, (truth_field, predicted_field, error_field, component, cmap) in enumerate(row_data):
        if row < 2:
            physical_norm = _row_norm(truth_field, predicted_field, valid)
            error_norm = Normalize(vmin=0.0, vmax=max(float(np.nanpercentile(error_field[valid], 99)), 1e-8))
        else:
            values = np.concatenate([truth_field[valid], predicted_field[valid]])
            physical_norm = Normalize(vmin=0.0, vmax=max(float(np.nanpercentile(values, 99)), 1e-8))
            error_norm = Normalize(vmin=0.0, vmax=max(float(np.nanpercentile(error_field[valid], 99)), 1e-8))
        meshes = []
        meshes.append(_field_panel(axes[row, 0], truth_field, x, y, valid, physical_norm, cmap,
                                   f"truth {component}", displacement_over_d))
        meshes.append(_field_panel(axes[row, 1], predicted_field, x, y, valid, physical_norm, cmap,
                                   f"APCE reconstruction {component}", displacement_over_d,
                                   add_sensors=False, sensor_xy=sensor_xy))
        meshes.append(_field_panel(axes[row, 2], error_field, x, y, valid, error_norm, "magma",
                                   f"absolute error {component}", displacement_over_d))
        axes[row, 1].set_ylabel("")
        axes[row, 2].set_ylabel("")
        if row < 2:
            for axis in axes[row, :]:
                axis.set_xlabel("")
                axis.tick_params(labelbottom=False)
        add_label(axes[row, 0], chr(ord("a") + row * 3))
        add_label(axes[row, 1], chr(ord("a") + row * 3 + 1))
        add_label(axes[row, 2], chr(ord("a") + row * 3 + 2))
        physical_cax = make_axes_locatable(axes[row, 1]).append_axes("right", size="3.5%", pad=0.045)
        cb = fig.colorbar(meshes[1], cax=physical_cax)
        cb.ax.tick_params(labelsize=5.5, length=2)
        cb.set_label("m s$^{-1}$", fontsize=5.8)
        error_cax = make_axes_locatable(axes[row, 2]).append_axes("right", size="3.5%", pad=0.045)
        cb_err = fig.colorbar(meshes[2], cax=error_cax)
        cb_err.ax.tick_params(labelsize=5.5, length=2)
        cb_err.set_label("absolute error", fontsize=5.8)

    ax_nrmse = fig.add_subplot(metric_gs[0, 0])
    ax_crps = fig.add_subplot(metric_gs[0, 1])
    ax_pod_spectrum = fig.add_subplot(metric_gs[0, 2])
    ax_energy_spectrum = fig.add_subplot(metric_gs[0, 3])
    t = case.time_s
    ax_nrmse.plot(t, frame_nrmse, color="#B64342", lw=0.8)
    ax_nrmse.axhline(float(np.mean(frame_nrmse[warmup:])), color="#767676", lw=0.7, ls=":")
    ax_nrmse.axvline(actual_time, color="#767676", lw=0.7, ls="--")
    ax_nrmse.set_xlabel("time (s)")
    ax_nrmse.set_ylabel("frame nRMSE")
    ax_nrmse.set_title("full-field nRMSE", loc="left", pad=2)
    ax_nrmse.text(0.98, 0.96, f"mean {np.mean(frame_nrmse[warmup:]):.3f}",
                  transform=ax_nrmse.transAxes, ha="right", va="top", fontsize=5.5)
    ax_nrmse.grid(axis="y", color="#D9D9D9", lw=0.4, alpha=0.7)
    ax_nrmse.set_ylim(0.0, 0.4)
    add_label(ax_nrmse, "j")

    crps_time = t[crps_step]
    ax_crps.plot(crps_time, normalized_crps, color="#B64342", lw=0.8)
    crps_after_warmup = normalized_crps[crps_step >= warmup]
    mean_crps = float(np.mean(crps_after_warmup))
    ax_crps.axhline(mean_crps, color="#767676", lw=0.7, ls=":")
    ax_crps.axvline(actual_time, color="#767676", lw=0.7, ls="--")
    ax_crps.set_xlabel("time (s)")
    ax_crps.set_ylabel("normalised CRPS")
    ax_crps.set_title("held-out-point CRPS", loc="left", pad=2)
    ax_crps.text(0.98, 0.96, f"mean {mean_crps:.3f}", transform=ax_crps.transAxes,
                 ha="right", va="top", fontsize=5.5)
    ax_crps.grid(axis="y", color="#D9D9D9", lw=0.4, alpha=0.7)
    ax_crps.set_ylim(bottom=0.0)
    add_label(ax_crps, "k")

    mode_frequency_truth, mode_power_truth = _energy_spectrum(truth_latent[:, 0], float(case.dt_s))
    mode_frequency_prediction, mode_power_prediction = _energy_spectrum(latent[:, 0], float(case.dt_s))
    keep_mode_truth = (mode_frequency_truth > 0) & (mode_frequency_truth <= 5.0)
    keep_mode_prediction = (mode_frequency_prediction > 0) & (mode_frequency_prediction <= 5.0)
    ax_pod_spectrum.loglog(mode_frequency_truth[keep_mode_truth], mode_power_truth[keep_mode_truth],
                           color="#222222", lw=0.9, label="truth")
    ax_pod_spectrum.loglog(mode_frequency_prediction[keep_mode_prediction], mode_power_prediction[keep_mode_prediction],
                           color="#B64342", lw=0.85, label="APCE")
    ax_pod_spectrum.set_xlabel("frequency (Hz)")
    ax_pod_spectrum.set_ylabel("normalised PSD")
    ax_pod_spectrum.set_title("POD mode-1 PSD", loc="left", pad=2)
    ax_pod_spectrum.set_xlim(0.01, 5.0)
    ax_pod_spectrum.grid(True, which="major", color="#D9D9D9", lw=0.4, alpha=0.7)
    ax_pod_spectrum.legend(fontsize=5.5, loc="lower left")
    add_label(ax_pod_spectrum, "l")

    frequency_truth, power_truth = _energy_spectrum(truth_energy, float(case.dt_s))
    frequency_prediction, power_prediction = _energy_spectrum(predicted_energy, float(case.dt_s))
    keep_truth = (frequency_truth > 0) & (frequency_truth <= 5.0)
    keep_prediction = (frequency_prediction > 0) & (frequency_prediction <= 5.0)
    ax_energy_spectrum.loglog(frequency_truth[keep_truth], power_truth[keep_truth], color="#222222", lw=0.9)
    ax_energy_spectrum.loglog(frequency_prediction[keep_prediction], power_prediction[keep_prediction], color="#B64342", lw=0.85)
    ax_energy_spectrum.set_xlabel("frequency (Hz)")
    ax_energy_spectrum.set_ylabel("normalised PSD")
    ax_energy_spectrum.set_title("kinetic-energy PSD", loc="left", pad=2)
    ax_energy_spectrum.set_xlim(0.01, 5.0)
    ax_energy_spectrum.grid(True, which="major", color="#D9D9D9", lw=0.4, alpha=0.7)
    add_label(ax_energy_spectrum, "m", x=-0.17)

    physical_nrmse = float(np.sqrt(error_square / max(truth_square, 1e-30)))
    energy_corr = float(np.corrcoef(truth_energy, predicted_energy)[0, 1])
    spectral_distance = float(np.trapezoid(np.abs(power_truth - power_prediction), frequency_truth))
    fig.subplots_adjust(top=0.985, bottom=0.105)
    fig.text(
        0.5, 0.006,
        rf"Training-only diagnostic fold | 20 x 40 spatial points (800 locations; 1,600 scalar u/v observations) | "
        rf"field nRMSE={physical_nrmse:.3f}, energy Pearson r={energy_corr:.3f}, spectrum L1={spectral_distance:.3f}",
        ha="center", va="bottom", fontsize=6.2,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / "nature_case1482_apce_uv_speed_energy"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    np.savez_compressed(
        args.output / "nature_case1482_apce_uv_speed_energy_source.npz",
        truth=truth, prediction=prediction, valid=valid, x_over_d=x, y_over_d=y,
        sensor_xy_over_d=sensor_xy, frame=np.asarray(frame), time_s=np.asarray(actual_time),
        truth_energy=truth_energy, predicted_energy=predicted_energy, frame_nrmse=frame_nrmse,
        normalized_crps=normalized_crps, crps_step=crps_step, crps_time_s=crps_time,
        truth_pod_mode1=truth_latent[:, 0], predicted_pod_mode1=latent[:, 0],
        mode_frequency_truth=mode_frequency_truth, mode_power_truth=mode_power_truth,
        mode_frequency_prediction=mode_frequency_prediction, mode_power_prediction=mode_power_prediction,
        frequency_truth=frequency_truth, power_truth=power_truth,
        frequency_prediction=frequency_prediction, power_prediction=power_prediction,
    )
    metadata: dict[str, Any] = {
        "case_id": case_id,
        "reduced_velocity": 14.82,
        "variant": args.variant,
        "method": "APCE",
        "figure_archetype": "image plate + quantitative validation",
        "representative_frame_rule": "maximum absolute cylinder displacement after 2 s warmup",
        "frame": frame,
        "time_s": actual_time,
        "sensor_layout": "20x40",
        "sensor_locations": int(sensor_xy.shape[0]),
        "scalar_observations": int(2 * sensor_xy.shape[0]),
        "full_grid": [int(case.y_mm.size), int(case.x_mm.size)],
        "physical_field_nrmse": physical_nrmse,
        "kinetic_energy_pearson_r": energy_corr,
        "frame_nrmse_mean_after_warmup": float(np.mean(frame_nrmse[warmup:])),
        "normalized_crps_mean_after_warmup": mean_crps,
        "crps_evaluation_dimensions": crps_evaluation_dimensions,
        "crps_frames": int(normalized_crps.size),
        "kinetic_energy_spectrum_l1": spectral_distance,
        "outputs": [str(stem.with_suffix(ext)) for ext in (".png", ".svg", ".pdf", ".tiff")],
    }
    write_json(args.output / "nature_case1482_apce_uv_speed_energy_metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
