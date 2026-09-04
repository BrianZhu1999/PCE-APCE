"""Quantify APCE field, spectral and vortex-dynamics fidelity on five held-out VIV cases."""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
from collections import defaultdict

import matplotlib as mpl
import numpy as np
import torch
import torch.nn.functional as torch_f
from scipy.signal import welch

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .common import load_config
from .io import VIVCase, list_cases
from .rom import PODModel


CASES = ("0463", "0556", "0679", "0803", "1359")
UR = np.asarray([4.63, 5.56, 6.79, 8.03, 13.59])
CASE_LABELS = [f"{value:.2f}" for value in UR]
TRUTH_COLOR = "#262626"
APCE_COLOR = "#B64342"
CASE_COLORS = ["#4C78A8", "#59A14F", "#F28E2B", "#B279A2", "#E15759"]


def trace_path(root: pathlib.Path, case_id: str, seed: int = 0) -> pathlib.Path:
    return root / f"viv_{case_id}_apce_seed{seed:03d}_layout20x40_ens064_covfull.npz"


def run_path(root: pathlib.Path, case_id: str, seed: int) -> pathlib.Path:
    return root / f"viv_{case_id}_apce_seed{seed:03d}_layout20x40_ens064_covfull.json"


def update_pair(stats: dict[str, float], truth: torch.Tensor, prediction: torch.Tensor) -> None:
    x = truth.double().reshape(-1)
    y = prediction.double().reshape(-1)
    stats["n"] += int(x.numel())
    stats["error2"] += float(torch.sum((y - x) ** 2))
    stats["truth2"] += float(torch.sum(x**2))
    stats["sum_x"] += float(torch.sum(x))
    stats["sum_y"] += float(torch.sum(y))
    stats["sum_x2"] += float(torch.sum(x**2))
    stats["sum_y2"] += float(torch.sum(y**2))
    stats["sum_xy"] += float(torch.sum(x * y))


def finish_pair(stats: dict[str, float]) -> tuple[float, float]:
    n = max(int(stats["n"]), 1)
    nrmse = np.sqrt(stats["error2"] / max(stats["truth2"], 1e-30))
    covariance = stats["sum_xy"] - stats["sum_x"] * stats["sum_y"] / n
    variance_x = stats["sum_x2"] - stats["sum_x"] ** 2 / n
    variance_y = stats["sum_y2"] - stats["sum_y"] ** 2 / n
    correlation = covariance / np.sqrt(max(variance_x * variance_y, 1e-30))
    return float(nrmse), float(correlation)


def series_metrics(truth: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    nrmse = np.sqrt(np.sum((prediction - truth) ** 2) / max(np.sum(truth**2), 1e-30))
    correlation = np.corrcoef(truth, prediction)[0, 1]
    return float(nrmse), float(correlation)


def spectrum(values: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    frequency, power = welch(
        values - np.mean(values), fs=1.0 / dt, window="hann",
        nperseg=min(512, values.size), noverlap=min(256, values.size // 2),
        detrend="constant", scaling="density",
    )
    area = np.trapezoid(power, frequency)
    return frequency, power / max(float(area), 1e-30)


def dominant_frequency(frequency: np.ndarray, power: np.ndarray, low: float = 0.05, high: float = 2.5) -> float:
    keep = (frequency >= low) & (frequency <= high)
    if not np.any(keep):
        return float("nan")
    indices = np.flatnonzero(keep)
    peak = int(indices[np.argmax(power[indices])])
    if peak <= 0 or peak >= power.size - 1:
        return float(frequency[peak])
    local = np.log(np.maximum(power[peak - 1 : peak + 2], 1e-30))
    denominator = local[0] - 2.0 * local[1] + local[2]
    offset = 0.0 if abs(float(denominator)) < 1e-14 else 0.5 * float(local[0] - local[2]) / float(denominator)
    offset = float(np.clip(offset, -0.5, 0.5))
    return float(frequency[peak] + offset * (frequency[1] - frequency[0]))


def gaussian_kernel(sigma: float, radius: int, device: torch.device) -> torch.Tensor:
    coordinate = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    one_dimensional = torch.exp(-0.5 * (coordinate / sigma) ** 2)
    one_dimensional /= one_dimensional.sum()
    kernel = one_dimensional[:, None] * one_dimensional[None, :]
    return kernel[None, None, :, :]


def masked_gaussian(field: torch.Tensor, valid: torch.Tensor, kernel: torch.Tensor, radius: int) -> tuple[torch.Tensor, torch.Tensor]:
    weights = torch_f.conv2d(valid[:, None].float(), kernel, padding=radius)
    numerator = torch_f.conv2d(field[:, None] * valid[:, None].float(), kernel, padding=radius)
    return (numerator / weights.clamp_min(1e-6))[:, 0], weights[:, 0] > 0.999


def evaluate_case(
    case: VIVCase,
    pod: PODModel,
    latent: np.ndarray,
    device: torch.device,
    diameter_m: float,
    warmup_s: float,
    block: int = 16,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    basis = torch.as_tensor(pod.basis, dtype=torch.float32, device=device)
    mean = torch.as_tensor(pod.mean, dtype=torch.float32, device=device)
    height, width = case.y_mm.size, case.x_mm.size
    dx = float(np.median(np.diff(case.x_mm))) * 1e-3
    dy = float(np.median(np.diff(case.y_mm))) * 1e-3
    area = dx * dy
    x_over_d = torch.as_tensor(case.x_mm / (diameter_m * 1000.0), dtype=torch.float32, device=device)
    y_over_d = torch.as_tensor(case.y_mm / (diameter_m * 1000.0), dtype=torch.float32, device=device)
    roi = ((x_over_d[1:-1][None, :] >= 0.5) & (x_over_d[1:-1][None, :] <= 8.5)
           & (y_over_d[1:-1][:, None] >= -2.2) & (y_over_d[1:-1][:, None] <= 2.2))
    smoothing_sigma = 1.5
    smoothing_radius = 3
    smoothing_kernel = gaussian_kernel(smoothing_sigma, smoothing_radius, device)

    pair_stats = {
        key: defaultdict(float) for key in ("u", "v", "speed", "vorticity")
    }
    count = case.time_s.size
    enstrophy_truth = np.empty(count, dtype=np.float64)
    enstrophy_prediction = np.empty(count, dtype=np.float64)
    circulation_truth = np.empty(count, dtype=np.float64)
    circulation_prediction = np.empty(count, dtype=np.float64)
    abs_circulation_truth = np.empty(count, dtype=np.float64)
    abs_circulation_prediction = np.empty(count, dtype=np.float64)
    upstream_sum = 0.0
    upstream_count = 0

    with torch.inference_mode():
        for start, truth_flat, valid_flat in case.iter_physical(block=block):
            stop = start + truth_flat.shape[0]
            truth = torch.as_tensor(truth_flat, dtype=torch.float32, device=device).reshape(-1, height, width, 2)
            valid = torch.as_tensor(valid_flat, dtype=torch.bool, device=device).reshape(-1, height, width, 2)[..., 0]
            state = torch.as_tensor(latent[start:stop], dtype=torch.float32, device=device)
            prediction = (mean[None, :] + state @ basis.mT).reshape(-1, height, width, 2)

            for component, name in enumerate(("u", "v")):
                update_pair(pair_stats[name], truth[..., component][valid], prediction[..., component][valid])
            truth_speed = torch.linalg.vector_norm(truth, dim=-1)
            prediction_speed = torch.linalg.vector_norm(prediction, dim=-1)
            update_pair(pair_stats["speed"], truth_speed[valid], prediction_speed[valid])

            upstream = valid & (x_over_d[None, None, :] < -0.8)
            upstream_sum += float(torch.sum(truth[..., 0][upstream]))
            upstream_count += int(torch.sum(upstream))

            truth_u, truth_smooth_valid = masked_gaussian(truth[..., 0], valid, smoothing_kernel, smoothing_radius)
            truth_v, _ = masked_gaussian(truth[..., 1], valid, smoothing_kernel, smoothing_radius)
            prediction_u, prediction_smooth_valid = masked_gaussian(prediction[..., 0], valid, smoothing_kernel, smoothing_radius)
            prediction_v, _ = masked_gaussian(prediction[..., 1], valid, smoothing_kernel, smoothing_radius)
            truth_omega = (
                (truth_v[:, 1:-1, 2:] - truth_v[:, 1:-1, :-2]) / (2.0 * dx)
                - (truth_u[:, 2:, 1:-1] - truth_u[:, :-2, 1:-1]) / (2.0 * dy)
            )
            prediction_omega = (
                (prediction_v[:, 1:-1, 2:] - prediction_v[:, 1:-1, :-2]) / (2.0 * dx)
                - (prediction_u[:, 2:, 1:-1] - prediction_u[:, :-2, 1:-1]) / (2.0 * dy)
            )
            interior = (
                truth_smooth_valid[:, 1:-1, 1:-1] & prediction_smooth_valid[:, 1:-1, 1:-1]
                & roi[None, :, :]
            )
            update_pair(pair_stats["vorticity"], truth_omega[interior], prediction_omega[interior])
            interior_float = interior.float()
            denominator = interior_float.sum(dim=(1, 2)).clamp_min(1.0)
            enstrophy_truth[start:stop] = (0.5 * (truth_omega**2 * interior_float).sum(dim=(1, 2)) / denominator).cpu()
            enstrophy_prediction[start:stop] = (0.5 * (prediction_omega**2 * interior_float).sum(dim=(1, 2)) / denominator).cpu()
            circulation_truth[start:stop] = ((truth_omega * interior_float).sum(dim=(1, 2)) * area).cpu()
            circulation_prediction[start:stop] = ((prediction_omega * interior_float).sum(dim=(1, 2)) * area).cpu()
            abs_circulation_truth[start:stop] = ((truth_omega.abs() * interior_float).sum(dim=(1, 2)) * area).cpu()
            abs_circulation_prediction[start:stop] = ((prediction_omega.abs() * interior_float).sum(dim=(1, 2)) * area).cpu()

    warmup = int(np.searchsorted(case.time_s, warmup_s))
    sl = slice(warmup, None)
    metrics: dict[str, float] = {}
    for key, stats in pair_stats.items():
        metrics[f"{key}_nrmse"], metrics[f"{key}_correlation"] = finish_pair(stats)

    series = {
        "enstrophy_truth": enstrophy_truth,
        "enstrophy_prediction": enstrophy_prediction,
        "circulation_truth": circulation_truth,
        "circulation_prediction": circulation_prediction,
        "abs_circulation_truth": abs_circulation_truth,
        "abs_circulation_prediction": abs_circulation_prediction,
    }
    for key in ("enstrophy", "circulation", "abs_circulation"):
        metrics[f"{key}_nrmse"], metrics[f"{key}_correlation"] = series_metrics(
            series[f"{key}_truth"][sl], series[f"{key}_prediction"][sl]
        )
    u_inf = upstream_sum / max(upstream_count, 1)
    metrics["free_stream_velocity_mps"] = float(u_inf)

    dt = case.dt_s
    for key in ("circulation",):
        f_truth, p_truth = spectrum(series[f"{key}_truth"][sl], dt)
        f_prediction, p_prediction = spectrum(series[f"{key}_prediction"][sl], dt)
        peak_truth = dominant_frequency(f_truth, p_truth)
        peak_prediction = dominant_frequency(f_prediction, p_prediction)
        metrics["shedding_frequency_truth_hz"] = peak_truth
        metrics["shedding_frequency_prediction_hz"] = peak_prediction
        metrics["strouhal_truth"] = peak_truth * diameter_m / max(abs(u_inf), 1e-12)
        metrics["strouhal_prediction"] = peak_prediction * diameter_m / max(abs(u_inf), 1e-12)
        metrics["strouhal_relative_error"] = abs(peak_prediction - peak_truth) / max(abs(peak_truth), 1e-12)
        metrics["circulation_spectral_l1"] = float(np.trapezoid(np.abs(p_prediction - p_truth), f_truth))
        series["circulation_frequency"] = f_truth
        series["circulation_psd_truth"] = p_truth
        series["circulation_psd_prediction"] = p_prediction

    displacement = case.cyl_displ_m[sl]
    f_displacement, p_displacement = spectrum(displacement, dt)
    cylinder_frequency = dominant_frequency(f_displacement, p_displacement)
    metrics["cylinder_frequency_hz"] = cylinder_frequency
    metrics["lockin_ratio_truth"] = metrics["shedding_frequency_truth_hz"] / max(cylinder_frequency, 1e-12)
    metrics["lockin_ratio_prediction"] = metrics["shedding_frequency_prediction_hz"] / max(cylinder_frequency, 1e-12)
    series["displacement_frequency"] = f_displacement
    series["displacement_psd"] = p_displacement
    return metrics, series


def add_heatmap(ax: plt.Axes, values: np.ndarray, columns: list[str], title: str, vmin: float, vmax: float, cmap: str) -> None:
    image = ax.imshow(values, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(np.arange(len(columns)), columns)
    ax.set_yticks(np.arange(len(CASE_LABELS)), CASE_LABELS)
    ax.set_ylabel(r"$U_r$")
    ax.set_title(title, loc="left")
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            color = "white" if values[row, col] > vmin + 0.62 * (vmax - vmin) else "#202020"
            ax.text(col, row, f"{values[row, col]:.2f}", ha="center", va="center", fontsize=5.7, color=color)
    cb = ax.figure.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
    cb.ax.tick_params(labelsize=5.5, length=2)


def save_figure(fig: plt.Figure, stem: pathlib.Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--variant", default="rank256_stride1")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()

    config = load_config(args.config)
    root = pathlib.Path(config["output_root"])
    model_root = root / "models" / args.variant
    trace_root = root / "runs" / args.variant / "traces"
    run_root = root / "runs" / args.variant / "runs"
    cases = list_cases(pathlib.Path(config["data_root"]))
    pod = PODModel.load(model_root / "pod_model.npz")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[dict[str, float | str]] = []
    series_by_case: dict[str, dict[str, np.ndarray]] = {}
    traces: dict[str, dict[str, np.ndarray]] = {}
    for case_id in CASES:
        case = VIVCase.open(cases[case_id])
        with np.load(trace_path(trace_root, case_id), allow_pickle=False) as trace:
            latent = np.asarray(trace["latent_estimate"], dtype=np.float32)
            truth_energy = np.asarray(trace["truth_energy"], dtype=np.float64)
            predicted_energy = np.asarray(trace["predicted_energy"], dtype=np.float64)
            time_s = np.asarray(trace["time_s"], dtype=np.float64)
        metrics, series = evaluate_case(
            case, pod, latent, device, float(config["cylinder_diameter_m"]), float(config["warmup_seconds"])
        )
        warmup = int(np.searchsorted(time_s, float(config["warmup_seconds"])))
        metrics["energy_nrmse"], metrics["energy_correlation"] = series_metrics(
            truth_energy[warmup:], predicted_energy[warmup:]
        )
        energy_f, energy_truth_psd = spectrum(truth_energy[warmup:], case.dt_s)
        _, energy_prediction_psd = spectrum(predicted_energy[warmup:], case.dt_s)
        metrics["energy_spectral_l1"] = float(np.trapezoid(np.abs(energy_prediction_psd - energy_truth_psd), energy_f))
        series.update({
            "time_s": time_s,
            "truth_energy": truth_energy,
            "predicted_energy": predicted_energy,
            "energy_frequency": energy_f,
            "energy_psd_truth": energy_truth_psd,
            "energy_psd_prediction": energy_prediction_psd,
        })

        run_payloads = [json.loads(run_path(run_root, case_id, seed).read_text(encoding="utf-8")) for seed in range(3)]
        for field in ("normalized_crps", "coverage_90", "normalized_interval_width_90", "blackout_mean_nrmse"):
            values = np.asarray([payload[field] for payload in run_payloads], dtype=float)
            metrics[field] = float(np.mean(values))
            metrics[field + "_sd"] = float(np.std(values, ddof=1))
        blackout = defaultdict(list)
        for seed in range(3):
            with np.load(trace_path(trace_root, case_id, seed), allow_pickle=False) as trace:
                rows = json.loads(str(trace["blackout_rows_json"].item()))
            for row in rows:
                blackout[float(row["horizon_s"])].append(float(row["evaluation_nrmse"]))
        horizons = np.asarray(sorted(blackout), dtype=float)
        series["blackout_horizons_s"] = horizons
        series["blackout_nrmse_mean"] = np.asarray([np.mean(blackout[h]) for h in horizons])
        series["blackout_nrmse_sd"] = np.asarray([np.std(blackout[h], ddof=1) for h in horizons])

        metrics_rows.append({"case_id": case_id, "reduced_velocity": int(case_id) / 100.0, **metrics})
        series_by_case[case_id] = series

    fieldnames = list(metrics_rows[0].keys())
    with (args.output / "fig08_five_cases_quantitative_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)

    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
        "font.size": 7.0, "axes.titlesize": 7.4, "axes.labelsize": 7.0,
        "xtick.labelsize": 6.0, "ytick.labelsize": 6.0,
        "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.7,
        "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
    })

    nrmse_columns = ["u", "v", r"$|\mathbf{v}|$", r"$\omega$"]
    nrmse_values = np.asarray([[row[f"{key}_nrmse"] for key in ("u", "v", "speed", "vorticity")] for row in metrics_rows])
    corr_values = np.asarray([[row[f"{key}_correlation"] for key in ("u", "v", "speed", "vorticity")] for row in metrics_rows])
    physics_values = np.asarray([[row[f"{key}_nrmse"] for key in ("energy", "enstrophy", "abs_circulation")] for row in metrics_rows])
    probabilistic_values = np.asarray([[row[key] for key in ("normalized_crps", "coverage_90", "normalized_interval_width_90")] for row in metrics_rows])

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.75), gridspec_kw={"wspace": 0.44, "hspace": 0.48})
    add_heatmap(axes[0, 0], nrmse_values, nrmse_columns, "field nRMSE", 0.0, max(0.65, float(nrmse_values.max())), "magma_r")
    add_heatmap(axes[0, 1], corr_values, nrmse_columns, "field Pearson correlation", 0.0, 1.0, "viridis")
    add_heatmap(axes[0, 2], physics_values, [r"$E$", r"$Z_\omega$", r"$|\Gamma|$"], "physics-series nRMSE", 0.0,
                max(0.65, float(physics_values.max())), "magma_r")
    add_heatmap(axes[1, 0], probabilistic_values, ["nCRPS", "90% cov.", "90% width"], "probabilistic calibration", 0.0,
                max(1.6, float(probabilistic_values.max())), "cividis")

    ax = axes[1, 1]
    all_blackout = []
    for color, row in zip(CASE_COLORS, metrics_rows):
        case_id = str(row["case_id"])
        series = series_by_case[case_id]
        ax.plot(series["blackout_horizons_s"], series["blackout_nrmse_mean"], marker="o", ms=3,
                lw=0.9, color=color, label=rf"$U_r={float(row['reduced_velocity']):.2f}$")
        all_blackout.append(series["blackout_nrmse_mean"])
    all_blackout = np.asarray(all_blackout)
    ax.plot(series_by_case[CASES[0]]["blackout_horizons_s"], all_blackout.mean(axis=0), color="#111111",
            marker="s", ms=3.2, lw=1.4, label="mean")
    ax.set_xlabel("blackout horizon (s)")
    ax.set_ylabel("nRMSE")
    ax.set_title("observation-blackout forecast", loc="left")
    ax.grid(axis="y", color="#D9D9D9", lw=0.4)
    ax.legend(fontsize=5.2, ncol=2, loc="upper left")

    ax = axes[1, 2]
    truth_st = np.asarray([row["strouhal_truth"] for row in metrics_rows], dtype=float)
    pred_st = np.asarray([row["strouhal_prediction"] for row in metrics_rows], dtype=float)
    lower = min(float(truth_st.min()), float(pred_st.min())) * 0.92
    upper = max(float(truth_st.max()), float(pred_st.max())) * 1.08
    ax.plot([lower, upper], [lower, upper], color="#777777", lw=0.8, ls="--")
    label_offsets = {
        "5.56": (-18, -7),
        "13.59": (4, 5),
    }
    for color, label, x, y in zip(CASE_COLORS, CASE_LABELS, truth_st, pred_st):
        ax.scatter(x, y, s=24, color=color, edgecolor="white", linewidth=0.4, zorder=3)
        ax.annotate(label, (x, y), xytext=label_offsets.get(label, (3, 2)),
                    textcoords="offset points", fontsize=5.5)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel(r"measured $St$")
    ax.set_ylabel(r"APCE $St$")
    ax.set_title("vortex-shedding frequency", loc="left")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#E1E1E1", lw=0.4)
    for index, axis in enumerate(axes.ravel()):
        axis.text(-0.14, 1.04, chr(ord("a") + index), transform=axis.transAxes,
                  fontsize=8, fontweight="bold", va="bottom")
    save_figure(fig, args.output / "fig08_five_cases_quantitative_summary")

    fig, axes = plt.subplots(5, 2, figsize=(7.2, 8.0), gridspec_kw={"wspace": 0.30, "hspace": 0.38})
    for row, (case_id, label) in enumerate(zip(CASES, CASE_LABELS)):
        series = series_by_case[case_id]
        for col, prefix in enumerate(("energy", "circulation")):
            ax = axes[row, col]
            frequency = series[f"{prefix}_frequency"]
            keep = (frequency > 0) & (frequency <= 2.5)
            ax.semilogy(frequency[keep], series[f"{prefix}_psd_truth"][keep], color=TRUTH_COLOR, lw=1.0, label="measured")
            ax.semilogy(frequency[keep], series[f"{prefix}_psd_prediction"][keep], color=APCE_COLOR, lw=0.9, label="APCE")
            if col == 0:
                ax.set_ylabel(rf"$U_r={label}$" + "\nnormalised PSD")
            else:
                ax.set_ylabel("normalised PSD")
            if row == 0:
                ax.set_title("kinetic-energy spectrum" if col == 0 else "wake-circulation spectrum")
            if row == 4:
                ax.set_xlabel("frequency (Hz)")
            ax.set_xlim(0.0, 2.5)
            ax.grid(axis="y", color="#D9D9D9", lw=0.4)
    axes[0, 0].legend(fontsize=5.8, ncol=2, loc="upper right")
    save_figure(fig, args.output / "fig09_five_cases_energy_circulation_spectra")

    fig, axes = plt.subplots(5, 2, figsize=(7.2, 8.0), gridspec_kw={"wspace": 0.30, "hspace": 0.38})
    for row, (case_id, label) in enumerate(zip(CASES, CASE_LABELS)):
        series = series_by_case[case_id]
        warmup = int(np.searchsorted(series["time_s"], float(config["warmup_seconds"])))
        time_s = series["time_s"][warmup:] - series["time_s"][warmup]
        pairs = ((series["truth_energy"][warmup:], series["predicted_energy"][warmup:], r"$E(t)$"),
                 (series["circulation_truth"][warmup:], series["circulation_prediction"][warmup:], r"$\Gamma(t)$"))
        for col, (truth, prediction, ylabel) in enumerate(pairs):
            ax = axes[row, col]
            ax.plot(time_s, truth, color=TRUTH_COLOR, lw=0.75, label="measured")
            ax.plot(time_s, prediction, color=APCE_COLOR, lw=0.65, alpha=0.9, label="APCE")
            ax.set_ylabel((rf"$U_r={label}$" + "\n" if col == 0 else "") + ylabel)
            if row == 0:
                ax.set_title("kinetic energy" if col == 0 else "signed wake circulation")
            if row == 4:
                ax.set_xlabel("time after warm-up (s)")
            ax.grid(axis="y", color="#D9D9D9", lw=0.4)
    axes[0, 0].legend(fontsize=5.8, ncol=2, loc="upper right")
    save_figure(fig, args.output / "fig10_five_cases_energy_circulation_timeseries")

    source = {}
    for case_id, series in series_by_case.items():
        for key, value in series.items():
            source[f"{case_id}_{key}"] = np.asarray(value)
    np.savez_compressed(args.output / "fig08_10_five_cases_quantitative_source.npz", **source)
    summary = {
        "method": "APCE", "variant": args.variant, "field_seed": 0,
        "probabilistic_seeds": [0, 1, 2], "cases": metrics_rows,
        "vorticity_roi": {"x_over_d": [0.5, 8.5], "y_over_d": [-2.2, 2.2]},
        "vorticity_filter": {"type": "mask-normalized Gaussian", "sigma_grid_points": 1.5, "radius_grid_points": 3},
        "force_coefficients_reported": False,
        "force_coefficients_reason": "No independent pressure, wall-shear or force-sensor truth is present in the NPZ archives.",
        "device": str(device),
    }
    (args.output / "fig08_10_five_cases_quantitative_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({"outputs": ["fig08", "fig09", "fig10"], "device": str(device)}, indent=2))


if __name__ == "__main__":
    main()
