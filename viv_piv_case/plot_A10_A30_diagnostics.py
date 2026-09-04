"""Generate independent A10--A30 wake-diagnostic figures for one VIV-PIV case."""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
from collections.abc import Iterable

import matplotlib as mpl
import numpy as np
import torch
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Circle
from scipy.ndimage import binary_erosion, gaussian_filter

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .common import load_config
from .io import VIVCase, list_cases
from .rom import PODModel


FIGURE_SPECS = {
    "A10": "vector_direction_error",
    "A11": "mean_u",
    "A12": "mean_v",
    "A13": "velocity_fluctuation_rms",
    "A14": "two_component_fluctuation_energy",
    "A15": "reynolds_shear_stress",
    "A16": "streamwise_normal_stress",
    "A17": "transverse_normal_stress",
    "A18": "vorticity",
    "A19": "vorticity_error",
    "A20": "strain_rate_magnitude",
    "A21": "q_criterion",
    "A22": "swirling_strength",
    "A23": "divergence",
    "A24": "recirculation_length",
    "A25": "centerline_velocity_deficit",
    "A26": "wake_width",
    "A27": "momentum_thickness",
    "A28": "circulation",
    "A29": "enstrophy",
    "A30": "wake_symmetry_index",
}


def _trace_path(root: pathlib.Path, case_id: str, method: str, seed: int, layout: str) -> pathlib.Path:
    return root / f"viv_{case_id}_{method}_seed{seed:03d}_layout{layout}_ens064_covfull.npz"


def _finite_percentile(values: Iterable[np.ndarray], percentile: float, floor: float = 1e-8) -> float:
    flat = np.concatenate([np.asarray(value)[np.isfinite(value)].ravel() for value in values])
    if flat.size == 0:
        return floor
    return max(float(np.percentile(np.abs(flat), percentile)), floor)


def _masked_smooth(field: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    """Spatial Gaussian smoothing normalized by the valid-data weights."""
    weights = gaussian_filter(valid.astype(np.float32), sigma=(0.0, sigma, sigma), mode="nearest")
    smoothed = np.empty_like(field, dtype=np.float32)
    for component in range(2):
        numerator = gaussian_filter(
            np.where(valid, field[..., component], 0.0),
            sigma=(0.0, sigma, sigma),
            mode="nearest",
        )
        smoothed[..., component] = np.divide(
            numerator,
            weights,
            out=np.zeros_like(numerator),
            where=weights > 1e-6,
        )
    return smoothed


def _gradients(
    field: np.ndarray,
    valid: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    sigma: float,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    smoothed = _masked_smooth(field, valid, sigma)
    du_dx = np.gradient(smoothed[..., 0], x_m, axis=2, edge_order=2)
    du_dy = np.gradient(smoothed[..., 0], y_m, axis=1, edge_order=2)
    dv_dx = np.gradient(smoothed[..., 1], x_m, axis=2, edge_order=2)
    dv_dy = np.gradient(smoothed[..., 1], y_m, axis=1, edge_order=2)
    interior = binary_erosion(
        valid,
        structure=np.ones((1, 3, 3), dtype=bool),
        iterations=2,
        border_value=0,
    )
    return {
        "du_dx": du_dx,
        "du_dy": du_dy,
        "dv_dx": dv_dx,
        "dv_dy": dv_dy,
    }, interior


def _derived_gradients(grad: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    du_dx = grad["du_dx"]
    du_dy = grad["du_dy"]
    dv_dx = grad["dv_dx"]
    dv_dy = grad["dv_dy"]
    sxx = du_dx
    syy = dv_dy
    sxy = 0.5 * (du_dy + dv_dx)
    omega_xy = 0.5 * (du_dy - dv_dx)
    strain_sq = sxx**2 + syy**2 + 2.0 * sxy**2
    rotation_sq = 2.0 * omega_xy**2
    trace = du_dx + dv_dy
    determinant = du_dx * dv_dy - du_dy * dv_dx
    discriminant = trace**2 - 4.0 * determinant
    return {
        "vorticity": dv_dx - du_dy,
        "strain_magnitude": np.sqrt(np.maximum(2.0 * strain_sq, 0.0)),
        "q_criterion": 0.5 * (rotation_sq - strain_sq),
        "swirling_strength": 0.5 * np.sqrt(np.maximum(-discriminant, 0.0)),
        "divergence": trace,
    }


def _symmetry_index(
    field: np.ndarray,
    valid: np.ndarray,
    x_over_d: np.ndarray,
    y_over_d: np.ndarray,
    cylinder_y_over_d: float,
) -> float:
    x_keep = (x_over_d >= 0.75) & (x_over_d <= 8.5)
    upper = np.flatnonzero(y_over_d >= cylinder_y_over_d)
    if upper.size == 0 or not np.any(x_keep):
        return float("nan")
    mirrored = np.asarray(
        [int(np.argmin(np.abs(y_over_d - (2.0 * cylinder_y_over_d - y_over_d[i])))) for i in upper],
        dtype=int,
    )
    keep = mirrored != upper
    upper = upper[keep]
    mirrored = mirrored[keep]
    if upper.size == 0:
        return float("nan")
    a = field[upper][:, x_keep]
    b = field[mirrored][:, x_keep]
    pair_valid = valid[upper][:, x_keep] & valid[mirrored][:, x_keep]
    if not np.any(pair_valid):
        return float("nan")
    residual_sq = (a[..., 0] - b[..., 0]) ** 2 + (a[..., 1] + b[..., 1]) ** 2
    reference_sq = 0.5 * np.sum(a**2 + b**2, axis=-1)
    numerator = math.sqrt(float(np.mean(residual_sq[pair_valid])))
    denominator = math.sqrt(float(np.mean(reference_sq[pair_valid])))
    return numerator / max(denominator, 1e-12)


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    output = np.full_like(values, np.nan, dtype=float)
    finite = np.isfinite(values)
    numerator = np.convolve(np.where(finite, values, 0.0), np.ones(window), mode="same")
    denominator = np.convolve(finite.astype(float), np.ones(window), mode="same")
    np.divide(numerator, denominator, out=output, where=denominator > 0)
    return output


def _publication_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams.update({
        "font.size": 7.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })


def _save(fig: plt.Figure, output: pathlib.Path, code: str) -> None:
    stem = output / f"{code}_{FIGURE_SPECS[code]}"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _decorate_field_axis(
    axis: plt.Axes,
    x_over_d: np.ndarray,
    y_over_d: np.ndarray,
    cylinder_y: float,
    title: str,
) -> None:
    axis.add_patch(Circle((0.0, cylinder_y), 0.5, facecolor="white", edgecolor="#222222", lw=0.65))
    axis.set_xlim(float(x_over_d.min()), float(x_over_d.max()))
    axis.set_ylim(float(y_over_d.min()), float(y_over_d.max()))
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"$x/D$")
    axis.set_ylabel(r"$y/D$")
    axis.set_title(title, fontweight="normal")


def _plot_compare_map(
    output: pathlib.Path,
    code: str,
    truth: np.ndarray,
    prediction: np.ndarray,
    valid: np.ndarray,
    x_over_d: np.ndarray,
    y_over_d: np.ndarray,
    cylinder_y: float,
    colorbar_label: str,
    cmap: str,
    centered: bool,
    percentile: float = 99.0,
    lower_zero: bool = False,
) -> None:
    values = [np.where(valid, truth, np.nan), np.where(valid, prediction, np.nan)]
    if centered:
        scale = _finite_percentile(values, percentile)
        norm = TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale)
    elif lower_zero:
        high = _finite_percentile(values, percentile)
        norm = Normalize(vmin=0.0, vmax=high)
    else:
        pooled = np.concatenate([value[np.isfinite(value)] for value in values])
        norm = Normalize(vmin=float(np.percentile(pooled, 1.0)), vmax=float(np.percentile(pooled, percentile)))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.35), constrained_layout=True)
    mesh = None
    for axis, field, title in zip(axes, values, ("Measured", "APCE")):
        mesh = axis.pcolormesh(x_over_d, y_over_d, field, shading="auto", cmap=cmap, norm=norm, rasterized=True)
        _decorate_field_axis(axis, x_over_d, y_over_d, cylinder_y, title)
    colorbar = fig.colorbar(mesh, ax=axes, orientation="vertical", fraction=0.025, pad=0.02)
    colorbar.set_label(colorbar_label)
    _save(fig, output, code)


def _plot_single_map(
    output: pathlib.Path,
    code: str,
    field: np.ndarray,
    valid: np.ndarray,
    x_over_d: np.ndarray,
    y_over_d: np.ndarray,
    cylinder_y: float,
    colorbar_label: str,
    cmap: str,
    vmin: float,
    vmax: float,
    title: str = "Diagnostic field",
) -> None:
    fig, axis = plt.subplots(figsize=(4.7, 2.65), constrained_layout=True)
    mesh = axis.pcolormesh(
        x_over_d,
        y_over_d,
        np.where(valid, field, np.nan),
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    _decorate_field_axis(axis, x_over_d, y_over_d, cylinder_y, title)
    colorbar = fig.colorbar(mesh, ax=axis, orientation="vertical", fraction=0.04, pad=0.025)
    colorbar.set_label(colorbar_label)
    _save(fig, output, code)


def _plot_line(
    output: pathlib.Path,
    code: str,
    x: np.ndarray,
    series: list[tuple[np.ndarray, str, str, str]],
    xlabel: str,
    ylabel: str,
    zero_line: bool = False,
) -> None:
    fig, axis = plt.subplots(figsize=(4.15, 2.85), constrained_layout=True)
    for values, label, color, linestyle in series:
        axis.plot(x, values, color=color, lw=1.35, linestyle=linestyle, label=label)
    if zero_line:
        axis.axhline(0.0, color="#767676", lw=0.7, linestyle=(0, (3, 2)))
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.legend(ncol=2, fontsize=6.5)
    axis.margins(x=0.01)
    _save(fig, output, code)


def _recirculation_length(x_over_d: np.ndarray, centerline_u: np.ndarray) -> float:
    eligible = np.flatnonzero((x_over_d >= 0.5) & np.isfinite(centerline_u))
    if eligible.size == 0:
        return float("nan")
    negative_positions = np.flatnonzero(centerline_u[eligible] < 0.0)
    if negative_positions.size == 0:
        return float("nan")
    start_position = int(negative_positions[0])
    segment = eligible[start_position:]
    positive = centerline_u[segment] >= 0.0
    for offset in range(max(0, positive.size - 2)):
        if np.all(positive[offset : offset + 3]):
            crossing_index = segment[offset]
            prior_index = max(crossing_index - 1, segment[0])
            x0, x1 = x_over_d[prior_index], x_over_d[crossing_index]
            u0, u1 = centerline_u[prior_index], centerline_u[crossing_index]
            if abs(u1 - u0) < 1e-12:
                return float(x1)
            return float(x0 - u0 * (x1 - x0) / (u1 - u0))
    return float(x_over_d[segment[-1]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--case", default="0679")
    parser.add_argument("--method", choices=("pce", "apce"), default="apce")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variant", default="rank256_stride1")
    parser.add_argument("--layout", default="20x40")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--block", type=int, default=16)
    parser.add_argument("--gradient-sigma", type=float, default=0.75)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    result_root = pathlib.Path(config["output_root"])
    model_root = result_root / "models" / args.variant
    trace_root = result_root / "runs" / args.variant / "traces"
    case_path = list_cases(pathlib.Path(config["data_root"]))[args.case]
    case = VIVCase.open(case_path)
    pod = PODModel.load(model_root / "pod_model.npz")
    trace_path = _trace_path(trace_root, args.case, args.method, args.seed, args.layout)
    with np.load(trace_path, allow_pickle=False) as trace:
        latent_estimate = np.asarray(trace["latent_estimate"], dtype=np.float32)
        trace_time = np.asarray(trace["time_s"], dtype=float)
    if latent_estimate.shape != (case.time_s.size, pod.rank):
        raise ValueError(f"Trace shape {latent_estimate.shape} does not match {(case.time_s.size, pod.rank)}")
    if not np.allclose(trace_time, case.time_s, rtol=0.0, atol=1e-8):
        raise ValueError("Trace and PIV time axes differ")

    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device}, but CUDA is unavailable")
    device = requested_device
    basis = torch.as_tensor(pod.basis, dtype=torch.float32, device=device)
    mean_vector = torch.as_tensor(pod.mean, dtype=torch.float32, device=device)

    diameter_m = float(config["cylinder_diameter_m"])
    x_m = np.asarray(case.x_mm, dtype=float) / 1000.0
    y_m = np.asarray(case.y_mm, dtype=float) / 1000.0
    x_over_d = x_m / diameter_m
    y_over_d = y_m / diameter_m
    dt = float(np.median(np.diff(case.time_s)))
    warmup = int(np.searchsorted(case.time_s, float(config["warmup_seconds"])))
    frames = np.arange(warmup, case.time_s.size, dtype=int)
    cylinder_centered = case.cyl_displ_m - np.median(case.cyl_displ_m[warmup:])
    representative_frame = warmup + int(np.argmax(np.abs(cylinder_centered[warmup:])))
    shape = (case.y_mm.size, case.x_mm.size, 2)
    scalar_shape = shape[:2]

    truth_sum = np.zeros(shape, dtype=np.float64)
    prediction_sum = np.zeros(shape, dtype=np.float64)
    truth_sq_sum = np.zeros(shape, dtype=np.float64)
    prediction_sq_sum = np.zeros(shape, dtype=np.float64)
    truth_uv_sum = np.zeros(scalar_shape, dtype=np.float64)
    prediction_uv_sum = np.zeros(scalar_shape, dtype=np.float64)
    count = np.zeros(scalar_shape, dtype=np.int64)
    upstream_sum = 0.0
    upstream_count = 0
    circulation_positive_truth = np.full(case.time_s.size, np.nan)
    circulation_negative_truth = np.full(case.time_s.size, np.nan)
    circulation_positive_prediction = np.full(case.time_s.size, np.nan)
    circulation_negative_prediction = np.full(case.time_s.size, np.nan)
    enstrophy_truth = np.full(case.time_s.size, np.nan)
    enstrophy_prediction = np.full(case.time_s.size, np.nan)
    symmetry_truth = np.full(case.time_s.size, np.nan)
    symmetry_prediction = np.full(case.time_s.size, np.nan)
    representative: dict[str, np.ndarray | float] = {}
    wake_roi = (
        (x_over_d[None, :] >= 0.75)
        & (x_over_d[None, :] <= 8.5)
        & (y_over_d[:, None] >= -2.2)
        & (y_over_d[:, None] <= 2.2)
    )
    upstream_roi = x_over_d[None, :] <= -1.0
    dx = abs(float(np.median(np.diff(x_m))))
    dy = abs(float(np.median(np.diff(y_m))))
    area = dx * dy

    for start in range(warmup, case.time_s.size, args.block):
        stop = min(start + args.block, case.time_s.size)
        truth, valid = case.physical_frames(start, stop)
        with torch.inference_mode():
            state = torch.as_tensor(latent_estimate[start:stop], dtype=torch.float32, device=device)
            decoded = mean_vector + state @ basis.mT
            prediction = decoded.reshape(stop - start, *shape).cpu().numpy()
        valid4 = valid[..., None]
        truth_valid = np.where(valid4, truth, 0.0)
        prediction_valid = np.where(valid4, prediction, 0.0)
        truth_sum += truth_valid.sum(axis=0, dtype=np.float64)
        prediction_sum += prediction_valid.sum(axis=0, dtype=np.float64)
        truth_sq_sum += np.square(truth_valid, dtype=np.float64).sum(axis=0)
        prediction_sq_sum += np.square(prediction_valid, dtype=np.float64).sum(axis=0)
        truth_uv_sum += np.where(valid, truth[..., 0] * truth[..., 1], 0.0).sum(axis=0, dtype=np.float64)
        prediction_uv_sum += np.where(valid, prediction[..., 0] * prediction[..., 1], 0.0).sum(axis=0, dtype=np.float64)
        count += valid.sum(axis=0, dtype=np.int64)
        upstream_valid = valid & upstream_roi[None, ...]
        upstream_sum += float(np.where(upstream_valid, truth[..., 0], 0.0).sum(dtype=np.float64))
        upstream_count += int(upstream_valid.sum())

        truth_grad, grad_valid = _gradients(truth, valid, x_m, y_m, args.gradient_sigma)
        prediction_grad, _ = _gradients(prediction, valid, x_m, y_m, args.gradient_sigma)
        truth_derived = _derived_gradients(truth_grad)
        prediction_derived = _derived_gradients(prediction_grad)
        active = grad_valid & wake_roi[None, ...]
        for local, frame in enumerate(range(start, stop)):
            local_active = active[local]
            omega_truth = truth_derived["vorticity"][local]
            omega_prediction = prediction_derived["vorticity"][local]
            circulation_positive_truth[frame] = area * np.sum(
                np.where(local_active, np.maximum(omega_truth, 0.0), 0.0), dtype=np.float64
            )
            circulation_negative_truth[frame] = area * np.sum(
                np.where(local_active, np.minimum(omega_truth, 0.0), 0.0), dtype=np.float64
            )
            circulation_positive_prediction[frame] = area * np.sum(
                np.where(local_active, np.maximum(omega_prediction, 0.0), 0.0), dtype=np.float64
            )
            circulation_negative_prediction[frame] = area * np.sum(
                np.where(local_active, np.minimum(omega_prediction, 0.0), 0.0), dtype=np.float64
            )
            enstrophy_truth[frame] = 0.5 * area * np.sum(
                np.where(local_active, omega_truth**2, 0.0), dtype=np.float64
            )
            enstrophy_prediction[frame] = 0.5 * area * np.sum(
                np.where(local_active, omega_prediction**2, 0.0), dtype=np.float64
            )
            cylinder_y = float(case.cyl_displ_m[frame] / diameter_m)
            symmetry_truth[frame] = _symmetry_index(
                truth[local], valid[local], x_over_d, y_over_d, cylinder_y
            )
            symmetry_prediction[frame] = _symmetry_index(
                prediction[local], valid[local], x_over_d, y_over_d, cylinder_y
            )
            if frame == representative_frame:
                representative = {
                    "truth": truth[local].copy(),
                    "prediction": prediction[local].copy(),
                    "valid": valid[local].copy(),
                    "grad_valid": grad_valid[local].copy(),
                    "cylinder_y": cylinder_y,
                }
                for key, value in truth_derived.items():
                    representative[f"truth_{key}"] = value[local].copy()
                for key, value in prediction_derived.items():
                    representative[f"prediction_{key}"] = value[local].copy()

    if not representative:
        raise RuntimeError("Representative frame was not captured")
    inflow_velocity = upstream_sum / max(upstream_count, 1)
    if not np.isfinite(inflow_velocity) or inflow_velocity <= 0:
        raise RuntimeError(f"Invalid upstream reference velocity: {inflow_velocity}")
    scale_rate = diameter_m / inflow_velocity
    time_keep = case.time_s[frames]
    count4 = count[..., None]
    truth_mean = np.divide(truth_sum, count4, out=np.full(shape, np.nan), where=count4 > 0)
    prediction_mean = np.divide(prediction_sum, count4, out=np.full(shape, np.nan), where=count4 > 0)
    truth_variance = np.maximum(
        np.divide(truth_sq_sum, count4, out=np.full(shape, np.nan), where=count4 > 0) - truth_mean**2,
        0.0,
    )
    prediction_variance = np.maximum(
        np.divide(prediction_sq_sum, count4, out=np.full(shape, np.nan), where=count4 > 0) - prediction_mean**2,
        0.0,
    )
    truth_covariance = (
        np.divide(truth_uv_sum, count, out=np.full(scalar_shape, np.nan), where=count > 0)
        - truth_mean[..., 0] * truth_mean[..., 1]
    )
    prediction_covariance = (
        np.divide(prediction_uv_sum, count, out=np.full(scalar_shape, np.nan), where=count > 0)
        - prediction_mean[..., 0] * prediction_mean[..., 1]
    )
    statistical_valid = count >= int(math.ceil(0.8 * frames.size))
    mean_cylinder_y = float(np.median(case.cyl_displ_m[frames]) / diameter_m)

    truth_rep = np.asarray(representative["truth"])
    prediction_rep = np.asarray(representative["prediction"])
    valid_rep = np.asarray(representative["valid"], dtype=bool)
    grad_valid_rep = np.asarray(representative["grad_valid"], dtype=bool)
    cylinder_y_rep = float(representative["cylinder_y"])
    truth_speed = np.linalg.norm(truth_rep, axis=-1)
    prediction_speed = np.linalg.norm(prediction_rep, axis=-1)
    direction_valid = valid_rep & (truth_speed >= 0.05 * inflow_velocity) & (prediction_speed >= 0.05 * inflow_velocity)
    cosine = np.sum(truth_rep * prediction_rep, axis=-1) / np.maximum(truth_speed * prediction_speed, 1e-12)
    direction_error = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    _publication_style()
    args.output.mkdir(parents=True, exist_ok=True)

    direction_limit = min(90.0, max(30.0, float(np.nanpercentile(direction_error[direction_valid], 99.0))))
    _plot_single_map(
        args.output, "A10", direction_error, direction_valid, x_over_d, y_over_d, cylinder_y_rep,
        r"direction error ($^\circ$)", "magma", 0.0, direction_limit,
        title="Vector direction error",
    )
    _plot_compare_map(
        args.output, "A11", truth_mean[..., 0] / inflow_velocity, prediction_mean[..., 0] / inflow_velocity,
        statistical_valid, x_over_d, y_over_d, mean_cylinder_y, r"$\overline{u}/U_\infty$", "viridis", False,
    )
    _plot_compare_map(
        args.output, "A12", truth_mean[..., 1] / inflow_velocity, prediction_mean[..., 1] / inflow_velocity,
        statistical_valid, x_over_d, y_over_d, mean_cylinder_y, r"$\overline{v}/U_\infty$", "RdBu_r", True,
    )

    truth_rms = np.sqrt(truth_variance) / inflow_velocity
    prediction_rms = np.sqrt(prediction_variance) / inflow_velocity
    rms_limit = _finite_percentile(
        [truth_rms[..., 0][statistical_valid], prediction_rms[..., 0][statistical_valid],
         truth_rms[..., 1][statistical_valid], prediction_rms[..., 1][statistical_valid]], 99.0
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.35), constrained_layout=True)
    rms_fields = (
        truth_rms[..., 0], prediction_rms[..., 0], truth_rms[..., 1], prediction_rms[..., 1]
    )
    rms_titles = ("Measured $u'$ RMS", "APCE $u'$ RMS", "Measured $v'$ RMS", "APCE $v'$ RMS")
    mesh = None
    for axis, field, title in zip(axes.ravel(), rms_fields, rms_titles):
        mesh = axis.pcolormesh(
            x_over_d, y_over_d, np.where(statistical_valid, field, np.nan), shading="auto",
            cmap="magma", vmin=0.0, vmax=rms_limit, rasterized=True,
        )
        _decorate_field_axis(axis, x_over_d, y_over_d, mean_cylinder_y, title)
    colorbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), orientation="vertical", fraction=0.025, pad=0.02)
    colorbar.set_label(r"RMS$/U_\infty$")
    _save(fig, args.output, "A13")

    truth_tke = 0.5 * np.sum(truth_variance, axis=-1) / inflow_velocity**2
    prediction_tke = 0.5 * np.sum(prediction_variance, axis=-1) / inflow_velocity**2
    _plot_compare_map(
        args.output, "A14", truth_tke, prediction_tke, statistical_valid, x_over_d, y_over_d,
        mean_cylinder_y, r"$k_{2C}/U_\infty^2$", "magma", False, lower_zero=True,
    )
    _plot_compare_map(
        args.output, "A15", truth_covariance / inflow_velocity**2,
        prediction_covariance / inflow_velocity**2, statistical_valid, x_over_d, y_over_d,
        mean_cylinder_y, r"$\overline{u'v'}/U_\infty^2$", "RdBu_r", True,
    )
    _plot_compare_map(
        args.output, "A16", truth_variance[..., 0] / inflow_velocity**2,
        prediction_variance[..., 0] / inflow_velocity**2, statistical_valid, x_over_d, y_over_d,
        mean_cylinder_y, r"$\overline{u'^2}/U_\infty^2$", "magma", False, lower_zero=True,
    )
    _plot_compare_map(
        args.output, "A17", truth_variance[..., 1] / inflow_velocity**2,
        prediction_variance[..., 1] / inflow_velocity**2, statistical_valid, x_over_d, y_over_d,
        mean_cylinder_y, r"$\overline{v'^2}/U_\infty^2$", "magma", False, lower_zero=True,
    )

    truth_vorticity = np.asarray(representative["truth_vorticity"]) * scale_rate
    prediction_vorticity = np.asarray(representative["prediction_vorticity"]) * scale_rate
    _plot_compare_map(
        args.output, "A18", truth_vorticity, prediction_vorticity, grad_valid_rep, x_over_d, y_over_d,
        cylinder_y_rep, r"$\omega_zD/U_\infty$", "RdBu_r", True, percentile=98.5,
    )
    vorticity_error = prediction_vorticity - truth_vorticity
    vorticity_error_limit = _finite_percentile([vorticity_error[grad_valid_rep]], 98.5)
    _plot_single_map(
        args.output, "A19", vorticity_error, grad_valid_rep, x_over_d, y_over_d, cylinder_y_rep,
        r"$\Delta\omega_zD/U_\infty$", "RdBu_r", -vorticity_error_limit, vorticity_error_limit,
        title="Vorticity error",
    )
    _plot_compare_map(
        args.output, "A20", np.asarray(representative["truth_strain_magnitude"]) * scale_rate,
        np.asarray(representative["prediction_strain_magnitude"]) * scale_rate,
        grad_valid_rep, x_over_d, y_over_d, cylinder_y_rep, r"$|S|D/U_\infty$", "magma", False,
        percentile=98.5, lower_zero=True,
    )
    _plot_compare_map(
        args.output, "A21", np.asarray(representative["truth_q_criterion"]) * scale_rate**2,
        np.asarray(representative["prediction_q_criterion"]) * scale_rate**2,
        grad_valid_rep, x_over_d, y_over_d, cylinder_y_rep, r"$QD^2/U_\infty^2$", "RdBu_r", True,
        percentile=98.5,
    )
    _plot_compare_map(
        args.output, "A22", np.asarray(representative["truth_swirling_strength"]) * scale_rate,
        np.asarray(representative["prediction_swirling_strength"]) * scale_rate,
        grad_valid_rep, x_over_d, y_over_d, cylinder_y_rep, r"$\lambda_{ci}D/U_\infty$", "magma", False,
        percentile=98.5, lower_zero=True,
    )
    _plot_compare_map(
        args.output, "A23", np.asarray(representative["truth_divergence"]) * scale_rate,
        np.asarray(representative["prediction_divergence"]) * scale_rate,
        grad_valid_rep, x_over_d, y_over_d, cylinder_y_rep, r"$(\nabla\cdot\mathbf{u})D/U_\infty$",
        "RdBu_r", True, percentile=98.5,
    )

    center_band = np.abs(y_over_d - mean_cylinder_y) <= 0.10
    centerline_truth = np.nanmean(np.where(statistical_valid[center_band], truth_mean[center_band, :, 0], np.nan), axis=0)
    centerline_prediction = np.nanmean(
        np.where(statistical_valid[center_band], prediction_mean[center_band, :, 0], np.nan), axis=0
    )
    recirculation_truth = _recirculation_length(x_over_d, centerline_truth)
    recirculation_prediction = _recirculation_length(x_over_d, centerline_prediction)
    fig, axis = plt.subplots(figsize=(4.15, 2.85), constrained_layout=True)
    axis.plot(x_over_d, centerline_truth / inflow_velocity, color="#272727", lw=1.4, label="Measured")
    axis.plot(x_over_d, centerline_prediction / inflow_velocity, color="#0F4D92", lw=1.4, label="APCE")
    axis.axhline(0.0, color="#767676", lw=0.7, linestyle=(0, (3, 2)))
    if np.isfinite(recirculation_truth):
        axis.axvline(recirculation_truth, color="#272727", lw=0.9, linestyle=(0, (3, 2)))
    if np.isfinite(recirculation_prediction):
        axis.axvline(recirculation_prediction, color="#0F4D92", lw=0.9, linestyle=(0, (3, 2)))
    axis.set_xlabel(r"$x/D$")
    axis.set_ylabel(r"centerline $\overline{u}/U_\infty$")
    axis.legend(fontsize=6.5)
    _save(fig, args.output, "A24")

    _plot_line(
        args.output, "A25", x_over_d,
        [((inflow_velocity - centerline_truth) / inflow_velocity, "Measured", "#272727", "-"),
         ((inflow_velocity - centerline_prediction) / inflow_velocity, "APCE", "#0F4D92", "-")],
        r"$x/D$", r"$(U_\infty-\overline{u}_c)/U_\infty$", zero_line=True,
    )

    def wake_width_and_momentum(mean_u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        width = np.full(x_over_d.size, np.nan)
        momentum = np.full(x_over_d.size, np.nan)
        for ix in range(x_over_d.size):
            valid_y = statistical_valid[:, ix] & np.isfinite(mean_u[:, ix])
            if valid_y.sum() < 8:
                continue
            local_y = y_over_d[valid_y]
            ratio = mean_u[valid_y, ix] / inflow_velocity
            deficit = np.maximum(1.0 - ratio, 0.0)
            deficit_integral = float(np.trapezoid(deficit, local_y))
            if deficit_integral > 1e-9:
                centroid = float(np.trapezoid(local_y * deficit, local_y) / deficit_integral)
                width[ix] = math.sqrt(max(float(np.trapezoid((local_y - centroid) ** 2 * deficit, local_y) / deficit_integral), 0.0))
            momentum[ix] = float(np.trapezoid(ratio * (1.0 - ratio), local_y))
        return width, momentum

    width_truth, momentum_truth = wake_width_and_momentum(truth_mean[..., 0])
    width_prediction, momentum_prediction = wake_width_and_momentum(prediction_mean[..., 0])
    downstream = x_over_d >= 0.75
    _plot_line(
        args.output, "A26", x_over_d[downstream],
        [(width_truth[downstream], "Measured", "#272727", "-"),
         (width_prediction[downstream], "APCE", "#0F4D92", "-")],
        r"$x/D$", r"deficit RMS width $\sigma_y/D$",
    )
    _plot_line(
        args.output, "A27", x_over_d[downstream],
        [(momentum_truth[downstream], "Measured", "#272727", "-"),
         (momentum_prediction[downstream], "APCE", "#0F4D92", "-")],
        r"$x/D$", r"momentum thickness $\theta/D$", zero_line=True,
    )

    circulation_scale = inflow_velocity * diameter_m
    _plot_line(
        args.output, "A28", time_keep,
        [(circulation_positive_truth[frames] / circulation_scale, r"Measured $\Gamma^+$", "#B64342", "-"),
         (circulation_positive_prediction[frames] / circulation_scale, r"APCE $\Gamma^+$", "#B64342", "--"),
         (circulation_negative_truth[frames] / circulation_scale, r"Measured $\Gamma^-$", "#0F4D92", "-"),
         (circulation_negative_prediction[frames] / circulation_scale, r"APCE $\Gamma^-$", "#0F4D92", "--")],
        "time (s)", r"circulation $\Gamma/(U_\infty D)$", zero_line=True,
    )
    _plot_line(
        args.output, "A29", time_keep,
        [(enstrophy_truth[frames] / inflow_velocity**2, "Measured", "#272727", "-"),
         (enstrophy_prediction[frames] / inflow_velocity**2, "APCE", "#0F4D92", "-")],
        "time (s)", r"enstrophy $\frac{1}{2}\int\omega_z^2dA/U_\infty^2$",
    )

    smooth_window = max(3, int(round(1.0 / dt)))
    fig, axis = plt.subplots(figsize=(4.15, 2.85), constrained_layout=True)
    axis.plot(time_keep, symmetry_truth[frames], color="#767676", lw=0.45, alpha=0.35)
    axis.plot(time_keep, symmetry_prediction[frames], color="#3775BA", lw=0.45, alpha=0.30)
    axis.plot(time_keep, _rolling_mean(symmetry_truth[frames], smooth_window), color="#272727", lw=1.35, label="Measured")
    axis.plot(time_keep, _rolling_mean(symmetry_prediction[frames], smooth_window), color="#0F4D92", lw=1.35, label="APCE")
    axis.set_xlabel("time (s)")
    axis.set_ylabel("moving-frame asymmetry index")
    axis.legend(fontsize=6.5)
    axis.margins(x=0.01)
    _save(fig, args.output, "A30")

    source_path = args.output / "A10_A30_source.npz"
    np.savez_compressed(
        source_path,
        x_over_d=x_over_d,
        y_over_d=y_over_d,
        time_s=time_keep,
        statistical_valid=statistical_valid,
        representative_valid=valid_rep,
        representative_gradient_valid=grad_valid_rep,
        direction_error_deg=direction_error,
        truth_mean=truth_mean,
        prediction_mean=prediction_mean,
        truth_variance=truth_variance,
        prediction_variance=prediction_variance,
        truth_covariance=truth_covariance,
        prediction_covariance=prediction_covariance,
        truth_vorticity=truth_vorticity,
        prediction_vorticity=prediction_vorticity,
        truth_strain=np.asarray(representative["truth_strain_magnitude"]) * scale_rate,
        prediction_strain=np.asarray(representative["prediction_strain_magnitude"]) * scale_rate,
        truth_q=np.asarray(representative["truth_q_criterion"]) * scale_rate**2,
        prediction_q=np.asarray(representative["prediction_q_criterion"]) * scale_rate**2,
        truth_lambda_ci=np.asarray(representative["truth_swirling_strength"]) * scale_rate,
        prediction_lambda_ci=np.asarray(representative["prediction_swirling_strength"]) * scale_rate,
        truth_divergence=np.asarray(representative["truth_divergence"]) * scale_rate,
        prediction_divergence=np.asarray(representative["prediction_divergence"]) * scale_rate,
        centerline_truth=centerline_truth,
        centerline_prediction=centerline_prediction,
        wake_width_truth=width_truth,
        wake_width_prediction=width_prediction,
        momentum_thickness_truth=momentum_truth,
        momentum_thickness_prediction=momentum_prediction,
        circulation_positive_truth=circulation_positive_truth[frames] / circulation_scale,
        circulation_negative_truth=circulation_negative_truth[frames] / circulation_scale,
        circulation_positive_prediction=circulation_positive_prediction[frames] / circulation_scale,
        circulation_negative_prediction=circulation_negative_prediction[frames] / circulation_scale,
        enstrophy_truth=enstrophy_truth[frames] / inflow_velocity**2,
        enstrophy_prediction=enstrophy_prediction[frames] / inflow_velocity**2,
        symmetry_truth=symmetry_truth[frames],
        symmetry_prediction=symmetry_prediction[frames],
    )
    summary_rows = [
        ("case_id", args.case),
        ("reduced_velocity", case.reduced_velocity),
        ("method", args.method),
        ("seed", args.seed),
        ("variant", args.variant),
        ("sensor_layout", args.layout),
        ("inflow_velocity_m_s", inflow_velocity),
        ("representative_frame", representative_frame),
        ("representative_time_s", case.time_s[representative_frame]),
        ("gradient_sigma_grid_cells", args.gradient_sigma),
        ("gradient_mask_erosion_cells", 2),
        ("recirculation_length_truth_D", recirculation_truth),
        ("recirculation_length_prediction_D", recirculation_prediction),
        ("direction_error_median_deg", float(np.nanmedian(direction_error[direction_valid]))),
        ("direction_error_p90_deg", float(np.nanpercentile(direction_error[direction_valid], 90.0))),
        ("mean_abs_divergence_truth", float(np.nanmean(np.abs(np.asarray(representative["truth_divergence"])[grad_valid_rep] * scale_rate)))),
        ("mean_abs_divergence_prediction", float(np.nanmean(np.abs(np.asarray(representative["prediction_divergence"])[grad_valid_rep] * scale_rate)))),
        ("mean_symmetry_truth", float(np.nanmean(symmetry_truth[frames]))),
        ("mean_symmetry_prediction", float(np.nanmean(symmetry_prediction[frames]))),
    ]
    with (args.output / "A10_A30_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["quantity", "value"])
        writer.writerows(summary_rows)

    metadata = {
        "figure_contract": {
            "core_conclusion": "Assess whether APCE preserves wake structure beyond pointwise velocity error.",
            "evidence_chain": {
                "A10": "vector orientation",
                "A11_A17": "mean and second-order statistics",
                "A18_A23": "gradient and incompressibility diagnostics",
                "A24_A30": "integral wake geometry and temporal structure",
            },
            "archetype": "independent diagnostic figures",
            "backend": "Python/matplotlib only",
            "exports": ["SVG", "PDF", "PNG", "NPZ", "CSV"],
        },
        "case_id": args.case,
        "reduced_velocity": case.reduced_velocity,
        "method": args.method,
        "trace": str(trace_path),
        "raw_data": str(case_path),
        "pod_model": str(model_root / "pod_model.npz"),
        "warmup_frames_excluded": warmup,
        "frames_analyzed": int(frames.size),
        "representative_frame_rule": "maximum absolute median-centered cylinder displacement after warm-up",
        "representative_frame": representative_frame,
        "representative_time_s": float(case.time_s[representative_frame]),
        "upstream_reference": "mean measured u for x/D <= -1 over analyzed valid samples",
        "inflow_velocity_m_s": inflow_velocity,
        "gradient_processing": {
            "spatial_gaussian_sigma_grid_cells": args.gradient_sigma,
            "mask_normalized_smoothing": True,
            "mask_erosion_grid_cells": 2,
        },
        "limitations": [
            "Two-component PIV does not provide the out-of-plane velocity or full three-dimensional TKE.",
            "Gradient diagnostics are resolution- and smoothing-sensitive and are screening results.",
            "Momentum thickness is a wake diagnostic, not a directly measured drag coefficient.",
        ],
        "figures": FIGURE_SPECS,
    }
    (args.output / "A10_A30_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "figures": len(FIGURE_SPECS), "summary": dict(summary_rows)}, indent=2))


if __name__ == "__main__":
    main()
