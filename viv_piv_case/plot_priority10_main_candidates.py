"""Generate ten independent main-text candidate figures for the VIV-PIV case."""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
from collections import defaultdict

import matplotlib as mpl
import numpy as np
import torch
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from scipy.signal import welch

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .common import load_config
from .io import VIVCase, list_cases
from .rom import PODModel


CASES = ("0463", "0556", "0679", "0803", "1359")
CASE_LABELS = tuple(f"{int(case_id) / 100:.2f}" for case_id in CASES)
CASE_COLORS = ("#484878", "#7884B4", "#42949E", "#9A4D8E", "#B64342")
TRUTH_COLOR = "#272727"
APCE_COLOR = "#0F4D92"
PCE_COLOR = "#C9793D"


def _style() -> None:
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
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 6.5,
    })


def _save(fig: plt.Figure, output: pathlib.Path, name: str) -> None:
    stem = output / name
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _single(path: pathlib.Path, pattern: str) -> pathlib.Path:
    matches = sorted(path.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one match for {path / pattern}, found {len(matches)}")
    return matches[0]


def _load_json(path: pathlib.Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _trace_path(root: pathlib.Path, case_id: str, method: str) -> pathlib.Path:
    return _single(
        root / "runs" / "rank256_stride1" / "traces",
        f"viv_{case_id}_{method}_seed000_layoutadaptive_fullfield_valid_ens064_covfull_shr050.npz",
    )


def _run_path(root: pathlib.Path, case_id: str, method: str) -> pathlib.Path:
    return _single(
        root / "runs" / "rank256_stride1" / "runs",
        f"viv_{case_id}_{method}_seed000_layoutadaptive_fullfield_valid_ens064_covfull_shr050.json",
    )


def _write_csv(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _correlation_from_sums(count: int, sx: float, sy: float, sxx: float, syy: float, sxy: float) -> float:
    covariance = sxy - sx * sy / max(count, 1)
    variance_x = sxx - sx * sx / max(count, 1)
    variance_y = syy - sy * sy / max(count, 1)
    return covariance / max(math.sqrt(max(variance_x * variance_y, 0.0)), 1e-30)


def _evaluate_case(
    case: VIVCase,
    pod: PODModel,
    latent: np.ndarray,
    sensor_flat_indices: np.ndarray,
    device: torch.device,
    diameter_m: float,
    block: int,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    basis = torch.as_tensor(pod.basis, dtype=torch.float32, device=device)
    mean = torch.as_tensor(pod.mean, dtype=torch.float32, device=device)
    height = case.y_mm.size
    width = case.x_mm.size
    pixel_count = height * width
    excluded_pixels = torch.as_tensor(
        np.unique(np.asarray(sensor_flat_indices, dtype=np.int64) // 2),
        dtype=torch.int64,
        device=device,
    )
    x_over_d = np.asarray(case.x_mm, dtype=float) / (diameter_m * 1000.0)
    y_over_d = np.asarray(case.y_mm, dtype=float) / (diameter_m * 1000.0)
    probe_ix = int(np.argmin(np.abs(x_over_d - 2.0)))
    probe_iy = int(np.argmin(np.abs(y_over_d - 0.0)))
    probe_pixel = probe_iy * width + probe_ix
    probe_truth = np.empty(case.time_s.size, dtype=np.float64)
    probe_prediction = np.empty(case.time_s.size, dtype=np.float64)

    vector_error = 0.0
    vector_truth = 0.0
    speed_error = 0.0
    speed_truth = 0.0
    speed_count = 0
    sx = sy = sxx = syy = sxy = 0.0
    upstream_sum = 0.0
    upstream_count = 0
    upstream_x = torch.as_tensor(x_over_d < -0.8, dtype=torch.bool, device=device)

    with torch.inference_mode():
        for start, truth_flat_np, valid_flat_np in case.iter_physical(block=block):
            stop = start + truth_flat_np.shape[0]
            truth = torch.as_tensor(truth_flat_np, dtype=torch.float32, device=device).reshape(-1, pixel_count, 2)
            valid = torch.as_tensor(valid_flat_np, dtype=torch.bool, device=device).reshape(-1, pixel_count, 2)[..., 0]
            state = torch.as_tensor(latent[start:stop], dtype=torch.float32, device=device)
            prediction = (mean[None, :] + state @ basis.mT).reshape(-1, pixel_count, 2)
            unobserved = valid.clone()
            if excluded_pixels.numel():
                unobserved[:, excluded_pixels] = False
            vector_error += float(torch.sum((prediction[unobserved] - truth[unobserved]) ** 2))
            vector_truth += float(torch.sum(truth[unobserved] ** 2))
            truth_speed = torch.linalg.vector_norm(truth, dim=-1)
            prediction_speed = torch.linalg.vector_norm(prediction, dim=-1)
            selected_truth = truth_speed[unobserved].to(torch.float64)
            selected_prediction = prediction_speed[unobserved].to(torch.float64)
            speed_error += float(torch.sum((selected_prediction - selected_truth) ** 2))
            speed_truth += float(torch.sum(selected_truth**2))
            speed_count += int(selected_truth.numel())
            sx += float(torch.sum(selected_truth))
            sy += float(torch.sum(selected_prediction))
            sxx += float(torch.sum(selected_truth**2))
            syy += float(torch.sum(selected_prediction**2))
            sxy += float(torch.sum(selected_truth * selected_prediction))
            truth_grid = truth.reshape(-1, height, width, 2)
            valid_grid = valid.reshape(-1, height, width)
            upstream = valid_grid & upstream_x[None, None, :]
            upstream_sum += float(torch.sum(truth_grid[..., 0][upstream]))
            upstream_count += int(torch.sum(upstream))
            probe_truth[start:stop] = truth[:, probe_pixel, 1].cpu().numpy()
            probe_prediction[start:stop] = prediction[:, probe_pixel, 1].cpu().numpy()

    metrics = {
        "vector_nrmse": math.sqrt(vector_error / max(vector_truth, 1e-30)),
        "speed_nrmse": math.sqrt(speed_error / max(speed_truth, 1e-30)),
        "speed_pearson": _correlation_from_sums(speed_count, sx, sy, sxx, syy, sxy),
        "free_stream_velocity_m_s": upstream_sum / max(upstream_count, 1),
        "probe_x_over_d": float(x_over_d[probe_ix]),
        "probe_y_over_d": float(y_over_d[probe_iy]),
    }
    return metrics, {
        "probe_truth": probe_truth,
        "probe_prediction": probe_prediction,
        "time_s": np.asarray(case.time_s, dtype=float),
    }


def _spectrum(values: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(values, dtype=np.float64)
    signal = signal - np.mean(signal)
    frequency, power = welch(
        signal,
        fs=1.0 / dt,
        window="hann",
        nperseg=min(512, signal.size),
        noverlap=min(256, signal.size // 2),
        detrend="constant",
        scaling="density",
    )
    integral = float(np.trapezoid(power, frequency))
    return frequency, power / max(integral, 1e-30)


def _dominant_frequency(frequency: np.ndarray, power: np.ndarray) -> float:
    keep = (frequency >= 0.10) & (frequency <= 2.5)
    if not np.any(keep):
        return float("nan")
    return float(frequency[keep][np.argmax(power[keep])])


def _rolling(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="same")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--result-root", type=pathlib.Path, required=True)
    parser.add_argument("--sparse-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--block", type=int, default=16)
    args = parser.parse_args()

    config = load_config(args.config)
    diameter_m = float(config["cylinder_diameter_m"])
    data_paths = list_cases(pathlib.Path(config["data_root"]))
    model_root = args.result_root / "models" / "rank256_stride1"
    pod = PODModel.load(model_root / "pod_model.npz")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device}, but CUDA is unavailable")
    args.output.mkdir(parents=True, exist_ok=True)
    _style()

    accuracy_rows: list[dict[str, object]] = []
    probe_series: dict[str, dict[str, np.ndarray]] = {}
    traces: dict[str, dict[str, np.ndarray]] = {}
    run_payloads: dict[tuple[str, str], dict[str, object]] = {}
    for case_id in CASES:
        case = VIVCase.open(data_paths[case_id])
        apce_trace_path = _trace_path(args.result_root, case_id, "apce")
        with np.load(apce_trace_path, allow_pickle=False) as trace:
            latent = np.asarray(trace["latent_estimate"], dtype=np.float32)
            traces[case_id] = {key: np.asarray(trace[key]) for key in trace.files}
        sensor_archive = np.load(
            model_root / "sensor_layouts" / "adaptive_fullfield_valid" / f"case_{case_id}.npz",
            allow_pickle=False,
        )
        sensor_indices = np.asarray(sensor_archive["sensor_flat_indices"], dtype=np.int64)
        metrics, series = _evaluate_case(
            case, pod, latent, sensor_indices, device, diameter_m, args.block
        )
        payload = _load_json(_run_path(args.result_root, case_id, "apce"))
        run_payloads[(case_id, "apce")] = payload
        run_payloads[(case_id, "pce")] = _load_json(_run_path(args.result_root, case_id, "pce"))
        recorded = float(payload["unobserved_full_field_physical_nrmse"])
        if not np.isclose(metrics["vector_nrmse"], recorded, rtol=2e-4, atol=2e-5):
            raise ValueError(f"Full-field nRMSE mismatch for {case_id}: {metrics['vector_nrmse']} vs {recorded}")
        accuracy_rows.append({
            "case_id": case_id,
            "reduced_velocity": int(case_id) / 100.0,
            **metrics,
        })
        probe_series[case_id] = series

    # 01: five-case held-out speed accuracy.
    x = np.arange(len(CASES))
    speed_nrmse = np.asarray([float(row["speed_nrmse"]) for row in accuracy_rows])
    speed_corr = np.asarray([float(row["speed_pearson"]) for row in accuracy_rows])
    fig, axis = plt.subplots(figsize=(4.25, 2.9), constrained_layout=True)
    line_nrmse = axis.plot(x, speed_nrmse, color=APCE_COLOR, marker="o", ms=4.0, lw=1.4, label="speed nRMSE")
    axis.set_ylabel("speed nRMSE", color=APCE_COLOR)
    axis.tick_params(axis="y", colors=APCE_COLOR)
    axis.set_ylim(0.0, max(0.30, 1.18 * float(speed_nrmse.max())))
    axis.set_xticks(x, CASE_LABELS)
    axis.set_xlabel(r"reduced velocity $U_r$")
    second = axis.twinx()
    line_corr = second.plot(x, speed_corr, color=TRUTH_COLOR, marker="s", ms=3.6, lw=1.2,
                            linestyle=(0, (3, 2)), label="Pearson correlation")
    second.set_ylabel("Pearson correlation", color=TRUTH_COLOR)
    second.set_ylim(min(0.75, float(speed_corr.min()) - 0.03), 1.0)
    second.spines["right"].set_visible(True)
    axis.legend(line_nrmse + line_corr, ["speed nRMSE", "Pearson correlation"], loc="upper center", ncol=2)
    axis.set_title("Five external VIV conditions", fontweight="normal")
    _save(fig, args.output, "01_five_case_accuracy")
    _write_csv(args.output / "01_five_case_accuracy.csv", accuracy_rows)

    # 02 and 03: representative full-field kinetic energy and spectrum.
    representative = "0679"
    trace = traces[representative]
    time_s = np.asarray(trace["time_s"], dtype=float)
    time_relative = time_s - time_s[0]
    truth_energy = np.asarray(trace["truth_energy"], dtype=float)
    prediction_energy = np.asarray(trace["predicted_energy"], dtype=float)
    energy_scale = float(np.mean(truth_energy))
    fig, axis = plt.subplots(figsize=(4.25, 2.75), constrained_layout=True)
    axis.plot(time_relative, truth_energy / energy_scale, color=TRUTH_COLOR, lw=1.0, label="Measured")
    axis.plot(time_relative, prediction_energy / energy_scale, color=APCE_COLOR, lw=0.9, label="APCE")
    axis.set_xlabel("time (s)")
    axis.set_ylabel(r"$E(t)/\langle E_{\rm meas}\rangle$")
    axis.set_title(r"Kinetic energy, $U_r=6.79$", fontweight="normal")
    axis.legend(ncol=2)
    axis.margins(x=0.01)
    _save(fig, args.output, "02_kinetic_energy_timeseries")
    _write_csv(args.output / "02_kinetic_energy_timeseries.csv", [
        {"time_s": float(t), "measured": float(a), "apce": float(b)}
        for t, a, b in zip(time_relative, truth_energy, prediction_energy)
    ])

    dt = float(np.median(np.diff(time_s)))
    energy_frequency, energy_truth_psd = _spectrum(truth_energy, dt)
    _, energy_prediction_psd = _spectrum(prediction_energy, dt)
    energy_peak_truth = _dominant_frequency(energy_frequency, energy_truth_psd)
    energy_peak_prediction = _dominant_frequency(energy_frequency, energy_prediction_psd)
    keep = (energy_frequency > 0.0) & (energy_frequency <= 2.5)
    fig, axis = plt.subplots(figsize=(4.15, 2.85), constrained_layout=True)
    axis.semilogy(energy_frequency[keep], energy_truth_psd[keep], color=TRUTH_COLOR, lw=1.2, label="Measured")
    axis.semilogy(energy_frequency[keep], energy_prediction_psd[keep], color=APCE_COLOR, lw=1.0, label="APCE")
    axis.axvline(energy_peak_truth, color=TRUTH_COLOR, lw=0.7, linestyle=(0, (3, 2)))
    axis.axvline(energy_peak_prediction, color=APCE_COLOR, lw=0.7, linestyle=(0, (3, 2)))
    axis.set_xlabel("frequency (Hz)")
    axis.set_ylabel("normalised PSD")
    axis.set_title(r"Kinetic-energy spectrum, $U_r=6.79$", fontweight="normal")
    axis.legend(ncol=2)
    _save(fig, args.output, "03_kinetic_energy_psd")
    _write_csv(args.output / "03_kinetic_energy_psd.csv", [
        {"frequency_hz": float(f), "measured_psd": float(a), "apce_psd": float(b)}
        for f, a, b in zip(energy_frequency, energy_truth_psd, energy_prediction_psd)
    ])

    # 04: shedding-frequency and Strouhal-number agreement from a fixed wake probe.
    strouhal_rows: list[dict[str, object]] = []
    for row in accuracy_rows:
        case_id = str(row["case_id"])
        series = probe_series[case_id]
        local_dt = float(np.median(np.diff(series["time_s"])))
        frequency, truth_psd = _spectrum(series["probe_truth"], local_dt)
        _, prediction_psd = _spectrum(series["probe_prediction"], local_dt)
        truth_frequency = _dominant_frequency(frequency, truth_psd)
        prediction_frequency = _dominant_frequency(frequency, prediction_psd)
        u_inf = float(row["free_stream_velocity_m_s"])
        truth_st = truth_frequency * diameter_m / u_inf
        prediction_st = prediction_frequency * diameter_m / u_inf
        strouhal_rows.append({
            "case_id": case_id,
            "reduced_velocity": float(row["reduced_velocity"]),
            "probe_x_over_d": float(row["probe_x_over_d"]),
            "probe_y_over_d": float(row["probe_y_over_d"]),
            "measured_frequency_hz": truth_frequency,
            "apce_frequency_hz": prediction_frequency,
            "measured_strouhal": truth_st,
            "apce_strouhal": prediction_st,
            "relative_frequency_error": abs(prediction_frequency - truth_frequency) / max(truth_frequency, 1e-12),
        })
    measured_st = np.asarray([float(row["measured_strouhal"]) for row in strouhal_rows])
    apce_st = np.asarray([float(row["apce_strouhal"]) for row in strouhal_rows])
    lower = 0.92 * min(float(measured_st.min()), float(apce_st.min()))
    upper = 1.08 * max(float(measured_st.max()), float(apce_st.max()))
    fig, axis = plt.subplots(figsize=(3.25, 3.1), constrained_layout=True)
    axis.plot([lower, upper], [lower, upper], color="#8A8A8A", lw=0.8, linestyle=(0, (3, 2)))
    for color, label, truth_st, pred_st, row in zip(CASE_COLORS, CASE_LABELS, measured_st, apce_st, strouhal_rows):
        axis.scatter(truth_st, pred_st, s=32, color=color, edgecolor="white", linewidth=0.5, zorder=3)
        axis.annotate(label, (truth_st, pred_st), xytext=(3, 3), textcoords="offset points", fontsize=6.0)
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"measured $St$")
    axis.set_ylabel(r"APCE $St$")
    axis.set_title("Wake-probe Strouhal number", fontweight="normal")
    _save(fig, args.output, "04_strouhal_agreement")
    _write_csv(args.output / "04_strouhal_agreement.csv", strouhal_rows)

    # 05: vector-direction error at the representative transition-condition frame.
    field_source = args.result_root / "figures" / "fig11_adaptive_fullfield_apce_uv_speed_source.npz"
    field_metadata = _load_json(args.result_root / "figures" / "fig11_adaptive_fullfield_apce_uv_speed_metadata.json")
    with np.load(field_source, allow_pickle=False) as source:
        truth_field = np.asarray(source[f"{representative}_truth"], dtype=float)
        prediction_field = np.asarray(source[f"{representative}_apce"], dtype=float)
        valid_field = np.asarray(source[f"{representative}_valid"], dtype=bool)
        x_over_d = np.asarray(source[f"{representative}_x_over_d"], dtype=float)
        y_over_d = np.asarray(source[f"{representative}_y_over_d"], dtype=float)
    frame_record = next(row for row in field_metadata["cases"] if row["case_id"] == representative)
    frame = int(frame_record["frame"])
    representative_case = VIVCase.open(data_paths[representative])
    cylinder_y = float(representative_case.cyl_displ_m[frame] / diameter_m)
    truth_speed = np.linalg.norm(truth_field, axis=-1)
    prediction_speed = np.linalg.norm(prediction_field, axis=-1)
    threshold = 0.05 * float(next(row["free_stream_velocity_m_s"] for row in accuracy_rows if row["case_id"] == representative))
    direction_valid = valid_field & (truth_speed >= threshold) & (prediction_speed >= threshold)
    cosine = np.sum(truth_field * prediction_field, axis=-1) / np.maximum(truth_speed * prediction_speed, 1e-12)
    direction_error = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    direction_limit = min(90.0, max(30.0, float(np.percentile(direction_error[direction_valid], 99.0))))
    fig, axis = plt.subplots(figsize=(4.7, 2.65), constrained_layout=True)
    image = axis.pcolormesh(
        x_over_d, y_over_d, np.where(direction_valid, direction_error, np.nan),
        shading="auto", cmap="magma", norm=Normalize(0.0, direction_limit), rasterized=True,
    )
    axis.add_patch(Circle((0.0, cylinder_y), 0.5, facecolor="white", edgecolor=TRUTH_COLOR, lw=0.65))
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"$x/D$")
    axis.set_ylabel(r"$y/D$")
    axis.set_title(r"Vector direction error, $U_r=6.79$", fontweight="normal")
    colorbar = fig.colorbar(image, ax=axis, fraction=0.04, pad=0.025)
    colorbar.set_label(r"direction error ($^\circ$)")
    _save(fig, args.output, "05_vector_direction_error")
    direction_summary = [{
        "case_id": representative,
        "frame": frame,
        "time_s": float(frame_record["time_s"]),
        "median_error_deg": float(np.median(direction_error[direction_valid])),
        "p90_error_deg": float(np.percentile(direction_error[direction_valid], 90.0)),
        "valid_pixels": int(direction_valid.sum()),
    }]
    _write_csv(args.output / "05_vector_direction_error.csv", direction_summary)

    # 06: sensor-count sensitivity using consistent full-field layouts.
    density_rows: list[dict[str, object]] = []
    for case_id in CASES:
        sparse_path = _single(
            args.sparse_root / "runs" / "rank256_stride1" / "runs",
            f"viv_{case_id}_apce_seed000_layoutsparse2x4_fullfield_covfull_shr050.json",
        )
        sparse_payload = _load_json(sparse_path)
        dense_payload = run_payloads[(case_id, "apce")]
        for label, payload in (("2x4 full field", sparse_payload), ("adaptive 20x40", dense_payload)):
            density_rows.append({
                "case_id": case_id,
                "reduced_velocity": int(case_id) / 100.0,
                "layout": label,
                "sensor_points": int(payload["sensor_density_points"]),
                "unobserved_full_field_nrmse": float(payload["unobserved_full_field_physical_nrmse"]),
            })
    fig, axis = plt.subplots(figsize=(3.75, 2.85), constrained_layout=True)
    positions = np.asarray([0.0, 1.0])
    paired = []
    for color, case_id, case_label in zip(CASE_COLORS, CASES, CASE_LABELS):
        rows = [row for row in density_rows if row["case_id"] == case_id]
        rows.sort(key=lambda row: int(row["sensor_points"]))
        values = np.asarray([float(row["unobserved_full_field_nrmse"]) for row in rows])
        paired.append(values)
        axis.plot(positions, values, color=color, lw=0.9, marker="o", ms=3.4, label=rf"$U_r={case_label}$")
    paired_array = np.asarray(paired)
    axis.plot(positions, paired_array.mean(axis=0), color=TRUTH_COLOR, lw=1.6, marker="s", ms=4.0, label="mean")
    axis.set_xticks(positions, ["8 points\n(2x4)", "745 points\n(adaptive 20x40)"])
    axis.set_ylabel("unobserved full-field nRMSE")
    axis.set_title("Sensor-count sensitivity", fontweight="normal")
    axis.legend(ncol=2, fontsize=5.7)
    _save(fig, args.output, "06_sensor_count_sensitivity")
    _write_csv(args.output / "06_sensor_count_sensitivity.csv", density_rows)

    # 07: blackout forecast degradation across 20 origins per case.
    blackout_rows: list[dict[str, object]] = []
    grouped: dict[str, dict[float, list[float]]] = {}
    for case_id in CASES:
        rows = json.loads(str(traces[case_id]["blackout_rows_json"].item()))
        local: dict[float, list[float]] = defaultdict(list)
        for row in rows:
            horizon = float(row["horizon_s"])
            value = float(row["evaluation_nrmse"])
            local[horizon].append(value)
            blackout_rows.append({"case_id": case_id, **row})
        grouped[case_id] = local
    horizons = np.asarray(sorted(grouped[CASES[0]]), dtype=float)
    case_blackout = []
    fig, axis = plt.subplots(figsize=(4.0, 2.9), constrained_layout=True)
    for color, case_id, case_label in zip(CASE_COLORS, CASES, CASE_LABELS):
        means = np.asarray([np.mean(grouped[case_id][h]) for h in horizons])
        case_blackout.append(means)
        axis.plot(horizons, means, color=color, marker="o", ms=3.2, lw=0.9, label=rf"$U_r={case_label}$")
    case_blackout_array = np.asarray(case_blackout)
    axis.plot(horizons, case_blackout_array.mean(axis=0), color=TRUTH_COLOR, marker="s", ms=3.8, lw=1.5, label="mean")
    axis.set_xlabel("blackout horizon (s)")
    axis.set_ylabel("held-out nRMSE")
    axis.set_title("Observation-blackout forecast", fontweight="normal")
    axis.set_xticks(horizons)
    axis.legend(ncol=2, fontsize=5.7)
    _save(fig, args.output, "07_blackout_degradation")
    _write_csv(args.output / "07_blackout_degradation.csv", blackout_rows)

    # 08: CRPS, coverage and interval-width calibration frontier.
    calibration_rows: list[dict[str, object]] = []
    for case_id in CASES:
        for method in ("pce", "apce"):
            payload = run_payloads[(case_id, method)]
            calibration_rows.append({
                "case_id": case_id,
                "reduced_velocity": int(case_id) / 100.0,
                "method": method.upper(),
                "normalized_crps": float(payload["normalized_crps"]),
                "coverage_90": float(payload["coverage_90"]),
                "normalized_interval_width_90": float(payload["normalized_interval_width_90"]),
            })
    coverage_values = np.asarray([float(row["coverage_90"]) for row in calibration_rows])
    norm = Normalize(vmin=min(0.90, float(coverage_values.min())), vmax=1.0)
    cmap = plt.get_cmap("viridis")
    fig, axis = plt.subplots(figsize=(3.8, 3.0), constrained_layout=True)
    for row in calibration_rows:
        marker = "s" if row["method"] == "APCE" else "o"
        axis.scatter(
            float(row["normalized_interval_width_90"]), float(row["normalized_crps"]),
            s=35, marker=marker, color=cmap(norm(float(row["coverage_90"]))),
            edgecolor=TRUTH_COLOR, linewidth=0.45, zorder=3,
        )
        label_offset = (3, 4) if row["method"] == "PCE" else (3, -9)
        axis.annotate(
            f"{float(row['reduced_velocity']):.2f}",
            (float(row["normalized_interval_width_90"]), float(row["normalized_crps"])),
            xytext=label_offset, textcoords="offset points", fontsize=5.5,
        )
    axis.set_xlabel("normalised 90% interval width")
    axis.set_ylabel("normalised CRPS")
    axis.set_title("Calibration--sharpness frontier", fontweight="normal")
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#B5B5B5", markeredgecolor=TRUTH_COLOR, label="PCE"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#B5B5B5", markeredgecolor=TRUTH_COLOR, label="APCE"),
    ]
    axis.legend(handles=legend, ncol=2, loc="upper left")
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(scalar, ax=axis, fraction=0.045, pad=0.025)
    colorbar.set_label("90% coverage")
    _save(fig, args.output, "08_calibration_frontier")
    _write_csv(args.output / "08_calibration_frontier.csv", calibration_rows)

    # 09: shadow candidate separation and analysis-feedback erasure.
    mechanism_trace = traces[representative]
    mechanism_time = np.asarray(mechanism_trace["time_s"], dtype=float)[1:]
    separation = np.asarray(mechanism_trace["separation"], dtype=float)
    erasure = np.asarray(mechanism_trace["erasure"], dtype=float)
    window = max(3, int(round(1.0 / dt)))
    fig, axis = plt.subplots(figsize=(4.15, 2.9), constrained_layout=True)
    separation_line = axis.plot(mechanism_time - mechanism_time[0], _rolling(separation, window),
                                color=APCE_COLOR, lw=1.25, label="shadow separation")
    axis.axhline(1.0, color=APCE_COLOR, lw=0.65, linestyle=(0, (3, 2)), alpha=0.7)
    axis.set_yscale("log")
    axis.set_ylabel("separation / score uncertainty", color=APCE_COLOR)
    axis.tick_params(axis="y", colors=APCE_COLOR)
    axis.set_xlabel("time (s)")
    second = axis.twinx()
    erasure_line = second.plot(mechanism_time - mechanism_time[0], _rolling(erasure, window),
                               color=PCE_COLOR, lw=1.1, label="analysis erasure ratio")
    second.axhline(1.0, color=PCE_COLOR, lw=0.65, linestyle=(0, (3, 2)), alpha=0.7)
    second.set_ylabel("post/pre candidate variance", color=PCE_COLOR)
    second.tick_params(axis="y", colors=PCE_COLOR)
    second.spines["right"].set_visible(True)
    axis.set_title(r"Shadow protection, $U_r=6.79$", fontweight="normal")
    axis.legend(separation_line + erasure_line, ["shadow separation", "analysis erasure ratio"], loc="lower right")
    _save(fig, args.output, "09_shadow_separation_erasure")
    _write_csv(args.output / "09_shadow_separation_erasure.csv", [
        {"time_s": float(t), "separation_ratio": float(a), "erasure_ratio": float(b)}
        for t, a, b in zip(mechanism_time, separation, erasure)
    ])

    # 10: APCE candidate weights and score gap.
    weights = np.asarray(mechanism_trace["weights"], dtype=float)
    candidate_grid = np.asarray(mechanism_trace["candidate_grid"], dtype=float)
    scores = np.asarray(mechanism_trace["scores"], dtype=float)
    sorted_scores = np.sort(scores, axis=1)
    score_gap = sorted_scores[:, -1] - sorted_scores[:, -2]
    weighted_coordinate = weights @ candidate_grid
    fig = plt.figure(figsize=(4.4, 3.75), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=(2.2, 1.0), hspace=0.08)
    upper_axis = fig.add_subplot(grid[0])
    lower_axis = fig.add_subplot(grid[1], sharex=upper_axis)
    local_time = mechanism_time - mechanism_time[0]
    image = upper_axis.pcolormesh(
        local_time, candidate_grid, weights.T, shading="nearest", cmap="Blues", vmin=0.0,
        vmax=max(0.40, float(np.percentile(weights, 99.5))), rasterized=True,
    )
    upper_axis.plot(local_time, weighted_coordinate, color="#B64342", lw=1.0, label="weighted coordinate")
    upper_axis.axhline(6.79, color=TRUTH_COLOR, lw=0.7, linestyle=(0, (3, 2)), label=r"test $U_r$")
    upper_axis.set_ylabel(r"candidate $U_r$")
    upper_axis.set_title("APCE candidate evidence", fontweight="normal")
    upper_axis.tick_params(labelbottom=False)
    upper_axis.legend(ncol=2, fontsize=5.8, loc="upper right")
    colorbar = fig.colorbar(image, ax=upper_axis, fraction=0.035, pad=0.025)
    colorbar.set_label("weight")
    lower_axis.plot(local_time, score_gap, color="#42949E", lw=0.9)
    lower_axis.set_xlabel("time (s)")
    lower_axis.set_ylabel("top-two score gap")
    _save(fig, args.output, "10_candidate_weights_score_gap")
    np.savez_compressed(
        args.output / "10_candidate_weights_score_gap_source.npz",
        time_s=mechanism_time,
        candidate_grid=candidate_grid,
        weights=weights,
        scores=scores,
        score_gap=score_gap,
        weighted_coordinate=weighted_coordinate,
    )

    metadata = {
        "figure_contract": {
            "core_conclusion": "Sparse full-field PIV evidence supports accurate, calibrated and dynamically meaningful VIV wake reconstruction.",
            "archetype": "ten independent quantitative candidate panels",
            "backend": "Python/matplotlib only",
            "exports": ["SVG", "PDF", "PNG", "CSV/NPZ source data"],
        },
        "protocol": {
            "main_layout": "adaptive_fullfield_valid",
            "main_sensor_points": 745,
            "sparse_layout": "sparse2x4_fullfield",
            "sparse_sensor_points": 8,
            "pod_rank": 256,
            "ensemble_size": 64,
            "method": "APCE unless explicitly compared with PCE",
            "test_cases": list(CASES),
            "seeds": "seed 0 for current full-field screening; blackout variability uses 20 origins per case",
        },
        "validation": {
            "full_field_metrics_recomputed_from_raw_fields": True,
            "sensor_points_excluded_from_accuracy_metrics": True,
            "strouhal_probe": {"x_over_d": 2.0, "y_over_d": 0.0},
            "direction_frame_rule": "maximum absolute centred cylinder displacement",
        },
        "review_risks": [
            "The current adaptive full-field protocol has one algorithmic seed; five conditions and blackout origins are not independent experimental repeats.",
            "The sensor-count panel currently contains two audited densities (8 and 745), not a continuous density sweep.",
            "Candidate weights are operational predictive evidence, not a Bayesian posterior or a true physical parameter estimate.",
        ],
        "summary": {
            "mean_speed_nrmse": float(speed_nrmse.mean()),
            "mean_speed_pearson": float(speed_corr.mean()),
            "energy_peak_measured_hz": energy_peak_truth,
            "energy_peak_apce_hz": energy_peak_prediction,
            "direction_median_deg": direction_summary[0]["median_error_deg"],
            "direction_p90_deg": direction_summary[0]["p90_error_deg"],
            "identifiable_time_fraction": float(np.mean(separation > 1.0)),
            "mean_erasure_ratio": float(np.mean(erasure)),
        },
    }
    (args.output / "priority10_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    readme = "# VIV-PIV 主文优先十图\n\n"
    readme += "协议：rank-256、APCE、745 个全场自适应有效测点；8 点结果仅用于极稀疏敏感性。每张图独立导出为 PNG、SVG 和 PDF。\n\n"
    names = [
        ("01", "五工况速度 nRMSE 与 Pearson 相关"),
        ("02", "全场动能时间序列"),
        ("03", "全场动能 PSD（功率谱密度）"),
        ("04", "尾迹测点 Strouhal 数一致性"),
        ("05", "速度矢量方向误差"),
        ("06", "8 点与 745 点测点敏感性"),
        ("07", "观测中断预测退化"),
        ("08", "CRPS--覆盖率--区间宽度校准前沿"),
        ("09", "shadow 分离与 analysis 擦除比"),
        ("10", "候选权重与 score gap"),
    ]
    for prefix, description in names:
        png = next(args.output.glob(f"{prefix}_*.png")).name
        readme += f"- [{prefix} {description}]({png})\n"
    readme += "\n详细数值、协议和证据边界见 `priority10_metadata.json`，各图绘图数据见同名 CSV 或 NPZ。\n"
    (args.output / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(metadata["summary"], indent=2))


if __name__ == "__main__":
    main()
