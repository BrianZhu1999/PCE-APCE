"""Prepare authoritative lightweight source data for Figure 5 panels h--j.

Run this script on Super-Server. It reads the formal5 APCE traces, POD model,
and public VIV-PIV archives, then writes a compact NPZ plus CSV/JSON provenance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import welch


CASES = ("0463", "0556", "0679", "0803", "1359")
UR = {case: int(case) / 100.0 for case in CASES}


def sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def spectrum(values: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(values, dtype=np.float64)
    signal = signal - np.mean(signal)
    nperseg = min(512, signal.size)
    frequency, power = welch(
        signal,
        fs=1.0 / dt,
        window="hann",
        nperseg=nperseg,
        noverlap=min(256, signal.size // 2),
        detrend="constant",
        scaling="density",
    )
    integral = float(np.trapezoid(power, frequency))
    return frequency, power / max(integral, 1e-30)


def dominant_frequency(frequency: np.ndarray, power: np.ndarray) -> float:
    keep = (frequency >= 0.10) & (frequency <= 2.5)
    if not np.any(keep):
        raise ValueError("No frequency samples in the registered Strouhal band")
    return float(frequency[keep][np.argmax(power[keep])])


def trace_path(result_root: Path, case_id: str, layout_suffix: str) -> Path:
    return (
        result_root
        / "runs"
        / "rank256_stride1"
        / "traces"
        / f"viv_{case_id}_apce_seed000_layoutadaptive_fullfield_valid{layout_suffix}_ens064_covfull_shr050.npz"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--layout-suffix",
        default="",
        help="Suffix after layoutadaptive_fullfield_valid, e.g. _x40y20 for the corrected 40-by-20 layout.",
    )
    parser.add_argument("--diameter-m", type=float, default=0.05)
    parser.add_argument("--warmup-s", type=float, default=2.0)
    args = parser.parse_args()

    sys.path.insert(0, str(args.code_root))
    from hybrid_uncertain_wave.viv_piv_case.io import VIVCase

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pod_path = args.result_root / "models" / "rank256_stride1" / "pod_model.npz"
    with np.load(pod_path, allow_pickle=False) as pod:
        mean = np.asarray(pod["mean"], dtype=np.float32)
        basis = np.asarray(pod["basis"], dtype=np.float32)

    npz_source: dict[str, np.ndarray] = {}
    h_rows: list[dict[str, object]] = []
    provenance_cases: list[dict[str, object]] = []
    all_speed: list[np.ndarray] = []
    frequency_resolution_hz: float | None = None

    for case_id in CASES:
        archive = args.data_root / f"reduced_velocity_{case_id}.npz"
        trace_file = trace_path(args.result_root, case_id, args.layout_suffix)
        case = VIVCase.open(archive)
        with np.load(trace_file, allow_pickle=False) as trace:
            latent = np.asarray(trace["latent_estimate"], dtype=np.float32)
            time_s = np.asarray(trace["time_s"], dtype=np.float64)
            candidate_grid = np.asarray(trace["candidate_grid"], dtype=np.float64)
            weights = np.asarray(trace["weights"], dtype=np.float64)
            scores = np.asarray(trace["scores"], dtype=np.float64)

        height = int(case.y_mm.size)
        width = int(case.x_mm.size)
        x_over_d = np.asarray(case.x_mm, dtype=np.float64) / (args.diameter_m * 1000.0)
        y_over_d = np.asarray(case.y_mm, dtype=np.float64) / (args.diameter_m * 1000.0)
        probe_ix = int(np.argmin(np.abs(x_over_d - 2.0)))
        probe_iy = int(np.argmin(np.abs(y_over_d - 0.0)))
        probe_pixel = probe_iy * width + probe_ix
        probe_rows = np.asarray([2 * probe_pixel, 2 * probe_pixel + 1], dtype=np.int64)

        norm_low = np.asarray(case.norm_values[0], dtype=np.float32)
        norm_high = np.asarray(case.norm_values[1], dtype=np.float32)
        probe_norm = np.asarray(case.velocities[:, probe_iy, probe_ix, :], dtype=np.float32)
        probe_truth = probe_norm * (norm_high - norm_low)[None, :] + norm_low[None, :]
        probe_prediction = mean[probe_rows][None, :] + latent @ basis[probe_rows, :].T

        upstream_columns = np.flatnonzero(x_over_d < -0.8)
        upstream_norm = np.asarray(case.velocities[:, :, upstream_columns, 0], dtype=np.float32)
        upstream_valid = np.asarray(case.mask[:, :, upstream_columns] > 0.5, dtype=bool)
        upstream_u = upstream_norm * (norm_high[0] - norm_low[0]) + norm_low[0]
        u_inf = float(np.mean(upstream_u[upstream_valid]))
        if not np.isfinite(u_inf) or u_inf <= 0.0:
            raise RuntimeError(f"Invalid upstream velocity for {case_id}: {u_inf}")

        dt = float(np.median(np.diff(case.time_s)))
        frequency, truth_psd = spectrum(probe_truth[:, 1], dt)
        _, prediction_psd = spectrum(probe_prediction[:, 1], dt)
        truth_frequency = dominant_frequency(frequency, truth_psd)
        prediction_frequency = dominant_frequency(frequency, prediction_psd)
        current_resolution = float((1.0 / dt) / min(512, case.time_s.size))
        if frequency_resolution_hz is None:
            frequency_resolution_hz = current_resolution
        elif not np.isclose(frequency_resolution_hz, current_resolution):
            raise ValueError("Inconsistent Welch frequency resolution across cases")
        h_rows.append({
            "case_id": case_id,
            "reduced_velocity": UR[case_id],
            "probe_x_over_d": float(x_over_d[probe_ix]),
            "probe_y_over_d": float(y_over_d[probe_iy]),
            "measured_frequency_hz": truth_frequency,
            "apce_frequency_hz": prediction_frequency,
            "measured_strouhal": truth_frequency * args.diameter_m / u_inf,
            "apce_strouhal": prediction_frequency * args.diameter_m / u_inf,
            "relative_frequency_error": abs(prediction_frequency - truth_frequency) / max(truth_frequency, 1e-12),
            "free_stream_velocity_m_s": u_inf,
            "welch_frequency_resolution_hz": current_resolution,
        })

        warmup = int(np.searchsorted(case.time_s, args.warmup_s))
        centred = case.cyl_displ_m - np.median(case.cyl_displ_m[warmup:])
        frame = warmup + int(np.argmax(np.abs(centred[warmup:])))
        truth_block, valid_block = case.physical_frames(frame, frame + 1)
        truth_field = np.asarray(truth_block[0], dtype=np.float32)
        valid = np.asarray(valid_block[0], dtype=bool)
        prediction_field = (
            mean + np.asarray(latent[frame], dtype=np.float32) @ basis.T
        ).reshape(height, width, 2)
        truth_speed = np.linalg.norm(truth_field, axis=-1)[valid].astype(np.float32)
        apce_speed = np.linalg.norm(prediction_field, axis=-1)[valid].astype(np.float32)
        npz_source[f"i_{case_id}_truth_speed"] = truth_speed
        npz_source[f"i_{case_id}_apce_speed"] = apce_speed
        all_speed.extend([truth_speed, apce_speed])

        score_sorted = np.sort(scores, axis=1)
        score_gap = score_sorted[:, -1] - score_sorted[:, -2]
        weighted_coordinate = weights @ candidate_grid
        npz_source[f"j_{case_id}_time_s"] = time_s[1:] - time_s[1]
        npz_source[f"j_{case_id}_candidate_grid"] = candidate_grid
        npz_source[f"j_{case_id}_weights"] = weights
        npz_source[f"j_{case_id}_score_gap"] = score_gap
        npz_source[f"j_{case_id}_weighted_coordinate"] = weighted_coordinate
        npz_source[f"j_{case_id}_target_ur"] = np.asarray(UR[case_id], dtype=np.float64)

        provenance_cases.append({
            "case_id": case_id,
            "reduced_velocity": UR[case_id],
            "truth_archive": str(archive),
            "apce_trace": str(trace_file),
            "representative_frame": int(frame),
            "representative_time_s": float(case.time_s[frame]),
            "valid_speed_samples": int(valid.sum()),
            "truth_speed_range_m_s": [float(truth_speed.min()), float(truth_speed.max())],
            "apce_speed_range_m_s": [float(apce_speed.min()), float(apce_speed.max())],
            "candidate_count": int(candidate_grid.size),
            "candidate_range": [float(candidate_grid.min()), float(candidate_grid.max())],
        })

    speed_max = float(max(np.max(values) for values in all_speed))
    speed_upper = np.ceil(speed_max * 100.0) / 100.0
    bin_edges = np.linspace(0.0, speed_upper, 97, dtype=np.float64)
    npz_source["i_speed_bin_edges_m_s"] = bin_edges
    npz_source["h_welch_frequency_resolution_hz"] = np.asarray(frequency_resolution_hz, dtype=np.float64)

    npz_path = args.output_dir / "figure5_hij_source.npz"
    np.savez_compressed(npz_path, **npz_source)
    h_csv_path = args.output_dir / "figure5_h_strouhal_source.csv"
    with h_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(h_rows[0]))
        writer.writeheader()
        writer.writerows(h_rows)

    provenance = {
        "figure": "figure5_hij_row",
        "backend": "Python/matplotlib",
        "formal_result_root": str(args.result_root),
        "public_data_root": str(args.data_root),
        "pod_model": str(pod_path),
        "diameter_m": args.diameter_m,
        "warmup_s_argument": args.warmup_s,
        "representative_frame_rule": "maximum absolute median-centred cylinder displacement after registered warm-up",
        "strouhal": {
            "probe_target_x_over_d": 2.0,
            "probe_target_y_over_d": 0.0,
            "welch_nperseg": 512,
            "welch_frequency_resolution_hz": frequency_resolution_hz,
            "dominant_frequency_band_hz": [0.10, 2.5],
            "note": "Measured and APCE peaks share the same resolved Welch bin in all five cases; no sub-bin interpolation is applied.",
        },
        "speed_pdf": {
            "sampling": "valid spatial points from one registered representative frame per case",
            "common_bin_count": int(bin_edges.size - 1),
            "common_bin_range_m_s": [float(bin_edges[0]), float(bin_edges[-1])],
        },
        "candidate_evidence_note": "Candidate weights are operational predictive evidence over candidate reduced velocities, not Bayesian posterior probabilities or direct physical inflow-speed measurements.",
        "cases": provenance_cases,
        "outputs": {
            "npz": str(npz_path),
            "h_csv": str(h_csv_path),
        },
    }
    provenance_path = args.output_dir / "figure5_hij_source_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    manifest = {
        "npz": {"path": str(npz_path), "sha256": sha256_file(npz_path)},
        "h_csv": {"path": str(h_csv_path), "sha256": sha256_file(h_csv_path)},
        "provenance": {"path": str(provenance_path), "sha256": sha256_file(provenance_path)},
    }
    manifest_path = args.output_dir / "figure5_hij_source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"provenance": provenance, "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
