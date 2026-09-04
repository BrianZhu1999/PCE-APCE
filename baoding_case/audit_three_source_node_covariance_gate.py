#!/usr/bin/env python3
"""Author-code-informed raw-covariance information gate for three targets.

This is not the authors' private Baoding implementation. It transfers the
supplied ReAVI code's node-wise sample covariance, source-energy and sensor-
noise precision mechanism to the wideband field data. Source/noise parameters
are fitted on one calibration second and frozen before the next-second spatial
likelihood audit. GPS is read only after scores have been frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import cupy as cp
import numpy as np


PAPER_NODES = (1, 3, 5, 6, 7, 8, 11, 13)
NODE_TO_IP = {1: 47, 3: 40, 5: 54, 6: 43, 7: 49, 8: 61, 11: 5, 13: 46}
CHANNEL_GROUPS = {
    "z": (19, 18, 17, 13, 14, 15, 16),
    "x": (9, 8, 7, 1, 2, 3),
    "y": (12, 11, 10, 4, 5, 6),
}
PAPER_XYV = np.asarray(
    (
        (38614853.4, 4337388.27, 25.85033, 23.00194),
        (38615012.2, 4336467.20, -41.09208, 6.67753),
        (38615647.2, 4337215.10, 3.20862, -41.49795),
    ),
    dtype=np.float64,
)
GPS_FILES = ("GPS1_plane1.gps", "GPS3_plane2.gps", "GPS4_plane2to3.gps")
FS = 3000
SOUND_SPEED_MPS = 340.0
PROCESSING_HEIGHT_M = 230.0
START_HHMMSS = 132754
TIME_SHUFFLE_SECONDS = (7, 13, 19, 23, 29, 31, 37, 41)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hms_seconds(value: str | int | float) -> float:
    text = str(int(float(value))).zfill(6)
    return int(text[:2]) * 3600 + int(text[2:4]) * 60 + int(text[4:])


def parse_nodes(path: Path) -> dict[int, tuple[float, float, float, int, int]]:
    nodes: dict[int, tuple[float, float, float, int, int]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        node = int(fields[2])
        horizontal_flip = int(float(fields[9])) if len(fields) > 9 else 0
        vertical_flip = int(float(fields[10])) if len(fields) > 10 else 0
        nodes[node] = (
            float(fields[3]),
            float(fields[4]),
            float(fields[5]),
            horizontal_flip,
            vertical_flip,
        )
    missing = sorted(set(PAPER_NODES) - set(nodes))
    if missing:
        raise ValueError(f"node file is missing paper nodes: {missing}")
    return nodes


def parse_gps(path: Path) -> list[tuple[float, float, float, float]]:
    rows: list[tuple[float, float, float, float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 8:
            continue
        try:
            rows.append(
                (
                    hms_seconds(fields[7]),
                    float(fields[4]),
                    float(fields[5]),
                    float(fields[6]),
                )
            )
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"no GPS rows parsed from {path}")
    return rows


def nearest_gps(rows: list[tuple[float, float, float, float]], time_s: float) -> np.ndarray:
    _, x, y, z = min(rows, key=lambda row: abs(row[0] - time_s))
    return np.asarray((x, y, z), dtype=np.float64)


def microphone_offsets(
    node: tuple[float, float, float, int, int], orientation_mode: str
) -> np.ndarray:
    horizontal_sign = -1.0 if orientation_mode == "nod_flip" and node[3] else 1.0
    vertical_sign = -1.0 if orientation_mode == "nod_flip" and node[4] else 1.0
    offsets = np.zeros((19, 3), dtype=np.float32)
    horizontal = (-1.5, -1.0, -0.5, 0.5, 1.0, 1.5)
    # Frozen legacy candidate selected by the existing calibration/evaluation gate.
    vertical = (-1.065, -0.765, -0.465, 0.0, 0.5, 1.0, 1.5)
    for channel, value in zip(CHANNEL_GROUPS["x"], horizontal):
        offsets[channel - 1, 0] = horizontal_sign * value
    for channel, value in zip(CHANNEL_GROUPS["y"], horizontal):
        offsets[channel - 1, 1] = horizontal_sign * value
    for channel, value in zip(CHANNEL_GROUPS["z"], vertical):
        offsets[channel - 1, 2] = vertical_sign * value
    return offsets


def stft_covariance(
    streams: list[np.ndarray],
    second_indices: list[int],
    snapshot_samples: int,
    hop_samples: int,
    nfft: int,
    frequency_bins: np.ndarray,
) -> tuple[cp.ndarray, int]:
    if snapshot_samples > FS:
        raise ValueError("snapshot_samples must not exceed one second")
    starts = list(range(0, FS - snapshot_samples + 1, hop_samples))
    if len(starts) < 4:
        raise ValueError("at least four covariance snapshots are required")
    window = cp.hanning(snapshot_samples).astype(cp.float32)
    normalization = cp.sqrt(cp.sum(window * window)).clip(1e-12)
    covariances = []
    for stream, second in zip(streams, second_indices):
        start = second * FS
        block_cpu = np.asarray(stream[:, start : start + FS], dtype=np.float32)
        if block_cpu.shape != (19, FS):
            raise ValueError(f"incomplete second {second}: {block_cpu.shape}")
        block = cp.asarray(block_cpu)
        block -= cp.mean(block, axis=1, keepdims=True)
        snapshots = cp.stack(
            [block[:, offset : offset + snapshot_samples] * window[None, :] for offset in starts],
            axis=0,
        )
        spectrum = cp.fft.rfft(snapshots, n=nfft, axis=-1)[:, :, frequency_bins]
        spectrum = spectrum.transpose(2, 1, 0).astype(cp.complex64) / normalization
        covariance = cp.einsum("fpl,fql->fpq", spectrum, cp.conj(spectrum)) / len(starts)
        covariances.append((covariance + cp.conj(covariance.transpose(0, 2, 1))) * 0.5)
    return cp.stack(covariances, axis=0), len(starts)


def steering(
    states_xyz: cp.ndarray,
    node_xyz: cp.ndarray,
    offsets: cp.ndarray,
    frequencies_hz: cp.ndarray,
    phase_sign: int,
) -> tuple[cp.ndarray, cp.ndarray]:
    # Output: node x frequency x target x microphone; node x target attenuation.
    microphones = node_xyz[:, None, :] + offsets
    delta = states_xyz[None, :, None, :] - microphones[:, None, :, :]
    distances = cp.sqrt(cp.sum(delta * delta, axis=-1).clip(1e-12))
    centre_delta = states_xyz[None, :, :] - node_xyz[:, None, :]
    centre_distance = cp.sqrt(cp.sum(centre_delta * centre_delta, axis=-1).clip(1e-12))
    relative_distance = distances - centre_distance[:, :, None]
    phase = cp.exp(
        phase_sign
        * 1j
        * 2.0
        * cp.pi
        * frequencies_hz[None, :, None, None]
        * relative_distance[:, None, :, :]
        / SOUND_SPEED_MPS
    )
    phase /= math.sqrt(offsets.shape[1])
    attenuation = 1.0 / cp.square(centre_distance)
    return phase.astype(cp.complex64), attenuation.astype(cp.float32)


def fit_source_noise(
    observed: cp.ndarray,
    states_xyz: cp.ndarray,
    node_xyz: cp.ndarray,
    offsets: cp.ndarray,
    frequencies_hz: cp.ndarray,
    phase_sign: int,
    iterations: int,
) -> dict[str, cp.ndarray | float]:
    vectors, attenuation = steering(
        states_xyz, node_xyz, offsets, frequencies_hz, phase_sign
    )
    eigenvalues = cp.linalg.eigvalsh(observed).real.clip(0.0)
    noise = cp.mean(eigenvalues[:, :, : observed.shape[-1] - states_xyz.shape[0]], axis=-1)
    trace_observed = cp.trace(observed, axis1=-2, axis2=-1).real
    floor = cp.maximum(cp.median(trace_observed, axis=1, keepdims=True) * 1e-8, 1e-8)
    noise = cp.maximum(noise, floor)

    gram_vectors = cp.einsum("nfmp,nfjp->nfmj", cp.conj(vectors), vectors)
    gram = cp.sum(
        attenuation[:, None, :, None]
        * attenuation[:, None, None, :]
        * cp.abs(gram_vectors) ** 2,
        axis=0,
    ).real
    ridge = cp.eye(states_xyz.shape[0], dtype=cp.float32)[None, :, :] * 1e-8
    energy = cp.ones((len(frequencies_hz), states_xyz.shape[0]), dtype=cp.float32)
    identity = cp.eye(observed.shape[-1], dtype=cp.complex64)[None, None, :, :]
    for _ in range(iterations):
        residual = observed - noise[:, :, None, None] * identity
        projected = cp.einsum(
            "nfmp,nfpq,nfmq->nfm", cp.conj(vectors), residual, vectors
        ).real
        rhs = cp.sum(attenuation[:, None, :] * projected, axis=0)
        candidate = cp.linalg.solve(gram + ridge, rhs[:, :, None])[:, :, 0]
        energy = cp.maximum(candidate, 0.0)
        source_trace = cp.einsum("fm,nm->nf", energy, attenuation)
        noise = cp.maximum((trace_observed - source_trace) / observed.shape[-1], floor)

    source_covariance = cp.einsum(
        "fm,nm,nfmp,nfmq->nfpq",
        energy,
        attenuation,
        vectors,
        cp.conj(vectors),
    )
    model = source_covariance + noise[:, :, None, None] * identity
    residual = observed - model
    residual_fraction = float(
        cp.asnumpy(
            cp.sum(cp.abs(residual) ** 2)
            / cp.maximum(cp.sum(cp.abs(observed) ** 2), 1e-12)
        )
    )
    source_trace = cp.einsum("fm,nm->nfm", energy, attenuation)
    snr = source_trace / noise[:, :, None].clip(1e-12)
    return {
        "vectors": vectors,
        "attenuation": attenuation,
        "energy": energy,
        "noise": noise,
        "residual_fraction": residual_fraction,
        "median_snr": float(cp.asnumpy(cp.median(snr))),
        "p90_snr": float(cp.asnumpy(cp.percentile(snr, 90.0))),
    }


def candidate_grid(center_xy: np.ndarray, radius_m: float, spacing_m: float) -> np.ndarray:
    axis = np.arange(-radius_m, radius_m + spacing_m * 0.5, spacing_m, dtype=np.float32)
    xx, yy = np.meshgrid(axis + center_xy[0], axis + center_xy[1], indexing="xy")
    zz = np.full_like(xx, PROCESSING_HEIGHT_M)
    return np.stack((xx.ravel(), yy.ravel(), zz.ravel()), axis=1)


def score_target(
    observed: cp.ndarray,
    energy: cp.ndarray,
    noise: cp.ndarray,
    target: int,
    candidates_xyz: np.ndarray,
    predicted_states_xyz: cp.ndarray,
    node_xyz: cp.ndarray,
    offsets: cp.ndarray,
    frequencies_hz: cp.ndarray,
    phase_sign: int,
    batch_size: int,
) -> np.ndarray:
    other_targets = [index for index in range(predicted_states_xyz.shape[0]) if index != target]
    other_vectors, other_attenuation = steering(
        predicted_states_xyz[other_targets],
        node_xyz,
        offsets,
        frequencies_hz,
        phase_sign,
    )
    other_covariance = cp.einsum(
        "fj,nj,nfjp,nfjq->nfpq",
        energy[:, other_targets],
        other_attenuation,
        other_vectors,
        cp.conj(other_vectors),
    )
    identity = cp.eye(observed.shape[-1], dtype=cp.complex64)[None, None, :, :]
    residual = observed - noise[:, :, None, None] * identity - other_covariance
    precision = 1.0 / noise.clip(1e-12)
    low, high = cp.percentile(precision, cp.asarray((5.0, 95.0), dtype=cp.float32))
    precision = cp.clip(precision, low, high)

    outputs = []
    for start in range(0, len(candidates_xyz), batch_size):
        candidates = cp.asarray(candidates_xyz[start : start + batch_size], dtype=cp.float32)
        microphones = node_xyz[:, None, :] + offsets
        delta = candidates[:, None, None, :] - microphones[None, :, :, :]
        distances = cp.sqrt(cp.sum(delta * delta, axis=-1).clip(1e-12))
        centre_delta = candidates[:, None, :] - node_xyz[None, :, :]
        centre_distance = cp.sqrt(cp.sum(centre_delta * centre_delta, axis=-1).clip(1e-12))
        relative_distance = distances - centre_distance[:, :, None]
        vectors = cp.exp(
            phase_sign
            * 1j
            * 2.0
            * cp.pi
            * frequencies_hz[None, None, :, None]
            * relative_distance[:, :, None, :]
            / SOUND_SPEED_MPS
        ).astype(cp.complex64)
        vectors /= math.sqrt(offsets.shape[1])
        attenuation = 1.0 / cp.square(centre_distance)
        beam = cp.einsum(
            "bnfp,nfpq,bnfq->bnf", cp.conj(vectors), residual, vectors
        ).real
        # Profile the target's nonnegative source energy at every frequency.
        # This is the analytic coordinate update corresponding to the supplied
        # code's source-energy step before its bearing/state update.
        precision_squared = cp.square(precision)[None, :, :]
        weighted_linear = cp.sum(
            attenuation[:, :, None] * beam * precision_squared, axis=1
        )
        weighted_curvature = cp.sum(
            cp.square(attenuation[:, :, None]) * precision_squared, axis=1
        ).clip(1e-20)
        profiled_improvement = cp.square(cp.maximum(weighted_linear, 0.0)) / weighted_curvature
        outputs.append(cp.sum(profiled_improvement, axis=1, dtype=cp.float64))
    return cp.asnumpy(cp.concatenate(outputs))


def score_summary(
    candidates: np.ndarray,
    scores: np.ndarray,
    truth_xyz: np.ndarray,
    target: int,
) -> dict[str, float | int | list[float]]:
    candidates = np.asarray(candidates, dtype=float).reshape(-1, 3)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if len(candidates) != len(scores):
        raise ValueError(
            f"candidate/score length mismatch: {len(candidates)} != {len(scores)}"
        )
    truth_index = int(np.argmin(np.sum((candidates[:, :2] - truth_xyz[None, :2]) ** 2, axis=1)))
    best_index = int(np.argmax(scores))
    truth_score = float(scores[truth_index])
    tie_tolerance = max(abs(truth_score) * 1e-12, 1e-12)
    greater = int(np.sum(scores > truth_score + tie_tolerance))
    tied = int(np.sum(np.abs(scores - truth_score) <= tie_tolerance))
    rank = 1.0 + greater + 0.5 * max(tied - 1, 0)
    percentile = 1.0 if len(scores) == 1 else 1.0 - (rank - 1) / (len(scores) - 1)
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    return {
        "target": target + 1,
        "candidate_count": int(len(scores)),
        "truth_xyz_offline_m": truth_xyz.tolist(),
        "nearest_grid_truth_xyz_m": candidates[truth_index].astype(float).tolist(),
        "best_xyz_m": candidates[best_index].astype(float).tolist(),
        "best_error_xy_m_offline": float(np.linalg.norm(candidates[best_index, :2] - truth_xyz[:2])),
        "truth_midrank_1_is_best": float(rank),
        "truth_rank_percentile": float(percentile),
        "truth_score": truth_score,
        "best_score": float(scores[best_index]),
        "truth_minus_median_over_mad": float((truth_score - median) / max(mad, 1e-12)),
        "score_dynamic_range": float(np.max(scores) - np.min(scores)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync-root", type=Path, required=True)
    parser.add_argument("--nod", type=Path, required=True)
    parser.add_argument("--gps-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    parser.add_argument("--calibration-second", type=int, default=0)
    parser.add_argument("--evaluation-second", type=int, default=1)
    parser.add_argument("--snapshot-samples", type=int, default=512)
    parser.add_argument("--hop-samples", type=int, default=96)
    parser.add_argument("--nfft", type=int, default=3000)
    parser.add_argument("--radius-m", type=float, default=160.0)
    parser.add_argument("--spacing-m", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fit-iterations", type=int, default=8)
    args = parser.parse_args()

    started = time.perf_counter()
    cp.cuda.Device(args.gpu).use()
    nodes = parse_nodes(args.nod)
    node_xyz_np = np.asarray([nodes[node][:3] for node in PAPER_NODES], dtype=np.float32)
    node_xyz = cp.asarray(node_xyz_np)
    frequencies_np = np.arange(3, 1498, 3, dtype=np.int64)
    frequencies = cp.asarray(frequencies_np, dtype=cp.float32)
    streams = []
    source_paths = []
    for node in PAPER_NODES:
        path = args.sync_root / f"node{node}_ip{NODE_TO_IP[node]}_3khz.npy"
        streams.append(np.load(path, mmap_mode="r"))
        source_paths.append(path)
    available_seconds = min(stream.shape[1] // FS for stream in streams)
    if max(args.calibration_second, args.evaluation_second, max(TIME_SHUFFLE_SECONDS)) >= available_seconds:
        raise ValueError(f"requested control second exceeds {available_seconds} available seconds")

    calibration_covariance, snapshot_count = stft_covariance(
        streams,
        [args.calibration_second] * len(PAPER_NODES),
        args.snapshot_samples,
        args.hop_samples,
        args.nfft,
        frequencies_np,
    )
    evaluation_covariance, _ = stft_covariance(
        streams,
        [args.evaluation_second] * len(PAPER_NODES),
        args.snapshot_samples,
        args.hop_samples,
        args.nfft,
        frequencies_np,
    )
    shuffled_seconds = [
        (args.evaluation_second + offset) % available_seconds
        for offset in TIME_SHUFFLE_SECONDS
    ]
    shuffled_covariance, _ = stft_covariance(
        streams,
        shuffled_seconds,
        args.snapshot_samples,
        args.hop_samples,
        args.nfft,
        frequencies_np,
    )

    calibration_states = np.column_stack(
        (PAPER_XYV[:, :2], np.full(3, PROCESSING_HEIGHT_M, dtype=np.float64))
    )
    predicted_xy = PAPER_XYV[:, :2] + PAPER_XYV[:, 2:] * (
        args.evaluation_second - args.calibration_second
    )
    predicted_states = np.column_stack(
        (predicted_xy, np.full(3, PROCESSING_HEIGHT_M, dtype=np.float64))
    )

    variant_fits = []
    fit_objects = {}
    for orientation_mode in ("none", "nod_flip"):
        offsets_np = np.stack(
            [microphone_offsets(nodes[node], orientation_mode) for node in PAPER_NODES]
        )
        offsets = cp.asarray(offsets_np)
        for phase_sign in (-1, 1):
            name = f"legacy_nonuniform_z__{orientation_mode}__phase_{phase_sign:+d}"
            fit = fit_source_noise(
                calibration_covariance,
                cp.asarray(calibration_states, dtype=cp.float32),
                node_xyz,
                offsets,
                frequencies,
                phase_sign,
                args.fit_iterations,
            )
            fit_objects[name] = (fit, offsets, phase_sign, orientation_mode)
            variant_fits.append(
                {
                    "name": name,
                    "orientation_mode": orientation_mode,
                    "phase_sign": phase_sign,
                    "calibration_covariance_residual_fraction": fit["residual_fraction"],
                    "median_fitted_source_snr_linear": fit["median_snr"],
                    "p90_fitted_source_snr_linear": fit["p90_snr"],
                }
            )
    selected_variant = min(
        variant_fits, key=lambda row: row["calibration_covariance_residual_fraction"]
    )["name"]
    fit, selected_offsets, selected_phase_sign, selected_orientation_mode = fit_objects[
        selected_variant
    ]

    controls = {
        "real_evaluation": evaluation_covariance,
        "zero_input": cp.zeros_like(evaluation_covariance),
        "node_time_shuffled": shuffled_covariance,
    }
    acoustic_scores: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for control_name, covariance in controls.items():
        acoustic_scores[control_name] = []
        for target in range(3):
            candidates = candidate_grid(
                predicted_states[target, :2], args.radius_m, args.spacing_m
            )
            scores = score_target(
                covariance,
                fit["energy"],
                fit["noise"],
                target,
                candidates,
                cp.asarray(predicted_states, dtype=cp.float32),
                node_xyz,
                selected_offsets,
                frequencies,
                selected_phase_sign,
                args.batch_size,
            )
            acoustic_scores[control_name].append((candidates, scores))

    # GPS is deliberately parsed only after the acoustic variant and all score
    # surfaces have been frozen.
    evaluation_time = hms_seconds(START_HHMMSS) + args.evaluation_second
    gps_tracks = [parse_gps(args.gps_root / filename) for filename in GPS_FILES]
    truths = [nearest_gps(track, evaluation_time) for track in gps_tracks]
    score_results = {
        control_name: [
            score_summary(candidates, scores, truths[target], target)
            for target, (candidates, scores) in enumerate(target_scores)
        ]
        for control_name, target_scores in acoustic_scores.items()
    }
    real = score_results["real_evaluation"]
    zero = score_results["zero_input"]
    shuffled = score_results["node_time_shuffled"]
    real_mean_rank = float(np.mean([row["truth_rank_percentile"] for row in real]))
    zero_mean_rank = float(np.mean([row["truth_rank_percentile"] for row in zero]))
    shuffled_mean_rank = float(
        np.mean([row["truth_rank_percentile"] for row in shuffled])
    )
    selected_fit = next(row for row in variant_fits if row["name"] == selected_variant)
    gate = {
        "all_real_truth_rank_percentiles_at_least_0_95": all(
            row["truth_rank_percentile"] >= 0.95 for row in real
        ),
        "all_real_best_xy_errors_at_most_75m": all(
            row["best_error_xy_m_offline"] <= 75.0 for row in real
        ),
        "real_mean_rank_exceeds_zero_by_0_10": real_mean_rank >= zero_mean_rank + 0.10,
        "real_mean_rank_exceeds_shuffled_by_0_10": real_mean_rank
        >= shuffled_mean_rank + 0.10,
        "calibration_covariance_explained_fraction_at_least_0_05": 1.0
        - selected_fit["calibration_covariance_residual_fraction"]
        >= 0.05,
    }
    gate["passed"] = all(gate.values())

    payload = {
        "task": "author-code-informed per-node wideband covariance information gate",
        "status": "passed" if gate["passed"] else "failed",
        "claim_status": "first-frame information gate only; not a field tracker and not the authors private Baoding implementation",
        "author_code_increment": {
            "sample_covariance": "Code Paper/MTT_Main.py: per-node Ro = sig sig^H / frame_len",
            "node_noise_precision": "Code Paper/DBN_MTT.py: node-specific lambda_e update",
            "source_energy": "Code Paper/DBN_MTT.py: source-energy s_I update",
            "wideband_transfer": "applied independently at 3,6,...,1497 Hz with parameters frozen from calibration to evaluation",
        },
        "protocol": {
            "paper_nodes": list(PAPER_NODES),
            "microphones_per_node": 19,
            "sample_rate_hz": FS,
            "frequency_grid_hz": "3,6,...,1497",
            "frequency_count": len(frequencies_np),
            "calibration_second": args.calibration_second,
            "evaluation_second": args.evaluation_second,
            "evaluation_time_hhmmss": START_HHMMSS + args.evaluation_second,
            "covariance_snapshot_samples": args.snapshot_samples,
            "covariance_hop_samples": args.hop_samples,
            "covariance_snapshots": snapshot_count,
            "nfft": args.nfft,
            "processing_height_m": PROCESSING_HEIGHT_M,
            "geometry_profile": "legacy_nonuniform_z",
            "variant_selection": "minimum calibration covariance residual; no GPS score used",
            "position_grid_radius_m": args.radius_m,
            "position_grid_spacing_m": args.spacing_m,
            "time_shuffled_node_seconds": shuffled_seconds,
            "gps_runtime_observation": False,
            "gps_role": "parsed only after acoustic score surfaces were frozen; offline rank/error scoring only",
        },
        "selected_variant": selected_variant,
        "variant_calibration_fits": variant_fits,
        "score_results": score_results,
        "rank_summary": {
            "real_mean_truth_rank_percentile": real_mean_rank,
            "zero_mean_truth_rank_percentile": zero_mean_rank,
            "node_time_shuffled_mean_truth_rank_percentile": shuffled_mean_rank,
        },
        "admission_gate": gate,
        "inputs": {
            "sync_root": str(args.sync_root),
            "node_file": str(args.nod),
            "node_file_sha256": sha256(args.nod),
            "source_files": {
                str(node): {"path": str(path), "sha256": sha256(path)}
                for node, path in zip(PAPER_NODES, source_paths)
            },
            "gps_files": {
                filename: {
                    "path": str(args.gps_root / filename),
                    "sha256": sha256(args.gps_root / filename),
                }
                for filename in GPS_FILES
            },
        },
        "runtime": {
            "gpu": args.gpu,
            "gpu_name": cp.cuda.runtime.getDeviceProperties(args.gpu)["name"].decode(),
            "seconds": time.perf_counter() - started,
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    output_json = args.output / "node_covariance_information_gate.json"
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "selected_variant": selected_variant,
                "rank_summary": payload["rank_summary"],
                "real_results": real,
                "admission_gate": gate,
                "runtime": payload["runtime"],
                "output": str(output_json),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
