#!/usr/bin/env python3
"""GPU-batched paper-inspired wideband DBN tracker for the Baoding field case.

This is an engineering reconstruction, not the authors' private implementation.
It keeps the paper's near-field wideband observation and motion-prior logic,
while using damped coordinate updates and PSD covariance estimates for a
reproducible 55-second three-target benchmark.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path

import cupy as cp
import numpy as np


PAPER_NODES = (1, 3, 5, 6, 7, 8, 11, 13)
NODE_TO_IP = {1: 47, 3: 40, 5: 54, 6: 43, 7: 49, 8: 61, 11: 5, 13: 46}
CHANNEL_GROUPS = {"z": [19, 18, 17, 13, 14, 15, 16], "x": [9, 8, 7, 1, 2, 3], "y": [12, 11, 10, 4, 5, 6]}
PAPER_XYV = np.asarray(((38614853.4, 4337388.27, 25.85033, 23.00194), (38615012.2, 4336467.20, -41.09208, 6.67753), (38615647.2, 4337215.10, 3.20862, -41.49795)), dtype=np.float64)
GPS_FILES = {1: "GPS1_plane1.gps", 2: "GPS3_plane2.gps", 3: "GPS4_plane2to3.gps"}
FS = 3000
SOUND_SPEED = 340.0
PROCESSING_HEIGHT_M = 230.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hms_seconds(value: str | int | float) -> float:
    text = str(int(float(value))).zfill(6)
    return int(text[:2]) * 3600 + int(text[2:4]) * 60 + int(text[4:])


def parse_nod(path: Path) -> dict[int, tuple[float, float, float, int, int]]:
    nodes = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) >= 6:
            nodes[int(fields[2])] = (float(fields[3]), float(fields[4]), float(fields[5]), int(float(fields[9])) if len(fields) > 9 else 0, int(float(fields[10])) if len(fields) > 10 else 0)
    return nodes


def sensor_geometry(node_xyz: tuple, orientation_mode: str = "none") -> np.ndarray:
    x0, y0, z0 = node_xyz[:3]
    horizontal_sign = -1.0 if orientation_mode == "nod_flip" and len(node_xyz) > 3 and node_xyz[3] else 1.0
    vertical_sign = -1.0 if orientation_mode == "nod_flip" and len(node_xyz) > 4 and node_xyz[4] else 1.0
    positions = np.zeros((19, 3), dtype=np.float64)
    offsets = (-3.0, -2.0, -1.0, 1.0, 2.0, 3.0)
    for channel, offset in zip(CHANNEL_GROUPS["x"], offsets):
        positions[channel - 1] = (x0 + horizontal_sign * 0.5 * offset, y0, z0)
    for channel, offset in zip(CHANNEL_GROUPS["y"], offsets):
        positions[channel - 1] = (x0, y0 + horizontal_sign * 0.5 * offset, z0)
    for channel, offset in zip(CHANNEL_GROUPS["z"], (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)):
        positions[channel - 1] = (x0, y0, z0 + vertical_sign * 0.5 * offset)
    return positions


def read_gps(path: Path) -> list[tuple[float, float, float]]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) >= 8:
            try:
                rows.append((hms_seconds(fields[7]), float(fields[4]), float(fields[5])))
            except ValueError:
                pass
    return rows


def nearest_gps(rows: list[tuple[float, float, float]], time_s: float) -> tuple[float, float]:
    _, x, y = min(rows, key=lambda row: abs(row[0] - time_s))
    return x, y


def steering(candidates: cp.ndarray, sensors_xy: cp.ndarray, frequencies: cp.ndarray) -> cp.ndarray:
    delta = candidates[:, None, :] - sensors_xy[None, :, :]
    distance = cp.sqrt(cp.sum(delta * delta, axis=-1) + PROCESSING_HEIGHT_M**2)
    gain = 1.0 / cp.maximum(distance, 1e-6)
    phase = cp.exp(-1j * 2.0 * cp.pi * frequencies[:, None, None] * distance[None, :, :] / SOUND_SPEED)
    return phase * gain[None, :, :]


def current_sources(states: cp.ndarray, spectrum: cp.ndarray, sensors_xy: cp.ndarray, frequencies: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    matrices = []
    for target in range(3):
        matrices.append(steering(states[target:target + 1, :2], sensors_xy, frequencies)[:, 0, :])
    matrix = cp.stack(matrices, axis=-1)  # frequency x sensor x target
    gram = cp.einsum("fpm,fpn->fmn", cp.conj(matrix), matrix)
    gram += cp.eye(3, dtype=cp.complex64)[None, :, :] * cp.asarray(1e-7, dtype=cp.float32)
    rhs = cp.einsum("fpm,fp->fm", cp.conj(matrix), spectrum)
    sources = cp.linalg.solve(gram, rhs)
    return matrix, sources


def candidate_scores(candidates: cp.ndarray, residual: cp.ndarray, sensors_xy: cp.ndarray, frequencies: cp.ndarray, predicted: cp.ndarray, covariance_xy: cp.ndarray, prior_weight: float, batch: int) -> cp.ndarray:
    outputs = []
    inv_cov = cp.linalg.inv(covariance_xy + cp.eye(2, dtype=cp.float32) * 1e-4)
    for start in range(0, len(candidates), batch):
        subset = candidates[start:start + batch]
        a = steering(subset, sensors_xy, frequencies)
        matched = cp.einsum("fcp,fp->fc", cp.conj(a), residual)
        norm = cp.sum(cp.abs(a) ** 2, axis=-1).clip(1e-12)
        score = cp.mean(cp.abs(matched) ** 2 / norm, axis=0)
        delta = subset - predicted[None, :]
        penalty = cp.einsum("ci,ij,cj->c", delta, inv_cov, delta)
        outputs.append(cp.log(score.clip(1e-20)) - prior_weight * penalty)
    return cp.concatenate(outputs)


def conditional_candidate_scores(candidates: cp.ndarray, spectrum: cp.ndarray, other_matrix: cp.ndarray, sensors_xy: cp.ndarray, frequencies: cp.ndarray, frequency_weights: cp.ndarray, predicted: cp.ndarray, covariance_xy: cp.ndarray, prior_weight: float, batch: int, coherence_mode: str) -> cp.ndarray:
    """Score one target after projecting out the other target subspace."""
    if other_matrix.shape[-1] == 0:
        return candidate_scores(candidates, spectrum, sensors_xy, frequencies, predicted, covariance_xy, prior_weight, batch)
    if coherence_mode == "node":
        return node_conditional_candidate_scores(candidates, spectrum, other_matrix, sensors_xy, frequencies, frequency_weights, predicted, covariance_xy, prior_weight, batch)
    gram = cp.einsum("fpi,fpj->fij", cp.conj(other_matrix), other_matrix)
    gram += cp.eye(other_matrix.shape[-1], dtype=cp.complex64)[None] * 1e-7
    inv_gram = cp.linalg.inv(gram)
    rhs_z = cp.einsum("fpi,fp->fi", cp.conj(other_matrix), spectrum)
    coef_z = cp.einsum("fij,fj->fi", inv_gram, rhs_z)
    z_residual = spectrum - cp.einsum("fpi,fi->fp", other_matrix, coef_z)
    outputs = []
    inv_cov = cp.linalg.inv(covariance_xy + cp.eye(2, dtype=cp.float32) * 1e-4)
    for start in range(0, len(candidates), batch):
        subset = candidates[start:start + batch]
        a = steering(subset, sensors_xy, frequencies)
        rhs_a = cp.einsum("fpi,fcp->fci", cp.conj(other_matrix), a)
        coef_a = cp.einsum("fij,fcj->fci", inv_gram, rhs_a)
        a_residual = a - cp.einsum("fpi,fci->fcp", other_matrix, coef_a)
        matched = cp.einsum("fcp,fp->fc", cp.conj(a_residual), z_residual)
        norm = cp.sum(cp.abs(a_residual) ** 2, axis=-1).clip(1e-12)
        score = cp.sum(frequency_weights[:, None] * (cp.abs(matched) ** 2 / norm), axis=0) / cp.sum(frequency_weights)
        delta = subset - predicted[None, :]
        penalty = cp.einsum("ci,ij,cj->c", delta, inv_cov, delta)
        outputs.append(cp.log(score.clip(1e-20)) - prior_weight * penalty)
    return cp.concatenate(outputs)


def node_conditional_candidate_scores(candidates: cp.ndarray, spectrum: cp.ndarray, other_matrix: cp.ndarray, sensors_xy: cp.ndarray, frequencies: cp.ndarray, frequency_weights: cp.ndarray, predicted: cp.ndarray, covariance_xy: cp.ndarray, prior_weight: float, batch: int) -> cp.ndarray:
    """Conditional likelihood with one nuisance source amplitude per node."""
    node_count, channels = 8, 19
    z = spectrum.reshape(len(frequencies), node_count, channels)
    other = other_matrix.reshape(len(frequencies), node_count, channels, other_matrix.shape[-1])
    gram = cp.einsum("fnpi,fnpj->fnij", cp.conj(other), other)
    gram += cp.eye(other.shape[-1], dtype=cp.complex64)[None, None] * 1e-7
    inv_gram = cp.linalg.inv(gram)
    rhs_z = cp.einsum("fnpi,fnp->fni", cp.conj(other), z)
    coef_z = cp.einsum("fnij,fnj->fni", inv_gram, rhs_z)
    z_residual = z - cp.einsum("fnpi,fni->fnp", other, coef_z)
    outputs = []
    inv_cov = cp.linalg.inv(covariance_xy + cp.eye(2, dtype=cp.float32) * 1e-4)
    for start in range(0, len(candidates), batch):
        subset = candidates[start:start + batch]
        a = steering(subset, sensors_xy, frequencies).reshape(len(frequencies), len(subset), node_count, channels).transpose(0, 2, 1, 3)
        rhs_a = cp.einsum("fnpi,fncp->fnci", cp.conj(other), a)
        coef_a = cp.einsum("fnij,fncj->fnci", inv_gram, rhs_a)
        a_residual = a - cp.einsum("fnpi,fnci->fncp", other, coef_a)
        matched = cp.einsum("fncp,fnp->fnc", cp.conj(a_residual), z_residual)
        norm = cp.sum(cp.abs(a_residual) ** 2, axis=-1).clip(1e-12)
        per_node = cp.abs(matched) ** 2 / norm
        score = cp.sum(frequency_weights[:, None, None] * per_node, axis=(0, 1)) / (cp.sum(frequency_weights) * node_count)
        delta = subset - predicted[None, :]
        penalty = cp.einsum("ci,ij,cj->c", delta, inv_cov, delta)
        outputs.append(cp.log(score.clip(1e-20)) - prior_weight * penalty)
    return cp.concatenate(outputs)


def grid(center: cp.ndarray, radius: float, spacing: float) -> cp.ndarray:
    axis = cp.arange(-radius, radius + spacing * 0.5, spacing, dtype=cp.float32)
    xx, yy = cp.meshgrid(axis, axis, indexing="xy")
    return cp.stack((xx.ravel() + center[0], yy.ravel() + center[1]), axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync-root", type=Path, required=True)
    parser.add_argument("--nod", type=Path, required=True)
    parser.add_argument("--gps-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=55)
    parser.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    parser.add_argument("--damping", type=float, default=0.45)
    parser.add_argument("--prior-weight", type=float, default=0.35)
    parser.add_argument("--coarse-radius", type=float, default=100.0)
    parser.add_argument("--coarse-spacing", type=float, default=5.0)
    parser.add_argument("--refine-radius", type=float, default=8.0)
    parser.add_argument("--refine-spacing", type=float, default=1.0)
    parser.add_argument("--velocity-damping", type=float, default=0.2)
    parser.add_argument("--batch", type=int, default=768)
    parser.add_argument("--global-radius", type=float, default=0.0)
    parser.add_argument("--global-spacing", type=float, default=20.0)
    parser.add_argument("--coherence-mode", choices=("global", "node"), default="global")
    parser.add_argument("--spectrum-side", choices=("negative", "positive"), default="negative")
    parser.add_argument("--orientation-mode", choices=("none", "nod_flip"), default="none")
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()

    cp.cuda.Device(args.gpu).use()
    cp.random.seed(args.seed)
    nodes = parse_nod(args.nod)
    sensors = np.vstack([sensor_geometry(nodes[node], args.orientation_mode) for node in PAPER_NODES])
    center = np.mean(np.asarray([nodes[node][:2] for node in PAPER_NODES]), axis=0)
    sensors_xy = cp.asarray(sensors[:, :2] - center[None, :], dtype=cp.float32)
    frequencies = cp.asarray(np.arange(3, 1498, 3), dtype=cp.float32)
    frequency_bins = np.arange(3, 1498, 3, dtype=np.int64)

    streams = []
    source_hashes = {}
    for node in PAPER_NODES:
        path = args.sync_root / f"node{node}_ip{NODE_TO_IP[node]}_3khz.npy"
        streams.append(np.load(path, mmap_mode="r"))
        source_hashes[str(node)] = sha256(path)

    initial_state = PAPER_XYV.copy()
    initial_state[:, :2] -= center[None, :]
    states = cp.asarray(initial_state, dtype=cp.float32)
    covariance = cp.tile(cp.diag(cp.asarray([100.0, 100.0, 64.0, 64.0], dtype=cp.float32))[None, :, :], (3, 1, 1))
    gps = {target: read_gps(args.gps_root / filename) for target, filename in GPS_FILES.items()}
    field_candidates = None
    if args.global_radius > 0:
        field_candidates = grid(cp.asarray([0.0, 0.0], dtype=cp.float32), args.global_radius, args.global_spacing)
    records = []
    start_wall = time.perf_counter()
    gpu_name = cp.cuda.runtime.getDeviceProperties(args.gpu)["name"].decode()

    for second in range(args.seconds):
        frame_cpu = np.vstack([stream[:, second * FS:(second + 1) * FS] for stream in streams]).astype(np.float32, copy=False)
        frame = cp.asarray(frame_cpu)
        spectrum = cp.fft.rfft(frame, axis=1)[:, frequency_bins].T.astype(cp.complex64) / FS
        # The upstream implementation reads DFT bins N-round(f*N/fs), i.e.
        # the negative-frequency coefficients. For real input this is the
        # conjugate of rfft(f); preserve that convention by default.
        if args.spectrum_side == "negative":
            spectrum = cp.conj(spectrum)
        predicted = states.copy()
        if second > 0:
            predicted[:, :2] = states[:, :2] + states[:, 2:]  # one-second state transition
        q = cp.diag(cp.asarray([25.0, 25.0, 16.0, 16.0], dtype=cp.float32))
        covariance = covariance + q[None, :, :]

        matrix, sources = current_sources(predicted, spectrum, sensors_xy, frequencies)
        reconstructed = cp.einsum("fpm,fm->fp", matrix, sources)
        full_residual = spectrum - reconstructed
        noise_power = cp.mean(cp.abs(full_residual) ** 2, axis=1).clip(1e-12)
        precision = 1.0 / noise_power
        low, high = cp.percentile(precision, cp.asarray([5.0, 95.0], dtype=cp.float32))
        frequency_weights = cp.clip(precision, low, high)
        frequency_weights /= cp.mean(frequency_weights)
        frame_costs = []
        posterior_covariances = []
        for target in range(3):
            # Coordinate-ascent detail from the DBN: after each target update,
            # rebuild the other-target steering subspace and source estimates.
            current_matrix, current_sources_amp = current_sources(states, spectrum, sensors_xy, frequencies)
            other_indices = [index for index in range(3) if index != target]
            other_matrix = current_matrix[:, :, other_indices]
            coarse = field_candidates if field_candidates is not None else grid(predicted[target, :2], args.coarse_radius, args.coarse_spacing)
            coarse_score = conditional_candidate_scores(coarse, spectrum, other_matrix, sensors_xy, frequencies, frequency_weights, predicted[target, :2], covariance[target, :2, :2], args.prior_weight, args.batch, args.coherence_mode)
            coarse_best = coarse[int(cp.argmax(coarse_score))]
            refined = grid(coarse_best, args.refine_radius, args.refine_spacing)
            refined_score = conditional_candidate_scores(refined, spectrum, other_matrix, sensors_xy, frequencies, frequency_weights, predicted[target, :2], covariance[target, :2, :2], args.prior_weight, args.batch, args.coherence_mode)
            best_index = int(cp.argmax(refined_score))
            best = refined[best_index]
            new_position = predicted[target, :2] + args.damping * (best - predicted[target, :2])
            measured_velocity = new_position - states[target, :2]
            new_velocity = (1.0 - args.velocity_damping) * predicted[target, 2:] + args.velocity_damping * measured_velocity

            weights = cp.exp((refined_score - cp.max(refined_score)).clip(-40.0, 0.0))
            weights /= cp.sum(weights)
            mean = cp.sum(refined * weights[:, None], axis=0)
            anomaly = refined - mean[None, :]
            cov_xy = cp.einsum("c,ci,cj->ij", weights, anomaly, anomaly)
            eigval, eigvec = cp.linalg.eigh((cov_xy + cov_xy.T) * 0.5)
            cov_xy = eigvec @ cp.diag(cp.maximum(eigval, 4.0)) @ eigvec.T
            posterior = cp.zeros((4, 4), dtype=cp.float32)
            posterior[:2, :2] = cov_xy
            posterior[2:, 2:] = cp.eye(2, dtype=cp.float32) * cp.maximum(cp.trace(cov_xy) * 0.25, 4.0)
            states[target, :2] = new_position
            states[target, 2:] = new_velocity
            covariance[target] = posterior
            frame_costs.append(float(cp.asnumpy(refined_score[best_index])))
            posterior_covariances.append(cp.asnumpy(posterior).tolist())

        time_s = hms_seconds(132754) + second
        state_cpu = cp.asnumpy(states)
        for target in range(1, 4):
            truth_x, truth_y = nearest_gps(gps[target], time_s)
            estimated_x = float(state_cpu[target - 1, 0] + center[0])
            estimated_y = float(state_cpu[target - 1, 1] + center[1])
            records.append({"frame_index": second, "time_s": time_s, "target": target, "estimated_x": estimated_x, "estimated_y": estimated_y, "estimated_vx": float(state_cpu[target - 1, 2]), "estimated_vy": float(state_cpu[target - 1, 3]), "truth_x": truth_x, "truth_y": truth_y, "position_error_m": math.hypot(estimated_x - truth_x, estimated_y - truth_y), "covariance": posterior_covariances[target - 1], "candidate_log_score": frame_costs[target - 1]})
        cp.cuda.get_current_stream().synchronize()

    runtime = time.perf_counter() - start_wall
    errors = {target: [float(row["position_error_m"]) for row in records if row["target"] == target] for target in range(1, 4)}
    ospa = []
    for frame_index in range(args.seconds):
        values = [float(row["position_error_m"]) for row in records if row["frame_index"] == frame_index]
        ospa.append(math.sqrt(sum(value * value for value in values) / len(values)))
    stable = [all(errors[target][index] <= 100.0 for target in range(1, 4)) for index in range(args.seconds)]
    first_unstable = next((index for index, value in enumerate(stable) if not value), None)
    payload = {"claim_status": "paper_inspired_gpu_wideband_dbn_55s", "gpu": args.gpu, "gpu_name": gpu_name, "runtime_s": runtime, "seed": args.seed, "config": {key: value for key, value in vars(args).items() if key not in ("output", "sync_root", "nod", "gps_root")}, "source_sync_root": str(args.sync_root), "source_sha256": source_hashes, "frequency_grid_hz": "3,6,...,1497", "frequency_count": 499, "sensor_count": 152, "frame_count": args.seconds, "frame_dt_s": 1.0, "duration_s": float(args.seconds), "target_metrics": {str(target): {"rmse_m": math.sqrt(sum(value * value for value in errors[target]) / len(errors[target])), "mean_error_m": sum(errors[target]) / len(errors[target]), "max_error_m": max(errors[target]), "final_error_m": errors[target][-1]} for target in range(1, 4)}, "mean_ospa_order2_m": sum(ospa) / len(ospa), "max_ospa_order2_m": max(ospa), "first_common_over_100m_frame": first_unstable, "stable_prefix_seconds": first_unstable if first_unstable is not None else args.seconds, "gps_runtime_correction": False, "covariance": "candidate-posterior PSD covariance with 4 m^2 eigenvalue floor", "records": records, "warning": "GPU paper-inspired engineering reconstruction; not the authors private DBN-LA-NM field implementation."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("claim_status", "gpu", "runtime_s", "frame_count", "target_metrics", "mean_ospa_order2_m", "max_ospa_order2_m", "stable_prefix_seconds")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
