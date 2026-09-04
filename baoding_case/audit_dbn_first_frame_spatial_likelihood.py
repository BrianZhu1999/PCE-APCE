#!/usr/bin/env python3
"""Audit first-frame real-acoustic spatial likelihood for paper DBN setup."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gpu_wideband_dbn_field as gpu


PAPER_XY = np.asarray(((38614853.4, 4337388.27), (38615012.2, 4336467.20), (38615647.2, 4337215.10)), dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync-root", type=Path, required=True)
    parser.add_argument("--nod", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--spectrum-side", choices=("positive", "negative"), default="positive")
    parser.add_argument("--radius-m", type=float, default=160.0)
    parser.add_argument("--spacing-m", type=float, default=2.0)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--orientation-mode", choices=("none", "nod_flip"), default="none")
    args = parser.parse_args()

    cp.cuda.Device(args.gpu).use()
    nodes = gpu.parse_nod(args.nod)
    center = np.mean(np.asarray([nodes[node][:2] for node in gpu.PAPER_NODES]), axis=0)
    sensors = np.vstack([gpu.sensor_geometry(nodes[node], args.orientation_mode) for node in gpu.PAPER_NODES])
    sensors_xy = cp.asarray(sensors[:, :2] - center[None, :], dtype=cp.float32)
    frequencies = cp.asarray(np.arange(3, 1498, 3), dtype=cp.float32)
    frequency_bins = np.arange(3, 1498, 3, dtype=np.int64)
    streams = [np.load(args.sync_root / f"node{node}_ip{gpu.NODE_TO_IP[node]}_3khz.npy", mmap_mode="r") for node in gpu.PAPER_NODES]
    start = args.frame_index * gpu.FS
    frame = cp.asarray(np.vstack([stream[:, start:start + gpu.FS] for stream in streams]).astype(np.float32, copy=False))
    spectrum = cp.fft.rfft(frame, axis=1)[:, frequency_bins].T.astype(cp.complex64) / gpu.FS
    if args.spectrum_side == "negative":
        spectrum = cp.conj(spectrum)
    states = cp.asarray(PAPER_XY - center[None, :], dtype=cp.float32)
    matrix, sources = gpu.current_sources(cp.hstack((states, cp.zeros((3, 2), dtype=cp.float32))), spectrum, sensors_xy, frequencies)
    reconstructed = cp.einsum("fpm,fm->fp", matrix, sources)
    residual = spectrum - reconstructed
    total_energy = float(cp.asnumpy(cp.mean(cp.abs(spectrum) ** 2)))
    residual_energy = float(cp.asnumpy(cp.mean(cp.abs(residual) ** 2)))
    source_energy = float(cp.asnumpy(cp.mean(cp.abs(reconstructed) ** 2)))

    frequency_weights = 1.0 / cp.mean(cp.abs(residual) ** 2, axis=1).clip(1e-12)
    low, high = cp.percentile(frequency_weights, cp.asarray([5.0, 95.0], dtype=cp.float32))
    frequency_weights = cp.clip(frequency_weights, low, high)
    frequency_weights /= cp.mean(frequency_weights)
    candidate_results = []
    for target in range(3):
        other_indices = [index for index in range(3) if index != target]
        other_matrix = matrix[:, :, other_indices]
        predicted = states[target]
        candidates = gpu.grid(predicted, args.radius_m, args.spacing_m)
        scores = gpu.conditional_candidate_scores(candidates, spectrum, other_matrix, sensors_xy, frequencies, frequency_weights, predicted, cp.eye(2, dtype=cp.float32) * 400.0, 0.0, args.batch, "global")
        best_index = int(cp.argmax(scores))
        scores_cpu = cp.asnumpy(scores)
        true_score = float(cp.asnumpy(scores[gpu.cp.argmin(cp.sum((candidates - predicted[None, :]) ** 2, axis=1))]))
        rank = int(1 + np.sum(scores_cpu > true_score))
        best = cp.asnumpy(candidates[best_index])
        true = cp.asnumpy(predicted)
        candidate_results.append({"target": target + 1, "grid_candidates": int(len(scores_cpu)), "true_position_local_m": true.tolist(), "best_position_local_m": best.tolist(), "best_minus_true_m": (best - true).tolist(), "true_score": true_score, "best_score": float(scores_cpu[best_index]), "true_score_rank_1_is_best": rank, "score_p95_minus_true": float(np.percentile(scores_cpu, 95) - true_score), "score_max_minus_true": float(np.max(scores_cpu) - true_score)})

    payload = {"claim_status": "first_frame_real_acoustic_spatial_likelihood_audit", "gpu": args.gpu, "sync_root": str(args.sync_root), "frame_index": args.frame_index, "spectrum_side": args.spectrum_side, "orientation_mode": args.orientation_mode, "sample_rate_hz": gpu.FS, "frame_samples": gpu.FS, "frequency_grid_hz": "3,6,...,1497", "frequency_count": len(frequencies), "sensor_count": len(sensors), "paper_initial_positions_projected_m": PAPER_XY.tolist(), "node_center_xy": center.tolist(), "total_spectral_energy": total_energy, "reconstructed_source_energy": source_energy, "residual_energy": residual_energy, "explained_energy_fraction": source_energy / max(total_energy, 1e-12), "candidate_results": candidate_results, "gps_runtime_used": False, "interpretation": "A positive audit requires all target true positions to rank near the spatial score maxima and explained energy to be nontrivial; this is an offline initialization audit, not a tracking result."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
