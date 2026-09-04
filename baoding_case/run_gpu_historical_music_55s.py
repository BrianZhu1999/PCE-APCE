#!/usr/bin/env python3
"""GPU historical-MUSIC frontend replay for the Baoding three-target window."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shuangyuan_dual_frontend as base


NODES = (1, 3, 5, 6, 7, 8, 11, 13)
NODE_TO_IP = {1: 47, 3: 40, 5: 54, 6: 43, 7: 49, 8: 61, 11: 5, 13: 46}
FS = 3050
NFFT = 128
FSNAP = 25
FRAME = NFFT * FSNAP
K = 3
C_SOUND = 340.0
SPACING = 0.5


def gpu_peaks(coeff: cp.ndarray, frequencies: cp.ndarray, axis: str) -> tuple[list[float], list[float]]:
    deploy = cp.asarray([-3, -2, -1, 1, 2, 3], dtype=cp.float32) * SPACING if axis == "azimuth" else cp.asarray([-2.13, -1.53, -0.93, 0, 1, 2, 3], dtype=cp.float32) * SPACING
    angles = cp.arange(0, 360 if axis == "azimuth" else 90, 1, dtype=cp.float32)
    total = cp.zeros(len(angles), dtype=cp.float32)
    for fi, frequency in enumerate(frequencies):
        covariance = coeff[:, :, fi] @ coeff[:, :, fi].conj().T / coeff.shape[1]
        _, vectors = cp.linalg.eigh(covariance)
        noise = vectors[:, : max(1, vectors.shape[1] - K)]
        if axis == "azimuth":
            deployment = cp.concatenate((deploy * cp.cos(cp.deg2rad(angles))[:, None], -deploy * cp.sin(cp.deg2rad(angles))[:, None]), axis=1)
        else:
            deployment = deploy[None, :] * cp.cos(cp.deg2rad(angles))[:, None]
        steering = cp.exp(1j * 2 * cp.pi * float(frequency) / C_SOUND * deployment.T)
        total += cp.sum(cp.abs(steering.conj().T @ noise) ** 2, axis=1).real
    score = 1.0 / cp.maximum(total, 1e-12)
    selected = []
    for index in cp.argsort(score)[::-1].get():
        value = float(angles[index])
        distances = [min(abs(value - other), 360 - abs(value - other)) for other in selected] if axis == "azimuth" else [abs(value - other) for other in selected]
        if all(distance >= 12 for distance in distances):
            selected.append(value)
        if len(selected) == K:
            break
    strengths = [float(score[int(cp.argmin(cp.abs(angles - value)))]) for value in selected]
    return selected, strengths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start-hhmmss", type=int, default=132754)
    parser.add_argument("--frames", type=int, default=55)
    parser.add_argument("--gpu", type=int, default=2)
    args = parser.parse_args()
    cp.cuda.Device(args.gpu).use()
    segment = args.remote_root / "20171107保定实验/project/20171107baoding/sanyuan_tongxinyuan_6"
    start_sample = int((base.hms_seconds(args.start_hhmmss) - base.hms_seconds(132614)) * FS)
    bins = np.arange(int(np.ceil(100 * NFFT / FS)), int(np.floor(500 * NFFT / FS)) + 1, dtype=int)
    frequencies = cp.asarray(bins * FS / NFFT, dtype=cp.float32)
    args.output_root.mkdir(parents=True, exist_ok=True)
    for node in NODES:
        data, metadata = base.decode_wavfm(segment / f"20171107baoding_132614_{NODE_TO_IP[node]}_19.wavfm")
        rows = []
        for frame_index in range(args.frames):
            block = data[:, start_sample + frame_index * FRAME:start_sample + (frame_index + 1) * FRAME]
            z, x, y = base.channel_vectors(block)
            def decompose(values):
                array = values.reshape(values.shape[0], FSNAP, NFFT)
                return cp.asarray(np.fft.fft(array, axis=-1)[:, :, bins] / np.sqrt(NFFT), dtype=cp.complex64)
            azimuth, az_strength = gpu_peaks(cp.concatenate((decompose(x), decompose(y)), axis=0), frequencies, "azimuth")
            zenith, zen_strength = gpu_peaks(decompose(z), frequencies, "elevation")
            row = {"frame_index": frame_index, "time_s": base.hms_seconds(args.start_hhmmss) + frame_index * FRAME / FS, "node_id": node, "ip_suffix": NODE_TO_IP[node]}
            for index in range(K):
                row[f"azimuth_{index + 1}_deg"] = azimuth[index]
                row[f"zenith_{index + 1}_deg"] = zenith[index]
                row[f"azimuth_strength_{index + 1}"] = az_strength[index]
                row[f"zenith_strength_{index + 1}"] = zen_strength[index]
            rows.append(row)
        output = args.output_root / f"node{node}" / f"triple_doa_node_{node}_132614.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        (output.parent / "frontend_manifest.json").write_text(json.dumps({"claim_status": "gpu_historical_music_three_peak_frontend", "node": node, "ip_suffix": NODE_TO_IP[node], "source_wavfm": str(segment / f"20171107baoding_132614_{NODE_TO_IP[node]}_19.wavfm"), "source_metadata": metadata, "frames": args.frames, "frame_samples": FRAME, "nfft": NFFT, "snapshots": FSNAP, "frequency_range_hz": [100, 500], "frequency_count": len(bins), "sources": K, "gps_runtime_used": False, "target_labels": "unassigned peaks", "csv": str(output)}, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"claim_status": "gpu_historical_music_55s_frontend", "nodes": list(NODES), "frames": args.frames, "frame_dt_s": FRAME / FS, "nfft": NFFT, "snapshots": FSNAP, "frequency_range_hz": [100, 500], "frequency_count": len(bins), "gpu": args.gpu, "gps_runtime_used": False, "association": "not yet applied; peaks remain rank-labelled", "output_root": str(args.output_root)}
    (args.output_root / "music_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
