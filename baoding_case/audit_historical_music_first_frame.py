#!/usr/bin/env python3
"""GPU audit of the historical Baoding apart_search MUSIC protocol."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shuangyuan_dual_frontend as base


PAPER_XY = np.asarray(((38614853.4, 4337388.27), (38615012.2, 4336467.20), (38615647.2, 4337215.10)), dtype=float)
NODES = (1, 3, 5, 6, 7, 8, 11, 13)
NODE_TO_IP = {1: 47, 3: 40, 5: 54, 6: 43, 7: 49, 8: 61, 11: 5, 13: 46}
NFFT = 128
FSNAP = 25
FS = 3050
SOUND_SPEED = 340.0
SPACING = 0.5
K = 3


def peaks(coeff: cp.ndarray, freqs: cp.ndarray, axis: str, device: int) -> tuple[list[float], list[float]]:
    deploy = cp.asarray([-3, -2, -1, 1, 2, 3], dtype=cp.float32) * SPACING if axis == "azimuth" else cp.asarray([-2.13, -1.53, -0.93, 0, 1, 2, 3], dtype=cp.float32) * SPACING
    angles = cp.arange(0, 360 if axis == "azimuth" else 90, 1, dtype=cp.float32)
    total = cp.zeros(len(angles), dtype=cp.float32)
    for fi, frequency in enumerate(freqs):
        xyz = coeff[:, :, fi]
        covariance = xyz @ xyz.conj().T / coeff.shape[1]
        _, vectors = cp.linalg.eigh(covariance)
        noise = vectors[:, : max(1, vectors.shape[1] - K)]
        if axis == "azimuth":
            deployment = cp.concatenate((deploy * cp.cos(cp.deg2rad(angles))[:, None], -deploy * cp.sin(cp.deg2rad(angles))[:, None]), axis=1)
        else:
            deployment = deploy[None, :] * cp.cos(cp.deg2rad(angles))[:, None]
        steering = cp.exp(1j * 2 * cp.pi * float(frequency) / SOUND_SPEED * deployment.T)
        projection = steering.conj().T @ noise
        total += cp.sum(cp.abs(projection) ** 2, axis=1).real
    score = 1.0 / cp.maximum(total, 1e-12)
    selected = []
    for index in cp.argsort(score)[::-1].get():
        value = float(angles[index])
        if all(min(abs(value - previous), 360 - abs(value - previous)) >= 12 for previous in selected) if axis == "azimuth" else all(abs(value - previous) >= 12 for previous in selected):
            selected.append(value)
        if len(selected) == K:
            break
    return selected, [float(score[int(cp.argmin(cp.abs(angles - value)))]) for value in selected]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--start-hhmmss", type=int, default=132754)
    args = parser.parse_args()
    cp.cuda.Device(args.gpu).use()
    segment = args.remote_root / "20171107保定实验/project/20171107baoding/sanyuan_tongxinyuan_6"
    nod = {}
    for line in (args.remote_root / "20171107保定实验/GPS_data/20171107baoding.nod").read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) >= 6:
            nod[int(fields[2])] = (float(fields[3]), float(fields[4]), float(fields[5]))
    center = np.mean(np.asarray([nod[node][:2] for node in NODES]), axis=0)
    start_sample = int((base.hms_seconds(args.start_hhmmss) - base.hms_seconds(132614)) * FS)
    observations = {}
    for node in NODES:
        path = segment / f"20171107baoding_132614_{NODE_TO_IP[node]}_19.wavfm"
        data, _ = base.decode_wavfm(path)
        block = data[:, start_sample:start_sample + NFFT * FSNAP]
        z, x, y = base.channel_vectors(block)
        bins = np.arange(int(np.ceil(100 * NFFT / FS)), int(np.floor(500 * NFFT / FS)) + 1, dtype=int)
        freqs = bins * FS / NFFT
        def decompose(values):
            return cp.asarray(np.fft.fft(values.reshape(values.shape[0], FSNAP, NFFT), axis=-1)[:, :, bins] / np.sqrt(NFFT), dtype=cp.complex64)
        x_coeff, y_coeff, z_coeff = decompose(x), decompose(y), decompose(z)
        az, az_strength = peaks(cp.concatenate((x_coeff, y_coeff), axis=0), cp.asarray(freqs, dtype=cp.float32), "azimuth", args.gpu)
        el, el_strength = peaks(z_coeff, cp.asarray(freqs, dtype=cp.float32), "elevation", args.gpu)
        observations[str(node)] = {"azimuth_deg": az, "zenith_deg": el, "azimuth_strength": az_strength, "zenith_strength": el_strength}
    payload = {"claim_status": "historical_matlab_music_first_frame_audit", "start_hhmmss": args.start_hhmmss, "gpu": args.gpu, "sample_rate_hz": FS, "nfft": NFFT, "snapshots": FSNAP, "frame_samples": NFFT * FSNAP, "frequency_range_hz": [100, 500], "frequency_bin_count": 16, "nodes": observations, "gps_runtime_used": False, "warning": "K=3 extension of the historical K=2 MUSIC frontend; no target association or DBN update is inferred."}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
