#!/usr/bin/env python3
"""Three-peak MUSIC smoke frontend for Baoding multi-source archives.

This is deliberately an inspection-stage frontend.  It reuses the validated
packet decoder and steering geometry from ``shuangyuan_dual_frontend.py`` but
extracts three separated peaks on each 1-D arm.  Azimuth/zenith rank pairing
is recorded as provisional; no target-labelled tracking or performance gate is
claimed by this module.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

import shuangyuan_dual_frontend as base


K = 3
MIN_SEPARATION_DEG = 12.0


def separated_peaks(score: np.ndarray, count: int, low: float, high: float) -> list[int]:
    candidates = []
    for i in range(len(score)):
        left = score[i - 1] if i else -np.inf
        right = score[i + 1] if i + 1 < len(score) else -np.inf
        if score[i] >= left and score[i] >= right:
            candidates.append(i)
    candidates.sort(key=lambda i: float(score[i]), reverse=True)
    selected: list[int] = []
    step = (high - low) / max(len(score) - 1, 1)
    for index in candidates:
        angle = low + index * step
        distances = [abs(angle - (low + j * step)) for j in selected]
        if low == 0.0 and high >= 360.0:
            distances = [min(d, 360.0 - d) for d in distances]
        if all(distance >= MIN_SEPARATION_DEG for distance in distances):
            selected.append(index)
        if len(selected) == count:
            break
    if len(selected) < count:
        for index in np.argsort(score)[::-1]:
            index = int(index)
            if index in selected:
                continue
            angle = low + index * step
            distances = [abs(angle - (low + j * step)) for j in selected]
            if low == 0.0 and high >= 360.0:
                distances = [min(d, 360.0 - d) for d in distances]
            if all(distance >= MIN_SEPARATION_DEG for distance in distances):
                selected.append(index)
            if len(selected) == count:
                break
    if len(selected) < count:
        raise RuntimeError(f"could not find {count} separated peaks")
    return selected


def three_peak_1d(coefficients: np.ndarray, frequencies: np.ndarray, axis: str):
    if axis == "azimuth":
        coarse = np.arange(0.0, 360.0, 5.0)
        low, high = 0.0, 360.0
    else:
        coarse = np.arange(0.0, 90.0 + 1e-9, 5.0)
        low, high = 0.0, 90.0
    score = base.steering_score(coefficients, np.zeros(coefficients.shape[0]), frequencies, coarse, axis)
    indices = separated_peaks(score, K, low, high)
    outputs, strengths = [], []
    for index in indices:
        center = float(coarse[index])
        fine = np.arange(center - 5.0, center + 5.0 + 1e-9, 1.0)
        fine = np.clip(fine, low, high)
        fine_score = base.steering_score(coefficients, np.zeros(coefficients.shape[0]), frequencies, fine, axis)
        winner = int(np.argmax(fine_score))
        outputs.append(float(fine[winner]))
        strengths.append(float(fine_score[winner]))
    order = np.argsort(np.asarray(strengths))[::-1]
    return np.asarray(outputs)[order], np.asarray(strengths)[order]


def music_three_peaks(block: np.ndarray, bins: np.ndarray, frequencies: np.ndarray) -> dict:
    z, x, y = base.channel_vectors(block)
    x_coeff = base.frequency_decompose(x, bins)
    y_coeff = base.frequency_decompose(y, bins)
    z_coeff = base.frequency_decompose(z, bins)
    azimuth, az_strength = three_peak_1d(np.concatenate([x_coeff, y_coeff], axis=0), frequencies, "azimuth")
    zenith, zen_strength = three_peak_1d(z_coeff, frequencies, "elevation")
    row = {}
    for index in range(K):
        row[f"azimuth_{index + 1}_deg"] = float(azimuth[index])
        row[f"zenith_{index + 1}_deg"] = float(zenith[index])
        row[f"azimuth_strength_{index + 1}"] = float(az_strength[index])
        row[f"zenith_strength_{index + 1}"] = float(zen_strength[index])
    return row


def run_node(segment: Path, node: int, start_hhmmss: int, limit_frames: int | None):
    ip = base.NODE_TO_IP[node]
    prefix = segment / f"20171107baoding_{start_hhmmss:06d}_{ip}_19"
    wavfm = prefix.with_suffix(".wavfm")
    gpstime = prefix.with_suffix(".gpstime")
    data, packet_meta = base.decode_wavfm(wavfm)
    timestamps = base.read_gpstime(gpstime)
    bins, frequencies = base.frequency_bins()
    num_frames = int(data.shape[1] // base.FRAME_SAMPLES) - 1
    if limit_frames is not None:
        num_frames = min(num_frames, limit_frames)
    rows = []
    for frame_number in range(num_frames):
        frame_start = frame_number * base.FRAME_SAMPLES
        tic = time.monotonic()
        output = music_three_peaks(data[:19, frame_start : frame_start + base.FRAME_SAMPLES], bins, frequencies)
        frame_time = base.hms_seconds(start_hhmmss) + frame_number * base.FRAME_SAMPLES / base.FS
        rows.append({
            "node_id": node,
            "ip_suffix": ip,
            "time_s": frame_time,
            "time_hhmmss": base.seconds_hhmmss(frame_time),
            "frame_start_sample": frame_start,
            "frontend_runtime_s": time.monotonic() - tic,
            **output,
        })
    manifest = {
        "node_id": node,
        "ip_suffix": ip,
        "wavfm": packet_meta,
        "gpstime": {"path": str(gpstime), "first": timestamps[0], "last": timestamps[-1]},
        "protocol": {
            "sources": K,
            "peak_min_separation_deg": MIN_SEPARATION_DEG,
            "peak_pairing": "azimuth and zenith independently ranked by MUSIC strength; provisional only",
            "sample_rate_hz": base.FS,
            "nfft": base.NFFT,
            "snapshots": base.FSNAP,
            "frame_samples": base.FRAME_SAMPLES,
            "frequency_bins": bins.tolist(),
            "frequencies_hz": frequencies.tolist(),
            "sound_speed_mps": base.C_SOUND,
        },
        "requested_start_hhmmss": start_hhmmss,
        "rows": len(rows),
        "claim_status": "frontend_inspection_only",
    }
    return rows, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-root", type=Path, required=True)
    parser.add_argument("--segment-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--node", type=int, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--limit-frames", type=int)
    args = parser.parse_args()
    segment = args.remote_root / "20171107保定实验/project/20171107baoding" / args.segment_name
    rows, manifest = run_node(segment, args.node, args.start, args.limit_frames)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = f"triple_doa_node_{args.node}_{args.start:06d}"
    with (args.output / f"{stem}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    (args.output / f"{stem}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "node": args.node, "rows": len(rows), "claim_status": manifest["claim_status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
