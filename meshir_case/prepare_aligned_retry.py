#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def estimate_s1_delay(rir: np.ndarray, positions: np.ndarray, source: np.ndarray, rate: float) -> tuple[float, float]:
    peak = np.argmax(np.abs(rir[: int(0.25 * rate)]), axis=0) / rate
    distance = np.linalg.norm(positions - source[None, :], axis=1)
    inverse_speed, delay = np.polyfit(distance, peak, 1)
    return float(delay), float(1.0 / inverse_speed)


def estimate_s32_delay(rir: np.ndarray, positions: np.ndarray, sources: np.ndarray, rate: float) -> float:
    residuals = []
    for source_index, source in enumerate(sources):
        values = np.abs(rir[source_index, : int(0.25 * rate)])
        maximum = values.max(axis=0)
        for microphone in range(values.shape[1]):
            hit = np.flatnonzero(values[:, microphone] >= 0.05 * maximum[microphone])
            if len(hit):
                arrival = hit[0] / rate
                distance = np.linalg.norm(positions[microphone] - source)
                residuals.append(arrival - distance / 343.0)
    return float(np.median(residuals))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, default=0.25)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.source_cache / "geometry.npz") as geometry:
        s1_positions, s1_source = geometry["s1_positions"], geometry["s1_source"][0]
        s32_positions, s32_sources = geometry["s32_positions"], geometry["s32_sources"]
    s1_raw = np.load(args.source_cache / "s1_rir_16k.npy", mmap_mode="r")
    s32_raw = np.load(args.source_cache / "s32_rir_16k.npy", mmap_mode="r")
    rate = 16000.0
    s1_delay, s1_speed = estimate_s1_delay(s1_raw, s1_positions, s1_source, rate)
    s32_delay = estimate_s32_delay(s32_raw, s32_positions, s32_sources, rate)
    length = int(round(args.duration_seconds * rate))
    s1_start = int(round(s1_delay * rate))
    s32_start = int(round(s32_delay * rate))
    if s1_start + length > len(s1_raw) or s32_start + length > s32_raw.shape[1]:
        raise ValueError("aligned crop exceeds cached data")
    np.save(args.output / "s1_rir_16k.npy", np.asarray(s1_raw[s1_start:s1_start + length], dtype=np.float32))
    np.save(args.output / "s32_rir_16k.npy", np.asarray(s32_raw[:, s32_start:s32_start + length], dtype=np.float32))
    np.savez_compressed(args.output / "geometry.npz", s1_positions=s1_positions, s1_source=s1_source[None], s32_positions=s32_positions, s32_sources=s32_sources)
    manifest = {
        "case": "meshir_aligned_retry",
        "source_cache": str(args.source_cache),
        "s1_recording_delay_seconds": s1_delay,
        "s1_fitted_speed_m_s": s1_speed,
        "s1_crop_start_sample": s1_start,
        "s32_recording_delay_seconds": s32_delay,
        "s32_crop_start_sample": s32_start,
        "sample_rate_hz": rate,
        "duration_seconds": args.duration_seconds,
        "future_data_used": False,
    }
    (args.output / "alignment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
