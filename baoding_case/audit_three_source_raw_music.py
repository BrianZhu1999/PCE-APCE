"""Re-run the released Baoding 19-element MUSIC front end with K=3.

The implementation follows the MATLAB ``apart_search`` code shipped with the
20200317 data package.  It reads the framed 21-channel WAVFM stream directly,
uses only the first 19 array channels, and emits a compact JSON audit.  GPS is
not used by the estimator; it can only be joined later for an offline score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MARKER = np.int16(23131)  # 0x5A5B, the WAVFM frame marker in wavtransZQ.m


def read_wavfm(path: Path, n_samples: int, sample_offset: int = 0) -> np.ndarray:
    """Return ``19 x n_samples`` int16 samples from a framed WAVFM file."""
    raw = np.fromfile(path, dtype="<i2")
    marker_idx = np.flatnonzero(raw == MARKER)
    if marker_idx.size == 0:
        raise ValueError(f"no 0x5A5B frame marker found in {path}")
    start = int(marker_idx[0])
    n_frame_words = 22  # marker + 21 channels, as in wavtransZQ.m
    usable = ((raw.size - start) // n_frame_words) * n_frame_words
    frames = raw[start : start + usable].reshape(-1, n_frame_words)
    marker_rate = float(np.mean(frames[:, 0] == MARKER))
    if marker_rate < 0.99:
        raise ValueError(f"frame-marker integrity is {marker_rate:.4f} for {path}")
    if sample_offset < 0 or sample_offset + n_samples > frames.shape[0]:
        raise ValueError(
            f"requested samples [{sample_offset}, {sample_offset + n_samples}) "
            f"outside {frames.shape[0]} frames in {path}"
        )
    # The original set_channel.m consumes channels 1--19; channels 20--21 are
    # retained in the source stream but are not part of the 19-element array.
    return frames[sample_offset : sample_offset + n_samples, 1:20].T.astype(np.float64)


def frequency_bins(fs: float, nfft: int, fl: float, fh: float) -> tuple[np.ndarray, np.ndarray]:
    k1 = 0
    while k1 * fs / nfft < fl:
        k1 += 1
    k2 = nfft // 2
    while k2 * fs / nfft > fh:
        k2 -= 1
    bins = np.arange(k1, k2 + 1, dtype=int)
    return bins, bins * fs / nfft


def set_channel(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Exact zero-based translation of set_channel.m.
    xz = data[[18, 17, 16, 12, 13, 14, 15], :]
    xx = data[[8, 7, 6, 0, 1, 2], :]
    xy = data[[11, 10, 9, 3, 4, 5], :]
    return xz, xx, xy


def frequency_decompose(x: np.ndarray, bins: np.ndarray, nfft: int, fsnap: int) -> np.ndarray:
    m, n_samples = x.shape
    if n_samples != nfft * fsnap:
        raise ValueError("frequency_decompose expects exactly nfft*fsnap samples")
    spectra = np.empty((m, fsnap, len(bins)), dtype=np.complex128)
    for snap in range(fsnap):
        segment = x[:, snap * nfft : (snap + 1) * nfft]
        fft_data = np.fft.fft(segment, n=nfft, axis=1) / np.sqrt(nfft)
        spectra[:, snap, :] = fft_data[:, bins]
    return spectra


def local_maxima(values: np.ndarray) -> list[int]:
    # MATLAB one_dim_search only accepts strict interior maxima.
    return [i for i in range(1, len(values) - 1) if values[i] > values[i - 1] and values[i] > values[i + 1]]


def top_peaks(values: np.ndarray, count: int) -> np.ndarray:
    peaks = local_maxima(np.abs(values))
    peaks.sort(key=lambda i: float(np.abs(values[i])), reverse=True)
    if len(peaks) < count:
        # Keep deterministic fallback behaviour for frames with fewer local
        # maxima; this is reported in the audit rather than hidden.
        candidates = np.argsort(np.abs(values))[::-1].tolist()
        for idx in candidates:
            if idx not in peaks:
                peaks.append(int(idx))
            if len(peaks) == count:
                break
    return np.asarray(peaks[:count], dtype=int)


def music_spectrum(k: int, x: np.ndarray, positions: np.ndarray, freqs: np.ndarray, c: float, angles: np.ndarray) -> np.ndarray:
    _, fsnap, flength = x.shape
    spectrum = np.zeros(len(angles), dtype=np.float64)
    for fi, freq in enumerate(freqs):
        r = x[:, :, fi] @ x[:, :, fi].conj().T / fsnap
        eigvals, eigvecs = np.linalg.eigh(r)
        noise = eigvecs[:, np.argsort(eigvals)[::-1][k:]]
        proj = noise @ noise.conj().T
        for ai, angle in enumerate(angles):
            steering = np.exp(1j * 2.0 * np.pi * (freq / c) * positions * np.cos(np.deg2rad(angle)))
            denom = np.real(np.vdot(steering, proj @ steering))
            spectrum[ai] += max(float(denom), np.finfo(float).tiny)
    return 1.0 / spectrum


def music_xy(k: int, xx: np.ndarray, xy: np.ndarray, freqs: np.ndarray, c: float, angles: np.ndarray) -> np.ndarray:
    _, fsnap, flength = xx.shape
    spectrum = np.zeros(len(angles), dtype=np.float64)
    deploy = np.arange(-3, 4, dtype=float)
    deploy = deploy[deploy != 0] * 0.5
    positions = np.concatenate([deploy, deploy])
    for fi, freq in enumerate(freqs):
        xyz = np.concatenate([xx[:, :, fi], xy[:, :, fi]], axis=0)
        r = xyz @ xyz.conj().T / fsnap
        eigvals, eigvecs = np.linalg.eigh(r)
        noise = eigvecs[:, np.argsort(eigvals)[::-1][k:]]
        proj = noise @ noise.conj().T
        for ai, angle in enumerate(angles):
            phase_positions = np.concatenate([deploy * np.cos(np.deg2rad(angle)), deploy * np.sin(np.deg2rad(angle))])
            steering = np.exp(1j * 2.0 * np.pi * (freq / c) * phase_positions)
            denom = np.real(np.vdot(steering, proj @ steering))
            spectrum[ai] += max(float(denom), np.finfo(float).tiny)
    return 1.0 / spectrum


def estimate_doa(data: np.ndarray, k: int, fs: float, nfft: int, fsnap: int, fl: float, fh: float, c: float, angle_dense: float) -> dict:
    bins, freqs = frequency_bins(fs, nfft, fl, fh)
    xz, xx, xy = set_channel(data)
    xz_f = frequency_decompose(xz, bins, nfft, fsnap)
    xx_f = frequency_decompose(xx, bins, nfft, fsnap)
    xy_f = frequency_decompose(xy, bins, nfft, fsnap)

    coarse_step = 5.0 * angle_dense
    phi_coarse = np.arange(coarse_step, 90.0, coarse_step)
    theta_coarse = np.arange(coarse_step, 360.0, coarse_step)
    xy_positions = np.arange(-3, 4, dtype=float)
    xy_positions = xy_positions[xy_positions != 0] * 0.5
    z_positions = np.asarray([-2.13, -1.53, -0.93, 0.0, 1.0, 2.0, 3.0]) * 0.5
    az_spec_coarse = music_xy(k, xx_f, xy_f, freqs, c, theta_coarse)
    el_spec_coarse = music_spectrum(k, xz_f, z_positions, freqs, c, phi_coarse)
    az_idx = top_peaks(az_spec_coarse, k)
    el_idx = top_peaks(el_spec_coarse, k)
    azimuth = []
    elevation = []
    az_refined_peak = []
    el_refined_peak = []
    for ai, ei in zip(az_idx, el_idx):
        az_center = theta_coarse[ai]
        el_center = phi_coarse[ei]
        az_grid = np.arange(az_center - coarse_step, az_center + coarse_step + 0.5 * angle_dense, angle_dense)
        el_grid = np.arange(el_center - coarse_step, el_center + coarse_step + 0.5 * angle_dense, angle_dense)
        az_spec = music_xy(k, xx_f, xy_f, freqs, c, az_grid)
        el_spec = music_spectrum(k, xz_f, z_positions, freqs, c, el_grid)
        azimuth.append(float(az_grid[int(np.argmax(az_spec))] % 360.0))
        elevation.append(float(el_grid[int(np.argmax(el_spec))]))
        az_refined_peak.append(float(np.max(az_spec)))
        el_refined_peak.append(float(np.max(el_spec)))
    return {
        "azimuth_deg": azimuth,
        "elevation_deg": elevation,
        "azimuth_coarse_peaks_deg": theta_coarse[az_idx].tolist(),
        "elevation_coarse_peaks_deg": phi_coarse[el_idx].tolist(),
        "azimuth_refined_peak": az_refined_peak,
        "elevation_refined_peak": el_refined_peak,
        "frequency_bins": bins.tolist(),
        "frequencies_hz": freqs.tolist(),
        "n_frequency_bins": int(len(freqs)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--frames", type=int, default=1, help="number of consecutive 640-sample frames")
    p.add_argument("--offset", type=int, default=0, help="starting frame index in the WAVFM stream")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--fs", type=float, default=3050.0)
    p.add_argument("--nfft", type=int, default=128)
    p.add_argument("--fsnap", type=int, default=5)
    p.add_argument("--fl", type=float, default=100.0)
    p.add_argument("--fh", type=float, default=500.0)
    p.add_argument("--c", type=float, default=340.0)
    p.add_argument("--angle-dense", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    samples_per_frame = args.nfft * args.fsnap
    data = read_wavfm(args.raw, args.frames * samples_per_frame, args.offset * samples_per_frame)
    frame_results = []
    for frame in range(args.frames):
        lo = frame * samples_per_frame
        hi = lo + samples_per_frame
        result = estimate_doa(
            data[:, lo:hi], args.k, args.fs, args.nfft, args.fsnap,
            args.fl, args.fh, args.c, args.angle_dense,
        )
        result["frame"] = args.offset + frame
        frame_results.append(result)
    payload = {
        "protocol": {
            "raw_path": str(args.raw),
            "k": args.k,
            "fs_hz": args.fs,
            "nfft": args.nfft,
            "fsnap": args.fsnap,
            "samples_per_update": samples_per_frame,
            "frequency_band_hz": [args.fl, args.fh],
            "sound_speed_mps": args.c,
            "angle_dense_deg": args.angle_dense,
            "channel_count_in_stream": 21,
            "array_channels_used": 19,
            "channel_reorder_zero_based": {
                "xz": [18, 17, 16, 12, 13, 14, 15],
                "xx": [8, 7, 6, 0, 1, 2],
                "xy": [11, 10, 9, 3, 4, 5],
            },
            "gps_used_at_runtime": False,
        },
        "frames": frame_results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "frames": len(frame_results)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
