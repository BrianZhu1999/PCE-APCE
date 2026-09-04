#!/usr/bin/env python3
"""Two-source MUSIC frontend for the 2017 Baoding ``shuangyuan_4`` archive.

The historical MATLAB ``apart_search`` implementation is reproduced here
without touching the Figure 2/3 runners.  ``.wavfm`` packets are decoded as
``[0x5a5b marker, 20 int16 samples, sample_index]`` at 3050 Hz.  The first
19 sample streams are mapped with the archived Ch1--Ch19 protocol.  The
archived MATLAB products were generated from per-channel WAV files whose
samples equal the packetized ``.wavfm`` streams after a fixed channel DC
removal; this replay therefore estimates and removes the same whole-file
integer DC offsets before MUSIC.  The two strongest azimuth/zenith MUSIC
peaks are returned every 640 samples (five 128-point snapshots,
approximately 0.21 s).

This module is a frontend only.  It does not use GPS to produce DOA.  GPS is
used by the downstream association/calibration audit, never as an assimilation
observation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np


IP_TO_NODE = {40: 3, 43: 6, 46: 13, 47: 1, 48: 2, 49: 7, 5: 11, 54: 5, 61: 8}
NODE_TO_IP = {node: ip for ip, node in IP_TO_NODE.items()}

# Historical set_channel.m (1-based channel numbering).
CHANNEL_GROUPS = {
    "z": [19, 18, 17, 13, 14, 15, 16],
    "x": [9, 8, 7, 1, 2, 3],
    "y": [12, 11, 10, 4, 5, 6],
}

FS = 3050
NFFT = 128
FSNAP = 5
FRAME_SAMPLES = NFFT * FSNAP
C_SOUND = 340.0
K_SOURCES = 2
F_LOW = 100.0
F_HIGH = 500.0
GEOMETRY_SPACING = 0.50
GPS_ACOUSTIC_INDEX_OFFSET_S = 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hms_seconds(value: float) -> float:
    integer = int(value)
    frac = float(value) - integer
    text = str(integer).zfill(6)
    return int(text[:2]) * 3600 + int(text[2:4]) * 60 + int(text[4:]) + frac


def seconds_hhmmss(value: float) -> int:
    value = float(value)
    hour = int(value // 3600)
    minute = int((value - hour * 3600) // 60)
    second = int(value - hour * 3600 - minute * 60)
    return hour * 10000 + minute * 100 + second


def circular_difference(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def decode_wavfm(path: Path, channels: int = 20) -> tuple[np.ndarray, dict]:
    """Decode the packetized .wavfm file into channels x samples."""
    raw = np.fromfile(path, dtype="<i2")
    packet_width = channels + 2
    if raw.size % packet_width:
        raise RuntimeError(
            f"{path.name}: {raw.size} int16 values are not divisible by "
            f"packet width {packet_width}"
        )
    packets = raw.reshape(-1, packet_width)
    marker = packets[:, 0]
    if not np.all(marker == 23131):
        bad = int(np.count_nonzero(marker != 23131))
        raise RuntimeError(f"{path.name}: {bad} packets have invalid marker")
    sample_index = packets[:, -1]
    data = packets[:, 1:-1].T.astype(np.float64, copy=False)
    # The historical per-channel WAV files in the Baoding archive are the
    # packetized streams after integer channel-wise DC removal.  Replaying this
    # centering from .wavfm is essential; otherwise the sample covariance is
    # dominated by array electronics offsets rather than acoustic phase.
    channel_dc_offsets = np.rint(np.mean(data, axis=1)).astype(np.float64)
    data = data - channel_dc_offsets[:, None]
    # The packet index is 1..3051 and restarts each second. It is retained in
    # the manifest as an integrity check; packet order is the time axis.
    metadata = {
        "path": str(path),
        "sha256": sha256(path),
        "packet_count": int(packets.shape[0]),
        "packet_width_int16": packet_width,
        "decoded_channels": channels,
        "sample_rate_hz": FS,
        "marker_value": 23131,
        "sample_index_min": int(sample_index.min()),
        "sample_index_max": int(sample_index.max()),
        "sample_index_unique": int(np.unique(sample_index).size),
        "sample_index_resets": int(np.count_nonzero(np.diff(sample_index) <= 0)),
        "duration_s": float(data.shape[1] / FS),
        "channel_dc_offsets_removed": [float(value) for value in channel_dc_offsets],
        "dc_centering_protocol": "round(mean(.wavfm channel over full file)); matches archived per-channel WAV replay",
    }
    return data, metadata


def read_gpstime(path: Path) -> list[tuple[int, float]]:
    values = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) >= 2:
            try:
                values.append((int(float(fields[0])), float(fields[1])))
            except ValueError:
                pass
    if not values:
        raise RuntimeError(f"{path}: no timestamps")
    return values


def frequency_bins() -> tuple[np.ndarray, np.ndarray]:
    # Exact equivalent of Finx.m: MATLAB indices k1+1:k2+1 for k1..k2.
    frequencies = np.arange(NFFT // 2 + 1, dtype=float) * FS / NFFT
    keep = np.flatnonzero((frequencies >= F_LOW) & (frequencies <= F_HIGH))
    return keep, frequencies[keep]


def channel_vectors(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def take(group: list[int]) -> np.ndarray:
        return data[np.asarray(group, dtype=int) - 1]

    return take(CHANNEL_GROUPS["z"]), take(CHANNEL_GROUPS["x"]), take(CHANNEL_GROUPS["y"])


def frequency_decompose(block: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Return channel x snapshot x selected-frequency complex coefficients."""
    frames = block[:, :FRAME_SAMPLES].reshape(block.shape[0], FSNAP, NFFT)
    spectrum = np.fft.fft(frames, axis=-1) / math.sqrt(NFFT)
    return spectrum[:, :, bins]


def local_peak_indices(score: np.ndarray, count: int) -> list[int]:
    candidates = []
    n = len(score)
    for i in range(n):
        left = score[i - 1] if i > 0 else 0.0
        right = score[i + 1] if i + 1 < n else 0.0
        if score[i] > left and score[i] > right:
            candidates.append(i)
    candidates.sort(key=lambda i: float(score[i]), reverse=True)
    selected = candidates[:count]
    if len(selected) < count:
        for i in np.argsort(score)[::-1]:
            if int(i) not in selected:
                selected.append(int(i))
            if len(selected) == count:
                break
    return selected


def steering_score(
    coefficients: np.ndarray,
    positions: np.ndarray,
    frequencies: np.ndarray,
    angles_deg: np.ndarray,
    axis: str,
) -> np.ndarray:
    """Wideband MUSIC spectrum for one 1-D axis."""
    if axis == "azimuth":
        deploy = np.asarray([-3, -2, -1, 1, 2, 3], dtype=float) * GEOMETRY_SPACING
        # The archived Baoding dual-source products use the clockwise azimuth
        # convention documented in the historical MATLAB comments.  With the
        # packet/WAV-equivalent channel streams this is the negative-y steering
        # branch; the positive-y branch reproduces the same zeniths but mirrors
        # azimuth as 360-theta.
        deployment = np.concatenate(
            [deploy * np.cos(np.deg2rad(angles_deg))[:, None],
             -deploy * np.sin(np.deg2rad(angles_deg))[:, None]],
            axis=1,
        )
    else:
        deploy = np.asarray([-2.13, -1.53, -0.93, 0, 1, 2, 3], dtype=float) * GEOMETRY_SPACING
        deployment = deploy[None, :] * np.cos(np.deg2rad(angles_deg))[:, None]

    total = np.zeros(len(angles_deg), dtype=float)
    for fidx, frequency in enumerate(frequencies):
        if axis == "azimuth":
            xyz = np.concatenate([coefficients[:6, :, fidx], coefficients[6:, :, fidx]], axis=0)
        else:
            xyz = coefficients[:, :, fidx]
        covariance = xyz @ xyz.conj().T / coefficients.shape[1]
        _, eigenvectors = np.linalg.eigh(covariance)
        noise = eigenvectors[:, : eigenvectors.shape[1] - K_SOURCES]
        phase = 2.0 * math.pi * float(frequency) / C_SOUND
        if axis == "azimuth":
            steering = np.exp(1j * phase * deployment.T)
        else:
            steering = np.exp(1j * phase * deployment.T)
        projection = steering.conj().T @ noise
        total += np.sum(np.abs(projection) ** 2, axis=1).real
    return 1.0 / np.maximum(total, 1e-12)


def two_peak_1d(
    coefficients: np.ndarray,
    positions: np.ndarray,
    frequencies: np.ndarray,
    axis: str,
) -> tuple[np.ndarray, np.ndarray]:
    if axis == "azimuth":
        coarse = np.arange(5.0, 360.0, 5.0)
        low, high = 0.0, 360.0
    else:
        coarse = np.arange(5.0, 90.0, 5.0)
        low, high = 0.0, 90.0
    score = steering_score(coefficients, positions, frequencies, coarse, axis)
    coarse_indices = local_peak_indices(score, K_SOURCES)
    outputs, strengths = [], []
    for index in coarse_indices:
        center = float(coarse[index])
        fine = np.arange(center - 5.0, center + 5.0 + 1e-9, 1.0)
        if axis != "azimuth":
            fine = np.clip(fine, low, high)
        fine_score = steering_score(coefficients, positions, frequencies, fine, axis)
        winner = int(np.argmax(fine_score))
        outputs.append(float(fine[winner]))
        strengths.append(float(fine_score[winner]))
    return np.asarray(outputs, dtype=float), np.asarray(strengths, dtype=float)


def music_two_peaks(block: np.ndarray, bins: np.ndarray, frequencies: np.ndarray) -> dict:
    z, x, y = channel_vectors(block)
    x_coeff = frequency_decompose(x, bins)
    y_coeff = frequency_decompose(y, bins)
    z_coeff = frequency_decompose(z, bins)
    # The historical azimuth routine uses X and Y six-element arms, while the
    # elevation routine uses the seven-element z arm.
    azimuth, az_strength = two_peak_1d(
        np.concatenate([x_coeff, y_coeff], axis=0),
        np.zeros(12),
        frequencies,
        "azimuth",
    )
    zenith, zen_strength = two_peak_1d(z_coeff, np.zeros(7), frequencies, "elevation")
    return {
        "azimuth_1_deg": float(azimuth[0]),
        "zenith_1_deg": float(zenith[0]),
        "azimuth_2_deg": float(azimuth[1]),
        "zenith_2_deg": float(zenith[1]),
        "azimuth_strength_1": float(az_strength[0]),
        "azimuth_strength_2": float(az_strength[1]),
        "zenith_strength_1": float(zen_strength[0]),
        "zenith_strength_2": float(zen_strength[1]),
    }


def run_node(
    segment: Path,
    node: int,
    start_hhmmss: int,
    end_hhmmss: int,
    limit_frames: int | None,
) -> tuple[list[dict], dict]:
    ip = NODE_TO_IP[node]
    prefix = segment / f"20171107baoding_125536_{ip}_19"
    wavfm = prefix.with_suffix(".wavfm")
    gpstime = prefix.with_suffix(".gpstime")
    data, packet_meta = decode_wavfm(wavfm)
    timestamps = read_gpstime(gpstime)
    timestamp_seconds = [item[0] for item in timestamps]
    start_s = hms_seconds(start_hhmmss)
    end_s = hms_seconds(end_hhmmss)
    target_begin_second = int(start_s + GPS_ACOUSTIC_INDEX_OFFSET_S)
    target_end_second = int(end_s + GPS_ACOUSTIC_INDEX_OFFSET_S)
    try:
        begin_row = timestamp_seconds.index(target_begin_second)
        end_row = timestamp_seconds.index(target_end_second)
    except ValueError as exc:
        raise RuntimeError(
            f"node {node}: gpstime does not contain historical acoustic-index "
            f"window {target_begin_second}..{target_end_second}"
        ) from exc
    start_index = begin_row * FS
    stop_index = min(data.shape[1], (end_row + 1) * FS)
    if stop_index <= start_index:
        raise RuntimeError(f"node {node}: requested interval is outside decoded .wavfm")
    data_length = stop_index - start_index
    num_frames = int(math.floor(data_length / FRAME_SAMPLES)) - 1
    if num_frames <= 0:
        raise RuntimeError(f"node {node}: no complete historical 640-sample frames in requested interval")
    bins, frequencies = frequency_bins()
    rows = []
    frame_number = 0
    while frame_number < num_frames and (limit_frames is None or len(rows) < limit_frames):
        frame_start = start_index + frame_number * FRAME_SAMPLES
        tic = time.monotonic()
        output = music_two_peaks(data[:19, frame_start:frame_start + FRAME_SAMPLES], bins, frequencies)
        frame_time = start_s + frame_number * FRAME_SAMPLES / FS
        rows.append(
            {
                "node_id": node,
                "ip_suffix": ip,
                "time_s": frame_time,
                "time_hhmmss": seconds_hhmmss(frame_time),
                "frame_start_sample": frame_start,
                "frontend_runtime_s": time.monotonic() - tic,
                **output,
            }
        )
        frame_number += 1
    manifest = {
        "node_id": node,
        "ip_suffix": ip,
        "wavfm": packet_meta,
        "gpstime": {"path": str(gpstime), "sha256": sha256(gpstime), "first": timestamps[0], "last": timestamps[-1]},
        "protocol": {
            "channels_used": list(range(1, 20)),
            "channels_excluded": [20],
            "channel_groups": CHANNEL_GROUPS,
            "sample_rate_hz": FS,
            "nfft": NFFT,
            "snapshots": FSNAP,
            "frame_samples": FRAME_SAMPLES,
            "frequency_bins": bins.tolist(),
            "frequencies_hz": frequencies.tolist(),
            "sound_speed_mps": C_SOUND,
            "geometry_spacing_m": GEOMETRY_SPACING,
            "sources": K_SOURCES,
            "coarse_angle_step_deg": 5,
            "fine_angle_step_deg": 1,
            "gps_acoustic_index_offset_s": GPS_ACOUSTIC_INDEX_OFFSET_S,
            "frame_time_rule": "floor(start_hhmmss + frame_index * 640 / 3050), matching archived deal_doa timing",
            "peak_order_rule": "descending MUSIC peak strength from one_dim_search, not sorted by angle",
            "azimuth_convention": "historical clockwise Baoding convention; negative-y steering branch",
        },
        "requested_interval": {"start_hhmmss": start_hhmmss, "end_hhmmss": end_hhmmss},
        "historical_window_samples": {
            "gpstime_begin_second": target_begin_second,
            "gpstime_end_second": target_end_second,
            "start_sample_zero_based": start_index,
            "stop_sample_exclusive": stop_index,
            "data_length_samples": data_length,
            "num_frames_floor_minus_one": num_frames,
        },
        "rows": len(rows),
    }
    return rows, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-root", type=Path, default=Path("<PRIVATE_DATA_ROOT>/2017保定实验/2017保定实验"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--node", type=int, required=True)
    parser.add_argument("--start", type=int, default=125540)
    parser.add_argument("--end", type=int, default=125900)
    parser.add_argument("--limit-frames", type=int)
    args = parser.parse_args()

    segment = args.remote_root / "20171107保定实验/project/20171107baoding/shuangyuan_4"
    rows, manifest = run_node(segment, args.node, args.start, args.end, args.limit_frames)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = f"dual_doa_node_{args.node}_{args.start}_{args.end}"
    with (args.output / f"{stem}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    manifest["script_sha256"] = sha256(Path(__file__))
    (args.output / f"{stem}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
