#!/usr/bin/env python3
"""Prepare the Baoding Fig. 15 window on the archived packet clock.

This follows the delivered wavtransZQ.m convention: retain the native packet
samples, subtract a channel mean, and divide 16-bit values by 32768. GPS time
selects the first packet second only and never enters observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import prepare_wideband_dbn_field_sync as base


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment", type=Path, required=True)
    parser.add_argument("--start-hhmmss", type=float, default=132754.0)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.seconds <= 0:
        raise ValueError("seconds must be positive")

    gpstime_path = args.segment / f"20171107baoding_132614_{base.NODE_TO_IP[1]}_19.gpstime"
    gps_rows = base.parse_gpstime(gpstime_path)
    start_seconds = base.hms_seconds(args.start_hhmmss)
    nearest = min(range(len(gps_rows)), key=lambda index: abs(gps_rows[index][1] - start_seconds))
    first_complete = max(0, nearest - 1)
    args.output_root.mkdir(parents=True, exist_ok=True)
    nodes = {}
    frequencies = np.arange(3, 1498, 3, dtype=np.int64)
    real_frames = []
    shifted_frames = []

    selected_by_node = {}

    for node in base.PAPER_NODES:
        ip = base.NODE_TO_IP[node]
        source = args.segment / f"20171107baoding_132614_{ip}_19.wavfm"
        raw = np.fromfile(source, dtype="<i2")
        if raw.size % base.PACKET_WIDTH:
            raise RuntimeError(f"invalid packet payload length: {source}")
        packets = raw.reshape(-1, base.PACKET_WIDTH)
        if np.any(packets[:, 0] != base.MARKER):
            raise RuntimeError(f"invalid packet marker: {source}")
        complete, repair = base.repair_segments(packets)
        selected = complete[first_complete : first_complete + args.seconds]
        if len(selected) != args.seconds:
            raise RuntimeError(f"{source}: requested interval unavailable")
        counts = [int(len(segment)) for segment in selected]
        selected_by_node[node] = selected
        values = np.concatenate([segment[:, 1:20] for segment in selected], axis=0).astype(np.float64)
        channel_mean = np.mean(values, axis=0)
        native = ((values - channel_mean[None, :]) / 32768.0).T.astype(np.float32)
        output = args.output_root / f"node{node}_ip{ip}_3khz.npy"
        np.save(output, native, allow_pickle=False)
        nodes[str(node)] = {
            "ip_suffix": ip,
            "source": str(source),
            "source_sha256": sha256(source),
            "selected_source_counts": counts,
            "channel_mean_before_scaling": channel_mean.tolist(),
            "output": str(output),
            "output_sha256": sha256(output),
            "shape": list(native.shape),
            "repair": repair,
        }

    for frame_index in range(args.seconds):
        real_nodes = []
        shifted_nodes = []
        for node_index, node in enumerate(base.PAPER_NODES):
            values = selected_by_node[node][frame_index][:, 1:20].astype(np.float64)
            values = (values - np.mean(values, axis=0, keepdims=True)) / 32768.0
            sample_count = values.shape[0]
            if sample_count <= 2 * int(frequencies[-1]):
                raise RuntimeError(f"node {node} frame {frame_index}: Nyquist below 1497 Hz")
            bins = np.remainder(sample_count - frequencies, sample_count)
            real_nodes.append((np.fft.fft(values, axis=0)[bins] / sample_count).astype(np.complex64))
            shifted = np.roll(values, shift=node_index * 73, axis=0)
            shifted_nodes.append((np.fft.fft(shifted, axis=0)[bins] / sample_count).astype(np.complex64))
        real_frames.append(np.concatenate(real_nodes, axis=1))
        shifted_frames.append(np.concatenate(shifted_nodes, axis=1))

    real_spectrum = np.stack(real_frames, axis=0)
    shifted_spectrum = np.stack(shifted_frames, axis=0)
    real_spectrum_path = args.output_root / "spectrum_real.npy"
    shifted_spectrum_path = args.output_root / "spectrum_node_shift.npy"
    np.save(real_spectrum_path, real_spectrum, allow_pickle=False)
    np.save(shifted_spectrum_path, shifted_spectrum, allow_pickle=False)

    manifest = {
        "claim_status": "native_packet_clock_wavtransZQ_convention",
        "segment": str(args.segment),
        "start_hhmmss": args.start_hhmmss,
        "selected_gpstime_row": nearest,
        "selected_gpstime_seconds": gps_rows[nearest][1],
        "first_complete_segment": first_complete,
        "seconds": args.seconds,
        "sample_rate_hz": "node-specific packet seconds (IP5 about 3000 Hz; other nodes about 3050 Hz)",
        "preprocessing": "per-packet-second channel mean subtraction; divide by 32768; exact integer-Hz DFT bins; no resampling",
        "frequency_grid_hz": "3,6,...,1497",
        "frequency_count": int(len(frequencies)),
        "spectrum_real": str(real_spectrum_path),
        "spectrum_real_sha256": sha256(real_spectrum_path),
        "spectrum_node_shift": str(shifted_spectrum_path),
        "spectrum_node_shift_sha256": sha256(shifted_spectrum_path),
        "spectrum_shape": list(real_spectrum.shape),
        "gps_role": "window selection only",
        "nodes": nodes,
    }
    manifest_path = args.output_root / "sync_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "nodes": len(nodes), "seconds": args.seconds}, ensure_ascii=False))


if __name__ == "__main__":
    main()
