#!/usr/bin/env python3
"""Interpolate every Baoding node onto one absolute GPS-time sample grid."""
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


def complete_segments(path: Path):
    raw = np.fromfile(path, dtype="<i2").reshape(-1, base.PACKET_WIDTH)
    complete, repair = base.repair_segments(raw)
    return complete, repair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment", type=Path, required=True)
    parser.add_argument("--start-hhmmss", type=float, default=132754.0)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    requested = base.hms_seconds(args.start_hhmmss)
    target_times = requested + np.arange(args.seconds * base.TARGET_FS, dtype=np.float64) / base.TARGET_FS
    args.output_root.mkdir(parents=True, exist_ok=True)
    node_manifests = {}

    for node in base.PAPER_NODES:
        ip = base.NODE_TO_IP[node]
        prefix = args.segment / f"20171107baoding_132614_{ip}_19"
        gpstime = base.parse_gpstime(prefix.with_suffix(".gpstime"))
        segments, repair = complete_segments(prefix.with_suffix(".wavfm"))
        # complete[0] corresponds to gpstime row 1 after the leading partial.
        available = min(len(segments), len(gpstime) - 1)
        segment_times = np.asarray([gpstime[index + 1][1] for index in range(available)], dtype=np.float64)
        selected_indices = [index for index, time_s in enumerate(segment_times) if requested - 2.0 <= time_s <= requested + args.seconds + 2.0]
        if not selected_indices:
            raise RuntimeError(f"node {node}: no timestamped segments near requested interval")
        values = []
        times = []
        for index in selected_indices:
            segment = segments[index]
            count = len(segment)
            # The gpstime timestamp labels the first sample of the corresponding
            # full packet-index second. Use the next timestamp when available to
            # infer its actual sampling interval and preserve clock drift.
            start = segment_times[index]
            if index + 1 < len(segment_times):
                stop = segment_times[index + 1]
            else:
                stop = start + 1.0
            sample_times = start + np.arange(count, dtype=np.float64) * ((stop - start) / count)
            values.append(segment[:, 1:20].astype(np.float64))
            times.append(sample_times)
        values_np = np.concatenate(values, axis=0)
        source_times = np.concatenate(times)
        offsets = np.rint(np.mean(values_np, axis=0))
        values_np -= offsets[None, :]
        synchronized = np.vstack([np.interp(target_times, source_times, values_np[:, channel]) for channel in range(19)]).astype(np.float32)
        output = args.output_root / f"node{node}_ip{ip}_3khz.npy"
        np.save(output, synchronized, allow_pickle=False)
        node_manifests[str(node)] = {
            "node_id": node, "ip_suffix": ip,
            "wavfm": str(prefix.with_suffix('.wavfm')), "wavfm_sha256": sha256(prefix.with_suffix('.wavfm')),
            "gpstime": str(prefix.with_suffix('.gpstime')), "gpstime_sha256": sha256(prefix.with_suffix('.gpstime')),
            "selected_segment_start_time": float(source_times[0]), "selected_segment_end_time": float(source_times[-1]),
            "target_start_time": float(target_times[0]), "target_end_time": float(target_times[-1]),
            "source_sample_count": int(len(source_times)), "target_sample_count": int(len(target_times)),
            "nearest_source_time_error_start_s": float(np.min(np.abs(source_times - target_times[0]))),
            "output": str(output), "output_sha256": sha256(output), "repair": repair,
        }

    manifest = {"claim_status": "absolute_gpstime_interpolated_3khz_sync", "segment": str(args.segment), "start_hhmmss": args.start_hhmmss, "start_seconds": requested, "seconds": args.seconds, "target_rate_hz": base.TARGET_FS, "method": "per-node packet samples interpolated to one absolute GPS-time grid using per-node gpstime boundaries", "gps_role": "sensor clock synchronization only; target GPS tracks are not read", "nodes": node_manifests}
    (args.output_root / "sync_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"claim_status": manifest["claim_status"], "nodes": {key: {"source_start": value["selected_segment_start_time"], "start_error_s": value["nearest_source_time_error_start_s"]} for key, value in node_manifests.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
