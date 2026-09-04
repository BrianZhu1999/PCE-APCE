#!/usr/bin/env python3
"""Continuously resample packet-aligned Baoding streams to a common 3 kHz clock.

Unlike the earlier second-wise repair, this workflow concatenates complete
packet-index seconds first and performs one continuous FFT resampling per node.
This avoids a fresh resampling-filter transient at every second boundary and
preserves inter-node phase as far as the archived packet clocks permit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample

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

    reference_gps = base.parse_gpstime(args.segment / f"20171107baoding_132614_{base.NODE_TO_IP[1]}_19.gpstime")
    start_seconds = base.hms_seconds(args.start_hhmmss)
    gps_row = min(range(len(reference_gps)), key=lambda index: abs(reference_gps[index][1] - start_seconds))
    first_complete = max(0, gps_row - 1)
    target_samples = args.seconds * base.TARGET_FS
    args.output_root.mkdir(parents=True, exist_ok=True)
    nodes = {}

    for node in base.PAPER_NODES:
        ip = base.NODE_TO_IP[node]
        source = args.segment / f"20171107baoding_132614_{ip}_19.wavfm"
        raw = np.fromfile(source, dtype="<i2")
        packets = raw.reshape(-1, base.PACKET_WIDTH)
        if np.any(packets[:, 0] != base.MARKER):
            raise RuntimeError(f"invalid marker in {source}")
        complete, repair = base.repair_segments(packets)
        selected = complete[first_complete:first_complete + args.seconds]
        if len(selected) != args.seconds:
            raise RuntimeError(f"{source}: only {len(selected)} complete seconds available")
        source_counts = [len(segment) for segment in selected]
        values = np.concatenate([segment[:, 1:20] for segment in selected], axis=0).astype(np.float64)
        offsets = np.rint(np.mean(packets[:, 1:20].astype(np.float64), axis=0))
        values -= offsets[None, :]
        # One resampling operation for the full interval preserves continuous
        # phase; only the two outer edges experience FFT-resampling wrap effects.
        synchronized = resample(values, target_samples, axis=0).T.astype(np.float32)
        output = args.output_root / f"node{node}_ip{ip}_3khz.npy"
        np.save(output, synchronized, allow_pickle=False)
        boundary = synchronized.reshape(19, args.seconds, base.TARGET_FS)
        jumps = np.linalg.norm(boundary[:, 1:, 0] - boundary[:, :-1, -1], axis=0)
        scale = np.sqrt(np.mean(boundary[:, 1:, :] ** 2, axis=(0, 2))) + 1e-9
        nodes[str(node)] = {
            "node_id": node, "ip_suffix": ip, "source": str(source),
            "source_sha256": sha256(source), "source_samples": int(len(values)),
            "source_counts_per_second": source_counts,
            "effective_source_rate_hz": float(len(values) / args.seconds),
            "target_samples": target_samples, "target_rate_hz": base.TARGET_FS,
            "output": str(output), "output_sha256": sha256(output),
            "boundary_jump_ratio_median": float(np.median(jumps / scale)),
            "boundary_jump_ratio_p95": float(np.percentile(jumps / scale, 95)),
            "repair": repair,
        }

    manifest = {
        "claim_status": "continuous_packet_aligned_3khz_sync",
        "segment": str(args.segment), "start_hhmmss": args.start_hhmmss,
        "selected_gpstime_row": gps_row, "first_complete_segment": first_complete,
        "seconds": args.seconds, "target_rate_hz": base.TARGET_FS,
        "resampling": "single scipy.signal.resample call per node over the full interval",
        "gps_role": "window selection only; never a tracking observation",
        "nodes": nodes,
    }
    (args.output_root / "sync_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"claim_status": manifest["claim_status"], "seconds": args.seconds, "nodes": {key: {"effective_source_rate_hz": value["effective_source_rate_hz"], "boundary_jump_ratio_median": value["boundary_jump_ratio_median"]} for key, value in nodes.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
