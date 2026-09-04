#!/usr/bin/env python3
"""Audit packet clocks before the wideband DBN field reproduction.

The Baoding archive contains packetized ``.wavfm`` streams whose effective
sample clocks are not identical across nodes. This read-only audit records
per-second packet counts, index anomalies, and the resampling contract needed
to put all nodes on the paper's nominal 3 kHz time base.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


PAPER_NODES = (1, 3, 5, 6, 7, 8, 11, 13)
NODE_TO_IP = {1: 47, 3: 40, 5: 54, 6: 43, 7: 49, 8: 61, 11: 5, 13: 46}
PACKET_CHANNELS = 20
PACKET_MARKER = 23131
PAPER_FS = 3000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hms_seconds(value: float) -> float:
    integer = int(value)
    text = str(integer).zfill(6)
    return int(text[:2]) * 3600 + int(text[2:4]) * 60 + int(text[4:]) + value - integer


def audit_stream(path: Path) -> dict:
    raw = np.fromfile(path, dtype="<i2")
    width = PACKET_CHANNELS + 2
    if raw.size % width:
        raise RuntimeError(f"{path}: {raw.size} values not divisible by {width}")
    packets = raw.reshape(-1, width)
    marker_bad = int(np.count_nonzero(packets[:, 0] != PACKET_MARKER))
    indices = packets[:, -1].astype(np.int64)
    reset = np.flatnonzero(np.diff(indices) <= 0) + 1
    starts = np.r_[0, reset]
    ends = np.r_[reset, len(indices)]
    counts = (ends - starts).astype(int)
    bad_index = int(np.count_nonzero((indices < 0) | (indices > 3051)))
    normal = counts[(counts >= 2900) & (counts <= 3100)]
    return {
        "path": str(path),
        "sha256": sha256(path),
        "packet_count": int(len(indices)),
        "packet_width_int16": width,
        "marker_bad_packets": marker_bad,
        "sample_index_min": int(indices.min()),
        "sample_index_max": int(indices.max()),
        "negative_or_gt_3051_indices": bad_index,
        "second_segment_count": int(len(counts)),
        "segment_packet_count_min": int(counts.min()),
        "segment_packet_count_median": float(np.median(counts)),
        "segment_packet_count_max": int(counts.max()),
        "segments_outside_2900_3100": int(len(counts) - len(normal)),
        "resampling_contract": {
            "target_sample_rate_hz": PAPER_FS,
            "source_clock_interpretation": "packet order within each GPS-second segment",
            "resampler": "polyphase rational resampling after segment-wise packet repair",
            "gps_role": "segment boundary and evaluation time only; never acoustic observation",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for node in PAPER_NODES:
        path = args.segment / f"20171107baoding_132614_{NODE_TO_IP[node]}_19.wavfm"
        item = audit_stream(path)
        item["node_id"] = node
        item["ip_suffix"] = NODE_TO_IP[node]
        rows.append(item)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "node_id", "ip_suffix", "packet_count", "packet_width_int16",
            "marker_bad_packets",
            "sample_index_min", "sample_index_max", "negative_or_gt_3051_indices",
            "second_segment_count", "segment_packet_count_min",
            "segment_packet_count_median", "segment_packet_count_max",
            "segments_outside_2900_3100", "sha256", "path",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)

    manifest = {
        "claim_status": "field_packet_sync_audit",
        "paper_sample_rate_hz": PAPER_FS,
        "paper_nodes": list(PAPER_NODES),
        "node_to_ip": NODE_TO_IP,
        "segment": str(args.segment),
        "gps_time_source": "per-node .gpstime files; detailed boundary audit remains required",
        "nodes": rows,
        "admission": {
            "status": "blocked_pending_segment_repair",
            "reason": "source packet clocks differ and at least one stream contains index anomalies",
            "next_step": "repair each GPS-second segment, resample to 3000 Hz, and verify cross-node frame coherence",
        },
        "csv": str(csv_path),
    }
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
