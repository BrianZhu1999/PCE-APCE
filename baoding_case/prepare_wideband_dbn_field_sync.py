#!/usr/bin/env python3
"""Repair and resample the Baoding packet streams onto the paper time base.

The source files are packetized and contain a partial segment at each edge.
Node IP5 also contains one internal segment split by a corrupted sample index;
the two adjacent pieces are merged before resampling. Only complete one-second
segments are admitted. GPS timestamps select the requested window but never
enter the acoustic observation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly


PAPER_NODES = (1, 3, 5, 6, 7, 8, 11, 13)
NODE_TO_IP = {1: 47, 3: 40, 5: 54, 6: 43, 7: 49, 8: 61, 11: 5, 13: 46}
CHANNELS = 19
PACKET_WIDTH = CHANNELS + 3
MARKER = 23131
TARGET_FS = 3000


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


def parse_gpstime(path: Path) -> list[tuple[int, float]]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) >= 2:
            try:
                rows.append((int(float(fields[0])), hms_seconds(float(fields[1]))))
            except ValueError:
                pass
    if not rows:
        raise RuntimeError(f"empty gpstime file: {path}")
    return rows


def split_segments(indices: np.ndarray) -> list[tuple[int, int]]:
    reset = np.flatnonzero(np.diff(indices) <= 0) + 1
    starts = np.r_[0, reset]
    ends = np.r_[reset, len(indices)]
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def repair_segments(packets: np.ndarray) -> tuple[list[np.ndarray], dict]:
    indices = packets[:, -1].astype(np.int64)
    raw_segments = split_segments(indices)
    # Merge an internal split caused by one invalid index. The expected source
    # segment is approximately one second; only internal adjacent pieces whose
    # combined length is 2900..3100 are eligible for this repair.
    segments: list[np.ndarray] = []
    merges = []
    i = 0
    while i < len(raw_segments):
        a, b = raw_segments[i]
        length = b - a
        if 0 < i < len(raw_segments) - 1 and length < 2900:
            c, d = raw_segments[i + 1]
            if 2900 <= length + (d - c) <= 3100:
                segments.append(packets[a:d])
                merges.append({"left_raw_segment": i, "right_raw_segment": i + 1, "combined_length": d - a})
                i += 2
                continue
        segments.append(packets[a:b])
        i += 1

    if len(segments) < 3:
        raise RuntimeError("too few packet segments after repair")
    first_len = len(segments[0])
    last_len = len(segments[-1])
    complete = segments[1:-1]
    bad_complete = [len(x) for x in complete if not 2900 <= len(x) <= 3100]
    if bad_complete:
        raise RuntimeError(f"unrepaired interior segment lengths: {bad_complete[:12]}")
    return complete, {
        "raw_segment_count": len(raw_segments),
        "repaired_segment_count": len(segments),
        "complete_segment_count": len(complete),
        "dropped_first_segment_samples": first_len,
        "dropped_last_segment_samples": last_len,
        "internal_merges": merges,
        "complete_source_samples_min": min(len(x) for x in complete),
        "complete_source_samples_median": float(np.median([len(x) for x in complete])),
        "complete_source_samples_max": max(len(x) for x in complete),
    }


def resample_segment(segment: np.ndarray) -> np.ndarray:
    source_n = int(segment.shape[0])
    ratio = Fraction(TARGET_FS, source_n).limit_denominator(10000)
    result = resample_poly(segment.T.astype(np.float64), ratio.numerator, ratio.denominator, axis=1).T
    if result.shape[0] > TARGET_FS:
        result = result[:TARGET_FS]
    elif result.shape[0] < TARGET_FS:
        result = np.pad(result, ((0, TARGET_FS - result.shape[0]), (0, 0)), mode="edge")
    return result


def decode_window(path: Path, first_segment: int, segment_count: int) -> tuple[np.ndarray, dict]:
    raw = np.fromfile(path, dtype="<i2")
    if raw.size % PACKET_WIDTH:
        raise RuntimeError(f"{path}: invalid packet payload length")
    packets = raw.reshape(-1, PACKET_WIDTH)
    if np.any(packets[:, 0] != MARKER):
        raise RuntimeError(f"{path}: invalid packet marker")
    complete, repair = repair_segments(packets)
    selected = complete[first_segment:first_segment + segment_count]
    if len(selected) != segment_count:
        raise RuntimeError(f"{path}: requested segment window is unavailable")
    blocks = []
    dc_offsets = []
    source_counts = []
    for segment in selected:
        values = segment[:, 1:20].astype(np.float64)
        offsets = np.rint(np.mean(values, axis=0))
        values = values - offsets[None, :]
        blocks.append(resample_segment(values))
        dc_offsets.append(offsets)
        source_counts.append(len(segment))
    data = np.concatenate(blocks, axis=0).T.astype(np.float32)
    repair["selected_source_counts"] = source_counts
    repair["selected_target_samples"] = int(data.shape[1])
    repair["dc_offset_mean_abs"] = float(np.mean(np.abs(np.concatenate(dc_offsets))))
    return data, repair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment", type=Path, required=True)
    parser.add_argument("--start-hhmmss", type=float, default=132754.0)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if args.seconds <= 0:
        raise ValueError("--seconds must be positive")
    gps_rows = parse_gpstime(args.segment / f"20171107baoding_132614_{NODE_TO_IP[1]}_19.gpstime")
    start_seconds = hms_seconds(args.start_hhmmss)
    nearest = min(range(len(gps_rows)), key=lambda i: abs(gps_rows[i][1] - start_seconds))
    # Complete segment 0 is the first full second after the leading partial;
    # gpstime row 0 labels that same first full second in the archive.
    # ``complete[0]`` corresponds to gpstime row 1 because raw segment 0 is
    # the leading partial second that is discarded. Map the requested GPS row
    # back to the repaired complete-segment index explicitly.
    first_segment = max(0, nearest - 1)
    args.output_root.mkdir(parents=True, exist_ok=True)
    node_manifests = {}
    for node in PAPER_NODES:
        ip = NODE_TO_IP[node]
        wavfm = args.segment / f"20171107baoding_132614_{ip}_19.wavfm"
        data, repair = decode_window(wavfm, first_segment, args.seconds)
        out = args.output_root / f"node{node}_ip{ip}_3khz.npy"
        np.save(out, data, allow_pickle=False)
        node_manifests[str(node)] = {
            "node_id": node,
            "ip_suffix": ip,
            "wavfm": str(wavfm),
            "wavfm_sha256": sha256(wavfm),
            "output": str(out),
            "shape_channels_samples": list(data.shape),
            "target_sample_rate_hz": TARGET_FS,
            "repair": repair,
        }

    manifest = {
        "claim_status": "paper_3khz_packet_repaired_window",
        "segment": str(args.segment),
        "paper_nodes": list(PAPER_NODES),
        "node_to_ip": NODE_TO_IP,
        "requested_start_hhmmss": args.start_hhmmss,
        "selected_gpstime": gps_rows[nearest][1],
        "selected_gpstime_row": nearest,
        "complete_segment_index": first_segment,
        "mapping_note": "complete segment index = gpstime row index - 1 after dropping leading partial segment",
        "seconds": args.seconds,
        "target_sample_rate_hz": TARGET_FS,
        "channels_per_node": CHANNELS,
        "gps_role": "window selection and provenance only; never runtime observation",
        "nodes": node_manifests,
        "next_gate": "cross-node frame coherence and short DBN field smoke",
    }
    (args.output_root / "sync_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
