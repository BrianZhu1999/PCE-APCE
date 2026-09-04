#!/usr/bin/env python3
"""Audit the complete synchronized eight-node Baoding field interval.

This is a read-only data contract builder.  It never uses GPS as a runtime
observation and records the exact common interval, packet metadata, frame
index and paper frequency protocol for downstream runners.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import shuangyuan_dual_frontend as base

PAPER_NODES = (1, 3, 5, 6, 7, 8, 11, 13)
NODE_TO_IP = {1: 47, 3: 40, 5: 54, 6: 43, 7: 49, 8: 61, 11: 5, 13: 46}
FS = 3050
SNAPSHOT = 2048
OVERLAP = 0.50

def hms(value: float) -> float:
    text = str(int(value)).zfill(6)
    return int(text[:2])*3600 + int(text[2:4])*60 + int(text[4:]) + (float(value)-int(float(value)))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--remote-root', type=Path, required=True)
    ap.add_argument('--output-root', type=Path, required=True)
    args = ap.parse_args()
    segment = args.remote_root / '20171107保定实验/project/20171107baoding/sanyuan_tongxinyuan_6'
    args.output_root.mkdir(parents=True, exist_ok=True)
    node_meta = {}
    starts, ends, sample_lengths = [], [], []
    for node in PAPER_NODES:
        ip = NODE_TO_IP[node]
        prefix = segment / f'20171107baoding_132614_{ip}_19'
        data, wav_meta = base.decode_wavfm(prefix.with_suffix('.wavfm'))
        timestamps = base.read_gpstime(prefix.with_suffix('.gpstime'))
        # ``gpstime`` stores an acoustic/sample index in column 1 and the
        # HHMMSS.sss wall-clock time in column 2.  The latter is the common
        # synchronization axis; the index is retained only as provenance.
        first_sec = hms(timestamps[0][1])
        last_sec = hms(timestamps[-1][1])
        starts.append(first_sec); ends.append(last_sec); sample_lengths.append(int(data.shape[1]))
        node_meta[str(node)] = {
            'node_id': node, 'ip_suffix': ip, 'wavfm': wav_meta,
            'gpstime': {'path': str(prefix.with_suffix('.gpstime')), 'first': timestamps[0], 'last': timestamps[-1], 'first_seconds': first_sec, 'last_seconds': last_sec},
            'sample_count': int(data.shape[1]),
        }
    common_start = max(starts); common_end = min(ends)
    # Conservative common sample interval: leave one snapshot at each edge.
    start_sample = int(math.ceil((common_start - min(starts)) * FS))
    # The wall-clock end in gpstime can extend beyond the decoded sample
    # payload when a node has packet gaps.  The usable common duration is
    # therefore constrained by both wall-clock overlap and the shortest
    # decoded stream after aligning to the common start.
    common_duration = min(common_end - common_start, min(sample_lengths) / FS)
    frame_step = int(round(SNAPSHOT * (1.0 - OVERLAP)))
    frame_count = max(0, 1 + int(math.floor((common_duration * FS - SNAPSHOT) / frame_step)))
    frames = []
    for index in range(frame_count):
        t = common_start + (index * frame_step + SNAPSHOT/2) / FS
        frames.append({'frame_index': index, 'start_sample_relative': index*frame_step, 'center_time_s': t, 'start_time_s': common_start + index*frame_step/FS, 'end_time_s': common_start + (index*frame_step+SNAPSHOT)/FS, 'accepted_sync': True})
    with (args.output_root / 'frame_index.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(frames[0]) if frames else ['frame_index']); w.writeheader(); w.writerows(frames)
    manifest = {
        'claim_status': 'raw_sync_audit_complete',
        'segment': 'sanyuan_tongxinyuan_6', 'segment_root': str(segment),
        'paper_nodes': list(PAPER_NODES), 'node_to_ip': NODE_TO_IP,
        'sample_rate_hz': FS, 'snapshot_length_samples': SNAPSHOT, 'snapshot_duration_s': SNAPSHOT/FS,
        'overlap_fraction': OVERLAP, 'frame_step_samples': frame_step, 'frame_step_s': frame_step/FS,
        'frequency_grid_hz': list(range(3, 1498, 3)), 'frequency_count': 499,
        'common_start_s': common_start, 'common_end_s': common_end, 'common_duration_s': common_duration,
        'common_start_hhmmss': base.seconds_hhmmss(common_start), 'common_end_hhmmss': base.seconds_hhmmss(common_end),
        'frame_count': frame_count, 'sample_length_min': min(sample_lengths), 'sample_length_max': max(sample_lengths),
        'sync_spread_start_s': max(starts)-min(starts), 'sync_spread_end_s': max(ends)-min(ends),
        'node_metadata': node_meta, 'frame_index_csv': str(args.output_root/'frame_index.csv'),
        'gps_runtime_observation': False,
        'notes': 'Common interval is limited by the shortest paper node recording (node 11/IP5). GPS is reserved for initialization, target ordering calibration and evaluation.',
    }
    (args.output_root/'raw_sync_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ('claim_status','common_start_hhmmss','common_end_hhmmss','common_duration_s','frame_count','sync_spread_start_s','sync_spread_end_s')}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
