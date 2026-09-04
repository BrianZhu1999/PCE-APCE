#!/usr/bin/env python3
"""Audit whether the Zhang--Bao paper field setup matches local Baoding data."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


PAPER_TARGETS = {
    "target1": (38614853.4, 4337388.27),
    "target2": (38615012.2, 4336467.20),
    "target3": (38615647.2, 4337215.10),
}
PAPER_IPS = (47, 43, 61, 5, 40, 46, 54, 49)
NODE_TO_IP = {1: 47, 2: 48, 3: 40, 5: 54, 6: 43, 7: 49, 8: 61, 11: 5, 13: 46}


def gps_rows(path: Path) -> list[tuple[float, float, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 8:
            continue
        try:
            rows.append((float(fields[4]), float(fields[5]), fields[7]))
        except ValueError:
            continue
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gps-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gps_files = {
        "GPS1": args.gps_root / "GPS1_plane1.gps",
        "GPS3": args.gps_root / "GPS3_plane2.gps",
        "GPS4": args.gps_root / "GPS4_plane2to3.gps",
    }
    matches = {}
    for target, coordinate in PAPER_TARGETS.items():
        best = None
        for gps_name, path in gps_files.items():
            for x, y, time_code in gps_rows(path):
                distance = math.hypot(x - coordinate[0], y - coordinate[1])
                candidate = (distance, gps_name, time_code, x, y)
                if best is None or candidate < best:
                    best = candidate
        matches[target] = {
            "paper_initial_xy": coordinate,
            "nearest_gps_file": best[1],
            "nearest_gps_time_hhmmss": best[2],
            "nearest_gps_xy": best[3:5],
            "distance_m": best[0],
        }
    result = {
        "paper": {
            "field_site": "Baoding Aviation Sports School, Hebei, China",
            "field_date": "November 2017",
            "array_nodes": 8,
            "microphones_per_node": 19,
            "sampling_rate_hz": 3000,
            "paper_node_ips": PAPER_IPS,
            "public_code": "https://github.com/zhangwenqiong2017/MTT_WB_DBN",
            "public_code_commit_audited": "8d05b864246f226b28b6baec0958deaefad41046",
        },
        "local_baoding_mapping": {
            "node_to_ip": NODE_TO_IP,
            "paper_node_ids": [node for node, ip in NODE_TO_IP.items() if ip in PAPER_IPS],
            "excluded_local_node_ids": [node for node, ip in NODE_TO_IP.items() if ip not in PAPER_IPS],
            "candidate_segment": "sanyuan_tongxinyuan_6",
        },
        "initial_target_matches": matches,
        "interpretation": "The paper field setup and three initial GPS coordinates match the local Baoding archive; this is strong evidence that tongxinyuan_6 is the paper-relevant segment. It does not prove that the paper's private real-data tracker implementation is in the public repository.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
