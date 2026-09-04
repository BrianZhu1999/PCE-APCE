#!/usr/bin/env python3
"""Prepare the paper-aligned Baoding tongxinyuan_6 truth/provenance bundle."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

PAPER_NODES = (1, 3, 5, 6, 7, 8, 11, 13)
TARGET_FILES = {1: "GPS1_plane1.gps", 2: "GPS3_plane2.gps", 3: "GPS4_plane2to3.gps"}
REMOTE_ROOT = "<PRIVATE_DATA_ROOT>/2017保定实验/2017保定实验"
REMOTE_SEGMENT = f"{REMOTE_ROOT}/20171107保定实验/project/20171107baoding/sanyuan_tongxinyuan_6"
REMOTE_GPS = f"{REMOTE_ROOT}/20171107保定实验/GPS_data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_gps(path: Path) -> list[dict[str, float]]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 8:
            continue
        try:
            rows.append({"x": float(fields[4]), "y": float(fields[5]), "z": float(fields[6]), "time_code": int(float(fields[7]))})
        except ValueError:
            continue
    return rows


def hms_seconds(value: int) -> int:
    text = str(value).zfill(6)
    return int(text[:2]) * 3600 + int(text[2:4]) * 60 + int(text[4:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gps-root", type=Path, required=True)
    parser.add_argument("--frame-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    frame_rows = []
    with args.frame_csv.open(encoding="utf-8", newline="") as stream:
        frame_rows = list(csv.DictReader(stream))
    times = [float(row["time_s"]) for row in frame_rows]
    if not times:
        raise RuntimeError("frame CSV is empty")
    node_coords = {}
    with (args.gps_root / "20171107baoding.nod").open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            fields = line.split()
            if len(fields) >= 6:
                node = int(fields[2])
                if node in PAPER_NODES:
                    node_coords[node] = {"ip": int(fields[0].split(".")[-1]), "x": float(fields[3]), "y": float(fields[4]), "z": float(fields[5])}
    if sorted(node_coords) != list(PAPER_NODES):
        raise RuntimeError(f"paper node coordinates incomplete: {sorted(node_coords)}")
    center = tuple(sum(node_coords[node][axis] for node in PAPER_NODES) / len(PAPER_NODES) for axis in ("x", "y", "z"))
    target_rows = {}
    for target, filename in TARGET_FILES.items():
        gps = parse_gps(args.gps_root / filename)
        selected = []
        for time_s in times:
            second = round(time_s)
            nearest = min(gps, key=lambda row: abs(hms_seconds(row["time_code"]) - second))
            selected.append({"time_s": time_s, "px": nearest["x"] - center[0], "py": nearest["y"] - center[1], "pz": nearest["z"] - center[2], "source_time_code": nearest["time_code"]})
        target_rows[target] = selected
    args.output_root.mkdir(parents=True, exist_ok=True)
    for target, rows in target_rows.items():
        frontend = args.output_root / f"target{target}" / "frontend"
        frontend.mkdir(parents=True, exist_ok=True)
        with (frontend / "gps_truth.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    manifest = {
        "claim_status": "paper_aligned_truth_bundle",
        "paper_reference": "Zhang et al., IEEE IoT Journal 9(6), DOI 10.1109/JIOT.2021.3108528",
        "public_code": "https://github.com/zhangwenqiong2017/MTT_WB_DBN",
        "remote_segment": REMOTE_SEGMENT,
        "remote_gps_root": REMOTE_GPS,
        "local_gps_root": str(args.gps_root),
        "paper_nodes": list(PAPER_NODES),
        "excluded_local_node": 2,
        "target_mapping": {"target1": "GPS1_plane1.gps", "target2": "GPS3_plane2.gps", "target3": "GPS4_plane2to3.gps"},
        "frame_source": str(args.frame_csv),
        "frame_count": len(times),
        "time_start_s": times[0], "time_end_s": times[-1],
        "center_xyz": list(center),
        "paper_initial_time_hhmmss": 132754,
        "gps_hashes": {name: sha256(args.gps_root / name) for name in TARGET_FILES.values()},
        "gps_role": "evaluation truth and paper-protocol initialization audit; not an acoustic observation",
    }
    (args.output_root / "paper_field_protocol_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
