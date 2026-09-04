#!/usr/bin/env python3
"""Aggregate the admitted Baoding dual-source DOAs to one-second frames.

The associated DOAs remain the only runtime observations. GPS is copied from
the frozen full-rate bundle solely as an offline truth stream. No GPS position
is used in DOA aggregation, initialization, triangulation or filtering.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


TARGETS = (1, 2)
NODES = (1, 2, 3, 5, 6, 7, 8, 11, 13)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def circular_mean_deg(values: list[float]) -> tuple[float, float]:
    radians = [math.radians(value) for value in values]
    cosine = statistics.mean(math.cos(value) for value in radians)
    sine = statistics.mean(math.sin(value) for value in radians)
    angle = math.degrees(math.atan2(sine, cosine)) % 360.0
    resultant = math.hypot(cosine, sine)
    return angle, resultant


def aggregate_truth(path: Path) -> list[dict[str, float]]:
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in read_csv(path):
        grouped.setdefault(int(math.floor(float(row["time_s"]))), []).append(row)
    output = []
    for second, rows in sorted(grouped.items()):
        output.append({
            "time_s": float(second),
            "px": statistics.median(float(row["px"]) for row in rows),
            "py": statistics.median(float(row["py"]) for row in rows),
            "pz": statistics.median(float(row["pz"]) for row in rows),
        })
    return output


def aggregate_target(
    target: int,
    full_root: Path,
    association: Path,
    output: Path,
    calibration_end_s: int,
) -> None:
    source_frontend = full_root / f"target{target}" / "frontend"
    source_manifest = json.loads((source_frontend / "frontend_manifest.json").read_text(encoding="utf-8"))
    grouped: dict[tuple[int, int], list[dict[str, str]]] = {}
    source_files = []
    for node in NODES:
        path = association / f"associated_global_node_{node}.csv"
        source_files.append(path)
        for row in read_csv(path):
            second = int(math.floor(float(row["time_s"])))
            grouped.setdefault((second, node), []).append(row)

    observations: list[dict[str, object]] = []
    seconds = sorted({second for second, _ in grouped})
    for source_index, second in enumerate(seconds):
        segment = "dual_calibration" if second < calibration_end_s else "dual_evaluation"
        for node in NODES:
            rows = grouped.get((second, node), [])
            if not rows:
                continue
            azimuths = [float(row[f"target{target}_az_deg"]) for row in rows]
            elevations = [90.0 - float(row[f"target{target}_zenith_deg"]) for row in rows]
            azimuth, resultant = circular_mean_deg(azimuths)
            elevation = statistics.median(elevations)
            elevation_mad = statistics.median(abs(value - elevation) for value in elevations)
            concentration = max(0.05, resultant * math.exp(-elevation_mad / 25.0))
            observations.append({
                "segment": segment,
                "time_s": float(second),
                "node_id": node,
                "azimuth_deg": azimuth,
                "elevation_deg": elevation,
                "concentration": concentration,
                "valid": True,
                "source_frame_index": source_index,
                "source_subframes": len(rows),
            })

    truth = aggregate_truth(source_frontend / "gps_truth.csv")
    write_csv(output / "observations.csv", observations)
    write_csv(output / "gps_truth.csv", truth)
    calibration_frames = len({int(row["time_s"]) for row in observations if row["segment"] == "dual_calibration"})
    evaluation_frames = len({int(row["time_s"]) for row in observations if row["segment"] == "dual_evaluation"})
    manifest = {
        "task": "Baoding shuangyuan_4 one-second associated-DOA frontend",
        "target": target,
        "nodes": source_manifest["nodes"],
        "center_xyz": source_manifest["center_xyz"],
        "coordinate_system": "node-centred local ENU metres for GPS truth; global node coordinates retained for downstream centring",
        "aggregation": {
            "source_update_interval_s": 0.2098360655737705,
            "output_update_interval_s": 1.0,
            "azimuth": "circular mean within floor(time_s) bins",
            "elevation": "median within floor(time_s) bins",
            "concentration": "azimuth resultant length with elevation-MAD attenuation",
            "calibration_end_s_exclusive": calibration_end_s,
            "calibration_frames": calibration_frames,
            "evaluation_frames": evaluation_frames,
        },
        "gps_role": "offline evaluation only; never used in aggregation or downstream runtime updates",
        "sources": {
            "full_rate_frontend": str(source_frontend),
            "full_rate_frontend_manifest_sha256": sha256(source_frontend / "frontend_manifest.json"),
            "associated_doa_files": [str(path) for path in source_files],
            "associated_doa_sha256": {path.name: sha256(path) for path in source_files},
        },
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha256(Path(__file__).resolve()),
    }
    write_json(output / "frontend_manifest.json", manifest)
    write_json(output / "frontend_calibration.json", {
        "target": target,
        "protocol": "inherits frozen global dual-target association; one-second aggregation adds no GPS correction",
        "gps_used_for_runtime": False,
        "source_association_gate": str(association / "shuangyuan4_global_association_gate.json"),
        "source_association_gate_sha256": sha256(association / "shuangyuan4_global_association_gate.json"),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--association", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-end-s", type=int, default=46600)
    args = parser.parse_args()
    for target in TARGETS:
        aggregate_target(
            target,
            args.full_root,
            args.association,
            args.output / f"target{target}" / "source_frontend",
            args.calibration_end_s,
        )
    print(json.dumps({
        "targets": list(TARGETS),
        "output": str(args.output),
        "calibration_end_s_exclusive": args.calibration_end_s,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
