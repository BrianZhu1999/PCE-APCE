#!/usr/bin/env python3
"""Package the stable paper-inspired DBN track as a PCE/APCE benchmark.

The input is a DBN-style Cartesian trajectory produced by the approximate
Baoding adapter. The package records finite-difference state estimates and an
empirical diagonal covariance derived from track innovations. GPS remains
evaluation provenance; it is never used to alter the DBN track.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


TARGET_Z = {1: 222.2, 2: 227.017, 3: 250.2}
GPS_FILES = {1: "GPS1_plane1.gps", 2: "GPS3_plane2.gps", 3: "GPS4_plane2to3.gps"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hms_seconds(value: int | float | str) -> float:
    text = str(value).replace(":", "").strip()
    integer = int(float(text))
    text = str(integer).zfill(6)
    return int(text[:2]) * 3600 + int(text[2:4]) * 60 + int(text[4:])


def read_target(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"empty DBN target CSV: {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start-hhmmss", type=float, default=132754)
    parser.add_argument("--dt-s", type=float, default=640.0 / 3050.0)
    parser.add_argument("--source-label", default="paper-inspired DBN-LA-NM Baoding adapter")
    args = parser.parse_args()

    targets = {target: read_target(args.input_root / f"target{target}_dbn_lanm_baseline.csv") for target in (1, 2, 3)}
    frame_count = min(len(rows) for rows in targets.values())
    args.output_root.mkdir(parents=True, exist_ok=True)

    target_manifests = {}
    for target, rows in targets.items():
        rows = rows[:frame_count]
        positions = [(float(row["estimated_x"]), float(row["estimated_y"])) for row in rows]
        times = [hms_seconds(args.start_hhmmss) + int(row["frame_index"]) * args.dt_s for row in rows]
        velocities = []
        for index, (x, y) in enumerate(positions):
            if index == 0:
                vx = (positions[1][0] - x) / args.dt_s if len(positions) > 1 else 0.0
                vy = (positions[1][1] - y) / args.dt_s if len(positions) > 1 else 0.0
            else:
                vx = (x - positions[index - 1][0]) / args.dt_s
                vy = (y - positions[index - 1][1]) / args.dt_s
            velocities.append((vx, vy))
        accelerations = []
        for index in range(1, len(velocities)):
            accelerations.append(((velocities[index][0] - velocities[index - 1][0]) / args.dt_s, (velocities[index][1] - velocities[index - 1][1]) / args.dt_s))
        accel_scale = max(1.0, statistics.median([math.hypot(ax, ay) for ax, ay in accelerations]) if accelerations else 1.0)
        position_sigma = max(5.0, min(100.0, 0.5 * args.dt_s * accel_scale + 5.0))
        velocity_sigma = max(2.0, min(50.0, accel_scale + 2.0))

        target_dir = args.output_root / f"target{target}"
        frontend = target_dir / "frontend"
        runs = target_dir / "runs"
        frontend.mkdir(parents=True, exist_ok=True)
        runs.mkdir(parents=True, exist_ok=True)
        track_path = frontend / "dbn_track.csv"
        fields = ["frame_index", "time_s", "time_hhmmss", "px", "py", "pz", "vx", "vy", "vz", "cov_px", "cov_py", "cov_pz", "cov_vx", "cov_vy", "cov_vz", "position_error_m"]
        with track_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for index, (row, (x, y), (vx, vy), time_s) in enumerate(zip(rows, positions, velocities, times)):
                writer.writerow({"frame_index": int(row["frame_index"]), "time_s": time_s, "time_hhmmss": int(args.start_hhmmss), "px": x, "py": y, "pz": TARGET_Z[target], "vx": vx, "vy": vy, "vz": 0.0, "cov_px": position_sigma**2, "cov_py": position_sigma**2, "cov_pz": 25.0**2, "cov_vx": velocity_sigma**2, "cov_vy": velocity_sigma**2, "cov_vz": 10.0**2, "position_error_m": row.get("position_error_m", "")})

        manifest = {"claim_status": "paper_inspired_dbn_benchmark_target", "target": target, "source_track": str(args.input_root / f"target{target}_dbn_lanm_baseline.csv"), "source_track_sha256": sha256(args.input_root / f"target{target}_dbn_lanm_baseline.csv"), "frame_count": frame_count, "start_hhmmss": args.start_hhmmss, "dt_s": args.dt_s, "target_height_m": TARGET_Z[target], "gps_evaluation_file": GPS_FILES[target], "state_definition": "[px, py, pz, vx, vy, vz]", "covariance_definition": "diagonal empirical innovation scale from finite-difference track acceleration; not posterior covariance from the private tracker", "position_sigma_m": position_sigma, "velocity_sigma_mps": velocity_sigma, "runtime_gps_correction": False, "source_label": args.source_label, "track_csv": str(track_path)}
        (frontend / "dbn_track_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        target_manifests[str(target)] = manifest

    root_manifest = {"claim_status": "paper_inspired_dbn_pce_apce_benchmark", "source_root": str(args.input_root), "source_root_sha256": {f"target{target}": sha256(args.input_root / f"target{target}_dbn_lanm_baseline.csv") for target in (1, 2, 3)}, "frame_count": frame_count, "start_hhmmss": args.start_hhmmss, "dt_s": args.dt_s, "targets": target_manifests, "pce_apce_role": "upstream scenario benchmark; observations are generated deterministically from DBN Cartesian tracks", "independent_acoustic_frontend": False, "formal_paper_reproduction": False, "warning": "This is an approximate paper-inspired DBN baseline for PCE/APCE testing, not the authors' private field implementation."}
    (args.output_root / "benchmark_manifest.json").write_text(json.dumps(root_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(root_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
