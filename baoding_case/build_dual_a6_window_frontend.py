#!/usr/bin/env python3
"""Build a fixed-window Cartesian PCE/APCE input from frozen A6 states."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a6-root", type=Path, required=True)
    parser.add_argument("--a6-manifest", type=Path, required=True)
    parser.add_argument("--truth-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-s", type=int, default=46673)
    parser.add_argument("--end-s", type=int, default=46697)
    args = parser.parse_args()

    expected_times = list(range(args.start_s, args.end_s + 1))
    for target in (1, 2):
        state_path = args.a6_root / "a6" / f"target{target}_state.csv"
        by_time = {int(round(float(row["time_s"]))): row for row in read_csv(state_path)}
        missing = [second for second in expected_times if second not in by_time]
        if missing:
            raise RuntimeError(f"target {target}: missing A6 seconds {missing}")
        observations: list[dict[str, object]] = []
        min_eigenvalue = float("inf")
        for source_index, second in enumerate(expected_times):
            source = by_time[second]
            covariance = np.asarray([
                [float(source[f"cov_{row}{column}"]) for column in range(3)]
                for row in range(3)
            ], dtype=float)
            covariance = 0.5 * (covariance + covariance.T)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            eigenvalues = np.maximum(eigenvalues, 1e-6)
            covariance = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            min_eigenvalue = min(min_eigenvalue, float(np.min(eigenvalues)))
            row: dict[str, object] = {
                "segment": "dual_evaluation",
                "time_s": float(second),
                "source_frame_index": source_index,
                "available_nodes": int(source["available_nodes"]),
                "inlier_nodes": int(source["inlier_nodes"]),
                "inlier_node_ids": source["inlier_node_ids"],
                "condition_number": float(source["condition_number"]),
                "reprojection_rms_deg": float(source["reprojection_rms_deg"]),
                "association_cost": float(source["objective"]),
                "valid": str(source["valid"]).lower() == "true",
                "y_E": float(source["px"]),
                "y_N": float(source["py"]),
                "y_U": float(source["pz"]),
            }
            for i in range(3):
                for j in range(3):
                    row[f"R_{i}{j}"] = float(covariance[i, j])
            observations.append(row)

        frontend = args.output / f"target{target}" / "frontend"
        frontend.mkdir(parents=True, exist_ok=True)
        observation_path = frontend / "observations_cartesian.csv"
        write_csv(observation_path, observations)
        truth_source = args.truth_root / f"target{target}" / "frontend" / "gps_truth.csv"
        truth_rows = [
            row for row in read_csv(truth_source)
            if args.start_s <= int(round(float(row["time_s"]))) <= args.end_s
        ]
        if len(truth_rows) != len(expected_times):
            raise RuntimeError(f"target {target}: expected {len(expected_times)} truth rows, found {len(truth_rows)}")
        write_csv(frontend / "gps_truth.csv", truth_rows)
        calibration_source = args.truth_root / f"target{target}" / "frontend" / "frontend_calibration.json"
        if calibration_source.exists():
            shutil.copy2(calibration_source, frontend / "frontend_calibration.json")
        manifest = {
            "task": "Baoding shuangyuan_4 A6 acoustic Cartesian fixed-window PCE/APCE bundle",
            "target": target,
            "coordinate_system": "node-centred local ENU metres; A6 px/py/pz retained",
            "window": {"start_s": args.start_s, "end_s": args.end_s, "frames": len(expected_times)},
            "segment": "dual_evaluation",
            "source_update_interval_s": 1.0,
            "aggregation": "direct frozen A6 one-second state output; no smoothing or GPS correction",
            "observation_covariance": "PSD-projected A6 state-covariance position block cov_00:cov_22",
            "finite_psd_covariance": min_eigenvalue > 0.0,
            "minimum_covariance_eigenvalue_m2": min_eigenvalue,
            "gps_role": "offline truth scoring only; no GPS in initialization, propagation, update or branch evidence",
            "source_a6_state": str(state_path),
            "source_a6_state_sha256": sha256(state_path),
            "source_a6_manifest": str(args.a6_manifest),
            "source_a6_manifest_sha256": sha256(args.a6_manifest),
            "gps_truth_source": str(truth_source),
            "gps_truth_source_sha256": sha256(truth_source),
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": sha256(Path(__file__).resolve()),
        }
        (frontend / "frontend_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps({"output": str(args.output), "window": [args.start_s, args.end_s]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
