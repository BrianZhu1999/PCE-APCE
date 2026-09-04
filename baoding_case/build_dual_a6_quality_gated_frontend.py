#!/usr/bin/env python3
"""Build a continuous A6 full-circle bundle with acoustic-only update gating."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a6-root", type=Path, required=True)
    parser.add_argument("--a6-manifest", type=Path, required=True)
    parser.add_argument("--truth-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-s", type=int, default=46613)
    parser.add_argument("--end-s", type=int, default=46699)
    parser.add_argument("--calibration-start-s", type=int, default=46540)
    parser.add_argument("--calibration-end-s", type=int, default=46560)
    parser.add_argument("--minimum-inlier-nodes", type=int, default=2)
    parser.add_argument("--maximum-reprojection-deg", type=float, default=25.0)
    args = parser.parse_args()

    expected_times = list(range(args.start_s, args.end_s + 1))
    for target in (1, 2):
        state_path = args.a6_root / "a6" / f"target{target}_state.csv"
        all_rows = read_csv(state_path)
        by_time = {int(round(float(row["time_s"]))): row for row in all_rows}
        missing = [second for second in expected_times if second not in by_time]
        if missing:
            raise RuntimeError(f"target {target}: missing A6 seconds {missing}")
        calibration_rows = [
            row for row in all_rows
            if args.calibration_start_s <= int(round(float(row["time_s"]))) <= args.calibration_end_s
        ]
        if not calibration_rows:
            raise RuntimeError(f"target {target}: empty calibration interval")
        calibration_reprojection_p90 = float(np.percentile(
            [float(row["reprojection_rms_deg"]) for row in calibration_rows], 90.0
        ))
        calibration_inlier_median = float(np.median(
            [int(row["inlier_nodes"]) for row in calibration_rows]
        ))

        observations: list[dict[str, object]] = []
        gate_counts = {"update": 0, "downweight_low_inliers": 0, "downweight_high_reprojection": 0, "skip_source_invalid": 0}
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
            source_valid = str(source["valid"]).lower() == "true"
            inliers = int(source["inlier_nodes"])
            reprojection = float(source["reprojection_rms_deg"])
            observation_update = source_valid
            if not source_valid:
                gate_reason = "skip_source_invalid"
            elif inliers < args.minimum_inlier_nodes:
                gate_reason = "downweight_low_inliers"
            elif reprojection > args.maximum_reprojection_deg:
                gate_reason = "downweight_high_reprojection"
            else:
                gate_reason = "update"
            # Preserve the frozen A6 covariance on geometrically admissible
            # frames.  Only violations of the declared two-ray/25-degree
            # physical gate reduce confidence, avoiding evaluation-wide
            # rescaling from held-out data.
            inlier_scale = (args.minimum_inlier_nodes / max(float(inliers), 0.5)) ** 2
            reprojection_scale = (reprojection / args.maximum_reprojection_deg) ** 2
            quality_covariance_multiplier = float(min(10000.0, max(1.0, inlier_scale, reprojection_scale)))
            covariance *= quality_covariance_multiplier
            gate_counts[gate_reason] += 1
            row: dict[str, object] = {
                "segment": "dual_evaluation",
                "time_s": float(second),
                "source_frame_index": source_index,
                "available_nodes": int(source["available_nodes"]),
                "inlier_nodes": inliers,
                "inlier_node_ids": source["inlier_node_ids"],
                "condition_number": float(source["condition_number"]),
                "reprojection_rms_deg": reprojection,
                "association_cost": float(source["objective"]),
                "valid": True,
                "observation_update": observation_update,
                "observation_gate_reason": gate_reason,
                "quality_covariance_multiplier": quality_covariance_multiplier,
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
            "task": "Baoding shuangyuan_4 A6 acoustic Cartesian continuous full-circle bundle with quality-gated updates",
            "target": target,
            "coordinate_system": "node-centred local ENU metres; A6 px/py/pz retained",
            "window": {"start_s": args.start_s, "end_s": args.end_s, "frames": len(expected_times)},
            "segment": "dual_evaluation",
            "source_update_interval_s": 1.0,
            "timeline_policy": "all 87 seconds retained; failed acoustic observations trigger prediction-only steps",
            "quality_gate": {
                "inputs": "A6 acoustic geometry fields only",
                "minimum_inlier_nodes": args.minimum_inlier_nodes,
                "maximum_reprojection_rms_deg": args.maximum_reprojection_deg,
                "calibration_interval_s": [args.calibration_start_s, args.calibration_end_s],
                "calibration_reprojection_p90_deg": calibration_reprojection_p90,
                "calibration_inlier_median": calibration_inlier_median,
                "covariance_scaling_reference": "only physical-gate violations; admissible A6 covariance is unchanged",
                "counts": gate_counts,
                "gps_used_for_gate": False,
            },
            "observation_covariance": "PSD-projected A6 state-covariance position block scaled by calibration-frozen acoustic geometry quality",
            "finite_psd_covariance": min_eigenvalue > 0.0,
            "minimum_covariance_eigenvalue_m2": min_eigenvalue,
            "gps_role": "offline truth scoring only; no GPS in initialization, propagation, update, gate or branch evidence",
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
