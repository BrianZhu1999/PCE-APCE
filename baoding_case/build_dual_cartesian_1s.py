#!/usr/bin/env python3
"""Build one-second dual-source Cartesian observations from acoustic rays.

The source is the admitted global-association triangulation, itself derived
from the real multi-node DOAs. Subframes are aggregated before PCE/APCE using
a robust median. GPS is copied only for offline scoring and never contributes
to the Cartesian observation or its covariance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np


TARGETS = (1, 2)
POS_NAMES = ("y_E", "y_N", "y_U")
R_NAMES = tuple(f"R_{row}{column}" for row in range(3) for column in range(3))


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


def aggregate_truth(path: Path) -> list[dict[str, float]]:
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in read_csv(path):
        grouped.setdefault(int(math.floor(float(row["time_s"]))), []).append(row)
    return [
        {
            "time_s": float(second),
            "px": statistics.median(float(row["px"]) for row in rows),
            "py": statistics.median(float(row["py"]) for row in rows),
            "pz": statistics.median(float(row["pz"]) for row in rows),
        }
        for second, rows in sorted(grouped.items())
    ]


def robust_covariance(
    points: np.ndarray,
    median_condition: float,
    median_inliers: float,
) -> np.ndarray:
    centre = np.median(points, axis=0)
    residual = points - centre
    radii = np.linalg.norm(residual, axis=1)
    if len(points) >= 3 and np.any(radii > 0.0):
        clip = max(float(np.percentile(radii, 80.0)), 1.0)
        scale = np.minimum(1.0, clip / np.maximum(radii, 1e-9))
        clipped = residual * scale[:, None]
        covariance = clipped.T @ clipped / max(len(clipped) - 1, 1)
    else:
        covariance = np.zeros((3, 3), dtype=float)
    geometry_std = float(np.clip(
        (8.0 + 2.0 * median_condition) * math.sqrt(6.0 / max(median_inliers, 1.0)),
        15.0,
        60.0,
    ))
    covariance = 0.5 * (covariance + covariance.T)
    values, vectors = np.linalg.eigh(covariance)
    values = np.maximum(values, geometry_std**2)
    return vectors @ np.diag(values) @ vectors.T


def offline_metrics(
    observations: list[dict[str, object]],
    truth_rows: list[dict[str, float]],
) -> dict[str, float | int]:
    truth = {
        int(row["time_s"]): np.asarray([row["px"], row["py"], row["pz"]], dtype=float)
        for row in truth_rows
    }
    errors = []
    for row in observations:
        second = int(float(row["time_s"]))
        if second not in truth:
            continue
        position = np.asarray([float(row[name]) for name in POS_NAMES])
        errors.append(float(np.linalg.norm(position - truth[second])))
    array = np.asarray(errors)
    return {
        "evaluated_frames": len(errors),
        "rmse_position_error_m": float(np.sqrt(np.mean(array**2))),
        "median_position_error_m": float(np.median(array)),
        "p90_position_error_m": float(np.percentile(array, 90.0)),
    }


def build_target(
    target: int,
    association: Path,
    full_root: Path,
    output: Path,
    calibration_end_s: int,
) -> None:
    triangulation_path = association / f"target{target}_triangulation_global.csv"
    source_frontend = full_root / f"target{target}" / "frontend"
    source_manifest = json.loads((source_frontend / "frontend_manifest.json").read_text(encoding="utf-8"))
    centre = np.asarray(source_manifest["center_xyz"], dtype=float)
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in read_csv(triangulation_path):
        if row.get("valid", "False").lower() != "true":
            continue
        if not all(row.get(name) for name in ("px", "py", "pz", "condition_number", "inlier_nodes")):
            continue
        grouped.setdefault(int(math.floor(float(row["time_s"]))), []).append(row)

    observations: list[dict[str, object]] = []
    for source_index, (second, rows) in enumerate(sorted(grouped.items())):
        points_global = np.asarray(
            [[float(row[name]) for name in ("px", "py", "pz")] for row in rows],
            dtype=float,
        )
        point_local = np.median(points_global, axis=0) - centre
        median_condition = statistics.median(float(row["condition_number"]) for row in rows)
        median_inliers = statistics.median(int(row["inlier_nodes"]) for row in rows)
        covariance = robust_covariance(points_global, median_condition, median_inliers)
        segment = "dual_calibration" if second < calibration_end_s else "dual_evaluation"
        output_row: dict[str, object] = {
            "segment": segment,
            "time_s": float(second),
            "source_frame_index": source_index,
            "available_nodes": int(round(statistics.median(int(row["available_nodes"]) for row in rows))),
            "inlier_nodes": int(round(median_inliers)),
            "inlier_node_ids": "",
            "condition_number": median_condition,
            "reprojection_rms_deg": "",
            "association_cost": "",
            "valid": True,
        }
        output_row.update({name: float(point_local[index]) for index, name in enumerate(POS_NAMES)})
        output_row.update({
            name: float(covariance[row_index, column_index])
            for row_index in range(3)
            for column_index, name in enumerate(R_NAMES[row_index * 3:(row_index + 1) * 3])
        })
        observations.append(output_row)

    truth_rows = aggregate_truth(source_frontend / "gps_truth.csv")
    write_csv(output / "observations_cartesian.csv", observations)
    write_csv(output / "gps_truth.csv", truth_rows)
    eigenvalues = []
    for row in observations:
        matrix = np.asarray(
            [[float(row[f"R_{i}{j}"]) for j in range(3)] for i in range(3)],
            dtype=float,
        )
        eigenvalues.extend(np.linalg.eigvalsh(matrix).tolist())
    calibration_frames = sum(row["segment"] == "dual_calibration" for row in observations)
    evaluation_frames = sum(row["segment"] == "dual_evaluation" for row in observations)
    manifest = {
        "task": "Baoding shuangyuan_4 one-second Cartesian acoustic frontend",
        "target": target,
        "coordinate_system": "node-centred local ENU metres",
        "source_update_interval_s": 0.2098360655737705,
        "output_update_interval_s": 1.0,
        "aggregation": "componentwise median of valid global-association triangulations within floor(time_s) bins",
        "covariance_model": "robust within-second Cartesian scatter plus condition/inlier-dependent PSD eigenvalue floor",
        "covariance_scale": 1.0,
        "calibration_end_s_exclusive": calibration_end_s,
        "calibration_frames": calibration_frames,
        "evaluation_frames": evaluation_frames,
        "finite_psd_covariance": bool(eigenvalues) and min(eigenvalues) > 0.0,
        "minimum_covariance_eigenvalue_m2": min(eigenvalues),
        "offline_gps_evaluation": offline_metrics(observations, truth_rows),
        "gps_role": "copied for offline evaluation only; never used for aggregation, covariance or tracking",
        "sources": {
            "triangulation": str(triangulation_path),
            "triangulation_sha256": sha256(triangulation_path),
            "full_rate_frontend": str(source_frontend),
            "full_rate_manifest_sha256": sha256(source_frontend / "frontend_manifest.json"),
            "association_gate": str(association / "shuangyuan4_global_association_gate.json"),
            "association_gate_sha256": sha256(association / "shuangyuan4_global_association_gate.json"),
        },
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha256(Path(__file__).resolve()),
    }
    write_json(output / "frontend_manifest.json", manifest)
    write_json(output / "cartesian_frontend_gate.json", {
        "target": target,
        "minimum_evaluation_frames": 120,
        "finite_psd_covariance_required": True,
        "evaluation_frames": evaluation_frames,
        "finite_psd_covariance": manifest["finite_psd_covariance"],
        "admitted": evaluation_frames >= 120 and manifest["finite_psd_covariance"],
        "gps_error_is_not_an_admission_gate": True,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--association", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-end-s", type=int, default=46600)
    args = parser.parse_args()
    for target in TARGETS:
        build_target(
            target,
            args.association,
            args.full_root,
            args.output / f"target{target}" / "frontend",
            args.calibration_end_s,
        )
    print(json.dumps({"targets": list(TARGETS), "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
