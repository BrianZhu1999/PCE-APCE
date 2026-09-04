#!/usr/bin/env python3
"""Build target-specific A6 Cartesian frontends with frozen reliability calibration."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np


PROFILES = ("baseline", "calibration_floor", "acoustic_reliability", "combined")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # The frozen PCE/APCE runner reads strict UTF-8 headers, so generated
    # frontend CSVs must not add a BOM to the first field name.
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


def psd(matrix: np.ndarray, floor: float = 1.0e-6) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    return vectors @ np.diag(np.maximum(values, floor)) @ vectors.T


def parse_nodes(value: str) -> list[int]:
    return [int(token) for token in value.split(";") if token.strip()]


def effective_sigma(node_ids: list[int], sigma_by_node: dict[int, float]) -> float:
    values = [max(float(sigma_by_node[node]), 1.0e-6) for node in node_ids if node in sigma_by_node]
    if not values:
        return 45.0
    return math.sqrt(len(values) / sum(1.0 / (value * value) for value in values))


def truth_map(path: Path) -> tuple[list[dict[str, str]], dict[int, np.ndarray]]:
    rows = read_csv(path)
    mapping = {
        int(round(float(row["time_s"]))): np.asarray(
            [float(row[key]) for key in ("px", "py", "pz")], dtype=float
        )
        for row in rows
    }
    return rows, mapping


def calibration_model(
    rows: list[dict[str, str]],
    truth: dict[int, np.ndarray],
    sigma_by_node: dict[int, float],
    start_s: int,
    end_s: int,
) -> dict[str, object]:
    selected = [
        row for row in rows
        if start_s <= int(round(float(row["time_s"]))) <= end_s
    ]
    if not selected:
        raise RuntimeError("empty reliability calibration interval")
    errors: list[np.ndarray] = []
    effective_sigmas: list[float] = []
    for row in selected:
        second = int(round(float(row["time_s"])))
        if second not in truth:
            raise RuntimeError(f"calibration truth missing at {second}")
        position = np.asarray([float(row[key]) for key in ("px", "py", "pz")])
        errors.append(position - truth[second])
        effective_sigmas.append(effective_sigma(parse_nodes(row["inlier_node_ids"]), sigma_by_node))
    error_matrix = np.asarray(errors)
    second_moment = error_matrix.T @ error_matrix / len(error_matrix)
    # Shrink the small-sample off-diagonal terms while retaining systematic
    # calibration bias in the diagonal second moment.
    covariance_floor = psd(0.5 * second_moment + 0.5 * np.diag(np.diag(second_moment)))
    condition = np.asarray([float(row["condition_number"]) for row in selected])
    reprojection = np.asarray([float(row["reprojection_rms_deg"]) for row in selected])
    return {
        "frames": len(selected),
        "condition_reference_p75": float(np.percentile(condition, 75.0)),
        "reprojection_reference_p75_deg": float(np.percentile(reprojection, 75.0)),
        "effective_node_sigma_reference_median_deg": float(np.median(effective_sigmas)),
        "position_error_rmse_axis_m": np.sqrt(np.mean(np.square(error_matrix), axis=0)).tolist(),
        "position_error_second_moment_m2": second_moment.tolist(),
        "covariance_floor_m2": covariance_floor.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a6-root", type=Path, required=True)
    parser.add_argument("--a6-manifest", type=Path, required=True)
    parser.add_argument("--truth-root", type=Path, required=True)
    parser.add_argument("--reliability-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--targets", type=int, nargs="+", choices=(1, 2), default=(1, 2))
    parser.add_argument("--start-s", type=int, required=True)
    parser.add_argument("--end-s", type=int, required=True)
    parser.add_argument("--calibration-start-s", type=int, default=46540)
    parser.add_argument("--calibration-end-s", type=int, default=46560)
    parser.add_argument("--minimum-inlier-nodes", type=int, default=2)
    parser.add_argument("--maximum-reprojection-deg", type=float, default=25.0)
    parser.add_argument("--maximum-dynamic-scale", type=float, default=25.0)
    args = parser.parse_args()

    expected_times = list(range(args.start_s, args.end_s + 1))
    reliability_rows = read_csv(args.reliability_csv)
    a6_manifest = json.loads(args.a6_manifest.read_text(encoding="utf-8"))
    model_sigma = a6_manifest["quality_model"]["node_target_sigma_deg"]

    for target in args.targets:
        state_path = args.a6_root / "a6" / f"target{target}_state.csv"
        all_rows = read_csv(state_path)
        by_time = {int(round(float(row["time_s"]))): row for row in all_rows}
        missing = [second for second in expected_times if second not in by_time]
        if missing:
            raise RuntimeError(f"target {target}: missing A6 seconds {missing}")

        truth_source = args.truth_root / f"target{target}" / "frontend" / "gps_truth.csv"
        truth_rows, truth = truth_map(truth_source)
        sigma_by_node = {
            int(row["node"]): float(model_sigma[f"node{row['node']}_target{target}"])
            for row in reliability_rows if int(row["target"]) == target
        }
        calibration = calibration_model(
            all_rows, truth, sigma_by_node,
            args.calibration_start_s, args.calibration_end_s,
        )
        covariance_floor = np.asarray(calibration["covariance_floor_m2"], dtype=float)
        condition_reference = max(float(calibration["condition_reference_p75"]), 1.0)
        reprojection_reference = max(float(calibration["reprojection_reference_p75_deg"]), 1.0)
        node_reference = max(float(calibration["effective_node_sigma_reference_median_deg"]), 1.0)

        observations: list[dict[str, object]] = []
        counts = {
            "updates": 0,
            "prediction_only": 0,
            "physical_downweighted": 0,
            "dynamic_downweighted": 0,
            "calibration_floor_applied": 0,
        }
        minimum_eigenvalue = float("inf")
        for source_index, second in enumerate(expected_times):
            source = by_time[second]
            base_covariance = psd(np.asarray([
                [float(source[f"cov_{row}{column}"]) for column in range(3)]
                for row in range(3)
            ], dtype=float))
            source_valid = str(source["valid"]).lower() == "true"
            inliers = int(source["inlier_nodes"])
            reprojection = float(source["reprojection_rms_deg"])
            condition = float(source["condition_number"])
            nodes = parse_nodes(source["inlier_node_ids"])
            node_sigma = effective_sigma(nodes, sigma_by_node)

            inlier_scale = (args.minimum_inlier_nodes / max(float(inliers), 0.5)) ** 2
            physical_reprojection_scale = (reprojection / args.maximum_reprojection_deg) ** 2
            physical_scale = float(max(1.0, inlier_scale, physical_reprojection_scale))
            condition_scale = (condition / condition_reference) ** 2
            reprojection_scale = (reprojection / reprojection_reference) ** 2
            node_scale = (node_sigma / node_reference) ** 2
            dynamic_scale = float(min(
                args.maximum_dynamic_scale,
                max(1.0, condition_scale, reprojection_scale, node_scale),
            ))

            if args.profile == "baseline":
                covariance = base_covariance * physical_scale
                applied_dynamic_scale = 1.0
                floor_applied = False
            elif args.profile == "calibration_floor":
                covariance = base_covariance * physical_scale + covariance_floor
                applied_dynamic_scale = 1.0
                floor_applied = True
            elif args.profile == "acoustic_reliability":
                covariance = base_covariance * max(physical_scale, dynamic_scale)
                applied_dynamic_scale = dynamic_scale
                floor_applied = False
            else:
                covariance = (base_covariance + covariance_floor) * max(physical_scale, dynamic_scale)
                applied_dynamic_scale = dynamic_scale
                floor_applied = True
            covariance = psd(covariance)
            minimum_eigenvalue = min(minimum_eigenvalue, float(np.min(np.linalg.eigvalsh(covariance))))
            observation_update = bool(source_valid)
            counts["updates" if observation_update else "prediction_only"] += 1
            counts["physical_downweighted"] += int(physical_scale > 1.0 + 1.0e-9)
            counts["dynamic_downweighted"] += int(applied_dynamic_scale > 1.0 + 1.0e-9)
            counts["calibration_floor_applied"] += int(floor_applied)

            row: dict[str, object] = {
                "segment": "dual_evaluation",
                "time_s": float(second),
                "source_frame_index": source_index,
                "available_nodes": int(source["available_nodes"]),
                "inlier_nodes": inliers,
                "inlier_node_ids": source["inlier_node_ids"],
                "condition_number": condition,
                "reprojection_rms_deg": reprojection,
                "association_cost": float(source["objective"]),
                "valid": True,
                "observation_update": observation_update,
                "observation_gate_reason": "update" if observation_update else "skip_source_invalid",
                "quality_covariance_multiplier": max(physical_scale, applied_dynamic_scale),
                "physical_covariance_multiplier": physical_scale,
                "dynamic_covariance_multiplier": applied_dynamic_scale,
                "condition_covariance_multiplier": float(max(1.0, condition_scale)),
                "reprojection_covariance_multiplier": float(max(1.0, reprojection_scale)),
                "node_reliability_covariance_multiplier": float(max(1.0, node_scale)),
                "effective_inlier_node_sigma_deg": node_sigma,
                "calibration_floor_trace_m2": float(np.trace(covariance_floor) if floor_applied else 0.0),
                "reliability_profile": args.profile,
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
        selected_truth = [
            row for row in truth_rows
            if args.start_s <= int(round(float(row["time_s"]))) <= args.end_s
        ]
        if len(selected_truth) != len(expected_times):
            raise RuntimeError(
                f"target {target}: expected {len(expected_times)} truth rows, found {len(selected_truth)}"
            )
        write_csv(frontend / "gps_truth.csv", selected_truth)
        calibration_source = args.truth_root / f"target{target}" / "frontend" / "frontend_calibration.json"
        if calibration_source.exists():
            shutil.copy2(calibration_source, frontend / "frontend_calibration.json")

        manifest = {
            "task": "Baoding dual-source A6 target-specific reliability-calibrated Cartesian frontend",
            "target": target,
            "profile": args.profile,
            "coordinate_system": "node-centred local ENU metres; A6 px/py/pz retained",
            "window": {"start_s": args.start_s, "end_s": args.end_s, "frames": len(expected_times)},
            "segment": "dual_evaluation",
            "source_update_interval_s": 1.0,
            "calibration_interval_s": [args.calibration_start_s, args.calibration_end_s],
            "calibration_model": calibration,
            "acoustic_reliability": {
                "condition_scale": "max(1, (condition/calibration p75)^2)",
                "reprojection_scale": "max(1, (reprojection/calibration p75)^2)",
                "node_scale": "max(1, (target-specific effective inlier sigma/calibration median)^2)",
                "combination": "maximum of condition, reprojection and node scales",
                "maximum_dynamic_scale": args.maximum_dynamic_scale,
                "counts": counts,
            },
            "observation_covariance": {
                "baseline": "PSD-projected A6 position covariance with the predeclared physical gate",
                "calibration_floor": "shrunken calibration residual second moment, including systematic bias",
                "profile_rule": args.profile,
                "minimum_eigenvalue_m2": minimum_eigenvalue,
                "finite_psd": bool(math.isfinite(minimum_eigenvalue) and minimum_eigenvalue > 0.0),
            },
            "gps_role": (
                "GPS is used only on 46540-46560 s to freeze the covariance floor and for offline scoring; "
                "no evaluation GPS enters acoustic reliability factors, observation values, state updates, or profile selection"
            ),
            "source_a6_state": str(state_path),
            "source_a6_state_sha256": sha256(state_path),
            "source_a6_manifest": str(args.a6_manifest),
            "source_a6_manifest_sha256": sha256(args.a6_manifest),
            "source_reliability_csv": str(args.reliability_csv),
            "source_reliability_csv_sha256": sha256(args.reliability_csv),
            "gps_truth_source": str(truth_source),
            "gps_truth_source_sha256": sha256(truth_source),
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": sha256(Path(__file__).resolve()),
        }
        (frontend / "frontend_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps({
        "output": str(args.output),
        "profile": args.profile,
        "targets": args.targets,
        "window": [args.start_s, args.end_s],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
