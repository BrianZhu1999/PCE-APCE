#!/usr/bin/env python3
"""Blind audit of the corrected raw-WAV MUSIC near-field frontend."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import torch

try:
    from .nearfield_audit import robust_triangulate, truth_angles
    from .run_baoding import sha256, signed_deg
except ImportError:
    from nearfield_audit import robust_triangulate, truth_angles
    from run_baoding import sha256, signed_deg


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-segment", default="danyuan_panxuan_2")
    parser.add_argument("--evaluation-segment", default="danyuan_panxuan_3")
    parser.add_argument("--calibration-seconds", type=int)
    args = parser.parse_args()
    manifest = json.loads((args.frontend / "frontend_manifest.json").read_text(encoding="utf-8"))
    observations = load_rows(args.frontend / "observations.csv")
    truth = {float(row["time_s"]): torch.tensor((float(row["px"]), float(row["py"]), float(row["pz"])), dtype=torch.float64) for row in load_rows(args.frontend / "gps_truth.csv")}
    nodes_global = {int(node): torch.tensor((value["x"], value["y"], value["z"]), dtype=torch.float64) for node, value in manifest["nodes"].items()}
    center = torch.stack(list(nodes_global.values())).mean(dim=0)
    nodes = {node: value - center for node, value in nodes_global.items()}
    by_segment: dict[str, dict[float, dict[int, tuple[float, float]]]] = {}
    for row in observations:
        by_segment.setdefault(row["segment"], {}).setdefault(float(row["time_s"]), {})[int(row["node_id"])] = (float(row["azimuth_deg"]), float(row["elevation_deg"]))
    calibration_name, evaluation_name = args.calibration_segment, args.evaluation_segment
    if calibration_name not in by_segment or evaluation_name not in by_segment:
        raise RuntimeError(f"missing requested segment(s): {calibration_name}, {evaluation_name}")
    calibration_times = sorted(by_segment[calibration_name])
    calibration_end = None
    if args.calibration_seconds is not None:
        calibration_end = calibration_times[0] + args.calibration_seconds
    calibration_frames = {
        second: frame for second, frame in by_segment[calibration_name].items()
        if calibration_end is None or second < calibration_end
    }
    evaluation_frames = {
        second: frame for second, frame in by_segment[evaluation_name].items()
        if calibration_name != evaluation_name or calibration_end is None or second >= calibration_end
    }
    node_sigma = {}
    node_metrics = {}
    for node in sorted(nodes):
        residuals, azimuth_errors, elevation_errors = [], [], []
        for second, frame in calibration_frames.items():
            if node not in frame or second not in truth:
                continue
            truth_az, truth_el = truth_angles(truth[second], nodes[node])
            azimuth_error = abs(signed_deg(frame[node][0] - truth_az))
            elevation_error = abs(frame[node][1] - truth_el)
            azimuth_errors.append(azimuth_error); elevation_errors.append(elevation_error)
            residuals.append(math.hypot(azimuth_error, elevation_error))
        median = statistics.median(residuals) if residuals else float("inf")
        node_sigma[node] = max(2.0, median)
        azimuth_mae = statistics.mean(azimuth_errors) if azimuth_errors else None
        elevation_mae = statistics.mean(elevation_errors) if elevation_errors else None
        eligible = bool(len(residuals) >= 20 and azimuth_mae is not None and azimuth_mae <= 10.0 and elevation_mae <= 10.0 and median <= 10.0)
        node_metrics[node] = {"calibration_rows": len(residuals), "azimuth_mae_deg": azimuth_mae, "elevation_mae_deg": elevation_mae, "joint_median_error_deg": median, "eligible": eligible}
    def frames(segment: str) -> dict[float, dict[int, tuple[float, float, float]]]:
        return {second: {node: (angles[0], angles[1], node_sigma[node]) for node, angles in frame.items() if node_metrics[node]["eligible"]} for second, frame in by_segment[segment].items()}
    thresholds = []
    for threshold in (5.0, 10.0, 15.0, 20.0, 30.0, 45.0):
        errors, valid = [], 0
        for second, frame in frames(calibration_name).items():
            position, inliers, _ = robust_triangulate(frame, nodes, threshold)
            if position is not None and second in truth:
                valid += 1; errors.append(float(torch.linalg.vector_norm(position - truth[second])))
        fraction = valid / max(len(calibration_frames), 1)
        objective = statistics.median(errors) if errors and fraction >= 0.80 else float("inf")
        thresholds.append((objective, threshold, fraction))
    _, selected_threshold, calibration_valid_fraction = min(thresholds)
    evaluation_rows, angular_az, angular_el = [], [], []
    for second, frame in {
        second: {node: (angles[0], angles[1], node_sigma[node]) for node, angles in frame.items() if node_metrics[node]["eligible"]}
        for second, frame in evaluation_frames.items()
    }.items():
        position, inliers, condition = robust_triangulate(frame, nodes, selected_threshold)
        row = {"time_s": second, "available_nodes": len(frame), "inlier_nodes": len(inliers), "condition_number": condition, "valid": position is not None}
        if second in truth:
            for node, angles in frame.items():
                truth_az, truth_el = truth_angles(truth[second], nodes[node])
                angular_az.append(abs(signed_deg(angles[0] - truth_az))); angular_el.append(abs(angles[1] - truth_el))
        if position is not None and second in truth:
            row["position_error_m"] = float(torch.linalg.vector_norm(position - truth[second]))
        evaluation_rows.append(row)
    valid_rows = [row for row in evaluation_rows if row["valid"] and "position_error_m" in row]
    errors = [row["position_error_m"] for row in valid_rows]
    gate = {
        "task": "corrected direct-WAV historical-MUSIC near-field audit",
        "calibration_segment": calibration_name,
        "evaluation_segment": evaluation_name,
        "calibration_seconds": args.calibration_seconds,
        "calibration_time_start_s": calibration_times[0],
        "calibration_time_end_s": calibration_end,
        "selected_ransac_threshold_deg": selected_threshold,
        "eligible_nodes": [node for node in sorted(node_metrics) if node_metrics[node]["eligible"]],
        "calibration_valid_fraction": calibration_valid_fraction,
        "node_calibration": node_metrics,
        "evaluation": {
            "frames": len(evaluation_rows),
            "valid_frames": len(valid_rows),
            "valid_fraction": len(valid_rows) / max(len(evaluation_rows), 1),
            "azimuth_mae_deg": statistics.mean(angular_az) if angular_az else None,
            "elevation_mae_deg": statistics.mean(angular_el) if angular_el else None,
            "median_position_error_m": statistics.median(errors) if errors else None,
            "p90_position_error_m": sorted(errors)[min(len(errors) - 1, int(0.90 * len(errors)))] if errors else None,
            "rmse_position_m": math.sqrt(sum(value * value for value in errors) / len(errors)) if errors else None,
        },
        "admission_thresholds": {"valid_fraction_min": 0.90, "azimuth_mae_deg_max": 10.0, "elevation_mae_deg_max": 10.0, "median_position_error_m_max": 200.0, "p90_position_error_m_max": 500.0},
        "provenance": {"frontend": str(args.frontend), "frontend_manifest_sha256": sha256(args.frontend / "frontend_manifest.json"), "frontend_calibration_sha256": sha256(args.frontend / "frontend_calibration.json"), "runner_sha256": sha256(Path(__file__))},
    }
    evaluation = gate["evaluation"]
    gate["raw_wav_frontend_admitted"] = bool(evaluation["valid_fraction"] >= 0.90 and evaluation["azimuth_mae_deg"] <= 10.0 and evaluation["elevation_mae_deg"] <= 10.0 and evaluation["median_position_error_m"] <= 200.0 and evaluation["p90_position_error_m"] <= 500.0)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "raw_frontend_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output / "raw_frontend_timeseries.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = sorted({key for row in evaluation_rows for key in row}); writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(evaluation_rows)
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
