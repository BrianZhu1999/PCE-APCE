#!/usr/bin/env python3
"""Deterministic, resumable diagnostics for shuangyuan_4 smoke results.

This script deliberately avoids shell heredocs and nested quoting.  It is
copied to the Super-Server and invoked as one ordinary command, so failures
produce a traceback and a JSON status file instead of silently ending a run.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def angle_error(observed: float, expected: float) -> float:
    return (observed - expected + 180.0) % 360.0 - 180.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--target", type=int, choices=(1, 2), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    target_root = args.result_root / f"target{args.target}"
    truth_rows = read_csv(target_root / "frontend" / "gps_truth.csv")
    obs_rows = read_csv(target_root / "frontend" / "observations.csv")
    manifest = read_json(target_root / "frontend" / "frontend_manifest.json")
    nodes = manifest["nodes"]
    center = np.mean(
        [[float(n["x"]), float(n["y"]), float(n["z"])] for n in nodes.values()],
        axis=0,
    )
    truth_by_time = {float(r["time_s"]): np.array([float(r["px"]), float(r["py"]), float(r["pz"])]) for r in truth_rows}

    first_time = min(truth_by_time)
    first_obs = [r for r in obs_rows if abs(float(r["time_s"]) - first_time) < 1e-9]
    az_errors, el_errors = [], []
    per_node = []
    truth = truth_by_time[first_time]
    for row in first_obs:
        node = nodes[str(int(row["node_id"]))]
        node_xyz = np.array([float(node["x"]), float(node["y"]), float(node["z"])]) - center
        delta = truth - node_xyz
        expected_az = math.degrees(math.atan2(delta[1], delta[0])) % 360.0
        expected_el = math.degrees(math.atan2(delta[2], math.hypot(delta[0], delta[1])))
        az_error_deg = angle_error(float(row["azimuth_deg"]), expected_az)
        el_error_deg = float(row["elevation_deg"]) - expected_el
        az_errors.append(abs(az_error_deg))
        el_errors.append(abs(el_error_deg))
        per_node.append(
            {
                "node_id": int(row["node_id"]),
                "observed_azimuth_deg": float(row["azimuth_deg"]),
                "expected_azimuth_deg": expected_az,
                "azimuth_error_deg": az_error_deg,
                "observed_elevation_deg": float(row["elevation_deg"]),
                "expected_elevation_deg": expected_el,
                "elevation_error_deg": el_error_deg,
            }
        )

    run_paths = [
        target_root / "runs" / f"{method}_seed_{args.seed}.json"
        for method in ("pce", "apce")
    ]
    run_summary = {}
    for path in run_paths:
        payload = read_json(path)
        records = payload["records"]
        xyz = np.asarray([[float(r["px"]), float(r["py"]), float(r["pz"])] for r in records])
        errors = np.asarray([float(r["position_error_m"]) for r in records])
        run_summary[payload["method"]] = {
            "records": len(records),
            "position_rmse_m": float(np.sqrt(np.mean(errors**2))),
            "position_max_m": float(np.max(errors)),
            "estimate_min_m": xyz.min(axis=0).tolist(),
            "estimate_max_m": xyz.max(axis=0).tolist(),
            "first_error_m": float(errors[0]),
            "last_error_m": float(errors[-1]),
        }

    result = {
        "status": "ok",
        "target": args.target,
        "seed": args.seed,
        "coordinate_system": "centered local ENU",
        "first_time_s": first_time,
        "node_center_xyz": center.tolist(),
        "first_frame_observation_count": len(first_obs),
        "first_frame_mean_abs_azimuth_error_deg": float(np.mean(az_errors)),
        "first_frame_mean_abs_elevation_error_deg": float(np.mean(el_errors)),
        "first_frame_per_node": per_node,
        "runs": run_summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
