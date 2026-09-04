#!/usr/bin/env python3
"""Offline oracle-identity upper bound for the author-informed triple gate."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import run_three_source_author_reliability_gate as gate
import run_three_source_global_association_gate as base


def metrics(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p90": None, "maximum": None}
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "maximum": float(np.max(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--nod", type=Path, required=True)
    parser.add_argument("--gps-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-seconds", type=float, default=30.0)
    parser.add_argument("--frame-limit", type=int)
    args = parser.parse_args()

    nodes, ip_to_node = base.parse_nodes(args.nod)
    deals, oracles = {}, {}
    for path in sorted(args.data_dir.glob("deal_doa_*.txt")):
        suffix = int(path.name.split("_")[2])
        node = ip_to_node[suffix]
        deal = base.read_numeric(path, 8)
        gps_doa = base.read_numeric(args.data_dir / f"gps_doa_{suffix}.txt", 11)
        deals[node] = deal
        oracles[node] = base.gps_oracle_angles(deal, gps_doa)
    active_nodes = sorted(deals)
    frame_count = min(len(deals[node]) for node in active_nodes)
    if args.frame_limit is not None:
        frame_count = min(frame_count, args.frame_limit)
    deals = {node: deals[node][:frame_count] for node in active_nodes}
    oracles = {node: oracles[node][:frame_count] for node in active_nodes}
    calibration_frames = int(round(args.calibration_seconds / base.FRAME_DT_S))

    transforms, candidates, permutations = {}, {}, {}
    for node in active_nodes:
        raw = deals[node][:, 1:7].reshape(frame_count, 3, 2)
        transforms[node] = base.calibrate_transform(raw, oracles[node], calibration_frames)
        candidates[node] = np.asarray([base.apply_transform(frame, transforms[node]) for frame in raw])
        permutations[node] = [
            base.permutation_cost(candidates[node][frame], oracles[node][frame, :, :2])[0]
            for frame in range(frame_count)
        ]
    reliability, selected_nodes = gate.calibration_precision(
        candidates, oracles, permutations, calibration_frames
    )

    gps1 = base.parse_gps(args.gps_root / "GPS1_plane1.gps")
    gps2 = base.parse_gps(args.gps_root / "GPS2_plane1.gps")
    tracks = [
        base.fuse_tracks(gps1, gps2),
        base.parse_gps(args.gps_root / "GPS3_plane2.gps"),
        base.parse_gps(args.gps_root / "GPS4_plane2to3.gps"),
    ]
    truth = [base.interpolate_track(deals[active_nodes[0]], track) for track in tracks]
    velocity_start = max(0, calibration_frames - int(round(5.0 / base.FRAME_DT_S)))
    elapsed = max((calibration_frames - 1 - velocity_start) * base.FRAME_DT_S, base.FRAME_DT_S)
    states, covariances = [], []
    for target in base.TARGETS:
        velocity = (truth[target][calibration_frames - 1] - truth[target][velocity_start]) / elapsed
        states.append(np.concatenate([truth[target][calibration_frames - 1], velocity]))
        covariances.append(np.diag([gate.INITIAL_POSITION_STD_M**2] * 3 + [gate.INITIAL_VELOCITY_STD_MPS**2] * 3))

    geometry_errors = {target: [] for target in base.TARGETS}
    state_errors = {target: [] for target in base.TARGETS}
    accepted = {target: 0 for target in base.TARGETS}
    ospa_values = []
    for frame in range(calibration_frames, frame_count):
        frame_estimates = []
        for target in base.TARGETS:
            predicted_state, predicted_covariance = gate.predict_state(states[target], covariances[target])
            observations, variances = {}, {}
            for node in selected_nodes[target]:
                candidate = candidates[node][frame, permutations[node][frame][target]]
                observations[node] = candidate
                row = reliability[node][target]
                variances[node] = np.square([row["sigma_azimuth_deg"], row["sigma_zenith_deg"]])
            position, _, _, measurement_covariance, _ = gate.weighted_ray_solution(observations, variances, nodes)
            if position is not None:
                geometry_errors[target].append(float(np.linalg.norm(position - truth[target][frame])))
            states[target], covariances[target], updated = gate.update_state(
                predicted_state, predicted_covariance, position, measurement_covariance
            )
            accepted[target] += int(updated)
            state_errors[target].append(float(np.linalg.norm(states[target][:3] - truth[target][frame])))
            frame_estimates.append(states[target][:3].copy())
        ospa_values.append(gate.ospa_order2(np.asarray(frame_estimates), np.asarray([truth[target][frame] for target in base.TARGETS])))

    post_frames = frame_count - calibration_frames
    result = {
        "audit": "offline correct-peak-identity upper bound with calibration-frozen target-specific reliability",
        "gps_role": "correct candidate identity for upper-bound diagnosis and offline scoring only; never an admissible online tracker",
        "frame_count": frame_count,
        "calibration_frames": calibration_frames,
        "post_calibration_frames": post_frames,
        "selected_nodes": {f"target{target + 1}": selected_nodes[target] for target in base.TARGETS},
        "targets": {
            f"target{target + 1}": {
                "raw_geometry_error_m": metrics(geometry_errors[target]),
                "state_error_m": metrics(state_errors[target]),
                "accepted_update_fraction": accepted[target] / max(post_frames, 1),
            }
            for target in base.TARGETS
        },
        "ospa_order2_m": metrics(ospa_values),
        "diagnostic_gate": {
            "upper_bound_supports_further_doa_association_work": bool(
                metrics(ospa_values)["mean"] is not None
                and metrics(ospa_values)["mean"] <= 150.0
                and all(accepted[target] / max(post_frames, 1) >= 0.90 for target in base.TARGETS)
            )
        },
        "scripts": {
            "script_sha256": gate.file_sha256(Path(__file__)),
            "gate_script_sha256": gate.file_sha256(Path(gate.__file__)),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "oracle_identity_upper_bound.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
