#!/usr/bin/env python3
"""Build Cartesian PCE/APCE inputs from a frozen joint-3D acoustic track.

Only raw associated ray-intersection positions are exported as observations.
Frames bridged by the acoustic tracker are retained on the common clock but
carry ``observation_update=False``.  RTS states and covariances are never used
as downstream observations.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np



def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track-result", type=Path, required=True)
    parser.add_argument("--hypotheses-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-frames", type=int, default=31)
    parser.add_argument("--expected-frames", type=int, default=122)
    parser.add_argument("--hold-covariance-std", type=float, default=180.0)
    args = parser.parse_args()

    track = json.loads(args.track_result.read_text(encoding="utf-8"))
    cache = json.loads(args.hypotheses_cache.read_text(encoding="utf-8"))
    times = np.asarray(track["times"], dtype=float)
    raw_positions = np.asarray(track["raw_positions"], dtype=float)
    association_path = track["measurement_association_path"]
    measurement_std = track["measurement_std_m"]
    truth_all = np.asarray(track["offline_truth"], dtype=float)
    identity = [int(value) for value in track["offline_gps_summary"]["global_identity_assignment"]]
    source_start, source_end = [int(value) for value in track["protocol"]["source_frame_labels"]]
    frame_count = len(times)

    if frame_count != args.expected_frames:
        raise RuntimeError(f"expected {args.expected_frames} frames, found {frame_count}")
    if not (0 < args.calibration_frames < frame_count):
        raise ValueError("calibration-frames must split the trajectory")
    if raw_positions.shape != (frame_count, 3, 3):
        raise RuntimeError(f"unexpected raw position shape: {raw_positions.shape}")
    if truth_all.shape != raw_positions.shape:
        raise RuntimeError(f"unexpected truth shape: {truth_all.shape}")
    if len(association_path) != frame_count or len(measurement_std) != frame_count:
        raise RuntimeError("association, covariance and time lengths differ")
    if source_end - source_start + 1 != frame_count:
        raise RuntimeError("source frame labels are not contiguous")

    cache_start = int(track["protocol"]["source_frame_indices"][0])
    cache_frames = cache["frame_hypotheses"][cache_start : cache_start + frame_count]
    if len(cache_frames) != frame_count:
        raise RuntimeError("hypothesis cache does not cover the selected trajectory")

    args.output.mkdir(parents=True, exist_ok=True)
    target_manifests = {}
    for target in range(3):
        frontend = args.output / f"target{target + 1}" / "frontend"
        frontend.mkdir(parents=True, exist_ok=True)
        observation_rows = []
        truth_rows = []
        update_count = 0
        for index, time_s in enumerate(times):
            point_index = association_path[index][target]
            update = point_index is not None
            sigma = measurement_std[index][target]
            if update:
                if sigma is None or not np.isfinite(float(sigma)):
                    raise RuntimeError(f"missing measurement sigma at frame {index}, target {target + 1}")
                point = cache_frames[index]["points"][int(point_index)]
                available_nodes = int(point.get("within15_nodes", 0))
                update_count += 1
            else:
                sigma = args.hold_covariance_std
                available_nodes = 0
            variance = float(sigma) ** 2
            phase = "calibration" if index < args.calibration_frames else "evaluation"
            observation_rows.append({
                "segment": "full_122",
                "phase": phase,
                "time_s": f"{time_s:.12f}",
                "source_frame_index": source_start + index,
                "y_E": f"{raw_positions[index, target, 0]:.12f}",
                "y_N": f"{raw_positions[index, target, 1]:.12f}",
                "y_U": f"{raw_positions[index, target, 2]:.12f}",
                "R_00": f"{variance:.12f}", "R_01": "0", "R_02": "0",
                "R_10": "0", "R_11": f"{variance:.12f}", "R_12": "0",
                "R_20": "0", "R_21": "0", "R_22": f"{variance:.12f}",
                "measurement_std_m": f"{float(sigma):.12f}",
                "available_nodes": available_nodes,
                "valid": "True",
                "observation_update": str(update),
                "observation_source": "joint3d_ray_intersection" if update else "prediction_clock_only",
            })
            truth_target = identity[target]
            truth_rows.append({
                "time_s": f"{time_s:.12f}",
                "source_frame_index": source_start + index,
                "phase": phase,
                "px": f"{truth_all[index, truth_target, 0]:.12f}",
                "py": f"{truth_all[index, truth_target, 1]:.12f}",
                "pz": f"{truth_all[index, truth_target, 2]:.12f}",
            })

        write_csv(frontend / "observations_cartesian.csv", observation_rows)
        write_csv(frontend / "gps_truth.csv", truth_rows)
        manifest = {
            "processing_stage": "cartesian_observations",
            "acoustic_track_label": target + 1,
            "offline_gps_identity_label": identity[target] + 1,
            "segment": "full_122",
            "state_definition": "[E,N,U,vE,vN,vU]",
            "observation_definition": "raw associated joint-3D ray-intersection position",
            "observation_covariance_definition": "isotropic variance from acoustic angular cost, range and pair residual",
            "frames": frame_count,
            "source_frame_labels": [source_start, source_end],
            "duration_s": float(times[-1] - times[0]),
            "calibration": {
                "frames": args.calibration_frames,
                "source_frame_labels": [source_start, source_start + args.calibration_frames - 1],
                "duration_s": float(times[args.calibration_frames - 1] - times[0]),
            },
            "evaluation": {
                "frames": frame_count - args.calibration_frames,
                "source_frame_labels": [source_start + args.calibration_frames, source_end],
                "duration_s": float(times[-1] - times[args.calibration_frames]),
            },
            "measurement_updates": update_count,
            "measurement_update_fraction": update_count / frame_count,
            "held_prediction_frames": frame_count - update_count,
            "gps_role": "offline identity assignment and scoring only",
            "gps_runtime_use": False,
            "rts_position_used_as_observation": False,
            "rts_covariance_used_as_observation": False,
            "source_track_result": args.track_result.name,
            "source_hypotheses_cache": args.hypotheses_cache.name,
            "update_schedule": "PCE/APCE analysis updates occur on observation-clock rows.",
        }
        (frontend / "frontend_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8"
        )
        target_manifests[str(target + 1)] = manifest

    root_manifest = {
        "processing_stage": "extended_prediction_intervals",
        "frames": frame_count,
        "intervals": frame_count - 1,
        "duration_s": float(times[-1] - times[0]),
        "source_frame_labels": [source_start, source_end],
        "calibration_frames": args.calibration_frames,
        "evaluation_frames": frame_count - args.calibration_frames,
        "targets": target_manifests,
        "observation_covariance_scale_to_apply": 1.8717,
        "gps_runtime_use": False,
        "rts_runtime_use": False,
    }
    (args.output / "bundle_manifest.json").write_text(
        json.dumps(root_manifest, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    print(json.dumps(root_manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
