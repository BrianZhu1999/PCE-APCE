#!/usr/bin/env python3
"""Select a common near-full-circle dual-source window from GPS geometry only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


TARGETS = (1, 2)


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
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_xy(path: Path) -> dict[int, np.ndarray]:
    return {
        int(round(float(row["time_s"]))): np.asarray([float(row["px"]), float(row["py"])])
        for row in read_csv(path)
    }


def load_valid_times(path: Path) -> set[int]:
    return {
        int(round(float(row["time_s"])))
        for row in read_csv(path)
        if row.get("valid", "true").lower() == "true"
    }


def circle_metrics(xy: np.ndarray) -> dict[str, float]:
    design = np.column_stack((2.0 * xy[:, 0], 2.0 * xy[:, 1], np.ones(len(xy))))
    rhs = np.sum(np.square(xy), axis=1)
    center_x, center_y, constant = np.linalg.lstsq(design, rhs, rcond=None)[0]
    center = np.asarray([center_x, center_y])
    radii = np.linalg.norm(xy - center, axis=1)
    radius = float(np.mean(radii))
    angles = np.unwrap(np.arctan2(xy[:, 1] - center_y, xy[:, 0] - center_x))
    increments = np.diff(angles)
    net_sweep = float(abs(np.degrees(angles[-1] - angles[0])))
    total_sweep = float(np.degrees(np.sum(np.abs(increments))))
    direction_consistency = net_sweep / max(total_sweep, 1e-12)
    return {
        "center_east_m": float(center_x),
        "center_north_m": float(center_y),
        "radius_m": radius,
        "net_sweep_deg": net_sweep,
        "total_sweep_deg": total_sweep,
        "direction_consistency": direction_consistency,
        "closure_m": float(np.linalg.norm(xy[-1] - xy[0])),
        "closure_radius_fraction": float(np.linalg.norm(xy[-1] - xy[0]) / max(radius, 1e-12)),
        "radial_cv": float(np.std(radii) / max(radius, 1e-12)),
        "radial_rmse_m": float(np.sqrt(np.mean(np.square(radii - radius)))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth-root", type=Path, required=True)
    parser.add_argument("--a6-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-length", type=int, default=40)
    parser.add_argument("--start-time-s", type=int, default=46561)
    parser.add_argument("--end-time-s", type=int, default=46740)
    parser.add_argument("--minimum-sweep-deg", type=float, default=330.0)
    parser.add_argument("--maximum-sweep-deg", type=float, default=420.0)
    parser.add_argument("--maximum-closure-radius-fraction", type=float, default=0.50)
    parser.add_argument("--minimum-direction-consistency", type=float, default=0.85)
    parser.add_argument("--maximum-radial-cv", type=float, default=0.35)
    args = parser.parse_args()

    truth_paths = {
        target: args.truth_root / f"target{target}" / "frontend" / "gps_truth.csv"
        for target in TARGETS
    }
    state_paths = {
        target: args.a6_root / "a6" / f"target{target}_state.csv"
        for target in TARGETS
    }
    truth = {target: load_xy(path) for target, path in truth_paths.items()}
    valid_times = {target: load_valid_times(path) for target, path in state_paths.items()}
    times = np.asarray(sorted(set.intersection(
        *(set(truth[target]) & valid_times[target] for target in TARGETS)
    )), dtype=int)
    times = times[(times >= args.start_time_s) & (times <= args.end_time_s)]
    if len(times) < args.minimum_length:
        raise RuntimeError("insufficient common valid A6/GPS times")

    candidates: list[dict[str, object]] = []
    for start in range(len(times)):
        for stop in range(start + args.minimum_length, len(times) + 1):
            window_times = times[start:stop]
            if not np.all(np.diff(window_times) == 1):
                continue
            row: dict[str, object] = {
                "start_time_s": int(window_times[0]),
                "end_time_s": int(window_times[-1]),
                "frames": int(len(window_times)),
                "duration_s": int(window_times[-1] - window_times[0]),
            }
            metrics = {}
            for target in TARGETS:
                xy = np.asarray([truth[target][int(time)] for time in window_times])
                target_metrics = circle_metrics(xy)
                metrics[target] = target_metrics
                row.update({f"target{target}_{key}": value for key, value in target_metrics.items()})
            row["minimum_target_sweep_deg"] = min(metrics[target]["net_sweep_deg"] for target in TARGETS)
            row["maximum_target_sweep_deg"] = max(metrics[target]["net_sweep_deg"] for target in TARGETS)
            row["maximum_target_sweep_deviation_from_360_deg"] = max(
                abs(metrics[target]["net_sweep_deg"] - 360.0) for target in TARGETS
            )
            row["maximum_closure_radius_fraction"] = max(metrics[target]["closure_radius_fraction"] for target in TARGETS)
            row["minimum_direction_consistency"] = min(metrics[target]["direction_consistency"] for target in TARGETS)
            row["maximum_radial_cv"] = max(metrics[target]["radial_cv"] for target in TARGETS)
            row["admitted"] = bool(
                row["minimum_target_sweep_deg"] >= args.minimum_sweep_deg
                and row["maximum_target_sweep_deg"] <= args.maximum_sweep_deg
                and row["maximum_closure_radius_fraction"] <= args.maximum_closure_radius_fraction
                and row["minimum_direction_consistency"] >= args.minimum_direction_consistency
                and row["maximum_radial_cv"] <= args.maximum_radial_cv
            )
            row["geometry_score"] = (
                float(row["maximum_target_sweep_deviation_from_360_deg"])
                + 90.0 * float(row["maximum_closure_radius_fraction"])
                + 120.0 * (1.0 - float(row["minimum_direction_consistency"]))
                + 120.0 * float(row["maximum_radial_cv"])
                + 0.03 * int(row["frames"])
            )
            candidates.append(row)

    candidates.sort(key=lambda row: (not bool(row["admitted"]), float(row["geometry_score"])))
    admitted = [row for row in candidates if bool(row["admitted"])]
    selected = admitted[0] if admitted else candidates[0]
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "dual_full_circle_window_candidates.csv", candidates)
    manifest = {
        "selection_status": "admitted" if admitted else "no_window_met_all_geometry_gates",
        "selection_uses_apce_error": False,
        "gps_role": "offline geometric window selection only; no GPS enters A6 or APCE state updates",
        "gates": {
            "minimum_target_sweep_deg": args.minimum_sweep_deg,
            "maximum_target_sweep_deg": args.maximum_sweep_deg,
            "maximum_closure_radius_fraction": args.maximum_closure_radius_fraction,
            "minimum_direction_consistency": args.minimum_direction_consistency,
            "maximum_radial_cv": args.maximum_radial_cv,
            "both_targets_required": True,
            "common_contiguous_one_second_A6_updates_required": True,
        },
        "common_valid_interval": {
            "start_time_s": int(times[0]),
            "end_time_s": int(times[-1]),
            "frames": int(len(times)),
        },
        "eligible_window_count": len(admitted),
        "selected": selected,
        "top_20": candidates[:20],
        "sources": {
            "truth": {str(target): str(path) for target, path in truth_paths.items()},
            "a6_state": {str(target): str(path) for target, path in state_paths.items()},
        },
        "source_hashes": {
            str(path): sha256(path)
            for path in (*truth_paths.values(), *state_paths.values())
        },
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    manifest_path = args.output / "dual_full_circle_window_audit.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "selection_status": manifest["selection_status"],
        "eligible_window_count": len(admitted),
        "selected": selected,
        "manifest": str(manifest_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
