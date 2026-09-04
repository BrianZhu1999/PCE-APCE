#!/usr/bin/env python3
"""Build an audited dual-source stable-segment publication source bundle."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


TARGETS = (1, 2)
SEEDS = (2026082601, 2026082602, 2026082603, 2026082604, 2026082605)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def circle_metrics(xy: np.ndarray) -> dict[str, float]:
    design = np.column_stack((2.0 * xy[:, 0], 2.0 * xy[:, 1], np.ones(len(xy))))
    rhs = np.sum(np.square(xy), axis=1)
    center_x, center_y, _ = np.linalg.lstsq(design, rhs, rcond=None)[0]
    center = np.asarray([center_x, center_y])
    radii = np.linalg.norm(xy - center, axis=1)
    radius = float(np.mean(radii))
    angles = np.unwrap(np.arctan2(xy[:, 1] - center_y, xy[:, 0] - center_x))
    increments = np.diff(angles)
    net_sweep = float(abs(np.degrees(angles[-1] - angles[0])))
    total_sweep = float(np.degrees(np.sum(np.abs(increments))))
    return {
        "center_east_m": float(center_x),
        "center_north_m": float(center_y),
        "radius_m": radius,
        "net_sweep_deg": net_sweep,
        "total_sweep_deg": total_sweep,
        "direction_consistency": net_sweep / max(total_sweep, 1e-12),
        "radial_cv": float(np.std(radii) / max(radius, 1e-12)),
    }


def load_runs(
    root: Path,
    target: int,
    expected_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, list[Path]]:
    paths = [root / f"target{target}" / "runs" / f"apce_seed_{seed}.json" for seed in SEEDS]
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(
        payload.get("status") != "valid" or len(payload.get("records", [])) != expected_frames
        for payload in payloads
    ):
        raise RuntimeError(f"target {target}: incomplete {expected_frames}-frame APCE matrix")
    times = np.asarray(
        [int(round(float(row["time_s"]))) for row in payloads[0]["records"]], dtype=int
    )
    if not all(
        np.array_equal(
            times,
            [int(round(float(row["time_s"]))) for row in payload["records"]],
        )
        for payload in payloads
    ):
        raise RuntimeError(f"target {target}: run timelines disagree")
    states = np.asarray([
        [[float(row[key]) for key in ("px", "py", "pz")] for row in payload["records"]]
        for payload in payloads
    ])
    widths = np.asarray([
        [float(row["interval_width_m"]) for row in payload["records"]]
        for payload in payloads
    ])
    coverage = float(np.mean([
        float(row["coverage_90"])
        for payload in payloads for row in payload["records"]
    ]))
    return times, np.median(states, axis=0), np.median(widths, axis=0), coverage, paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend-root", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--acoustic-selection-manifest", type=Path, required=True)
    parser.add_argument("--profile-selection", type=Path)
    parser.add_argument("--window-key", default="60")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    acoustic_selection = json.loads(args.acoustic_selection_manifest.read_text(encoding="utf-8"))
    selected_acoustic = acoustic_selection["lengths"][str(args.window_key)]["selected"]
    expected_frames = int(selected_acoustic["frames"])
    start_time = int(selected_acoustic["start_time_s"])
    end_time = int(selected_acoustic["end_time_s"])

    truth: dict[int, np.ndarray] = {}
    estimates: dict[int, np.ndarray] = {}
    widths: dict[int, np.ndarray] = {}
    coverage: dict[int, float] = {}
    frontend_rows: dict[int, list[dict[str, str]]] = {}
    geometry: dict[int, dict[str, float]] = {}
    metrics: dict[int, dict[str, float | bool | int]] = {}
    run_paths: list[Path] = []
    source_paths: list[Path] = [args.acoustic_selection_manifest]
    profile_selection: dict[str, object] | None = None
    if args.profile_selection is not None:
        profile_selection = json.loads(args.profile_selection.read_text(encoding="utf-8"))
        source_paths.append(args.profile_selection)
    times: np.ndarray | None = None

    for target in TARGETS:
        frontend = args.frontend_root / f"target{target}" / "frontend"
        truth_path = frontend / "gps_truth.csv"
        observation_path = frontend / "observations_cartesian.csv"
        manifest_path = frontend / "frontend_manifest.json"
        source_paths.extend((truth_path, observation_path, manifest_path))
        frontend_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if profile_selection is not None:
            selected_profiles = profile_selection.get("selected_profile_by_target", {})
            expected_profile = selected_profiles.get(str(target))
            if expected_profile is None or frontend_manifest.get("profile") != expected_profile:
                raise RuntimeError(
                    f"target {target}: frontend reliability profile does not match frozen selection"
                )
        truth_rows = read_csv(truth_path)
        truth_times = np.asarray([int(round(float(row["time_s"]))) for row in truth_rows], dtype=int)
        truth[target] = np.asarray([
            [float(row[key]) for key in ("px", "py", "pz")] for row in truth_rows
        ])
        if (
            len(truth_times) != expected_frames
            or int(truth_times[0]) != start_time
            or int(truth_times[-1]) != end_time
            or not np.all(np.diff(truth_times) == 1)
        ):
            raise RuntimeError(f"target {target}: truth does not match acoustically selected window")
        run_times, estimates[target], widths[target], coverage[target], paths = load_runs(
            args.formal_root, target, expected_frames
        )
        run_paths.extend(paths)
        if not np.array_equal(truth_times, run_times):
            raise RuntimeError(f"target {target}: truth/run timeline mismatch")
        times = run_times if times is None else times
        if not np.array_equal(times, run_times):
            raise RuntimeError("target timelines disagree")

        observations = read_csv(observation_path)
        if len(observations) != expected_frames:
            raise RuntimeError(f"target {target}: observation length mismatch")
        frontend_rows[target] = observations
        error = np.linalg.norm(estimates[target] - truth[target], axis=1)
        steps = np.linalg.norm(np.diff(estimates[target], axis=0), axis=1)
        frontend_error = np.asarray([
            np.linalg.norm(
                np.asarray([float(row[key]) for key in ("y_E", "y_N", "y_U")])
                - truth[target][index]
            )
            for index, row in enumerate(observations)
        ])
        minimum_eigenvalue = min(
            float(np.min(np.linalg.eigvalsh(np.asarray([
                [float(row[f"R_{i}{j}"]) for j in range(3)] for i in range(3)
            ]))))
            for row in observations
        )
        geometry[target] = circle_metrics(truth[target][:, :2])
        metrics[target] = {
            "rmse_m": float(np.sqrt(np.mean(np.square(error)))),
            "median_error_m": float(np.median(error)),
            "p90_error_m": float(np.percentile(error, 90.0)),
            "frontend_rmse_m": float(np.sqrt(np.mean(np.square(frontend_error)))),
            "median_marginal_width_m": float(np.median(widths[target])),
            "mean_component_coverage_90": coverage[target],
            "maximum_step_m": float(np.max(steps)),
            "minimum_covariance_eigenvalue_m2": minimum_eigenvalue,
            "finite_uncertainty": bool(np.isfinite(widths[target]).all() and (widths[target] >= 0.0).all()),
            "covariance_psd": bool(minimum_eigenvalue > 0.0),
            "quality_scaled_frames": int(sum(
                float(row["quality_covariance_multiplier"]) > 1.0 + 1e-9
                for row in observations
            )),
        }

    assert times is not None
    direct = (
        np.linalg.norm(estimates[1] - truth[1], axis=1)
        + np.linalg.norm(estimates[2] - truth[2], axis=1)
    )
    swapped = (
        np.linalg.norm(estimates[1] - truth[2], axis=1)
        + np.linalg.norm(estimates[2] - truth[1], axis=1)
    )
    identity_fraction = float(np.mean(direct <= swapped))
    estimated_separation = np.linalg.norm(estimates[1] - estimates[2], axis=1)
    truth_separation = np.linalg.norm(truth[1] - truth[2], axis=1)

    source_rows: list[dict[str, object]] = []
    for index, time_s in enumerate(times):
        row: dict[str, object] = {"time_s": int(time_s), "elapsed_s": int(time_s - times[0])}
        for target in TARGETS:
            for dimension, name in enumerate(("east", "north", "up")):
                row[f"target{target}_gps_{name}_m"] = float(truth[target][index, dimension])
                row[f"target{target}_apce_{name}_median_5seeds_m"] = float(estimates[target][index, dimension])
            row[f"target{target}_apce_error_m"] = float(
                np.linalg.norm(estimates[target][index] - truth[target][index])
            )
            row[f"target{target}_apce_median_marginal_width_m"] = float(widths[target][index])
            row[f"target{target}_quality_covariance_multiplier"] = float(
                frontend_rows[target][index]["quality_covariance_multiplier"]
            )
            row[f"target{target}_inlier_nodes"] = int(frontend_rows[target][index]["inlier_nodes"])
            row[f"target{target}_reprojection_rms_deg"] = float(
                frontend_rows[target][index]["reprojection_rms_deg"]
            )
        source_rows.append(row)
    source_path = args.output / "dual_stable_segment_quality_gated_timeseries.csv"
    write_csv(source_path, source_rows)

    selected: dict[str, object] = {
        "length_frames": int(len(times)),
        "duration_s": int(times[-1] - times[0]),
        "start_time_s": int(times[0]),
        "end_time_s": int(times[-1]),
        "identity_match_fraction": identity_fraction,
        "minimum_estimated_target_separation_m": float(np.min(estimated_separation)),
        "minimum_truth_target_separation_m": float(np.min(truth_separation)),
    }
    for target in TARGETS:
        selected.update({f"target{target}_{key}": value for key, value in metrics[target].items()})
        selected[f"target{target}_gps_sweep_deg"] = float(geometry[target]["net_sweep_deg"])
    anomaly_start, anomaly_end = acoustic_selection["excluded_acoustic_anomaly_interval_s"]
    excludes_anomaly = end_time < int(anomaly_start) or start_time > int(anomaly_end)
    selected.update({
        "worst_target_rmse_m": max(float(metrics[target]["rmse_m"]) for target in TARGETS),
        "mean_target_rmse_m": float(np.mean([metrics[target]["rmse_m"] for target in TARGETS])),
        "worst_target_p90_m": max(float(metrics[target]["p90_error_m"]) for target in TARGETS),
        "admitted_identity": identity_fraction >= 0.90,
        "admitted_jump": max(float(metrics[target]["maximum_step_m"]) for target in TARGETS) <= 200.0,
        "admitted_error": all(math.isfinite(float(metrics[target]["rmse_m"])) for target in TARGETS),
        "admitted_uncertainty": all(bool(metrics[target]["finite_uncertainty"]) for target in TARGETS),
        "admitted_covariance": all(bool(metrics[target]["covariance_psd"]) for target in TARGETS),
        "admitted_acoustic_selection": excludes_anomaly,
        "stable_segment_admitted": bool(
            identity_fraction >= 0.90
            and max(float(metrics[target]["maximum_step_m"]) for target in TARGETS) <= 200.0
            and all(bool(metrics[target]["finite_uncertainty"]) for target in TARGETS)
            and all(bool(metrics[target]["covariance_psd"]) for target in TARGETS)
            and excludes_anomaly
        ),
        "complete_circle_geometry_admitted": False,
    })
    selection = {
        "selection_status": "60-frame continuous stable segment selected from A6 acoustic diagnostics before APCE scoring",
        "selection_uses_apce_error": False,
        "selection_uses_gps_geometry": False,
        "gps_role": "offline trajectory characterization, identity audit and error scoring only; no GPS enters acoustic selection, quality scaling, robust innovation, APCE state update or branch evidence",
        "admission": {
            "identity_match_fraction_min": 0.90,
            "maximum_one_second_step_m": 200.0,
            "finite_nonnegative_uncertainty_required": True,
            "frontend_observation_covariance_psd_required": True,
            "contiguous_one_second_updates_required": True,
            "acoustic_anomaly_excluded": [int(anomaly_start), int(anomaly_end)],
            "complete_circle_required": False,
        },
        "selected": selected,
        "acoustic_selection": selected_acoustic,
        "geometry": {str(target): geometry[target] for target in TARGETS},
        "reliability_profile_selection": profile_selection,
        "formal_root": str(args.formal_root),
        "frontend_root": str(args.frontend_root),
        "source_csv": str(source_path),
        "source_hashes": {
            str(path): sha256(path)
            for path in [*source_paths, source_path, *run_paths]
        },
        "seeds": {str(target): list(SEEDS) for target in TARGETS},
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    manifest_path = args.output / "dual_stable_segment_quality_gated_selection.json"
    manifest_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "source": str(source_path),
        "manifest": str(manifest_path),
        "selected": selected,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
