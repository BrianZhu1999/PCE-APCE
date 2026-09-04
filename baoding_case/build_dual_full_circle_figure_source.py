#!/usr/bin/env python3
"""Build the audited 87-frame dual-source publication source bundle."""
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


def load_runs(root: Path, target: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, list[Path]]:
    paths = [root / f"target{target}" / "runs" / f"apce_seed_{seed}.json" for seed in SEEDS]
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(payload.get("status") != "valid" or len(payload.get("records", [])) != 87 for payload in payloads):
        raise RuntimeError(f"target {target}: incomplete APCE matrix")
    times = np.asarray([int(round(float(row["time_s"]))) for row in payloads[0]["records"]], dtype=int)
    if not all(np.array_equal(times, [int(round(float(row["time_s"]))) for row in payload["records"]]) for payload in payloads):
        raise RuntimeError(f"target {target}: run timelines disagree")
    states = np.asarray([
        [[float(row[key]) for key in ("px", "py", "pz")] for row in payload["records"]]
        for payload in payloads
    ])
    widths = np.asarray([[float(row["interval_width_m"]) for row in payload["records"]] for payload in payloads])
    coverage = float(np.mean([
        float(row["coverage_90"])
        for payload in payloads for row in payload["records"]
    ]))
    return times, np.median(states, axis=0), np.median(widths, axis=0), coverage, paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend-root", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    geometry = json.loads(args.geometry_manifest.read_text(encoding="utf-8"))
    selected_geometry = geometry["selected"]
    truth: dict[int, np.ndarray] = {}
    estimates: dict[int, np.ndarray] = {}
    widths: dict[int, np.ndarray] = {}
    coverage: dict[int, float] = {}
    run_paths: list[Path] = []
    frontend_rows: dict[int, list[dict[str, str]]] = {}
    metrics: dict[int, dict[str, float | bool]] = {}
    times: np.ndarray | None = None

    for target in TARGETS:
        truth_rows = read_csv(args.frontend_root / f"target{target}" / "frontend" / "gps_truth.csv")
        truth_times = np.asarray([int(round(float(row["time_s"]))) for row in truth_rows], dtype=int)
        truth[target] = np.asarray([[float(row[key]) for key in ("px", "py", "pz")] for row in truth_rows])
        run_times, estimates[target], widths[target], coverage[target], paths = load_runs(args.formal_root, target)
        run_paths.extend(paths)
        if not np.array_equal(truth_times, run_times):
            raise RuntimeError(f"target {target}: truth/run timeline mismatch")
        times = run_times if times is None else times
        if not np.array_equal(times, run_times):
            raise RuntimeError("target timelines disagree")
        error = np.linalg.norm(estimates[target] - truth[target], axis=1)
        steps = np.linalg.norm(np.diff(estimates[target], axis=0), axis=1)
        observations = read_csv(args.frontend_root / f"target{target}" / "frontend" / "observations_cartesian.csv")
        frontend_rows[target] = observations
        frontend_error = np.asarray([
            np.linalg.norm(
                np.asarray([float(row[key]) for key in ("y_E", "y_N", "y_U")]) - truth[target][index]
            ) for index, row in enumerate(observations)
        ])
        minimum_eigenvalue = min(
            float(np.min(np.linalg.eigvalsh(np.asarray([
                [float(row[f"R_{i}{j}"]) for j in range(3)] for i in range(3)
            ])))) for row in observations
        )
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
            "quality_scaled_frames": int(sum(float(row["quality_covariance_multiplier"]) > 1.0 + 1e-9 for row in observations)),
        }

    assert times is not None
    direct = np.linalg.norm(estimates[1] - truth[1], axis=1) + np.linalg.norm(estimates[2] - truth[2], axis=1)
    swapped = np.linalg.norm(estimates[1] - truth[2], axis=1) + np.linalg.norm(estimates[2] - truth[1], axis=1)
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
            row[f"target{target}_apce_error_m"] = float(np.linalg.norm(estimates[target][index] - truth[target][index]))
            row[f"target{target}_apce_median_marginal_width_m"] = float(widths[target][index])
            row[f"target{target}_quality_covariance_multiplier"] = float(frontend_rows[target][index]["quality_covariance_multiplier"])
            row[f"target{target}_inlier_nodes"] = int(frontend_rows[target][index]["inlier_nodes"])
            row[f"target{target}_reprojection_rms_deg"] = float(frontend_rows[target][index]["reprojection_rms_deg"])
        source_rows.append(row)
    source_path = args.output / "dual_full_circle_quality_gated_timeseries.csv"
    write_csv(source_path, source_rows)

    selected: dict[str, object] = {
        "length_frames": int(len(times)), "start_time_s": int(times[0]), "end_time_s": int(times[-1]),
        "identity_match_fraction": identity_fraction,
        "minimum_estimated_target_separation_m": float(np.min(estimated_separation)),
        "minimum_truth_target_separation_m": float(np.min(truth_separation)),
    }
    for target in TARGETS:
        selected.update({f"target{target}_{key}": value for key, value in metrics[target].items()})
    selected.update({
        "worst_target_rmse_m": max(float(metrics[target]["rmse_m"]) for target in TARGETS),
        "mean_target_rmse_m": float(np.mean([metrics[target]["rmse_m"] for target in TARGETS])),
        "worst_target_p90_m": max(float(metrics[target]["p90_error_m"]) for target in TARGETS),
        "admitted_identity": identity_fraction >= 0.90,
        "admitted_jump": max(float(metrics[target]["maximum_step_m"]) for target in TARGETS) <= 200.0,
        "admitted_error": all(math.isfinite(float(metrics[target]["rmse_m"])) for target in TARGETS),
        "admitted_uncertainty": all(bool(metrics[target]["finite_uncertainty"]) for target in TARGETS),
        "admitted_covariance": all(bool(metrics[target]["covariance_psd"]) for target in TARGETS),
        "complete_circle_geometry_admitted": bool(selected_geometry["admitted"]),
        "target1_gps_sweep_deg": float(selected_geometry["target1_net_sweep_deg"]),
        "target2_gps_sweep_deg": float(selected_geometry["target2_net_sweep_deg"]),
    })
    selection = {
        "selection_status": "common complete-circle geometry selected before quality-gated APCE scoring",
        "selection_uses_apce_error": False,
        "gps_role": "offline common-circle geometry selection, identity audit and error scoring only; no GPS enters quality scaling, robust innovation, APCE state update or branch evidence",
        "admission": {
            "identity_match_fraction_min": 0.90,
            "maximum_one_second_step_m": 200.0,
            "error_gate": "finite long-window RMSE reported without a post-hoc numerical cap",
            "finite_nonnegative_uncertainty_required": True,
            "frontend_observation_covariance_psd_required": True,
            "contiguous_one_second_updates_required": True,
            "complete_circle_geometry_required": True,
        },
        "selected": selected,
        "geometry": selected_geometry,
        "formal_root": str(args.formal_root),
        "frontend_root": str(args.frontend_root),
        "source_csv": str(source_path),
        "source_hashes": {str(path): sha256(path) for path in [args.geometry_manifest, source_path, *run_paths]},
        "seeds": {str(target): list(SEEDS) for target in TARGETS},
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    manifest_path = args.output / "dual_full_circle_quality_gated_selection.json"
    manifest_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"source": str(source_path), "manifest": str(manifest_path), "selected": selected}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
