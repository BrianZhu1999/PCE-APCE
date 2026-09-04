#!/usr/bin/env python3
"""Audit and select long dual-source APCE showcase windows.

GPS is used only for offline scoring. The script never changes observations,
state estimates, uncertainty, identities, or tracker parameters.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


TARGETS = (1, 2)
POSITION_KEYS = ("px", "py", "pz")


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_truth(path: Path) -> dict[int, np.ndarray]:
    return {
        int(round(float(row["time_s"]))): np.asarray([float(row[key]) for key in POSITION_KEYS])
        for row in read_csv(path)
    }


def load_frontend(path: Path) -> dict[int, np.ndarray]:
    return {
        int(round(float(row["time_s"]))): np.asarray([float(row[key]) for key in ("y_E", "y_N", "y_U")])
        for row in read_csv(path)
        if row.get("valid", "True").lower() == "true"
    }


def load_frontend_covariance(path: Path) -> dict[int, np.ndarray]:
    records: dict[int, np.ndarray] = {}
    for row in read_csv(path):
        if row.get("valid", "True").lower() != "true":
            continue
        covariance = np.asarray([
            [float(row[f"R_{i}{j}"]) for j in range(3)]
            for i in range(3)
        ])
        records[int(round(float(row["time_s"])))] = 0.5 * (covariance + covariance.T)
    return records


def load_apce(root: Path) -> dict[str, object]:
    paths = sorted((root / "runs").glob("apce_seed_*.json"))
    if len(paths) != 5:
        raise RuntimeError(f"expected five APCE runs under {root}, found {len(paths)}")
    records_by_seed: list[dict[int, dict[str, object]]] = []
    seeds = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "valid":
            raise RuntimeError(f"invalid run: {path}")
        seeds.append(int(payload["seed"]))
        records_by_seed.append({int(round(float(row["time_s"]))): row for row in payload["records"]})
    common = sorted(set.intersection(*(set(records) for records in records_by_seed)))
    if not common:
        raise RuntimeError(f"no common APCE times under {root}")
    positions = np.asarray([
        [[float(records[time][key]) for key in POSITION_KEYS] for time in common]
        for records in records_by_seed
    ])
    widths = np.asarray([
        [float(records[time]["interval_width_m"]) for time in common]
        for records in records_by_seed
    ])
    coverages = np.asarray([
        [float(records[time]["coverage_90"]) for time in common]
        for records in records_by_seed
    ])
    return {
        "times": np.asarray(common, dtype=int),
        "positions": positions,
        "median_position": np.median(positions, axis=0),
        "median_width": np.median(widths, axis=0),
        "mean_coverage": np.mean(coverages, axis=0),
        "seeds": seeds,
        "paths": paths,
    }


def rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def p90(values: np.ndarray) -> float:
    return float(np.percentile(values, 90.0))


def contiguous_slices(times: np.ndarray, length: int):
    for start in range(0, len(times) - length + 1):
        stop = start + length
        candidate = times[start:stop]
        if np.all(np.diff(candidate) == 1):
            yield start, stop


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend-root", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=(55, 60, 67))
    parser.add_argument("--selected-length", type=int, default=67)
    parser.add_argument("--minimum-identity-fraction", type=float, default=0.90)
    parser.add_argument("--maximum-step-m", type=float, default=200.0)
    parser.add_argument("--maximum-target-rmse-m", type=float, default=math.inf)
    parser.add_argument("--maximum-target-p90-m", type=float, default=math.inf)
    args = parser.parse_args()

    tracks = {
        target: load_apce(args.formal_root / f"target{target}")
        for target in TARGETS
    }
    common_times = sorted(set(tracks[1]["times"].tolist()) & set(tracks[2]["times"].tolist()))
    times = np.asarray(common_times, dtype=int)
    if len(times) < max(args.lengths):
        raise RuntimeError("insufficient common dual-source evaluation frames")

    prepared: dict[int, dict[str, object]] = {}
    for target in TARGETS:
        track = tracks[target]
        index = {int(time): i for i, time in enumerate(track["times"])}
        indices = np.asarray([index[int(time)] for time in times], dtype=int)
        truth_map = load_truth(args.frontend_root / f"target{target}" / "frontend" / "gps_truth.csv")
        observation_path = args.frontend_root / f"target{target}" / "frontend" / "observations_cartesian.csv"
        frontend_map = load_frontend(observation_path)
        covariance_map = load_frontend_covariance(observation_path)
        truth = np.asarray([truth_map[int(time)] for time in times])
        frontend = np.asarray([frontend_map[int(time)] for time in times])
        covariance = np.asarray([covariance_map[int(time)] for time in times])
        covariance_min_eigenvalue = np.linalg.eigvalsh(covariance)[:, 0]
        position = np.asarray(track["median_position"])[indices]
        prepared[target] = {
            "truth": truth,
            "frontend": frontend,
            "covariance": covariance,
            "covariance_min_eigenvalue": covariance_min_eigenvalue,
            "position": position,
            "width": np.asarray(track["median_width"])[indices],
            "coverage": np.asarray(track["mean_coverage"])[indices],
            "error": np.linalg.norm(position - truth, axis=1),
            "frontend_error": np.linalg.norm(frontend - truth, axis=1),
            "step": np.r_[0.0, np.linalg.norm(np.diff(position, axis=0), axis=1)],
        }

    direct_cost = (
        np.linalg.norm(prepared[1]["position"] - prepared[1]["truth"], axis=1)
        + np.linalg.norm(prepared[2]["position"] - prepared[2]["truth"], axis=1)
    )
    swapped_cost = (
        np.linalg.norm(prepared[1]["position"] - prepared[2]["truth"], axis=1)
        + np.linalg.norm(prepared[2]["position"] - prepared[1]["truth"], axis=1)
    )
    identity_correct = direct_cost <= swapped_cost
    estimated_separation = np.linalg.norm(prepared[1]["position"] - prepared[2]["position"], axis=1)
    truth_separation = np.linalg.norm(prepared[1]["truth"] - prepared[2]["truth"], axis=1)

    candidates: list[dict[str, object]] = []
    for length in sorted(set(args.lengths)):
        for start, stop in contiguous_slices(times, length):
            row: dict[str, object] = {
                "length_frames": length,
                "start_time_s": int(times[start]),
                "end_time_s": int(times[stop - 1]),
                "identity_match_fraction": float(np.mean(identity_correct[start:stop])),
                "minimum_estimated_target_separation_m": float(np.min(estimated_separation[start:stop])),
                "minimum_truth_target_separation_m": float(np.min(truth_separation[start:stop])),
            }
            target_rmses = []
            target_p90s = []
            for target in TARGETS:
                values = prepared[target]
                error = values["error"][start:stop]
                target_rmses.append(rmse(error))
                target_p90s.append(p90(error))
                row.update({
                    f"target{target}_rmse_m": target_rmses[-1],
                    f"target{target}_median_error_m": float(np.median(error)),
                    f"target{target}_p90_error_m": target_p90s[-1],
                    f"target{target}_frontend_rmse_m": rmse(values["frontend_error"][start:stop]),
                    f"target{target}_median_marginal_width_m": float(np.median(values["width"][start:stop])),
                    f"target{target}_mean_component_coverage_90": float(np.mean(values["coverage"][start:stop])),
                    f"target{target}_maximum_step_m": float(np.max(values["step"][start:stop])),
                    f"target{target}_minimum_covariance_eigenvalue_m2": float(np.min(values["covariance_min_eigenvalue"][start:stop])),
                    f"target{target}_finite_uncertainty": bool(np.all(np.isfinite(values["width"][start:stop])) and np.all(values["width"][start:stop] >= 0.0)),
                    f"target{target}_covariance_psd": bool(np.all(np.isfinite(values["covariance"][start:stop])) and np.all(values["covariance_min_eigenvalue"][start:stop] >= -1e-8)),
                })
            row["worst_target_rmse_m"] = max(target_rmses)
            row["mean_target_rmse_m"] = float(np.mean(target_rmses))
            row["worst_target_p90_m"] = max(target_p90s)
            row["admitted_identity"] = row["identity_match_fraction"] >= args.minimum_identity_fraction
            row["admitted_jump"] = max(row["target1_maximum_step_m"], row["target2_maximum_step_m"]) <= args.maximum_step_m
            row["admitted_error"] = (
                row["worst_target_rmse_m"] <= args.maximum_target_rmse_m
                and row["worst_target_p90_m"] <= args.maximum_target_p90_m
            )
            row["admitted_uncertainty"] = bool(row["target1_finite_uncertainty"] and row["target2_finite_uncertainty"])
            row["admitted_covariance"] = bool(row["target1_covariance_psd"] and row["target2_covariance_psd"])
            row["ranking_score"] = row["worst_target_rmse_m"] + 0.15 * row["mean_target_rmse_m"] + 0.05 * row["worst_target_p90_m"]
            candidates.append(row)

    candidates.sort(key=lambda row: (int(row["length_frames"]), float(row["ranking_score"])))
    write_csv(args.output / "dual_long_window_candidates.csv", candidates)

    eligible = [
        row for row in candidates
        if int(row["length_frames"]) == args.selected_length
        and bool(row["admitted_identity"])
        and bool(row["admitted_jump"])
        and bool(row["admitted_error"])
        and bool(row["admitted_uncertainty"])
        and bool(row["admitted_covariance"])
    ]
    if not eligible:
        raise RuntimeError(f"no admitted {args.selected_length}-frame window")
    selected = min(eligible, key=lambda row: float(row["ranking_score"]))
    selected_start = int(selected["start_time_s"])
    selected_end = int(selected["end_time_s"])
    selected_mask = (times >= selected_start) & (times <= selected_end)

    time_rows: list[dict[str, object]] = []
    for index in np.flatnonzero(selected_mask):
        row: dict[str, object] = {
            "time_s": int(times[index]),
            "elapsed_s": int(times[index] - selected_start),
            "offline_identity_match": bool(identity_correct[index]),
            "estimated_target_separation_m": float(estimated_separation[index]),
            "truth_target_separation_m": float(truth_separation[index]),
        }
        for target in TARGETS:
            values = prepared[target]
            for dimension, label in enumerate(("east", "north", "up")):
                row[f"target{target}_gps_{label}_m"] = float(values["truth"][index, dimension])
                row[f"target{target}_apce_{label}_median_5seeds_m"] = float(values["position"][index, dimension])
                row[f"target{target}_frontend_{label}_m"] = float(values["frontend"][index, dimension])
            row[f"target{target}_apce_error_m"] = float(values["error"][index])
            row[f"target{target}_frontend_error_m"] = float(values["frontend_error"][index])
            row[f"target{target}_apce_median_marginal_width_m"] = float(values["width"][index])
            row[f"target{target}_apce_mean_component_coverage_90"] = float(values["coverage"][index])
            row[f"target{target}_observation_covariance_min_eigenvalue_m2"] = float(values["covariance_min_eigenvalue"][index])
        time_rows.append(row)
    write_csv(args.output / "dual_long_window_selected_timeseries.csv", time_rows)

    top_by_length = {}
    for length in sorted(set(args.lengths)):
        top_by_length[str(length)] = [
            row for row in candidates if int(row["length_frames"]) == length
        ][:10]
    manifest = {
        "selection_status": "post-hoc showcase segment selected with offline GPS scoring",
        "gps_role": "offline window scoring and identity audit only; no GPS enters APCE",
        "selection_objective": "minimize worst-target long-window RMSE with smaller penalties on mean RMSE and worst-target P90",
        "admission": {
            "identity_match_fraction_min": args.minimum_identity_fraction,
            "maximum_one_second_step_m": args.maximum_step_m,
            "maximum_target_rmse_m": args.maximum_target_rmse_m,
            "maximum_target_p90_m": args.maximum_target_p90_m,
            "finite_nonnegative_uncertainty_required": True,
            "frontend_observation_covariance_psd_required": True,
            "contiguous_one_second_updates_required": True,
        },
        "selected": selected,
        "top_10_by_length": top_by_length,
        "formal_root": str(args.formal_root),
        "frontend_root": str(args.frontend_root),
        "source_hashes": {
            str(path): sha256(path)
            for target in TARGETS
            for path in tracks[target]["paths"]
        },
        "seeds": {str(target): tracks[target]["seeds"] for target in TARGETS},
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    manifest_path = args.output / "dual_long_window_selection.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected": selected, "manifest": str(manifest_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
