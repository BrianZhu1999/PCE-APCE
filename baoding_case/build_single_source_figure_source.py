#!/usr/bin/env python3
"""Build the single-source CSV interface used by the Baoding composite figure."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


CONFIG_KEYS = (
    "ensemble_size",
    "q_min_accel_mps2",
    "q_max_accel_mps2",
    "position_init_std_m",
    "velocity_init_std_mps",
    "observation_covariance_scale",
    "turn_rate_radps",
)


def load_runs(root: Path, method: str) -> dict[str, object]:
    paths = sorted((root / "runs").glob(f"{method}_seed_*.json"))
    if len(paths) != 5:
        raise RuntimeError(f"expected five {method} runs, found {len(paths)}")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    records = [payload["records"] for payload in payloads]
    times = np.asarray([float(row["time_s"]) for row in records[0]], dtype=float)
    if not all(np.allclose(times, [float(row["time_s"]) for row in rows]) for rows in records[1:]):
        raise RuntimeError("run time grids disagree")
    positions = np.asarray(
        [[[float(row[name]) for name in ("px", "py", "pz")] for row in rows] for rows in records],
        dtype=float,
    )
    widths = np.asarray([[float(row["interval_width_m"]) for row in rows] for rows in records], dtype=float)
    coverages = np.asarray([[float(row["coverage_90"]) for row in rows] for rows in records], dtype=float)
    seeds = [int(payload["seed"]) for payload in payloads]
    configurations = [{key: payload[key] for key in CONFIG_KEYS} for payload in payloads]
    if any(configuration != configurations[0] for configuration in configurations[1:]):
        raise RuntimeError(f"{method} run configurations disagree")
    return {
        "times": times,
        "position": np.median(positions, axis=0),
        "width": np.median(widths, axis=0),
        "coverage_mean": float(np.mean(coverages)),
        "seeds": seeds,
        "configuration": configurations[0],
        "paths": [str(path) for path in paths],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend", type=Path, required=True)
    parser.add_argument("--apce-runs", type=Path, required=True)
    parser.add_argument("--pce-runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    truth_rows = read_csv(args.frontend / "gps_truth.csv")
    truth = {float(row["time_s"]): np.asarray([float(row[key]) for key in ("px", "py", "pz")]) for row in truth_rows}
    apce_runs = load_runs(args.apce_runs, "apce")
    pce_runs = load_runs(args.pce_runs, "pce")
    times = np.asarray(apce_runs["times"], dtype=float)
    apce = np.asarray(apce_runs["position"], dtype=float)
    apce_width = np.asarray(apce_runs["width"], dtype=float)
    pce_times = np.asarray(pce_runs["times"], dtype=float)
    pce = np.asarray(pce_runs["position"], dtype=float)
    seeds = list(apce_runs["seeds"])
    if not np.allclose(times, pce_times) or len(times) != 67:
        raise RuntimeError("expected matching 67-frame APCE/PCE grids")
    if apce_runs["configuration"] != pce_runs["configuration"]:
        raise RuntimeError("APCE and PCE figure-source configurations disagree")
    gps = np.asarray([truth[float(time)] for time in times])
    errors = np.linalg.norm(apce - gps, axis=1)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, time in enumerate(times):
        rows.append({
            "time_s": float(time), "elapsed_s": float(time - times[0]),
            "gps_east_m": float(gps[index, 0]), "gps_north_m": float(gps[index, 1]), "gps_up_m": float(gps[index, 2]),
            "apce_east_median_5seeds_m": float(apce[index, 0]), "apce_north_median_5seeds_m": float(apce[index, 1]), "apce_up_median_5seeds_m": float(apce[index, 2]),
            "apce_position_error_of_median_m": float(errors[index]), "apce_median_mean_marginal_90pct_interval_width_m": float(apce_width[index]),
            "pce_east_median_5seeds_m": float(pce[index, 0]), "pce_north_median_5seeds_m": float(pce[index, 1]), "pce_up_median_5seeds_m": float(pce[index, 2]),
        })
    source = args.output / "single_source_figure_source.csv"
    with source.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    configuration = dict(apce_runs["configuration"])
    registry = {
        "figure_contract": {"core_conclusion": "A6 frontend with the established single-source backend preserves the low-error 67-second trajectory; interval coverage is reported without changing the visual scale.", "gps_role": "offline calibration and scoring only"},
        "gps_role": "offline calibration and scoring only; no GPS in held-out update",
        "window": {"start_time_s": float(times[0]), "end_time_s": float(times[-1]), "frames": len(times), "update_interval_s": 1.0, "selection_status": "fixed 67-second window"},
        "configuration": {**configuration, "seeds": seeds, "frontend": "A6 calibration-frozen node confidence and DOA bias compensation"},
        "metrics": {"apce_median_trajectory_rmse_m": float(np.sqrt(np.mean(errors**2))), "apce_median_error_m": float(np.median(errors)), "apce_p90_error_m": float(np.percentile(errors, 90)), "apce_median_width_m": float(np.median(apce_width)), "apce_mean_component_coverage_90": float(apce_runs["coverage_mean"])},
        "sources": {"frontend": str(args.frontend), "apce_runs": str(args.apce_runs), "pce_runs": str(args.pce_runs), "source_csv_sha256": sha256(source)},
    }
    (args.output / "single_source_figure_registry.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(registry, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
