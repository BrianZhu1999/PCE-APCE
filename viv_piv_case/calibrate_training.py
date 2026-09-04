"""Distributed, training-only calibration of VIV-PIV uncertainty controls.

The runner deliberately never opens an external test result.  Every job uses
one leave-one-training-case-out model fold, and the held-out training case is
excluded from POD/DMDc fitting, candidate construction, observation-noise
statistics and full observation covariance estimation.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import itertools
import json
import os
import pathlib
import subprocess
import sys
from collections import defaultdict
from typing import Any

import numpy as np

from .common import load_config, write_json


DEFAULT_GRID = {
    "initial_ensemble_scale": [0.10, 0.20, 0.30],
    "process_noise_scale": [0.50, 1.00, 2.00],
    "state_inflation": [1.00, 1.20, 1.50, 2.00],
    "observation_covariance_shrinkage": [0.05, 0.18, 0.35],
}


def _parse_list(value: str | None, fallback: list[str]) -> list[str]:
    if value is None:
        return list(fallback)
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("Expected at least one value")
    return result


def _parse_seeds(value: str) -> list[int]:
    result: list[int] = []
    for token in _parse_list(value, []):
        if "-" in token:
            first, last = token.split("-", 1)
            result.extend(range(int(first), int(last) + 1))
        else:
            result.append(int(token))
    return sorted(set(result))


def _normalise_case_ids(values: list[str]) -> list[str]:
    return [value.replace(",", "").zfill(4)[-4:] for value in values]


def _grid_from_json(path: pathlib.Path | None) -> list[dict[str, float]]:
    if path is None:
        keys = list(DEFAULT_GRID)
        return [
            {key: float(value) for key, value in zip(keys, values)}
            for values in itertools.product(*(DEFAULT_GRID[key] for key in keys))
        ]
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("--configurations-json must contain a JSON list")
    expected = set(DEFAULT_GRID)
    rows: list[dict[str, float]] = []
    for row in raw:
        if set(row) != expected:
            raise ValueError(f"Each configuration must contain exactly {sorted(expected)}")
        rows.append({key: float(row[key]) for key in DEFAULT_GRID})
    return rows


def _configuration_id(parameters: dict[str, float]) -> str:
    canonical = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _calibration_loss(row: dict[str, Any]) -> float:
    # CRPS rewards calibrated sharp forecasts. The two modest terms prevent a
    # nominal 90% interval from being achieved solely by indiscriminate spread.
    return (
        float(row["normalized_crps"])
        + 0.50 * abs(float(row["coverage_90"]) - 0.90)
        + 0.05 * float(row["normalized_interval_width_90"])
        + 0.10 * float(row["evaluation_nrmse"])
    )


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["configuration_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Training-only uncertainty calibration for PCE/APCE.")
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--stage", required=True, help="Stable output label, e.g. coarse or validation.")
    parser.add_argument("--layout", default="20x40")
    parser.add_argument("--folds", default=None, help="Comma-separated configured training cases.")
    parser.add_argument("--methods", default="pce,apce")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--configurations-json", type=pathlib.Path, default=None)
    parser.add_argument("--evidence-window-frames", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    all_training_cases = list(config["train_cases"])
    folds = _normalise_case_ids(_parse_list(args.folds, all_training_cases))
    unknown = sorted(set(folds) - set(all_training_cases))
    if unknown:
        raise ValueError(f"Calibration folds must be configured training cases: {unknown}")
    methods = _parse_list(args.methods, [])
    unknown_methods = sorted(set(methods) - {"pce", "apce"})
    if unknown_methods:
        raise ValueError(f"Only PCE/APCE are valid calibration methods: {unknown_methods}")
    seeds = _parse_seeds(args.seeds)
    gpus = _parse_list(args.gpus, [])
    parameters = _grid_from_json(args.configurations_json)
    if not seeds or not gpus or not parameters:
        raise ValueError("Calibration needs configurations, seeds and GPUs")
    if args.evidence_window_frames < 1:
        raise ValueError("--evidence-window-frames must be at least one")

    root = pathlib.Path(config["output_root"]) / "calibration" / "uncertainty" / args.stage
    trial_root = root / "trials"
    log_root = root / "logs"
    trial_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    tasks = [
        (params, fold, method, seed)
        for params in parameters
        for fold in folds
        for method in methods
        for seed in seeds
    ]

    def worker(gpu: str, assigned: list[tuple[dict[str, float], str, str, int]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        for params, fold, method, seed in assigned:
            configuration_id = _configuration_id(params)
            variant = f"rank{int(config['rank'])}_stride1_calibration_holdout{fold}"
            output = trial_root / f"{configuration_id}_{fold}_{method}_seed{seed:03d}.json"
            if args.skip_existing and output.exists():
                payload = json.loads(output.read_text(encoding="utf-8"))
                if payload.get("valid") and payload.get("status") == "completed":
                    rows.append({"configuration_id": configuration_id, "fold": fold, "method": method, "seed": seed, "gpu": gpu, "status": "skipped_completed", "path": str(output)})
                    continue
            command = [
                sys.executable, "-m", "viv_piv_case.calibration_worker", "--config", str(args.config),
                "--case", fold, "--method", method, "--seed", str(seed), "--variant", variant,
                "--layout", args.layout, "--initial-ensemble-scale", str(params["initial_ensemble_scale"]),
                "--process-noise-scale", str(params["process_noise_scale"]), "--state-inflation", str(params["state_inflation"]),
                "--covariance-shrinkage", str(params["observation_covariance_shrinkage"]),
                "--evidence-window-frames", str(args.evidence_window_frames),
                "--output", str(output), "--device", "cpu",
            ]
            log = log_root / f"{configuration_id}_{fold}_{method}_seed{seed:03d}.log"
            with log.open("w", encoding="utf-8") as handle:
                completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=environment, check=False)
            rows.append({
                "configuration_id": configuration_id, "fold": fold, "method": method, "seed": seed,
                "gpu": gpu, "status": "completed" if completed.returncode == 0 else "failed",
                "returncode": completed.returncode, "path": str(output), "log": str(log),
            })
        return rows

    partitions = [tasks[index::len(gpus)] for index in range(len(gpus))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(worker, gpu, partition) for gpu, partition in zip(gpus, partitions)]
        launch_rows = [row for future in futures for row in future.result()]
    _write_csv(root / "launch_manifest.csv", launch_rows)

    result_rows: list[dict[str, Any]] = []
    for path in sorted(trial_root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("valid") and payload.get("status") == "completed":
            row = dict(payload)
            row["path"] = str(path)
            row["configuration_id"] = _configuration_id({key: float(payload[key]) for key in DEFAULT_GRID})
            row["calibration_loss"] = _calibration_loss(row)
            result_rows.append(row)
    _write_csv(root / "trial_metrics.csv", result_rows)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in result_rows:
        grouped[str(row["configuration_id"])].append(row)
    ranking: list[dict[str, Any]] = []
    expected = len(folds) * len(methods) * len(seeds)
    for configuration_id, rows in grouped.items():
        parameters_row = {key: float(rows[0][key]) for key in DEFAULT_GRID}
        ranking.append({
            "configuration_id": configuration_id,
            **parameters_row,
            "expected_trials": expected,
            "completed_trials": len(rows),
            "complete": len(rows) == expected,
            "mean_calibration_loss": float(np.mean([row["calibration_loss"] for row in rows])),
            "mean_nrmse": float(np.mean([row["evaluation_nrmse"] for row in rows])),
            "mean_crps": float(np.mean([row["normalized_crps"] for row in rows])),
            "mean_coverage_90": float(np.mean([row["coverage_90"] for row in rows])),
            "mean_interval_width_90": float(np.mean([row["normalized_interval_width_90"] for row in rows])),
        })
    ranking.sort(key=lambda row: (not bool(row["complete"]), float(row["mean_calibration_loss"])))
    _write_csv(root / "configuration_ranking.csv", ranking)
    manifest = {
        "protocol": "strict training-only leave-one-case-out uncertainty calibration",
        "stage": args.stage,
        "layout": args.layout,
        "folds": folds,
        "methods": methods,
        "seeds": seeds,
        "gpus": gpus,
        "parameters": parameters,
        "task_count": len(tasks),
        "evidence_window_frames": int(args.evidence_window_frames),
        "completed_valid_trials": len(result_rows),
        "failed_launches": sum(row["status"] == "failed" for row in launch_rows),
        "selection_objective": "mean CRPS + 0.50*|coverage_90-0.90| + 0.05*interval_width_90 + 0.10*nRMSE",
        "ranking": str(root / "configuration_ranking.csv"),
        "trials": str(root / "trial_metrics.csv"),
    }
    write_json(root / "calibration_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False))
    if manifest["failed_launches"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
