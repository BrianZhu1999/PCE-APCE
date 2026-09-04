from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np


METHOD_LABELS = {
    "aug_enkf": "Aug-EnKF",
    "bma_static": "BMA",
    "pce": "PCE",
    "apce": "APCE",
}
METRICS = ("nrmse", "crps", "alpha_absolute_error", "coverage_90", "interval_width_90")
FOCAL_METHODS = ("pce", "apce")
BASELINES = ("aug_enkf", "bma_static")


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_sha256(rows: list[dict[str, Any]]) -> str:
    encoded = json.dumps(clean_json(rows), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["_json_path"] = str(path)
    return payload


def discover(output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_root = output / "artifacts" / "run_json" / "kolmogorov64"
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if not run_root.exists():
        return completed, failures
    for path in sorted(run_root.rglob("*.json")):
        payload = read_json(path)
        if payload.get("status") == "completed" and payload.get("numerical_status") == "valid":
            completed.append(payload)
        else:
            failures.append(payload)
    return completed, failures


def as_float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def mean_ci(values: np.ndarray, seed: int, n_bootstrap: int = 5000) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(n_bootstrap, values.size))
    draws = values[indices].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summary_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups: dict[tuple[float, int, int, int, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (
            float(record["reynolds"]),
            int(record["forcing_wavenumber"]),
            int(record["sensor_grid"]),
            int(record["obs_interval"]),
            str(record["method"]),
        )
        groups.setdefault(key, []).append(record)
    for (reynolds, forcing_wavenumber, sensor_grid, interval, method), items in sorted(groups.items()):
        row: dict[str, Any] = {
            "case": "kolmogorov64",
            "reynolds": reynolds,
            "forcing_wavenumber": forcing_wavenumber,
            "sensor_grid": sensor_grid,
            "observed_points": sensor_grid * sensor_grid,
            "obs_interval": interval,
            "method": method,
            "label": METHOD_LABELS.get(method, method),
            "valid_n": len(items),
            "seeds": ",".join(str(int(item["seed"])) for item in sorted(items, key=lambda x: int(x["seed"]))),
        }
        for metric in METRICS:
            values = np.asarray([as_float(item, metric) for item in items], dtype=float)
            finite = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(np.mean(finite)) if finite.size else float("nan")
            row[f"{metric}_median"] = float(np.median(finite)) if finite.size else float("nan")
            row[f"{metric}_sd"] = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
            low, high = mean_ci(finite, seed=17_000 + interval * 101 + len(rows) * 13 + len(metric))
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        runtime = np.asarray([as_float(item, "runtime_seconds") for item in items], dtype=float)
        row["runtime_seconds_mean"] = float(np.nanmean(runtime)) if runtime.size else float("nan")
        memory = np.asarray([as_float(item, "peak_gpu_memory_mb") for item in items], dtype=float)
        row["peak_gpu_memory_mb_max"] = float(np.nanmax(memory)) if memory.size else float("nan")
        rows.append(row)
    return rows


def paired_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[float, int, int, int, str, int], dict[str, Any]] = {}
    for record in records:
        by_key[
            (
                float(record["reynolds"]),
                int(record["forcing_wavenumber"]),
                int(record["sensor_grid"]),
                int(record["obs_interval"]),
                str(record["method"]),
                int(record["seed"]),
            )
        ] = record
    groups: dict[tuple[float, int, int, int], list[int]] = {}
    for record in records:
        key = (
            float(record["reynolds"]),
            int(record["forcing_wavenumber"]),
            int(record["sensor_grid"]),
            int(record["obs_interval"]),
        )
        groups.setdefault(key, []).append(int(record["seed"]))
    rows: list[dict[str, Any]] = []
    for (reynolds, forcing_wavenumber, sensor_grid, interval), seeds in sorted(groups.items()):
        unique_seeds = sorted(set(seeds))
        for method in FOCAL_METHODS:
            for baseline in BASELINES:
                paired = [
                    (by_key[(reynolds, forcing_wavenumber, sensor_grid, interval, method, seed)],
                     by_key[(reynolds, forcing_wavenumber, sensor_grid, interval, baseline, seed)])
                    for seed in unique_seeds
                    if (reynolds, forcing_wavenumber, sensor_grid, interval, method, seed) in by_key
                    and (reynolds, forcing_wavenumber, sensor_grid, interval, baseline, seed) in by_key
                ]
                if not paired:
                    continue
                row: dict[str, Any] = {
                    "case": "kolmogorov64",
                    "reynolds": reynolds,
                    "forcing_wavenumber": forcing_wavenumber,
                    "sensor_grid": sensor_grid,
                    "observed_points": sensor_grid * sensor_grid,
                    "obs_interval": interval,
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "baseline": baseline,
                    "baseline_label": METHOD_LABELS.get(baseline, baseline),
                    "paired_seed_count": len(paired),
                    "paired_seeds": ",".join(str(int(left["seed"])) for left, _ in paired),
                }
                for metric in METRICS:
                    if metric == "coverage_90":
                        differences = np.asarray(
                            [
                                abs(float(right[metric]) - 0.90) - abs(float(left[metric]) - 0.90)
                                for left, right in paired
                            ],
                            dtype=float,
                        )
                        wins = [abs(float(left[metric]) - 0.90) < abs(float(right[metric]) - 0.90) for left, right in paired]
                        gain_name = "coverage_90_error_gain"
                    else:
                        differences = np.asarray(
                            [float(right[metric]) - float(left[metric]) for left, right in paired],
                            dtype=float,
                        )
                        wins = [float(left[metric]) < float(right[metric]) for left, right in paired]
                        gain_name = f"{metric}_gain"
                    low, high = mean_ci(differences, seed=29_000 + interval * 101 + len(rows) * 17 + len(metric))
                    row[f"{gain_name}_mean"] = float(np.mean(differences))
                    row[f"{gain_name}_median"] = float(np.median(differences))
                    row[f"{gain_name}_ci95_low"] = low
                    row[f"{gain_name}_ci95_high"] = high
                    row[f"{metric}_win_count"] = int(sum(wins))
                    row[f"{metric}_loss_count"] = int(len(wins) - sum(wins))
                rows.append(row)
    return rows


def failure_rows(failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in failures:
        rows.append(
            {
                "case": payload.get("case", "kolmogorov64"),
                "method": payload.get("method", ""),
                "seed": payload.get("seed", ""),
                "reynolds": payload.get("reynolds", ""),
                "forcing_wavenumber": payload.get("forcing_wavenumber", ""),
                "sensor_grid": payload.get("sensor_grid", ""),
                "obs_interval": payload.get("obs_interval", ""),
                "status": payload.get("status", ""),
                "numerical_status": payload.get("numerical_status", ""),
                "error_type": payload.get("error_type", ""),
                "error": payload.get("error", ""),
                "json_path": payload.get("_json_path", ""),
            }
        )
    return rows


def short_report(records: list[dict[str, Any]], summary: list[dict[str, Any]], paired: list[dict[str, Any]]) -> str:
    if records:
        reynolds = float(records[0]["reynolds"])
        forcing_wavenumber = int(records[0]["forcing_wavenumber"])
        sensor_grid = int(records[0]["sensor_grid"])
        observed_points = sensor_grid * sensor_grid
        state_dim = int(records[0]["state_dim"])
    else:
        reynolds = 575.0
        forcing_wavenumber = 4
        sensor_grid = 16
        observed_points = 256
        state_dim = 4096
    lines = [
        "# Figure 4 KOL-64 sparse 2-D aggregate report",
        "",
        f"- Completed valid records: {len(records)}",
        f"- Protocol: Re={reynolds}, k={forcing_wavenumber}, state_dim={state_dim}, sensor grid {sensor_grid}x{sensor_grid} ({observed_points} sensors), methods Aug-EnKF/BMA/PCE/APCE.",
        "- Positive paired gains mean the focal method has lower error than the baseline; coverage gain uses smaller absolute deviation from 0.90.",
        "",
        "## Method means",
        "",
    ]
    for row in summary:
        lines.append(
            "- Re {reynolds:.0f}, k {forcing_wavenumber}, obs {sensor_grid}x{sensor_grid}, time {obs_interval}, {label}: "
            "nRMSE {nrmse_mean:.4g}, CRPS {crps_mean:.4g}, alpha MAE {alpha_absolute_error_mean:.4g}, "
            "coverage {coverage_90_mean:.4g}, width {interval_width_90_mean:.4g} (n={valid_n})".format(**row)
        )
    lines.extend(["", "## Paired focal comparisons", ""])
    for row in paired:
        lines.append(
            "- Re {reynolds:.0f}, k {forcing_wavenumber}, obs {sensor_grid}x{sensor_grid}, time {obs_interval}, "
            "{method_label} vs {baseline_label}: nRMSE wins {nrmse_win_count}/{paired_seed_count}, "
            "CRPS wins {crps_win_count}/{paired_seed_count}, alpha-MAE wins {alpha_absolute_error_win_count}/{paired_seed_count}; "
            "mean nRMSE gain {nrmse_gain_mean:.4g}, mean CRPS gain {crps_gain_mean:.4g}, "
            "mean alpha-MAE gain {alpha_absolute_error_gain_mean:.4g}".format(**row)
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Figure 4 KOL-64 sparse 2-D runs.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output
    source = output / "source_data"
    records, failures = discover(output)
    summary = summary_rows(records)
    paired = paired_rows(records)
    failed = failure_rows(failures)
    run_csv = source / "kolmogorov64_run_source_data.csv"
    summary_csv = source / "kolmogorov64_method_summary.csv"
    paired_csv = source / "kolmogorov64_paired_gains.csv"
    failure_csv = source / "kolmogorov64_failure_records.csv"
    report_path = source / "KOL64_SPARSE2D_REPORT.md"
    write_csv(run_csv, records)
    write_csv(summary_csv, summary)
    write_csv(paired_csv, paired)
    write_csv(failure_csv, failed)
    report_path.write_text(short_report(records, summary, paired), encoding="utf-8")
    manifest = {
        "created_at_unix": time.time(),
        "output": str(output),
        "source_data": str(source),
        "valid_records": len(records),
        "failure_records": len(failed),
        "protocol": {
            "case": "kolmogorov64",
            "reynolds_values": sorted({float(row["reynolds"]) for row in records}),
            "forcing_wavenumbers": sorted({int(row["forcing_wavenumber"]) for row in records}),
            "sensor_grids": sorted({int(row["sensor_grid"]) for row in records}),
            "observed_points": sorted({int(row["observed_points"]) for row in records}),
            "obs_intervals": sorted({int(row["obs_interval"]) for row in records}),
            "methods": sorted({str(row["method"]) for row in records}),
            "metrics": list(METRICS),
            "paired_gain_definition": "positive means the focal method improves over the baseline; coverage uses 0.90 absolute-error reduction",
        },
        "files": {
            "run_source_data": str(run_csv),
            "run_source_data_sha256": file_sha256(run_csv),
            "method_summary": str(summary_csv),
            "method_summary_sha256": file_sha256(summary_csv),
            "paired_gains": str(paired_csv),
            "paired_gains_sha256": file_sha256(paired_csv),
            "failure_records": str(failure_csv),
            "failure_records_sha256": file_sha256(failure_csv),
            "report": str(report_path),
            "report_sha256": file_sha256(report_path),
            "aggregate_sha256": aggregate_sha256(records),
        },
    }
    manifest_path = source / "manifest.json"
    manifest_path.write_text(json.dumps(clean_json(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(clean_json(manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
