from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
METRICS = (
    "forecast_nrmse",
    "forecast_crps",
    "blackout_alpha_absolute_error",
    "forecast_correlation_error",
    "forecast_coverage_90",
    "forecast_interval_width_90",
    "skill_horizon_time_015",
    "skill_horizon_time_020",
    "skill_horizon_time_030",
)
LOWER_BETTER = {
    "forecast_nrmse",
    "forecast_crps",
    "blackout_alpha_absolute_error",
    "forecast_correlation_error",
    "forecast_interval_width_90",
}
COVERAGE_TARGET = 0.90
BASELINES = ("aug_enkf", "bma_static")
FOCAL_METHODS = ("pce", "apce")
SERIES_KEYS = (
    "forecast_steps",
    "forecast_lead_time",
    "lead_nrmse",
    "lead_crps",
    "lead_coverage_90",
    "lead_interval_width_90",
    "lead_correlation_error",
)


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
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
    run_root = output / "artifacts" / "run_json" / "lorenz96_1024"
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


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fields = sorted({key for row in rows for key in row})
    else:
        fields = fieldnames
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def scalar_source_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        row: dict[str, Any] = {}
        for key, value in record.items():
            if key in SERIES_KEYS:
                continue
            if isinstance(value, (str, int, float)) or value is None:
                row[key] = value
        rows.append(row)
    return rows


def lead_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        steps = record.get("forecast_steps", [])
        leads = record.get("forecast_lead_time", [])
        nrmse = record.get("lead_nrmse", [])
        crps = record.get("lead_crps", [])
        coverage = record.get("lead_coverage_90", [])
        width = record.get("lead_interval_width_90", [])
        corr = record.get("lead_correlation_error", [])
        length = max(len(steps), len(leads), len(nrmse), len(crps), len(coverage), len(width), len(corr))
        for index in range(length):
            rows.append(
                {
                    "case": record.get("case", "lorenz96_1024"),
                    "method": record.get("method", ""),
                    "label": record.get("label", METHOD_LABELS.get(str(record.get("method", "")), "")),
                    "seed": record.get("seed", ""),
                    "obs_interval": record.get("obs_interval", ""),
                    "blackout_start_step": record.get("blackout_start_step", ""),
                    "step": steps[index] if index < len(steps) else "",
                    "lead_time": leads[index] if index < len(leads) else "",
                    "lead_nrmse": nrmse[index] if index < len(nrmse) else "",
                    "lead_crps": crps[index] if index < len(crps) else "",
                    "lead_coverage_90": coverage[index] if index < len(coverage) else "",
                    "lead_interval_width_90": width[index] if index < len(width) else "",
                    "lead_correlation_error": corr[index] if index < len(corr) else "",
                }
            )
    return rows


def summary_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault((int(record["obs_interval"]), str(record["method"])), []).append(record)
    rows: list[dict[str, Any]] = []
    for (interval, method), items in sorted(groups.items()):
        row: dict[str, Any] = {
            "case": "lorenz96_1024",
            "obs_interval": interval,
            "method": method,
            "label": METHOD_LABELS.get(method, method),
            "valid_n": len(items),
            "seeds": ",".join(str(int(item["seed"])) for item in sorted(items, key=lambda item: int(item["seed"]))),
        }
        for metric in METRICS:
            values = np.asarray([as_float(item, metric) for item in items], dtype=float)
            finite = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(np.mean(finite)) if finite.size else float("nan")
            row[f"{metric}_median"] = float(np.median(finite)) if finite.size else float("nan")
            row[f"{metric}_sd"] = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
            low, high = mean_ci(finite, seed=41_000 + interval * 101 + len(rows) * 19 + len(metric))
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        runtime = np.asarray([as_float(item, "runtime_seconds") for item in items], dtype=float)
        row["runtime_seconds_mean"] = float(np.nanmean(runtime)) if runtime.size else float("nan")
        memory = np.asarray([as_float(item, "peak_gpu_memory_mb") for item in items], dtype=float)
        row["peak_gpu_memory_mb_max"] = float(np.nanmax(memory)) if memory.size else float("nan")
        rows.append(row)
    return rows


def paired_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[int, str, int], dict[str, Any]] = {}
    for record in records:
        by_key[(int(record["obs_interval"]), str(record["method"]), int(record["seed"]))] = record
    intervals = sorted({int(record["obs_interval"]) for record in records})
    rows: list[dict[str, Any]] = []
    for interval in intervals:
        seeds = sorted({int(record["seed"]) for record in records if int(record["obs_interval"]) == interval})
        for method in FOCAL_METHODS:
            for baseline in BASELINES:
                paired = [
                    (by_key[(interval, method, seed)], by_key[(interval, baseline, seed)])
                    for seed in seeds
                    if (interval, method, seed) in by_key and (interval, baseline, seed) in by_key
                ]
                if not paired:
                    continue
                row: dict[str, Any] = {
                    "case": "lorenz96_1024",
                    "obs_interval": interval,
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "baseline": baseline,
                    "baseline_label": METHOD_LABELS.get(baseline, baseline),
                    "paired_seed_count": len(paired),
                    "paired_seeds": ",".join(str(int(left["seed"])) for left, _ in paired),
                }
                for metric in METRICS:
                    if metric == "forecast_coverage_90":
                        differences = np.asarray(
                            [
                                abs(float(right[metric]) - COVERAGE_TARGET)
                                - abs(float(left[metric]) - COVERAGE_TARGET)
                                for left, right in paired
                            ],
                            dtype=float,
                        )
                        wins = [
                            abs(float(left[metric]) - COVERAGE_TARGET) < abs(float(right[metric]) - COVERAGE_TARGET)
                            for left, right in paired
                        ]
                        gain_name = "forecast_coverage_90_error_gain"
                    elif metric in LOWER_BETTER:
                        differences = np.asarray(
                            [float(right[metric]) - float(left[metric]) for left, right in paired],
                            dtype=float,
                        )
                        wins = [float(left[metric]) < float(right[metric]) for left, right in paired]
                        gain_name = f"{metric}_gain"
                    else:
                        differences = np.asarray(
                            [float(left[metric]) - float(right[metric]) for left, right in paired],
                            dtype=float,
                        )
                        wins = [float(left[metric]) > float(right[metric]) for left, right in paired]
                        gain_name = f"{metric}_gain"
                    low, high = mean_ci(differences, seed=53_000 + interval * 101 + len(rows) * 23 + len(metric))
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
                "case": payload.get("case", "lorenz96_1024"),
                "method": payload.get("method", ""),
                "seed": payload.get("seed", ""),
                "obs_interval": payload.get("obs_interval", ""),
                "blackout_start_step": payload.get("blackout_start_step", ""),
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
        observed_points = int(records[0]["observed_points"])
        state_dim = int(records[0]["state_dim"])
        blackout_step = int(records[0]["blackout_start_step"])
        steps = int(records[0]["steps"])
    else:
        observed_points, state_dim, blackout_step, steps = 128, 1024, 200, 300
    downsample = state_dim // max(observed_points, 1)
    lines = [
        "# Figure 4 Lorenz-96-1024 blackout forecast aggregate report",
        "",
        f"- Completed valid records: {len(records)}",
        f"- Protocol: D={state_dim}, {observed_points} observed states ({downsample}× spatial downsampling), "
        f"time interval 8 by default, assimilation through step {blackout_step}, free forecast to step {steps}.",
        "- Positive paired gains mean the focal method improves over the baseline; coverage gain uses smaller absolute deviation from 0.90.",
        "",
        "## Method means",
        "",
    ]
    for row in summary:
        lines.append(
            "- time {obs_interval}, {label}: forecast nRMSE {forecast_nrmse_mean:.4g}, "
            "forecast CRPS {forecast_crps_mean:.4g}, blackout alpha MAE {blackout_alpha_absolute_error_mean:.4g}, "
            "corr. error {forecast_correlation_error_mean:.4g}, skill@0.20 {skill_horizon_time_020_mean:.4g} "
            "(n={valid_n})".format(**row)
        )
    lines.extend(["", "## Paired focal comparisons", ""])
    for row in paired:
        lines.append(
            "- time {obs_interval}, {method_label} vs {baseline_label}: forecast nRMSE wins "
            "{forecast_nrmse_win_count}/{paired_seed_count}, CRPS wins {forecast_crps_win_count}/{paired_seed_count}, "
            "alpha-MAE wins {blackout_alpha_absolute_error_win_count}/{paired_seed_count}; "
            "mean forecast nRMSE gain {forecast_nrmse_gain_mean:.4g}, mean CRPS gain {forecast_crps_gain_mean:.4g}, "
            "mean alpha-MAE gain {blackout_alpha_absolute_error_gain_mean:.4g}".format(**row)
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Figure 4 Lorenz-96 blackout forecast runs.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output
    source = output / "source_data"
    records, failures = discover(output)
    scalar_rows = scalar_source_rows(records)
    leads = lead_rows(records)
    summary = summary_rows(records)
    paired = paired_rows(records)
    failed = failure_rows(failures)

    run_csv = source / "lorenz96_1024_blackout_run_source_data.csv"
    lead_csv = source / "lorenz96_1024_blackout_lead_time_source_data.csv"
    summary_csv = source / "lorenz96_1024_blackout_method_summary.csv"
    paired_csv = source / "lorenz96_1024_blackout_paired_gains.csv"
    failure_csv = source / "lorenz96_1024_blackout_failure_records.csv"
    report_path = source / "L96_1024_BLACKOUT_FORECAST_REPORT.md"

    write_csv(run_csv, scalar_rows)
    write_csv(lead_csv, leads)
    write_csv(summary_csv, summary)
    write_csv(paired_csv, paired)
    write_csv(failure_csv, failed)
    report_path.write_text(short_report(records, summary, paired), encoding="utf-8")

    observed_points = int(records[0]["observed_points"]) if records else 128
    state_dim = int(records[0]["state_dim"]) if records else 1024
    manifest = {
        "created_at_unix": time.time(),
        "output": str(output),
        "source_data": str(source),
        "valid_records": len(records),
        "failure_records": len(failed),
        "protocol": {
            "case": "lorenz96_1024",
            "state_dim": state_dim,
            "observed_points": observed_points,
            "spatial_downsampling_factor": state_dim // max(observed_points, 1),
            "obs_intervals": sorted({int(row["obs_interval"]) for row in records}),
            "blackout_start_steps": sorted({int(row["blackout_start_step"]) for row in records}),
            "methods": sorted({str(row["method"]) for row in records}),
            "metrics": list(METRICS),
            "paired_gain_definition": "positive means the focal method improves over the baseline; coverage uses 0.90 absolute-error reduction",
        },
        "files": {
            "run_source_data": str(run_csv),
            "run_source_data_sha256": file_sha256(run_csv),
            "lead_time_source_data": str(lead_csv),
            "lead_time_source_data_sha256": file_sha256(lead_csv),
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
