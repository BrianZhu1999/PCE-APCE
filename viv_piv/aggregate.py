"""Aggregate VIV-PIV reconstruction metrics from run outputs."""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
from typing import Any

import numpy as np

from .common import load_config, write_json


METRIC_FIELDS = (
    "evaluation_nrmse",
    "normalized_crps",
    "coverage_90",
    "normalized_interval_width_90",
    "blackout_mean_nrmse",
    "full_field_physical_nrmse",
    "full_field_fluctuation_nrmse",
    "unobserved_full_field_physical_nrmse",
    "unobserved_full_field_fluctuation_nrmse",
    "observed_scalar_dimensions_excluded",
    "kinetic_energy_nrmse",
    "kinetic_energy_correlation",
    "kinetic_energy_peak_relative_error",
    "effective_candidate_count",
    "final_weight_entropy",
    "candidate_count",
    "local_grid_stable",
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(config_path: pathlib.Path, variant: str | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    output_root = pathlib.Path(config["output_root"])
    variant = variant or f"rank{int(config['rank'])}_stride1"
    run_root = output_root / "runs" / variant
    records: list[dict[str, Any]] = []
    blackout_rows: list[dict[str, Any]] = []
    identifiability_rows: list[dict[str, Any]] = []
    for path in sorted((run_root / "runs").glob("viv_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        record = {
            "run_id": payload.get("run_id"),
            "case_id": payload.get("case_id"),
            "method": payload.get("method"),
            "seed": payload.get("seed"),
            "sensor_density_points": payload.get("sensor_density_points", 40),
            "uses_known_cylinder_displacement_input": payload.get("uses_known_cylinder_displacement_input", True),
        }
        for field in METRIC_FIELDS:
            value = payload.get(field)
            record[field] = value
        record.update({"status": payload.get("status"), "valid": payload.get("valid"), "device": payload.get("device")})
        records.append(record)
        trace_path = pathlib.Path(payload["trace_path"]) if payload.get("trace_path") else run_root / "traces" / f"{payload.get('run_id')}.npz"
        trace_exists = trace_path.exists()
        if trace_exists:
            with np.load(trace_path, allow_pickle=False) as trace:
                raw = str(trace["blackout_rows_json"].item()) if "blackout_rows_json" in trace.files else "[]"
                if payload.get("method") in {"pce", "apce"} and "separation" in trace.files and trace["separation"].size:
                    separation = np.asarray(trace["separation"], dtype=float)
                    weights = np.asarray(trace["weights"], dtype=float)
                    warmup = int(round(float(config["warmup_seconds"]) / float(config["time_step_s"])))
                    early = weights[warmup : min(warmup + 5, weights.shape[0])]
                    early_max = float(early.max()) if early.size else float(weights[:1].max())
                    significant_fraction = float(np.mean(separation > 1.0))
                    identifiability_rows.append({
                        "run_id": payload.get("run_id"),
                        "case_id": payload.get("case_id"),
                        "method": payload.get("method"),
                        "seed": payload.get("seed"),
                        "significant_separation_fraction": significant_fraction,
                        "median_separation_ratio": float(np.median(separation)),
                        "mean_erasure_ratio": float(np.mean(trace["erasure"])) if "erasure" in trace.files else np.nan,
                        "early_max_weight": early_max,
                    })
            try:
                rows = json.loads(raw)
            except json.JSONDecodeError:
                rows = []
            for row in rows:
                blackout_rows.append({
                    "run_id": payload.get("run_id"),
                    "case_id": payload.get("case_id"),
                    "method": payload.get("method"),
                    "seed": payload.get("seed"),
                    **row,
                })

    summary_dir = output_root / "summaries" / variant
    fields = ["run_id", "case_id", "method", "seed", "sensor_density_points", "uses_known_cylinder_displacement_input", *METRIC_FIELDS, "status", "valid", "device"]
    _write_csv(summary_dir / "summary_metrics.csv", records, fields)
    _write_csv(summary_dir / "blackout_metrics.csv", blackout_rows, ["run_id", "case_id", "method", "seed", "origin_index", "origin_time_s", "horizon_s", "evaluation_nrmse"])
    _write_csv(summary_dir / "identifiability.csv", identifiability_rows, ["run_id", "case_id", "method", "seed", "significant_separation_fraction", "median_separation_ratio", "mean_erasure_ratio", "early_max_weight"])
    model_manifest_path = output_root / "models" / variant / "model_manifest.json"
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8")) if model_manifest_path.exists() else {}
    train_cases = {str(x) for x in config["train_cases"]}
    test_cases = {str(x) for x in config["test_cases"]}
    candidate_cases = {str(x) for x in model_manifest.get("candidate_cases", [])}
    summary = {
        "protocol": "VIV-PIV held-out reconstruction",
        "variant": variant,
        "train_cases": sorted(train_cases),
        "test_cases": sorted(test_cases),
        "candidate_cases": sorted(candidate_cases),
        "result_count": len(records),
        "summary_metrics": str(summary_dir / "summary_metrics.csv"),
    }
    write_json(summary_dir / "summary.json", _json_ready(summary))
    write_json(summary_dir / "summary_metrics.json", _json_ready(records))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate VIV-PIV PCE/APCE runs.")
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--variant", default=None)
    args = parser.parse_args()
    result = aggregate(args.config, args.variant)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
