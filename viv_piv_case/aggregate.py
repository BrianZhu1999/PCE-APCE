"""Aggregate VIV-PIV runs and perform leakage/finite-value audits.

The aggregator is deliberately independent of the plotting script.  It never
opens the held-out full fields; it only reads run JSON and the compact traces
written by ``run_case``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


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
    run_manifest: list[dict[str, Any]] = []
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
        run_manifest.append({
            "run_id": payload.get("run_id"),
            "case_id": payload.get("case_id"),
            "method": payload.get("method"),
            "seed": payload.get("seed"),
            "sensor_density_points": payload.get("sensor_density_points", 40),
            "uses_known_cylinder_displacement_input": payload.get("uses_known_cylinder_displacement_input", True),
            "status": payload.get("status"),
            "valid": payload.get("valid"),
            "json_path": str(path),
            "json_sha256": sha256_file(path),
            "trace_path": str(trace_path),
            "trace_exists": trace_exists,
            "trace_sha256": payload.get("trace_sha256"),
            "local_grid_stable": payload.get("local_grid_stable"),
        })
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
                        "early_nonphysical_collapse": early_max > 0.95,
                        "interpretability_gate_pass": significant_fraction >= 0.30 and early_max <= 0.95,
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
    _write_csv(summary_dir / "identifiability.csv", identifiability_rows, ["run_id", "case_id", "method", "seed", "significant_separation_fraction", "median_separation_ratio", "mean_erasure_ratio", "early_max_weight", "early_nonphysical_collapse", "interpretability_gate_pass"])
    _write_csv(summary_dir / "run_manifest.csv", run_manifest, list(run_manifest[0].keys()) if run_manifest else ["run_id"])

    model_manifest_path = output_root / "models" / variant / "model_manifest.json"
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8")) if model_manifest_path.exists() else {}
    train_cases = {str(x) for x in config["train_cases"]}
    test_cases = {str(x) for x in config["test_cases"]}
    candidate_cases = {str(x) for x in model_manifest.get("candidate_cases", [])}
    leakage_checks = {
        "train_test_disjoint": train_cases.isdisjoint(test_cases),
        "candidate_cases_subset_train": candidate_cases.issubset(train_cases),
        "model_manifest_confirms_test_data_not_used_for_fit": model_manifest.get("test_data_used_for_model_fit") is False,
        "all_result_cases_held_out": all(str(r.get("case_id")) in test_cases for r in records),
        "all_weights_normalized": True,
        "all_finite_trace_values": True,
    }
    failures: list[str] = []
    for record in records:
        trace_path_str = next((m["trace_path"] for m in run_manifest if m["run_id"] == record["run_id"]), None)
        path = pathlib.Path(trace_path_str) if trace_path_str else None
        if path is not None and path.exists():
            with np.load(path, allow_pickle=False) as trace:
                for key in ("weights", "scores", "entropy", "separation", "erasure"):
                    if key in trace.files and not np.isfinite(trace[key]).all():
                        leakage_checks["all_finite_trace_values"] = False
                        failures.append(f"nonfinite:{record['run_id']}:{key}")
                if "weights" in trace.files and trace["weights"].size:
                    sums = trace["weights"].sum(axis=1)
                    if not np.allclose(sums, 1.0, atol=1e-8):
                        leakage_checks["all_weights_normalized"] = False
                        failures.append(f"weight_sum:{record['run_id']}")
    for name, passed in leakage_checks.items():
        if not passed:
            failures.append(name)
    audit = {
        "protocol": "VIV-PIV PCE/APCE external-case audit",
        "variant": variant,
        "train_cases": sorted(train_cases),
        "test_cases": sorted(test_cases),
        "candidate_cases": sorted(candidate_cases),
        "result_count": len(records),
        "checks": leakage_checks,
        "failures": failures,
        "pass": not failures,
        "model_manifest": str(model_manifest_path),
        "summary_metrics": str(summary_dir / "summary_metrics.csv"),
    }
    write_json(summary_dir / "leakage_audit.json", _json_ready(audit))
    write_json(summary_dir / "summary_metrics.json", _json_ready(records))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate VIV-PIV PCE/APCE runs and audit leakage.")
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--variant", default=None)
    args = parser.parse_args()
    result = aggregate(args.config, args.variant)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
