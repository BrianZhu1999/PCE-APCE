from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

METHOD_LABELS = {"aug_enkf": "Aug-EnKF", "bma_static": "BMA", "pce": "PCE", "apce": "APCE"}
METHODS = ("aug_enkf", "bma_static", "pce", "apce")
SENSOR_GRIDS = (16, 8)
SEEDS = (2026081600, 2026081601, 2026081602, 2026081603, 2026081604)
METRICS = ("nrmse", "crps", "alpha_absolute_error", "reynolds_relative_error", "coverage_90_error", "interval_width_90")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["_json_path"] = str(path)
    return payload


def discover(output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    root = output / "artifacts" / "run_json"
    if root.exists():
        for path in sorted(root.rglob("*.json")):
            row = read_json(path)
            if row.get("status") == "completed" and row.get("numerical_status") == "valid":
                completed.append(row)
            else:
                failures.append(row)
    return completed, failures


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trace_diagnostics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "trace_exists": False,
            "trace_sha256": "",
            "max_abs_divergence_mean_trace": float("nan"),
            "max_abs_mean_state_trace": float("nan"),
        }
    with np.load(path, allow_pickle=False) as data:
        mean_states = np.asarray(data["mean_states"], dtype=np.float64)
    nx = ny = 64
    if mean_states.ndim != 2 or mean_states.shape[1] != 2 * nx * ny:
        raise ValueError(f"Unexpected mean-state trace shape in {path}: {mean_states.shape}")
    fields = mean_states.reshape(mean_states.shape[0], 2, nx, ny)
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=2.0 * np.pi / nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=2.0 * np.pi / ny)
    u_hat = np.fft.fft2(fields[:, 0], axes=(-2, -1))
    v_hat = np.fft.fft2(fields[:, 1], axes=(-2, -1))
    divergence = np.fft.ifft2(
        1j * kx[None, :, None] * u_hat + 1j * ky[None, None, :] * v_hat,
        axes=(-2, -1),
    ).real
    return {
        "trace_exists": True,
        "trace_sha256": file_sha256(path),
        "max_abs_divergence_mean_trace": float(np.max(np.abs(divergence))),
        "max_abs_mean_state_trace": float(np.max(np.abs(mean_states))),
    }


def complete_run_matrix(
    output: Path,
    completed: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    sensor_grids: tuple[int, ...] = SENSOR_GRIDS,
    seeds: tuple[int, ...] = SEEDS,
) -> list[dict[str, Any]]:
    discovered = {
        (int(row["sensor_grid"]), int(row["seed"]), str(row["method"])): dict(row)
        for row in completed + failures
    }
    rows: list[dict[str, Any]] = []
    for sensor_grid in sensor_grids:
        for seed in seeds:
            for method in METHODS:
                key = (sensor_grid, seed, method)
                row = discovered.get(
                    key,
                    {
                        "sensor_grid": sensor_grid,
                        "spatial_downsampling_factor_per_axis": 64 // sensor_grid,
                        "seed": seed,
                        "method": method,
                        "label": METHOD_LABELS[method],
                        "status": "missing",
                        "numerical_status": "missing",
                        "error_type": "missing_run_record",
                        "error": "No run JSON was found for this preregistered combination.",
                    },
                )
                nrmse = f(row, "nrmse")
                if row.get("status") != "completed" or row.get("numerical_status") != "valid":
                    row["performance_status"] = "numerical_failure_or_missing"
                    row["nrmse_failure_gt_025"] = ""
                else:
                    failed = bool(nrmse > 0.25)
                    row["performance_status"] = "performance_failure_nrmse_gt_025" if failed else "admitted_nrmse_le_025"
                    row["nrmse_failure_gt_025"] = failed
                trace_value = str(row.get("trace_npz", ""))
                trace = Path(trace_value) if trace_value else output / "__missing_trace__"
                row.update(trace_diagnostics(trace))
                rows.append(row)
    return rows


def f(row: dict[str, Any], key: str) -> float:
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def mean_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return (float(values[0]), float(values[0])) if values.size else (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, values.size, size=(5000, values.size))].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in records:
        groups.setdefault((int(row["sensor_grid"]), str(row["method"])), []).append(row)
    output: list[dict[str, Any]] = []
    for (sensor_grid, method), items in sorted(groups.items()):
        row: dict[str, Any] = {
            "sensor_grid": sensor_grid,
            "spatial_downsampling_factor_per_axis": 64 // sensor_grid,
            "method": method,
            "label": METHOD_LABELS.get(method, method),
            "n": len(items),
            "seeds": ",".join(str(int(item["seed"])) for item in sorted(items, key=lambda x: int(x["seed"]))),
        }
        for metric in METRICS:
            values = np.asarray([f(item, metric) for item in items])
            finite = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(np.mean(finite)) if finite.size else float("nan")
            row[f"{metric}_median"] = float(np.median(finite)) if finite.size else float("nan")
            row[f"{metric}_sd"] = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
            low, high = mean_ci(finite, 12000 + sensor_grid * 100 + len(output) * 17 + len(metric))
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        row["nrmse_failure_count_gt_025"] = int(sum(f(item, "nrmse") > 0.25 for item in items))
        output.append(row)
    return output


def paired(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(int(r["sensor_grid"]), int(r["seed"]), str(r["method"])): r for r in records}
    output: list[dict[str, Any]] = []
    for sensor_grid in sorted({int(r["sensor_grid"]) for r in records}):
        for focal in ("pce", "apce"):
            for baseline in ("aug_enkf", "bma_static"):
                pairs = [
                    (by_key[(sensor_grid, seed, focal)], by_key[(sensor_grid, seed, baseline)])
                    for seed in sorted({int(r["seed"]) for r in records if int(r["sensor_grid"]) == sensor_grid})
                    if (sensor_grid, seed, focal) in by_key and (sensor_grid, seed, baseline) in by_key
                ]
                if not pairs:
                    continue
                row: dict[str, Any] = {
                    "sensor_grid": sensor_grid,
                    "spatial_downsampling_factor_per_axis": 64 // sensor_grid,
                    "method": focal,
                    "method_label": METHOD_LABELS[focal],
                    "baseline": baseline,
                    "baseline_label": METHOD_LABELS[baseline],
                    "paired_n": len(pairs),
                    "paired_seeds": ",".join(str(int(left["seed"])) for left, _ in pairs),
                }
                for metric in METRICS:
                    # For alpha/reynolds errors and all point/probabilistic errors, lower is better.
                    diffs = np.asarray([f(right, metric) - f(left, metric) for left, right in pairs])
                    finite = diffs[np.isfinite(diffs)]
                    low, high = mean_ci(finite, 23000 + sensor_grid + len(output) * 11 + len(metric))
                    row[f"{metric}_gain_mean"] = float(np.mean(finite)) if finite.size else float("nan")
                    row[f"{metric}_gain_ci95_low"] = low
                    row[f"{metric}_gain_ci95_high"] = high
                    row[f"{metric}_win_count"] = int(sum(f(left, metric) < f(right, metric) for left, right in pairs))
                    row[f"{metric}_loss_count"] = len(pairs) - row[f"{metric}_win_count"]
                output.append(row)
    return output


def failure_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "json_path": row.get("_json_path", ""),
            "sensor_grid": row.get("sensor_grid", ""),
            "method": row.get("method", ""),
            "seed": row.get("seed", ""),
            "status": row.get("status", ""),
            "numerical_status": row.get("numerical_status", ""),
            "error_type": row.get("error_type", ""),
            "error": row.get("error", ""),
        }
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate KOL velocity-observation smoke runs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sensor-grids", type=int, nargs="+", default=list(SENSOR_GRIDS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args()
    sensor_grids = tuple(args.sensor_grids)
    seeds = tuple(args.seeds)
    records, failures = discover(args.output)
    full_records = complete_run_matrix(args.output, records, failures, sensor_grids=sensor_grids, seeds=seeds)
    missing_records = [row for row in full_records if row.get("status") == "missing"]
    all_failures = failures + missing_records
    source = args.output / "source_data"
    run_csv = source / "kolmogorov64_velocityobs_run_source_data.csv"
    summary_csv = source / "kolmogorov64_velocityobs_method_summary.csv"
    paired_csv = source / "kolmogorov64_velocityobs_paired_gains.csv"
    failure_csv = source / "kolmogorov64_velocityobs_failure_records.csv"
    write_csv(run_csv, full_records)
    write_csv(summary_csv, summaries(records))
    write_csv(paired_csv, paired(records))
    write_csv(failure_csv, failure_rows(all_failures))
    report = source / "KOL64_VELOCITYOBS_SMOKE_REPORT.md"
    lines = [
        "# KOL-64 sparse velocity-observation smoke report",
        "",
        f"Preregistered records: {len(full_records)}; valid records: {len(records)}; "
        f"failure records: {len(failures)}; missing records: {len(missing_records)}.",
        "Old vorticity-observation/axis-swapped/single-save-step results are protocol-invalid and excluded.",
        "",
        "## Method summaries",
        "",
    ]
    for row in summaries(records):
        lines.append(
            f"- sensor {row['sensor_grid']}x{row['sensor_grid']} ({row['label']}): nRMSE {row['nrmse_mean']:.5g}, "
            f"CRPS {row['crps_mean']:.5g}, alpha MAE {row['alpha_absolute_error_mean']:.5g}, "
            f"Re relative error {row['reynolds_relative_error_mean']:.5g}, coverage error {row['coverage_90_error_mean']:.5g}, "
            f"nRMSE>0.25 {row['nrmse_failure_count_gt_025']}/{row['n']}"
        )
    lines.extend(["", "## Paired comparisons", ""])
    for row in paired(records):
        lines.append(
            f"- sensor {row['sensor_grid']}x{row['sensor_grid']}: {row['method_label']} vs {row['baseline_label']}; "
            f"nRMSE wins {row['nrmse_win_count']}/{row['paired_n']}, CRPS wins {row['crps_win_count']}/{row['paired_n']}, "
            f"alpha-MAE wins {row['alpha_absolute_error_win_count']}/{row['paired_n']}"
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "protocol": "kolmogorov64_velocity_observation_smoke_v2_weighted_aug_temporal_sweep",
        "expected_records": len(sensor_grids) * len(seeds) * len(METHODS),
        "full_table_records": len(full_records),
        "valid_records": len(records),
        "failure_records": len(failures),
        "missing_records": len(missing_records),
        "methods": list(METHODS),
        "sensor_grids": list(sensor_grids),
        "seeds": list(seeds),
        "reynolds_values": sorted({float(row["reynolds"]) for row in records if "reynolds" in row}),
        "forcing_wavenumbers": sorted({int(row["forcing_wavenumber"]) for row in records if "forcing_wavenumber" in row}),
        "observation_intervals": sorted({int(row["obs_interval"]) for row in records if "obs_interval" in row}),
        "metrics": list(METRICS),
        "legacy_results": "excluded: vorticity observation, swapped spatial axes, and no internal RK4 substeps",
        "files": {
            "run_source_data": str(run_csv),
            "method_summary": str(summary_csv),
            "paired_gains": str(paired_csv),
            "failure_records": str(failure_csv),
            "report": str(report),
        },
    }
    manifest["aggregate_sha256"] = hashlib.sha256(json.dumps(full_records, sort_keys=True, default=str).encode()).hexdigest()
    (source / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
