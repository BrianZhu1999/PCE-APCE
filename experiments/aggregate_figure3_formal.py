from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from hilda_da.systems.applied_odes import final_figure3_case_names


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METHOD_LABELS = {
    "denkf": "DEnKF",
    "letkf": "LETKF",
    "iensf": "IEnSF",
    "aug_enkf": "Aug-EnKF",
    "bma_static": "BMA",
    "pce": "PCE",
    "apce": "APCE",
}

TRACE_INDEX_CSV_NAME = "figure3_v4_trace_index.csv"
SOURCE_HASH_FILES = (
    Path(__file__).resolve(),
    PROJECT_ROOT / "paper_experiments" / "run_figure3_applied_ode.py",
    PROJECT_ROOT / "hilda_da" / "systems" / "applied_odes.py",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_trace_index(root: Path, cases: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        for trace_path in sorted((root / case / "runs").glob("*.npz")):
            json_path = trace_path.with_suffix(".json")
            metadata: dict[str, Any] = {}
            if json_path.is_file():
                try:
                    metadata = json.loads(json_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    metadata = {"status": "metadata_json_decode_failed"}
            rows.append(
                {
                    "case": metadata.get("case", case),
                    "method": metadata.get("method", ""),
                    "label": metadata.get("label", ""),
                    "seed": metadata.get("seed", ""),
                    "numerical_status": metadata.get("numerical_status", ""),
                    "trace_npz_path": str(trace_path),
                    "run_json_path": str(json_path),
                }
            )
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hash_manifest() -> dict[str, Any]:
    files: dict[str, str] = {}
    for path in SOURCE_HASH_FILES:
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        files[relative] = file_sha256(path) if path.is_file() else "MISSING"
    encoded = json.dumps(files, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return {
        "aggregate_sha256": hashlib.sha256(encoded).hexdigest(),
        "files": files,
    }


def bootstrap_ci(values: np.ndarray, seed: int, n_bootstrap: int = 5000) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, values.size, size=(n_bootstrap, values.size))].mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def sign_test_p_less(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0.0]
    n = int(nonzero.size)
    if n == 0:
        return 1.0
    wins = int(np.sum(nonzero < 0.0))
    return float(sum(math.comb(n, k) for k in range(wins, n + 1)) / (2**n))


def holm_adjust(rows: list[dict[str, Any]]) -> None:
    indexed = sorted(enumerate(rows), key=lambda item: float(item[1]["p_raw"]))
    total = len(indexed)
    previous = 0.0
    for rank, (index, row) in enumerate(indexed):
        adjusted = min(1.0, max(previous, (total - rank) * float(row["p_raw"])))
        rows[index]["p_holm"] = adjusted
        previous = adjusted


def summarize(records: list[dict[str, str]], seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cases = sorted({row["case"] for row in records})
    for case in cases:
        methods = sorted({row["method"] for row in records if row["case"] == case})
        for method in methods:
            subset = [
                row for row in records
                if row["case"] == case and row["method"] == method and row.get("numerical_status") == "valid"
            ]
            item: dict[str, Any] = {
                "case": case,
                "method": method,
                "label": METHOD_LABELS.get(method, method),
                "n": len(subset),
                "valid": bool(subset),
                "seeds": ",".join(row["seed"] for row in subset),
            }
            if subset:
                for key in (
                    "nrmse",
                    "rmse",
                    "crps",
                    "coverage_90",
                    "interval_width_90",
                    "alpha_absolute_error",
                    "physical_validity_error",
                    "peak_time_error",
                    "auc_relative_error",
                    "runtime_seconds",
                    "peak_gpu_memory_mb",
                ):
                    values = np.asarray([float(row[key]) for row in subset], dtype=float)
                    item[f"{key}_mean"] = float(values.mean())
                    item[f"{key}_sd"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
                    item[f"{key}_ci95"] = bootstrap_ci(values, seed + len(rows)) if values.size > 1 else [float(values[0]), float(values[0])]
            rows.append(item)
    return rows


def paired_comparisons(records: list[dict[str, str]], seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metrics = ("nrmse", "crps", "coverage_90", "interval_width_90", "physical_validity_error", "runtime_seconds")
    valid = [row for row in records if row.get("numerical_status") == "valid"]
    for case in sorted({row["case"] for row in valid}):
        case_rows = [row for row in valid if row["case"] == case]
        for reference in ("apce", "pce"):
            if reference not in {row["method"] for row in case_rows}:
                continue
            baselines = sorted({row["method"] for row in case_rows if row["method"] != reference})
            for baseline in baselines:
                ref_by_seed = {row["seed"]: row for row in case_rows if row["method"] == reference}
                base_by_seed = {row["seed"]: row for row in case_rows if row["method"] == baseline}
                paired = sorted(set(ref_by_seed) & set(base_by_seed))
                for metric in metrics:
                    differences = np.asarray(
                        [float(ref_by_seed[s][metric]) - float(base_by_seed[s][metric]) for s in paired],
                        dtype=float,
                    )
                    ci = bootstrap_ci(differences, seed + len(rows)) if differences.size > 1 else [float(differences[0]), float(differences[0])]
                    rows.append(
                        {
                            "case": case,
                            "reference": reference,
                            "baseline": baseline,
                            "metric": metric,
                            "n": int(differences.size),
                            "mean_difference_reference_minus_baseline": float(differences.mean()),
                            "ci95_low": ci[0],
                            "ci95_high": ci[1],
                            "p_raw": sign_test_p_less(differences),
                            "paired_seeds": ",".join(paired),
                        }
                    )
    holm_adjust(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Figure 3 formal case directories.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--cases",
        default=",".join(final_figure3_case_names()),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=2026081200)
    args = parser.parse_args()
    output = args.output or args.root
    cases = [item.strip() for item in args.cases.split(",") if item.strip()]
    records: list[dict[str, str]] = []
    manifests: dict[str, Any] = {}
    for case in cases:
        case_root = args.root / case
        run_path = case_root / "source_data" / "figure3_v4_run_source_data.csv"
        manifest_path = case_root / "figure3_v4_config_manifest.json"
        if not run_path.is_file():
            raise FileNotFoundError(run_path)
        records.extend(read_csv(run_path))
        if manifest_path.is_file():
            manifests[case] = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_data = output / "source_data"
    write_csv(source_data / "figure3_v4_run_source_data.csv", records)
    summary = summarize(records, args.seed)
    comparisons = paired_comparisons(records, args.seed + 10_000)
    write_csv(source_data / "figure3_v4_method_summary.csv", summary)
    write_csv(source_data / "figure3_v4_paired_comparisons.csv", comparisons)
    trace_rows = collect_trace_index(args.root, cases)
    write_csv(source_data / TRACE_INDEX_CSV_NAME, trace_rows)
    payload = {
        "protocol": "figure3-final-hybrid-ode-v4-formal-combined",
        "cases": cases,
        "records": len(records),
        "valid_records": sum(1 for row in records if row.get("numerical_status") == "valid"),
        "case_manifests": manifests,
        "trace_npz_count": len(trace_rows),
        "trace_index_file": str(source_data / TRACE_INDEX_CSV_NAME),
        "source_hash": source_hash_manifest(),
    }
    (output / "figure3_v4_config_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"records": len(records), "valid": payload["valid_records"], "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
