from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(
    "<HILDA_RESULTS_ROOT>/results/"
    "figure3_selected5_freq1to8_formal_50seeds_allmethods_20260813"
)
COMBINED = ROOT / "combined"
SOURCE = COMBINED / "source_data"
CASES = ["pk_infusion", "chemical", "pendulum", "fhn", "robertson"]
METHODS = ["denkf", "letkf", "iensf", "aug_enkf", "bma_static", "pce", "apce"]
METHOD_LABELS = {
    "denkf": "DEnKF",
    "letkf": "LETKF",
    "iensf": "IEnSF",
    "aug_enkf": "Aug-EnKF",
    "bma_static": "BMA",
    "pce": "PCE",
    "apce": "APCE",
}
METRICS = [
    "nrmse",
    "crps",
    "alpha_absolute_error",
    "coverage_90",
    "interval_width_90",
    "core_runtime_seconds",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def key_for(row: dict[str, Any], freq: int | None = None) -> tuple[str, str, str, int]:
    scenario = str(row.get("observation_scenario") or (f"freq{freq}" if freq else ""))
    return (
        scenario,
        str(row["case"]),
        str(row["method"]),
        int(row["seed"]),
    )


def load_json(path: Path, freq: int) -> dict[str, Any]:
    row = json.loads(path.read_text(encoding="utf-8"))
    row["_path"] = str(path)
    row["_freq"] = freq
    if not row.get("observation_scenario"):
        row["observation_scenario"] = f"freq{freq}"
    return row


def is_valid(row: dict[str, Any]) -> bool:
    return row.get("numerical_status") == "valid" or row.get("valid") is True


def main() -> None:
    original: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    original_failures: list[dict[str, Any]] = []
    for freq in range(1, 9):
        for path in sorted((ROOT / f"freq{freq}" / "runs").glob("*.json")):
            row = load_json(path, freq)
            key = key_for(row, freq)
            original[key] = row
            if not is_valid(row):
                original_failures.append(
                    {
                        "task_key": "|".join(map(str, key)),
                        "original_path": str(path),
                        "original_status": row.get("numerical_status", ""),
                        "original_nrmse": row.get("nrmse", ""),
                        "original_max_abs_state": row.get("max_abs_state", ""),
                    }
                )

    retries: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    retry_root = ROOT / "retries" / "pendulum_iensf_40x4_20260813"
    for path in sorted(retry_root.glob("shard*_gpu*/runs/*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["_path"] = str(path)
        row["_freq"] = int(str(row["observation_scenario"]).replace("freq", ""))
        key = key_for(row)
        retries[key] = row

    authoritative: dict[tuple[str, str, str, int], dict[str, Any]] = dict(original)
    retry_audit: list[dict[str, Any]] = []
    for key, retry in retries.items():
        old = authoritative.get(key)
        authoritative[key] = retry
        retry_audit.append(
            {
                "task_key": "|".join(map(str, key)),
                "original_path": str(old.get("_path", "")) if old else "",
                "original_status": old.get("numerical_status", "") if old else "",
                "retry_path": str(retry.get("_path", "")),
                "retry_status": retry.get("numerical_status", ""),
                "retry_config": "IEnSF sampling_time_step_count=40; refinement_iterations=4; endpoint_epsilon=0.001",
                "authoritative": bool(is_valid(retry)),
            }
        )

    expected_keys = {
        (f"freq{freq}", case, method, seed)
        for freq in range(1, 9)
        for case in CASES
        for method in METHODS
        for seed in range(2026081200, 2026081250)
    }
    missing = sorted(expected_keys - set(authoritative))
    invalid = [
        {
            "task_key": "|".join(map(str, key)),
            "path": str(row.get("_path", "")),
            "status": row.get("numerical_status", ""),
        }
        for key, row in authoritative.items()
        if key in expected_keys and not is_valid(row)
    ]

    clean_rows: list[dict[str, Any]] = []
    for key in sorted(expected_keys):
        row = dict(authoritative[key])
        row["authoritative_source"] = "retry" if key in retries else "original"
        row["task_key"] = "|".join(map(str, key))
        row.pop("_path", None)
        row.pop("_freq", None)
        clean_rows.append(row)

    write_csv(SOURCE / "figure3_freq1to8_formal_authoritative_run_source_data.csv", clean_rows)
    write_csv(SOURCE / "figure3_freq1to8_formal_retry_audit.csv", retry_audit)
    write_csv(SOURCE / "figure3_freq1to8_formal_original_failure_audit.csv", original_failures)

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in clean_rows:
        grouped[
            (
                str(row["observation_scenario"]),
                str(row["case"]),
                str(row["method"]),
            )
        ].append(row)
    summary: list[dict[str, Any]] = []
    for (scenario, case, method), rows in sorted(grouped.items()):
        out: dict[str, Any] = {
            "observation_scenario": scenario,
            "obs_interval_factor": rows[0].get("obs_interval_factor", ""),
            "case": case,
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "n": len(rows),
            "valid": sum(is_valid(r) for r in rows),
        }
        for metric in METRICS:
            vals = [
                float(r[metric])
                for r in rows
                if r.get(metric) not in (None, "")
                and math.isfinite(float(r[metric]))
            ]
            out[f"{metric}_mean"] = statistics.fmean(vals) if vals else math.nan
            out[f"{metric}_sd"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        summary.append(out)
    write_csv(SOURCE / "figure3_freq1to8_formal_method_summary.csv", summary)

    summary_index = {
        (r["observation_scenario"], r["case"], r["method"]): r for r in summary
    }
    paired: list[dict[str, Any]] = []
    for scenario in [f"freq{i}" for i in range(1, 9)]:
        for case in CASES:
            for target in ["pce", "apce"]:
                for reference in ["aug_enkf", "bma_static"]:
                    a = summary_index[(scenario, case, target)]
                    b = summary_index[(scenario, case, reference)]
                    out = {
                        "observation_scenario": scenario,
                        "case": case,
                        "target_method": target,
                        "target_label": METHOD_LABELS[target],
                        "reference_method": reference,
                        "reference_label": METHOD_LABELS[reference],
                    }
                    for metric in METRICS:
                        out[f"delta_{metric}"] = float(
                            a[f"{metric}_mean"] - b[f"{metric}_mean"]
                        )
                    paired.append(out)
    write_csv(SOURCE / "figure3_freq1to8_formal_paired_comparisons.csv", paired)

    trace_rows: list[dict[str, Any]] = []
    for key in sorted(expected_keys):
        row = authoritative[key]
        json_path = Path(str(row.get("_path", "")))
        npz_path = json_path.with_suffix(".npz")
        trace_rows.append(
            {
                "task_key": "|".join(map(str, key)),
                "observation_scenario": key[0],
                "case": key[1],
                "method": key[2],
                "seed": key[3],
                "authoritative_source": "retry" if key in retries else "original",
                "run_json_path": str(json_path),
                "trace_npz_path": str(npz_path),
                "trace_exists": npz_path.is_file(),
                "numerical_status": row.get("numerical_status", ""),
            }
        )
    write_csv(SOURCE / "figure3_freq1to8_formal_authoritative_trace_index.csv", trace_rows)

    source_hash = sha256_file(
        SOURCE / "figure3_freq1to8_formal_authoritative_run_source_data.csv"
    )
    manifest = {
        "protocol": "figure3-selected5-freq1to8-formal-50seed-allmethods-authoritative-with-retries",
        "expected_records": len(expected_keys),
        "authoritative_records": len(clean_rows),
        "authoritative_valid_records": sum(is_valid(r) for r in clean_rows),
        "missing_records": len(missing),
        "invalid_records": len(invalid),
        "original_records": len(original),
        "original_invalid_records": len(original_failures),
        "retry_records": len(retries),
        "retry_valid_records": sum(is_valid(r) for r in retries.values()),
        "retry_profile": {
            "method": "IEnSF",
            "case": "pendulum",
            "sampling_time_step_count": 40,
            "refinement_iterations": 4,
            "endpoint_epsilon": 0.001,
            "max_score_component": 1000.0,
        },
        "cases": CASES,
        "methods": METHODS,
        "seed_range": [2026081200, 2026081249],
        "frequencies": list(range(1, 9)),
        "source_data_sha256": source_hash,
        "missing": missing,
        "invalid": invalid,
        "failure_policy": "Original failures are retained in audit; valid retry is authoritative by task key.",
        "note": "The retry changes only IEnSF probability-flow numerical integration resolution; benchmark, observations, seeds, truth, noise and evaluation thresholds are unchanged.",
    }
    (COMBINED / "figure3_freq1to8_formal_authoritative_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
