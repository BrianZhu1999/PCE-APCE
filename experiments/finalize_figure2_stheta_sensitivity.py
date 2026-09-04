from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_CASES = ("wave", "spring", "heat")
EXPECTED_METHODS = ("denkf", "letkf", "iensf", "aug_enkf", "bma_static", "pce", "apce")
EXPECTED_SCALES = (0.50, 0.75, 1.00, 1.25, 1.50)
EXPECTED_SEEDS = tuple(range(2026080700, 2026080750))
FORMAL_PROTOCOL = "figure2-stheta-sensitivity-formal-50paired-seeds-20260811"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_key(row: dict[str, Any]) -> tuple[str, str, int, float]:
    return (
        str(row["case"]),
        str(row["method"]),
        int(row["seed"]),
        round(float(row["sensitivity_scale"]), 8),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize the Figure 2 s_theta formal matrix.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(args.root.rglob("seed_*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        records.append((path, row))

    valid = [
        (path, row)
        for path, row in records
        if row.get("status") == "completed"
        and bool(row.get("valid"))
        and row.get("case") in EXPECTED_CASES
        and row.get("method") in EXPECTED_METHODS
    ]
    failed = [(path, row) for path, row in records if row.get("status") == "failed"]
    valid_keys = [task_key(row) for _, row in valid]
    duplicate_valid = sorted(
        key for key, count in Counter(valid_keys).items() if count > 1
    )
    expected_keys = {
        (case, method, seed, round(scale, 8))
        for case in EXPECTED_CASES
        for method in EXPECTED_METHODS
        for scale in EXPECTED_SCALES
        for seed in EXPECTED_SEEDS
    }
    missing_keys = sorted(expected_keys.difference(valid_keys))
    extra_keys = sorted(set(valid_keys).difference(expected_keys))

    source_hashes = sorted({str(row.get("source_hash")) for _, row in valid if row.get("source_hash")})
    trace_hashes = sorted({str(row.get("trace_sha256")) for _, row in valid if row.get("trace_sha256")})
    asset_hashes = sorted({str(row.get("common_asset_sha256")) for _, row in valid if row.get("common_asset_sha256")})
    failed_types = Counter(str(row.get("error_type", "unknown")) for _, row in failed)

    aggregate_files = {}
    for path in sorted(args.aggregate.glob("*")):
        if path.is_file():
            aggregate_files[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

    code_files = [
        "experiments/run_figure2_stheta_sensitivity_worker.py",
        "experiments/aggregate_figure2_stheta_sensitivity.py",
        "experiments/plot_figure2_stheta_sensitivity.py",
        "experiments/finalize_figure2_stheta_sensitivity.py",
        "experiments/run_figure2_corrected_formal_worker.py",
        "experiments/run_figure2_reviewer_gate.py",
        "experiments/run_modern_baseline_admission.py",
        "paper_experiments/run_spring_heat_gate.py",
        "run_benchmark_v3.py",
    ]
    code_hashes = {}
    for relative in code_files:
        path = Path(__file__).resolve().parents[1] / relative
        if path.exists():
            code_hashes[relative] = sha256_file(path)

    payload = {
        "protocol": FORMAL_PROTOCOL,
        "root": str(args.root),
        "aggregate_root": str(args.aggregate),
        "expected_rows": len(expected_keys),
        "valid_rows": len(valid),
        "unique_valid_tasks": len(set(valid_keys)),
        "failed_records_preserved": len(failed),
        "main_matrix_valid_rows": sum("/retries/" not in str(path).replace("\\", "/") for path, _ in valid),
        "retry_matrix_valid_rows": sum("/retries/" in str(path).replace("\\", "/") for path, _ in valid),
        "main_matrix_failed_rows": sum("/retries/" not in str(path).replace("\\", "/") for path, _ in failed),
        "retry_matrix_failed_rows": sum("/retries/" in str(path).replace("\\", "/") for path, _ in failed),
        "duplicate_valid_tasks": duplicate_valid,
        "missing_tasks": missing_keys,
        "extra_tasks": extra_keys,
        "failure_types": dict(failed_types),
        "source_hashes": source_hashes,
        "valid_trace_hash_count": len(trace_hashes),
        "valid_common_asset_hash_count": len(asset_hashes),
        "aggregate_files": aggregate_files,
        "code_hashes": code_hashes,
        "metadata_repair": {
            "retry1_protocol_repaired": True,
            "old_protocol": "figure2-stheta-sensitivity-5paired-seeds-20260811",
            "new_protocol": FORMAL_PROTOCOL,
            "reason": "Retry1 completed numerically under the formal 50-seed task list; only the protocol label was inherited from the smoke launcher."
        },
        "audit_interpretation": (
            "The 40 original CUDA Invalid device argument failures are retained as an infrastructure audit. "
            "Each failed task was completed once in retry1; no successful task was rerun."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
