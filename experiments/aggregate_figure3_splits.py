from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from paper_experiments.run_figure3_applied_ode import paired_comparisons, summarize, write_csv


RUN_CSV_NAME = "figure3_v4_run_source_data.csv"
SUMMARY_CSV_NAME = "figure3_v4_method_summary.csv"
COMPARISONS_CSV_NAME = "figure3_v4_paired_comparisons.csv"
TRACE_INDEX_CSV_NAME = "figure3_v4_trace_index.csv"
SOURCE_HASH_FILES = (
    Path(__file__).resolve(),
    PROJECT_ROOT / "paper_experiments" / "run_figure3_applied_ode.py",
    PROJECT_ROOT / "hilda_da" / "systems" / "applied_odes.py",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def collect_trace_index(split_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace_path in sorted(split_root.glob("split_gpu*/runs/*.npz")):
        json_path = trace_path.with_suffix(".json")
        metadata: dict[str, Any] = {}
        if json_path.is_file():
            try:
                metadata = json.loads(json_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metadata = {"status": "metadata_json_decode_failed"}
        rows.append(
            {
                "split_source": trace_path.parents[1].name,
                "case": metadata.get("case", ""),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate split Figure 3 ODE runs.")
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026081100)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    split_paths = sorted(args.split_root.glob(f"split_gpu*/source_data/{RUN_CSV_NAME}"))
    if not split_paths:
        raise FileNotFoundError(f"No split Figure 3 CSVs found under {args.split_root}")
    for path in split_paths:
        for row in read_csv(path):
            row["split_source"] = path.parents[1].name
            rows.append(row)

    args.output.mkdir(parents=True, exist_ok=True)
    source_data = args.output / "source_data"
    write_csv(source_data / RUN_CSV_NAME, rows)
    summary = summarize(rows, args.seed)
    comparisons = paired_comparisons(rows, args.seed + 10_000)
    write_csv(source_data / SUMMARY_CSV_NAME, summary)
    write_csv(source_data / COMPARISONS_CSV_NAME, comparisons)
    trace_rows = collect_trace_index(args.split_root)
    write_csv(source_data / TRACE_INDEX_CSV_NAME, trace_rows)
    run_logs = sorted(args.split_root.glob("split_gpu*/run.log"))
    split_manifests = sorted(args.split_root.glob("split_gpu*/figure3_v4_config_manifest.json"))

    payload = {
        "protocol": "figure3-final-hybrid-ode-v4-split-aggregate",
        "split_root": str(args.split_root),
        "split_files": [str(path) for path in split_paths],
        "split_manifests": [str(path) for path in split_manifests],
        "run_logs": [str(path) for path in run_logs],
        "trace_npz_count": len(trace_rows),
        "trace_index_file": str(source_data / TRACE_INDEX_CSV_NAME),
        "source_hash": source_hash_manifest(),
        "records": len(rows),
        "valid_records": sum(1 for row in rows if row.get("numerical_status") == "valid"),
        "cases": sorted({row["case"] for row in rows}),
        "methods": sorted({row["method"] for row in rows}),
    }
    (args.output / "figure3_v4_config_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
