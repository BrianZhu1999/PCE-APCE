#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--aggregate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = [
        args.case_root / "config.json",
        args.case_root / "requirements.txt",
        args.case_root / "prepare_data.py",
        args.case_root / "identify_models.py",
        args.case_root / "run_pilot.py",
        args.case_root / "aggregate_pilot.py",
        args.case_root / "f16_gvt" / "data.py",
        args.case_root / "f16_gvt" / "preprocess.py",
        args.case_root / "f16_gvt" / "identification.py",
        args.case_root / "f16_gvt" / "candidates.py",
        args.case_root / "f16_gvt" / "assimilation.py",
        args.case_root / "f16_gvt" / "metrics.py",
        args.case_root / "models" / "level1_base_model.npz",
        args.case_root / "models" / "level1_identification.json",
        args.case_root / "models" / "modal_uncertainty_path.json",
        args.aggregate_root / "f16_gvt_run_source_data.csv",
        args.aggregate_root / "f16_gvt_summary.csv",
        args.aggregate_root / "f16_gvt_paired_comparisons.csv",
        args.aggregate_root / "f16_gvt_admission.json",
    ]
    missing = [str(path) for path in files if not path.exists()]
    manifest = {
        "case": "f16_gvt_7p3hz",
        "missing_files": missing,
        "hashes": {str(path.relative_to(args.case_root)): sha256(path) for path in files if path.exists()},
        "row_counts": {
            "run_source_data": csv_rows(args.aggregate_root / "f16_gvt_run_source_data.csv"),
            "summary": csv_rows(args.aggregate_root / "f16_gvt_summary.csv"),
            "paired_comparisons": csv_rows(args.aggregate_root / "f16_gvt_paired_comparisons.csv"),
        },
        "manuscript_modified": False,
        "formal_matrix_launched": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
