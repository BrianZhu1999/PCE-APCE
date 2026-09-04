#!/usr/bin/env python3
"""Content-audit a newly supplied Baoding archive against remote datasets.

This tool is intentionally read-only for every input tree. It hashes files on
the remote server and writes manifests/reporting only below ``--output``.
Exact duplicate decisions are based on SHA-256, not directory or file names.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CHUNK_BYTES = 8 * 1024 * 1024
SAMPLE_LIMIT = 40


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def extension(path: Path) -> str:
    return path.suffix.lower() or "[no_extension]"


def build_manifest(label: str, root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not root.is_dir():
        raise RuntimeError(f"input directory does not exist: {root}")
    records: list[dict[str, Any]] = []
    suffix_counts: Counter[str] = Counter()
    suffix_bytes: Counter[str] = Counter()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        size = path.stat().st_size
        suffix = extension(path)
        record = {
            "dataset": label,
            "relative_path": path.relative_to(root).as_posix(),
            "bytes": size,
            "extension": suffix,
            "sha256": sha256(path),
        }
        records.append(record)
        suffix_counts[suffix] += 1
        suffix_bytes[suffix] += size
    summary = {
        "label": label,
        "root": str(root),
        "files": len(records),
        "bytes": sum(row["bytes"] for row in records),
        "extension_counts": dict(sorted(suffix_counts.items())),
        "extension_bytes": dict(sorted(suffix_bytes.items())),
    }
    return records, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["dataset", "relative_path", "bytes", "extension", "sha256"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def hash_index(records: list[dict[str, Any]]) -> dict[tuple[int, str], list[dict[str, Any]]]:
    output: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        output[(int(row["bytes"]), str(row["sha256"]))].append(row)
    return output


def overlap(candidate: list[dict[str, Any]], reference: list[dict[str, Any]]) -> dict[str, Any]:
    reference_index = hash_index(reference)
    matches: list[dict[str, Any]] = []
    matched_bytes = 0
    for row in candidate:
        duplicates = reference_index.get((int(row["bytes"]), str(row["sha256"])), [])
        if duplicates:
            matched_bytes += int(row["bytes"])
            matches.append(
                {
                    "candidate_relative_path": row["relative_path"],
                    "bytes": row["bytes"],
                    "reference_relative_paths": [item["relative_path"] for item in duplicates[:5]],
                }
            )
    return {
        "candidate_files_exactly_duplicated": len(matches),
        "candidate_bytes_exactly_duplicated": matched_bytes,
        "candidate_file_fraction": len(matches) / len(candidate) if candidate else 0.0,
        "candidate_byte_fraction": matched_bytes / sum(int(row["bytes"]) for row in candidate) if candidate else 0.0,
        "sample_exact_matches": matches[:SAMPLE_LIMIT],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--reference",
        nargs=2,
        action="append",
        metavar=("LABEL", "PATH"),
        default=[],
        help="repeatable reference dataset label and absolute path",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.reference:
        raise SystemExit("at least one --reference LABEL PATH pair is required")
    args.output.mkdir(parents=True, exist_ok=True)
    candidate_records, candidate_summary = build_manifest("candidate_20200317", args.candidate)
    write_csv(args.output / "candidate_manifest.csv", candidate_records)
    reference_payload: dict[str, Any] = {}
    all_reference_keys: set[tuple[int, str]] = set()
    for label, raw_path in args.reference:
        records, summary = build_manifest(label, Path(raw_path))
        write_csv(args.output / f"reference_{label}_manifest.csv", records)
        reference_payload[label] = {
            "summary": summary,
            "exact_content_overlap_with_candidate": overlap(candidate_records, records),
        }
        all_reference_keys.update(hash_index(records))

    candidate_unique = [
        row
        for row in candidate_records
        if (int(row["bytes"]), str(row["sha256"])) not in all_reference_keys
    ]
    write_csv(args.output / "candidate_not_in_any_reference.csv", candidate_unique)
    result = {
        "audit_type": "content-based Baoding dataset comparison",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": candidate_summary,
        "references": reference_payload,
        "candidate_files_not_exactly_present_in_any_reference": len(candidate_unique),
        "candidate_bytes_not_exactly_present_in_any_reference": sum(int(row["bytes"]) for row in candidate_unique),
        "sample_candidate_files_not_in_any_reference": candidate_unique[:SAMPLE_LIMIT],
        "method": {
            "duplicate_definition": "same byte size and SHA-256",
            "hash_algorithm": "SHA-256",
            "inputs_read_only": True,
            "warning": "Files with different bytes are not treated as duplicates even if names or topics are similar.",
        },
    }
    (args.output / "content_comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
