from __future__ import annotations

import argparse
import csv
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from hilda_da.pdebench import PDEBenchManifestRecord, verify_manifest_checksum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a verified subset of PDEBench files")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--filename", action="append", required=True)
    return parser.parse_args()


def select_records(manifest: Path, filenames: list[str]) -> list[PDEBenchManifestRecord]:
    if len(set(filenames)) != len(filenames):
        raise ValueError("Requested filenames must be unique")
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_name: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_name.setdefault(row.get("Filename", ""), []).append(row)
    records = []
    for filename in filenames:
        matches = by_name.get(filename, [])
        if len(matches) != 1:
            raise ValueError(f"Expected one manifest row for {filename!r}, found {len(matches)}")
        row = matches[0]
        records.append(
            PDEBenchManifestRecord(
                pde=row["PDE"],
                filename=row["Filename"],
                url=row["URL"],
                relative_path=row["Path"],
                expected_md5=row["MD5"],
            )
        )
    return records


def target_for(root: Path, record: PDEBenchManifestRecord) -> Path:
    relative = Path(record.relative_path) / record.filename
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe PDEBench manifest path: {relative}")
    target = (root / relative).resolve()
    resolved_root = root.resolve()
    if target != resolved_root and resolved_root not in target.parents:
        raise ValueError(f"PDEBench target escapes root: {target}")
    return target


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def download_one(root: Path, record: PDEBenchManifestRecord) -> dict[str, Any]:
    target = target_for(root, record)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        actual_md5 = verify_manifest_checksum(target, record)
        status = "verified_existing"
    else:
        partial = target.with_suffix(target.suffix + ".part")
        subprocess.run(
            [
                "curl", "-fL", "--retry", "5", "--continue-at", "-",
                "--output", str(partial), record.url,
            ],
            check=True,
        )
        actual_md5 = verify_manifest_checksum(partial, record)
        partial.replace(target)
        status = "downloaded_and_verified"
    stat = target.stat()
    return {
        "filename": record.filename,
        "pde": record.pde,
        "url": record.url,
        "target": str(target),
        "expected_md5": record.expected_md5,
        "actual_md5": actual_md5,
        "size_bytes": stat.st_size,
        "status": status,
        "completed_unix": time.time(),
    }


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    records = select_records(args.manifest, args.filename)
    ledger: dict[str, Any] = {
        "schema_version": 1,
        "manifest": str(args.manifest.resolve()),
        "root": str(args.root.resolve()),
        "requested": args.filename,
        "files": {},
        "started_unix": time.time(),
        "completed": False,
    }
    lock = threading.Lock()
    atomic_json(args.ledger, ledger)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_one, args.root, record): record for record in records}
        for future in as_completed(futures):
            record = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = {
                    "filename": record.filename,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            with lock:
                ledger["files"][record.filename] = result
                atomic_json(args.ledger, ledger)
    failures = [value for value in ledger["files"].values() if value["status"] == "failed"]
    ledger["completed"] = not failures and len(ledger["files"]) == len(records)
    ledger["finished_unix"] = time.time()
    atomic_json(args.ledger, ledger)
    if failures:
        raise SystemExit(f"PDEBench download failures: {len(failures)}")
    print(json.dumps({"completed": True, "files": len(records)}))


if __name__ == "__main__":
    main()
