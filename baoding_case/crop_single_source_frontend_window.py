#!/usr/bin/env python3
"""Create a read-only cropped single-source frontend bundle."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def filter_csv(source: Path, target: Path, segment: str, start: float, end: float) -> int:
    with source.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = [row for row in reader if row.get("segment") == segment and start <= float(row["time_s"]) <= end]
        fields = reader.fieldnames or []
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segment", default="danyuan_panxuan_3")
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    args = parser.parse_args()
    if args.end < args.start:
        raise ValueError("end must be >= start")
    args.output.mkdir(parents=True, exist_ok=True)
    n = filter_csv(args.source / "observations_cartesian.csv", args.output / "observations_cartesian.csv", args.segment, args.start, args.end)
    for name in ("frontend_manifest.json", "frontend_calibration.json", "gps_truth.csv"):
        source = args.source / name
        if source.exists():
            shutil.copy2(source, args.output / name)
    manifest = {
        "claim_status": "cropped_single_source_quality_window",
        "source_frontend": str(args.source),
        "segment": args.segment,
        "start_time_s": args.start,
        "end_time_s": args.end,
        "rows_copied": n,
        "selection_rule": "selected from acoustic-only quality ranking; GPS not used for selection",
        "gps_role": "offline evaluation only",
    }
    (args.output / "window_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
