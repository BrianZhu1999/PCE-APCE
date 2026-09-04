from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.input_directory.glob("admission_*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        raise SystemExit("no admission JSON files found")
    args.output_directory.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (args.output_directory / "admission_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for case in ("wave", "spring", "heat"):
        for method in (
            "denkf",
            "letkf",
            "ensf",
            "iensf",
            "ensf_lr",
            "ensf_lr_ridge",
        ):
            subset = [row for row in rows if row["case"] == case and row["method"] == method]
            if not subset:
                continue
            row = subset[0].copy()
            row["n_records"] = len(subset)
            summary.append(row)
    (args.output_directory / "admission_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
