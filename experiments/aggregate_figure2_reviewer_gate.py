from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_figure2_reviewer_gate import (
    read_existing_rows,
    summarize,
    write_csv,
    write_report,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate split reviewer-gate Figure 2 smoke runs.")
    parser.add_argument("--split-root", type=Path)
    parser.add_argument("--new-method-runs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--existing-source-data",
        type=Path,
        default=PROJECT_ROOT / "ncs_chinese_submission" / "source_data" / "figure2_run_source_data_20260807.csv",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.split_root is None and args.new_method_runs is None:
        raise SystemExit("Provide --split-root or --new-method-runs.")

    new_rows: list[dict[str, str]] = []
    if args.new_method_runs is not None:
        new_rows.extend(read_csv(args.new_method_runs))
    if args.split_root is not None:
        for path in sorted(args.split_root.glob("split_gpu*/new_method_runs.csv")):
            for row in read_csv(path):
                row["split_source"] = path.parent.name
                new_rows.append(row)
    write_csv(args.output / "figure2_reviewer_gate_new_method_runs_20260810.csv", new_rows)
    (args.output / "figure2_reviewer_gate_new_method_runs_20260810.json").write_text(
        json.dumps(new_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    seeds = {int(float(row["seed"])) for row in new_rows if row.get("seed")}
    cases = {row["case"] for row in new_rows if row.get("case")}
    existing_rows = read_existing_rows(args.existing_source_data, cases, seeds)
    combined = existing_rows + new_rows
    write_csv(args.output / "figure2_reviewer_gate_combined_with_existing_20260810.csv", combined)

    summary = summarize(combined)
    write_csv(args.output / "figure2_reviewer_gate_summary_20260810.csv", summary)
    (args.output / "figure2_reviewer_gate_summary_20260810.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(args.output / "FIGURE2_REVIEWER_GATE_SMOKE_20260810.md", summary, sorted(cases))
    print(
        json.dumps(
            {
                "new_rows": len(new_rows),
                "existing_rows": len(existing_rows),
                "combined_rows": len(combined),
                "summary_rows": len(summary),
                "output": str(args.output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
