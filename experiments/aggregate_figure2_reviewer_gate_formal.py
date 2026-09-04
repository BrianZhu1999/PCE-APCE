from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from run_figure2_reviewer_gate import summarize, write_csv, write_report


def read_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(root.glob("*/*/seed_*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if row.get("case") and row.get("method") and row.get("seed") is not None:
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate resumable Figure 2 V2 formal worker outputs.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.root)
    write_csv(args.output / "figure2_reviewer_gate_v2_formal_new_method_runs.csv", rows)
    summary = summarize(rows)
    write_csv(args.output / "figure2_reviewer_gate_v2_formal_summary.csv", summary)
    (args.output / "figure2_reviewer_gate_v2_formal_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(args.output / "FIGURE2_REVIEWER_GATE_V2_FORMAL_REPORT.md", summary, sorted({r["case"] for r in rows}))
    print(
        json.dumps(
            {
                "rows": len(rows),
                "summary_rows": len(summary),
                "valid_rows": sum(bool(r.get("valid", False)) for r in rows),
                "output": str(args.output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
