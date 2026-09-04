#!/usr/bin/env python3
"""Write a bounded smoke gate without converting failed performance to success."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    result = Path(__import__("sys").argv[1])
    summary = list(csv.DictReader((result / "runs" / "method_summary.csv").open(encoding="utf-8")))
    finite = all(math.isfinite(float(row["position_rmse_m"])) and math.isfinite(float(row["coverage_90"])) for row in summary)
    rows = [float(row["position_rmse_m"]) for row in summary]
    coverage = [float(row["coverage_90"]) for row in summary]
    gate = {
        "scope": "2017 Baoding single-helicopter full single-flight smoke",
        "frontend": "direct WAV MUSIC, d=0.50 m candidate",
        "valid_runs": len(summary),
        "expected_runs": 25,
        "all_run_metrics_finite": finite,
        "position_rmse_median": sorted(rows)[len(rows) // 2] if rows else None,
        "coverage_90_mean": sum(coverage) / len(coverage) if coverage else None,
        "smoke_matrix_complete": len(summary) == 25,
        "performance_gate": False,
        "formal_admission": False,
        "failure_reasons": [
            "direct WAV MUSIC frontend remains poorly calibrated for long-range single-flight bearing;",
            "position RMSE is kilometre-scale or larger and 90% coverage is not calibrated;",
            "d=0.56 m candidate was audited separately and was worse than d=0.50 m on the paired PCE seed."
        ],
        "authoritative_inputs": {
            "archive": "<PRIVATE_DATA_ROOT>/2017保定实验/2017保定实验",
            "result": str(result),
            "frontends": [
                str(result.parent / "calibration_d050_v5/frontend"),
                str(result.parent / "calibration_d056_v1/frontend")
            ]
        }
    }
    (result / "smoke_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
