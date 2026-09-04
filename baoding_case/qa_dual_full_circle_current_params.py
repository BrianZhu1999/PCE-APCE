#!/usr/bin/env python3
"""QA the frozen-parameter Baoding dual-source full-circle audit bundle."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure-root", type=Path, required=True)
    parser.add_argument("--visual-review-confirmed", action="store_true")
    args = parser.parse_args()
    root = args.figure_root
    stem = root / "dual_full_circle_current_params_apce"
    registry = json.loads((root / "dual_full_circle_current_params_registry.json").read_text(encoding="utf-8"))
    with (root / "dual_full_circle_current_params_source.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    svg = stem.with_suffix(".svg").read_text(encoding="utf-8")
    with Image.open(stem.with_suffix(".tiff")) as image:
        tiff_dpi = image.info.get("dpi", (0.0, 0.0))
        tiff_size = image.size
    geometry = registry["window"]
    config = registry["configuration"]
    metrics = registry["metrics"]
    checks = {
        "all_exports_nonempty": all(stem.with_suffix(f".{suffix}").stat().st_size > 0 for suffix in ("png", "pdf", "svg", "tiff")),
        "source_has_87_contiguous_rows": len(rows) == 87 and all(int(rows[index + 1]["time_s"]) - int(rows[index]["time_s"]) == 1 for index in range(86)),
        "both_gps_sweeps_are_complete_circles": all(abs(float(geometry[f"target{target}_net_sweep_deg"]) - 360.0) < 1.0 for target in (1, 2)),
        "identity_fraction_is_one": abs(float(registry["identity_match_fraction"]) - 1.0) < 1e-12,
        "frozen_configuration_exact": config == {"q_min_accel_mps2": 2.0, "q_max_accel_mps2": 12.0, "observation_covariance_scale": 1.0, "turn_rate_radps": 0.2, "ensemble_members": 48, "seeds": 5},
        "metrics_finite": all(math.isfinite(float(value)) for record in metrics.values() for value in record.values()),
        "standard_gate_failure_recorded": not bool(registry["standard_gate_passed"]),
        "failure_is_t1_maximum_step": float(metrics["target1_apce"]["maximum_step_m"]) > 200.0 and float(metrics["target2_apce"]["maximum_step_m"]) <= 200.0,
        "maximum_step_time_recorded": int(metrics["target1_apce"]["maximum_step_end_time_s"]) == 46659,
        "editable_svg_text": "Target 1: 87-frame complete circle" in svg and "Target 2: 87-frame complete circle" in svg,
        "tiff_600dpi": all(abs(float(value) - 600.0) < 1.0 for value in tiff_dpi),
        "manual_visual_review_confirmed": bool(args.visual_review_confirmed),
    }
    result = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "details": {"tiff_dpi": [float(value) for value in tiff_dpi], "tiff_size_px": list(tiff_size), "metrics": metrics},
    }
    output = root / "dual_full_circle_current_params_qa.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
