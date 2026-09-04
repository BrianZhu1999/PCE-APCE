#!/usr/bin/env python3
"""Freeze target-specific reliability profiles from calibration-only matrices."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


PROFILES = ("baseline", "calibration_floor", "acoustic_reliability", "combined")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def method_metrics(path: Path, method: str) -> dict[str, float]:
    rows = [row for row in read_csv(path) if row["method"] == method]
    if len(rows) != 5:
        raise RuntimeError(f"{path}: expected five {method} rows, found {len(rows)}")
    return {
        "mean_rmse_m": float(np.mean([float(row["position_rmse_m"]) for row in rows])),
        "mean_crps_m": float(np.mean([float(row["crps_position_m"]) for row in rows])),
        "mean_coverage_90": float(np.mean([float(row["coverage_90"]) for row in rows])),
        "mean_interval_width_m": float(np.mean([float(row["interval_width_m"]) for row in rows])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metrics: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    hashes: dict[str, str] = {}
    for profile in PROFILES:
        metrics[profile] = {}
        for target in (1, 2):
            path = args.matrix_root / profile / f"target{target}" / "runs" / "method_summary.csv"
            metrics[profile][str(target)] = {
                method: method_metrics(path, method) for method in ("pce", "apce")
            }
            hashes[str(path)] = sha256(path)

    selected: dict[str, str] = {}
    decisions: dict[str, object] = {}
    for target in (1, 2):
        key = str(target)
        baseline = metrics["baseline"][key]
        admitted: list[str] = []
        for profile in PROFILES:
            current = metrics[profile][key]
            pce_bounded = current["pce"]["mean_rmse_m"] <= baseline["pce"]["mean_rmse_m"] * 1.15
            width_bounded = current["apce"]["mean_interval_width_m"] <= max(
                baseline["apce"]["mean_interval_width_m"] * 4.0, 1.0
            )
            if pce_bounded and width_bounded:
                admitted.append(profile)
        winner = min(
            admitted,
            key=lambda profile: (
                metrics[profile][key]["apce"]["mean_rmse_m"],
                metrics[profile][key]["apce"]["mean_crps_m"],
                PROFILES.index(profile),
            ),
        )
        selected[key] = winner
        decisions[key] = {
            "selected_profile": winner,
            "admitted_profiles": admitted,
            "baseline_apce_mean_rmse_m": baseline["apce"]["mean_rmse_m"],
            "selected_apce_mean_rmse_m": metrics[winner][key]["apce"]["mean_rmse_m"],
            "relative_apce_rmse_change": (
                metrics[winner][key]["apce"]["mean_rmse_m"]
                / baseline["apce"]["mean_rmse_m"] - 1.0
            ),
        }

    payload = {
        "selection_status": "frozen before 60-frame evaluation",
        "calibration_interval_s": [46540, 46560],
        "evaluation_interval_s": [46593, 46652],
        "selection_metric": "minimum five-seed mean APCE RMSE, then CRPS, among calibration-admitted profiles",
        "admission": {
            "maximum_pce_rmse_ratio_vs_baseline": 1.15,
            "maximum_apce_interval_width_ratio_vs_baseline": 4.0,
            "baseline_always_admitted": True,
        },
        "selected_profile_by_target": selected,
        "decisions": decisions,
        "calibration_metrics": metrics,
        "gps_role": (
            "GPS is used only to score the declared 46540-46560 s calibration matrices; "
            "the 46593-46652 s evaluation interval is not read during profile selection"
        ),
        "source_hashes": hashes,
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
