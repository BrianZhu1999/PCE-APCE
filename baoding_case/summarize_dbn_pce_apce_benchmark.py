#!/usr/bin/env python3
"""Summarize and integrity-check the DBN-derived PCE/APCE benchmark."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    integrity = []
    for target in (1, 2, 3):
        frontend = args.root / f"target{target}" / "frontend"
        track = frontend / "dbn_track.csv"
        obs = frontend / "observations.csv"
        manifest = frontend / "frontend_manifest.json"
        integrity.append({"target": target, "dbn_track_exists": track.is_file(), "observations_exists": obs.is_file(), "frontend_manifest_exists": manifest.is_file(), "dbn_track_sha256": sha256(track) if track.is_file() else None, "observation_rows": sum(1 for _ in obs.open(encoding="utf-8")) - 1 if obs.is_file() else 0})
        for method in ("pce", "apce"):
            for path in sorted((args.root / f"target{target}" / "runs").glob(f"{method}_seed_*.json")):
                payload = json.loads(path.read_text(encoding="utf-8"))
                records = [row for row in payload.get("records", []) if row.get("position_error_m") is not None]
                errors = [float(row["position_error_m"]) for row in records]
                rows.append({"target": target, "method": method, "seed": payload.get("seed"), "records": len(records), "rmse_m": math.sqrt(sum(value * value for value in errors) / len(errors)) if errors else None, "crps_m": sum(float(row["crps_position_m"]) for row in records) / len(records) if records else None, "coverage_90": sum(float(row["coverage_90"]) for row in records) / len(records) if records else None, "interval_width_m": sum(float(row["interval_width_m"]) for row in records) / len(records) if records else None, "run_status": payload.get("status"), "input_provenance": payload.get("input_provenance")})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        fields = [key for key in rows[0] if key != "input_provenance"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows([{key: row[key] for key in fields} for row in rows])
    seeds = sorted({row["seed"] for row in rows})
    expected_records = max((row["records"] for row in rows), default=0)
    expected_runs = 3 * 2 * len(seeds)
    summary = {"claim_status": "paper_inspired_dbn_pce_apce_benchmark", "root": str(args.root), "targets": [1, 2, 3], "methods": ["pce", "apce"], "seeds": seeds, "expected_records_per_run": expected_records, "expected_runs": expected_runs, "valid_runs": sum(row["run_status"] == "valid" and row["records"] == expected_records for row in rows), "integrity": integrity, "metrics_csv": str(csv_path), "independent_acoustic_frontend": False, "formal_uncertainty_calibration": False, "formal_paper_reproduction": False, "interpretation": "PCE/APCE downstream benchmark on a DBN-like semi-synthetic observation stream calibrated from real Baoding DBN residuals; useful as a reproducible long-window engineering scenario, not as an independent acoustic or calibration claim."}
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
