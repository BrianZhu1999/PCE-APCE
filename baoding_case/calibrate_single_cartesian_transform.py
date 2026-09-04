#!/usr/bin/env python3
"""Fit a calibration-only Cartesian transform for the Baoding single-source bundle.

The transform is estimated from ``danyuan_panxuan_2`` and then frozen while it
is applied to the later evaluation segment.  GPS is used only for this
calibration fit and for the runner's offline audit; it is never written into
the transformed observations as an assimilation input.

This script deliberately keeps the first candidate conservative: a constant
translation in local ENU coordinates.  Translation does not change the
observation covariance.  More expressive transforms should only be admitted
after a separate calibration/hold-out comparison.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np


POS = ("y_E", "y_N", "y_U")
TRUTH = ("px", "py", "pz")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nearest_truth(truth: dict[float, np.ndarray], time_s: float) -> np.ndarray | None:
    if not truth:
        return None
    key = min(truth, key=lambda value: abs(value - time_s))
    return truth[key] if abs(key - time_s) <= 2.0 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-segment", default="danyuan_panxuan_2")
    parser.add_argument("--evaluation-segment", default="danyuan_panxuan_3")
    args = parser.parse_args()

    source_obs = args.source / "observations_cartesian.csv"
    source_truth = args.source / "gps_truth.csv"
    source_manifest = args.source / "frontend_manifest.json"
    with source_obs.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with source_truth.open(encoding="utf-8-sig", newline="") as stream:
        truth = {
            float(row["time_s"]): np.asarray([float(row[key]) for key in TRUTH], dtype=float)
            for row in csv.DictReader(stream)
        }

    residuals: list[np.ndarray] = []
    for row in rows:
        if row.get("segment") != args.calibration_segment or row.get("valid", "False").lower() != "true":
            continue
        estimate = np.asarray([float(row[key]) for key in POS], dtype=float)
        target = nearest_truth(truth, float(row["time_s"]))
        if target is not None:
            residuals.append(target - estimate)
    if len(residuals) < 8:
        raise RuntimeError("fewer than eight calibration pairs")
    residual_array = np.stack(residuals)
    # The mean is the primary estimate; the median is recorded as a robustness
    # diagnostic and is not selected using the evaluation segment.
    translation = residual_array.mean(axis=0)
    median_translation = np.median(residual_array, axis=0)

    args.output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (args.output / "observations_cartesian.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            if out.get("valid", "False").lower() == "true":
                for key, delta in zip(POS, translation):
                    out[key] = f"{float(out[key]) + float(delta):.12g}"
            writer.writerow(out)
    if source_truth.exists():
        (args.output / "gps_truth.csv").write_bytes(source_truth.read_bytes())

    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest["source_frontend"] = str(args.source)
    manifest["source_observations_sha256"] = sha256(source_obs)
    manifest["calibration_transform"] = {
        "type": "constant_translation_ENU",
        "calibration_segment": args.calibration_segment,
        "evaluation_segment": args.evaluation_segment,
        "pairs": len(residuals),
        "translation_m": translation.tolist(),
        "median_translation_diagnostic_m": median_translation.tolist(),
        "calibration_residual_mean_m": residual_array.mean(axis=0).tolist(),
        "calibration_residual_std_m": residual_array.std(axis=0, ddof=1).tolist(),
        "gps_role": "calibration fit and offline audit only; never an assimilation observation",
    }
    (args.output / "frontend_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    gate = {
        "task": "calibration-only transformed single-source Cartesian frontend",
        "calibration_segment": args.calibration_segment,
        "evaluation_segment": args.evaluation_segment,
        "transform": manifest["calibration_transform"],
        "source_manifest_sha256": sha256(source_manifest),
        "runner_note": "evaluate on the frozen evaluation segment; do not refit using it",
    }
    (args.output / "calibration_transform_manifest.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
