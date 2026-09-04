#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    errors = []
    smoke = args.root / "smoke_aligned"
    for method in ("denkf", "bma", "pce", "apce"):
        record = json.loads((smoke / f"{method}.json").read_text(encoding="utf-8"))
        if record["reduced_order_model_used"]:
            errors.append(f"ROM flag in {method}")
        if record["future_observations_used"]:
            errors.append(f"future leakage in {method}")
        if max(record["cfl_numbers"]) >= 1.0:
            errors.append(f"unstable CFL in {method}")
        numeric = [value for value in record.values() if isinstance(value, (int, float))]
        if not np.isfinite(numeric).all():
            errors.append(f"nonfinite metrics in {method}")
    required = [
        args.root / "aggregate" / "s1_full_dynamics_source_data.csv",
        args.root / "aggregate" / "admission.json",
        args.root / "aggregate" / "S1_FULL_DYNAMICS_SMOKE_REPORT.md",
        args.root / "assets" / "s1_full_dynamics_smoke.svg",
        args.root / "assets" / "s1_full_dynamics_smoke.pdf",
        args.root / "assets" / "s1_full_dynamics_smoke.png",
        args.root / "assets" / "s1_full_dynamics_smoke.tiff",
        args.root / "assets" / "s1_full_dynamics_figure_source_data.csv",
        args.root / "assets" / "s1_full_dynamics_figure_registry.json",
    ]
    for path in required:
        if not path.is_file():
            errors.append(f"missing {path}")
    code_files = sorted(HERE.glob("*.py")) + sorted((HERE / "fullwave").glob("*.py")) + sorted((HERE / "tests").glob("*.py")) + [HERE / "config.json"]
    manifest = {
        "case": "meshir_s1_full_grid",
        "errors": errors,
        "passed": not errors,
        "reduced_order_model_used": False,
        "full_state_dimension": 7938,
        "authoritative_source_bundle": "<HILDA_RESULTS_ROOT>/experiments/meshir_pilot_20260818/cache/",
        "code_hashes": {str(path.relative_to(HERE)): sha256(path) for path in code_files if path.is_file()},
        "output_hashes": {str(path.relative_to(args.root)): sha256(path) for path in required if path.is_file()},
        "manuscript_modified": False,
    }
    destination = args.root / "aggregate" / "audit_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
