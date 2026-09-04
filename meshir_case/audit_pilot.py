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
    records = list((args.root / "smoke").glob("s1/fold*/*/*/seed_*.json")) + list((args.root / "smoke3").glob("s32/fold*/*/*/seed_*.json"))
    for path in records:
        record = json.loads(path.read_text(encoding="utf-8"))
        if not record.get("completed"):
            errors.append(f"incomplete {path}")
        if record.get("device") not in ("cuda:2", "cuda:3"):
            errors.append(f"disallowed device {path}")
        if record.get("test_truth_used_for_fit"):
            errors.append(f"test leakage flag {path}")
        numeric = [value for key, value in record.items() if isinstance(value, (int, float)) and key not in ("seed", "fold")]
        if not np.isfinite(numeric).all():
            errors.append(f"non-finite metrics {path}")
    model_errors = []
    for path in sorted((args.root / "models").glob("*.npz")):
        with np.load(path) as model:
            if "rho" in model and not 0 <= float(model["rho"]) < 1:
                model_errors.append(f"unstable residual rho {path}")
            for key in model.files:
                if model[key].dtype.kind in "fc" and not np.isfinite(model[key]).all():
                    model_errors.append(f"nonfinite {key} {path}")
    errors.extend(model_errors)
    required = [
        args.root / "cache" / "data_manifest.json",
        args.root / "models" / "model_manifest.json",
        args.root / "aggregate" / "s1_run_source_data.csv",
        args.root / "aggregate" / "s32_run_source_data.csv",
        args.root / "aggregate" / "summary_metrics.csv",
        args.root / "aggregate" / "paired_comparisons.csv",
        args.root / "aggregate" / "meshir_admission.json",
        args.root / "aggregate" / "MESHIR_PILOT_ADMISSION_REPORT.md",
        args.root / "assets" / "meshir_pre_admission_diagnostic.svg",
        args.root / "assets" / "meshir_pre_admission_diagnostic.pdf",
        args.root / "assets" / "meshir_pre_admission_diagnostic.png",
        args.root / "assets" / "meshir_pre_admission_diagnostic.tiff",
        args.root / "assets" / "meshir_pre_admission_figure_source_data.csv",
        args.root / "assets" / "meshir_pre_admission_figure_registry.json",
    ]
    for path in required:
        if not path.is_file():
            errors.append(f"missing {path}")
    code_files = sorted(HERE.glob("*.py")) + sorted((HERE / "meshir").glob("*.py")) + sorted((HERE / "tests").glob("*.py")) + [HERE / "config.json"]
    manifest = {
        "case": "meshir_s1_s32_pilot",
        "smoke_records": len(records),
        "full_192_run_matrix_launched": False,
        "errors": errors,
        "passed": not errors,
        "code_hashes": {str(path.relative_to(HERE)): sha256(path) for path in code_files if path.is_file()},
        "output_hashes": {str(path.relative_to(args.root)): sha256(path) for path in required if path.is_file()},
        "manuscript_modified": False,
    }
    destination = args.root / "aggregate" / "meshir_audit_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
