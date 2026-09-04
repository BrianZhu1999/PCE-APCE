from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

CASES = ("wave", "spring", "heat")
METHODS = ("denkf", "letkf", "iensf", "aug_enkf", "bma_static", "pce", "apce")
LABELS = {
    "denkf": "DEnKF",
    "letkf": "LETKF",
    "iensf": "IEnSF",
    "aug_enkf": "Aug-EnKF",
    "bma_static": "BMA",
    "pce": "PCE",
    "apce": "APCE",
}
METRICS = (
    "nrmse", "rmse", "crps", "coverage_90", "interval_width_90",
    "alpha_absolute_error", "runtime_seconds", "forward_member_steps",
    "peak_gpu_memory_mb",
)


def numeric(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def valid(row: dict[str, Any]) -> bool:
    return (
        row.get("status", "completed") == "completed"
        and str(row.get("valid", "")).lower() in {"true", "1", "yes"}
        and row.get("case") in CASES
        and row.get("method") in METHODS
        and math.isfinite(numeric(row.get("sensitivity_scale")))
    )


def read_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("seed_*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if valid(row):
            row["label"] = LABELS[row["method"]]
            row["_path"] = str(path)
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row if not key.startswith("_")})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["case"], float(row["sensitivity_scale"]), row["method"]), []).append(row)
    output: list[dict[str, Any]] = []
    for case in CASES:
        scales = sorted({scale for c, scale, _ in groups if c == case})
        for scale in scales:
            for method in METHODS:
                subset = groups.get((case, scale, method), [])
                if not subset:
                    continue
                item = {
                    "case": case,
                    "sensitivity_scale": scale,
                    "method": method,
                    "label": LABELS[method],
                    "n": len(subset),
                }
                for metric in METRICS:
                    values = np.asarray([numeric(row.get(metric)) for row in subset])
                    values = values[np.isfinite(values)]
                    item[f"{metric}_mean"] = float(values.mean()) if values.size else ""
                    item[f"{metric}_sd"] = float(values.std(ddof=1)) if values.size > 1 else (0.0 if values.size else "")
                output.append(item)
    return output


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Figure 2 s_theta sensitivity results.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-seeds", type=int, default=5)
    args = parser.parse_args()
    rows = read_rows(args.root)
    if not rows:
        raise SystemExit("No valid sensitivity rows found.")
    summary = summarize(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "figure2_stheta_sensitivity_run_source_data.csv", rows)
    write_csv(args.output / "figure2_stheta_sensitivity_summary.csv", summary)
    expected = len(CASES) * len(METHODS) * len({float(r["sensitivity_scale"]) for r in rows}) * args.expected_seeds
    payload = {
        "protocol": rows[0].get("protocol"),
        "root": str(args.root),
        "valid_rows": len(rows),
        "expected_rows_if_complete": expected,
        "scales": sorted({float(r["sensitivity_scale"]) for r in rows}),
        "cases": CASES,
        "methods": METHODS,
        "source_hashes": sorted({r.get("source_hash") for r in rows}),
        "file_sha256": sha256_file(args.output / "figure2_stheta_sensitivity_run_source_data.csv"),
    }
    (args.output / "figure2_stheta_sensitivity_aggregate.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
