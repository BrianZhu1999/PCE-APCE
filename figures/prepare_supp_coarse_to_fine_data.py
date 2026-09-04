from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASES = ("wave", "spring", "heat")
METHODS = ("pce", "apce")
METHOD_LABELS = {"pce": "PCE", "apce": "APCE"}
CASE_LABELS = {"wave": "Wave", "spring": "Spring", "heat": "Heat"}
BOOTSTRAP_SEED = 2026081203
N_BOOTSTRAP = 10_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_float(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not np.isfinite(value):
        raise ValueError(f"Non-finite {key} for {row.get('case')}/{row.get('method')}/{row.get('seed')}")
    return value


def bootstrap_mean_ci(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or data.size == 0 or not np.isfinite(data).all():
        raise ValueError("Bootstrap input must be a non-empty finite vector")
    rng = np.random.default_rng(seed)
    means = np.empty(N_BOOTSTRAP, dtype=float)
    batch = 500
    for start in range(0, N_BOOTSTRAP, batch):
        stop = min(start + batch, N_BOOTSTRAP)
        draws = rng.integers(0, data.size, size=(stop - start, data.size))
        means[start:stop] = data[draws].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(data.mean()), float(low), float(high)


def prepare_records(formal_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = [
        row
        for row in formal_rows
        if row.get("case") in CASES and row.get("method") in METHODS
    ]
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        grouped[(row["case"], row["method"])].append(row)

    records: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}
    seed_offset = 0
    for case in CASES:
        checks[case] = {}
        for method in METHODS:
            rows = sorted(grouped[(case, method)], key=lambda row: int(row["seed"]))
            if len(rows) != 50:
                raise ValueError(f"Expected 50 formal rows for {case}/{method}, found {len(rows)}")
            if len({int(row["seed"]) for row in rows}) != 50:
                raise ValueError(f"Duplicate seeds for {case}/{method}")

            nrmse_log_ratio: list[float] = []
            alpha_error_change: list[float] = []
            captured = 0
            for row in rows:
                if row.get("status") != "completed" or row.get("valid", "").lower() != "true":
                    raise ValueError(f"Invalid formal record for {case}/{method}/{row.get('seed')}")
                alpha_true = finite_float(row, "alpha_true")
                grid_min = finite_float(row, "local_alpha_grid_min")
                grid_max = finite_float(row, "local_alpha_grid_max")
                grid_points = int(float(row["local_alpha_grid_points"]))
                coarse_nrmse = finite_float(row, "coarse_pass_nrmse")
                refined_nrmse = finite_float(row, "nrmse")
                coarse_alpha_error = finite_float(row, "coarse_alpha_mean_error")
                refined_alpha_error = finite_float(row, "alpha_absolute_error")
                if not (grid_min <= grid_max and grid_points >= 2):
                    raise ValueError(f"Invalid local grid for {case}/{method}/{row['seed']}")
                if coarse_nrmse <= 0 or refined_nrmse <= 0:
                    raise ValueError(f"nRMSE must be positive for {case}/{method}/{row['seed']}")
                capture = grid_min <= alpha_true <= grid_max
                captured += int(capture)
                log_ratio = float(np.log2(refined_nrmse / coarse_nrmse))
                error_change = refined_alpha_error - coarse_alpha_error
                nrmse_log_ratio.append(log_ratio)
                alpha_error_change.append(error_change)
                records.append(
                    {
                        "record_type": "seed",
                        "case": case,
                        "case_label": CASE_LABELS[case],
                        "method": METHOD_LABELS[method],
                        "seed": int(row["seed"]),
                        "alpha_true": f"{alpha_true:.16g}",
                        "coarse_alpha_estimate": row["coarse_alpha_mean"],
                        "refined_alpha_estimate": row["alpha_estimate"],
                        "coarse_alpha_error": f"{coarse_alpha_error:.16g}",
                        "refined_alpha_error": f"{refined_alpha_error:.16g}",
                        "alpha_error_change": f"{error_change:.16g}",
                        "coarse_nrmse": f"{coarse_nrmse:.16g}",
                        "refined_nrmse": f"{refined_nrmse:.16g}",
                        "nrmse_log2_ratio": f"{log_ratio:.16g}",
                        "local_grid_min": f"{grid_min:.16g}",
                        "local_grid_max": f"{grid_max:.16g}",
                        "local_grid_points": grid_points,
                        "truth_captured": str(capture).lower(),
                        "mean": "",
                        "ci_low": "",
                        "ci_high": "",
                        "n": "",
                    }
                )

            metric_values = {
                "nrmse_log2_ratio": np.asarray(nrmse_log_ratio, dtype=float),
                "alpha_error_change": np.asarray(alpha_error_change, dtype=float),
            }
            summary: dict[str, Any] = {
                "n": len(rows),
                "truth_capture_count": captured,
                "truth_capture_rate": captured / len(rows),
                "refined_nrmse_lower_count": int(np.sum(metric_values["nrmse_log2_ratio"] < 0)),
                "refined_alpha_error_lower_count": int(np.sum(metric_values["alpha_error_change"] < 0)),
            }
            if captured != len(rows):
                raise ValueError(f"Local grid failed to capture alpha_true for {case}/{method}")
            for metric_index, (metric, values) in enumerate(metric_values.items()):
                mean, low, high = bootstrap_mean_ci(
                    values, BOOTSTRAP_SEED + seed_offset + metric_index
                )
                summary[metric] = {"mean": mean, "ci_low": low, "ci_high": high}
                records.append(
                    {
                        "record_type": "summary",
                        "case": case,
                        "case_label": CASE_LABELS[case],
                        "method": METHOD_LABELS[method],
                        "seed": "",
                        "alpha_true": "",
                        "coarse_alpha_estimate": "",
                        "refined_alpha_estimate": "",
                        "coarse_alpha_error": "",
                        "refined_alpha_error": "",
                        "alpha_error_change": "" if metric != "alpha_error_change" else f"{mean:.16g}",
                        "coarse_nrmse": "",
                        "refined_nrmse": "",
                        "nrmse_log2_ratio": "" if metric != "nrmse_log2_ratio" else f"{mean:.16g}",
                        "local_grid_min": "",
                        "local_grid_max": "",
                        "local_grid_points": "",
                        "truth_captured": "",
                        "mean": f"{mean:.16g}",
                        "ci_low": f"{low:.16g}",
                        "ci_high": f"{high:.16g}",
                        "n": len(rows),
                    }
                )
            checks[case][METHOD_LABELS[method]] = summary
            seed_offset += 10
    return records, checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare formal coarse-to-fine source data.")
    parser.add_argument(
        "--formal-csv",
        type=Path,
        default=ROOT
        / "ncs_chinese_submission"
        / "source_data"
        / "figure2_corrected_dimension_score_formal_50seeds_20260811"
        / "figure2_corrected_formal_run_source_data.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "ncs_english_latex" / "source_data"
    )
    args = parser.parse_args()

    rows = load_rows(args.formal_csv)
    records, checks = prepare_records(rows)
    output_path = args.output_dir / "supp_coarse_to_fine_validation_source_data.csv"
    manifest_path = args.output_dir / "supp_coarse_to_fine_validation_manifest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0])
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    manifest = {
        "schema_version": 1,
        "figure_contract": {
            "core_conclusion": (
                "Local refinement consistently reduces state nRMSE and retains the generating cognitive "
                "coordinate inside the local search interval, while changes in cognitive-coordinate error "
                "remain system- and method-dependent."
            ),
            "archetype": "quantitative grid",
            "backend": "Python/matplotlib only",
            "evidence_boundary": (
                "All panels use the corrected formal 50-paired-seed records. Refinement is not claimed to "
                "uniformly reduce cognitive-coordinate error."
            ),
        },
        "input": {"path": str(args.formal_csv), "sha256": sha256(args.formal_csv), "rows": len(rows)},
        "source_data": {"path": str(output_path), "sha256": sha256(output_path), "rows": len(records)},
        "bootstrap": {"resamples": N_BOOTSTRAP, "base_seed": BOOTSTRAP_SEED},
        "validations": checks,
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "outputs": {},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"source_data": str(output_path), "rows": len(records), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
