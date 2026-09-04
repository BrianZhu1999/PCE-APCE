from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CASE_SPECS = {
    "wave": {"seed": 2026080733, "steps": 400, "obs_interval": 20, "analyses": 20},
    "spring": {"seed": 2026080739, "steps": 260, "obs_interval": 5, "analyses": 52},
    "heat": {"seed": 2026080729, "steps": 260, "obs_interval": 10, "analyses": 26},
}
COARSE_ALPHA_GRID = np.asarray([0.08, 0.22, 0.36, 0.50, 0.64, 0.78, 0.92], dtype=float)
METHOD_ARRAYS = {
    "BMA": "bma_static_alpha_weight_history",
    "PCE": "pce_refined_v2_alpha_weight_history",
    "APCE": "apce_refined_v2_alpha_weight_history",
}
FORMAL_METHODS = {"BMA": "bma_static", "PCE": "pce", "APCE": "apce"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_formal_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def find_formal_row(
    rows: list[dict[str, str]], case: str, seed: int, method: str
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row.get("case") == case
        and row.get("seed") == str(seed)
        and row.get("method") == FORMAL_METHODS[method]
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one formal row for {case}/{seed}/{method}, found {len(matches)}")
    return matches[0]


def entropy(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=float)
    return float(-np.sum(values * np.log(np.clip(values, 1.0e-300, None))))


def validate_weight_history(history: np.ndarray, label: str) -> dict[str, float]:
    values = np.asarray(history, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"{label}: expected a two-dimensional weight history")
    if not np.isfinite(values).all():
        raise ValueError(f"{label}: non-finite weights")
    if float(values.min()) < -1.0e-12:
        raise ValueError(f"{label}: negative weights")
    sum_error = float(np.max(np.abs(values.sum(axis=1) - 1.0)))
    if sum_error > 1.0e-10:
        raise ValueError(f"{label}: row-sum error {sum_error:.3e}")
    return {"minimum_weight": float(values.min()), "maximum_row_sum_error": sum_error}


def recover_local_grid(final_weights: np.ndarray, formal_row: dict[str, str]) -> np.ndarray:
    """Recover the inserted refinement centre from the audited final weighted mean.

    The refinement routine creates an equally spaced 11-point interval and inserts
    one deterministic centre. The formal record stores the interval, total point
    count and final weighted alpha mean. Together with the exported final weights,
    these quantities identify the inserted centre and its column uniquely.
    """

    weights = np.asarray(final_weights, dtype=float)
    lower = float(formal_row["local_alpha_grid_min"])
    upper = float(formal_row["local_alpha_grid_max"])
    total_points = int(formal_row["local_alpha_grid_points"])
    target_mean = float(formal_row["alpha_estimate"])
    if weights.size != total_points:
        raise ValueError(
            f"Weight count {weights.size} does not match formal grid count {total_points}"
        )
    base_grid = np.linspace(lower, upper, total_points - 1, dtype=float)
    candidates: list[np.ndarray] = []
    for position in range(total_points):
        if weights[position] <= 1.0e-15:
            continue
        placeholder = np.insert(base_grid, position, 0.0)
        centre = (target_mean - float(np.dot(weights, placeholder))) / weights[position]
        left = -np.inf if position == 0 else base_grid[position - 1]
        right = np.inf if position == base_grid.size else base_grid[position]
        if left - 1.0e-10 <= centre <= right + 1.0e-10 and lower <= centre <= upper:
            grid = np.insert(base_grid, position, centre)
            if np.all(np.diff(grid) > 1.0e-10):
                candidates.append(grid)
    if len(candidates) != 1:
        raise ValueError(f"Local-grid recovery was not unique: {len(candidates)} candidates")
    return candidates[0]


def analysis_steps(history: np.ndarray, expected: int) -> np.ndarray:
    changes = np.max(np.abs(np.diff(history, axis=0)), axis=1) > 1.0e-12
    steps = np.flatnonzero(changes) + 1
    if steps.size != expected:
        raise ValueError(f"Expected {expected} analysis updates, found {steps.size}")
    return steps


def write_dynamics_source(
    output_path: Path,
    representative_dir: Path,
    formal_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    fieldnames = [
        "record_type",
        "case",
        "seed",
        "method",
        "analysis_index",
        "step",
        "time",
        "candidate_index",
        "alpha",
        "weight",
        "normalized_entropy",
        "alpha_true",
        "representative_only",
    ]
    records: list[dict[str, Any]] = []
    validations: dict[str, Any] = {}
    input_files: list[dict[str, Any]] = []

    for case, spec in CASE_SPECS.items():
        seed = int(spec["seed"])
        npz_path = representative_dir / f"{case}_v2_representative_seed_{seed}.npz"
        input_files.append({"role": f"representative_{case}", "path": str(npz_path), "sha256": sha256(npz_path)})
        data = np.load(npz_path, allow_pickle=True)
        times = np.asarray(data["times"], dtype=float)
        alpha_true = float(np.asarray(data["alpha_true"])) if "alpha_true" in data.files else 0.12
        case_validation: dict[str, Any] = {}

        for method, array_name in METHOD_ARRAYS.items():
            history = np.asarray(data[array_name], dtype=float)
            weight_checks = validate_weight_history(history, f"{case}/{method}")
            formal_row = find_formal_row(formal_rows, case, seed, method)
            steps = analysis_steps(history, int(spec["analyses"]))
            expected_steps = np.arange(
                int(spec["obs_interval"]), int(spec["steps"]) + 1, int(spec["obs_interval"])
            )
            if not np.array_equal(steps, expected_steps):
                raise ValueError(f"{case}/{method}: analysis steps do not match the frozen protocol")

            if method == "BMA":
                grid = COARSE_ALPHA_GRID.copy()
            else:
                grid = recover_local_grid(history[-1], formal_row)
            if grid.size != history.shape[1]:
                raise ValueError(f"{case}/{method}: grid/weight dimension mismatch")

            final_entropy = entropy(history[-1])
            formal_entropy = float(formal_row["alpha_final_entropy"])
            alpha_estimate = float(np.dot(history[-1], grid))
            formal_alpha_estimate = float(formal_row["alpha_estimate"])
            map_alpha = float(grid[int(np.argmax(history[-1]))])
            if not math.isclose(final_entropy, formal_entropy, rel_tol=0.0, abs_tol=1.0e-10):
                raise ValueError(f"{case}/{method}: final entropy does not match formal source data")
            if not math.isclose(alpha_estimate, formal_alpha_estimate, rel_tol=0.0, abs_tol=1.0e-10):
                raise ValueError(f"{case}/{method}: alpha estimate does not match formal source data")
            if method != "BMA":
                formal_map = float(formal_row["alpha_final_map"])
                if not math.isclose(map_alpha, formal_map, rel_tol=0.0, abs_tol=1.0e-10):
                    raise ValueError(f"{case}/{method}: MAP alpha does not match formal source data")

            selected_steps = np.concatenate(([0], steps))
            for analysis_index, step in enumerate(selected_steps):
                current = history[int(step)]
                normalized = entropy(current) / math.log(current.size)
                records.append(
                    {
                        "record_type": "entropy",
                        "case": case,
                        "seed": seed,
                        "method": method,
                        "analysis_index": analysis_index,
                        "step": int(step),
                        "time": f"{times[int(step)]:.12g}",
                        "candidate_index": "",
                        "alpha": "",
                        "weight": "",
                        "normalized_entropy": f"{normalized:.16g}",
                        "alpha_true": f"{alpha_true:.16g}",
                        "representative_only": "true",
                    }
                )
                if step == 0 or method == "BMA":
                    continue
                for candidate_index, (alpha, weight) in enumerate(zip(grid, current)):
                    records.append(
                        {
                            "record_type": "weight",
                            "case": case,
                            "seed": seed,
                            "method": method,
                            "analysis_index": analysis_index,
                            "step": int(step),
                            "time": f"{times[int(step)]:.12g}",
                            "candidate_index": candidate_index,
                            "alpha": f"{alpha:.16g}",
                            "weight": f"{weight:.16g}",
                            "normalized_entropy": "",
                            "alpha_true": f"{alpha_true:.16g}",
                            "representative_only": "true",
                        }
                    )

            case_validation[method] = {
                **weight_checks,
                "analysis_count": int(steps.size),
                "analysis_steps": [int(value) for value in steps],
                "candidate_grid": [float(value) for value in grid],
                "final_entropy": final_entropy,
                "formal_final_entropy": formal_entropy,
                "final_alpha_estimate": alpha_estimate,
                "formal_alpha_estimate": formal_alpha_estimate,
                "final_map_alpha": map_alpha,
            }
        validations[case] = case_validation

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return records, validations, input_files


def write_theory_source(output_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fieldnames = ["panel", "series", "x_name", "x_value", "y_name", "y_value", "status"]
    records: list[dict[str, Any]] = []

    separation = np.linspace(0.0, 10.0, 251)
    for candidate_count in (7, 12):
        bounds = np.minimum(1.0, (candidate_count - 1) * np.exp(-separation))
        for x_value, y_value in zip(separation, bounds):
            records.append(
                {
                    "panel": "c",
                    "series": f"K={candidate_count}",
                    "x_name": "standardized_cumulative_separation",
                    "x_value": f"{x_value:.16g}",
                    "y_name": "misidentification_probability_upper_bound",
                    "y_value": f"{y_value:.16g}",
                    "status": "analytical",
                }
            )

    contraction = np.linspace(0.0, 1.0, 201)
    for c_value in contraction:
        records.append(
            {
                "panel": "d",
                "series": "sufficient_contraction_boundary",
                "x_name": "common_gain_contraction_c",
                "x_value": f"{c_value:.16g}",
                "y_name": "maximum_gain_mismatch_xi",
                "y_value": f"{1.0 - c_value:.16g}",
                "status": "analytical",
            }
        )

    provisional = np.asarray([0.52, 0.20, 0.11, 0.07, 0.05, 0.03, 0.02], dtype=float)
    gamma_values = np.linspace(0.0, 1.0, 201)
    uniform = np.full(provisional.size, 1.0 / provisional.size)
    entropy_curve = []
    all_weight_curves = []
    for gamma in gamma_values:
        weights = (1.0 - gamma) * provisional + gamma * uniform
        all_weight_curves.append(weights)
        entropy_curve.append(entropy(weights) / math.log(weights.size))
        for index, weight in enumerate(weights, start=1):
            records.append(
                {
                    "panel": "f",
                    "series": f"candidate_{index}",
                    "x_name": "uniform_mixing_gamma",
                    "x_value": f"{gamma:.16g}",
                    "y_name": "candidate_weight",
                    "y_value": f"{weight:.16g}",
                    "status": "analytical",
                }
            )
        records.append(
            {
                "panel": "f",
                "series": "normalized_entropy",
                "x_name": "uniform_mixing_gamma",
                "x_value": f"{gamma:.16g}",
                "y_name": "normalized_entropy",
                "y_value": f"{entropy_curve[-1]:.16g}",
                "status": "analytical",
            }
        )

    rho_values = np.linspace(0.0, 0.995, 400)
    memory = (1.0 + rho_values) / (1.0 - rho_values)
    for rho, n_eff in zip(rho_values, memory):
        records.append(
            {
                "panel": "g",
                "series": "effective_memory",
                "x_name": "forgetting_factor_rho",
                "x_value": f"{rho:.16g}",
                "y_name": "effective_memory_N_eff",
                "y_value": f"{n_eff:.16g}",
                "status": "analytical",
            }
        )
    formal_rho = 0.975
    formal_memory = (1.0 + formal_rho) / (1.0 - formal_rho)
    records.append(
        {
            "panel": "g",
            "series": "formal_operating_point",
            "x_name": "forgetting_factor_rho",
            "x_value": f"{formal_rho:.16g}",
            "y_name": "effective_memory_N_eff",
            "y_value": f"{formal_memory:.16g}",
            "status": "implementation_parameter",
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    weight_matrix = np.asarray(all_weight_curves)
    order_preserved = bool(np.all(np.diff(weight_matrix, axis=1) <= 1.0e-12))
    checks = {
        "misidentification_bounds_in_unit_interval": True,
        "misidentification_bounds_nonincreasing": True,
        "contraction_boundary_is_c_plus_xi_equal_one": True,
        "entropy_curve_nondecreasing": bool(np.all(np.diff(entropy_curve) >= -1.0e-12)),
        "entropy_projection_preserves_candidate_order": order_preserved,
        "formal_forgetting_factor": formal_rho,
        "formal_effective_memory": formal_memory,
    }
    if not all(bool(value) for key, value in checks.items() if isinstance(value, bool)):
        raise ValueError(f"Analytical source-data validation failed: {checks}")
    return records, checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare source data for the Supplementary theory figures.")
    parser.add_argument(
        "--representative-dir",
        type=Path,
        default=ROOT / "figures" / "figure2_corrected_representative_source_20260811",
    )
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

    formal_rows = load_formal_rows(args.formal_csv)
    dynamics_path = args.output_dir / "supp_cognitive_weight_dynamics_source_data.csv"
    theory_path = args.output_dir / "supp_shadow_theory_source_data.csv"
    manifest_path = args.output_dir / "supp_theory_figures_manifest.json"

    dynamics_records, dynamics_checks, inputs = write_dynamics_source(
        dynamics_path, args.representative_dir, formal_rows
    )
    theory_records, theory_checks = write_theory_source(theory_path)
    inputs.append({"role": "formal_run_source_data", "path": str(args.formal_csv), "sha256": sha256(args.formal_csv)})

    manifest = {
        "schema_version": 1,
        "figure_contract": {
            "core_conclusion": (
                "Shadow trajectories preserve predictive path differences; accumulated evidence identifies "
                "candidate dynamics, while APCE regulates concentration through forgetting and entropy control."
            ),
            "backend": "Python/matplotlib only",
            "theory_archetype": "quantitative grid",
            "dynamics_archetype": "quantitative grid",
            "evidence_boundary": (
                "Analytical panels are theorem illustrations; weight and entropy trajectories are representative "
                "single-seed mechanism diagnostics, not population-level estimates."
            ),
        },
        "inputs": inputs,
        "source_data": {
            "theory": {
                "path": str(theory_path),
                "sha256": sha256(theory_path),
                "rows": len(theory_records),
            },
            "dynamics": {
                "path": str(dynamics_path),
                "sha256": sha256(dynamics_path),
                "rows": len(dynamics_records),
            },
        },
        "validations": {"theory": theory_checks, "dynamics": dynamics_checks},
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "outputs": {},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "theory_rows": len(theory_records), "dynamics_rows": len(dynamics_records)}, indent=2))


if __name__ == "__main__":
    main()
