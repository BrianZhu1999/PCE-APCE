from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hilda_da.metrics import paired_bootstrap_ci, paired_effect_size


TRAINING_FREE_METHODS = (
    "hilda",
    "denkf",
    "letkf",
    "ensf",
    "iensf",
    "ensf_lr_ridge",
    "enff_f2p",
)

STEP_METRICS = {
    "state_rmse": "mean_state_rmse",
    "observation_rmse": "mean_observation_rmse",
    "crps": "mean_crps",
    "energy_score": "mean_energy_score",
    "coverage": "mean_coverage",
    "interval_width": "mean_interval_width",
    "alpha_absolute_error": "mean_alpha_absolute_error",
}
PROVENANCE_METRICS = {
    "wall_time": "elapsed_seconds",
    "peak_gpu_memory": "peak_gpu_memory_bytes",
}
PAIRED_METRICS = tuple(
    metric for metric in (*STEP_METRICS, *PROVENANCE_METRICS)
    if metric != "alpha_absolute_error"
)
METRIC_UNITS = {
    "state_rmse": "state units",
    "observation_rmse": "observation units",
    "crps": "state units",
    "energy_score": "state units",
    "coverage": "fraction",
    "interval_width": "state units",
    "alpha_absolute_error": "alpha",
    "wall_time": "seconds",
    "peak_gpu_memory": "bytes",
}

# These flags affect execution or a method implementation, not the scientific
# condition shared by methods for seed-wise pairing.
CONDITION_EXCLUDED_FLAGS = {
    "method",
    "seed",
    "output_root",
    "run_id",
    "resume_run",
    "device",
    "checkpoint_interval",
    "energy_score_chunk_size",
    "iensf_gamma",
    "iensf_variance_split_mode",
    "iensf_flow_steps",
    "iensf_refinement_iterations",
}
CONFIG_VALIDATION_EXCLUDED_FLAGS = {
    "checkpoint_interval",
    "run_id",
    "resume_run",
    "device",
}


@dataclass(frozen=True)
class ManifestJob:
    job_id: str
    arguments: dict[str, str]
    system: str
    method: str
    seed: int
    condition_id: str
    condition: dict[str, str]
    run_directory: Path


@dataclass(frozen=True)
class CompletedRun:
    job: ManifestJob
    values: dict[str, float]


class RunValidationError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate immutable formal runs with paired seed statistics"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Override every --output-root in the manifest",
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--reference-method",
        choices=TRAINING_FREE_METHODS,
        default=None,
        help="Only compare this method against all others; default is all pairs",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260805)
    return parser.parse_args()


def _argument_map(arguments: Any, job_id: str) -> dict[str, str]:
    if not isinstance(arguments, list) or not all(isinstance(value, str) for value in arguments):
        raise ValueError(f"Job {job_id} needs a string arguments array")
    result: dict[str, str] = {}
    index = 0
    while index < len(arguments):
        flag = arguments[index]
        if not flag.startswith("--") or index + 1 >= len(arguments):
            raise ValueError(f"Job {job_id} has malformed arguments near {flag!r}")
        key = flag[2:].replace("-", "_")
        value = arguments[index + 1]
        if value.startswith("--") or key in result:
            raise ValueError(f"Job {job_id} has malformed or duplicate flag {flag}")
        result[key] = value
        index += 2
    return result


def _condition(arguments: dict[str, str]) -> tuple[str, dict[str, str]]:
    payload = {
        key: value
        for key, value in sorted(arguments.items())
        if key not in CONDITION_EXCLUDED_FLAGS and key != "system"
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12], payload


def load_manifest(path: Path, results_root: Path | None = None) -> tuple[dict[str, Any], list[ManifestJob]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("jobs"), list):
        raise ValueError("Manifest must contain a jobs array")
    jobs: list[ManifestJob] = []
    identifiers: set[str] = set()
    pairing_keys: set[tuple[str, str, int, str]] = set()
    for item in raw["jobs"]:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ValueError("Every manifest job needs a string id")
        job_id = item["id"]
        if job_id in identifiers:
            raise ValueError(f"Duplicate job id: {job_id}")
        identifiers.add(job_id)
        arguments = _argument_map(item.get("arguments"), job_id)
        try:
            system = arguments["system"]
            method = arguments["method"]
            seed = int(arguments["seed"])
        except (KeyError, ValueError) as error:
            raise ValueError(f"Job {job_id} requires valid system, method, and seed") from error
        condition_id, condition = _condition(arguments)
        pairing_key = (system, condition_id, seed, method)
        if pairing_key in pairing_keys:
            raise ValueError(f"Duplicate system/condition/seed/method entry: {pairing_key}")
        pairing_keys.add(pairing_key)
        root = results_root if results_root is not None else Path(arguments.get("output_root", "results"))
        jobs.append(
            ManifestJob(
                job_id=job_id,
                arguments=arguments,
                system=system,
                method=method,
                seed=seed,
                condition_id=condition_id,
                condition=condition,
                run_directory=root / job_id,
            )
        )
    declared_count = raw.get("job_count")
    if declared_count is not None and declared_count != len(jobs):
        raise ValueError(
            f"Manifest job_count is {declared_count}, but jobs contains {len(jobs)} entries"
        )
    return raw, jobs


def _read_json(path: Path, expected_type: type) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RunValidationError(f"cannot read {path.name}: {error}") from error
    if not isinstance(value, expected_type):
        raise RunValidationError(f"{path.name} must contain a {expected_type.__name__}")
    return value


def _finite_number(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunValidationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise RunValidationError(f"{label} must be finite" + (" and non-negative" if nonnegative else ""))
    return result


def _scalar_matches(expected: str, actual: Any) -> bool:
    if isinstance(actual, bool):
        return expected.lower() == str(actual).lower()
    if isinstance(actual, (int, float)) and not isinstance(actual, bool):
        try:
            return Decimal(expected) == Decimal(str(actual))
        except InvalidOperation:
            return False
    return expected == str(actual)


def _validate_configuration(job: ManifestJob, config: dict[str, Any]) -> None:
    for key, expected in job.arguments.items():
        if key in CONFIG_VALIDATION_EXCLUDED_FLAGS:
            continue
        if key not in config:
            raise RunValidationError(f"config.json is missing manifest field {key}")
        if not _scalar_matches(expected, config[key]):
            raise RunValidationError(
                f"config.json field {key} does not match manifest: {config[key]!r} != {expected!r}"
            )


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise RunValidationError("cannot average an empty metric")
    return math.fsum(materialized) / len(materialized)


def validate_completed_run(job: ManifestJob) -> CompletedRun:
    required = ("config.json", "provenance.json", "summary.json", "metrics.json")
    missing = [name for name in required if not (job.run_directory / name).is_file()]
    if missing:
        raise RunValidationError(f"missing artifacts: {', '.join(missing)}")
    config = _read_json(job.run_directory / "config.json", dict)
    provenance = _read_json(job.run_directory / "provenance.json", dict)
    summary = _read_json(job.run_directory / "summary.json", dict)
    metrics = _read_json(job.run_directory / "metrics.json", list)

    _validate_configuration(job, config)
    if provenance.get("run_id") != job.job_id:
        raise RunValidationError("provenance run_id does not match manifest job id")
    if provenance.get("configuration") != config:
        raise RunValidationError("provenance configuration does not match config.json")
    if provenance.get("completed") is not True or provenance.get("failed") is True:
        message = provenance.get("failure_message")
        raise RunValidationError(f"run is not completed successfully: {message or 'no failure message'}")
    if not metrics or not all(isinstance(item, dict) for item in metrics):
        raise RunValidationError("metrics.json must contain at least one metric object")
    if summary.get("analysis_times") != len(metrics):
        raise RunValidationError("summary analysis_times does not match metrics.json length")

    if job.method != "hilda":
        has_alpha_estimate = any(
            item.get("alpha_estimate") is not None
            or item.get("alpha_absolute_error") is not None
            for item in metrics
        )
        if has_alpha_estimate or summary.get("mean_alpha_absolute_error") is not None:
            raise RunValidationError(
                "a non-HILDA baseline must not contain an alpha estimate or alpha error"
            )

    values: dict[str, float] = {}
    for metric, summary_key in STEP_METRICS.items():
        observed = [
            _finite_number(item[metric], f"metrics.json[{index}].{metric}")
            for index, item in enumerate(metrics)
            if item.get(metric) is not None
        ]
        if not observed:
            if summary.get(summary_key) is not None:
                raise RunValidationError(f"{summary_key} is non-null without per-step values")
            continue
        if len(observed) != len(metrics):
            raise RunValidationError(f"metric {metric} is only present for part of the run")
        recomputed = _mean(observed)
        reported = _finite_number(summary.get(summary_key), f"summary.json.{summary_key}")
        if not math.isclose(recomputed, reported, rel_tol=1e-9, abs_tol=1e-12):
            raise RunValidationError(
                f"summary {summary_key} does not match metrics.json mean"
            )
        values[metric] = recomputed

    for metric, provenance_key in PROVENANCE_METRICS.items():
        values[metric] = _finite_number(
            provenance.get(provenance_key),
            f"provenance.json.{provenance_key}",
            nonnegative=True,
        )
    return CompletedRun(job=job, values=values)


def _incomplete_reason(job: ManifestJob) -> str | None:
    config_path = job.run_directory / "config.json"
    checkpoint_path = job.run_directory / "checkpoint.pt"
    if not config_path.is_file() or not checkpoint_path.is_file():
        return None
    provenance_path = job.run_directory / "provenance.json"
    if not provenance_path.is_file():
        return "checkpoint_present_without_final_provenance"
    try:
        provenance = _read_json(provenance_path, dict)
    except RunValidationError:
        return None
    if provenance.get("completed") is not True and provenance.get("failed") is not True:
        return "checkpoint_present_for_unfinished_run"
    return None


def _sample_standard_deviation(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = _mean(values)
    return math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _bootstrap_seed(base_seed: int, *parts: str) -> int:
    digest = hashlib.sha256("\x00".join(parts).encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:8], "big")) % (2**63 - 1)


def _effect_size(first: torch.Tensor, second: torch.Tensor) -> tuple[float | None, str]:
    if first.numel() < 2:
        return None, "insufficient_pairs"
    value = float(paired_effect_size(first, second))
    if math.isinf(value):
        return None, "positive_infinity" if value > 0.0 else "negative_infinity"
    return value, "finite"


def aggregate_results(
    manifest_path: Path,
    *,
    results_root: Path | None = None,
    reference_method: str | None = None,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 20260805,
) -> dict[str, Any]:
    if bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be positive")
    if reference_method is not None and reference_method not in TRAINING_FREE_METHODS:
        raise ValueError(f"reference_method must be one of {TRAINING_FREE_METHODS}")
    manifest, jobs = load_manifest(manifest_path, results_root)
    completed: list[CompletedRun] = []
    job_status: list[dict[str, Any]] = []
    for job in jobs:
        base = {
            "job_id": job.job_id,
            "system": job.system,
            "condition_id": job.condition_id,
            "method": job.method,
            "seed": job.seed,
            "run_directory": str(job.run_directory),
        }
        if job.method not in TRAINING_FREE_METHODS:
            job_status.append({**base, "status": "excluded", "reason": "method_not_training_free"})
            continue
        if not job.run_directory.exists():
            job_status.append({**base, "status": "missing", "reason": "run_directory_missing"})
            continue
        try:
            result = validate_completed_run(job)
        except RunValidationError as error:
            incomplete = _incomplete_reason(job)
            if incomplete is not None:
                job_status.append({**base, "status": "incomplete", "reason": incomplete})
            else:
                job_status.append({**base, "status": "failed", "reason": str(error)})
            continue
        completed.append(result)
        job_status.append({**base, "status": "completed", "reason": None})

    conditions: dict[tuple[str, str], dict[str, str]] = {}
    expected: dict[tuple[str, str, str], set[int]] = {}
    values: dict[tuple[str, str, str, int], dict[str, float]] = {}
    for job in jobs:
        if job.method not in TRAINING_FREE_METHODS:
            continue
        conditions[(job.system, job.condition_id)] = job.condition
        expected.setdefault((job.system, job.condition_id, job.method), set()).add(job.seed)
    for run in completed:
        key = (run.job.system, run.job.condition_id, run.job.method, run.job.seed)
        values[key] = run.values

    method_summaries: list[dict[str, Any]] = []
    run_metrics: list[dict[str, Any]] = []
    for run in sorted(
        completed,
        key=lambda item: (item.job.system, item.job.condition_id, item.job.method, item.job.seed),
    ):
        for metric, value in sorted(run.values.items()):
            run_metrics.append(
                {
                    "job_id": run.job.job_id,
                    "system": run.job.system,
                    "condition_id": run.job.condition_id,
                    "method": run.job.method,
                    "seed": run.job.seed,
                    "metric": metric,
                    "value": value,
                    "unit": METRIC_UNITS[metric],
                }
            )

    for system, condition_id, method in sorted(expected):
        expected_seeds = expected[(system, condition_id, method)]
        completed_seeds = sorted(
            seed for seed in expected_seeds if (system, condition_id, method, seed) in values
        )
        metric_names = sorted(
            {
                metric
                for seed in completed_seeds
                for metric in values[(system, condition_id, method, seed)]
            }
        )
        for metric in metric_names:
            samples = [
                values[(system, condition_id, method, seed)][metric]
                for seed in completed_seeds
                if metric in values[(system, condition_id, method, seed)]
            ]
            method_summaries.append(
                {
                    "system": system,
                    "condition_id": condition_id,
                    "method": method,
                    "metric": metric,
                    "unit": METRIC_UNITS[metric],
                    "expected_runs": len(expected_seeds),
                    "completed_runs": len(completed_seeds),
                    "metric_count": len(samples),
                    "mean": _mean(samples),
                    "sample_standard_deviation": _sample_standard_deviation(samples),
                }
            )

    paired_comparisons: list[dict[str, Any]] = []
    for system, condition_id in sorted(conditions):
        methods = sorted(
            method
            for candidate_system, candidate_condition, method in expected
            if candidate_system == system and candidate_condition == condition_id
        )
        if reference_method is None:
            method_pairs = list(combinations(methods, 2))
        elif reference_method in methods:
            method_pairs = [(reference_method, method) for method in methods if method != reference_method]
        else:
            method_pairs = []
        for first_method, second_method in method_pairs:
            expected_pair_seeds = sorted(
                expected[(system, condition_id, first_method)]
                & expected[(system, condition_id, second_method)]
            )
            for metric in PAIRED_METRICS:
                paired_seeds = [
                    seed
                    for seed in expected_pair_seeds
                    if metric in values.get((system, condition_id, first_method, seed), {})
                    and metric in values.get((system, condition_id, second_method, seed), {})
                ]
                if not paired_seeds:
                    continue
                first = torch.tensor(
                    [values[(system, condition_id, first_method, seed)][metric] for seed in paired_seeds],
                    dtype=torch.float64,
                )
                second = torch.tensor(
                    [values[(system, condition_id, second_method, seed)][metric] for seed in paired_seeds],
                    dtype=torch.float64,
                )
                ci = paired_bootstrap_ci(
                    first,
                    second,
                    confidence=0.95,
                    resamples=bootstrap_resamples,
                    seed=_bootstrap_seed(
                        bootstrap_seed,
                        system,
                        condition_id,
                        first_method,
                        second_method,
                        metric,
                    ),
                )
                effect, effect_status = _effect_size(first, second)
                paired_comparisons.append(
                    {
                        "system": system,
                        "condition_id": condition_id,
                        "first_method": first_method,
                        "second_method": second_method,
                        "contrast": f"{first_method} - {second_method}",
                        "metric": metric,
                        "unit": METRIC_UNITS[metric],
                        "expected_pair_count": len(expected_pair_seeds),
                        "paired_count": len(paired_seeds),
                        "paired_seeds": paired_seeds,
                        "unavailable_seeds": sorted(set(expected_pair_seeds) - set(paired_seeds)),
                        "mean_difference": float(ci.estimate),
                        "bootstrap_95_ci_lower": float(ci.lower),
                        "bootstrap_95_ci_upper": float(ci.upper),
                        "cohen_dz": effect,
                        "cohen_dz_status": effect_status,
                    }
                )

    counts = {status: sum(item["status"] == status for item in job_status) for status in (
        "completed", "incomplete", "missing", "failed", "excluded"
    )}
    return {
        "schema_version": 1,
        "manifest": str(manifest_path.resolve()),
        "manifest_protocol_version": manifest.get("protocol_version"),
        "manifest_profile": manifest.get("profile"),
        "comparison_mode": (
            {"type": "all_pairs"}
            if reference_method is None
            else {"type": "reference", "reference_method": reference_method}
        ),
        "bootstrap": {
            "method": "paired percentile bootstrap",
            "confidence": 0.95,
            "resamples": bootstrap_resamples,
            "base_seed": bootstrap_seed,
        },
        "contrast_definition": "first_method - second_method",
        "metric_applicability": {
            "alpha_absolute_error": ["hilda"],
            "paired_metrics": list(PAIRED_METRICS),
            "note": "Null baseline alpha fields are unavailable, not alpha estimates.",
        },
        "job_counts": {"manifest": len(jobs), **counts},
        "conditions": [
            {"system": system, "condition_id": condition_id, "parameters": condition}
            for (system, condition_id), condition in sorted(conditions.items())
        ],
        "job_status": job_status,
        "run_metrics": run_metrics,
        "method_summaries": method_summaries,
        "paired_comparisons": paired_comparisons,
    }


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _csv_text(rows: list[dict[str, Any]], fieldnames: list[str]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        normalized = dict(row)
        for key, value in normalized.items():
            if isinstance(value, (list, dict)):
                normalized[key] = json.dumps(value, separators=(",", ":"), ensure_ascii=True)
        writer.writerow(normalized)
    return buffer.getvalue()


def write_outputs(output_directory: Path, aggregate: dict[str, Any]) -> list[Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        "formal_aggregate.json": json.dumps(
            aggregate, indent=2, ensure_ascii=False, allow_nan=False
        ) + "\n",
        "job_status.csv": _csv_text(
            aggregate["job_status"],
            ["job_id", "system", "condition_id", "method", "seed", "run_directory", "status", "reason"],
        ),
        "run_metrics.csv": _csv_text(
            aggregate["run_metrics"],
            ["job_id", "system", "condition_id", "method", "seed", "metric", "value", "unit"],
        ),
        "method_summary.csv": _csv_text(
            aggregate["method_summaries"],
            [
                "system", "condition_id", "method", "metric", "unit", "expected_runs",
                "completed_runs", "metric_count", "mean", "sample_standard_deviation",
            ],
        ),
        "paired_comparisons.csv": _csv_text(
            aggregate["paired_comparisons"],
            [
                "system", "condition_id", "first_method", "second_method", "contrast",
                "metric", "unit", "expected_pair_count", "paired_count", "paired_seeds",
                "unavailable_seeds", "mean_difference", "bootstrap_95_ci_lower",
                "bootstrap_95_ci_upper", "cohen_dz", "cohen_dz_status",
            ],
        ),
    }
    paths = []
    for name, content in outputs.items():
        path = output_directory / name
        _atomic_write_text(path, content)
        paths.append(path)
    return paths


def main() -> None:
    args = parse_args()
    aggregate = aggregate_results(
        args.manifest,
        results_root=args.results_root,
        reference_method=args.reference_method,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    outputs = write_outputs(args.output_directory, aggregate)
    print(
        json.dumps(
            {
                "outputs": [str(path) for path in outputs],
                "job_counts": aggregate["job_counts"],
                "paired_comparisons": len(aggregate["paired_comparisons"]),
            }
        )
    )


if __name__ == "__main__":
    main()
