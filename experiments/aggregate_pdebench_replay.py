from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hilda_da.metrics import paired_bootstrap_ci, paired_effect_size


METHODS = ("hilda", "denkf", "letkf", "ensf", "iensf", "ensf_lr_ridge", "enff_f2p")
BASELINES = tuple(method for method in METHODS if method != "hilda")
FILE_INDICES = tuple(range(5))
TRAJECTORY_INDICES = tuple(range(4))
EXPECTED_JOB_COUNT = len(FILE_INDICES) * len(TRAJECTORY_INDICES) * len(METHODS)
PROTOCOL_VERSION = "2026-08-05-pdebench-replay-v1"

MEAN_METRICS = (
    "state_rmse",
    "observation_rmse",
    "crps",
    "energy_score",
    "coverage",
    "interval_width",
    "cycle_seconds",
)
METRICS = (*MEAN_METRICS, "peak_gpu_memory_bytes")
METRIC_UNITS = {
    "state_rmse": "state units",
    "observation_rmse": "observation units",
    "crps": "state units",
    "energy_score": "state units",
    "coverage": "fraction",
    "interval_width": "state units",
    "cycle_seconds": "seconds",
    "peak_gpu_memory_bytes": "bytes",
}
FORBIDDEN_ALPHA_FIELDS = {
    "alpha_true",
    "alpha_estimate",
    "alpha_rmse",
    "alpha_absolute_error",
}
CONFIG_EXCLUDED_FLAGS = {"checkpoint_interval", "run_id", "resume_run", "device"}
PATH_FLAGS = {"data", "manifest", "collection_validation", "output_root"}
JOB_ID_PATTERN = re.compile(
    r"^pdebench_f(?P<file>\d+)_tr(?P<trajectory>\d+)_(?P<method>.+)_s(?P<seed>\d+)$"
)
DATA_FILE_PATTERN = re.compile(r"(?:^|[\\/])ns_incom_inhom_2d_512-(?P<file>\d+)\.h5$")


@dataclass(frozen=True)
class ReplayJob:
    job_id: str
    arguments: dict[str, str]
    file_index: int
    trajectory_index: int
    method: str
    seed: int
    run_directory: Path

    @property
    def trajectory_id(self) -> str:
        return f"f{self.file_index}_tr{self.trajectory_index}"


@dataclass(frozen=True)
class CompletedReplay:
    job: ReplayJob
    values: dict[str, float]


class ReplayValidationError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate paired PDEBench NS_Incom replay trajectories"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260806)
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


def _data_file_index(value: str) -> int | None:
    match = DATA_FILE_PATTERN.search(value)
    return int(match.group("file")) if match else None


def load_replay_manifest(
    path: Path,
    results_root: Path | None = None,
) -> tuple[dict[str, Any], list[ReplayJob]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("jobs"), list):
        raise ValueError("Replay manifest must contain a jobs array")
    if raw.get("profile") != "pdebench_replay":
        raise ValueError("Replay manifest profile must be pdebench_replay")
    if raw.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported PDEBench replay protocol: {raw.get('protocol_version')!r}")
    if raw.get("job_count") != len(raw["jobs"]):
        raise ValueError("Replay manifest job_count does not match jobs length")

    identifiers: set[str] = set()
    recognized: dict[tuple[int, int, str], ReplayJob] = {}
    jobs: list[ReplayJob] = []
    trajectory_seeds: dict[tuple[int, int], int] = {}
    for item in raw["jobs"]:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ValueError("Every replay job needs a string id")
        job_id = item["id"]
        if job_id in identifiers:
            raise ValueError(f"Duplicate replay job id: {job_id}")
        identifiers.add(job_id)
        match = JOB_ID_PATTERN.fullmatch(job_id)
        if match is None:
            raise ValueError(f"Replay job id does not encode file/trajectory/method/seed: {job_id}")
        arguments = _argument_map(item.get("arguments"), job_id)
        try:
            file_index = int(match.group("file"))
            trajectory_index = int(match.group("trajectory"))
            method = match.group("method")
            seed = int(match.group("seed"))
            argument_trajectory = int(arguments["trajectory_index"])
            argument_seed = int(arguments["seed"])
            argument_method = arguments["method"]
            argument_file = _data_file_index(arguments["data"])
        except (KeyError, ValueError) as error:
            raise ValueError(f"Replay job {job_id} has invalid identity arguments") from error
        if (
            argument_file != file_index
            or argument_trajectory != trajectory_index
            or argument_method != method
            or argument_seed != seed
        ):
            raise ValueError(f"Replay job {job_id} identity disagrees with its arguments")
        root = results_root if results_root is not None else Path(arguments.get("output_root", "results"))
        job = ReplayJob(
            job_id=job_id,
            arguments=arguments,
            file_index=file_index,
            trajectory_index=trajectory_index,
            method=method,
            seed=seed,
            run_directory=root / job_id,
        )
        jobs.append(job)
        if method not in METHODS:
            continue
        key = (file_index, trajectory_index, method)
        if key in recognized:
            raise ValueError(f"Duplicate replay identity: {key}")
        recognized[key] = job
        trajectory_key = (file_index, trajectory_index)
        previous_seed = trajectory_seeds.setdefault(trajectory_key, seed)
        if previous_seed != seed:
            raise ValueError(f"Methods for trajectory {trajectory_key} do not share a seed")

    expected = {
        (file_index, trajectory_index, method)
        for file_index in FILE_INDICES
        for trajectory_index in TRAJECTORY_INDICES
        for method in METHODS
    }
    actual = set(recognized)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"Replay manifest must identify exactly {EXPECTED_JOB_COUNT} frozen jobs; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    if len(set(trajectory_seeds.values())) != len(FILE_INDICES) * len(TRAJECTORY_INDICES):
        raise ValueError("Every PDEBench trajectory must have a distinct paired seed")
    return raw, jobs


def _read_json(path: Path, expected_type: type) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReplayValidationError(f"cannot read {path.name}: {error}") from error
    if not isinstance(value, expected_type):
        raise ReplayValidationError(f"{path.name} must contain a {expected_type.__name__}")
    return value


def _finite_number(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayValidationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ReplayValidationError(f"{label} must be finite" + (" and non-negative" if nonnegative else ""))
    return result


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ReplayValidationError("cannot average an empty metric")
    return math.fsum(materialized) / len(materialized)


def _scalar_matches(expected: str, actual: Any) -> bool:
    if isinstance(actual, bool):
        return expected.lower() == str(actual).lower()
    if isinstance(actual, (int, float)) and not isinstance(actual, bool):
        try:
            return Decimal(expected) == Decimal(str(actual))
        except InvalidOperation:
            return False
    return expected == str(actual)


def _normalized_path(value: Any) -> str:
    return str(value).replace("\\", "/").rstrip("/")


def _path_matches(expected: str, actual: Any) -> bool:
    expected_normalized = _normalized_path(expected)
    actual_normalized = _normalized_path(actual)
    return (
        actual_normalized == expected_normalized
        or actual_normalized.endswith("/" + expected_normalized.lstrip("/"))
    )


def _validate_config(job: ReplayJob, config: dict[str, Any]) -> None:
    for key, expected in job.arguments.items():
        if key in CONFIG_EXCLUDED_FLAGS:
            continue
        if key not in config:
            raise ReplayValidationError(f"config.json is missing manifest field {key}")
        if key == "data":
            matches = _data_file_index(str(config[key])) == job.file_index
        elif key in PATH_FLAGS:
            matches = _path_matches(expected, config[key])
        else:
            matches = _scalar_matches(expected, config[key])
        if not matches:
            raise ReplayValidationError(f"config.json field {key} does not match the manifest")

    stride = int(job.arguments["spatial_stride"])
    expected_times = list(
        range(
            int(job.arguments["time_start"]),
            int(job.arguments["time_stop"]),
            int(job.arguments["time_step"]),
        )
    )
    required = {
        "source_grid": [512, 512],
        "assimilation_grid": [512 // stride, 512 // stride],
        "channel_order": ["velocity_x", "velocity_y"],
        "state_dim": 2 * (512 // stride) ** 2,
        "selected_raw_time_indices": expected_times,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ReplayValidationError(f"config.json has invalid {key}")
    forecast = config.get("forecast_protocol")
    if not isinstance(forecast, dict):
        raise ReplayValidationError("config.json is missing forecast_protocol")
    if (
        forecast.get("name") != job.arguments["forecast_model"]
        or forecast.get("future_target_frames_used_by_forecast") is not False
        or forecast.get("normalization") != "none"
    ):
        raise ReplayValidationError("config.json forecast protocol violates replay contract")


def _contains_forbidden_alpha_fields(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            key == "alpha" or key.startswith("alpha_") or _contains_forbidden_alpha_fields(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_alpha_fields(child) for child in value)
    return False


def validate_completed_replay(job: ReplayJob) -> CompletedReplay:
    required = ("config.json", "provenance.json", "summary.json", "metrics.json")
    missing = [name for name in required if not (job.run_directory / name).is_file()]
    if missing:
        raise ReplayValidationError(f"missing artifacts: {', '.join(missing)}")
    config = _read_json(job.run_directory / "config.json", dict)
    provenance = _read_json(job.run_directory / "provenance.json", dict)
    summary = _read_json(job.run_directory / "summary.json", dict)
    metrics = _read_json(job.run_directory / "metrics.json", list)
    _validate_config(job, config)

    if _contains_forbidden_alpha_fields(summary) or _contains_forbidden_alpha_fields(metrics):
        raise ReplayValidationError("external replay must not contain alpha truth or alpha error metrics")
    if provenance.get("completed") is not True or provenance.get("failure") is not None:
        raise ReplayValidationError(f"run is not completed successfully: {provenance.get('failure')}")
    provenance_required = {
        "run_id": job.job_id,
        "method": job.method,
        "dataset": "PDEBench NS_Incom",
        "source_grid": [512, 512],
        "assimilation_grid": config["assimilation_grid"],
        "channels": ["velocity_x", "velocity_y"],
        "pressure_present": False,
        "normalization": "none",
        "forecast_uses_future_target_frames": False,
        "forecast_model": job.arguments["forecast_model"],
    }
    for key, expected in provenance_required.items():
        if provenance.get(key) != expected:
            raise ReplayValidationError(f"provenance.json has invalid {key}")
    source = provenance.get("source")
    if not isinstance(source, dict):
        raise ReplayValidationError("provenance.json is missing source details")
    if (
        source.get("trajectory_index") != job.trajectory_index
        or source.get("schema") != "ns_incom"
        or list(source.get("channel_names", [])) != ["velocity_x", "velocity_y"]
        or list(source.get("channel_indices", [])) != [0, 1]
        or _data_file_index(str(source.get("source_path", ""))) != job.file_index
        or list(source.get("spatial_stride", [int(job.arguments["spatial_stride"])] * 2))
        != [int(job.arguments["spatial_stride"])] * 2
    ):
        raise ReplayValidationError("provenance source does not match the replay identity")

    expected_times = config["selected_raw_time_indices"]
    if len(metrics) != len(expected_times) or not metrics or not all(isinstance(item, dict) for item in metrics):
        raise ReplayValidationError("metrics.json does not contain the full selected trajectory")
    for position, (item, raw_time_index) in enumerate(zip(metrics, expected_times)):
        if item.get("cycle") != position or item.get("raw_time_index") != raw_time_index:
            raise ReplayValidationError("metrics.json cycle or raw time order is inconsistent")
        if (
            item.get("forecast_model") not in {None, config["forecast_model"]}
            or item.get("coverage_level") not in {None, config["coverage_level"]}
        ):
            raise ReplayValidationError("metrics.json protocol fields disagree with config.json")
    if summary.get("cycle_count") != len(metrics):
        raise ReplayValidationError("summary cycle_count does not match metrics.json")
    means = summary.get("means")
    if not isinstance(means, dict):
        raise ReplayValidationError("summary means must be an object")

    values: dict[str, float] = {}
    for metric in MEAN_METRICS:
        observed = [
            _finite_number(item.get(metric), f"metrics.json[{index}].{metric}", nonnegative=True)
            for index, item in enumerate(metrics)
        ]
        if metric == "coverage" and any(value > 1.0 for value in observed):
            raise ReplayValidationError("coverage values must lie in [0, 1]")
        recomputed = _mean(observed)
        reported = _finite_number(means.get(metric), f"summary.json.means.{metric}", nonnegative=True)
        if not math.isclose(recomputed, reported, rel_tol=1e-9, abs_tol=1e-12):
            raise ReplayValidationError(f"summary mean for {metric} does not match metrics.json")
        values[metric] = recomputed
    observed_peaks = [
        int(_finite_number(item.get("peak_gpu_memory_bytes"), "peak_gpu_memory_bytes", nonnegative=True))
        for item in metrics
    ]
    reported_peak = int(
        _finite_number(
            summary.get("peak_gpu_memory_bytes"),
            "summary.json.peak_gpu_memory_bytes",
            nonnegative=True,
        )
    )
    if reported_peak != max(observed_peaks):
        raise ReplayValidationError("summary peak GPU memory is not the trajectory maximum")
    values["peak_gpu_memory_bytes"] = float(reported_peak)
    return CompletedReplay(job=job, values=values)


def _incomplete_reason(job: ReplayJob) -> str | None:
    if not (job.run_directory / "config.json").is_file() or not (job.run_directory / "checkpoint.pt").is_file():
        return None
    provenance_path = job.run_directory / "provenance.json"
    if not provenance_path.is_file():
        return "checkpoint_present_without_final_provenance"
    try:
        provenance = _read_json(provenance_path, dict)
    except ReplayValidationError:
        return None
    if provenance.get("completed") is not True and provenance.get("failure") is None:
        return "checkpoint_present_for_unfinished_run"
    return None


def _sample_standard_deviation(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = _mean(values)
    return math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _bootstrap_seed(base_seed: int, baseline: str, metric: str) -> int:
    digest = hashlib.sha256(f"{baseline}\x00{metric}".encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:8], "big")) % (2**63 - 1)


def _effect_size(first: torch.Tensor, second: torch.Tensor) -> tuple[float | None, str]:
    if first.numel() < 2:
        return None, "insufficient_pairs"
    value = float(paired_effect_size(first, second))
    if math.isinf(value):
        return None, "positive_infinity" if value > 0.0 else "negative_infinity"
    return value, "finite"


def aggregate_replays(
    manifest_path: Path,
    *,
    results_root: Path | None = None,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 20260806,
) -> dict[str, Any]:
    if bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be positive")
    manifest, jobs = load_replay_manifest(manifest_path, results_root)
    completed: list[CompletedReplay] = []
    job_status: list[dict[str, Any]] = []
    for job in jobs:
        base = {
            "job_id": job.job_id,
            "file_index": job.file_index,
            "trajectory_index": job.trajectory_index,
            "trajectory_id": job.trajectory_id,
            "method": job.method,
            "seed": job.seed,
            "run_directory": str(job.run_directory),
        }
        if job.method not in METHODS:
            job_status.append({**base, "status": "excluded", "reason": "method_not_training_free"})
        elif not job.run_directory.exists():
            job_status.append({**base, "status": "missing", "reason": "run_directory_missing"})
        else:
            try:
                replay = validate_completed_replay(job)
            except ReplayValidationError as error:
                incomplete = _incomplete_reason(job)
                status = "incomplete" if incomplete is not None else "failed"
                job_status.append({**base, "status": status, "reason": incomplete or str(error)})
            else:
                completed.append(replay)
                job_status.append({**base, "status": "completed", "reason": None})

    completed_values = {
        (run.job.file_index, run.job.trajectory_index, run.job.method): run.values
        for run in completed
    }
    trajectory_metrics = [
        {
            "job_id": run.job.job_id,
            "file_index": run.job.file_index,
            "trajectory_index": run.job.trajectory_index,
            "trajectory_id": run.job.trajectory_id,
            "method": run.job.method,
            "seed": run.job.seed,
            "metric": metric,
            "value": value,
            "unit": METRIC_UNITS[metric],
        }
        for run in sorted(completed, key=lambda value: (value.job.file_index, value.job.trajectory_index, value.job.method))
        for metric, value in sorted(run.values.items())
    ]

    method_summaries = []
    for method in METHODS:
        method_runs = [run for run in completed if run.job.method == method]
        for metric in METRICS:
            samples = [run.values[metric] for run in method_runs]
            if not samples:
                continue
            method_summaries.append(
                {
                    "method": method,
                    "metric": metric,
                    "unit": METRIC_UNITS[metric],
                    "expected_trajectories": len(FILE_INDICES) * len(TRAJECTORY_INDICES),
                    "completed_trajectories": len(method_runs),
                    "mean": _mean(samples),
                    "sample_standard_deviation": _sample_standard_deviation(samples),
                }
            )

    trajectory_keys = [(file_index, trajectory_index) for file_index in FILE_INDICES for trajectory_index in TRAJECTORY_INDICES]
    paired_comparisons = []
    for baseline in BASELINES:
        for metric in METRICS:
            paired = [
                key for key in trajectory_keys
                if metric in completed_values.get((*key, "hilda"), {})
                and metric in completed_values.get((*key, baseline), {})
            ]
            if not paired:
                continue
            hilda_values = torch.tensor(
                [completed_values[(*key, "hilda")][metric] for key in paired], dtype=torch.float64
            )
            baseline_values = torch.tensor(
                [completed_values[(*key, baseline)][metric] for key in paired], dtype=torch.float64
            )
            ci = paired_bootstrap_ci(
                hilda_values,
                baseline_values,
                confidence=0.95,
                resamples=bootstrap_resamples,
                seed=_bootstrap_seed(bootstrap_seed, baseline, metric),
            )
            effect, effect_status = _effect_size(hilda_values, baseline_values)
            paired_set = set(paired)
            paired_comparisons.append(
                {
                    "first_method": "hilda",
                    "second_method": baseline,
                    "contrast": f"hilda - {baseline}",
                    "metric": metric,
                    "unit": METRIC_UNITS[metric],
                    "expected_pair_count": len(trajectory_keys),
                    "paired_count": len(paired),
                    "paired_trajectories": [f"f{file_index}_tr{trajectory_index}" for file_index, trajectory_index in paired],
                    "unavailable_trajectories": [
                        f"f{file_index}_tr{trajectory_index}"
                        for file_index, trajectory_index in trajectory_keys
                        if (file_index, trajectory_index) not in paired_set
                    ],
                    "mean_difference": float(ci.estimate),
                    "bootstrap_95_ci_lower": float(ci.lower),
                    "bootstrap_95_ci_upper": float(ci.upper),
                    "cohen_dz": effect,
                    "cohen_dz_status": effect_status,
                }
            )

    statuses = ("completed", "incomplete", "missing", "failed", "excluded")
    counts = {status: sum(row["status"] == status for row in job_status) for status in statuses}
    return {
        "schema_version": 1,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "manifest_protocol_version": manifest["protocol_version"],
        "manifest_profile": manifest["profile"],
        "expected_replay_jobs": EXPECTED_JOB_COUNT,
        "recognized_replay_jobs": EXPECTED_JOB_COUNT,
        "job_counts": {"manifest": len(jobs), **counts},
        "bootstrap": {
            "method": "paired percentile bootstrap",
            "confidence": 0.95,
            "resamples": bootstrap_resamples,
            "base_seed": bootstrap_seed,
        },
        "contrast_definition": "hilda - baseline",
        "pairing_key": ["file_index", "trajectory_index"],
        "metric_applicability": {
            "included": list(METRICS),
            "excluded": sorted(FORBIDDEN_ALPHA_FIELDS),
            "note": "External PDEBench replay has no alpha truth; Liu-coordinate diagnostics are not aggregate performance metrics.",
        },
        "job_status": job_status,
        "trajectory_metrics": trajectory_metrics,
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
        "pdebench_replay_aggregate.json": json.dumps(aggregate, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        "pdebench_job_status.csv": _csv_text(
            aggregate["job_status"],
            ["job_id", "file_index", "trajectory_index", "trajectory_id", "method", "seed", "run_directory", "status", "reason"],
        ),
        "pdebench_trajectory_metrics.csv": _csv_text(
            aggregate["trajectory_metrics"],
            ["job_id", "file_index", "trajectory_index", "trajectory_id", "method", "seed", "metric", "value", "unit"],
        ),
        "pdebench_method_summary.csv": _csv_text(
            aggregate["method_summaries"],
            ["method", "metric", "unit", "expected_trajectories", "completed_trajectories", "mean", "sample_standard_deviation"],
        ),
        "pdebench_paired_comparisons.csv": _csv_text(
            aggregate["paired_comparisons"],
            [
                "first_method", "second_method", "contrast", "metric", "unit",
                "expected_pair_count", "paired_count", "paired_trajectories",
                "unavailable_trajectories", "mean_difference", "bootstrap_95_ci_lower",
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
    aggregate = aggregate_replays(
        args.manifest,
        results_root=args.results_root,
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
