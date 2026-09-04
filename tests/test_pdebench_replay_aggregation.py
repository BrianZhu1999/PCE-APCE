from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from experiments.aggregate_pdebench_replay import (
    EXPECTED_JOB_COUNT,
    aggregate_replays,
    load_replay_manifest,
    write_outputs,
)
from experiments.build_pdebench_replay_manifest import build_jobs


def _argument_map(arguments: list[str]) -> dict[str, str]:
    return {
        arguments[index][2:].replace("-", "_"): arguments[index + 1]
        for index in range(0, len(arguments), 2)
    }


def _make_manifest(root: Path) -> tuple[Path, Path, list[dict[str, object]]]:
    project_root = Path(__file__).resolve().parents[1]
    matrix = json.loads(
        (project_root / "experiments" / "pdebench_replay_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    results_root = root / "results"
    matrix["result_root"] = str(results_root)
    matrix["data"]["root"] = str(root / "data")
    matrix["data"]["manifest"] = str(root / "official_urls.csv")
    matrix["data"]["collection_validation"] = str(root / "collection.json")
    jobs = build_jobs(matrix)
    manifest = root / "pdebench_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_version": matrix["protocol_version"],
                "profile": "pdebench_replay",
                "job_count": len(jobs),
                "jobs": jobs,
            }
        ),
        encoding="utf-8",
    )
    return manifest, results_root, jobs


def _resolved_config(arguments: list[str]) -> dict[str, object]:
    raw = _argument_map(arguments)
    integer_fields = {
        "trajectory_index",
        "time_start",
        "time_stop",
        "time_step",
        "spatial_stride",
        "seed",
        "ensemble_size",
        "sensor_count",
        "noise_smoothing_radius",
        "energy_score_chunk_size",
    }
    float_fields = {
        "observation_noise",
        "initial_spread",
        "process_noise",
        "hilda_path_log_scale",
        "coverage_level",
    }
    config: dict[str, object] = {}
    for key, value in raw.items():
        if key == "checkpoint_interval":
            continue
        if key in integer_fields:
            config[key] = int(value)
        elif key in float_fields:
            config[key] = float(value)
        else:
            config[key] = value
    stride = int(raw["spatial_stride"])
    config.update(
        {
            "source_grid": [512, 512],
            "assimilation_grid": [512 // stride, 512 // stride],
            "channel_order": ["velocity_x", "velocity_y"],
            "state_dim": 2 * (512 // stride) ** 2,
            "selected_raw_time_indices": list(
                range(
                    int(raw["time_start"]),
                    int(raw["time_stop"]),
                    int(raw["time_step"]),
                )
            ),
            "forecast_protocol": {
                "name": raw["forecast_model"],
                "future_target_frames_used_by_forecast": False,
                "normalization": "none",
            },
        }
    )
    return config


def _write_run(
    results_root: Path,
    job: dict[str, object],
    *,
    state_rmse: float,
    completed: bool = True,
) -> Path:
    job_id = str(job["id"])
    arguments = list(job["arguments"])
    raw = _argument_map(arguments)
    config = _resolved_config(arguments)
    run_directory = results_root / job_id
    run_directory.mkdir(parents=True)
    raw_indices = list(config["selected_raw_time_indices"])
    metrics = []
    for cycle, raw_time_index in enumerate(raw_indices):
        metrics.append(
            {
                "cycle": cycle,
                "raw_time_index": raw_time_index,
                "state_rmse": state_rmse,
                "observation_rmse": state_rmse / 2.0,
                "crps": state_rmse / 3.0,
                "energy_score": state_rmse * 2.0,
                "coverage": 0.8,
                "interval_width": 1.2,
                "cycle_seconds": 0.5 + state_rmse,
                "peak_gpu_memory_bytes": 1000 + cycle,
                "liu_coordinate_estimate": 0.5 if raw["method"] == "hilda" else None,
            }
        )
    means = {
        name: sum(float(item[name]) for item in metrics) / len(metrics)
        for name in (
            "state_rmse",
            "observation_rmse",
            "crps",
            "energy_score",
            "coverage",
            "interval_width",
            "cycle_seconds",
        )
    }
    summary = {
        "cycle_count": len(metrics),
        "means": means,
        "peak_gpu_memory_bytes": max(
            item["peak_gpu_memory_bytes"] for item in metrics
        ),
    }
    provenance = {
        "schema_version": 1,
        "completed": completed,
        "failure": None if completed else "RuntimeError: synthetic failure",
        "run_id": job_id,
        "method": raw["method"],
        "dataset": "PDEBench NS_Incom",
        "source_grid": [512, 512],
        "assimilation_grid": config["assimilation_grid"],
        "channels": ["velocity_x", "velocity_y"],
        "pressure_present": False,
        "normalization": "none",
        "forecast_uses_future_target_frames": False,
        "forecast_model": raw["forecast_model"],
        "source": {
            "source_path": raw["data"],
            "schema": "ns_incom",
            "trajectory_index": int(raw["trajectory_index"]),
            "channel_indices": [0, 1],
            "channel_names": ["velocity_x", "velocity_y"],
        },
    }
    for name, value in (
        ("config.json", config),
        ("provenance.json", provenance),
        ("summary.json", summary),
        ("metrics.json", metrics),
    ):
        (run_directory / name).write_text(json.dumps(value), encoding="utf-8")
    (run_directory / "checkpoint.pt").write_bytes(b"synthetic checkpoint")
    return run_directory


def _job(jobs: list[dict[str, object]], file_index: int, trajectory_index: int, method: str) -> dict[str, object]:
    prefix = f"pdebench_f{file_index}_tr{trajectory_index}_{method}_"
    return next(job for job in jobs if str(job["id"]).startswith(prefix))


class PDEBenchReplayAggregationTests(unittest.TestCase):
    def test_frozen_manifest_strictly_identifies_140_jobs(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        manifest = project_root / "experiments" / "pdebench_replay_manifest_v3.json"
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing_root = Path(temporary_directory) / "missing"
            raw, jobs = load_replay_manifest(manifest, missing_root)
            self.assertEqual(raw["job_count"], EXPECTED_JOB_COUNT)
            self.assertEqual(len(jobs), EXPECTED_JOB_COUNT)
            self.assertEqual(
                len({(job.file_index, job.trajectory_index, job.method) for job in jobs}),
                EXPECTED_JOB_COUNT,
            )
            data_arguments = [job.arguments["data"] for job in jobs]
            self.assertTrue(
                all(
                    value.startswith("<HILDA_RESULTS_ROOT>/data/")
                    for value in data_arguments
                )
            )
            self.assertFalse(any("\\" in value for value in data_arguments))
            aggregate = aggregate_replays(manifest, results_root=missing_root)
            self.assertEqual(aggregate["job_counts"]["missing"], EXPECTED_JOB_COUNT)
            self.assertEqual(aggregate["recognized_replay_jobs"], EXPECTED_JOB_COUNT)

    def test_pairs_hilda_with_baseline_by_exact_trajectory_and_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest, results_root, jobs = _make_manifest(root)
            _write_run(results_root, _job(jobs, 0, 0, "hilda"), state_rmse=1.0)
            _write_run(results_root, _job(jobs, 0, 1, "hilda"), state_rmse=2.0)
            _write_run(results_root, _job(jobs, 0, 0, "denkf"), state_rmse=2.0)
            _write_run(results_root, _job(jobs, 0, 1, "denkf"), state_rmse=4.0)
            incomplete_job = _job(jobs, 0, 0, "letkf")
            incomplete_directory = results_root / str(incomplete_job["id"])
            incomplete_directory.mkdir(parents=True)
            (incomplete_directory / "config.json").write_text("{}", encoding="utf-8")
            (incomplete_directory / "checkpoint.pt").write_bytes(b"checkpoint")
            _write_run(
                results_root,
                _job(jobs, 0, 0, "ensf"),
                state_rmse=3.0,
                completed=False,
            )

            aggregate = aggregate_replays(
                manifest,
                results_root=results_root,
                bootstrap_seed=11,
            )
            self.assertEqual(aggregate["bootstrap"]["resamples"], 10_000)
            self.assertEqual(
                aggregate["job_counts"],
                {
                    "manifest": 140,
                    "completed": 4,
                    "incomplete": 1,
                    "missing": 134,
                    "failed": 1,
                    "excluded": 0,
                },
            )
            state = next(
                row
                for row in aggregate["paired_comparisons"]
                if row["second_method"] == "denkf" and row["metric"] == "state_rmse"
            )
            self.assertEqual(state["paired_trajectories"], ["f0_tr0", "f0_tr1"])
            self.assertAlmostEqual(state["mean_difference"], -1.5)
            self.assertAlmostEqual(state["cohen_dz"], -1.5 / math.sqrt(0.5))
            aggregate_sections = (
                aggregate["trajectory_metrics"],
                aggregate["method_summaries"],
                aggregate["paired_comparisons"],
            )
            self.assertFalse(
                any("alpha" in str(row.get("metric", "")) for section in aggregate_sections for row in section)
            )

            outputs = write_outputs(root / "aggregate", aggregate)
            self.assertEqual(len(outputs), 5)
            self.assertTrue(all(path.is_file() for path in outputs))
            self.assertFalse(any((root / "aggregate").glob("*.tmp")))
            persisted = json.loads(outputs[0].read_text(encoding="utf-8"))
            self.assertEqual(persisted["job_counts"], aggregate["job_counts"])

    def test_alpha_error_field_invalidates_external_replay_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest, results_root, jobs = _make_manifest(root)
            job = _job(jobs, 0, 0, "hilda")
            run_directory = _write_run(results_root, job, state_rmse=1.0)
            metrics_path = run_directory / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics[0]["alpha_absolute_error"] = 0.1
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            aggregate = aggregate_replays(manifest, results_root=results_root)
            self.assertEqual(aggregate["job_counts"]["failed"], 1)
            self.assertIn("must not contain alpha", next(row for row in aggregate["job_status"] if row["status"] == "failed")["reason"])
            self.assertEqual(aggregate["trajectory_metrics"], [])

    def test_extra_trained_method_is_audited_as_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest, results_root, jobs = _make_manifest(root)
            source = _job(jobs, 0, 0, "hilda")
            arguments = list(source["arguments"])
            method_index = arguments.index("--method") + 1
            arguments[method_index] = "trained_operator"
            extra = {
                "id": "pdebench_f0_tr0_trained_operator_s2026080600",
                "arguments": arguments,
            }
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["jobs"].append(extra)
            payload["job_count"] += 1
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            aggregate = aggregate_replays(manifest, results_root=results_root)
            self.assertEqual(aggregate["job_counts"]["excluded"], 1)
            self.assertEqual(aggregate["job_counts"]["missing"], EXPECTED_JOB_COUNT)


if __name__ == "__main__":
    unittest.main()
