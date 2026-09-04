from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from experiments.aggregate_formal_results import aggregate_results, write_outputs


def _arguments(method: str, seed: int, output_root: Path) -> list[str]:
    return [
        "--system", "spring",
        "--method", method,
        "--seed", str(seed),
        "--ensemble-size", "20",
        "--steps", "10",
        "--observation-interval", "5",
        "--observation-count", "1",
        "--observation-noise", "0.05",
        "--observation-transform", "linear",
        "--alpha-mode", "fixed",
        "--alpha-true", "0.72",
        "--fixed-model-alpha", "0.5",
        "--coverage-level", "0.9",
        "--energy-score-chunk-size", "64",
        "--checkpoint-interval", "1",
        "--output-root", str(output_root),
    ]


def _config(arguments: list[str]) -> dict[str, object]:
    result: dict[str, object] = {}
    integer_fields = {
        "seed", "ensemble_size", "steps", "observation_interval", "observation_count",
        "energy_score_chunk_size",
    }
    float_fields = {
        "observation_noise", "alpha_true", "fixed_model_alpha", "coverage_level",
    }
    for index in range(0, len(arguments), 2):
        key = arguments[index][2:].replace("-", "_")
        if key == "checkpoint_interval":
            continue
        value: object = arguments[index + 1]
        if key in integer_fields:
            value = int(str(value))
        elif key in float_fields:
            value = float(str(value))
        result[key] = value
    result["dt"] = 0.01
    result["state_dim"] = 2
    result["hilda"] = {}
    return result


def _write_run(
    root: Path,
    job_id: str,
    arguments: list[str],
    *,
    state_rmse: float,
    alpha_error: float | None,
    completed: bool = True,
    baseline_alpha_estimate: float | None = None,
) -> None:
    run = root / job_id
    run.mkdir(parents=True)
    configuration = _config(arguments)
    method = configuration["method"]
    metrics = []
    for step in (5, 10):
        metrics.append(
            {
                "step": step,
                "state_rmse": state_rmse,
                "observation_rmse": state_rmse / 2.0,
                "crps": state_rmse / 3.0,
                "energy_score": state_rmse * 2.0,
                "coverage": 0.8,
                "interval_width": 1.2,
                "alpha_estimate": (
                    baseline_alpha_estimate
                    if method != "hilda"
                    else (0.72 - alpha_error if alpha_error is not None else None)
                ),
                "alpha_absolute_error": (
                    baseline_alpha_estimate if method != "hilda" else alpha_error
                ),
            }
        )
    summary = {
        "mean_state_rmse": state_rmse,
        "mean_observation_rmse": state_rmse / 2.0,
        "mean_crps": state_rmse / 3.0,
        "mean_energy_score": state_rmse * 2.0,
        "mean_coverage": 0.8,
        "mean_interval_width": 1.2,
        "mean_alpha_absolute_error": (
            baseline_alpha_estimate if method != "hilda" else alpha_error
        ),
        "analysis_times": 2,
    }
    provenance = {
        "run_id": job_id,
        "configuration": configuration,
        "elapsed_seconds": 10.0 + state_rmse,
        "peak_gpu_memory_bytes": 1000,
        "completed": completed,
        "failed": not completed,
        "failure_message": None if completed else "synthetic failure",
    }
    for name, value in (
        ("config.json", configuration),
        ("provenance.json", provenance),
        ("summary.json", summary),
        ("metrics.json", metrics),
    ):
        (run / name).write_text(json.dumps(value), encoding="utf-8")


class FormalResultAggregationTests(unittest.TestCase):
    def test_pairs_only_completed_matching_seeds_and_writes_atomic_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            results_root = temporary / "results"
            jobs = []
            arguments_by_id = {}
            for method in ("hilda", "denkf"):
                for seed in (1, 2, 3):
                    job_id = f"spring_{method}_{seed}"
                    arguments = _arguments(method, seed, results_root)
                    arguments_by_id[job_id] = arguments
                    jobs.append({"id": job_id, "arguments": arguments})
            manifest = temporary / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "protocol_version": "test-v1",
                        "profile": "primary",
                        "job_count": len(jobs),
                        "jobs": jobs,
                    }
                ),
                encoding="utf-8",
            )
            _write_run(
                results_root, "spring_hilda_1", arguments_by_id["spring_hilda_1"],
                state_rmse=1.0, alpha_error=0.1,
            )
            _write_run(
                results_root, "spring_hilda_2", arguments_by_id["spring_hilda_2"],
                state_rmse=2.0, alpha_error=0.2,
            )
            _write_run(
                results_root, "spring_denkf_1", arguments_by_id["spring_denkf_1"],
                state_rmse=2.0, alpha_error=None,
            )
            _write_run(
                results_root, "spring_denkf_2", arguments_by_id["spring_denkf_2"],
                state_rmse=4.0, alpha_error=None,
            )
            _write_run(
                results_root, "spring_denkf_3", arguments_by_id["spring_denkf_3"],
                state_rmse=3.0, alpha_error=None, completed=False,
            )

            aggregate = aggregate_results(
                manifest,
                results_root=results_root,
                reference_method="hilda",
                bootstrap_resamples=2_000,
                bootstrap_seed=7,
            )
            self.assertEqual(
                aggregate["job_counts"],
                {
                    "manifest": 6,
                    "completed": 4,
                    "incomplete": 0,
                    "missing": 1,
                    "failed": 1,
                    "excluded": 0,
                },
            )
            state_comparison = next(
                row for row in aggregate["paired_comparisons"] if row["metric"] == "state_rmse"
            )
            self.assertEqual(state_comparison["paired_seeds"], [1, 2])
            self.assertEqual(state_comparison["unavailable_seeds"], [3])
            self.assertAlmostEqual(state_comparison["mean_difference"], -1.5)
            self.assertAlmostEqual(state_comparison["cohen_dz"], -1.5 / math.sqrt(0.5))
            self.assertFalse(
                any(
                    row["method"] == "denkf" and row["metric"] == "alpha_absolute_error"
                    for row in aggregate["method_summaries"]
                )
            )
            self.assertFalse(
                any(row["metric"] == "alpha_absolute_error" for row in aggregate["paired_comparisons"])
            )

            output_directory = temporary / "aggregate"
            output_paths = write_outputs(output_directory, aggregate)
            self.assertEqual(len(output_paths), 5)
            self.assertTrue(all(path.is_file() for path in output_paths))
            self.assertFalse(any(output_directory.glob("*.tmp")))
            persisted = json.loads((output_directory / "formal_aggregate.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["job_counts"], aggregate["job_counts"])

    def test_rejects_non_hilda_alpha_as_a_failed_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            results_root = temporary / "results"
            arguments = _arguments("denkf", 1, results_root)
            manifest = temporary / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "job_count": 1,
                        "jobs": [{"id": "bad_baseline", "arguments": arguments}],
                    }
                ),
                encoding="utf-8",
            )
            _write_run(
                results_root,
                "bad_baseline",
                arguments,
                state_rmse=1.0,
                alpha_error=None,
                baseline_alpha_estimate=0.4,
            )
            aggregate = aggregate_results(
                manifest,
                results_root=results_root,
                bootstrap_resamples=10,
            )
            self.assertEqual(aggregate["job_counts"]["failed"], 1)
            self.assertIn("must not contain an alpha estimate", aggregate["job_status"][0]["reason"])
            self.assertEqual(aggregate["run_metrics"], [])

    def test_excludes_methods_outside_frozen_training_free_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            results_root = temporary / "results"
            arguments = _arguments("trained_neural_operator", 1, results_root)
            manifest = temporary / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "job_count": 1,
                        "jobs": [{"id": "trained", "arguments": arguments}],
                    }
                ),
                encoding="utf-8",
            )
            aggregate = aggregate_results(manifest, results_root=results_root)
            self.assertEqual(aggregate["job_counts"]["excluded"], 1)
            self.assertEqual(aggregate["method_summaries"], [])
            self.assertEqual(aggregate["paired_comparisons"], [])

    def test_classifies_checkpointed_unfinished_run_as_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            results_root = temporary / "results"
            arguments = _arguments("hilda", 1, results_root)
            manifest = temporary / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "job_count": 1,
                        "jobs": [{"id": "unfinished", "arguments": arguments}],
                    }
                ),
                encoding="utf-8",
            )
            run_directory = results_root / "unfinished"
            run_directory.mkdir(parents=True)
            (run_directory / "config.json").write_text("{}", encoding="utf-8")
            (run_directory / "checkpoint.pt").write_bytes(b"checkpoint")
            aggregate = aggregate_results(manifest, results_root=results_root)
            self.assertEqual(aggregate["job_counts"]["incomplete"], 1)
            self.assertEqual(aggregate["job_counts"]["failed"], 0)
            self.assertEqual(aggregate["job_status"][0]["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
