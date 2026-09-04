from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.aggregate_quick_advantage import analyze_advantage
from experiments.build_quick_advantage_manifest import build_jobs


class QuickAdvantageManifestTests(unittest.TestCase):
    def test_builds_complete_paired_training_free_matrix(self) -> None:
        root = Path(__file__).resolve().parents[1]
        matrix = json.loads(
            (root / "experiments" / "quick_advantage_matrix.json").read_text(encoding="utf-8")
        )
        jobs = build_jobs(matrix)
        expected = 2 * len(matrix["methods"]) * matrix["seed_protocol"]["replicates"]
        self.assertEqual(len(jobs), expected)
        self.assertEqual(len({job["id"] for job in jobs}), expected)
        self.assertEqual(
            {job["arguments"][job["arguments"].index("--alpha-true") + 1] for job in jobs},
            {"0.5", "0.72"},
        )
        self.assertFalse(any("train" in value.lower() for job in jobs for value in job["arguments"]))
        for job in jobs:
            self.assertEqual(
                job["arguments"][job["arguments"].index("--output-root") + 1],
                "<HILDA_RESULTS_ROOT>/results/quick_advantage_v1",
            )


class QuickAdvantageAggregationTests(unittest.TestCase):
    def test_computes_paired_degradation_and_difference_in_differences(self) -> None:
        conditions = [
            {
                "system": "wave",
                "condition_id": "match",
                "parameters": {"alpha_true": "0.5", "fixed_model_alpha": "0.5"},
            },
            {
                "system": "wave",
                "condition_id": "miss",
                "parameters": {"alpha_true": "0.72", "fixed_model_alpha": "0.5"},
            },
        ]
        rows = []
        values = {
            ("hilda", "match"): [1.0, 1.2, 0.8],
            ("hilda", "miss"): [1.1, 1.3, 0.9],
            ("denkf", "match"): [0.9, 1.0, 1.1],
            ("denkf", "miss"): [1.8, 2.0, 2.2],
        }
        for (method, condition_id), samples in values.items():
            for seed, value in enumerate(samples, start=1):
                rows.append({
                    "system": "wave",
                    "condition_id": condition_id,
                    "method": method,
                    "seed": seed,
                    "metric": "state_rmse",
                    "value": value,
                    "unit": "state units",
                })
        aggregate = {
            "conditions": conditions,
            "run_metrics": rows,
            "job_counts": {"manifest": 12, "completed": 12},
        }
        result = analyze_advantage(aggregate, bootstrap_resamples=2_000, bootstrap_seed=7)
        hilda_delta = next(row for row in result["condition_deltas"] if row["method"] == "hilda")
        interaction = result["interactions"][0]
        self.assertAlmostEqual(hilda_delta["mean_delta"], 0.1)
        self.assertAlmostEqual(interaction["mean_interaction"], -0.9)
        self.assertTrue(interaction["direction_favors_hilda"])
        self.assertTrue(interaction["ci_excludes_zero_in_hilda_direction"])
        mismatch = next(
            row for row in result["direct_comparisons"]
            if row["condition"] == "misspecified" and row["metric"] == "state_rmse"
        )
        self.assertAlmostEqual(mismatch["mean_difference"], -0.9)
        self.assertEqual(result["verdict"]["status"], "consistent_absolute_hilda_advantage")
        self.assertTrue(result["verdict"]["execution_complete"])


if __name__ == "__main__":
    unittest.main()
