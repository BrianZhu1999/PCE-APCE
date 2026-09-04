from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.build_formal_manifest import build_primary_jobs
from experiments.download_pdebench_subset import select_records, target_for
from experiments.run_cpu_compatibility_smoke import METHODS, SYSTEMS, build_smoke_jobs
from experiments.run_workstation_batch import (
    BatchJob,
    classify_existing_run,
    resolve_runner,
    run_directory,
)


class ManifestTests(unittest.TestCase):
    def test_frozen_primary_matrix_expands_to_expected_job_count(self) -> None:
        root = Path(__file__).resolve().parents[1]
        matrix = json.loads(
            (root / "experiments" / "formal_experiment_matrix.json").read_text(encoding="utf-8")
        )
        jobs = build_primary_jobs(matrix)
        expected = sum(item["replicates"] for item in matrix["systems"].values()) * len(matrix["methods"])
        self.assertEqual(len(jobs), expected)
        self.assertEqual(len({job["id"] for job in jobs}), expected)
        self.assertFalse(any("train" in value.lower() for job in jobs for value in job["arguments"]))
        for job in jobs:
            self.assertIn("--fixed-model-alpha", job["arguments"])
            self.assertIn("--coverage-level", job["arguments"])
            self.assertIn("--energy-score-chunk-size", job["arguments"])
            self.assertEqual(
                job["arguments"][job["arguments"].index("--output-root") + 1],
                "<HILDA_RESULTS_ROOT>/results/formal_v2",
            )

    def test_cpu_compatibility_matrix_covers_every_method_system_pair(self) -> None:
        jobs = build_smoke_jobs(Path("smoke"))
        self.assertEqual(len(jobs), len(METHODS) * len(SYSTEMS))
        self.assertEqual(len({job.job_id for job in jobs}), len(jobs))
        enff_jobs = [
            job
            for job in jobs
            if job.arguments[job.arguments.index("--system") + 1]
            == "navier_stokes_enff"
        ]
        self.assertEqual(len(enff_jobs), len(METHODS))
        for job in enff_jobs:
            self.assertEqual(
                job.arguments[job.arguments.index("--enff-grid-size") + 1],
                "8",
            )


class ResumeClassificationTests(unittest.TestCase):
    def test_classifies_new_completed_resume_and_invalid_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self.assertEqual(classify_existing_run(root / "missing"), "new")
            completed = root / "completed"
            completed.mkdir()
            (completed / "provenance.json").write_text('{"completed": true}', encoding="utf-8")
            self.assertEqual(classify_existing_run(completed), "completed")
            resumable = root / "resumable"
            resumable.mkdir()
            (resumable / "config.json").write_text("{}", encoding="utf-8")
            (resumable / "checkpoint.pt").write_bytes(b"checkpoint")
            self.assertEqual(classify_existing_run(resumable), "resume")
            invalid = root / "invalid"
            invalid.mkdir()
            self.assertEqual(classify_existing_run(invalid), "invalid")

    def test_run_directory_uses_job_id_and_output_root(self) -> None:
        job = BatchJob("stable_id", ("--output-root", "formal_results"))
        self.assertEqual(run_directory(job, Path("/project")), Path("/project/formal_results/stable_id"))

    def test_runner_can_select_the_pdebench_entry_point(self) -> None:
        project = Path("/project")
        self.assertEqual(
            resolve_runner(project, Path("experiments/run_pdebench_assimilation.py")),
            (project / "experiments" / "run_pdebench_assimilation.py").resolve(),
        )


class PDEBenchDownloadPlanTests(unittest.TestCase):
    def test_selects_exact_manifest_subset_and_rejects_unsafe_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest = root / "urls.csv"
            manifest.write_text(
                "PDE,Filename,URL,Path,MD5\n"
                "NS_Incom,a.h5,https://example.test/a,2D/NS_incom/,aaa\n"
                "NS_Incom,b.h5,https://example.test/b,2D/NS_incom/,bbb\n",
                encoding="utf-8",
            )
            records = select_records(manifest, ["b.h5", "a.h5"])
            self.assertEqual([record.filename for record in records], ["b.h5", "a.h5"])
            self.assertEqual(target_for(root, records[0]), root / "2D" / "NS_incom" / "b.h5")
            unsafe = type(records[0])(
                records[0].pde,
                records[0].filename,
                records[0].url,
                "../outside",
                records[0].expected_md5,
            )
            with self.assertRaises(ValueError):
                target_for(root, unsafe)


if __name__ == "__main__":
    unittest.main()
