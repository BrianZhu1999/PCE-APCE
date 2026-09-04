from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments.build_pdebench_replay_manifest import build_jobs
from experiments.run_pdebench_assimilation import (
    EXPECTED_CHANNELS,
    METHODS,
    PDEBenchFrameStream,
    analysis_trend_increment,
    channel_scales,
    seeded_sensor_mask,
    smooth_relative_noise,
    summarize,
    validate_collection_record,
    validate_ns_incom_schema,
)
from hilda_da.pdebench import PDEBenchHDF5Adapter


class PDEBenchAssimilationProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.path = self.root / "ns_incom_inhom_2d_512-0.h5"
        velocity = np.arange(2 * 6 * 4 * 5 * 2, dtype=np.float32).reshape(
            2, 6, 4, 5, 2
        )
        with h5py.File(self.path, "w") as handle:
            handle.create_dataset("velocity", data=velocity)
            handle.create_dataset(
                "t", data=np.broadcast_to(np.arange(6, dtype=np.float32), (2, 6))
            )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_schema_contract_is_velocity_only_and_rejects_nonofficial_grid(self) -> None:
        adapter = PDEBenchHDF5Adapter(self.path)
        self.assertEqual(adapter.schema.kind, "ns_incom")
        self.assertEqual(adapter.schema.channel_names, EXPECTED_CHANNELS)
        self.assertNotIn("pressure", adapter.schema.channel_names)
        with self.assertRaisesRegex(ValueError, "official"):
            validate_ns_incom_schema(adapter)
        validate_ns_incom_schema(adapter, expected_grid=(4, 5))

    def test_dispatch_contract_contains_only_frozen_training_free_methods(self) -> None:
        self.assertEqual(
            METHODS,
            ("hilda", "denkf", "letkf", "ensf", "iensf", "ensf_lr_ridge", "enff_f2p"),
        )

    def test_frame_stream_reads_one_selected_frame_at_a_time(self) -> None:
        adapter = PDEBenchHDF5Adapter(self.path)
        stream = PDEBenchFrameStream(
            adapter,
            trajectory_index=1,
            time_start=1,
            time_stop=6,
            time_step=2,
            spatial_stride=1,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        self.assertEqual(stream.raw_indices, (1, 3, 5))
        for position in range(len(stream)):
            frame = stream.frame(position)
            self.assertEqual(frame.states.shape, (1, 4 * 5 * 2))
            self.assertEqual(frame.channel_names, EXPECTED_CHANNELS)
            self.assertEqual(float(frame.times[0]), float(stream.raw_indices[position]))

    def test_seeded_sensor_mask_observes_both_velocity_components(self) -> None:
        adapter = PDEBenchHDF5Adapter(self.path)
        frame = PDEBenchFrameStream(
            adapter,
            trajectory_index=0,
            time_start=0,
            time_stop=1,
            time_step=1,
            spatial_stride=1,
            dtype=torch.float32,
            device=torch.device("cpu"),
        ).frame(0)
        first = seeded_sensor_mask(frame.spatial_shape, 4, 17)
        second = seeded_sensor_mask(frame.spatial_shape, 4, 17)
        self.assertTrue(torch.equal(first, second))
        observation = frame.sparse_observation(first)
        indices = observation.indices.reshape(-1, 2)
        self.assertEqual(observation.indices.numel(), 8)
        self.assertTrue(torch.equal(indices[:, 1], indices[:, 0] + 1))

    def test_smoothed_noise_preserves_shape_scale_and_zero_spatial_mean(self) -> None:
        adapter = PDEBenchHDF5Adapter(self.path)
        frame = PDEBenchFrameStream(
            adapter,
            trajectory_index=0,
            time_start=0,
            time_stop=1,
            time_step=1,
            spatial_stride=1,
            dtype=torch.float32,
            device=torch.device("cpu"),
        ).frame(0)
        scales = channel_scales(frame.states[0], frame.spatial_shape)
        generator = torch.Generator().manual_seed(9)
        noise = smooth_relative_noise(
            5,
            frame.spatial_shape,
            scales,
            0.1,
            1,
            dtype=torch.float32,
            device=torch.device("cpu"),
            generator=generator,
        ).reshape(5, 4 * 5, 2)
        self.assertEqual(noise.shape, (5, 20, 2))
        self.assertTrue(torch.allclose(noise.mean(dim=1), torch.zeros(5, 2), atol=1e-5))
        actual_rms = torch.sqrt(noise.square().mean(dim=1))
        self.assertTrue(
            torch.allclose(actual_rms, 0.1 * scales.unsqueeze(0), rtol=1e-4, atol=1e-5)
        )

    def test_metric_protocol_has_no_alpha_error_output_fields(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "run_pdebench_assimilation.py"
        ).read_text(encoding="utf-8")
        forbidden = ("alpha_rmse", "alpha_absolute_error", '"alpha_true"')
        self.assertFalse(any(value in source for value in forbidden))
        self.assertIn('"liu_coordinate_estimate"', source)
        self.assertIn('"pressure_present": False', source)

    def test_forecast_trend_uses_only_two_past_analysis_estimates(self) -> None:
        previous = torch.tensor([1.0, 3.0])
        latest = torch.tensor([2.5, 2.0])
        self.assertTrue(
            torch.equal(
                analysis_trend_increment("linear_extrapolation", previous, latest),
                torch.tensor([1.5, -1.0]),
            )
        )
        self.assertIsNone(analysis_trend_increment("persistence", previous, latest))
        self.assertIsNone(analysis_trend_increment("linear_extrapolation", None, latest))

    def test_collection_record_locks_validated_file_size(self) -> None:
        adapter = PDEBenchHDF5Adapter(self.path)
        report = self.root / "collection_validation.json"
        report.write_text(
            json.dumps(
                {
                    "validated": True,
                    "files": {
                        self.path.name: {
                            "actual_md5": "validated-md5",
                            "size_bytes": self.path.stat().st_size,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        record = validate_collection_record(report, self.path, adapter)
        self.assertEqual(record["size_bytes"], self.path.stat().st_size)
        payload = json.loads(report.read_text(encoding="utf-8"))
        payload["files"][self.path.name]["size_bytes"] += 1
        report.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(ValueError):
            validate_collection_record(report, self.path, adapter)

    def test_frozen_replay_matrix_expands_to_twenty_trajectories_by_seven_methods(self) -> None:
        root = Path(__file__).resolve().parents[1]
        matrix = json.loads(
            (root / "experiments" / "pdebench_replay_matrix.json").read_text(
                encoding="utf-8"
            )
        )
        jobs = build_jobs(matrix)
        self.assertEqual(len(jobs), 20 * len(METHODS))
        self.assertEqual(len({job["id"] for job in jobs}), len(jobs))
        for job in jobs:
            arguments = job["arguments"]
            self.assertEqual(
                arguments[arguments.index("--forecast-model") + 1],
                "linear_extrapolation",
            )
            self.assertIn("--collection-validation", arguments)
            self.assertFalse(any("train" in value.lower() for value in arguments))

    def test_summary_uses_maximum_not_mean_for_cumulative_peak_memory(self) -> None:
        records = []
        for memory in (100, 250):
            records.append(
                {
                    "state_rmse": 1.0,
                    "observation_rmse": 1.0,
                    "crps": 1.0,
                    "energy_score": 1.0,
                    "coverage": 0.9,
                    "interval_width": 1.0,
                    "cycle_seconds": 2.0,
                    "peak_gpu_memory_bytes": memory,
                }
            )
        summary = summarize(records)
        self.assertEqual(summary["peak_gpu_memory_bytes"], 250)
        self.assertNotIn("peak_gpu_memory_bytes", summary["means"])


if __name__ == "__main__":
    unittest.main()
