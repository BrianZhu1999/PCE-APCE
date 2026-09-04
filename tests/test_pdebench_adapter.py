from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from hilda_da.pdebench import (
    PDEBenchHDF5Adapter,
    TrajectorySlice,
    file_md5,
    load_manifest_record,
    verify_manifest_checksum,
)


class PDEBenchAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_tensor_schema_slice_and_manifest_provenance(self) -> None:
        path = self.root / "1D_Burgers_Sols_Nu0.01.hdf5"
        data = np.arange(2 * 6 * 8, dtype=np.float32).reshape(2, 6, 8)
        with h5py.File(path, "w") as handle:
            handle.create_dataset("tensor", data=data)
            handle.create_dataset("x-coordinate", data=np.linspace(0.0, 1.0, 8))
            handle.create_dataset("t-coordinate", data=np.linspace(0.0, 0.5, 6))
            handle.attrs["generator"] = "PDEBench"
        manifest = self.root / "pdebench_data_urls.csv"
        manifest.write_text(
            "PDE,Filename,URL,Path,MD5\n"
            "Burgers,1D_Burgers_Sols_Nu0.01.hdf5,https://example.test/data,"
            "1D/Burgers/Train/,0123456789abcdef0123456789abcdef\n",
            encoding="utf-8",
        )

        adapter = PDEBenchHDF5Adapter(path, manifest_path=manifest)
        trajectory = adapter.load_trajectory(
            TrajectorySlice(
                trajectory_index=1,
                time_start=1,
                time_stop=6,
                time_step=2,
                spatial_stride=2,
            ),
            dtype=torch.float64,
        )

        expected = torch.from_numpy(data[1, 1:6:2, ::2]).to(torch.float64)
        self.assertEqual(adapter.schema.kind, "tensor")
        self.assertEqual(trajectory.states.shape, (3, 4))
        self.assertTrue(torch.equal(trajectory.states, expected))
        self.assertEqual(trajectory.spatial_shape, (4,))
        self.assertTrue(
            torch.allclose(
                trajectory.times,
                torch.tensor([0.1, 0.3, 0.5], dtype=torch.float64),
            )
        )
        self.assertEqual(trajectory.coordinates[0].numel(), 4)
        self.assertEqual(
            trajectory.provenance.manifest_record.expected_md5,
            "0123456789abcdef0123456789abcdef",
        )
        self.assertEqual(trajectory.provenance.root_attributes["generator"], "PDEBench")
        self.assertIn("source_path", json.dumps(trajectory.provenance.as_dict()))

    def test_cfd_fields_and_sparse_sensor_mask(self) -> None:
        path = self.root / "2D_CFD_Test.hdf5"
        base = np.arange(1 * 3 * 2 * 3, dtype=np.float32).reshape(1, 3, 2, 3)
        with h5py.File(path, "w") as handle:
            handle.create_dataset("density", data=base)
            handle.create_dataset("pressure", data=base + 100.0)
            handle.create_dataset("Vx", data=base + 200.0)
            handle.create_dataset("Vy", data=base + 300.0)
            handle.create_dataset("x-coordinate", data=np.arange(2))
            handle.create_dataset("y-coordinate", data=np.arange(3))

        adapter = PDEBenchHDF5Adapter(path, cfd_fields=("density", "Vx"))
        trajectory = adapter.load_trajectory()
        sensor_mask = torch.tensor([[True, False, False], [False, False, True]])
        operator = trajectory.sparse_observation(sensor_mask, channel_indices=(1,))

        self.assertEqual(adapter.schema.kind, "cfd_fields")
        self.assertEqual(trajectory.states.shape, (3, 12))
        self.assertEqual(trajectory.channel_names, ("density", "Vx"))
        self.assertTrue(torch.equal(operator.indices, torch.tensor([1, 11])))
        observed = operator(trajectory.states)
        self.assertTrue(torch.equal(observed[:, 0], torch.from_numpy(base[:, :, 0, 0]).squeeze(0) + 200.0))
        self.assertTrue(torch.equal(observed[:, 1], torch.from_numpy(base[:, :, 1, 2]).squeeze(0) + 200.0))

    def test_grouped_schema_selects_requested_trajectory_and_channels(self) -> None:
        path = self.root / "2D_diff-react_NA_NA.h5"
        first = np.zeros((4, 2, 3, 2), dtype=np.float32)
        second = np.arange(4 * 2 * 3 * 2, dtype=np.float32).reshape(4, 2, 3, 2)
        with h5py.File(path, "w") as handle:
            handle.create_dataset("0000/data", data=first)
            handle.create_dataset("0001/data", data=second)

        adapter = PDEBenchHDF5Adapter(path)
        trajectory = adapter.load_trajectory(
            TrajectorySlice(
                trajectory_index=1,
                time_start=1,
                spatial_stride=(1, 2),
                channel_indices=(1,),
            )
        )

        expected = torch.from_numpy(second[1:, :, ::2, 1]).reshape(3, -1)
        self.assertEqual(adapter.schema.kind, "grouped_data")
        self.assertEqual(adapter.schema.trajectory_count, 2)
        self.assertEqual(trajectory.provenance.dataset_paths, ("0001/data",))
        self.assertEqual(trajectory.provenance.source_shapes, (second.shape,))
        self.assertEqual(trajectory.provenance.trajectory_name, "0001")
        self.assertEqual(trajectory.spatial_shape, (2, 2))
        self.assertEqual(trajectory.channel_names, ("channel_1",))
        self.assertTrue(torch.equal(trajectory.states, expected))

    def test_ns_incom_schema_uses_batched_times_and_velocity_channels(self) -> None:
        path = self.root / "ns_incom_inhom_2d_512-0.h5"
        velocity = np.arange(2 * 5 * 4 * 6 * 2, dtype=np.float32).reshape(2, 5, 4, 6, 2)
        times = np.stack((np.linspace(0.0, 0.4, 5), np.linspace(1.0, 1.4, 5))).astype(np.float32)
        with h5py.File(path, "w") as handle:
            handle.create_dataset("velocity", data=velocity)
            handle.create_dataset("t", data=times)
            handle.create_dataset("particles", data=np.zeros((2, 5, 4, 6, 1), dtype=np.float32))
            handle.create_dataset("force", data=np.zeros((2, 4, 6, 2), dtype=np.float32))

        adapter = PDEBenchHDF5Adapter(path)
        trajectory = adapter.load_trajectory(
            TrajectorySlice(
                trajectory_index=1,
                time_start=1,
                time_stop=5,
                time_step=2,
                spatial_stride=(2, 3),
            )
        )

        expected = torch.from_numpy(velocity[1, 1:5:2, ::2, ::3, :]).reshape(2, -1)
        self.assertEqual(adapter.schema.kind, "ns_incom")
        self.assertEqual(adapter.schema.trajectory_count, 2)
        self.assertEqual(trajectory.spatial_shape, (2, 2))
        self.assertEqual(trajectory.channel_names, ("velocity_x", "velocity_y"))
        self.assertTrue(torch.equal(trajectory.states, expected))
        self.assertTrue(torch.allclose(trajectory.times, torch.tensor([1.1, 1.3])))

    def test_rejects_invalid_mask_and_ambiguous_manifest(self) -> None:
        path = self.root / "sample.hdf5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("tensor", data=np.zeros((1, 2, 3), dtype=np.float32))
        adapter = PDEBenchHDF5Adapter(path)
        trajectory = adapter.load_trajectory()
        with self.assertRaises(ValueError):
            trajectory.sparse_observation(torch.zeros(3, dtype=torch.bool))
        with self.assertRaises(ValueError):
            trajectory.sparse_observation(torch.ones(2, dtype=torch.bool))

        manifest = self.root / "empty.csv"
        manifest.write_text("PDE,Filename,URL,Path,MD5\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_manifest_record(manifest, path.name)

    def test_streaming_manifest_checksum_accepts_match_and_rejects_mismatch(self) -> None:
        path = self.root / "payload.h5"
        path.write_bytes(b"pdebench-checksum-payload")
        expected = file_md5(path, chunk_size=3)
        manifest = self.root / "pdebench_data_urls.csv"
        manifest.write_text(
            "PDE,Filename,URL,Path,MD5\n"
            f"NS_Incom,{path.name},https://example.test/data,2D/NS_incom/,{expected}\n",
            encoding="utf-8",
        )
        record = load_manifest_record(manifest, path.name)
        self.assertEqual(verify_manifest_checksum(path, record, chunk_size=4), expected)
        wrong_record = type(record)(
            record.pde,
            record.filename,
            record.url,
            record.relative_path,
            "0" * 32,
        )
        with self.assertRaises(ValueError):
            verify_manifest_checksum(path, wrong_record)


if __name__ == "__main__":
    unittest.main()
