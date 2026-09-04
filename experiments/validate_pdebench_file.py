from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from hilda_da.pdebench import (
    PDEBenchHDF5Adapter,
    TrajectorySlice,
    verify_manifest_checksum,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and inspect one official PDEBench file")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-index", type=int, default=0)
    parser.add_argument("--time-stop", type=int, default=2)
    parser.add_argument("--spatial-stride", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = PDEBenchHDF5Adapter(args.data, manifest_path=args.manifest)
    if adapter.manifest_record is None:
        raise RuntimeError("Manifest record was not loaded")
    actual_md5 = verify_manifest_checksum(args.data, adapter.manifest_record)
    trajectory = adapter.load_trajectory(
        TrajectorySlice(
            trajectory_index=args.trajectory_index,
            time_stop=args.time_stop,
            spatial_stride=args.spatial_stride,
        ),
        dtype=torch.float32,
        device="cpu",
    )
    report = {
        "schema_version": 1,
        "validated": True,
        "actual_md5": actual_md5,
        "schema": asdict(adapter.schema),
        "loaded_slice": {
            "state_shape": list(trajectory.states.shape),
            "spatial_shape": list(trajectory.spatial_shape),
            "channel_names": list(trajectory.channel_names),
            "finite": bool(torch.isfinite(trajectory.states).all()),
            "minimum": float(trajectory.states.min()),
            "maximum": float(trajectory.states.max()),
            "time_count": None if trajectory.times is None else trajectory.times.numel(),
            "coordinate_lengths": [value.numel() for value in trajectory.coordinates],
        },
        "provenance": trajectory.provenance.as_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report["loaded_slice"]))


if __name__ == "__main__":
    main()
