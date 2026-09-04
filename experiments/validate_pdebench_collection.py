from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hilda_da.pdebench import PDEBenchHDF5Adapter, TrajectorySlice


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a verified PDEBench NS_Incom collection")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--download-ledger", type=Path, required=True)
    parser.add_argument("--initial-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ledger = json.loads(args.download_ledger.read_text(encoding="utf-8"))
    initial = json.loads(args.initial_validation.read_text(encoding="utf-8"))
    if ledger.get("completed") is not True or initial.get("validated") is not True:
        raise ValueError("All source checksum validations must be complete")
    verified = {
        "ns_incom_inhom_2d_512-0.h5": {
            "actual_md5": initial["actual_md5"],
            "size_bytes": initial["provenance"]["file_size_bytes"],
        },
        **ledger["files"],
    }
    records = []
    for file_index in range(5):
        filename = f"ns_incom_inhom_2d_512-{file_index}.h5"
        path = args.root / filename
        source = verified[filename]
        if not path.is_file() or path.stat().st_size != int(source["size_bytes"]):
            raise ValueError(f"Validated file is missing or has changed size: {filename}")
        adapter = PDEBenchHDF5Adapter(path, manifest_path=args.manifest)
        if (
            adapter.schema.kind != "ns_incom"
            or adapter.schema.trajectory_count != 4
            or adapter.schema.spatial_shape != (512, 512)
            or adapter.schema.channel_names != ("velocity_x", "velocity_y")
        ):
            raise ValueError(f"Unexpected NS_Incom schema for {filename}: {adapter.schema}")
        for trajectory_index in range(4):
            trajectory = adapter.load_trajectory(
                TrajectorySlice(
                    trajectory_index=trajectory_index,
                    time_stop=1,
                    spatial_stride=128,
                ),
                dtype=torch.float32,
            )
            records.append(
                {
                    "file_index": file_index,
                    "trajectory_index": trajectory_index,
                    "state_shape": list(trajectory.states.shape),
                    "finite": bool(torch.isfinite(trajectory.states).all()),
                    "time": float(trajectory.times[0]),
                }
            )
    report = {
        "schema_version": 1,
        "validated": all(item["finite"] for item in records),
        "file_count": 5,
        "trajectory_count": len(records),
        "grid": [512, 512],
        "channels": ["velocity_x", "velocity_y"],
        "files": {
            name: {
                "actual_md5": value["actual_md5"],
                "size_bytes": value["size_bytes"],
            }
            for name, value in verified.items()
        },
        "trajectory_audit": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    if not report["validated"]:
        raise ValueError("PDEBench collection contains non-finite values")
    print(json.dumps({"validated": True, "files": 5, "trajectories": len(records)}))


if __name__ == "__main__":
    main()
