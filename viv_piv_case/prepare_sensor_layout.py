"""Extract a compact read-only sensor layout from the VIV-PIV archives."""
from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np

from .common import load_config, software_environment, write_json
from .io import VIVCase, list_cases, sha256_file
from .rom import PODModel


def _parse_cases(value: str | None) -> list[str] | None:
    if value is None:
        return None
    result = [item.strip().replace(",", "").zfill(4)[-4:] for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("At least one case id is required")
    if len(set(result)) != len(result):
        raise ValueError(f"Duplicate case ids: {value}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a compact VIV-PIV sensor layout.")
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--layout", required=True)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--train-cases", default=None, help="Comma-separated model-fitting cases for split labels.")
    parser.add_argument("--case-ids", default=None, help="Comma-separated cases to extract; defaults to all configured cases.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.layout not in config["sensor_layouts"]:
        raise ValueError(f"Unknown sensor layout {args.layout!r}")
    layout = config["sensor_layouts"][args.layout]
    variant = args.variant or f"rank{int(config['rank'])}_stride1"
    model_root = pathlib.Path(config["output_root"]) / "models" / variant
    output_root = model_root / "sensor_layouts" / args.layout
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists() and not args.force:
        print(manifest_path)
        return
    output_root.mkdir(parents=True, exist_ok=True)
    pod = PODModel.load(model_root / "pod_model.npz")
    paths = list_cases(pathlib.Path(config["data_root"]))
    train_ids = _parse_cases(args.train_cases) or list(config["train_cases"])
    case_ids = _parse_cases(args.case_ids) or [*config["train_cases"], *config["test_cases"]]
    case_ids = list(dict.fromkeys(case_ids))
    missing = sorted(set(case_ids) - set(paths))
    if missing:
        raise ValueError(f"Requested cases are not available under {config['data_root']}: {missing}")
    diameter_mm = float(config["cylinder_diameter_m"]) * 1000.0
    if "x_over_d_values" in layout:
        target_x_over_d = np.asarray(layout["x_over_d_values"], dtype=float)
        if target_x_over_d.size != int(layout["x_points"]):
            raise ValueError(f"Layout {args.layout} x_over_d_values length does not match x_points")
    else:
        target_x_over_d = np.linspace(*map(float, layout["x_over_d_range"]), int(layout["x_points"]))
    if "y_over_d_values" in layout:
        target_y_over_d = np.asarray(layout["y_over_d_values"], dtype=float)
        if target_y_over_d.size != int(layout["y_points"]):
            raise ValueError(f"Layout {args.layout} y_over_d_values length does not match y_points")
    else:
        target_y_over_d = np.linspace(*map(float, layout["y_over_d_range"]), int(layout["y_points"]))
    target_x = target_x_over_d * diameter_mm
    target_y = target_y_over_d * diameter_mm
    started = time.perf_counter()
    records = []
    for case_id in case_ids:
        case = VIVCase.open(paths[case_id])
        x_indices = np.asarray([int(np.argmin(np.abs(case.x_mm - value))) for value in target_x], dtype=np.int64)
        y_indices = np.asarray([int(np.argmin(np.abs(case.y_mm - value))) for value in target_y], dtype=np.int64)
        if np.unique(x_indices).size != int(layout["x_points"]) or np.unique(y_indices).size != int(layout["y_points"]):
            raise ValueError(f"Layout {args.layout} aliases grid positions in case {case_id}")
        pairs = [(int(iy), int(ix)) for ix in x_indices for iy in y_indices]
        pair_y = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
        pair_x = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
        normalized = np.asarray(case.velocities[:, pair_y, pair_x, :], dtype=np.float32)
        valid = np.asarray(case.mask[:, pair_y, pair_x] > 0.5, dtype=bool)
        valid_fraction = float(valid.mean())
        if valid_fraction < 1.0:
            raise ValueError(f"Layout {args.layout} intersects the mask in case {case_id}: {valid_fraction:.6f}")
        low = case.norm_values[0].astype(np.float32)
        high = case.norm_values[1].astype(np.float32)
        observations = normalized * (high - low)[None, None, :] + low[None, None, :]
        observations = observations.reshape(case.time_s.size, -1)
        flat_indices = np.asarray(
            [(iy * case.x_mm.size + ix) * 2 + component for iy, ix in pairs for component in (0, 1)],
            dtype=np.int64,
        )
        sensor_mean, sensor_basis = pod.observation_matrix(flat_indices)
        coordinates = np.asarray([(case.x_mm[ix], case.y_mm[iy]) for iy, ix in pairs], dtype=np.float32)
        output_path = output_root / f"case_{case_id}.npz"
        np.savez_compressed(
            output_path,
            sensor_observations=observations.astype(np.float32),
            sensor_flat_indices=flat_indices,
            sensor_mean=sensor_mean.astype(np.float32),
            sensor_basis=sensor_basis.astype(np.float32),
            sensor_coordinates_mm=coordinates,
            x_indices=x_indices,
            y_indices=y_indices,
            valid_fraction=np.asarray(valid_fraction),
            split=np.asarray("train" if case_id in train_ids else "test"),
        )
        records.append({
            "case_id": case_id,
            "split": "train" if case_id in train_ids else "test",
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "spatial_points": len(pairs),
            "scalar_observations": int(observations.shape[1]),
            "valid_fraction": valid_fraction,
        })
    manifest = {
        "layout": args.layout,
        "variant": variant,
        "x_points": int(layout["x_points"]),
        "y_points": int(layout["y_points"]),
        "spatial_points": int(layout["x_points"]) * int(layout["y_points"]),
        "scalar_observations": 2 * int(layout["x_points"]) * int(layout["y_points"]),
        "x_over_d_range": layout["x_over_d_range"],
        "y_over_d_range": layout["y_over_d_range"],
        "x_over_d_values": target_x_over_d.tolist(),
        "y_over_d_values": target_y_over_d.tolist(),
        "train_cases": train_ids,
        "case_ids": case_ids,
        "records": records,
        "environment": software_environment(),
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(manifest_path, manifest)
    print(manifest_path)


if __name__ == "__main__":
    main()
