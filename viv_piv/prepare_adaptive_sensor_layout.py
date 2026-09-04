"""Prepare a mask-aware full-field sensor layout with a fixed target count."""
from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np

from .common import load_config, write_json
from .io import VIVCase, list_cases
from .rom import PODModel


def _parse_cases(value: str | None) -> list[str] | None:
    if value is None:
        return None
    result = [item.strip().replace(",", "").zfill(4)[-4:] for item in value.split(",") if item.strip()]
    if not result or len(set(result)) != len(result):
        raise ValueError("case list must be non-empty and unique")
    return result


def _valid_all_cases(cases: list[VIVCase], block: int = 64) -> np.ndarray:
    shape = cases[0].mask.shape[1:]
    valid_all = np.ones(shape, dtype=bool)
    for case in cases:
        if case.mask.shape[1:] != shape:
            raise ValueError("all cases must share the same mask shape")
        for start in range(0, case.time_s.size, block):
            stop = min(start + block, case.time_s.size)
            valid_all &= np.all(np.asarray(case.mask[start:stop] > 0.5), axis=0)
    return valid_all


def _adaptive_pairs(
    reference: VIVCase,
    valid_all: np.ndarray,
    nx: int,
    ny: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    diameter_mm = 50.0
    x = np.asarray(reference.x_mm, dtype=float) / diameter_mm
    y = np.asarray(reference.y_mm, dtype=float) / diameter_mm
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    target_x = np.linspace(xmin, xmax, nx)
    target_y = np.linspace(ymin, ymax, ny)
    targets = np.asarray([(tx, ty) for tx in target_x for ty in target_y], dtype=float)
    pair_x_all = np.asarray([int(np.argmin(np.abs(x - tx))) for tx, _ in targets], dtype=np.int64)
    pair_y_all = np.asarray([int(np.argmin(np.abs(y - ty))) for _, ty in targets], dtype=np.int64)
    grid_pairs = np.column_stack([pair_y_all, pair_x_all])
    if np.unique(grid_pairs, axis=0).shape[0] != targets.shape[0]:
        raise ValueError("ideal lattice aliases PIV grid locations")
    keep = valid_all[pair_y_all, pair_x_all]
    pair_x = pair_x_all[keep]
    pair_y = pair_y_all[keep]
    coordinates = np.column_stack([reference.x_mm[pair_x], reference.y_mm[pair_y]]).astype(np.float32)
    ideal_coordinates = (targets[keep] * diameter_mm).astype(np.float32)
    return pair_x, pair_y, coordinates, ideal_coordinates, keep


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare mask-aware full-field VIV-PIV sensor layout.")
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--layout", required=True)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--target-points", type=int, default=800)
    parser.add_argument("--case-ids", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.layout not in config["sensor_layouts"]:
        raise ValueError(f"unknown layout {args.layout}")
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
    case_ids = _parse_cases(args.case_ids) or [*config["train_cases"], *config["test_cases"]]
    case_ids = list(dict.fromkeys(case_ids))
    cases = [VIVCase.open(paths[case_id]) for case_id in case_ids]
    reference = cases[0]
    started = time.perf_counter()
    valid_all = _valid_all_cases(cases)
    layout = config["sensor_layouts"][args.layout]
    nx = int(layout["x_points"])
    ny = int(layout["y_points"])
    if args.target_points != nx * ny:
        raise ValueError(f"--target-points must equal x_points*y_points ({nx * ny})")
    pair_x, pair_y, coordinates, ideal_coordinates, retained_targets = _adaptive_pairs(
        reference, valid_all, nx, ny
    )
    flat_indices = np.asarray(
        [(iy * reference.x_mm.size + ix) * 2 + component
         for iy, ix in zip(pair_y, pair_x) for component in (0, 1)], dtype=np.int64)
    train_ids = {str(x) for x in config["train_cases"]}
    records = []
    for case in cases:
        normalized = np.asarray(case.velocities[:, pair_y, pair_x, :], dtype=np.float32)
        valid = np.asarray(case.mask[:, pair_y, pair_x] > 0.5, dtype=bool)
        valid_fraction = float(valid.mean())
        if valid_fraction < 1.0:
            raise ValueError(f"adaptive layout still intersects the mask in {case.case_id}: {valid_fraction:.6f}")
        low = case.norm_values[0].astype(np.float32)
        high = case.norm_values[1].astype(np.float32)
        observations = normalized * (high - low)[None, None, :] + low[None, None, :]
        observations = observations.reshape(case.time_s.size, -1)
        sensor_mean, sensor_basis = pod.observation_matrix(flat_indices)
        output_path = output_root / f"case_{case.case_id}.npz"
        np.savez_compressed(
            output_path,
            sensor_observations=observations.astype(np.float32),
            sensor_flat_indices=flat_indices,
            sensor_mean=sensor_mean.astype(np.float32),
            sensor_basis=sensor_basis.astype(np.float32),
            sensor_coordinates_mm=coordinates,
            ideal_sensor_coordinates_mm=ideal_coordinates,
            x_indices=pair_x,
            y_indices=pair_y,
            valid_fraction=np.asarray(valid_fraction),
            split=np.asarray("train" if case.case_id in train_ids else "test"),
        )
        records.append({
            "case_id": case.case_id,
            "split": "train" if case.case_id in train_ids else "test",
            "path": str(output_path),
            "spatial_points": int(pair_x.size),
            "scalar_observations": int(observations.shape[1]),
            "valid_fraction": valid_fraction,
        })
    manifest = {
        "layout": args.layout,
        "variant": variant,
        "adaptive_mask_aware": True,
        "target_points": int(args.target_points),
        "dropped_masked_points": int(args.target_points - pair_x.size),
        "retained_target_indices": np.flatnonzero(retained_targets),
        "spatial_points": int(pair_x.size),
        "scalar_observations": int(2 * pair_x.size),
        "x_over_d_range": layout["x_over_d_range"],
        "y_over_d_range": layout["y_over_d_range"],
        "reference_grid_x_over_d": [float(x) for x in reference.x_mm[pair_x] / 50.0],
        "reference_grid_y_over_d": [float(y) for y in reference.y_mm[pair_y] / 50.0],
        "valid_across_all_cases": True,
        "case_ids": case_ids,
        "records": records,
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(manifest_path, manifest)
    print(manifest_path)


if __name__ == "__main__":
    main()
