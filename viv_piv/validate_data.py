from __future__ import annotations

import argparse
import csv
import pathlib
import zipfile

import numpy as np

from .common import load_config, write_json
from .io import VIVCase, list_cases, stored_memmap


def _member_headers(path: pathlib.Path) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            key = pathlib.Path(info.filename).stem
            array = stored_memmap(path, key)
            output[key] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "compressed_bytes": int(info.compress_size),
                "raw_bytes": int(info.file_size),
                "zip_compression": int(info.compress_type),
            }
    return output


def describe_case(path: pathlib.Path, config: dict[str, object]) -> dict[str, object]:
    case = VIVCase.open(path)
    velocities = case.velocities
    mask = case.mask
    sample_indices = np.arange(0, case.time_s.size, 10, dtype=int)
    mask_values: set[float] = set()
    fluid_fraction: list[float] = []
    for index in sample_indices:
        frame = np.asarray(mask[index])
        mask_values.update(float(value) for value in np.unique(frame))
        fluid_fraction.append(float(np.mean(frame > 0.5)))
    dt = np.diff(case.time_s)
    result: dict[str, object] = {
        "case_id": case.case_id,
        "label": case.label,
        "reduced_velocity": case.reduced_velocity,
        "split": "test" if case.case_id in config["test_cases"] else "train",
        "file_bytes": int(path.stat().st_size),
        "members": _member_headers(path),
        "velocity_shape": list(velocities.shape),
        "velocity_dtype": str(velocities.dtype),
        "mask_shape": list(mask.shape),
        "mask_dtype": str(mask.dtype),
        "mask_values_sampled": sorted(mask_values),
        "fluid_fraction_min": float(np.min(fluid_fraction)),
        "fluid_fraction_mean": float(np.mean(fluid_fraction)),
        "fluid_fraction_max": float(np.max(fluid_fraction)),
        "x_min_mm": float(case.x_mm.min()),
        "x_max_mm": float(case.x_mm.max()),
        "y_min_mm": float(case.y_mm.min()),
        "y_max_mm": float(case.y_mm.max()),
        "x_step_mm": float(np.median(np.diff(case.x_mm))),
        "y_step_mm": float(np.median(np.diff(case.y_mm))),
        "time_start_s": float(case.time_s[0]),
        "time_end_s": float(case.time_s[-1]),
        "duration_s": float(case.time_s[-1] - case.time_s[0]),
        "dt_min_s": float(dt.min()),
        "dt_median_s": float(np.median(dt)),
        "dt_max_s": float(dt.max()),
        "displacement_min_m": float(case.cyl_displ_m.min()),
        "displacement_max_m": float(case.cyl_displ_m.max()),
        "displacement_std_m": float(case.cyl_displ_m.std()),
        "norm_values": case.norm_values.tolist(),
    }
    return result


def write_csv(path: pathlib.Path, records: list[dict[str, object]]) -> None:
    scalar_keys = sorted(
        key for key in records[0]
        if all(not isinstance(record.get(key), (dict, list)) for record in records)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        writer.writerows({key: record.get(key, "") for key in scalar_keys} for record in records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the public VIV-PIV NPZ collection.")
    parser.add_argument("--config", type=pathlib.Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    data_root = pathlib.Path(config["data_root"])
    output_root = pathlib.Path(config["output_root"]) / "validation"
    paths = list_cases(data_root)
    expected = set(config["train_cases"]) | set(config["test_cases"])
    if set(paths) != expected:
        raise RuntimeError(f"Case registry mismatch: found={sorted(paths)} expected={sorted(expected)}")
    records = [describe_case(paths[case_id], config) for case_id in sorted(paths)]
    total_bytes = sum(int(record["file_bytes"]) for record in records)
    payload = {
        "doi": config["doi"],
        "license": config["license"],
        "coordinate_unit_decision": "source arrays are millimetres; divide by 1000 for metres",
        "coordinate_unit_evidence": "the 515.6 mm streamwise span is 10.31 cylinder diameters for D=0.05 m",
        "case_count": len(records),
        "total_npz_bytes": total_bytes,
        "train_cases": list(config["train_cases"]),
        "test_cases": list(config["test_cases"]),
        "records": records,
    }
    write_json(output_root / "viv_piv_dataset.json", payload)
    write_csv(output_root / "viv_piv_dataset.csv", records)
    print(f"validated={len(records)} bytes={total_bytes} output={output_root}")


if __name__ == "__main__":
    main()
