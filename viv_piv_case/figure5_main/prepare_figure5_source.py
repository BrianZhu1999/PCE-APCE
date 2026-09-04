"""Build a compact, auditable source bundle for the VIV-PIV main Figure 5."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil

import numpy as np

from viv_piv_case.common import load_config, write_json
from viv_piv_case.io import VIVCase, list_cases
from viv_piv_case.rom import PODModel


CASE_ID = "0679"
SEED = 0
LAYOUT = "adaptive_fullfield_valid"
METHODS = ("pce", "apce", "aug_enkf", "bma")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trace_path(trace_root: pathlib.Path, method: str) -> pathlib.Path:
    return trace_root / (
        f"viv_{CASE_ID}_{method}_seed{SEED:03d}_layout{LAYOUT}_ens064_covfull_shr050.npz"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--variant", default="rank256_stride1")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    result_root = pathlib.Path(config["output_root"])
    model_root = result_root / "models" / args.variant
    trace_root = result_root / "runs" / args.variant / "traces"
    summary_root = result_root / "summaries" / args.variant
    blackout_root = result_root / "figures" / "blackout_gifs"
    source_root = blackout_root / "sources"
    args.output.mkdir(parents=True, exist_ok=True)

    selected_source = source_root / f"viv_{CASE_ID}_apce_seed{SEED:03d}_blackout_source.npz"
    with np.load(selected_source, allow_pickle=False) as archive:
        origin = int(archive["origin_index"])
        latent_apce = np.asarray(archive["latent_estimate"], dtype=np.float32)
        horizon_s = np.asarray(archive["horizon_s"], dtype=np.float64)

    cases = list_cases(pathlib.Path(config["data_root"]))
    case = VIVCase.open(cases[CASE_ID])
    pod = PODModel.load(model_root / "pod_model.npz")
    final_index = origin + horizon_s.size - 1
    truth_origin, valid_origin = case.physical_frames(origin, origin + 1)
    truth_final, valid_final = case.physical_frames(final_index, final_index + 1)
    truth_origin = truth_origin[0]
    truth_final = truth_final[0]
    valid_origin = valid_origin[0]
    valid_final = valid_final[0]
    apce_flat = pod.mean.astype(np.float32) + pod.basis.astype(np.float32) @ latent_apce[-1]
    apce_final = apce_flat.reshape(truth_final.shape)

    truth_origin_speed = np.linalg.norm(truth_origin, axis=-1)
    truth_final_speed = np.linalg.norm(truth_final, axis=-1)
    apce_final_speed = np.linalg.norm(apce_final, axis=-1)
    truth_rms = float(np.sqrt(np.mean(np.sum(truth_final[valid_final] ** 2, axis=-1))))
    vector_error = np.linalg.norm(apce_final - truth_final, axis=-1) / max(truth_rms, 1e-12)
    truth_origin_speed = np.where(valid_origin, truth_origin_speed, np.nan).astype(np.float32)
    truth_final_speed = np.where(valid_final, truth_final_speed, np.nan).astype(np.float32)
    apce_final_speed = np.where(valid_final, apce_final_speed, np.nan).astype(np.float32)
    vector_error = np.where(valid_final, vector_error, np.nan).astype(np.float32)

    sensor_path = model_root / "sensor_layouts" / LAYOUT / f"case_{CASE_ID}.npz"
    with np.load(sensor_path, allow_pickle=False) as sensor:
        sensor_coordinates_mm = np.asarray(sensor["sensor_coordinates_mm"], dtype=np.float32)
        sensor_flat_indices = np.asarray(sensor["sensor_flat_indices"], dtype=np.int64)

    trace_payload: dict[str, np.ndarray] = {}
    source_files: list[pathlib.Path] = [selected_source, sensor_path, cases[CASE_ID]]
    for method in METHODS:
        path = trace_path(trace_root, method)
        source_files.append(path)
        with np.load(path, allow_pickle=False) as trace:
            if method == "pce":
                trace_payload["truth_energy"] = np.asarray(trace["truth_energy"], dtype=np.float64)
                trace_payload["time_s"] = np.asarray(trace["time_s"], dtype=np.float64)
            trace_payload[f"{method}_energy"] = np.asarray(trace["predicted_energy"], dtype=np.float64)
            if method in {"pce", "apce"}:
                grid = np.asarray(trace["candidate_grid"], dtype=np.float64)
                weights = np.asarray(trace["weights"], dtype=np.float64)
                trace_payload[f"{method}_candidate_grid"] = grid
                trace_payload[f"{method}_candidate_mean"] = weights @ grid
                trace_payload[f"{method}_normalized_entropy"] = (
                    -np.sum(np.maximum(weights, 1e-300) * np.log(np.maximum(weights, 1e-300)), axis=1)
                    / np.log(grid.size)
                )

    diameter_mm = float(config["cylinder_diameter_m"]) * 1000.0
    compact_path = args.output / "figure5_viv_piv_compact_source.npz"
    np.savez_compressed(
        compact_path,
        case_id=np.asarray(CASE_ID),
        reduced_velocity=np.asarray(int(CASE_ID) / 100.0),
        seed=np.asarray(SEED),
        origin_index=np.asarray(origin),
        origin_time_s=np.asarray(case.time_s[origin]),
        final_index=np.asarray(final_index),
        final_time_s=np.asarray(case.time_s[final_index]),
        blackout_horizon_s=np.asarray(horizon_s[-1]),
        x_over_d=np.asarray(case.x_mm, dtype=np.float32) / diameter_mm,
        y_over_d=np.asarray(case.y_mm, dtype=np.float32) / diameter_mm,
        sensor_x_over_d=sensor_coordinates_mm[:, 0] / diameter_mm,
        sensor_y_over_d=sensor_coordinates_mm[:, 1] / diameter_mm,
        sensor_flat_indices=sensor_flat_indices,
        cylinder_origin_y_over_d=np.asarray(case.cyl_displ_m[origin] / float(config["cylinder_diameter_m"])),
        cylinder_final_y_over_d=np.asarray(case.cyl_displ_m[final_index] / float(config["cylinder_diameter_m"])),
        cylinder_time_s=np.asarray(case.time_s, dtype=np.float64),
        cylinder_displacement_over_d=np.asarray(case.cyl_displ_m, dtype=np.float64) / float(config["cylinder_diameter_m"]),
        truth_origin_speed=truth_origin_speed,
        truth_final_speed=truth_final_speed,
        apce_final_speed=apce_final_speed,
        apce_vector_error_normalized=vector_error,
        **trace_payload,
    )

    copied: list[pathlib.Path] = []
    for name in ("summary_metrics.csv", "blackout_metrics.csv", "identifiability.csv", "run_manifest.csv", "leakage_audit.json"):
        source = summary_root / name
        destination = args.output / name
        shutil.copy2(source, destination)
        copied.append(destination)
    config_copy = args.output / "config_adaptive_fullfield_formal5.json"
    shutil.copy2(args.config, config_copy)
    copied.append(config_copy)

    manifest = {
        "figure": "figure5_viv_piv_real_experiment",
        "remote_result_bundle": str(result_root),
        "compact_source": str(compact_path),
        "compact_source_sha256": sha256_file(compact_path),
        "representative_selection": {
            "case_id": CASE_ID,
            "reduced_velocity": int(CASE_ID) / 100.0,
            "seed": SEED,
            "blackout_origin_rule": "maximum truth kinetic energy among the 20 predefined origins",
            "origin_index": origin,
            "origin_time_s": float(case.time_s[origin]),
            "horizon_s": float(horizon_s[-1]),
        },
        "source_files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in source_files
        ],
        "copied_summary_files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in copied
        ],
    }
    write_json(args.output / "figure5_viv_piv_source_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
