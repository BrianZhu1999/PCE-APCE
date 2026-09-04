#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from meshir.assimilation import run_assimilation
from meshir.geometry import geometric_localization
from meshir.metrics import evaluate_field, nrmse


HERE = Path(__file__).resolve().parent
METHODS = ("DEnKF", "BMA", "PCE", "APCE")
CONDITIONS = ("standard", "blackout")


def run_one(
    method: str,
    condition: str,
    truth: np.ndarray,
    model: dict[str, np.ndarray],
    config: dict,
    seed: int,
    device: str,
    rng: np.random.Generator,
) -> tuple[dict, dict[str, np.ndarray]]:
    sample_rate = 16000.0
    analysis_end = int(round(float(config["dataset"]["s1_analysis_end_seconds"]) * sample_rate))
    prediction_end = int(round(float(config["dataset"]["s1_prediction_end_seconds"]) * sample_rate))
    truth = truth[:prediction_end]
    mean_full = np.asarray(model["mean_full"], dtype=np.float32)
    centered_truth = truth - mean_full[None, :]
    candidate_paths = np.asarray(model["candidate_paths"], dtype=np.float32)
    fixed_path = np.asarray(model["fixed_path"], dtype=np.float32)
    candidate_paths = candidate_paths[:, :prediction_end]
    fixed_path = fixed_path[:prediction_end]
    candidates = len(candidate_paths) if method != "DEnKF" else 1
    ensemble = int(config["assimilation"]["ensemble_size_pilot"])
    state_dim = candidate_paths.shape[-1]
    initial_noise = rng.standard_normal((candidates, ensemble, state_dim)).astype(np.float32)
    forecast_noise = rng.standard_normal((prediction_end, candidates, ensemble, state_dim)).astype(np.float32)
    result = run_assimilation(
        method, centered_truth, candidate_paths, fixed_path,
        np.asarray(model["basis_full"], dtype=np.float32),
        np.asarray(model["observed_indices"], dtype=int), float(model["rho"]),
        model["state_covariance"], model["process_covariance"], model["observation_covariance"],
        {**config["assimilation"], **config["preprocess"], "blackout_fraction_of_analysis": config["dataset"]["blackout_fraction_of_analysis"]},
        analysis_end, prediction_end, condition, device,
        initial_noise, forecast_noise, sample_rate,
        candidate_toa=(
            np.linalg.norm(np.asarray(model["candidate_source_positions"])[:, None, :] - np.asarray(model["positions"])[np.asarray(model["observed_indices"])[None, :], :], axis=2) / 343.0
            if "candidate_source_positions" in model else None
        ),
        observed_peak_times=(
            np.argmax(np.abs(truth[:int(0.02 * sample_rate), np.asarray(model["observed_indices"], dtype=int)]), axis=0) / sample_rate
            if "candidate_source_positions" in model else None
        ),
        toa_sigma_seconds=float(config["assimilation"].get("toa_sigma_seconds", 0.0005)),
    )
    mean = result.mean_field + mean_full[None, :]
    lower = result.lower_field + mean_full[None, :]
    upper = result.upper_field + mean_full[None, :]
    heldout = np.asarray(model["heldout_indices"], dtype=int)
    metrics = evaluate_field(truth, mean, lower, upper, heldout, analysis_end, prediction_end, sample_rate)
    candidate_errors = []
    for path in candidate_paths:
        candidate_field = np.einsum("td,fd->tf", path, model["basis_full"]) + mean_full[None, :]
        candidate_errors.append(nrmse(truth[analysis_end:prediction_end, heldout], candidate_field[analysis_end:prediction_end, heldout]))
    fixed_field = np.einsum("td,fd->tf", fixed_path, model["basis_full"]) + mean_full[None, :]
    fixed_error = nrmse(truth[analysis_end:prediction_end, heldout], fixed_field[analysis_end:prediction_end, heldout])
    metrics.update({
        "candidate_oracle_prediction_nrmse": float(np.min(candidate_errors)),
        "fixed_prediction_nrmse": float(fixed_error),
        "oracle_improvement_fraction": float((fixed_error - np.min(candidate_errors)) / max(fixed_error, 1e-12)),
        "median_separation_ratio": float(np.median(result.separation_history)) if len(result.separation_history) else 0.0,
        "weight_entropy_final": float(-np.sum(result.final_weights * np.log(np.maximum(result.final_weights, 1e-300)))),
        "weight_update_inside_blackout": bool(np.any(result.blackout_mask[np.rint(result.weight_time * sample_rate).astype(int)])) if len(result.weight_time) else False,
    })
    snapshots = np.asarray([mean[index] for index in [min(int(0.02 * sample_rate), prediction_end - 1), min(analysis_end, prediction_end - 1), min(int(0.12 * sample_rate), prediction_end - 1), min(int(0.20 * sample_rate) - 1, prediction_end - 1)]])
    payload = {
        "snapshots": snapshots,
        "truth_snapshots": np.asarray([truth[index] for index in [min(int(0.02 * sample_rate), prediction_end - 1), min(analysis_end, prediction_end - 1), min(int(0.12 * sample_rate), prediction_end - 1), min(int(0.20 * sample_rate) - 1, prediction_end - 1)]]),
        "final_weights": result.final_weights,
        "weight_time": result.weight_time,
        "weight_history": result.weight_history,
        "lower_probe": lower[:, heldout[: min(32, len(heldout))]],
        "upper_probe": upper[:, heldout[: min(32, len(heldout))]],
        "paired_noise_digest": np.array(result.paired_noise_digest),
    }
    return metrics, payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("s1", "s32"), required=True)
    parser.add_argument("--folds", nargs="+", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--device", choices=("cuda:2", "cuda:3"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    all_truth = np.load(args.cache / ("s1_rir_16k.npy" if args.case == "s1" else "s32_rir_16k.npy"), mmap_mode="r")
    with np.load(args.cache / "geometry.npz") as geometry:
        positions = geometry["s1_positions"] if args.case == "s1" else geometry["s32_positions"]
        sources = geometry["s32_sources"] if args.case == "s32" else None
    completed = []
    for fold in args.folds:
        model_path = args.models / f"{args.case}_fold{fold}.npz"
        with np.load(model_path, allow_pickle=False) as handle:
            model = {key: handle[key] for key in handle.files}
        test_sources = [0] if args.case == "s1" else np.asarray(model["test_source_indices"], dtype=int).tolist()
        for seed in args.seeds:
            for condition in CONDITIONS:
                for method in METHODS:
                    rng = np.random.default_rng(2026081800 + 1000 * fold + 100 * seed + (0 if args.case == "s1" else 50000))
                    source_metrics = []
                    representative_payload = None
                    for source in test_sources:
                        truth = np.asarray(all_truth if args.case == "s1" else all_truth[source], dtype=np.float32)
                        metrics, payload = run_one(method, condition, truth, model, config, seed, args.device, rng)
                        if representative_payload is None:
                            representative_payload = payload
                        if args.case == "s32":
                            estimate, baseline_toa_error = geometric_localization(
                                truth[:int(0.02 * 16000), np.asarray(model["observed_indices"], dtype=int)].T,
                                positions[np.asarray(model["observed_indices"], dtype=int)],
                                np.asarray(model["sources"]),
                                343.0, 16000.0,
                            )
                            true_position = sources[source]
                            if true_position.shape[0] == 3:
                                true_position = true_position[:2]
                            candidate_positions = np.asarray(model["candidate_source_positions"])
                            if method == "DEnKF":
                                source_estimate = np.mean(candidate_positions, axis=0)
                            else:
                                source_estimate = np.sum(np.asarray(result_weights := np.asarray(payload["final_weights"]))[:, None] * candidate_positions, axis=0)
                            metrics["localization_error_m"] = float(np.linalg.norm(source_estimate[:2] - true_position[:2]))
                            metrics["geometric_baseline_error_m"] = float(np.linalg.norm(estimate[:2] - true_position[:2]))
                            metrics["geometric_baseline_toa_error_s"] = float(baseline_toa_error)
                        source_metrics.append({"source_index": int(source), **metrics})
                    aggregated = {}
                    for key in source_metrics[0]:
                        if key == "source_index":
                            continue
                        values = [row[key] for row in source_metrics if isinstance(row[key], (float, int))]
                        if values:
                            aggregated[key] = float(np.mean(values))
                    output_dir = args.output_root / args.case / f"fold{fold}" / condition / method.lower()
                    output_dir.mkdir(parents=True, exist_ok=True)
                    stem = output_dir / f"seed_{seed:02d}"
                    record = {
                        "case": args.case,
                        "fold": fold,
                        "condition": condition,
                        "method": method,
                        "seed": seed,
                        "device": args.device,
                        "test_sources": test_sources,
                        "observed_count": int(len(model["observed_indices"])),
                        "heldout_count": int(len(model["heldout_indices"])),
                        "test_truth_used_for_fit": False,
                        "paired_noise_digest": str(representative_payload["paired_noise_digest"]),
                        "source_metrics": source_metrics,
                        **aggregated,
                        "completed": True,
                    }
                    stem.with_suffix(".json").write_text(json.dumps(record, indent=2), encoding="utf-8")
                    np.savez_compressed(stem.with_suffix(".npz"), **representative_payload)
                    completed.append(str(stem.with_suffix(".json")))
    manifest = {"case": args.case, "folds": args.folds, "seeds": args.seeds, "methods": list(METHODS), "conditions": list(CONDITIONS), "device": args.device, "completed_runs": len(completed), "records": completed}
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / f"worker_manifest_{args.case}_{args.device.replace(':','')}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"case": args.case, "completed_runs": len(completed), "device": args.device}, indent=2))


if __name__ == "__main__":
    main()
