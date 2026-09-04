#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from meshir.data import farthest_point_sampling, s1_spatial_folds
from meshir.geometry import idw_extension
from meshir.rom import build_spatial_basis, candidate_paths, fit_pod, residual_statistics, estimate_decay_rate


def save_s1_fold(output: Path, rir: np.ndarray, positions: np.ndarray, fold: int, config: dict) -> dict:
    folds = s1_spatial_folds(positions)
    heldout = folds[fold]
    calibration = np.setdiff1d(np.arange(len(positions)), heldout)
    observed_local = farthest_point_sampling(positions[calibration], int(config["dataset"]["s1_observation_count_pilot"]))
    observed = calibration[observed_local]
    basis_cal, coefficients, mean_cal = fit_pod(rir[:, calibration], int(config["rom"]["rank_s1"]))
    basis_full = build_spatial_basis(positions[calibration], basis_cal, positions, int(config["rom"]["spatial_knn"]))
    mean_full = idw_extension(positions[calibration], mean_cal[:, None], positions, int(config["rom"]["spatial_knn"]))[:, 0]
    rate = 16000.0
    decay = estimate_decay_rate(coefficients, rate)
    paths, parameters = candidate_paths(coefficients, rate, config["rom"]["speed_scale"], config["rom"]["damping_scale"], decay)
    fixed = paths.mean(axis=0)
    rho, state_cov, process_cov, observation_cov = residual_statistics([coefficients], fixed)
    np.savez_compressed(
        output / f"s1_fold{fold}.npz",
        basis_full=basis_full, mean_full=mean_full, candidate_paths=paths,
        candidate_parameters=parameters, fixed_path=fixed, rho=np.array(rho),
        state_covariance=state_cov, process_covariance=process_cov,
        observation_covariance=observation_cov, observed_indices=observed,
        heldout_indices=heldout, calibration_indices=calibration,
        positions=positions, fold=np.array(fold), decay_rate=np.array(decay),
    )
    return {"fold": fold, "observed": int(len(observed)), "heldout": int(len(heldout)), "decay_rate": decay, "candidate_count": len(paths)}


def save_s32_fold(output: Path, rir: np.ndarray, positions: np.ndarray, sources: np.ndarray, fold: int, config: dict) -> dict:
    source_folds = np.array_split(np.arange(len(sources)), 4)
    test_sources = source_folds[fold]
    train_sources = np.setdiff1d(np.arange(len(sources)), test_sources)
    observed = farthest_point_sampling(positions, int(config["dataset"]["s32_observation_count_pilot"]))
    matrix = rir[train_sources].reshape(-1, rir.shape[2])
    basis, coefficients_all, mean_full = fit_pod(matrix, int(config["rom"]["rank_s32"]))
    coefficients = coefficients_all.reshape(len(train_sources), rir.shape[1], -1)
    source_paths = coefficients
    fixed = source_paths.mean(axis=0)
    trajectories = [source_paths[index] for index in range(len(train_sources))]
    rho, state_cov, process_cov, observation_cov = residual_statistics(trajectories, fixed)
    np.savez_compressed(
        output / f"s32_fold{fold}.npz",
        basis_full=basis, mean_full=mean_full, candidate_paths=source_paths,
        candidate_source_positions=sources[train_sources], candidate_source_indices=train_sources,
        test_source_indices=test_sources, fixed_path=fixed, rho=np.array(rho),
        state_covariance=state_cov, process_covariance=process_cov,
        observation_covariance=observation_cov, observed_indices=observed,
        heldout_indices=np.setdiff1d(np.arange(len(positions)), observed),
        positions=positions, sources=sources, fold=np.array(fold),
    )
    return {"fold": fold, "train_sources": train_sources.tolist(), "test_sources": test_sources.tolist(), "observed": int(len(observed)), "candidate_count": len(source_paths)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.cache / "geometry.npz") as geometry:
        s1_positions, s32_positions, s32_sources = geometry["s1_positions"], geometry["s32_positions"], geometry["s32_sources"]
    s1 = np.load(args.cache / "s1_rir_16k.npy", mmap_mode="r")
    s32 = np.load(args.cache / "s32_rir_16k.npy", mmap_mode="r")
    records = {"s1": [save_s1_fold(args.output, s1, s1_positions, fold, config) for fold in range(4)], "s32": [save_s32_fold(args.output, s32, s32_positions, s32_sources, fold, config) for fold in range(4)]}
    manifest = {"case": "meshir_s1_s32_pilot", "models": records, "test_truth_used_for_fit": False}
    (args.output / "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
