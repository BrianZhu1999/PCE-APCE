#!/usr/bin/env python3
"""Run the measured 3D acoustic-field reconstruction protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .assimilation import run
from .boundary_candidates import rbf_boundary_candidate_series_from_sparse
from .metrics import evaluate_region
from .model import cfl_number


HERE = Path(__file__).resolve().parent
METHODS = ("DEnKF", "BMA", "PCE", "APCE")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda:2", "cuda:3"), default="cuda:2")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nominal-index", type=int, default=None)
    parser.add_argument("--store-candidate-history", action="store_true")
    args = parser.parse_args()

    config = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    config["assimilation"]["seed"] = int(args.seed)
    config["assimilation"]["store_candidate_history"] = bool(args.store_candidate_history)
    if args.nominal_index is not None:
        config["physics"]["nominal_boundary_closure_index"] = int(args.nominal_index)
    with np.load(args.source_npz) as source:
        truth = np.asarray(source["truth"], dtype=np.float32)
        positions = np.asarray(source["positions"], dtype=float)
        mapping = np.asarray(source["mapping"], dtype=int)
        boundary_flat = np.asarray(source["boundary_flat"], dtype=int)
        interior_flat = np.asarray(source["interior_flat"], dtype=int)
        heldout_interior = np.asarray(source["heldout_interior"], dtype=int)
        heldout_boundary = np.asarray(source["heldout_boundary"], dtype=int)
    grid_positions = positions[mapping.reshape(-1)]
    boundary_series, candidate_labels = rbf_boundary_candidate_series_from_sparse(
        truth, boundary_flat, grid_positions
    )
    analysis_end = min(1024, len(truth))
    rate = 16000.0
    signal_std = float(np.std(truth[:analysis_end].reshape(analysis_end, -1)[:, interior_flat]))
    process_std = float(config["physics"]["process_noise_fraction"]) * max(signal_std, 1e-8)
    ensemble = int(config["assimilation"]["ensemble_size"])
    candidate_count = len(candidate_labels)
    rng = np.random.default_rng(2026082100 + int(args.seed))
    initial_noise = (
        0.1 * process_std * rng.standard_normal((candidate_count, ensemble, 2, 9, 21, 21))
    ).astype(np.float32)
    run_config = json.loads(json.dumps(config))
    cfl = cfl_number(
        float(config["physics"]["nominal_speed_m_s"]),
        1.0 / rate,
        float(config["grid"]["spacing_m"]),
    )
    if cfl >= 1.0:
        raise ValueError(f"unstable CFL number {cfl}")

    args.output.mkdir(parents=True, exist_ok=True)
    preparation = {
        "case": "acoustic_field_reconstruction",
        "source_npz": str(args.source_npz),
        "candidate_labels": candidate_labels,
        "candidate_parameter": "RBF boundary-closure family",
        "nominal_boundary_closure_index": int(config["physics"]["nominal_boundary_closure_index"]),
        "fixed_speed_m_s": float(config["physics"]["nominal_speed_m_s"]),
        "localization_radius_m": float(config["assimilation"]["localization_radius_m"]),
        "ensemble_inflation": float(config["assimilation"]["ensemble_inflation"]),
        "ensemble_size": ensemble,
        "model_band_hz": 350.0,
    }
    (args.output / "preparation.json").write_text(json.dumps(preparation, indent=2), encoding="utf-8")
    records = []
    for method in args.methods:
        print(f"method={method} device={args.device} seed={args.seed}", flush=True)
        result = run(
            method,
            truth,
            boundary_series,
            interior_flat,
            np.asarray([float(config["physics"]["nominal_speed_m_s"])]),
            run_config,
            args.device,
            initial_noise,
            noise_seed=2026082117 + int(args.seed),
            process_noise_std=process_std,
        )
        metrics = {}
        metrics.update(evaluate_region(
            truth, result.mean, result.standard_deviation, heldout_interior,
            analysis_end, rate, "interior",
        ))
        metrics.update(evaluate_region(
            truth, result.mean, result.standard_deviation, heldout_boundary,
            analysis_end, rate, "boundary",
        ))
        record = {
            **preparation,
            "method": method,
            "seed": int(args.seed),
            "device": args.device,
            "final_weights": result.final_weights.tolist(),
            "median_separation_ratio": float(np.median(result.separation_history)) if len(result.separation_history) else 0.0,
            "cfl_number": cfl,
            "process_noise_std": process_std,
            **metrics,
            "completed": True,
        }
        stem = args.output / method.lower()
        stem.with_suffix(".json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        arrays = {
            "truth": truth,
            "mean": result.mean,
            "standard_deviation": result.standard_deviation,
            "boundary_flat": boundary_flat,
            "interior_flat": interior_flat,
            "heldout_interior": heldout_interior,
            "heldout_boundary": heldout_boundary,
            "positions": positions,
            "mapping": mapping,
            "candidate_labels": np.asarray(candidate_labels),
            "final_weights": result.final_weights,
            "weight_time": result.weight_time,
            "weight_history": result.weight_history,
            "separation_history": result.separation_history,
            "score_history": result.score_history,
        }
        if result.candidate_mean is not None:
            arrays["candidate_mean"] = result.candidate_mean
        np.savez_compressed(stem.with_suffix(".npz"), **arrays)
        records.append(str(stem.with_suffix(".json")))
    manifest = {
        **preparation,
        "methods": list(args.methods),
        "seed": int(args.seed),
        "records": records,
        "completed_runs": len(records),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
