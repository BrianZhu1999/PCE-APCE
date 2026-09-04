#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from f16_gvt.assimilation import run_two_pass
from f16_gvt.candidates import ModalCandidateFamily
from f16_gvt.identification import load_model
from f16_gvt.metrics import evaluate_pass


HERE = Path(__file__).resolve().parent
METHODS = ("DEnKF", "BMA", "PCE", "APCE")
CONDITIONS = ("standard", "blackout")


def load_payload(cache: Path, level: int) -> dict[str, np.ndarray]:
    data = np.load(cache / f"fullmsine_level{level}_processed.npz")
    return {key: data[key] for key in data.files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", type=int, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--cache", type=Path, default=HERE / "cache")
    parser.add_argument("--models", type=Path, default=HERE / "models")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    allowed = set(int(value) for value in config["levels"]["validation"])
    if not set(args.levels).issubset(allowed):
        raise ValueError(f"pilot levels must be held-out validation levels {sorted(allowed)}")
    model = load_model(args.models)
    path = json.loads((args.models / "modal_uncertainty_path.json").read_text(encoding="utf-8"))
    family = ModalCandidateFamily(
        model,
        path,
        int(config["candidates"]["envelope_quantization_bins"]),
        float(config["candidates"]["maximum_frequency_scale"]),
        float(config["candidates"]["maximum_damping_log_scale"]),
    )
    ensemble_size = int(config["assimilation"]["ensemble_size_smoke"])
    max_candidates = int(config["candidates"]["local_points"]) + 1
    observed = [int(value) for value in config["observed_channels_zero_based"]]
    heldout = int(config["heldout_channels_zero_based"][0])
    burnin = int(round(float(config["identification"]["warmup_seconds"]) * float(config["processed_rate_hz"])))
    completed = []
    for level in args.levels:
        payload = load_payload(args.cache, level)
        force = payload["force"].reshape(-1)
        acceleration = payload["acceleration"].reshape(-1, 3)
        for seed in args.seeds:
            rng = np.random.default_rng(2026081700 + 100 * level + seed)
            initial_noise = rng.standard_normal((max_candidates, ensemble_size, model.order)).astype(np.float32)
            forecast_noise = rng.standard_normal((len(force), max_candidates, ensemble_size, model.order)).astype(np.float32)
            for condition in args.conditions:
                for method in args.methods:
                    print(f"level={level} seed={seed} condition={condition} method={method} device={args.device}", flush=True)
                    result = run_two_pass(
                        method,
                        model,
                        family,
                        force,
                        acceleration,
                        observed,
                        config,
                        args.device,
                        initial_noise,
                        forecast_noise,
                        condition,
                    )
                    selected = result.local
                    metrics = evaluate_pass(
                        selected,
                        acceleration,
                        heldout,
                        burnin,
                        float(config["processed_rate_hz"]),
                        tuple(float(value) for value in config["filter_band_hz"]),
                    )
                    blackout_times = np.flatnonzero(selected.blackout_mask) / float(config["processed_rate_hz"])
                    weight_update_inside_blackout = bool(
                        blackout_times.size
                        and np.any((selected.weight_time >= blackout_times[0]) & (selected.weight_time <= blackout_times[-1]))
                    )
                    output_dir = args.output_root / f"level{level}" / condition / method.lower()
                    output_dir.mkdir(parents=True, exist_ok=True)
                    stem = output_dir / f"seed_{seed:02d}"
                    record = {
                        "case": "f16_gvt_7p3hz",
                        "level": level,
                        "condition": condition,
                        "method": method,
                        "seed": seed,
                        "device": args.device,
                        "ensemble_size": ensemble_size,
                        "observed_channels_zero_based": observed,
                        "heldout_channels_zero_based": [heldout],
                        "background_level": 1,
                        "estimation_levels": config["levels"]["estimation"],
                        "validation_level": level,
                        "validation_used_for_selection": False,
                        "coarse_grid": result.coarse.grid.tolist(),
                        "local_grid": selected.grid.tolist(),
                        "final_weights": selected.final_weights.tolist(),
                        "paired_noise_digest": selected.paired_noise_digest,
                        "weight_update_inside_blackout": weight_update_inside_blackout,
                        "completed": True,
                        **{key: value for key, value in metrics.items() if not isinstance(value, list)},
                        "candidate_nrmse": metrics["candidate_nrmse"],
                    }
                    stem.with_suffix(".json").write_text(json.dumps(record, indent=2), encoding="utf-8")
                    np.savez_compressed(
                        stem.with_suffix(".npz"),
                        time=np.arange(len(force)) / float(config["processed_rate_hz"]),
                        force=force,
                        truth=acceleration,
                        mean=selected.mean_physical,
                        branch_mean=selected.branch_mean_physical,
                        lower90=selected.lower90_physical,
                        upper90=selected.upper90_physical,
                        evaluation_indices=selected.evaluation_indices,
                        grid=selected.grid,
                        final_weights=selected.final_weights,
                        weight_time=selected.weight_time,
                        weight_history=selected.weight_history,
                        score_history=selected.score_history,
                        blackout_mask=selected.blackout_mask,
                    )
                    completed.append(str(stem.with_suffix(".json")))
    manifest = {
        "levels": args.levels,
        "seeds": args.seeds,
        "methods": args.methods,
        "conditions": args.conditions,
        "device": args.device,
        "completed_runs": len(completed),
        "records": completed,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    suffix = (
        "_".join(str(value) for value in args.levels)
        + "__" + "_".join(str(value) for value in args.seeds)
        + "__" + "_".join(args.conditions)
        + "__" + "_".join(args.methods)
    )
    (args.output_root / f"worker_manifest_{suffix}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
