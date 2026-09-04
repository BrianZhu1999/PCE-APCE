from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import run_modern_baseline_admission as modern  # noqa: E402
from experiments import run_wave_repair_validation as wave_repair  # noqa: E402
from figures import export_figure2_v2_representative_source as v2rep  # noqa: E402
from paper_experiments import run_spring_heat_gate as sh  # noqa: E402


METHODS = ("letkf", "aug_enkf", "bma_static", "pce", "apce")


def _seed_values(seed_args: list[int], seed_start: int | None, seed_end: int | None) -> list[int]:
    seeds = list(seed_args)
    if seed_start is not None or seed_end is not None:
        if seed_start is None or seed_end is None:
            raise ValueError("Both --seed-start and --seed-end are required when using a seed range.")
        seeds.extend(range(seed_start, seed_end + 1))
    if not seeds:
        seeds = list(range(2026080700, 2026080750))
    return sorted(dict.fromkeys(int(seed) for seed in seeds))


def _wave_curves(seed: int, device: torch.device) -> dict[str, np.ndarray]:
    assets = modern.make_wave_assets(seed)
    node = int(assets.nx // 2)
    curves: dict[str, np.ndarray] = {
        "times": np.asarray(assets.times, dtype=float),
        "truth": np.asarray(assets.truth_states[:, node], dtype=float),
        "node": np.asarray(node),
        "alpha_true": np.asarray(float(assets.alpha_true)),
    }
    curves["letkf"] = np.asarray(wave_repair.trace_single_path(assets, "letkf", device)[:, node], dtype=float)
    curves["aug_enkf"] = np.asarray(v2rep._wave_aug_trace(assets, device)[:, node], dtype=float)
    bma_trace, _ = v2rep._wave_bma_trace(assets, device)
    curves["bma_static"] = np.asarray(bma_trace[:, node], dtype=float)
    pce_trace, _ = v2rep._wave_refined_v2_trace(assets, "pce_refined_v2", device)
    apce_trace, _ = v2rep._wave_refined_v2_trace(assets, "apce_refined_v2", device)
    curves["pce"] = np.asarray(pce_trace[:, node], dtype=float)
    curves["apce"] = np.asarray(apce_trace[:, node], dtype=float)
    return curves


def _spring_curves(seed: int, device: torch.device) -> dict[str, np.ndarray]:
    config = sh.config_for_case("spring", seed)
    scenario = sh.generate_scenario(config, device)
    times = np.arange(config.steps + 1, dtype=float) * float(config.dt)
    curves: dict[str, np.ndarray] = {
        "times": times,
        "truth": np.asarray(scenario.truth.detach().cpu().numpy()[:, 0], dtype=float),
        "alpha_true": np.asarray(float(config.alpha_true)),
    }
    letkf_result = sh.run_method(scenario, "letkf", device, record_trace=True)
    curves["letkf"] = np.asarray(letkf_result["mean_states"], dtype=float)[:, 0]
    curves["aug_enkf"] = np.asarray(v2rep._spring_heat_aug_trace(scenario, device), dtype=float)[:, 0]
    bma_trace, _ = v2rep._spring_heat_bma_trace(scenario, device)
    curves["bma_static"] = np.asarray(bma_trace, dtype=float)[:, 0]
    pce_trace, _ = v2rep._spring_heat_refined_v2_trace(scenario, "pce_refined_v2", device)
    apce_trace, _ = v2rep._spring_heat_refined_v2_trace(scenario, "apce_refined_v2", device)
    curves["pce"] = np.asarray(pce_trace, dtype=float)[:, 0]
    curves["apce"] = np.asarray(apce_trace, dtype=float)[:, 0]
    return curves


def _heat_curves(seed: int, device: torch.device) -> dict[str, np.ndarray]:
    config = sh.config_for_case("heat", seed)
    scenario = sh.generate_scenario(config, device)
    nx = int(scenario.truth.shape[-1])
    curves: dict[str, np.ndarray] = {
        "space": np.linspace(0.0, 1.0, nx),
        "truth": np.asarray(scenario.truth.detach().cpu().numpy()[-1], dtype=float),
        "alpha_true": np.asarray(float(config.alpha_true)),
    }
    letkf_result = sh.run_method(scenario, "letkf", device, record_trace=True)
    curves["letkf"] = np.asarray(letkf_result["mean_states"], dtype=float)[-1]
    curves["aug_enkf"] = np.asarray(v2rep._spring_heat_aug_trace(scenario, device), dtype=float)[-1]
    bma_trace, _ = v2rep._spring_heat_bma_trace(scenario, device)
    curves["bma_static"] = np.asarray(bma_trace, dtype=float)[-1]
    pce_trace, _ = v2rep._spring_heat_refined_v2_trace(scenario, "pce_refined_v2", device)
    apce_trace, _ = v2rep._spring_heat_refined_v2_trace(scenario, "apce_refined_v2", device)
    curves["pce"] = np.asarray(pce_trace, dtype=float)[-1]
    curves["apce"] = np.asarray(apce_trace, dtype=float)[-1]
    return curves


def export_one(case: str, seed: int, output: Path, device: torch.device) -> dict[str, str]:
    output.mkdir(parents=True, exist_ok=True)
    if case == "wave":
        arrays = _wave_curves(seed, device)
    elif case == "spring":
        arrays = _spring_curves(seed, device)
    elif case == "heat":
        arrays = _heat_curves(seed, device)
    else:
        raise ValueError(case)
    arrays["seed"] = np.asarray(seed)
    arrays["case"] = np.asarray(case)
    path = output / f"{case}_seed_{seed}_seedband_curves.npz"
    np.savez_compressed(path, **arrays)
    return {"case": case, "seed": str(seed), "path": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export per-seed curve traces for Figure 2 seed-band panels.")
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--seed-end", type=int)
    parser.add_argument("--case", choices=("wave", "spring", "heat"), action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    seeds = _seed_values(args.seed, args.seed_start, args.seed_end)
    cases = args.case or ["wave", "spring", "heat"]
    device = torch.device(args.device)
    completed: list[dict[str, str]] = []
    failed: list[dict[str, str]] = []
    for seed in seeds:
        for case in cases:
            try:
                completed.append(export_one(case, seed, args.output, device))
            except Exception as exc:  # pragma: no cover - runtime audit record
                failed.append({"case": case, "seed": str(seed), "error": repr(exc)})
                status_path = args.output / f"FAILED_{case}_seed_{seed}.json"
                status_path.write_text(json.dumps(failed[-1], indent=2, ensure_ascii=False), encoding="utf-8")
                raise
    manifest = {
        "device": str(device),
        "seeds": seeds,
        "cases": cases,
        "methods": METHODS,
        "completed": completed,
        "failed": failed,
    }
    manifest_path = args.output / f"manifest_{seeds[0]}_{seeds[-1]}_{str(device).replace(':', '')}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"completed": len(completed), "failed": len(failed), "manifest": str(manifest_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
