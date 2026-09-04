from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_experiments.run_spring_heat_gate import (  # noqa: E402
    METHOD_LABELS,
    config_for_case,
    generate_scenario,
    run_method,
)


FIGURE_METHODS = ("denkf", "pce", "apce")


def export_case(case: str, seed: int, output: Path, device: torch.device) -> dict[str, object]:
    config = config_for_case(case, seed)
    scenario = generate_scenario(config, device)
    times = np.arange(config.steps + 1, dtype=float) * config.dt
    arrays: dict[str, np.ndarray] = {
        "times": times,
        "truth_states": scenario.truth.detach().cpu().numpy(),
        "observation_indices": scenario.observation_indices.detach().cpu().numpy(),
        "primary_indices": scenario.primary_indices.detach().cpu().numpy(),
        "alpha_grid": scenario.alpha_grid.detach().cpu().numpy(),
    }
    metrics: dict[str, dict[str, float]] = {}
    for method in FIGURE_METHODS:
        result = run_method(scenario, method, device, record_trace=True)
        arrays[f"{method}_mean_states"] = np.asarray(result.pop("mean_states"))
        if "alpha_weight_history" in result:
            arrays[f"{method}_alpha_weight_history"] = np.asarray(result.pop("alpha_weight_history"))
        metrics[method] = {key: float(value) for key, value in result.items() if isinstance(value, (int, float))}

    if case == "heat":
        # Heat1D uses an evenly spaced unit interval grid with fixed endpoints.
        arrays["space"] = np.linspace(0.0, 1.0, scenario.truth.shape[-1])
    else:
        arrays["state_components"] = np.asarray(["displacement", "velocity"])

    out_file = output / f"{case}_representative_seed_{seed}.npz"
    np.savez_compressed(out_file, **arrays)
    metrics_file = output / f"{case}_representative_seed_{seed}_metrics.json"
    metrics_file.write_text(
        json.dumps(
            {
                "case": case,
                "seed": seed,
                "methods": {method: METHOD_LABELS[method] for method in FIGURE_METHODS},
                "metrics": metrics,
                "config": {
                    "steps": config.steps,
                    "dt": config.dt,
                    "obs_interval": config.obs_interval,
                    "ensemble_size": config.ensemble_size,
                    "alpha_true": config.alpha_true,
                    "fixed_alpha": config.fixed_alpha,
                    "alpha_grid": list(config.alpha_grid),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"source": str(out_file), "metrics": str(metrics_file)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export representative Spring/Heat traces for NCS figures.")
    parser.add_argument("--seed", type=int, default=2026080600)
    parser.add_argument("--output", type=Path, default=ROOT / "figures" / "spring_heat_repair_template")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    manifest = {
        case: export_case(case, args.seed, args.output, device)
        for case in ("spring", "heat")
    }
    manifest_path = args.output / "representative_source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
