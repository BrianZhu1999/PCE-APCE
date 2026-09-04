from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import run_benchmark_v3 as v3
from experiments.wave_scenario_assets import WaveScenarioAssets


ALPHA_LEVELS = (0.12, 0.50, 0.88)
SEEDS = tuple(range(5))


def freeze(output: Path, base_seed: int) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for alpha_true in ALPHA_LEVELS:
        for seed_index in SEEDS:
            seed = base_seed + seed_index
            cfg = dataclasses.replace(
                v3.make_config("quick"),
                seed=seed,
                nx=41,
                ensemble_size=18,
                n_alpha=7,
                t_end=1.0,
                dt=0.0025,
                obs_interval=20,
                n_sensors=6,
                alpha_true=alpha_true,
            )
            assets = WaveScenarioAssets.from_legacy_scenario(v3.generate_scenario(cfg))
            name = f"alpha_{alpha_true:.2f}_seed_{seed}"
            directory = output / name
            assets.save(directory)
            records.append({
                "name": name,
                "alpha_true": alpha_true,
                "seed": seed,
                "nx": assets.nx,
                "ensemble_size": assets.ensemble_size,
                "n_steps": assets.n_steps,
                "array_digest": assets.array_digest,
                "path": str(directory),
            })
    manifest = {
        "schema_version": 1,
        "profile": "gate_b_wave_assets",
        "alpha_true_levels": list(ALPHA_LEVELS),
        "paired_seed_indices": list(SEEDS),
        "base_seed": base_seed,
        "state_variable": "displacement_only_for_metrics; velocity retained in state",
        "job_count": len(records),
        "records": records,
    }
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    manifest["manifest_sha256"] = hashlib.sha256(payload).hexdigest()
    (output / "gate_b_assets_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, default=2026080700)
    args = parser.parse_args()
    manifest = freeze(args.output, args.base_seed)
    print("FROZEN", manifest["job_count"], manifest["manifest_sha256"][:16])
    for record in manifest["records"]:
        print("ASSET", record["name"], str(record["array_digest"])[:16])


if __name__ == "__main__":
    main()
