from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.run_wave_repair_validation import trace_pce_family, trace_single_path
from experiments.wave_scenario_assets import WaveScenarioAssets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export representative APCE/PCE wave-repair source arrays."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--asset-name", default="alpha_0.12_seed_2026080700")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    matches = [record for record in manifest["records"] if record["name"] == args.asset_name]
    if not matches:
        raise ValueError(f"Asset not found in manifest: {args.asset_name}")

    device = torch.device(args.device)
    assets = WaveScenarioAssets.load(Path(matches[0]["path"]))
    denkf_trace = trace_single_path(assets, "denkf", device)
    pce_trace, pce_weights = trace_pce_family(assets, "pce", device)
    apce_trace, apce_weights = trace_pce_family(assets, "apce", device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        asset_name=np.asarray(args.asset_name),
        seed=np.asarray(assets.seed),
        alpha_true=np.asarray(assets.alpha_true),
        times=assets.times,
        observation_indices=assets.observation_indices,
        observation_mask=assets.observation_mask,
        truth_states=assets.truth_states,
        denkf_mean_states=denkf_trace,
        pce_mean_states=pce_trace,
        apce_mean_states=apce_trace,
        pce_final_weights=pce_weights,
        apce_final_weights=apce_weights,
    )
    print(
        json.dumps(
            {
                "asset": args.asset_name,
                "device": str(device),
                "shape": list(assets.truth_states.shape),
                "methods": ["truth", "denkf", "pce", "apce"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
