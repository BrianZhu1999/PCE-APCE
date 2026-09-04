#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from f16_gvt.candidates import ModalCandidateFamily, fit_bootstrap_path
from f16_gvt.identification import fit_select_refit, save_model


HERE = Path(__file__).resolve().parent


def load_payload(cache: Path, level: int) -> dict[str, np.ndarray]:
    data = np.load(cache / f"fullmsine_level{level}_processed.npz")
    return {key: data[key] for key in data.files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=HERE / "cache")
    parser.add_argument("--output-root", type=Path, default=HERE / "models")
    args = parser.parse_args()
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    level1 = load_payload(args.cache, 1)
    model, identification = fit_select_refit(level1, config)
    save_model(model, identification, args.output_root)
    estimation = {level: load_payload(args.cache, level) for level in config["levels"]["estimation"]}
    path = fit_bootstrap_path(model, estimation, config)
    family = ModalCandidateFamily(
        model,
        path,
        int(config["candidates"]["envelope_quantization_bins"]),
        float(config["candidates"]["maximum_frequency_scale"]),
        float(config["candidates"]["maximum_damping_log_scale"]),
    )
    grid = np.asarray(config["candidates"]["coarse_alpha"], dtype=float)
    candidate_audit = family.audit_grid(grid)
    path["coarse_candidate_audit"] = candidate_audit
    path["validation_levels_used"] = False
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "modal_uncertainty_path.json").write_text(json.dumps(path, indent=2), encoding="utf-8")
    result = {
        "selected_order": model.order,
        "target_mode": identification["final_target_mode"],
        "bootstrap_successful": path["successful_replicates"],
        "bootstrap_requested": path["requested_replicates"],
        "principal_explained_fraction": path["explained_fraction"],
        "all_coarse_candidates_stable": all(row["stable"] for row in candidate_audit),
        "validation_levels_used": False,
    }
    (args.output_root / "model_candidate_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
