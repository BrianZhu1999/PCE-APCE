from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand the frozen HILDA experiment matrix")
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("primary",), default="primary")
    return parser.parse_args()


def build_primary_jobs(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    output_root = matrix["result_root"]
    seed_base = int(matrix["seed_protocol"]["base"])
    jobs: list[dict[str, Any]] = []
    for system_name, system in matrix["systems"].items():
        for method in matrix["methods"]:
            for replicate in range(int(system["replicates"])):
                seed = seed_base + replicate
                job_id = f"primary_{system_name}_{method}_s{seed}"
                observation_transform = system.get(
                    "primary_observation_transform",
                    matrix["primary_protocol"]["observation_transform"],
                )
                arguments = [
                    "--system", system_name,
                    "--method", method,
                    "--seed", str(seed),
                    "--ensemble-size", str(matrix["primary_protocol"]["ensemble_size"]),
                    "--steps", str(system["steps"]),
                    "--observation-interval", str(system["observation_interval"]),
                    "--observation-count", str(system["observation_count"]),
                    "--observation-noise", str(system["observation_noise"]),
                    "--observation-transform", observation_transform,
                    "--alpha-mode", matrix["primary_protocol"]["alpha_mode"],
                    "--alpha-true", str(matrix["alpha_protocol"]["on_grid"]),
                    "--fixed-model-alpha", str(matrix["primary_protocol"]["fixed_model_alpha"]),
                    "--coverage-level", str(matrix["primary_protocol"]["coverage_level"]),
                    "--energy-score-chunk-size", str(
                        matrix["primary_protocol"]["energy_score_chunk_size"]
                    ),
                    "--checkpoint-interval", str(matrix["primary_protocol"]["checkpoint_interval"]),
                    "--output-root", output_root,
                ]
                if "enff_grid_size" in system:
                    arguments.extend(("--enff-grid-size", str(system["enff_grid_size"])))
                if "dtype" in system:
                    arguments.extend(("--dtype", system["dtype"]))
                jobs.append({"id": job_id, "arguments": arguments})
    return jobs


def main() -> None:
    args = parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    jobs = build_primary_jobs(matrix)
    manifest = {
        "schema_version": 1,
        "protocol_version": matrix["protocol_version"],
        "profile": args.profile,
        "job_count": len(jobs),
        "jobs": jobs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "jobs": len(jobs)}))


if __name__ == "__main__":
    main()
