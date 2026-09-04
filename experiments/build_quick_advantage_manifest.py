from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the paired HILDA advantage manifest")
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_jobs(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    methods = matrix["methods"]
    conditions = matrix["conditions"]
    if set(conditions) != {"matched", "misspecified"}:
        raise ValueError("conditions must be exactly matched and misspecified")
    protocol = matrix["protocol"]
    system = matrix["system"]
    fixed_alpha = float(protocol["fixed_model_alpha"])
    if float(conditions["matched"]["alpha_true"]) != fixed_alpha:
        raise ValueError("matched alpha_true must equal fixed_model_alpha")
    if float(conditions["misspecified"]["alpha_true"]) == fixed_alpha:
        raise ValueError("misspecified alpha_true must differ from fixed_model_alpha")

    jobs: list[dict[str, Any]] = []
    seed_base = int(matrix["seed_protocol"]["base"])
    replicates = int(matrix["seed_protocol"]["replicates"])
    for condition_name, condition in conditions.items():
        for method in methods:
            for replicate in range(replicates):
                seed = seed_base + replicate
                job_id = f"quick_{condition_name}_{system['name']}_{method}_s{seed}"
                arguments = [
                    "--system", str(system["name"]),
                    "--method", str(method),
                    "--seed", str(seed),
                    "--ensemble-size", str(protocol["ensemble_size"]),
                    "--steps", str(system["steps"]),
                    "--observation-interval", str(system["observation_interval"]),
                    "--observation-count", str(system["observation_count"]),
                    "--observation-noise", str(system["observation_noise"]),
                    "--observation-transform", str(system["observation_transform"]),
                    "--alpha-mode", str(protocol["alpha_mode"]),
                    "--alpha-true", str(condition["alpha_true"]),
                    "--fixed-model-alpha", str(protocol["fixed_model_alpha"]),
                    "--coverage-level", str(protocol["coverage_level"]),
                    "--energy-score-chunk-size", str(protocol["energy_score_chunk_size"]),
                    "--checkpoint-interval", str(protocol["checkpoint_interval"]),
                    "--dtype", str(protocol["dtype"]),
                    "--output-root", str(matrix["result_root"]),
                ]
                jobs.append({"id": job_id, "arguments": arguments})
    return jobs


def main() -> None:
    args = parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    jobs = build_jobs(matrix)
    manifest = {
        "schema_version": 1,
        "protocol_version": matrix["protocol_version"],
        "profile": "quick_advantage",
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
