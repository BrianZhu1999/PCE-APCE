from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the immutable Figure 2 50-seed manifest.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result-root", default="<HILDA_RESULTS_ROOT>/results/figure2_formal_50seeds_20260807")
    parser.add_argument("--seed-base", type=int, default=2026080700)
    parser.add_argument("--replicates", type=int, default=50)
    args = parser.parse_args()

    jobs = []
    for case in ("wave", "spring", "heat"):
        for method in ("denkf", "letkf", "iensf", "pce", "apce", "oracle_alpha"):
            for replicate in range(args.replicates):
                seed = args.seed_base + replicate
                job_id = f"figure2_{case}_{method}_s{seed}"
                output = f"{args.result_root}/{job_id}.json"
                jobs.append({
                    "id": job_id,
                    "arguments": [
                        "--case", case,
                        "--method", method,
                        "--seed", str(seed),
                        "--output", output,
                    ],
                })
    manifest = {
        "schema_version": 1,
        "protocol": "figure2-formal-50paired-seeds-20260807-v1",
        "cases": ["wave", "spring", "heat"],
        "methods": ["denkf", "letkf", "iensf", "pce", "apce", "oracle_alpha"],
        "replicates": args.replicates,
        "seed_base": args.seed_base,
        "job_count": len(jobs),
        "result_root": args.result_root,
        "jobs": jobs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "job_count": len(jobs)}))


if __name__ == "__main__":
    main()
