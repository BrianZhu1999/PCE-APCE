from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand the frozen PDEBench replay matrix")
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_jobs(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    data = matrix["data"]
    protocol = matrix["replay_protocol"]
    jobs = []
    sample_index = 0
    for file_index in data["file_indices"]:
        filename = data["filename_template"].format(file_index=file_index)
        data_path = str(PurePosixPath(data["root"]) / filename)
        for trajectory_index in data["trajectory_indices"]:
            seed = int(matrix["seed_base"]) + sample_index
            for method in matrix["methods"]:
                job_id = (
                    f"pdebench_f{file_index}_tr{trajectory_index}_{method}_s{seed}"
                )
                arguments = [
                    "--data", data_path,
                    "--manifest", data["manifest"],
                    "--collection-validation", data["collection_validation"],
                    "--trajectory-index", str(trajectory_index),
                    "--time-start", str(protocol["time_start"]),
                    "--time-stop", str(protocol["time_stop"]),
                    "--time-step", str(protocol["time_step"]),
                    "--spatial-stride", str(protocol["spatial_stride"]),
                    "--method", method,
                    "--seed", str(seed),
                    "--ensemble-size", str(protocol["ensemble_size"]),
                    "--sensor-count", str(protocol["sensor_count"]),
                    "--observation-noise", str(protocol["observation_noise"]),
                    "--observation-transform", protocol["observation_transform"],
                    "--initial-spread", str(protocol["initial_spread"]),
                    "--process-noise", str(protocol["process_noise"]),
                    "--noise-smoothing-radius", str(protocol["noise_smoothing_radius"]),
                    "--hilda-path-log-scale", str(protocol["hilda_path_log_scale"]),
                    "--forecast-model", protocol["forecast_model"],
                    "--coverage-level", str(protocol["coverage_level"]),
                    "--energy-score-chunk-size", str(protocol["energy_score_chunk_size"]),
                    "--dtype", protocol["dtype"],
                    "--checkpoint-interval", str(protocol["checkpoint_interval"]),
                    "--output-root", matrix["result_root"],
                ]
                jobs.append({"id": job_id, "arguments": arguments})
            sample_index += 1
    return jobs


def main() -> None:
    args = parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    jobs = build_jobs(matrix)
    manifest = {
        "schema_version": 1,
        "protocol_version": matrix["protocol_version"],
        "profile": "pdebench_replay",
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
