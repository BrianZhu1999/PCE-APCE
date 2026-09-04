#!/usr/bin/env python3
"""Run frozen target-specific dual-source APCE formal evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import run_cartesian_pce_apce as runner


SEEDS = (2026082601, 2026082602, 2026082603, 2026082604, 2026082605)
CONFIG = {
    1: {
        "calibration_name": "midq_s8_turn010",
        "q_min": 1.0,
        "q_max": 8.0,
        "observation_covariance_scale": 8.0,
        "turn_rate_radps": 0.10,
    },
    2: {
        "calibration_name": "lowq_s1_turn010",
        "q_min": 0.5,
        "q_max": 4.0,
        "observation_covariance_scale": 1.0,
        "turn_rate_radps": 0.10,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-validation", type=Path, required=True)
    parser.add_argument("--target", type=int, choices=(1, 2), required=True)
    parser.add_argument("--device", choices=("cuda:2", "cuda:3"), required=True)
    args = parser.parse_args()

    target = args.target
    config = CONFIG[target]
    calibration_manifest = args.calibration_validation / f"target{target}" / "calibration_validation_manifest.json"
    validation = json.loads(calibration_manifest.read_text(encoding="utf-8"))
    if validation["selected"]["config"] != config["calibration_name"]:
        raise RuntimeError("frozen configuration does not match calibration selection")
    frontend = args.root / f"target{target}" / "frontend"
    destination = args.output / f"target{target}"
    completed = []
    for seed in SEEDS:
        payload = runner.run_track(
            frontend=frontend,
            output=destination / "runs",
            method="apce",
            seed=seed,
            device_name=args.device,
            segment="dual_evaluation",
            ensemble_size=48,
            q_min=float(config["q_min"]),
            q_max=float(config["q_max"]),
            position_init_std=50.0,
            velocity_init_std=10.0,
            observation_covariance_scale=float(config["observation_covariance_scale"]),
            turn_rate_radps=float(config["turn_rate_radps"]),
        )
        completed.append({"seed": seed, "status": payload["status"], "records": len(payload["records"])})
    manifest = {
        "target": target,
        "method": "apce",
        "seeds": list(SEEDS),
        "segment": "dual_evaluation",
        "configuration": {
            **config,
            "ensemble_size": 48,
            "position_init_std_m": 50.0,
            "velocity_init_std_mps": 10.0,
        },
        "calibration_validation_manifest": str(calibration_manifest),
        "calibration_validation_manifest_sha256": sha256(calibration_manifest),
        "gps_role": "offline evaluation only",
        "completed": completed,
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "formal_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
