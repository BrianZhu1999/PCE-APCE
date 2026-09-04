"""Decode saved latent estimates against sealed full fields after inference.

This command is evaluation-only: the decoded full field is never passed back to
state updates, candidate scores, hyperparameters or stopping decisions.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch

from .common import load_config, write_json
from .io import VIVCase, list_cases, sha256_file
from .metrics import full_field_metrics
from .rom import PODModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Postprocess one sealed VIV-PIV full field.")
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_config(args.config)
    case_id = str(args.case).replace(",", "").zfill(4)[-4:]
    if case_id not in config["test_cases"]:
        raise ValueError("Full-field evaluation is restricted to configured external tests")
    variant = args.variant or f"rank{int(config['rank'])}_stride1"
    run_root = pathlib.Path(config["output_root"]) / "runs" / variant
    run_id = args.run_id or f"viv_{case_id}_{args.method}_seed{args.seed:03d}"
    json_path = run_root / "runs" / f"{run_id}.json"
    trace_path = run_root / "traces" / f"{run_id}.npz"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not payload.get("valid"):
        raise ValueError(f"Cannot postprocess invalid run {run_id}")
    with np.load(trace_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    latent = np.asarray(arrays["latent_estimate"], dtype=np.float64)
    model_root = pathlib.Path(config["output_root"]) / "models" / variant
    pod = PODModel.load(model_root / "pod_model.npz")
    case_path = list_cases(pathlib.Path(config["data_root"]))[case_id]
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    excluded = None
    sensor_layout = payload.get("sensor_layout")
    if sensor_layout and sensor_layout != "8x5":
        sensor_path = model_root / "sensor_layouts" / sensor_layout / f"case_{case_id}.npz"
        with np.load(sensor_path, allow_pickle=False) as sensor:
            excluded = np.asarray(sensor["sensor_flat_indices"], dtype=np.int64)
    metrics, traces = full_field_metrics(
        VIVCase.open(case_path), pod, latent, device, excluded_flat_indices=excluded
    )
    arrays.update(traces)
    np.savez_compressed(trace_path, **arrays)
    payload.update(metrics)
    payload["full_field_postprocess_only"] = True
    payload["full_field_used_for_inference"] = False
    payload["trace_sha256"] = sha256_file(trace_path)
    write_json(json_path, payload)
    print(json.dumps({"run_id": run_id, **metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
