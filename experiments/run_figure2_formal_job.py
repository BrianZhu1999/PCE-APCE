from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments import run_modern_baseline_admission as modern
from experiments import run_wave_repair_validation as wave_repair
from hilda_da.baselines import denkf_analysis
from hilda_da.observations import SparseObservation
from paper_experiments import run_spring_heat_gate as spring_heat


METHODS = ("denkf", "letkf", "iensf", "pce", "apce", "oracle_alpha")
LABELS = {
    "denkf": "DEnKF",
    "letkf": "LETKF",
    "iensf": "IEnSF",
    "pce": "PCE",
    "apce": "APCE",
    "oracle_alpha": "Oracle-alpha",
}


def source_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "hilda_da").rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    for path in sorted((root / "experiments").glob("run_*formal*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _common_metadata(case: str, method: str, seed: int, result: dict[str, Any]) -> dict[str, Any]:
    row = dict(result)
    if "nrmse" not in row and "displacement_nrmse" in row:
        row["nrmse"] = row["displacement_nrmse"]
    if "rmse" not in row and "displacement_rmse" in row:
        row["rmse"] = row["displacement_rmse"]
    row.update(
        case=case,
        method=method,
        label=LABELS[method],
        seed=seed,
        valid=bool(all(math.isfinite(float(row[key])) for key in ("nrmse", "rmse", "crps", "coverage_90", "interval_width_90"))),
        protocol="figure2-formal-50paired-seeds-20260807-v1",
        source_hash=source_hash(PROJECT_ROOT),
    )
    return row


def run_spring_heat_job(case: str, method: str, seed: int, device: torch.device) -> dict[str, Any]:
    scenario = spring_heat.generate_scenario(spring_heat.config_for_case(case, seed), device)
    if method in {"pce", "apce", "oracle_alpha"}:
        result = spring_heat.run_method(scenario, method, device)
    else:
        result = modern.run_spring_heat(case, method, seed, device)
    return _common_metadata(case, method, seed, result)


def run_wave_oracle(assets: modern.WaveScenarioAssets, device: torch.device) -> dict[str, Any]:
    cfg = wave_repair.configuration(assets)
    dtype = torch.float64
    ensemble = torch.as_tensor(assets.initial_ensemble, dtype=dtype, device=device)
    truth = torch.as_tensor(assets.truth_states, dtype=dtype, device=device)
    indices = torch.as_tensor(assets.observation_indices, dtype=torch.int64, device=device)
    operator = SparseObservation(indices)
    covariance = cfg.obs_noise**2 * torch.eye(indices.numel(), dtype=dtype, device=device)
    weights = torch.full((assets.ensemble_size,), 1.0 / assets.ensemble_size, dtype=dtype, device=device)
    metrics = wave_repair.MetricAccumulator(assets.nx)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(assets.n_steps + 1):
        metrics.add(ensemble, truth[step], weights)
        if step == assets.n_steps:
            break
        ensemble = wave_repair.propagate_numpy(
            ensemble,
            wave_repair.v3.alpha_to_theta(float(assets.alpha_true), cfg),
            step,
            assets,
            cfg,
        )
        if not assets.observation_mask[step + 1]:
            continue
        observation = torch.as_tensor(assets.observations[step + 1], dtype=dtype, device=device)
        ensemble = denkf_analysis(ensemble, observation, operator, covariance)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = metrics.finalize()
    result.update(
        runtime_seconds=time.perf_counter() - started,
        forward_member_steps=assets.n_steps * assets.ensemble_size,
        peak_gpu_memory_mb=(
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda" else 0.0
        ),
        alpha_estimate=float(assets.alpha_true),
        alpha_absolute_error=0.0,
    )
    return result


def run_wave_job(method: str, seed: int, device: torch.device) -> dict[str, Any]:
    assets = modern.make_wave_assets(seed)
    if method in {"denkf", "letkf", "iensf"}:
        result = modern.run_wave(method, seed, device)
    elif method in {"pce", "apce"}:
        result = wave_repair.run_pce_family(assets, method, device)
    elif method == "oracle_alpha":
        result = run_wave_oracle(assets, device)
    else:
        raise ValueError(method)
    return _common_metadata("wave", method, seed, result)


def main() -> None:
    parser = argparse.ArgumentParser(description="One immutable Figure 2 formal job.")
    parser.add_argument("--case", choices=("wave", "spring", "heat"), required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    started = time.perf_counter()
    try:
        if args.case == "wave":
            row = run_wave_job(args.method, args.seed, device)
        else:
            row = run_spring_heat_job(args.case, args.method, args.seed, device)
        row["torch_version"] = torch.__version__
        row["cuda_available"] = bool(torch.cuda.is_available())
        row["device_name"] = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        row["elapsed_seconds_wall"] = time.perf_counter() - started
        row["status"] = "completed"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(args.output)
        print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
    except Exception as exc:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        failure = {
            "case": args.case,
            "method": args.method,
            "seed": args.seed,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds_wall": time.perf_counter() - started,
        }
        args.output.write_text(json.dumps(failure, indent=2, ensure_ascii=False), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
