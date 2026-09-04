from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hilda_da.baselines import denkf_analysis, letkf_analysis
from hilda_da.observations import SparseObservation
from paper_experiments import run_lorenz_ks_gate as gate


METHODS: tuple[str, ...] = (
    "misspecified_forecast",
    "denkf",
    "letkf",
    "aug_enkf",
    "bma_static",
    "oracle_alpha",
    "pce",
    "apce",
)

METHOD_LABELS: dict[str, str] = {
    "misspecified_forecast": "Misspecified forecast",
    "denkf": "DEnKF",
    "letkf": "LETKF",
    "aug_enkf": "Aug-EnKF",
    "bma_static": "BMA",
    "oracle_alpha": "Oracle-alpha",
    "pce": "PCE",
    "apce": "APCE",
}


def ks1024_config(seed: int) -> gate.CaseConfig:
    return gate.CaseConfig(
        name="ks",
        seed=int(seed),
        steps=260,
        dt=0.050,
        obs_interval=5,
        obs_stride=0,
        ensemble_size=28,
        obs_noise=0.23,
        state_dim=1024,
        alpha_true=0.12,
        fixed_alpha=0.50,
        alpha_grid=(0.08, 0.22, 0.36, 0.50, 0.64, 0.78, 0.92),
        pce_temperature=0.42,
        apce_temperature=0.55,
        apce_min_temperature=0.14,
        apce_forgetting=0.985,
        apce_entropy_floor=0.74,
        evidence_shrinkage=0.18,
    )


def sparse128_observation_indices(config: gate.CaseConfig, device: torch.device) -> torch.Tensor:
    if config.name != "ks":
        raise ValueError("This worker is KS-only.")
    if config.state_dim != 1024:
        raise ValueError(f"Expected KS state_dim=1024, got {config.state_dim}.")
    stride = config.state_dim // 128
    indices = torch.arange(0, config.state_dim, stride, dtype=torch.int64, device=device)
    if int(indices.numel()) != 128:
        raise RuntimeError(f"Expected exactly 128 observed points, got {indices.numel()}.")
    return indices


def install_ks1024_patch() -> None:
    # The legacy Lorenz/KS gate is kept as the numerical reference; this patch
    # only changes the KS spatial observation geometry for the new Figure 4
    # admission test.
    gate.observation_indices = sparse128_observation_indices  # type: ignore[assignment]


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{k: clean_json(v) for k, v in row.items()} for row in rows])


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for method in METHODS:
        subset = [row for row in rows if row.get("method") == method and row.get("valid")]
        item: dict[str, Any] = {
            "case": "ks1024_sparse128",
            "method": method,
            "label": METHOD_LABELS[method],
            "n_valid": len(subset),
            "n_total": len([row for row in rows if row.get("method") == method]),
        }
        for key in (
            "nrmse",
            "rmse",
            "crps",
            "coverage_90",
            "interval_width_90",
            "alpha_absolute_error",
            "runtime_seconds",
            "peak_gpu_memory_mb",
            "forward_member_steps",
        ):
            values = np.asarray([float(row[key]) for row in subset if row.get(key) not in (None, "")], dtype=float)
            item[key] = float(np.mean(values)) if values.size else math.nan
            item[f"{key}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        summary.append(item)
    return summary


def save_common_assets(output: Path, scenario: gate.Scenario) -> Path:
    asset_dir = output / "artifacts" / "common_assets" / f"seed_{scenario.config.seed}"
    asset_dir.mkdir(parents=True, exist_ok=True)
    obs_steps = np.asarray(sorted(scenario.observations), dtype=np.int64)
    obs_values = np.stack([scenario.observations[int(step)].detach().cpu().numpy() for step in obs_steps])
    path = asset_dir / "ks1024_sparse128_common_assets.npz"
    np.savez_compressed(
        path,
        times=scenario.times.detach().cpu().numpy(),
        coordinates=scenario.coordinates.detach().cpu().numpy(),
        observation_indices=scenario.observation_indices.detach().cpu().numpy(),
        observation_steps=obs_steps,
        observations=obs_values,
        truth_states=scenario.truth.detach().cpu().numpy(),
        initial_ensemble=scenario.initial_ensemble.detach().cpu().numpy(),
        forecast_noise=scenario.forecast_noise.detach().cpu().numpy(),
        alpha_grid=scenario.alpha_grid.detach().cpu().numpy(),
    )
    return path


def run_aug_enkf_method(
    scenario: gate.Scenario,
    device: torch.device,
    *,
    record_trace: bool = False,
) -> dict[str, Any]:
    config = scenario.config
    system = gate.make_system(config, device, torch.float64)
    ensemble = scenario.initial_ensemble.clone()
    n_members = int(config.ensemble_size)
    alpha_ensemble = torch.linspace(0.04, 0.96, n_members, dtype=ensemble.dtype, device=device)
    operator = SparseObservation(scenario.observation_indices)
    covariance = config.obs_noise**2 * torch.eye(
        scenario.observation_indices.numel(),
        dtype=ensemble.dtype,
        device=device,
    )
    weights = torch.full((n_members,), 1.0 / n_members, dtype=ensemble.dtype, device=device)
    metrics = gate.TrajectoryMetrics()
    mean_states: list[torch.Tensor] = []
    alpha_history: list[torch.Tensor] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(config.steps + 1):
        if record_trace:
            mean_states.append(ensemble.mean(dim=0).detach().cpu())
            alpha_history.append(alpha_ensemble.detach().cpu())
        metrics.add(ensemble, scenario.truth[step], weights)
        if step == config.steps:
            break
        propagated = []
        for member in range(n_members):
            propagated.append(
                gate.step_with_noise(
                    system,
                    ensemble[member : member + 1],
                    config.dt,
                    float(alpha_ensemble[member]),
                    scenario.forecast_noise[step, member : member + 1],
                ).squeeze(0)
            )
        ensemble = torch.stack(propagated, dim=0)
        if step + 1 not in scenario.observations:
            continue
        augmented = torch.cat([ensemble, alpha_ensemble[:, None]], dim=1)
        augmented = denkf_analysis(augmented, scenario.observations[step + 1], operator, covariance)
        ensemble = system.project(augmented[:, : config.state_dim])
        alpha_ensemble = augmented[:, config.state_dim].clamp(0.02, 0.98)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    alpha_estimate = float(alpha_ensemble.mean())
    result = metrics.finalize()
    result.update(
        runtime_seconds=float(time.perf_counter() - started),
        forward_member_steps=config.steps * n_members,
        peak_gpu_memory_mb=(
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else 0.0
        ),
        alpha_estimate=alpha_estimate,
        alpha_absolute_error=abs(alpha_estimate - config.alpha_true),
        alpha_final_std=float(alpha_ensemble.std(unbiased=True)),
    )
    if record_trace:
        result["mean_states"] = torch.stack(mean_states).numpy()
        result["alpha_member_history"] = torch.stack(alpha_history).numpy()
    return result


def run_bma_static_method(
    scenario: gate.Scenario,
    device: torch.device,
    *,
    record_trace: bool = False,
) -> dict[str, Any]:
    config = scenario.config
    system = gate.make_system(config, device, torch.float64)
    path_count = int(scenario.alpha_grid.numel())
    branches = scenario.initial_ensemble.unsqueeze(0).repeat(path_count, 1, 1)
    log_weights = torch.zeros(path_count, dtype=branches.dtype, device=device)
    weights = torch.softmax(log_weights, dim=0)
    operator = SparseObservation(scenario.observation_indices)
    covariance = config.obs_noise**2 * torch.eye(
        scenario.observation_indices.numel(),
        dtype=branches.dtype,
        device=device,
    )
    metrics = gate.TrajectoryMetrics()
    mean_states: list[torch.Tensor] = []
    alpha_weights: list[torch.Tensor] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(config.steps + 1):
        flat = branches.reshape(-1, branches.shape[-1])
        flat_weights = weights.unsqueeze(1).expand(-1, config.ensemble_size).reshape(-1) / config.ensemble_size
        if record_trace:
            branch_means = branches.mean(dim=1)
            mean_states.append((weights.unsqueeze(-1) * branch_means).sum(dim=0).detach().cpu())
            alpha_weights.append(weights.detach().cpu())
        metrics.add(flat, scenario.truth[step], flat_weights)
        if step == config.steps:
            break
        for path_index, alpha in enumerate(scenario.alpha_grid):
            branches[path_index] = gate.step_with_noise(
                system,
                branches[path_index],
                config.dt,
                float(alpha),
                scenario.forecast_noise[step],
            )
        if step + 1 not in scenario.observations:
            continue
        observation = scenario.observations[step + 1]
        branch_observations = torch.stack([operator(branch) for branch in branches])
        evidence = torch.stack(
            [
                gate.evidence_score(
                    branch_observations[path_index],
                    observation,
                    config.obs_noise,
                    config.evidence_shrinkage,
                    None,
                )
                for path_index in range(path_count)
            ]
        )
        log_weights = log_weights + evidence - evidence.mean()
        weights = torch.softmax(log_weights, dim=0)
        for path_index in range(path_count):
            branches[path_index] = denkf_analysis(branches[path_index], observation, operator, covariance)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    alpha_estimate = float((scenario.alpha_grid * weights).sum())
    result = metrics.finalize()
    result.update(
        runtime_seconds=float(time.perf_counter() - started),
        forward_member_steps=config.steps * path_count * config.ensemble_size,
        peak_gpu_memory_mb=(
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else 0.0
        ),
        alpha_estimate=alpha_estimate,
        alpha_absolute_error=abs(alpha_estimate - config.alpha_true),
        alpha_final_entropy=float(gate.entropy(weights)),
    )
    if record_trace:
        result["mean_states"] = torch.stack(mean_states).numpy()
        result["alpha_weight_history"] = torch.stack(alpha_weights).numpy()
    return result


def run_method(
    scenario: gate.Scenario,
    method: str,
    device: torch.device,
    *,
    record_trace: bool = False,
) -> dict[str, Any]:
    if method in {"misspecified_forecast", "denkf", "letkf", "oracle_alpha", "pce", "apce"}:
        return gate.run_method(scenario, method, device, record_trace=record_trace)  # type: ignore[arg-type]
    if method == "aug_enkf":
        return run_aug_enkf_method(scenario, device, record_trace=record_trace)
    if method == "bma_static":
        return run_bma_static_method(scenario, device, record_trace=record_trace)
    raise ValueError(method)


def save_method_trace(output: Path, seed: int, method: str, result: dict[str, Any]) -> str | None:
    arrays: dict[str, np.ndarray] = {}
    for key in ("mean_states", "alpha_weight_history", "alpha_member_history"):
        if key in result:
            arrays[key] = np.asarray(result.pop(key))
    if not arrays:
        return None
    trace_dir = output / "artifacts" / "method_traces" / "ks1024_sparse128" / method
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"seed_{seed}.npz"
    np.savez_compressed(path, **arrays)
    return str(path)


def run_seed(
    seed: int,
    methods: tuple[str, ...],
    output: Path,
    device: torch.device,
    *,
    record_trace: bool,
) -> list[dict[str, Any]]:
    install_ks1024_patch()
    config = ks1024_config(seed)
    scenario = gate.generate_scenario(config, device)
    common_asset_path = save_common_assets(output, scenario)
    common_asset_hash = file_sha256(common_asset_path)
    rows: list[dict[str, Any]] = []
    run_json_dir = output / "artifacts" / "run_json" / "ks1024_sparse128"
    run_json_dir.mkdir(parents=True, exist_ok=True)
    for method in methods:
        started = time.perf_counter()
        row: dict[str, Any] = {
            "case": "ks1024_sparse128",
            "seed": int(seed),
            "method": method,
            "label": METHOD_LABELS[method],
            "state_dim": int(config.state_dim),
            "observed_points": int(scenario.observation_indices.numel()),
            "spatial_observation_stride": 8,
            "obs_interval": int(config.obs_interval),
            "steps": int(config.steps),
            "dt": float(config.dt),
            "alpha_true": float(config.alpha_true),
            "common_asset_path": str(common_asset_path),
            "common_asset_sha256": common_asset_hash,
            "status": "started",
            "valid": False,
            "device": str(device),
        }
        try:
            result = run_method(scenario, method, device, record_trace=record_trace)
            trace_path = save_method_trace(output, seed, method, result)
            row.update(result)
            row.update(
                status="completed",
                valid=bool(np.isfinite(float(result["nrmse"]))),
                method_trace_path=trace_path or "",
                wall_seconds_total=float(time.perf_counter() - started),
            )
        except Exception as exc:
            row.update(
                status="failed",
                valid=False,
                failure_type=type(exc).__name__,
                failure_message=str(exc),
                wall_seconds_total=float(time.perf_counter() - started),
            )
        run_json_path = run_json_dir / f"{method}_seed_{seed}.json"
        run_json_path.write_text(json.dumps(clean_json(row), ensure_ascii=False, indent=2), encoding="utf-8")
        row["run_json_path"] = str(run_json_path)
        rows.append(row)
        print(
            f"KS1024 seed={seed} method={method} status={row['status']} "
            f"nrmse={row.get('nrmse', '')} alpha_err={row.get('alpha_absolute_error', '')}",
            flush=True,
        )
    return rows


def parse_seed_file(path: Path) -> list[int]:
    seeds: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        seeds.append(int(text.split(",")[0]))
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description="Figure 4 KS 1D 1024-grid / 128-sensor sparse admission worker.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seed-file", type=Path, default=None)
    parser.add_argument("--base-seed", type=int, default=2026081400)
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--record-trace", action="store_true")
    args = parser.parse_args()

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if args.seed_file is not None:
        seeds = parse_seed_file(args.seed_file)
    elif args.seed is not None:
        seeds = [int(args.seed)]
    else:
        seeds = [int(args.base_seed) + i for i in range(int(args.n_seeds))]

    script_path = Path(__file__).resolve()
    legacy_path = Path(gate.__file__).resolve()
    all_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for seed in seeds:
        all_rows.extend(
            run_seed(
                int(seed),
                tuple(args.methods),
                output,
                device,
                record_trace=bool(args.record_trace),
            )
        )
    worker_dir = output / "workers"
    worker_dir.mkdir(parents=True, exist_ok=True)
    worker_tag = f"worker_{os.getpid()}_{int(time.time() * 1000)}_{str(device).replace(':', '')}"
    write_csv(worker_dir / f"{worker_tag}_run_source_data.csv", all_rows)
    write_csv(worker_dir / f"{worker_tag}_summary.csv", summarize(all_rows))
    manifest = {
        "protocol": "figure4-ks1024-sparse128-admission",
        "case": "ks1024_sparse128",
        "state_dim": 1024,
        "observed_points": 128,
        "spatial_observation_stride": 8,
        "temporal_obs_interval": 5,
        "methods": list(args.methods),
        "seeds": seeds,
        "record_trace": bool(args.record_trace),
        "device": str(device),
        "output": str(output),
        "script_path": str(script_path),
        "script_sha256": file_sha256(script_path),
        "legacy_ks_gate_path": str(legacy_path),
        "legacy_ks_gate_sha256": file_sha256(legacy_path),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "elapsed_seconds": float(time.perf_counter() - started),
        "config": asdict(ks1024_config(seeds[0])),
    }
    (worker_dir / f"{worker_tag}_manifest.json").write_text(
        json.dumps(clean_json(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
