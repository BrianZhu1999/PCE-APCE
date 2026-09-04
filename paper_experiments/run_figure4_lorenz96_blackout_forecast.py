from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

core = importlib.import_module("paper_experiments.run_figure4_lorenz96_1024_scaling")


MethodName = Literal["aug_enkf", "bma_static", "pce", "apce"]
METHODS: tuple[MethodName, ...] = ("aug_enkf", "bma_static", "pce", "apce")
METHOD_LABELS = {
    "aug_enkf": "Aug-EnKF",
    "bma_static": "BMA",
    "pce": "PCE",
    "apce": "APCE",
}

BLACKOUT_START_STEP = 200
SAVED_DT = 0.01


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def padded_history(history: list[torch.Tensor]) -> np.ndarray:
    if not history:
        return np.empty((0, 0), dtype=np.float64)
    width = max(int(item.numel()) for item in history)
    output = np.full((len(history), width), np.nan, dtype=np.float64)
    for index, item in enumerate(history):
        values = item.detach().cpu().numpy().reshape(-1)
        output[index, : values.size] = values
    return output


def two_point_correlation_error(estimate: torch.Tensor, truth: torch.Tensor) -> float:
    state_dim = int(truth.numel())
    lags = torch.linspace(1, state_dim // 2, 32, dtype=torch.float64, device=truth.device).round().to(torch.int64)

    def correlation(field: torch.Tensor) -> torch.Tensor:
        centered = field - field.mean()
        denominator = centered.square().mean().clamp_min(1.0e-12)
        return torch.stack([(centered * torch.roll(centered, -int(lag), dims=0)).mean() / denominator for lag in lags])

    return float((correlation(estimate) - correlation(truth)).abs().mean())


class ForecastMetrics:
    """Forecast-only metrics after a fixed blackout step."""

    def __init__(self) -> None:
        self.blackout_start_step = int(BLACKOUT_START_STEP)
        self.saved_dt = float(SAVED_DT)
        self.step = 0
        self.steps: list[int] = []
        self.nrmse: list[float] = []
        self.crps: list[float] = []
        self.coverage: list[float] = []
        self.width: list[float] = []
        self.correlation_error: list[float] = []
        self.squared_error = 0.0
        self.truth_square = 0.0
        self.points = 0

    def add(
        self,
        ensemble: torch.Tensor,
        truth: torch.Tensor,
        weights: torch.Tensor,
        *,
        point_estimate: torch.Tensor,
        probabilistic: bool,
    ) -> None:
        step = self.step
        self.step += 1
        if step <= self.blackout_start_step:
            return
        normalized = weights.to(dtype=ensemble.dtype, device=ensemble.device).clamp_min(1.0e-300)
        normalized = normalized / normalized.sum().clamp_min(1.0e-300)
        error_sq = (point_estimate - truth).square().sum()
        truth_sq = truth.square().sum().clamp_min(1.0e-30)
        self.steps.append(int(step))
        self.nrmse.append(float(torch.sqrt(error_sq / truth_sq)))
        self.crps.append(float(core.weighted_ensemble_crps(ensemble, truth, normalized)))
        coverage, width = core.weighted_central_interval_coverage_width(ensemble, truth, normalized, level=0.90)
        self.coverage.append(float(coverage))
        self.width.append(float(width))
        self.correlation_error.append(two_point_correlation_error(point_estimate, truth))
        self.squared_error += float(error_sq)
        self.truth_square += float(truth_sq)
        self.points += int(truth.numel())

    def _skill_horizon_time(self, threshold: float) -> float:
        if not self.nrmse:
            return math.nan
        passed = np.asarray(self.nrmse, dtype=float) > float(threshold)
        if passed.any():
            lead_steps = int(np.flatnonzero(passed)[0]) + 1
        else:
            lead_steps = len(self.nrmse)
        return float(lead_steps * self.saved_dt)

    def finalize(self) -> dict[str, Any]:
        forecast_nrmse = math.sqrt(self.squared_error / max(self.truth_square, 1.0e-30))
        forecast_crps = float(np.mean(self.crps)) if self.crps else math.nan
        forecast_coverage = float(np.mean(self.coverage)) if self.coverage else math.nan
        forecast_width = float(np.mean(self.width)) if self.width else math.nan
        forecast_corr = float(np.mean(self.correlation_error)) if self.correlation_error else math.nan
        return {
            "nrmse": forecast_nrmse,
            "rmse": math.sqrt(self.squared_error / max(self.points, 1)),
            "crps": forecast_crps,
            "coverage_90": forecast_coverage,
            "interval_width_90": forecast_width,
            "forecast_nrmse": forecast_nrmse,
            "forecast_crps": forecast_crps,
            "forecast_coverage_90": forecast_coverage,
            "forecast_interval_width_90": forecast_width,
            "forecast_correlation_error": forecast_corr,
            "forecast_steps": self.steps,
            "forecast_lead_time": [float((step - self.blackout_start_step) * self.saved_dt) for step in self.steps],
            "lead_nrmse": self.nrmse,
            "lead_crps": self.crps,
            "lead_coverage_90": self.coverage,
            "lead_interval_width_90": self.width,
            "lead_correlation_error": self.correlation_error,
            "skill_horizon_time_015": self._skill_horizon_time(0.15),
            "skill_horizon_time_020": self._skill_horizon_time(0.20),
            "skill_horizon_time_030": self._skill_horizon_time(0.30),
        }


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(seed)
    return generator


def load_shared_assets(config: core.L96ScalingConfig, asset_root: Path, device: torch.device) -> core.SharedAssets:
    path = core.create_shared_assets(config, asset_root, device)
    with np.load(path, allow_pickle=False) as data:
        stored_config = json.loads(str(data["config_json"].item()))
        immutable_fields = ("state_dim", "observed_points", "steps", "dt", "ensemble_size", "alpha_true")
        for field in immutable_fields:
            if stored_config[field] != asdict(config)[field]:
                raise RuntimeError(f"Shared asset mismatch for {field}: {stored_config[field]} != {asdict(config)[field]}")
        return core.SharedAssets(
            config=config,
            truth=torch.as_tensor(data["truth"], dtype=torch.float64, device=device),
            initial_ensemble=torch.as_tensor(data["initial_ensemble"], dtype=torch.float64, device=device),
            forecast_noise=torch.as_tensor(data["forecast_noise"], dtype=torch.float64, device=device),
            observation_noise=torch.as_tensor(data["observation_noise"], dtype=torch.float64, device=device),
            observation_indices=torch.as_tensor(data["observation_indices"], dtype=torch.int64, device=device),
            asset_path=path,
            asset_sha256=file_sha256(path),
        )


def materialize_blackout_scenario(shared: core.SharedAssets, blackout_start_step: int) -> core.Scenario:
    scenario = core.materialize_scenario(shared)
    observations = {step: observation for step, observation in scenario.observations.items() if int(step) <= int(blackout_start_step)}
    return core.Scenario(
        config=scenario.config,
        truth=scenario.truth,
        observations=observations,
        initial_ensemble=scenario.initial_ensemble,
        forecast_noise=scenario.forecast_noise,
        observation_indices=scenario.observation_indices,
        localization=scenario.localization,
        augmented_localization=scenario.augmented_localization,
        asset_path=scenario.asset_path,
        asset_sha256=scenario.asset_sha256,
    )


def blackout_alpha(result: dict[str, Any], config: core.L96ScalingConfig, method: MethodName, step_index: int) -> tuple[float, float]:
    if method == "aug_enkf":
        history = np.asarray(result.get("alpha_mean_history", []), dtype=float)
        if history.size <= step_index:
            estimate = float(history[-1]) if history.size else math.nan
        else:
            estimate = float(history[step_index])
        return estimate, estimate
    if method == "bma_static":
        weights = np.asarray(result.get("alpha_weight_history", []), dtype=float)
        if weights.ndim != 2 or weights.shape[0] <= step_index:
            return math.nan, math.nan
        row = weights[step_index]
        grid = np.linspace(config.alpha_min, config.alpha_max, config.bma_alpha_grid_size, dtype=float)
        valid = np.isfinite(row) & np.isfinite(grid[: row.size])
        row = row[valid]
        grid = grid[: row.size][valid]
        if row.size == 0 or float(row.sum()) <= 0:
            return math.nan, math.nan
        row = row / row.sum()
        mean = float(np.dot(row, grid))
        return mean, float(grid[int(np.argmax(row))])
    weights = np.asarray(result.get("alpha_weight_history", []), dtype=float)
    grids = np.asarray(result.get("alpha_grid_history", []), dtype=float)
    if weights.ndim != 2 or grids.ndim != 2 or weights.shape[0] <= step_index or grids.shape[0] <= step_index:
        return math.nan, math.nan
    row_w = weights[step_index]
    row_g = grids[step_index]
    valid = np.isfinite(row_w) & np.isfinite(row_g)
    row_w = row_w[valid]
    row_g = row_g[valid]
    if row_w.size == 0 or float(row_w.sum()) <= 0:
        return math.nan, math.nan
    row_w = row_w / row_w.sum()
    mean = float(np.dot(row_w, row_g))
    return mean, float(row_g[int(np.argmax(row_w))])


def trace_path(output: Path, method: MethodName, interval: int, seed: int) -> Path:
    return output / "artifacts" / "method_traces" / "lorenz96_1024" / f"time{interval}" / method / f"seed_{seed}.npz"


def save_trace(output: Path, method: MethodName, scenario: core.Scenario, result: dict[str, Any]) -> str:
    path = trace_path(output, method, scenario.config.obs_interval, scenario.config.seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "times": np.arange(scenario.config.steps + 1, dtype=float) * scenario.config.dt,
        "observation_indices": scenario.observation_indices.detach().cpu().numpy(),
        "alpha_true": np.asarray(scenario.config.alpha_true),
        "obs_interval": np.asarray(scenario.config.obs_interval),
        "state_dim": np.asarray(scenario.config.state_dim),
        "blackout_start_step": np.asarray(int(BLACKOUT_START_STEP), dtype=np.int64),
    }
    for key in ("mean_states", "alpha_mean_history", "alpha_weight_history", "alpha_grid_history"):
        if key in result:
            payload[key] = np.asarray(result[key])
    np.savez_compressed(path, **payload)
    return str(path)


def completed_payload(
    scenario: core.Scenario,
    method: MethodName,
    result: dict[str, Any],
    output: Path,
    record_trace: bool,
) -> dict[str, Any]:
    trace = save_trace(output, method, scenario, result) if record_trace else ""
    scalars = {key: value for key, value in result.items() if isinstance(value, (str, float, int))}
    series_keys = (
        "forecast_steps",
        "forecast_lead_time",
        "lead_nrmse",
        "lead_crps",
        "lead_coverage_90",
        "lead_interval_width_90",
        "lead_correlation_error",
    )
    series = {key: clean_json(result[key]) for key in series_keys if key in result}
    return {
        "run_id": f"lorenz96_1024_t{scenario.config.obs_interval}_{method}_seed{scenario.config.seed}_blackout{BLACKOUT_START_STEP}",
        "status": "completed",
        "numerical_status": core.numerical_status(result, scenario),
        "case": "lorenz96_1024",
        "method": method,
        "label": METHOD_LABELS[method],
        "seed": scenario.config.seed,
        "state_dim": scenario.config.state_dim,
        "observed_points": scenario.config.observed_points,
        "spatial_downsampling_factor": scenario.config.state_dim // scenario.config.observed_points,
        "observation_indices": ",".join(str(int(value)) for value in scenario.observation_indices.detach().cpu().tolist()),
        "obs_interval": scenario.config.obs_interval,
        "observation_count": len(scenario.observations),
        "dt": scenario.config.dt,
        "steps": scenario.config.steps,
        "ensemble_size": scenario.config.ensemble_size,
        "alpha_true": scenario.config.alpha_true,
        "tuning_profile": scenario.config.tuning_profile,
        "blackout_start_step": int(BLACKOUT_START_STEP),
        "blackout_time": float(BLACKOUT_START_STEP) * float(scenario.config.dt),
        "assimilation_final_observation_step": int(BLACKOUT_START_STEP),
        "forecast_start_step": int(BLACKOUT_START_STEP) + 1,
        "analysis_updates_after_blackout": 0,
        "evidence_updates_after_blackout": 0,
        "regrids_after_blackout": 0,
        "asset_npz": str(scenario.asset_path),
        "asset_sha256": scenario.asset_sha256,
        "trace_npz": trace,
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        **scalars,
        **series,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def run_json_path(output: Path, method: MethodName, interval: int, seed: int) -> Path:
    return output / "artifacts" / "run_json" / "lorenz96_1024" / f"time{interval}" / method / f"seed_{seed}.json"


def parse_method(text: str) -> MethodName:
    if text not in METHODS:
        raise ValueError(f"--method must be one of {METHODS}")
    return text  # type: ignore[return-value]


def config_from_args(args: argparse.Namespace) -> core.L96ScalingConfig:
    base = core.L96ScalingConfig(
        seed=args.seed,
        state_dim=args.state_dim,
        observed_points=args.observed_points,
        obs_interval=args.obs_interval,
        steps=args.steps,
    )
    return core.apply_tuning_profile(base, args.tuning_profile)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lorenz-96 D=1024 blackout forecast worker.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--state-dim", type=int, default=1024)
    parser.add_argument("--observed-points", type=int, default=128)
    parser.add_argument("--obs-interval", type=int, default=8)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--blackout-start-step", type=int, default=200)
    parser.add_argument("--method", default="apce")
    parser.add_argument("--tuning-profile", default="baseline")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda:2" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prepare-asset-only", action="store_true")
    parser.add_argument("--no-record-trace", action="store_true")
    args = parser.parse_args()
    global BLACKOUT_START_STEP, SAVED_DT
    BLACKOUT_START_STEP = int(args.blackout_start_step)
    config = config_from_args(args)
    SAVED_DT = float(config.dt)
    device = torch.device(args.device)
    asset_root = args.asset_root or (args.output / "shared_assets")
    if args.prepare_asset_only:
        path = core.create_shared_assets(config, asset_root, device)
        print(json.dumps({"status": "asset_ready", "asset": str(path), "sha256": file_sha256(path)}, ensure_ascii=False))
        return
    method = parse_method(args.method)
    run_path = run_json_path(args.output, method, config.obs_interval, config.seed)
    try:
        shared = load_shared_assets(config, asset_root, device)
        scenario = materialize_blackout_scenario(shared, BLACKOUT_START_STEP)
        original_metrics = core.RunningMetrics
        core.RunningMetrics = ForecastMetrics  # type: ignore[assignment]
        try:
            result = core.run_method(scenario, method, record_trace=not args.no_record_trace)
        finally:
            core.RunningMetrics = original_metrics  # type: ignore[assignment]
        alpha_estimate, alpha_map = blackout_alpha(result, config, method, BLACKOUT_START_STEP)
        result.update(
            blackout_start_step=BLACKOUT_START_STEP,
            blackout_alpha_estimate=alpha_estimate,
            blackout_alpha_map=alpha_map,
            blackout_alpha_absolute_error=abs(alpha_estimate - config.alpha_true) if math.isfinite(alpha_estimate) else math.nan,
            alpha_estimate=alpha_estimate,
            alpha_map=alpha_map,
            alpha_absolute_error=abs(alpha_estimate - config.alpha_true) if math.isfinite(alpha_estimate) else math.nan,
        )
        payload = completed_payload(scenario, method, result, args.output, not args.no_record_trace)
        write_json(run_path, payload)
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "numerical_status": payload["numerical_status"],
                    "method": method,
                    "seed": config.seed,
                    "obs_interval": config.obs_interval,
                    "blackout_start_step": BLACKOUT_START_STEP,
                    "nrmse": payload["nrmse"],
                    "crps": payload["crps"],
                    "alpha_mae": payload["blackout_alpha_absolute_error"],
                    "skill_horizon_time_020": payload.get("skill_horizon_time_020"),
                },
                ensure_ascii=False,
            )
        )
    except Exception as error:  # noqa: BLE001
        payload = {
            "run_id": f"lorenz96_1024_t{config.obs_interval}_{method}_seed{config.seed}_blackout{BLACKOUT_START_STEP}",
            "status": "failed",
            "case": "lorenz96_1024",
            "method": method,
            "seed": config.seed,
            "obs_interval": config.obs_interval,
            "blackout_start_step": BLACKOUT_START_STEP,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": file_sha256(Path(__file__).resolve()),
        }
        write_json(run_path, payload)
        raise


if __name__ == "__main__":
    main()
