from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from . import comparison_methods as baselines
from . import wave_evaluation as wave_methods
from . import classical_protocol as protocol
from . import spring_heat


PROTOCOL = "classical-systems-published-protocol"
METHODS = ("denkf", "letkf", "iensf", "aug_enkf", "bma_static", "pce", "apce")
LABELS = {
    "denkf": "DEnKF",
    "letkf": "LETKF",
    "iensf": "IEnSF",
    "aug_enkf": "Aug-EnKF",
    "bma_static": "BMA",
    "pce": "PCE",
    "apce": "APCE",
}
def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _as_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def save_common_assets(case: str, seed: int, device: torch.device, asset_root: Path) -> dict[str, Any]:
    destination = asset_root / case / f"seed_{seed}"
    arrays_path = destination / "arrays.npz"
    metadata_path = destination / "metadata.json"
    lock_dir = destination / ".asset_write_lock"

    while True:
        if arrays_path.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["asset_path"] = str(arrays_path)
            return metadata
        try:
            destination.mkdir(parents=True, exist_ok=True)
            lock_dir.mkdir()
            break
        except FileExistsError:
            time.sleep(0.05)

    try:
        if arrays_path.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["asset_path"] = str(arrays_path)
            return metadata

        if case == "wave":
            assets = baselines.make_wave_assets(seed)
            arrays = {
                "times": assets.times,
                "truth_states": assets.truth_states,
                "observations": assets.observations,
                "observation_mask": assets.observation_mask.astype(np.uint8),
                "observation_indices": assets.observation_indices,
                "initial_ensemble": assets.initial_ensemble,
                "forecast_noise": assets.forecast_noise,
                "truth_noise": assets.truth_noise,
                "observation_noise": assets.observation_noise,
            }
            metadata = {
                "case": case,
                "seed": seed,
                "alpha_true": float(assets.alpha_true),
                "ensemble_size": int(assets.ensemble_size),
                "steps": int(assets.n_steps),
                "analysis_rng_seed_rules": {
                    "comparison_baselines": int(seed + 910_000),
                    "wave_pce_family": "seed + stable_offset(method)",
                },
            }
        else:
            scenario = spring_heat.generate_scenario(spring_heat.config_for_case(case, seed), device)
            observation_steps = np.asarray(sorted(scenario.observations), dtype=np.int64)
            observation_values = np.stack(
                [_as_numpy(scenario.observations[int(step)]) for step in observation_steps],
                axis=0,
            )
            arrays = {
                "truth_states": _as_numpy(scenario.truth),
                "observation_steps": observation_steps,
                "observations": observation_values,
                "observation_indices": _as_numpy(scenario.observation_indices),
                "initial_ensemble": _as_numpy(scenario.initial_ensemble),
                "forecast_noise": _as_numpy(scenario.forecast_noise),
                "alpha_grid": _as_numpy(scenario.alpha_grid),
                "primary_indices": _as_numpy(scenario.primary_indices),
            }
            metadata = {
                "case": case,
                "seed": seed,
                "config": asdict(scenario.config),
                "analysis_rng_seed_rules": {
                    "comparison_baselines": int(seed + 910_000),
                    "pce_family": int(seed + 20_000),
                },
            }

        _atomic_npz(arrays_path, {key: np.ascontiguousarray(value) for key, value in arrays.items()})
        metadata.update(
            protocol=PROTOCOL,
            asset_path=str(arrays_path),
        )
        _atomic_json(metadata_path, metadata)
    finally:
        try:
            lock_dir.rmdir()
        except OSError:
            pass
    return metadata


class TraceRecorder:
    def __init__(self) -> None:
        self.blocks: list[dict[str, Any]] = []
        self._by_instance: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.alpha_analysis_history: list[np.ndarray] = []

    def _block(
        self,
        source: str,
        instance: object,
        shape_key: tuple[Any, ...],
    ) -> dict[str, Any]:
        key = (source, id(instance), shape_key)
        if key not in self._by_instance:
            block: dict[str, Any] = {
                "source": source,
                "shape_key": repr(shape_key),
                "mean": [],
                "std": [],
                "truth": [],
                "member_weights": [],
            }
            self._by_instance[key] = block
            self.blocks.append(block)
        return self._by_instance[key]

    def add_unweighted(
        self,
        source: str,
        instance: object,
        ensemble: torch.Tensor,
        truth: torch.Tensor,
    ) -> None:
        shape_key = (tuple(ensemble.shape), tuple(truth.shape))
        block = self._block(source, instance, shape_key)
        values = ensemble.detach()
        block["mean"].append(values.mean(dim=0).cpu().numpy())
        block["std"].append(values.std(dim=0, unbiased=False).cpu().numpy())
        block["truth"].append(truth.detach().cpu().numpy())

    def add_weighted(
        self,
        source: str,
        instance: object,
        ensemble: torch.Tensor,
        truth: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        shape_key = (tuple(ensemble.shape), tuple(truth.shape), tuple(weights.shape))
        block = self._block(source, instance, shape_key)
        values = ensemble.detach()
        normalized = weights.detach() / weights.detach().sum().clamp_min(1.0e-300)
        mean = (normalized.unsqueeze(-1) * values).sum(dim=0)
        variance = (normalized.unsqueeze(-1) * (values - mean).square()).sum(dim=0)
        block["mean"].append(mean.cpu().numpy())
        block["std"].append(variance.clamp_min(0.0).sqrt().cpu().numpy())
        block["truth"].append(truth.detach().cpu().numpy())
        block["member_weights"].append(normalized.cpu().numpy())

    def arrays(self) -> dict[str, np.ndarray]:
        output: dict[str, np.ndarray] = {}
        for index, block in enumerate(self.blocks):
            prefix = f"block_{index:02d}"
            output[f"{prefix}_mean"] = np.stack(block["mean"], axis=0)
            output[f"{prefix}_std"] = np.stack(block["std"], axis=0)
            output[f"{prefix}_truth"] = np.stack(block["truth"], axis=0)
            if block["member_weights"]:
                output[f"{prefix}_member_weights"] = np.stack(block["member_weights"], axis=0)
            output[f"{prefix}_source"] = np.asarray(block["source"])
            output[f"{prefix}_shape_key"] = np.asarray(block["shape_key"])
        if self.alpha_analysis_history:
            output["augmented_alpha_analysis_history"] = np.stack(self.alpha_analysis_history, axis=0)
        return output


@contextmanager
def record_traces(method: str) -> Iterator[TraceRecorder]:
    recorder = TraceRecorder()
    original_baseline_add = baselines.Metrics.add
    original_spring_add = spring_heat.TrajectoryMetrics.add
    original_wave_add = wave_methods.MetricAccumulator.add
    original_protocol_denkf = protocol.denkf_analysis

    def baseline_add(instance: object, ensemble: torch.Tensor, truth: torch.Tensor) -> None:
        recorder.add_unweighted("baselines.Metrics", instance, ensemble, truth)
        original_baseline_add(instance, ensemble, truth)

    def spring_add(
        instance: object,
        ensemble: torch.Tensor,
        truth: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        recorder.add_weighted("spring_heat.TrajectoryMetrics", instance, ensemble, truth, weights)
        original_spring_add(instance, ensemble, truth, weights)

    def wave_add(
        instance: object,
        ensemble: torch.Tensor,
        truth: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        recorder.add_weighted("wave_methods.MetricAccumulator", instance, ensemble, truth, weights)
        original_wave_add(instance, ensemble, truth, weights)

    def protocol_denkf(
        ensemble: torch.Tensor,
        observation: torch.Tensor,
        operator: Any,
        covariance: torch.Tensor,
    ) -> torch.Tensor:
        analysed = original_protocol_denkf(ensemble, observation, operator, covariance)
        if method == "aug_enkf":
            recorder.alpha_analysis_history.append(analysed[..., -1].detach().cpu().numpy())
        return analysed

    baselines.Metrics.add = baseline_add  # type: ignore[method-assign]
    spring_heat.TrajectoryMetrics.add = spring_add  # type: ignore[method-assign]
    wave_methods.MetricAccumulator.add = wave_add  # type: ignore[method-assign]
    protocol.denkf_analysis = protocol_denkf
    try:
        yield recorder
    finally:
        baselines.Metrics.add = original_baseline_add  # type: ignore[method-assign]
        spring_heat.TrajectoryMetrics.add = original_spring_add  # type: ignore[method-assign]
        wave_methods.MetricAccumulator.add = original_wave_add  # type: ignore[method-assign]
        protocol.denkf_analysis = original_protocol_denkf


def run_method(case: str, method: str, seed: int, device: torch.device) -> dict[str, Any]:
    started = time.perf_counter()
    if method in {"denkf", "letkf", "iensf"}:
        if case == "wave":
            result = baselines.run_wave(method, seed, device)
        else:
            result = baselines.run_spring_heat(case, method, seed, device)
        result.update(
            case=case,
            method=method,
            label=LABELS[method],
            seed=seed,
            protocol=PROTOCOL,
            alpha_estimate=None,
            alpha_absolute_error=None,
            status="completed",
        )
    else:
        result = protocol.run_case_method_seed(
            case,
            method,
            seed,
            device,
            protocol=PROTOCOL,
        )
        result.update(method=method, label=LABELS[method])

    result.update(
        elapsed_seconds_wall=float(time.perf_counter() - started),
    )
    result["valid"] = bool(result.get("valid", protocol.finite_metrics(result)))
    return result


def parse_tasks(path: Path) -> list[tuple[str, str, int]]:
    tasks: list[tuple[str, str, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        case, method, seed_text = [item.strip() for item in line.split(",")]
        if method not in METHODS:
            raise ValueError(f"unsupported method in task file: {method}")
        tasks.append((case, method, int(seed_text)))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one published classical-system benchmark task.")
    parser.add_argument("--task-file", type=Path)
    parser.add_argument("--case", choices=("wave", "spring", "heat"))
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.set_num_threads(max(1, int(os.environ.get("PCE_TORCH_THREADS", "2"))))
    if args.task_file is not None:
        if any(value is not None for value in (args.case, args.method, args.seed)):
            parser.error("--task-file cannot be combined with --case, --method, or --seed")
        tasks = parse_tasks(args.task_file)
    else:
        if args.case is None or args.method is None or args.seed is None:
            parser.error("provide --task-file or all of --case, --method, and --seed")
        tasks = [(args.case, args.method, args.seed)]
    artifact_root = args.artifact_root or (args.output / "artifacts")
    completed = 0
    failed = 0
    started = time.perf_counter()

    for index, (case, method, seed) in enumerate(tasks, start=1):
        out_path = args.output / case / method / f"seed_{seed}.json"
        trace_path = artifact_root / "method_traces" / case / method / f"seed_{seed}.npz"
        if out_path.exists() and trace_path.exists():
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
                if existing.get("status") == "completed" and existing.get("valid", False):
                    completed += 1
                    print(f"SKIP {index}/{len(tasks)} {case} {method} {seed}", flush=True)
                    continue
            except Exception:
                pass

        print(f"RUN {index}/{len(tasks)} {case} {method} {seed}", flush=True)
        job_started = time.perf_counter()
        try:
            asset_metadata = save_common_assets(
                case,
                seed,
                device,
                artifact_root / "common_assets",
            )
            with record_traces(method) as recorder:
                row = run_method(case, method, seed, device)
            trace_arrays = recorder.arrays()
            _atomic_npz(trace_path, trace_arrays)
            row.update(
                common_asset_path=asset_metadata["asset_path"],
                trace_path=str(trace_path),
                trace_blocks=len(recorder.blocks),
                device=str(device),
                elapsed_seconds=float(time.perf_counter() - job_started),
            )
            _atomic_json(out_path, row)
            completed += 1
            print(
                f"OK {case} {method} {seed} nrmse={row.get('nrmse')} "
                f"crps={row.get('crps')} trace_blocks={len(recorder.blocks)}",
                flush=True,
            )
        except Exception as exc:
            failed += 1
            failure = {
                "case": case,
                "method": method,
                "label": LABELS.get(method, method),
                "seed": seed,
                "protocol": PROTOCOL,
                "status": "failed",
                "valid": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "device": str(device),
                "elapsed_seconds": float(time.perf_counter() - job_started),
            }
            _atomic_json(out_path, failure)
            print(f"FAILED {case} {method} {seed}: {type(exc).__name__}: {exc}", flush=True)

    status = {
        "status": "completed" if failed == 0 else "completed_with_failures",
        "tasks": len(tasks),
        "completed": completed,
        "failed": failed,
        "elapsed_seconds": float(time.perf_counter() - started),
        "device": str(device),
        "protocol": PROTOCOL,
    }
    _atomic_json(args.output / "run_summary.json", status)
    print(json.dumps(status, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
