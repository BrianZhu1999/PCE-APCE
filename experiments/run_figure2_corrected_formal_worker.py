from __future__ import annotations

import argparse
import hashlib
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

from experiments import run_modern_baseline_admission as modern
from experiments import run_wave_repair_validation as wave_repair
from experiments import run_figure2_reviewer_gate as reviewer
from paper_experiments import run_spring_heat_gate as spring_heat


PROTOCOL = "figure2-corrected-dimension-score-50paired-seeds-20260811"
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
REVIEWER_METHODS = {
    "aug_enkf": "aug_enkf",
    "bma_static": "bma_static",
    "pce": "pce_refined_v2",
    "apce": "apce_refined_v2",
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            assets = modern.make_wave_assets(seed)
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
                "asset_digest": assets.array_digest,
                "analysis_rng_seed_rules": {
                    "modern_baselines": int(seed + 910_000),
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
                    "modern_baselines": int(seed + 910_000),
                    "pce_family": int(seed + 20_000),
                },
            }

        _atomic_npz(arrays_path, {key: np.ascontiguousarray(value) for key, value in arrays.items()})
        metadata.update(
            protocol=PROTOCOL,
            arrays_sha256=_sha256(arrays_path),
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
    original_modern_add = modern.Metrics.add
    original_spring_add = spring_heat.TrajectoryMetrics.add
    original_wave_add = wave_repair.MetricAccumulator.add
    original_reviewer_denkf = reviewer.denkf_analysis

    def modern_add(instance: object, ensemble: torch.Tensor, truth: torch.Tensor) -> None:
        recorder.add_unweighted("modern.Metrics", instance, ensemble, truth)
        original_modern_add(instance, ensemble, truth)

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
        recorder.add_weighted("wave_repair.MetricAccumulator", instance, ensemble, truth, weights)
        original_wave_add(instance, ensemble, truth, weights)

    def reviewer_denkf(
        ensemble: torch.Tensor,
        observation: torch.Tensor,
        operator: Any,
        covariance: torch.Tensor,
    ) -> torch.Tensor:
        analysed = original_reviewer_denkf(ensemble, observation, operator, covariance)
        if method == "aug_enkf":
            recorder.alpha_analysis_history.append(analysed[..., -1].detach().cpu().numpy())
        return analysed

    modern.Metrics.add = modern_add  # type: ignore[method-assign]
    spring_heat.TrajectoryMetrics.add = spring_add  # type: ignore[method-assign]
    wave_repair.MetricAccumulator.add = wave_add  # type: ignore[method-assign]
    reviewer.denkf_analysis = reviewer_denkf
    try:
        yield recorder
    finally:
        modern.Metrics.add = original_modern_add  # type: ignore[method-assign]
        spring_heat.TrajectoryMetrics.add = original_spring_add  # type: ignore[method-assign]
        wave_repair.MetricAccumulator.add = original_wave_add  # type: ignore[method-assign]
        reviewer.denkf_analysis = original_reviewer_denkf


def run_method(case: str, method: str, seed: int, device: torch.device) -> dict[str, Any]:
    started = time.perf_counter()
    if method in {"denkf", "letkf", "iensf"}:
        if case == "wave":
            result = modern.run_wave(method, seed, device)
        else:
            result = modern.run_spring_heat(case, method, seed, device)
        result.update(
            case=case,
            method=method,
            label=LABELS[method],
            seed=seed,
            protocol=PROTOCOL,
            source_hash=reviewer.source_hash(reviewer.PROJECT_ROOT),
            alpha_estimate=None,
            alpha_absolute_error=None,
            status="completed",
        )
    else:
        implementation_method = REVIEWER_METHODS[method]
        result = reviewer.run_case_method_seed(
            case,
            implementation_method,
            seed,
            device,
            protocol=PROTOCOL,
        )
        result.update(
            method=method,
            label=LABELS[method],
            implementation_method=implementation_method,
        )

    result.update(
        torch_version=torch.__version__,
        cuda_available=bool(torch.cuda.is_available()),
        device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        elapsed_seconds_wall=float(time.perf_counter() - started),
        worker_pid=os.getpid(),
    )
    result["valid"] = bool(result.get("valid", reviewer.finite_metrics(result)))
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
    parser = argparse.ArgumentParser(description="Corrected-score Figure 2 formal worker.")
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--protocol", default=PROTOCOL)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.set_num_threads(max(1, int(os.environ.get("FIG2_TORCH_THREADS", "2"))))
    tasks = parse_tasks(args.task_file)
    completed = 0
    failed = 0
    started = time.perf_counter()

    for index, (case, method, seed) in enumerate(tasks, start=1):
        out_path = args.output / case / method / f"seed_{seed}.json"
        trace_path = args.artifact_root / "method_traces" / case / method / f"seed_{seed}.npz"
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
                args.artifact_root / "common_assets",
            )
            with record_traces(method) as recorder:
                row = run_method(case, method, seed, device)
            trace_arrays = recorder.arrays()
            _atomic_npz(trace_path, trace_arrays)
            row.update(
                common_asset_path=asset_metadata["asset_path"],
                common_asset_sha256=asset_metadata["arrays_sha256"],
                trace_path=str(trace_path),
                trace_sha256=_sha256(trace_path),
                trace_blocks=len(recorder.blocks),
                worker_device=str(device),
                worker_elapsed_seconds=float(time.perf_counter() - job_started),
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
                "protocol": args.protocol,
                "status": "failed",
                "valid": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "worker_device": str(device),
                "worker_elapsed_seconds": float(time.perf_counter() - job_started),
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
        "protocol": args.protocol,
        "pid": os.getpid(),
    }
    _atomic_json(args.output / "worker_status.json", status)
    print(json.dumps(status, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
