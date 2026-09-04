from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from experiments import run_figure2_corrected_formal_worker as formal
from experiments import run_modern_baseline_admission as modern
from experiments import run_figure2_reviewer_gate as reviewer
from experiments.wave_scenario_assets import WaveScenarioAssets
from paper_experiments import run_spring_heat_gate as spring_heat
import run_benchmark_v3 as wave_v3
from hilda_da.systems.one_dimensional import Heat1D, HeatConfig, SpringConfig, SpringOscillator


CASES = ("wave", "spring", "heat")
METHODS = formal.METHODS
SCALES = (0.50, 0.75, 1.00, 1.25, 1.50)
PROTOCOL = os.environ.get(
    "FIG2_STHETA_PROTOCOL",
    "figure2-stheta-sensitivity-5paired-seeds-20260811",
)


def scale_slug(scale: float) -> str:
    return f"{scale:.2f}".replace(".", "p")


def source_hash(root: Path) -> str:
    digest = hashlib.sha256()
    files = [
        "experiments/run_figure2_stheta_sensitivity_worker.py",
        "experiments/run_figure2_corrected_formal_worker.py",
        "experiments/run_figure2_reviewer_gate.py",
        "experiments/run_modern_baseline_admission.py",
        "paper_experiments/run_spring_heat_gate.py",
        "run_benchmark_v3.py",
    ]
    for relative in files:
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _scaled_wave_assets(seed: int, scale: float) -> WaveScenarioAssets:
    base = wave_v3.make_config("quick")
    cfg = replace(
        base,
        seed=seed,
        nx=41,
        ensemble_size=18,
        n_alpha=7,
        t_end=1.0,
        dt=0.0025,
        obs_interval=20,
        n_sensors=6,
        alpha_true=0.12,
        epistemic_scale=float(base.epistemic_scale) * scale,
    )
    return WaveScenarioAssets.from_legacy_scenario(wave_v3.generate_scenario(cfg))


def _scaled_make_system(config: spring_heat.CaseConfig, scale: float) -> SpringOscillator | Heat1D:
    if config.name == "spring":
        return SpringOscillator(
            SpringConfig(
                damping=0.10,
                frequency=1.0,
                cubic_stiffness=0.06,
                forcing_amplitude=0.72,
                forcing_frequency=0.75,
                stochastic_scale=0.10,
                epistemic_scale=0.42 * scale,
            )
        )
    return Heat1D(
        HeatConfig(
            nx=64,
            diffusivity=0.060,
            reaction=0.08,
            stochastic_scale=0.010,
            epistemic_scale=0.18 * scale,
        )
    )


@contextmanager
def scaled_entrypoints(scale: float) -> Iterator[None]:
    original_wave_assets = modern.make_wave_assets
    original_make_system = spring_heat.make_system

    def wave_assets(seed: int) -> WaveScenarioAssets:
        return _scaled_wave_assets(seed, scale)

    def make_system(config: spring_heat.CaseConfig) -> SpringOscillator | Heat1D:
        return _scaled_make_system(config, scale)

    modern.make_wave_assets = wave_assets  # type: ignore[method-assign]
    spring_heat.make_system = make_system  # type: ignore[method-assign]
    try:
        yield
    finally:
        modern.make_wave_assets = original_wave_assets  # type: ignore[method-assign]
        spring_heat.make_system = original_make_system  # type: ignore[method-assign]


def save_assets(case: str, seed: int, scale: float, device: torch.device, root: Path) -> dict[str, Any]:
    # The scale is part of the data-generating process and must therefore be
    # part of the asset path. A lock also prevents two methods from reading a
    # partially written metadata file for the same case/scale/seed.
    scaled_root = root / f"scale_{scale_slug(scale)}"
    destination = scaled_root / case / f"seed_{seed}"
    lock = destination / ".sensitivity_asset_lock"
    while True:
        try:
            destination.mkdir(parents=True, exist_ok=True)
            lock.mkdir()
            break
        except FileExistsError:
            time.sleep(0.05)
    try:
        with scaled_entrypoints(scale):
            metadata = formal.save_common_assets(case, seed, device, scaled_root)
    finally:
        try:
            lock.rmdir()
        except OSError:
            pass
    metadata.update(
        sensitivity_scale=float(scale),
        sensitivity_scale_slug=scale_slug(scale),
        sensitivity_protocol=PROTOCOL,
    )
    metadata_path = Path(metadata["asset_path"]).with_name("metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def run_one(
    case: str, method: str, seed: int, scale: float, device: torch.device
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    started = time.perf_counter()
    with scaled_entrypoints(scale), formal.record_traces(method) as recorder:
        row = formal.run_method(case, method, seed, device)
    row.update(
        case=case,
        method=method,
        seed=int(seed),
        sensitivity_scale=float(scale),
        sensitivity_scale_slug=scale_slug(scale),
        protocol=PROTOCOL,
        source_hash=source_hash(PROJECT_ROOT),
        sensitivity_valid=bool(row.get("valid", False)),
        elapsed_seconds_wall=float(time.perf_counter() - started),
        peak_gpu_memory_mb=(
            float(torch.cuda.max_memory_allocated(device) / (1024.0**2))
            if device.type == "cuda"
            else 0.0
        ),
        trace_blocks=len(recorder.blocks),
    )
    return row, recorder.arrays()


def parse_tasks(path: Path) -> list[tuple[str, str, int, float]]:
    tasks: list[tuple[str, str, int, float]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        case, method, seed_text, scale_text = [item.strip() for item in line.split(",")]
        if case not in CASES or method not in METHODS:
            raise ValueError(f"unsupported task: {line}")
        tasks.append((case, method, int(seed_text), float(scale_text)))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Figure 2 s_theta sensitivity worker.")
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.set_num_threads(max(1, int(os.environ.get("FIG2_TORCH_THREADS", "2"))))
    tasks = parse_tasks(args.task_file)
    completed = 0
    failed = 0
    started = time.perf_counter()

    for index, (case, method, seed, scale) in enumerate(tasks, start=1):
        slug = scale_slug(scale)
        out_path = args.output / case / f"scale_{slug}" / method / f"seed_{seed}.json"
        trace_path = args.artifact_root / "method_traces" / case / f"scale_{slug}" / method / f"seed_{seed}.npz"
        if out_path.exists() and trace_path.exists():
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
                if existing.get("status") == "completed" and existing.get("valid", False):
                    completed += 1
                    print(f"SKIP {index}/{len(tasks)} {case} {method} {scale} {seed}", flush=True)
                    continue
            except Exception:
                pass
        print(f"RUN {index}/{len(tasks)} {case} {method} scale={scale:.2f} seed={seed}", flush=True)
        job_started = time.perf_counter()
        try:
            asset_metadata = save_assets(case, seed, scale, device, args.artifact_root / "common_assets")
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            row, trace_arrays = run_one(case, method, seed, scale, device)
            formal._atomic_npz(trace_path, trace_arrays)
            row.update(
                common_asset_path=asset_metadata["asset_path"],
                common_asset_sha256=asset_metadata["arrays_sha256"],
                trace_path=str(trace_path),
                trace_sha256=formal._sha256(trace_path),
                worker_device=str(device),
                worker_elapsed_seconds=float(time.perf_counter() - job_started),
            )
            formal._atomic_json(out_path, row)
            completed += 1
            print(f"OK {case} {method} scale={scale:.2f} seed={seed} nrmse={row.get('nrmse')}", flush=True)
        except Exception as exc:
            failed += 1
            formal._atomic_json(
                out_path,
                {
                    "case": case,
                    "method": method,
                    "seed": seed,
                    "sensitivity_scale": scale,
                    "sensitivity_scale_slug": slug,
                    "protocol": PROTOCOL,
                    "status": "failed",
                    "valid": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "worker_device": str(device),
                    "worker_elapsed_seconds": float(time.perf_counter() - job_started),
                },
            )
            print(f"FAILED {case} {method} scale={scale:.2f} seed={seed}: {type(exc).__name__}: {exc}", flush=True)

    formal._atomic_json(
        args.output / "worker_status.json",
        {
            "status": "completed" if failed == 0 else "completed_with_failures",
            "tasks": len(tasks),
            "completed": completed,
            "failed": failed,
            "elapsed_seconds": float(time.perf_counter() - started),
            "device": str(device),
            "protocol": PROTOCOL,
        },
    )


if __name__ == "__main__":
    main()
