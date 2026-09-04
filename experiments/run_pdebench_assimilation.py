from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional

from hilda_da.alpha import AlphaEvidenceTracker, liu_quantile
from hilda_da.baselines import denkf_analysis, letkf_analysis
from hilda_da.config import HILDAConfig
from hilda_da.filter import HILDAFilter
from hilda_da.metrics import (
    observation_rmse,
    state_rmse,
    weighted_central_interval_coverage_width,
    weighted_ensemble_crps,
    weighted_multivariate_energy_score,
)
from hilda_da.pdebench import (
    PDEBenchHDF5Adapter,
    PDEBenchTrajectory,
    TrajectorySlice,
    verify_manifest_checksum,
)
from hilda_da.strong_baselines import (
    EnFFF2PConfig,
    EnSFConfig,
    IEnSFConfig,
    enff_f2p_analysis,
    ensf_analysis,
    ensf_lr_ridge_analysis,
    iensf_analysis,
)


METHODS = ("hilda", "denkf", "letkf", "ensf", "iensf", "ensf_lr_ridge", "enff_f2p")
EXPECTED_SOURCE_GRID = (512, 512)
EXPECTED_CHANNELS = ("velocity_x", "velocity_y")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assimilate an official PDEBench NS_Incom trajectory without training"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--collection-validation", type=Path, default=None)
    parser.add_argument("--verify-manifest-md5", action="store_true")
    parser.add_argument("--trajectory-index", type=int, default=0)
    parser.add_argument("--time-start", type=int, default=0)
    parser.add_argument("--time-stop", type=int, default=None)
    parser.add_argument("--time-step", type=int, default=1)
    parser.add_argument("--spatial-stride", type=int, default=1)
    parser.add_argument("--method", choices=METHODS, default="hilda")
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--ensemble-size", type=int, default=20)
    parser.add_argument("--sensor-count", type=int, default=64)
    parser.add_argument("--observation-noise", type=float, default=0.05)
    parser.add_argument(
        "--observation-transform",
        choices=("linear", "atan", "square_signed"),
        default="linear",
    )
    parser.add_argument("--initial-spread", type=float, default=0.05)
    parser.add_argument("--process-noise", type=float, default=0.01)
    parser.add_argument(
        "--forecast-model",
        choices=("linear_extrapolation", "persistence"),
        default="linear_extrapolation",
    )
    parser.add_argument("--noise-smoothing-radius", type=int, default=4)
    parser.add_argument("--hilda-path-log-scale", type=float, default=0.35)
    parser.add_argument("--iensf-gamma", type=float, default=None)
    parser.add_argument(
        "--iensf-variance-split-mode",
        choices=("variance_consistent", "literal"),
        default="variance_consistent",
    )
    parser.add_argument("--iensf-flow-steps", type=int, default=40)
    parser.add_argument("--iensf-refinement-iterations", type=int, default=4)
    parser.add_argument("--ensf-flow-steps", type=int, default=100)
    parser.add_argument("--enff-flow-steps", type=int, default=5)
    parser.add_argument("--enff-guidance-lambda", type=float, default=0.001)
    parser.add_argument("--coverage-level", type=float, default=0.9)
    parser.add_argument("--energy-score-chunk-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--resume-run", type=Path, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.ensemble_size < 4:
        raise ValueError("ensemble-size must be at least 4")
    if args.time_step < 1 or args.spatial_stride < 1:
        raise ValueError("time-step and spatial-stride must be positive")
    if args.time_start < 0 or (
        args.time_stop is not None and args.time_stop <= args.time_start
    ):
        raise ValueError("time slice must have a non-negative start and stop after start")
    if args.sensor_count < 1 or 2 * args.sensor_count > HILDAConfig().flow.max_patch_observations:
        raise ValueError("sensor-count must lie in [1, 64] for two-channel HILDA patches")
    if args.observation_noise <= 0.0:
        raise ValueError("observation-noise must be positive")
    if args.initial_spread <= 0.0 or args.process_noise < 0.0:
        raise ValueError("initial-spread must be positive and process-noise non-negative")
    if args.noise_smoothing_radius < 0:
        raise ValueError("noise-smoothing-radius must be non-negative")
    if args.hilda_path_log_scale < 0.0:
        raise ValueError("hilda-path-log-scale must be non-negative")
    if not 0.0 < args.coverage_level < 1.0:
        raise ValueError("coverage-level must lie strictly between zero and one")
    if args.energy_score_chunk_size < 1 or args.checkpoint_interval < 1:
        raise ValueError("chunk and checkpoint intervals must be positive")
    if args.ensf_flow_steps < 2 or args.enff_flow_steps < 2:
        raise ValueError("flow methods require at least two path-time points")


def validate_ns_incom_schema(
    adapter: PDEBenchHDF5Adapter,
    *,
    expected_grid: tuple[int, int] = EXPECTED_SOURCE_GRID,
) -> None:
    schema = adapter.schema
    if schema.kind != "ns_incom":
        raise ValueError(f"Expected PDEBench NS_Incom schema, got {schema.kind!r}")
    if schema.spatial_shape != expected_grid:
        raise ValueError(
            f"Expected the official {expected_grid} source grid, got {schema.spatial_shape}"
        )
    if schema.channel_names != EXPECTED_CHANNELS:
        raise ValueError(
            f"Expected velocity-only channels {EXPECTED_CHANNELS}, got {schema.channel_names}"
        )


def validate_collection_record(
    report_path: Path,
    data_path: Path,
    adapter: PDEBenchHDF5Adapter,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("validated") is not True:
        raise ValueError("PDEBench collection report is not validated")
    files = report.get("files")
    if not isinstance(files, dict) or data_path.name not in files:
        raise ValueError("PDEBench collection report does not contain the selected file")
    record = files[data_path.name]
    if not isinstance(record, dict):
        raise ValueError("PDEBench collection file record must be an object")
    size_bytes = int(record.get("size_bytes", -1))
    if data_path.stat().st_size != size_bytes:
        raise ValueError("PDEBench file size differs from the validated collection record")
    actual_md5 = str(record.get("actual_md5", "")).lower()
    if adapter.manifest_record is not None and (
        actual_md5 != adapter.manifest_record.expected_md5.lower()
    ):
        raise ValueError("Collection MD5 does not match the official manifest")
    return {"actual_md5": actual_md5, "size_bytes": size_bytes}


class PDEBenchFrameStream:
    """Load exactly one selected HDF5 frame at a time through the shared adapter."""

    def __init__(
        self,
        adapter: PDEBenchHDF5Adapter,
        *,
        trajectory_index: int,
        time_start: int,
        time_stop: int | None,
        time_step: int,
        spatial_stride: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if not 0 <= trajectory_index < adapter.schema.trajectory_count:
            raise IndexError("trajectory-index is outside the PDEBench file")
        time_count = adapter.schema.source_shapes[0][1]
        stop = time_count if time_stop is None else min(time_stop, time_count)
        self.raw_indices = tuple(range(time_start, stop, time_step))
        if not self.raw_indices:
            raise ValueError("The requested time slice contains no frames")
        self.adapter = adapter
        self.trajectory_index = trajectory_index
        self.spatial_stride = spatial_stride
        self.dtype = dtype
        self.device = device

    def __len__(self) -> int:
        return len(self.raw_indices)

    def frame(self, position: int) -> PDEBenchTrajectory:
        raw_index = self.raw_indices[position]
        trajectory = self.adapter.load_trajectory(
            TrajectorySlice(
                trajectory_index=self.trajectory_index,
                time_start=raw_index,
                time_stop=raw_index + 1,
                spatial_stride=self.spatial_stride,
                channel_indices=(0, 1),
            ),
            dtype=self.dtype,
            device=self.device,
        )
        if trajectory.states.shape[0] != 1:
            raise RuntimeError("Frame stream materialized more than one time sample")
        return trajectory


def seeded_sensor_mask(
    spatial_shape: tuple[int, int],
    sensor_count: int,
    seed: int,
) -> torch.Tensor:
    point_count = math.prod(spatial_shape)
    if not 1 <= sensor_count <= point_count:
        raise ValueError("sensor-count exceeds the assimilation grid")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    selected = torch.randperm(point_count, generator=generator)[:sensor_count]
    mask = torch.zeros(point_count, dtype=torch.bool)
    mask[selected] = True
    return mask.reshape(spatial_shape)


def channel_scales(state: torch.Tensor, spatial_shape: tuple[int, int]) -> torch.Tensor:
    values = state.reshape(*spatial_shape, len(EXPECTED_CHANNELS))
    flat = values.reshape(-1, values.shape[-1])
    scales = torch.sqrt(torch.mean((flat - flat.mean(dim=0)).square(), dim=0))
    floor = torch.finfo(state.dtype).eps * state.abs().mean().clamp_min(1.0)
    return scales.clamp_min(floor)


def smooth_relative_noise(
    ensemble_size: int,
    spatial_shape: tuple[int, int],
    scales: torch.Tensor,
    relative_amplitude: float,
    smoothing_radius: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    ny, nx = spatial_shape
    noise = torch.randn(
        ensemble_size,
        len(EXPECTED_CHANNELS),
        ny,
        nx,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    if smoothing_radius:
        radius = min(smoothing_radius, max(0, min(ny, nx) // 2 - 1))
        if radius:
            noise = functional.pad(noise, (radius, radius, radius, radius), mode="circular")
            noise = functional.avg_pool2d(noise, kernel_size=2 * radius + 1, stride=1)
    noise = noise - noise.mean(dim=(-2, -1), keepdim=True)
    rms = torch.sqrt(noise.square().mean(dim=(-2, -1), keepdim=True)).clamp_min(
        torch.finfo(dtype).eps
    )
    noise = noise / rms
    noise = noise * (relative_amplitude * scales.reshape(1, -1, 1, 1))
    return noise.permute(0, 2, 3, 1).reshape(ensemble_size, -1)


def hilda_process_scales(
    tracker: AlphaEvidenceTracker,
    log_scale: float,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.exp(log_scale * liu_quantile(tracker.alpha)).to(dtype=dtype)


def analysis_trend_increment(
    forecast_model: str,
    previous_estimate: torch.Tensor | None,
    latest_estimate: torch.Tensor | None,
) -> torch.Tensor | None:
    if forecast_model == "persistence" or previous_estimate is None or latest_estimate is None:
        return None
    if forecast_model != "linear_extrapolation":
        raise ValueError(forecast_model)
    if previous_estimate.shape != latest_estimate.shape:
        raise ValueError("Analysis estimates must have matching shapes")
    return latest_estimate - previous_estimate


def build_iensf_config(args: argparse.Namespace) -> IEnSFConfig:
    gamma = args.iensf_gamma
    if gamma is None:
        gamma = 1.0 if args.observation_transform == "linear" else 0.25
    return IEnSFConfig(
        gamma=gamma,
        variance_split_mode=args.iensf_variance_split_mode,
        sampling_time_step_count=args.iensf_flow_steps,
        refinement_iterations=args.iensf_refinement_iterations,
    )


def analyze_ensemble(
    method: str,
    ensemble: torch.Tensor,
    previous_filtering: torch.Tensor | None,
    observation: torch.Tensor,
    observation_operator,
    observation_covariance: torch.Tensor,
    *,
    hilda: HILDAFilter,
    tracker: AlphaEvidenceTracker,
    args: argparse.Namespace,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    diagnostic: dict[str, Any] = {}
    if method == "hilda":
        analysis = hilda.analyze_paths(
            ensemble,
            tracker,
            observation,
            observation_operator,
            observation_covariance,
        )
        ensemble = analysis.ensembles
        metric_ensemble = ensemble.reshape(-1, ensemble.shape[-1])
        metric_weights = (
            analysis.evidence_weights.to(metric_ensemble)
            .unsqueeze(1)
            .expand(-1, ensemble.shape[1])
            .reshape(-1)
            / ensemble.shape[1]
        )
        diagnostic = {
            "liu_coordinate_estimate": analysis.alpha_estimate,
            "active_path_count": int(tracker.alpha.numel()),
            "mean_final_innovation": float(
                sum(item.final_innovation for item in analysis.diagnostics)
                / len(analysis.diagnostics)
            ),
        }
        return ensemble, None, analysis.state_estimate, metric_ensemble, metric_weights, diagnostic

    if method == "denkf":
        ensemble = denkf_analysis(
            ensemble, observation, observation_operator, observation_covariance
        )
    elif method == "letkf":
        ensemble = letkf_analysis(
            ensemble, observation, observation_operator, observation_covariance
        )
    elif method == "ensf":
        ensemble = ensf_analysis(
            ensemble,
            observation,
            observation_operator,
            observation_covariance,
            EnSFConfig(sampling_time_step_count=args.ensf_flow_steps),
            generator,
        )
    elif method == "iensf":
        ensemble = iensf_analysis(
            ensemble,
            observation,
            observation_operator,
            observation_covariance,
            build_iensf_config(args),
            generator,
        )
    elif method == "ensf_lr_ridge":
        ensemble = ensf_lr_ridge_analysis(
            ensemble,
            observation,
            observation_operator,
            observation_covariance,
            generator=generator,
        )
    elif method == "enff_f2p":
        if previous_filtering is None:
            raise RuntimeError("EnFF-F2P requires the previous filtering ensemble")
        ensemble = enff_f2p_analysis(
            previous_filtering,
            ensemble,
            observation,
            observation_operator,
            observation_covariance,
            EnFFF2PConfig(
                sampling_time_step_count=args.enff_flow_steps,
                guidance_lambda=args.enff_guidance_lambda,
            ),
            generator,
        )
        previous_filtering = ensemble.clone()
    else:
        raise ValueError(method)
    estimate = ensemble.mean(dim=0)
    metric_weights = torch.full(
        (ensemble.shape[0],),
        1.0 / ensemble.shape[0],
        dtype=ensemble.dtype,
        device=ensemble.device,
    )
    diagnostic["mean_final_innovation"] = float(
        torch.linalg.vector_norm(observation_operator(ensemble).mean(0) - observation)
    )
    return ensemble, previous_filtering, estimate, ensemble, metric_weights, diagnostic


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def atomic_checkpoint(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def source_hash(project_root: Path) -> str:
    digest = hashlib.sha256()
    paths = list((project_root / "hilda_da").rglob("*.py"))
    paths.append(Path(__file__).resolve())
    for path in sorted(paths):
        digest.update(path.relative_to(project_root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def resolved_config(
    args: argparse.Namespace,
    adapter: PDEBenchHDF5Adapter,
    stream: PDEBenchFrameStream,
    first_frame: PDEBenchTrajectory,
) -> dict[str, Any]:
    config = vars(args).copy()
    for key in ("run_id", "resume_run", "checkpoint_interval"):
        config.pop(key)
    config["data"] = str(args.data.resolve())
    config["manifest"] = str(args.manifest.resolve()) if args.manifest else None
    config["collection_validation"] = (
        str(args.collection_validation.resolve()) if args.collection_validation else None
    )
    config["output_root"] = str(args.output_root.resolve())
    config["source_grid"] = list(adapter.schema.spatial_shape)
    config["assimilation_grid"] = list(first_frame.spatial_shape)
    config["channel_order"] = list(first_frame.channel_names)
    config["state_dim"] = first_frame.state_dim
    config["selected_raw_time_indices"] = list(stream.raw_indices)
    config["forecast_protocol"] = {
        "name": args.forecast_model,
        "future_target_frames_used_by_forecast": False,
        "normalization": "none",
        "deterministic_increment": (
            "latest_analysis_minus_previous_analysis"
            if args.forecast_model == "linear_extrapolation"
            else "zero"
        ),
        "stochastic_increment": "spatially_correlated_relative_noise",
        "hilda_path_role": "Liu-coordinate modulation of process-noise amplitude",
    }
    config["hilda"] = asdict(HILDAConfig())
    if args.method == "iensf":
        config["iensf"] = asdict(build_iensf_config(args))
    return config


def resume_signature(config: dict[str, Any]) -> dict[str, Any]:
    result = config.copy()
    result.pop("output_root", None)
    return result


def summarize(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    names = (
        "state_rmse",
        "observation_rmse",
        "crps",
        "energy_score",
        "coverage",
        "interval_width",
        "cycle_seconds",
    )
    return {
        "cycle_count": len(metrics),
        "means": {
            name: sum(float(item[name]) for item in metrics) / len(metrics)
            for name in names
        },
        "peak_gpu_memory_bytes": max(
            int(item["peak_gpu_memory_bytes"]) for item in metrics
        ),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    adapter = PDEBenchHDF5Adapter(args.data, manifest_path=args.manifest)
    validate_ns_incom_schema(adapter)
    collection_record = None
    if args.collection_validation is not None:
        collection_record = validate_collection_record(
            args.collection_validation,
            args.data.resolve(),
            adapter,
        )
    if args.verify_manifest_md5:
        if adapter.manifest_record is None:
            raise ValueError("--verify-manifest-md5 requires --manifest")
        verify_manifest_checksum(args.data, adapter.manifest_record)
    stream = PDEBenchFrameStream(
        adapter,
        trajectory_index=args.trajectory_index,
        time_start=args.time_start,
        time_stop=args.time_stop,
        time_step=args.time_step,
        spatial_stride=args.spatial_stride,
        dtype=dtype,
        device=device,
    )
    first_frame = stream.frame(0)
    if first_frame.channel_names != EXPECTED_CHANNELS:
        raise RuntimeError("PDEBench replay must retain both velocity channels in fixed order")
    mask = seeded_sensor_mask(first_frame.spatial_shape, args.sensor_count, args.seed + 1)
    observation_operator = first_frame.sparse_observation(
        mask, transform=args.observation_transform
    )
    observation_covariance = args.observation_noise**2 * torch.eye(
        observation_operator.indices.numel(), dtype=dtype, device=device
    )
    scales = channel_scales(first_frame.states[0], first_frame.spatial_shape)
    configuration = resolved_config(args, adapter, stream, first_frame)
    configuration_hash = hashlib.sha256(
        json.dumps(configuration, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]

    if args.resume_run is None:
        run_id = args.run_id or (
            f"pdebench_ns_incom_f{args.data.stem.split('-')[-1]}_"
            f"tr{args.trajectory_index}_{args.method}_s{args.seed}_{configuration_hash}"
        )
        if Path(run_id).name != run_id or run_id in {".", ".."}:
            raise ValueError("run-id must be a single directory name")
        run_directory = args.output_root / run_id
        if run_directory.exists():
            raise FileExistsError(f"Immutable run already exists: {run_directory}")
        run_directory.mkdir(parents=True)
        atomic_json(run_directory / "config.json", configuration)
    else:
        run_directory = args.resume_run.resolve()
        stored_path = run_directory / "config.json"
        checkpoint_path = run_directory / "checkpoint.pt"
        if not stored_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError("Resume requires config.json and checkpoint.pt")
        stored = json.loads(stored_path.read_text(encoding="utf-8"))
        if resume_signature(stored) != resume_signature(configuration):
            raise ValueError("Resume arguments do not match the immutable configuration")
        configuration = stored
        run_id = run_directory.name

    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    process_generator = torch.Generator(device=device).manual_seed(args.seed)
    observation_generator = torch.Generator(device=device).manual_seed(args.seed + 2)
    hilda = HILDAFilter()
    tracker = AlphaEvidenceTracker.create(
        hilda.config.alpha, device=device, dtype=torch.float64
    )
    metrics: list[dict[str, Any]] = []
    start_position = 0
    resume_count = 0
    elapsed_before_resume = 0.0
    previous_analysis_estimate: torch.Tensor | None = None
    latest_analysis_estimate: torch.Tensor | None = None

    if args.resume_run is None:
        initial_noise = smooth_relative_noise(
            args.ensemble_size,
            first_frame.spatial_shape,
            scales,
            args.initial_spread,
            args.noise_smoothing_radius,
            dtype=dtype,
            device=device,
            generator=process_generator,
        )
        base_ensemble = first_frame.states[0].unsqueeze(0) + initial_noise
        if args.method == "hilda":
            ensemble = base_ensemble.unsqueeze(0).expand(
                tracker.alpha.numel(), -1, -1
            ).clone()
            previous_filtering = None
        else:
            ensemble = base_ensemble
            previous_filtering = ensemble.clone() if args.method == "enff_f2p" else None
    else:
        checkpoint = torch.load(
            run_directory / "checkpoint.pt", map_location=device, weights_only=False
        )
        if checkpoint.get("completed", False):
            raise RuntimeError("Run is already complete")
        start_position = int(checkpoint["next_position"])
        ensemble = checkpoint["ensemble"].to(device=device, dtype=dtype)
        previous_filtering = checkpoint["previous_filtering"]
        if previous_filtering is not None:
            previous_filtering = previous_filtering.to(device=device, dtype=dtype)
        scales = checkpoint["channel_scales"].to(device=device, dtype=dtype)
        metrics = checkpoint["metrics"]
        previous_analysis_estimate = checkpoint["previous_analysis_estimate"]
        latest_analysis_estimate = checkpoint["latest_analysis_estimate"]
        if previous_analysis_estimate is not None:
            previous_analysis_estimate = previous_analysis_estimate.to(
                device=device, dtype=dtype
            )
        if latest_analysis_estimate is not None:
            latest_analysis_estimate = latest_analysis_estimate.to(
                device=device, dtype=dtype
            )
        process_generator.set_state(checkpoint["process_generator_state"].cpu())
        observation_generator.set_state(checkpoint["observation_generator_state"].cpu())
        if args.method == "hilda":
            tracker.alpha = checkpoint["tracker_alpha"].to(device=device, dtype=torch.float64)
            tracker.log_scores = checkpoint["tracker_log_scores"].to(
                device=device, dtype=torch.float64
            )
            tracker.low_evidence_counts = checkpoint["tracker_low_evidence_counts"].to(
                device=device, dtype=torch.int64
            )
        elapsed_before_resume = float(checkpoint.get("elapsed_seconds", 0.0))
        resume_count = int(checkpoint.get("resume_count", 0)) + 1

    started = time.time()
    next_position = start_position

    def checkpoint_payload(completed: bool) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "completed": completed,
            "next_position": next_position,
            "ensemble": ensemble.detach(),
            "previous_filtering": (
                previous_filtering.detach() if previous_filtering is not None else None
            ),
            "channel_scales": scales.detach(),
            "metrics": metrics,
            "previous_analysis_estimate": (
                previous_analysis_estimate.detach()
                if previous_analysis_estimate is not None
                else None
            ),
            "latest_analysis_estimate": (
                latest_analysis_estimate.detach()
                if latest_analysis_estimate is not None
                else None
            ),
            "process_generator_state": process_generator.get_state(),
            "observation_generator_state": observation_generator.get_state(),
            "tracker_alpha": tracker.alpha.detach() if args.method == "hilda" else None,
            "tracker_log_scores": (
                tracker.log_scores.detach() if args.method == "hilda" else None
            ),
            "tracker_low_evidence_counts": (
                tracker.low_evidence_counts.detach() if args.method == "hilda" else None
            ),
            "elapsed_seconds": elapsed_before_resume + time.time() - started,
            "peak_gpu_memory_bytes": (
                max(int(item["peak_gpu_memory_bytes"]) for item in metrics)
                if metrics
                else 0
            ),
            "resume_count": resume_count,
        }

    completed = False
    failure: str | None = None
    try:
        for position in range(start_position, len(stream)):
            cycle_started = time.time()
            frame = first_frame if position == 0 else stream.frame(position)
            truth = frame.states[0]
            if position > 0:
                trend_increment = analysis_trend_increment(
                    args.forecast_model,
                    previous_analysis_estimate,
                    latest_analysis_estimate,
                )
                if trend_increment is not None:
                    ensemble = ensemble + trend_increment
            if position > 0 and args.process_noise:
                base_noise = smooth_relative_noise(
                    args.ensemble_size,
                    frame.spatial_shape,
                    scales,
                    args.process_noise,
                    args.noise_smoothing_radius,
                    dtype=dtype,
                    device=device,
                    generator=process_generator,
                )
                if args.method == "hilda":
                    path_scales = hilda_process_scales(
                        tracker, args.hilda_path_log_scale, dtype=dtype
                    )
                    ensemble = ensemble + path_scales[:, None, None] * base_noise[None]
                else:
                    ensemble = ensemble + base_noise

            clean_observation = observation_operator(truth.unsqueeze(0)).squeeze(0)
            observation = clean_observation + args.observation_noise * torch.randn(
                clean_observation.shape,
                dtype=dtype,
                device=device,
                generator=observation_generator,
            )
            (
                ensemble,
                previous_filtering,
                estimate,
                metric_ensemble,
                metric_weights,
                diagnostic,
            ) = analyze_ensemble(
                args.method,
                ensemble,
                previous_filtering,
                observation,
                observation_operator,
                observation_covariance,
                hilda=hilda,
                tracker=tracker,
                args=args,
                generator=process_generator,
            )
            previous_analysis_estimate = latest_analysis_estimate
            latest_analysis_estimate = estimate.detach().clone()
            coverage, interval_width = weighted_central_interval_coverage_width(
                metric_ensemble,
                truth,
                metric_weights,
                level=args.coverage_level,
            )
            predicted_observation = observation_operator(estimate.unsqueeze(0)).squeeze(0)
            state_error = state_rmse(estimate, truth)
            observed_error = observation_rmse(
                predicted_observation, clean_observation
            )
            crps = weighted_ensemble_crps(metric_ensemble, truth, metric_weights)
            energy_score = weighted_multivariate_energy_score(
                metric_ensemble,
                truth,
                metric_weights,
                chunk_size=args.energy_score_chunk_size,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak_memory = int(torch.cuda.max_memory_allocated(device))
            else:
                peak_memory = 0
            record = {
                "cycle": position,
                "raw_time_index": stream.raw_indices[position],
                "time": (
                    float(frame.times[0]) if frame.times is not None else None
                ),
                "state_rmse": float(state_error),
                "observation_rmse": float(observed_error),
                "crps": float(crps),
                "energy_score": float(energy_score),
                "coverage": float(coverage),
                "interval_width": float(interval_width),
                "coverage_level": args.coverage_level,
                "forecast_model": args.forecast_model,
                "cycle_seconds": time.time() - cycle_started,
                "peak_gpu_memory_bytes": peak_memory,
                **diagnostic,
            }
            metrics.append(record)
            next_position = position + 1
            if len(metrics) % args.checkpoint_interval == 0:
                atomic_checkpoint(
                    run_directory / "checkpoint.pt", checkpoint_payload(False)
                )
        completed = True
        atomic_checkpoint(run_directory / "checkpoint.pt", checkpoint_payload(True))
        atomic_json(run_directory / "metrics.json", metrics)
        atomic_json(run_directory / "summary.json", summarize(metrics))
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        atomic_checkpoint(run_directory / "checkpoint.pt", checkpoint_payload(False))
        raise
    finally:
        provenance = {
            "schema_version": 1,
            "completed": completed,
            "failure": failure,
            "run_id": run_id,
            "method": args.method,
            "dataset": "PDEBench NS_Incom",
            "source_grid": list(adapter.schema.spatial_shape),
            "assimilation_grid": list(first_frame.spatial_shape),
            "channels": list(EXPECTED_CHANNELS),
            "pressure_present": False,
            "normalization": "none",
            "forecast_uses_future_target_frames": False,
            "forecast_model": args.forecast_model,
            "collection_validation": collection_record,
            "source": first_frame.provenance.as_dict(),
            "source_hash": source_hash(Path(__file__).resolve().parents[1]),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "elapsed_seconds": elapsed_before_resume + time.time() - started,
            "resume_count": resume_count,
        }
        atomic_json(run_directory / "provenance.json", provenance)


if __name__ == "__main__":
    main()
