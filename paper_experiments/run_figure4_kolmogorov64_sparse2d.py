from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hilda_da.alpha import liu_quantile
from hilda_da.observations import SparseObservation
from paper_experiments import run_figure4_lorenz96_1024_scaling as core


MethodName = Literal["aug_enkf", "bma_static", "pce", "apce"]
METHODS: tuple[MethodName, ...] = ("aug_enkf", "bma_static", "pce", "apce")
METHOD_LABELS = core.METHOD_LABELS

DEFAULT_KOL_DATA = Path(
    "<EXTERNAL_DATA_ROOT>/S3GM/S3GM-main/S3GM-main/data/kolmogorov_flow_test.npy"
)
TEST_REYNOLDS = (50.0, 125.0, 575.0, 1100.0, 1500.0)
TEST_WAVENUMBERS = (2, 4, 6, 8)

TUNING_PROFILE_OVERRIDES: dict[str, dict[str, Any]] = {
    "baseline": {},
    "pce_temp_026": {"pce_temperature": 0.26},
    "pce_temp_030": {"pce_temperature": 0.30},
    "pce_temp_036": {"pce_temperature": 0.36},
    "apce_floor_060": {"apce_entropy_floor": 0.60},
    "apce_floor_070": {"apce_entropy_floor": 0.70},
    "alpha_conservative": {
        "branch_member_alpha_jitter": 0.030,
        "branch_augmented_alpha_analysis_strength": 0.25,
        "global_augmented_alpha_analysis_strength": 0.05,
    },
    "dynamic_regrid": {"dynamic_regrid_from_alpha_members": True},
}


@dataclass(frozen=True)
class KOL64Config:
    seed: int
    tuning_profile: str = "baseline"
    data_path: str = str(DEFAULT_KOL_DATA)
    reynolds: float = 575.0
    forcing_wavenumber: int = 4
    sample_index: int = 0
    nx: int = 64
    ny: int = 64
    state_dim: int = 4096
    sensor_grid: int = 16
    observed_points: int = 256
    obs_interval: int = 2
    steps: int = 100
    dt: float = 0.1
    ensemble_size: int = 64
    obs_noise: float = 0.12
    alpha_true: float = 0.50
    alpha_min: float = 0.08
    alpha_max: float = 0.92
    fixed_alpha: float = 0.50
    reynolds_log_span: float = 0.90
    linear_drag: float = 0.10
    forcing_amplitude: float = 1.0
    stochastic_scale: float = 0.010
    initial_spread: float = 0.10
    state_clip: float = 30.0
    coarse_alpha_grid: tuple[float, ...] = (0.08, 0.22, 0.36, 0.50, 0.64, 0.78, 0.92)
    bma_alpha_grid_size: int = 21
    pce_temperature: float = 0.30
    apce_temperature: float = 0.46
    apce_min_temperature: float = 0.14
    apce_forgetting: float = 0.985
    apce_entropy_floor: float = 0.68
    apce_recycle_entropy_projected_scores: bool = True
    evidence_shrinkage: float = 0.28
    apce_dimension_floor: float = 0.30
    apce_dimension_gain: float = 0.70
    branch_member_alpha_jitter: float = 0.040
    aug_alpha_jitter: float = 0.035
    aug_alpha_random_walk_std: float = 0.004
    branch_augmented_alpha_analysis_strength: float = 0.32
    global_augmented_alpha_analysis_strength: float = 0.08
    global_state_analysis_strength: float = 0.12
    local_grid_points: int = 11
    local_grid_radius: float = 0.18
    local_grid_min_spacing: float = 0.012
    dynamic_regrid_from_alpha_members: bool = False
    localization_scale: float = 4.0
    probabilistic_metric_stride: int = 5
    max_valid_amplitude_ratio: float = 80.0


@dataclass(frozen=True)
class SharedAssets:
    config: KOL64Config
    truth: torch.Tensor
    initial_ensemble: torch.Tensor
    forecast_noise: torch.Tensor
    observation_noise: torch.Tensor
    observation_indices: torch.Tensor
    source_velocity: torch.Tensor
    source_frame_start: int
    asset_path: Path
    asset_sha256: str


@dataclass(frozen=True)
class Scenario:
    config: KOL64Config
    truth: torch.Tensor
    observations: dict[int, torch.Tensor]
    initial_ensemble: torch.Tensor
    forecast_noise: torch.Tensor
    observation_indices: torch.Tensor
    localization: torch.Tensor
    augmented_localization: torch.Tensor
    source_velocity: torch.Tensor
    source_frame_start: int
    asset_path: Path
    asset_sha256: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(seed)
    return generator


def randn(shape: tuple[int, ...], device: torch.device, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(shape, dtype=torch.float64, device=device, generator=generator)


def gaspari_cohn(distance_ratio: torch.Tensor) -> torch.Tensor:
    return core.gaspari_cohn(distance_ratio)


def reynolds_index(value: float) -> int:
    distances = [abs(float(item) - float(value)) for item in TEST_REYNOLDS]
    index = int(np.argmin(distances))
    if distances[index] > 1.0e-8:
        raise ValueError(f"reynolds must be one of {TEST_REYNOLDS}; got {value}")
    return index


def wavenumber_index(value: int) -> int:
    if int(value) not in TEST_WAVENUMBERS:
        raise ValueError(f"forcing_wavenumber must be one of {TEST_WAVENUMBERS}; got {value}")
    return TEST_WAVENUMBERS.index(int(value))


def spectral_vorticity_from_velocity(velocity: np.ndarray) -> np.ndarray:
    if velocity.ndim != 4 or velocity.shape[-1] != 2:
        raise ValueError(f"Expected velocity shape (time, ny, nx, 2), got {velocity.shape}")
    u = velocity[..., 0]
    v = velocity[..., 1]
    ny, nx = int(u.shape[-2]), int(u.shape[-1])
    kx = 2.0 * math.pi * np.fft.fftfreq(nx, d=(2.0 * math.pi) / nx)
    ky = 2.0 * math.pi * np.fft.fftfreq(ny, d=(2.0 * math.pi) / ny)
    ky_grid, kx_grid = np.meshgrid(ky, kx, indexing="ij")
    u_hat = np.fft.fft2(u, axes=(-2, -1))
    v_hat = np.fft.fft2(v, axes=(-2, -1))
    omega = np.fft.ifft2(1j * kx_grid * v_hat - 1j * ky_grid * u_hat, axes=(-2, -1)).real
    return omega.astype(np.float64, copy=False)


def smooth_periodic_noise_2d(noise: torch.Tensor, passes: int = 1) -> torch.Tensor:
    field = noise.reshape(*noise.shape[:-1], 64, 64)
    for _ in range(passes):
        field = (
            0.12 * torch.roll(field, shifts=(2, 0), dims=(-2, -1))
            + 0.18 * torch.roll(field, shifts=(1, 0), dims=(-2, -1))
            + 0.12 * torch.roll(field, shifts=(-2, 0), dims=(-2, -1))
            + 0.18 * torch.roll(field, shifts=(-1, 0), dims=(-2, -1))
            + 0.12 * torch.roll(field, shifts=(0, 2), dims=(-2, -1))
            + 0.18 * torch.roll(field, shifts=(0, 1), dims=(-2, -1))
            + 0.12 * torch.roll(field, shifts=(0, -2), dims=(-2, -1))
            + 0.18 * torch.roll(field, shifts=(0, -1), dims=(-2, -1))
            + 0.20 * field
        )
    scale = field.std(dim=(-2, -1), keepdim=True).clamp_min(1.0e-12)
    return (field / scale).reshape_as(noise)


class KolmogorovVorticity64:
    """Periodic 2-D Kolmogorov-flow vorticity model used as the forecast core.

    The reference trajectory is loaded from the S3GM/NMI Kolmogorov dataset.
    This class supplies the explicit coarse-grained PDE forecast model:

    omega_t = -u omega_x - v omega_y + nu(alpha) laplacian(omega)
              - 0.1 omega - A k cos(k y).
    """

    def __init__(self, config: KOL64Config) -> None:
        self.config = config

    def _reshape(self, state: torch.Tensor) -> torch.Tensor:
        return state.reshape(*state.shape[:-1], self.config.ny, self.config.nx)

    def _wave_numbers(self, field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kx = 2.0 * math.pi * torch.fft.fftfreq(
            self.config.nx,
            d=(2.0 * math.pi) / self.config.nx,
            dtype=field.dtype,
            device=field.device,
        )
        ky = 2.0 * math.pi * torch.fft.fftfreq(
            self.config.ny,
            d=(2.0 * math.pi) / self.config.ny,
            dtype=field.dtype,
            device=field.device,
        )
        ky_grid, kx_grid = torch.meshgrid(ky, kx, indexing="ij")
        squared = kx_grid.square() + ky_grid.square()
        return ky_grid, kx_grid, squared

    def _dealias(self, spectrum: torch.Tensor) -> torch.Tensor:
        mode_x = torch.fft.fftfreq(self.config.nx, device=spectrum.device).abs() * self.config.nx
        mode_y = torch.fft.fftfreq(self.config.ny, device=spectrum.device).abs() * self.config.ny
        mask = (mode_y[:, None] <= self.config.ny // 3) & (mode_x[None, :] <= self.config.nx // 3)
        return torch.where(mask, spectrum, torch.zeros_like(spectrum))

    def reynolds_from_alpha(self, alpha: torch.Tensor | float, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        alpha_tensor = torch.as_tensor(alpha, dtype=dtype, device=device)
        q = liu_quantile(alpha_tensor)
        q_true = liu_quantile(torch.as_tensor(self.config.alpha_true, dtype=dtype, device=device))
        log_re = math.log(self.config.reynolds) + self.config.reynolds_log_span * (q - q_true)
        return torch.exp(log_re)

    def drift(self, state: torch.Tensor, alpha: torch.Tensor | float) -> torch.Tensor:
        vorticity = self._reshape(state)
        ky, kx, squared_wave = self._wave_numbers(vorticity)
        spectrum = self._dealias(torch.fft.fft2(vorticity, dim=(-2, -1)))
        inverse_laplacian = torch.where(squared_wave > 0, squared_wave.reciprocal(), torch.zeros_like(squared_wave))
        streamfunction_spectrum = spectrum * inverse_laplacian
        velocity_x = torch.fft.ifft2(1j * ky * streamfunction_spectrum, dim=(-2, -1)).real
        velocity_y = torch.fft.ifft2(-1j * kx * streamfunction_spectrum, dim=(-2, -1)).real
        gradient_x = torch.fft.ifft2(1j * kx * spectrum, dim=(-2, -1)).real
        gradient_y = torch.fft.ifft2(1j * ky * spectrum, dim=(-2, -1)).real
        laplacian = torch.fft.ifft2(-squared_wave * spectrum, dim=(-2, -1)).real
        reynolds = self.reynolds_from_alpha(alpha, dtype=state.dtype, device=state.device)
        viscosity = reynolds.reciprocal()
        while viscosity.ndim < vorticity.ndim:
            viscosity = viscosity.unsqueeze(-1)
        grid_y = torch.linspace(
            0.0,
            2.0 * math.pi,
            self.config.ny + 1,
            dtype=state.dtype,
            device=state.device,
        )[:-1]
        forcing = -self.config.forcing_amplitude * self.config.forcing_wavenumber * torch.cos(
            self.config.forcing_wavenumber * grid_y
        )[:, None]
        tendency = (
            -(velocity_x * gradient_x + velocity_y * gradient_y)
            + viscosity * laplacian
            - self.config.linear_drag * vorticity
            + forcing
        )
        return tendency.reshape_as(state)

    def project(self, state: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(
            state,
            nan=0.0,
            posinf=self.config.state_clip,
            neginf=-self.config.state_clip,
        ).clamp(-self.config.state_clip, self.config.state_clip)

    def step(self, state: torch.Tensor, alpha: torch.Tensor | float, noise: torch.Tensor) -> torch.Tensor:
        dt = self.config.dt
        k1 = self.drift(state, alpha)
        k2 = self.drift(state + 0.5 * dt * k1, alpha)
        k3 = self.drift(state + 0.5 * dt * k2, alpha)
        k4 = self.drift(state + dt * k3, alpha)
        deterministic = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        return self.project(deterministic + math.sqrt(dt) * self.config.stochastic_scale * noise)


core.Lorenz96 = KolmogorovVorticity64


def observation_indices(config: KOL64Config, device: torch.device) -> torch.Tensor:
    if config.nx != 64 or config.ny != 64:
        raise ValueError("This worker is intentionally fixed to 64x64 KOL fields.")
    if config.sensor_grid < 1 or config.nx % config.sensor_grid != 0 or config.ny % config.sensor_grid != 0:
        raise ValueError("--sensor-grid must evenly divide 64.")
    stride_x = config.nx // config.sensor_grid
    stride_y = config.ny // config.sensor_grid
    ys = torch.arange(0, config.ny, stride_y, dtype=torch.int64, device=device)
    xs = torch.arange(0, config.nx, stride_x, dtype=torch.int64, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return (yy.reshape(-1) * config.nx + xx.reshape(-1)).to(dtype=torch.int64)


def periodic_2d_localization(config: KOL64Config, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    device = indices.device
    dtype = torch.float64
    state = torch.arange(config.state_dim, dtype=torch.int64, device=device)
    sy = (state // config.nx).to(dtype=dtype)[:, None]
    sx = (state % config.nx).to(dtype=dtype)[:, None]
    oy = (indices // config.nx).to(dtype=dtype)[None, :]
    ox = (indices % config.nx).to(dtype=dtype)[None, :]
    dx = (sx - ox).abs()
    dy = (sy - oy).abs()
    dx = torch.minimum(dx, config.nx - dx)
    dy = torch.minimum(dy, config.ny - dy)
    distance = torch.sqrt(dx.square() + dy.square())
    physical = gaspari_cohn(distance / config.localization_scale)
    augmented = torch.cat((physical, torch.ones((1, physical.shape[1]), dtype=dtype, device=device)), dim=0)
    return physical, augmented


def asset_path(asset_root: Path, config: KOL64Config) -> Path:
    return (
        asset_root
        / "kolmogorov64"
        / f"re{int(config.reynolds)}_k{config.forcing_wavenumber}_sample{config.sample_index}"
        / f"obs{config.sensor_grid}x{config.sensor_grid}_steps{config.steps}_seed{config.seed}.npz"
    )


def load_velocity_segment(config: KOL64Config, seed: int) -> tuple[np.ndarray, int]:
    data_path = Path(config.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"KOL data not found: {data_path}")
    raw = np.load(data_path, mmap_mode="r")
    if raw.shape == (20, 400, 64, 64, 2):
        data = raw.reshape(5, 4, 400, 64, 64, 2)
    elif raw.shape == (5, 4, 400, 64, 64, 2):
        data = raw
    else:
        raise ValueError(f"Unexpected KOL data shape: {raw.shape}")
    re_index = reynolds_index(config.reynolds)
    k_index = wavenumber_index(config.forcing_wavenumber)
    sequence = np.asarray(data[re_index, k_index], dtype=np.float64)
    available = sequence.shape[0]
    needed = config.steps + 1
    if needed + 20 > available:
        raise ValueError(f"steps={config.steps} too long for available KOL frames={available}")
    rng = np.random.default_rng(seed + 73_001 + 101 * int(config.sample_index))
    start = int(rng.integers(10, available - needed - 10 + 1))
    return sequence[start : start + needed], start


def create_shared_assets(config: KOL64Config, asset_root: Path, device: torch.device) -> Path:
    asset_root.mkdir(parents=True, exist_ok=True)
    path = asset_path(asset_root, config)
    if path.exists():
        return path
    velocity, frame_start = load_velocity_segment(config, config.seed)
    truth_np = spectral_vorticity_from_velocity(velocity).reshape(config.steps + 1, config.state_dim)
    truth_scale = float(np.std(truth_np))
    generator = make_generator(device, config.seed)
    truth = torch.as_tensor(truth_np, dtype=torch.float64, device=device)
    initial_noise = smooth_periodic_noise_2d(randn((config.ensemble_size, config.state_dim), device, generator), passes=2)
    forecast_noise = smooth_periodic_noise_2d(
        randn((config.steps, config.ensemble_size, config.state_dim), device, generator),
        passes=2,
    )
    observation_noise = randn((config.steps, config.observed_points), device, generator)
    system = KolmogorovVorticity64(config)
    initial_ensemble = system.project(truth[0].unsqueeze(0) + config.initial_spread * max(truth_scale, 1.0e-8) * initial_noise)
    idx = observation_indices(config, device)
    tmp_path = path.with_name(f"{path.stem}.tmp_{os.getpid()}.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        tmp_path,
        truth=truth.detach().cpu().numpy(),
        initial_ensemble=initial_ensemble.detach().cpu().numpy(),
        forecast_noise=forecast_noise.detach().cpu().numpy(),
        observation_noise=observation_noise.detach().cpu().numpy(),
        observation_indices=idx.detach().cpu().numpy(),
        source_velocity=velocity.astype(np.float32, copy=False),
        source_frame_start=np.asarray(frame_start),
        config_json=np.asarray(json.dumps(asdict(config), ensure_ascii=False)),
    )
    tmp_path.replace(path)
    return path


def load_shared_assets(config: KOL64Config, asset_root: Path, device: torch.device) -> SharedAssets:
    path = create_shared_assets(config, asset_root, device)
    with np.load(path, allow_pickle=False) as data:
        stored_config = json.loads(str(data["config_json"].item()))
        immutable_fields = (
            "reynolds",
            "forcing_wavenumber",
            "sensor_grid",
            "observed_points",
            "steps",
            "dt",
            "ensemble_size",
            "alpha_true",
        )
        current = asdict(config)
        for field in immutable_fields:
            if stored_config[field] != current[field]:
                raise RuntimeError(f"Shared asset mismatch for {field}: {stored_config[field]} != {current[field]}")
        return SharedAssets(
            config=config,
            truth=torch.as_tensor(data["truth"], dtype=torch.float64, device=device),
            initial_ensemble=torch.as_tensor(data["initial_ensemble"], dtype=torch.float64, device=device),
            forecast_noise=torch.as_tensor(data["forecast_noise"], dtype=torch.float64, device=device),
            observation_noise=torch.as_tensor(data["observation_noise"], dtype=torch.float64, device=device),
            observation_indices=torch.as_tensor(data["observation_indices"], dtype=torch.int64, device=device),
            source_velocity=torch.as_tensor(data["source_velocity"], dtype=torch.float64, device=device),
            source_frame_start=int(np.asarray(data["source_frame_start"]).item()),
            asset_path=path,
            asset_sha256=file_sha256(path),
        )


def materialize_scenario(shared: SharedAssets) -> Scenario:
    config = shared.config
    observations = {
        step: shared.truth[step, shared.observation_indices] + config.obs_noise * shared.observation_noise[step - 1]
        for step in range(config.obs_interval, config.steps + 1, config.obs_interval)
    }
    physical, augmented = periodic_2d_localization(config, shared.observation_indices)
    return Scenario(
        config=config,
        truth=shared.truth,
        observations=observations,
        initial_ensemble=shared.initial_ensemble,
        forecast_noise=shared.forecast_noise,
        observation_indices=shared.observation_indices,
        localization=physical,
        augmented_localization=augmented,
        source_velocity=shared.source_velocity,
        source_frame_start=shared.source_frame_start,
        asset_path=shared.asset_path,
        asset_sha256=shared.asset_sha256,
    )


def covariance(scenario: Scenario) -> torch.Tensor:
    size = int(scenario.observation_indices.numel())
    return scenario.config.obs_noise**2 * torch.eye(size, dtype=torch.float64, device=scenario.truth.device)


core.covariance = covariance


def numerical_status(result: dict[str, Any], scenario: Scenario) -> str:
    required = ("nrmse", "rmse", "crps", "coverage_90", "interval_width_90", "max_abs_state")
    if not all(math.isfinite(float(result[key])) for key in required):
        return "nonfinite"
    truth_scale = float(torch.max(torch.abs(scenario.truth)).detach().cpu())
    if truth_scale and float(result["max_abs_state"]) > scenario.config.max_valid_amplitude_ratio * truth_scale:
        return "diverged"
    return "valid"


def trace_path(output: Path, method: MethodName, config: KOL64Config) -> Path:
    return (
        output
        / "artifacts"
        / "method_traces"
        / "kolmogorov64"
        / f"re{int(config.reynolds)}_k{config.forcing_wavenumber}"
        / f"obs{config.sensor_grid}x{config.sensor_grid}_time{config.obs_interval}"
        / method
        / f"seed_{config.seed}.npz"
    )


def save_trace(output: Path, method: MethodName, scenario: Scenario, result: dict[str, Any]) -> str:
    path = trace_path(output, method, scenario.config)
    path.parent.mkdir(parents=True, exist_ok=True)
    obs_matrix = np.full((scenario.config.steps + 1, scenario.config.observed_points), np.nan, dtype=np.float64)
    for step, values in scenario.observations.items():
        obs_matrix[step] = values.detach().cpu().numpy()
    payload: dict[str, Any] = {
        "times": np.arange(scenario.config.steps + 1, dtype=float) * scenario.config.dt,
        "truth_states": scenario.truth.detach().cpu().numpy(),
        "source_velocity": scenario.source_velocity.detach().cpu().numpy(),
        "source_frame_start": np.asarray(scenario.source_frame_start),
        "observations": obs_matrix,
        "observation_indices": scenario.observation_indices.detach().cpu().numpy(),
        "alpha_true": np.asarray(scenario.config.alpha_true),
        "reynolds": np.asarray(scenario.config.reynolds),
        "forcing_wavenumber": np.asarray(scenario.config.forcing_wavenumber),
        "obs_interval": np.asarray(scenario.config.obs_interval),
        "sensor_grid": np.asarray(scenario.config.sensor_grid),
        "nx": np.asarray(scenario.config.nx),
        "ny": np.asarray(scenario.config.ny),
    }
    for key in ("mean_states", "alpha_mean_history", "alpha_weight_history", "alpha_grid_history"):
        if key in result:
            payload[key] = result[key]
    np.savez_compressed(path, **payload)
    return str(path)


def run_json_path(output: Path, method: MethodName, config: KOL64Config) -> Path:
    return (
        output
        / "artifacts"
        / "run_json"
        / "kolmogorov64"
        / f"re{int(config.reynolds)}_k{config.forcing_wavenumber}"
        / f"obs{config.sensor_grid}x{config.sensor_grid}_time{config.obs_interval}"
        / method
        / f"seed_{config.seed}.json"
    )


def completed_payload(
    scenario: Scenario,
    method: MethodName,
    result: dict[str, Any],
    output: Path,
    record_trace: bool,
) -> dict[str, Any]:
    status = numerical_status(result, scenario)
    trace = save_trace(output, method, scenario, result) if record_trace else ""
    scalars = {key: value for key, value in result.items() if isinstance(value, (str, float, int))}
    return {
        "run_id": (
            f"kolmogorov64_re{int(scenario.config.reynolds)}_k{scenario.config.forcing_wavenumber}_"
            f"obs{scenario.config.sensor_grid}x{scenario.config.sensor_grid}_t{scenario.config.obs_interval}_"
            f"{method}_seed{scenario.config.seed}"
        ),
        "status": "completed",
        "numerical_status": status,
        "case": "kolmogorov64",
        "case_role": "coarse_grained_explicit_pde_forecast_against_public_kolmogorov_flow_data",
        "method": method,
        "label": METHOD_LABELS[method],
        "seed": scenario.config.seed,
        "reynolds": scenario.config.reynolds,
        "forcing_wavenumber": scenario.config.forcing_wavenumber,
        "sample_index": scenario.config.sample_index,
        "source_frame_start": scenario.source_frame_start,
        "nx": scenario.config.nx,
        "ny": scenario.config.ny,
        "state_dim": scenario.config.state_dim,
        "sensor_grid": scenario.config.sensor_grid,
        "observed_points": scenario.config.observed_points,
        "spatial_downsampling_factor_per_axis": scenario.config.nx // scenario.config.sensor_grid,
        "observation_indices": ",".join(str(int(value)) for value in scenario.observation_indices.detach().cpu().tolist()),
        "obs_interval": scenario.config.obs_interval,
        "observation_count": len(scenario.observations),
        "dt": scenario.config.dt,
        "steps": scenario.config.steps,
        "ensemble_size": scenario.config.ensemble_size,
        "alpha_true": scenario.config.alpha_true,
        "alpha_parameterization": "alpha maps to Reynolds/viscosity; alpha_true maps exactly to the dataset Reynolds number",
        "reynolds_log_span": scenario.config.reynolds_log_span,
        "linear_drag": scenario.config.linear_drag,
        "forcing_amplitude": scenario.config.forcing_amplitude,
        "tuning_profile": scenario.config.tuning_profile,
        "coarse_alpha_grid": ",".join(f"{value:.4g}" for value in scenario.config.coarse_alpha_grid),
        "bma_alpha_grid_size": scenario.config.bma_alpha_grid_size,
        "pce_temperature": scenario.config.pce_temperature,
        "apce_temperature": scenario.config.apce_temperature,
        "apce_forgetting": scenario.config.apce_forgetting,
        "apce_entropy_floor": scenario.config.apce_entropy_floor,
        "apce_recycle_entropy_projected_scores": scenario.config.apce_recycle_entropy_projected_scores,
        "apce_dimension_floor": scenario.config.apce_dimension_floor,
        "apce_dimension_gain": scenario.config.apce_dimension_gain,
        "branch_member_alpha_jitter": scenario.config.branch_member_alpha_jitter,
        "branch_augmented_alpha_analysis_strength": scenario.config.branch_augmented_alpha_analysis_strength,
        "global_augmented_alpha_analysis_strength": scenario.config.global_augmented_alpha_analysis_strength,
        "global_state_analysis_strength": scenario.config.global_state_analysis_strength,
        "dynamic_regrid_from_alpha_members": scenario.config.dynamic_regrid_from_alpha_members,
        "localization_scale": scenario.config.localization_scale,
        "localization_compact_support": 2.0 * scenario.config.localization_scale,
        "probabilistic_metric_stride": scenario.config.probabilistic_metric_stride,
        "data_path": scenario.config.data_path,
        "asset_npz": str(scenario.asset_path),
        "asset_sha256": scenario.asset_sha256,
        "trace_npz": trace,
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        **scalars,
    }


def parse_method(text: str) -> MethodName:
    if text not in METHODS:
        raise ValueError(f"--method must be one of {METHODS}")
    return text  # type: ignore[return-value]


def apply_tuning_profile(config: KOL64Config, profile_text: str) -> KOL64Config:
    requested = [item.strip() for item in profile_text.split("+") if item.strip()]
    if not requested:
        requested = ["baseline"]
    if len(requested) > 1:
        requested = [item for item in requested if item != "baseline"]
    overrides: dict[str, Any] = {}
    for name in requested:
        if name not in TUNING_PROFILE_OVERRIDES:
            choices = ", ".join(sorted(TUNING_PROFILE_OVERRIDES))
            raise ValueError(f"Unknown --tuning-profile {name!r}; choose from {choices}")
        overrides.update(TUNING_PROFILE_OVERRIDES[name])
    return replace(config, tuning_profile="+".join(requested), **overrides)


def config_from_args(args: argparse.Namespace) -> KOL64Config:
    if args.sensor_grid < 1 or 64 % args.sensor_grid != 0:
        raise ValueError("--sensor-grid must evenly divide 64")
    observed_points = int(args.sensor_grid) * int(args.sensor_grid)
    base = KOL64Config(
        seed=args.seed,
        data_path=str(args.data_path),
        reynolds=float(args.reynolds),
        forcing_wavenumber=int(args.forcing_wavenumber),
        sample_index=int(args.sample_index),
        sensor_grid=int(args.sensor_grid),
        observed_points=observed_points,
        obs_interval=int(args.obs_interval),
        steps=int(args.steps),
        obs_noise=float(args.obs_noise),
        localization_scale=float(args.localization_scale),
    )
    return apply_tuning_profile(base, args.tuning_profile)


def main() -> None:
    parser = argparse.ArgumentParser(description="Figure 4 KOL-64 sparse 2-D vorticity assimilation worker.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_KOL_DATA)
    parser.add_argument("--reynolds", type=float, default=575.0)
    parser.add_argument("--forcing-wavenumber", type=int, default=4)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--sensor-grid", type=int, default=16)
    parser.add_argument("--obs-interval", type=int, default=2)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--obs-noise", type=float, default=0.12)
    parser.add_argument("--localization-scale", type=float, default=4.0)
    parser.add_argument("--method", default="apce")
    parser.add_argument("--tuning-profile", default="baseline")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda:2" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prepare-asset-only", action="store_true")
    parser.add_argument("--no-record-trace", action="store_true")
    args = parser.parse_args()
    config = config_from_args(args)
    device = torch.device(args.device)
    asset_root = args.asset_root or (args.output / "shared_assets")
    if args.prepare_asset_only:
        path = create_shared_assets(config, asset_root, device)
        print(json.dumps({"status": "asset_ready", "asset": str(path), "sha256": file_sha256(path)}, ensure_ascii=False))
        return
    method = parse_method(args.method)
    run_path = run_json_path(args.output, method, config)
    try:
        shared = load_shared_assets(config, asset_root, device)
        scenario = materialize_scenario(shared)
        result = core.run_method(scenario, method, record_trace=not args.no_record_trace)
        payload = completed_payload(scenario, method, result, args.output, not args.no_record_trace)
        write_json(run_path, payload)
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "numerical_status": payload["numerical_status"],
                    "method": method,
                    "seed": config.seed,
                    "reynolds": config.reynolds,
                    "forcing_wavenumber": config.forcing_wavenumber,
                    "sensor_grid": config.sensor_grid,
                    "obs_interval": config.obs_interval,
                    "nrmse": payload["nrmse"],
                    "crps": payload["crps"],
                    "alpha_mae": payload["alpha_absolute_error"],
                },
                ensure_ascii=False,
            )
        )
    except Exception as error:  # noqa: BLE001
        payload = {
            "run_id": (
                f"kolmogorov64_re{int(config.reynolds)}_k{config.forcing_wavenumber}_"
                f"obs{config.sensor_grid}x{config.sensor_grid}_t{config.obs_interval}_{method}_seed{config.seed}"
            ),
            "status": "failed",
            "case": "kolmogorov64",
            "method": method,
            "seed": config.seed,
            "reynolds": config.reynolds,
            "forcing_wavenumber": config.forcing_wavenumber,
            "sensor_grid": config.sensor_grid,
            "obs_interval": config.obs_interval,
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
