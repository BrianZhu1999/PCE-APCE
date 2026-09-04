from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hilda_da.alpha import liu_quantile
from hilda_da.baselines import denkf_analysis
from hilda_da.math_utils import stable_cholesky
from hilda_da.metrics import weighted_central_interval_coverage_width, weighted_ensemble_crps


MethodName = Literal["aug_enkf", "bma_static", "pce", "apce"]
METHODS: tuple[MethodName, ...] = ("aug_enkf", "bma_static", "pce", "apce")
METHOD_LABELS = {
    "aug_enkf": "Aug-EnKF",
    "bma_static": "BMA",
    "pce": "PCE",
    "apce": "APCE",
}

DEFAULT_DATA = Path(
    "<EXTERNAL_DATA_ROOT>/S3GM/S3GM-main/S3GM-main/data/kolmogorov_flow_test.npy"
)
WINDOW_STARTS = {
    2026081600: 10,
    2026081601: 75,
    2026081602: 140,
    2026081603: 205,
    2026081604: 270,
}
RE_LEVELS = (50.0, 125.0, 575.0, 1100.0, 1500.0)
K_LEVELS = (2, 4, 6, 8)
ALPHA_MIN = 0.08
ALPHA_MAX = 0.92
RE_MIN = 50.0
RE_MAX = 1500.0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(int(seed))
    return generator


def liu_inverse(value: torch.Tensor | float) -> torch.Tensor:
    q = torch.as_tensor(value)
    return 1.0 / (1.0 + torch.exp(-math.pi / math.sqrt(3.0) * q))


def re_alpha_constants() -> tuple[float, float]:
    q0 = float(liu_quantile(torch.tensor(ALPHA_MIN)))
    q1 = float(liu_quantile(torch.tensor(ALPHA_MAX)))
    b = (math.log(RE_MAX) - math.log(RE_MIN)) / (q1 - q0)
    a = math.log(RE_MIN) - b * q0
    return a, b


RE_LOG_A, RE_LOG_B = re_alpha_constants()
ALPHA_TRUE_RE575 = float(liu_inverse(torch.tensor((math.log(575.0) - RE_LOG_A) / RE_LOG_B)))


def alpha_for_re(reynolds: float) -> float:
    if reynolds < RE_MIN or reynolds > RE_MAX:
        raise ValueError(f"Reynolds number must be within [{RE_MIN}, {RE_MAX}], got {reynolds}")
    return float(liu_inverse(torch.tensor((math.log(reynolds) - RE_LOG_A) / RE_LOG_B)))


@dataclass(frozen=True)
class KOL64VelocityConfig:
    seed: int
    sensor_grid: int
    window_start: int
    data_path: str = str(DEFAULT_DATA)
    reynolds: float = 575.0
    forcing_wavenumber: int = 4
    nx: int = 64
    ny: int = 64
    channels: int = 2
    steps: int = 58
    observation_interval: int = 1
    save_dt: float = 0.1
    solver_substeps: int = 4
    ensemble_size: int = 64
    obs_noise_fraction: float = 0.02
    initial_spread_fraction: float = 0.03
    process_noise_fraction: float = 0.002
    alpha_true: float = ALPHA_TRUE_RE575
    alpha_min: float = ALPHA_MIN
    alpha_max: float = ALPHA_MAX
    coarse_alpha_grid: tuple[float, ...] = (0.08, 0.22, 0.36, 0.50, 0.64, 0.78, 0.92)
    bma_alpha_grid_size: int = 21
    pce_temperature: float = 0.62
    apce_temperature: float = 0.54
    apce_min_temperature: float = 0.08
    apce_forgetting: float = 0.975
    apce_entropy_floor: float = 0.34
    evidence_shrinkage: float = 0.22
    dimension_weight_floor: float = 0.35
    dimension_weight_gain: float = 0.65
    branch_member_alpha_jitter: float = 0.012
    alpha_random_walk_std: float = 0.004
    branch_augmented_alpha_analysis_strength: float = 0.60
    global_augmented_alpha_analysis_strength: float = 0.25
    global_analysis_strength: float = 1.0
    local_grid_points: int = 11
    local_grid_radius: float = 0.18
    local_grid_min_spacing: float = 0.012
    localization_scale: float = 4.0
    max_valid_amplitude_ratio: float = 20.0

    @property
    def state_dim(self) -> int:
        return self.channels * self.nx * self.ny

    @property
    def observed_points(self) -> int:
        return self.channels * self.sensor_grid * self.sensor_grid


@dataclass(frozen=True)
class SharedAssets:
    config: KOL64VelocityConfig
    truth: torch.Tensor
    observation_noise: torch.Tensor
    initial_noise: torch.Tensor
    forecast_noise: torch.Tensor
    sensor_indices: torch.Tensor
    sensor_spatial_indices: torch.Tensor
    velocity_rms: float
    asset_path: Path
    asset_sha256: str


@dataclass(frozen=True)
class Scenario:
    config: KOL64VelocityConfig
    truth: torch.Tensor
    observations: dict[int, torch.Tensor]
    initial_ensemble: torch.Tensor
    forecast_noise: torch.Tensor
    sensor_indices: torch.Tensor
    sensor_spatial_indices: torch.Tensor
    localization: torch.Tensor
    augmented_localization: torch.Tensor
    obs_sigma: float
    velocity_rms: float
    asset_path: Path
    asset_sha256: str


def fft_wavenumbers(config: KOL64VelocityConfig, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kx = 2.0 * math.pi * torch.fft.fftfreq(config.nx, d=2.0 * math.pi / config.nx, device=device, dtype=dtype)
    ky = 2.0 * math.pi * torch.fft.fftfreq(config.ny, d=2.0 * math.pi / config.ny, device=device, dtype=dtype)
    kx_grid, ky_grid = torch.meshgrid(kx, ky, indexing="ij")
    return kx_grid, ky_grid, kx_grid.square() + ky_grid.square()


def dealias_mask(config: KOL64VelocityConfig, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mx = torch.fft.fftfreq(config.nx, device=device).abs() * config.nx
    my = torch.fft.fftfreq(config.ny, device=device).abs() * config.ny
    return ((mx[:, None] <= config.nx // 3) & (my[None, :] <= config.ny // 3)).to(dtype)


class KolmogorovVelocitySystem:
    """Pseudo-spectral velocity-form Kolmogorov flow with periodic projection."""

    def __init__(self, config: KOL64VelocityConfig, device: torch.device, dtype: torch.dtype = torch.float64) -> None:
        self.config = config
        self.device = device
        self.dtype = dtype
        self.kx, self.ky, self.k2 = fft_wavenumbers(config, device, dtype)
        self.mask = dealias_mask(config, device, dtype)
        self.inv_k2 = torch.where(self.k2 > 0, self.k2.reciprocal(), torch.zeros_like(self.k2))
        y = torch.arange(config.ny, dtype=dtype, device=device) * (2.0 * math.pi / config.ny)
        self.forcing = torch.zeros((2, config.nx, config.ny), dtype=dtype, device=device)
        self.forcing[0] = torch.sin(config.forcing_wavenumber * y)[None, :]

    def _reshape(self, state: torch.Tensor) -> torch.Tensor:
        return state.reshape(*state.shape[:-1], 2, self.config.nx, self.config.ny)

    def _flatten(self, state: torch.Tensor) -> torch.Tensor:
        return state.reshape(*state.shape[:-3], self.config.state_dim)

    def project(self, state: torch.Tensor) -> torch.Tensor:
        field = self._reshape(torch.nan_to_num(state, nan=0.0, posinf=20.0, neginf=-20.0))
        u_hat = torch.fft.fft2(field[..., 0, :, :], dim=(-2, -1))
        v_hat = torch.fft.fft2(field[..., 1, :, :], dim=(-2, -1))
        divergence = self.kx * u_hat + self.ky * v_hat
        u_hat = u_hat - self.kx * divergence * self.inv_k2
        v_hat = v_hat - self.ky * divergence * self.inv_k2
        projected = torch.stack((torch.fft.ifft2(u_hat, dim=(-2, -1)).real, torch.fft.ifft2(v_hat, dim=(-2, -1)).real), dim=-3)
        return self._flatten(projected).clamp(-20.0, 20.0)

    def reynolds_from_alpha(self, alpha: torch.Tensor | float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        q = liu_quantile(torch.as_tensor(alpha, dtype=dtype, device=device))
        return torch.exp(RE_LOG_A + RE_LOG_B * q)

    def rhs(self, state: torch.Tensor, alpha: torch.Tensor | float) -> torch.Tensor:
        field = self._reshape(state)
        u_hat = torch.fft.fft2(field[..., 0, :, :], dim=(-2, -1))
        v_hat = torch.fft.fft2(field[..., 1, :, :], dim=(-2, -1))
        u_hat = u_hat * self.mask
        v_hat = v_hat * self.mask
        divergence = self.kx * u_hat + self.ky * v_hat
        u_proj = u_hat - self.kx * divergence * self.inv_k2
        v_proj = v_hat - self.ky * divergence * self.inv_k2
        u = torch.fft.ifft2(u_proj, dim=(-2, -1)).real
        v = torch.fft.ifft2(v_proj, dim=(-2, -1)).real
        ux = torch.fft.ifft2(1j * self.kx * u_hat, dim=(-2, -1)).real
        uy = torch.fft.ifft2(1j * self.ky * u_hat, dim=(-2, -1)).real
        vx = torch.fft.ifft2(1j * self.kx * v_hat, dim=(-2, -1)).real
        vy = torch.fft.ifft2(1j * self.ky * v_hat, dim=(-2, -1)).real
        nonlinear_u = torch.fft.fft2(-(u * ux + v * uy), dim=(-2, -1)) * self.mask
        nonlinear_v = torch.fft.fft2(-(u * vx + v * vy), dim=(-2, -1)) * self.mask
        nonlinear_div = self.kx * nonlinear_u + self.ky * nonlinear_v
        nonlinear_u = nonlinear_u - self.kx * nonlinear_div * self.inv_k2
        nonlinear_v = nonlinear_v - self.ky * nonlinear_div * self.inv_k2
        reynolds = self.reynolds_from_alpha(alpha, state.dtype, state.device)
        while reynolds.ndim < state.ndim + 1:
            reynolds = reynolds.unsqueeze(-1)
        viscosity = reynolds.reciprocal()
        lap_u = -self.k2 * u_hat
        lap_v = -self.k2 * v_hat
        rhs_u = nonlinear_u + viscosity * lap_u - 0.1 * u_hat + torch.fft.fft2(self.forcing[0], dim=(-2, -1))
        rhs_v = nonlinear_v + viscosity * lap_v - 0.1 * v_hat
        out = torch.stack((torch.fft.ifft2(rhs_u, dim=(-2, -1)).real, torch.fft.ifft2(rhs_v, dim=(-2, -1)).real), dim=-3)
        return self._flatten(out)

    def step(self, state: torch.Tensor, alpha: torch.Tensor | float, noise: torch.Tensor | None = None) -> torch.Tensor:
        h = self.config.save_dt / self.config.solver_substeps
        result = state
        for _ in range(self.config.solver_substeps):
            k1 = self.rhs(result, alpha)
            k2 = self.rhs(result + 0.5 * h * k1, alpha)
            k3 = self.rhs(result + 0.5 * h * k2, alpha)
            k4 = self.rhs(result + h * k3, alpha)
            result = result + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            result = self.project(result)
        if noise is not None:
            result = self.project(result + self.config.process_noise_fraction * noise)
        return result


def load_truth(config: KOL64VelocityConfig) -> np.ndarray:
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
    re_index = int(np.argmin([abs(v - config.reynolds) for v in RE_LEVELS]))
    k_index = int(np.argmin([abs(v - config.forcing_wavenumber) for v in K_LEVELS]))
    if abs(RE_LEVELS[re_index] - config.reynolds) > 1e-8 or K_LEVELS[k_index] != config.forcing_wavenumber:
        raise ValueError("The smoke protocol uses an unsupported Re or forcing wavenumber")
    stop = config.window_start + config.steps + 1
    if config.window_start < 0 or stop > 400:
        raise ValueError(f"window [{config.window_start}, {stop}) is outside the 400-frame trajectory")
    segment = np.asarray(data[re_index, k_index, config.window_start:stop], dtype=np.float64)
    if segment.shape[-1] != 2:
        raise ValueError("KOL source must contain two velocity components")
    return segment


def sensor_indices(config: KOL64VelocityConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if config.nx % config.sensor_grid or config.ny % config.sensor_grid:
        raise ValueError("sensor_grid must evenly divide 64")
    stride_x = config.nx // config.sensor_grid
    stride_y = config.ny // config.sensor_grid
    xs = torch.arange(0, config.nx, stride_x, dtype=torch.int64, device=device)
    ys = torch.arange(0, config.ny, stride_y, dtype=torch.int64, device=device)
    xx, yy = torch.meshgrid(xs, ys, indexing="ij")
    spatial = (xx * config.ny + yy).reshape(-1)
    indices = torch.cat((spatial, config.nx * config.ny + spatial))
    return indices, spatial


def periodic_resize(field: torch.Tensor, target_x: int, target_y: int) -> torch.Tensor:
    source_x, source_y = field.shape[-2:]
    spectrum = torch.fft.fftshift(torch.fft.fft2(field, dim=(-2, -1)), dim=(-2, -1))
    output = torch.zeros((*field.shape[:-2], target_x, target_y), dtype=spectrum.dtype, device=field.device)
    left_x = (target_x - source_x) // 2
    left_y = (target_y - source_y) // 2
    output[..., left_x:left_x + source_x, left_y:left_y + source_y] = spectrum
    return torch.fft.ifft2(torch.fft.ifftshift(output, dim=(-2, -1)), dim=(-2, -1)).real * (target_x * target_y) / (source_x * source_y)


def make_divergence_free_noise(config: KOL64VelocityConfig, raw: torch.Tensor, system: KolmogorovVelocitySystem) -> torch.Tensor:
    noise = raw.reshape(*raw.shape[:-1], 2, config.nx, config.ny)
    noise_hat = torch.fft.fft2(noise, dim=(-2, -1)) * system.mask
    noise = torch.fft.ifft2(noise_hat, dim=(-2, -1)).real
    noise = system.project(noise.reshape(*noise.shape[:-3], config.state_dim)).reshape_as(noise)
    scale = noise.square().mean(dim=(-3, -2, -1), keepdim=True).sqrt().clamp_min(1e-12)
    return (noise / scale).reshape_as(raw)


def asset_path(asset_root: Path, config: KOL64VelocityConfig) -> Path:
    return asset_root / "kolmogorov64_velocityobs" / f"re{int(config.reynolds)}_k{config.forcing_wavenumber}" / f"s{config.sensor_grid}_window{config.window_start}_seed{config.seed}.npz"


def create_shared_assets(config: KOL64VelocityConfig, asset_root: Path, device: torch.device) -> Path:
    path = asset_path(asset_root, config)
    if path.exists():
        return path
    generator = make_generator(device, config.seed)
    segment_np = load_truth(config)
    truth_field = torch.as_tensor(segment_np, dtype=torch.float64, device=device).permute(0, 3, 1, 2)
    truth = torch.cat((truth_field[:, 0].reshape(config.steps + 1, -1), truth_field[:, 1].reshape(config.steps + 1, -1)), dim=-1)
    indices, spatial = sensor_indices(config, device)
    velocity_rms = float(truth.square().mean().sqrt().detach().cpu())
    obs_noise_field = torch.randn((config.steps + 1, 2, config.nx, config.ny), dtype=torch.float64, device=device, generator=generator)
    initial_noise = torch.randn((config.ensemble_size, config.state_dim), dtype=torch.float64, device=device, generator=generator)
    forecast_noise = torch.randn((config.steps, config.ensemble_size, config.state_dim), dtype=torch.float64, device=device, generator=generator)
    system = KolmogorovVelocitySystem(config, device)
    initial_noise = make_divergence_free_noise(config, initial_noise, system)
    forecast_noise = make_divergence_free_noise(config, forecast_noise.reshape(-1, config.state_dim), system).reshape_as(forecast_noise)
    forecast_noise = velocity_rms * forecast_noise
    initial_observation = truth[0, indices] + config.obs_noise_fraction * velocity_rms * obs_noise_field[0].reshape(config.state_dim)[indices]
    sensor_values = torch.stack((initial_observation[: spatial.numel()], initial_observation[spatial.numel():]), dim=0).reshape(2, config.sensor_grid, config.sensor_grid)
    interpolated = periodic_resize(sensor_values, config.nx, config.ny)
    initial_center = torch.cat((interpolated[0].reshape(-1), interpolated[1].reshape(-1)))
    initial_ensemble = system.project(initial_center.unsqueeze(0) + config.initial_spread_fraction * velocity_rms * initial_noise)
    obs_noise = obs_noise_field.reshape(config.steps + 1, config.state_dim).index_select(-1, indices)
    tmp = path.with_name(f"{path.stem}.tmp_{os.getpid()}.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        tmp,
        truth=truth.detach().cpu().numpy(),
        observation_noise=obs_noise.detach().cpu().numpy(),
        initial_noise=initial_noise.detach().cpu().numpy(),
        forecast_noise=forecast_noise.detach().cpu().numpy(),
        sensor_indices=indices.detach().cpu().numpy(),
        sensor_spatial_indices=spatial.detach().cpu().numpy(),
        velocity_rms=np.asarray(velocity_rms),
        config_json=np.asarray(json.dumps(asdict(config), ensure_ascii=False)),
    )
    tmp.replace(path)
    return path


def load_shared_assets(config: KOL64VelocityConfig, asset_root: Path, device: torch.device) -> SharedAssets:
    path = create_shared_assets(config, asset_root, device)
    for attempt in range(30):
        try:
            with np.load(path, allow_pickle=False) as data:
                stored = json.loads(str(data["config_json"].item()))
                for field in ("seed", "sensor_grid", "window_start", "steps", "solver_substeps", "reynolds", "forcing_wavenumber"):
                    if stored[field] != asdict(config)[field]:
                        raise RuntimeError(f"shared asset mismatch for {field}: {stored[field]} != {asdict(config)[field]}")
                return SharedAssets(
                    config=config,
                    truth=torch.as_tensor(data["truth"], dtype=torch.float64, device=device),
                    observation_noise=torch.as_tensor(data["observation_noise"], dtype=torch.float64, device=device),
                    initial_noise=torch.as_tensor(data["initial_noise"], dtype=torch.float64, device=device),
                    forecast_noise=torch.as_tensor(data["forecast_noise"], dtype=torch.float64, device=device),
                    sensor_indices=torch.as_tensor(data["sensor_indices"], dtype=torch.int64, device=device),
                    sensor_spatial_indices=torch.as_tensor(data["sensor_spatial_indices"], dtype=torch.int64, device=device),
                    velocity_rms=float(np.asarray(data["velocity_rms"]).item()),
                    asset_path=path,
                    asset_sha256=file_sha256(path),
                )
        except (FileNotFoundError, OSError, ValueError):
            if attempt == 29:
                raise
            time.sleep(0.1)
            path = create_shared_assets(config, asset_root, device)
    raise RuntimeError(f"failed to load shared asset after retries: {path}")


def gaspari_cohn(distance_ratio: torch.Tensor) -> torch.Tensor:
    distance = distance_ratio.abs()
    out = torch.zeros_like(distance)
    inner = distance <= 1.0
    middle = (distance > 1.0) & (distance <= 2.0)
    x = distance[inner]
    out[inner] = 1.0 - (5.0 / 3.0) * x.square() + 0.625 * x.pow(3) + 0.5 * x.pow(4) - 0.25 * x.pow(5)
    x = distance[middle]
    out[middle] = 4.0 - 5.0 * x + (5.0 / 3.0) * x.square() + 0.625 * x.pow(3) - 0.5 * x.pow(4) + (1.0 / 12.0) * x.pow(5) - 2.0 / (3.0 * x)
    return out.clamp(0.0, 1.0)


def localization(config: KOL64VelocityConfig, spatial: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    state_spatial = torch.arange(config.nx * config.ny, dtype=torch.int64, device=device)
    sx, sy = state_spatial // config.ny, state_spatial % config.ny
    ox, oy = spatial // config.ny, spatial % config.ny
    dx = (sx[:, None] - ox[None, :]).abs().to(torch.float64)
    dy = (sy[:, None] - oy[None, :]).abs().to(torch.float64)
    dx = torch.minimum(dx, config.nx - dx)
    dy = torch.minimum(dy, config.ny - dy)
    taper = gaspari_cohn(torch.sqrt(dx.square() + dy.square()) / config.localization_scale)
    physical = torch.cat((torch.cat((taper, taper), dim=1), torch.cat((taper, taper), dim=1)), dim=0)
    augmented = torch.cat((physical, torch.ones((1, physical.shape[1]), dtype=torch.float64, device=device)), dim=0)
    return physical, augmented


def materialize(shared: SharedAssets, device: torch.device) -> Scenario:
    config = shared.config
    system = KolmogorovVelocitySystem(config, device)
    sigma = config.obs_noise_fraction * shared.velocity_rms
    observations = {step: shared.truth[step, shared.sensor_indices] + shared.observation_noise[step] * sigma for step in range(config.steps + 1)}
    physical, augmented = localization(config, shared.sensor_spatial_indices, device)
    initial_observation = observations[0]
    s = config.sensor_grid
    values = torch.stack((initial_observation[: s * s], initial_observation[s * s:]), dim=0).reshape(2, s, s)
    initial_center = torch.cat((periodic_resize(values, config.nx, config.ny)[0].reshape(-1), periodic_resize(values, config.nx, config.ny)[1].reshape(-1)))
    initial_ensemble = system.project(initial_center.unsqueeze(0) + config.initial_spread_fraction * shared.velocity_rms * shared.initial_noise)
    return Scenario(config, shared.truth, observations, initial_ensemble, shared.forecast_noise, shared.sensor_indices, shared.sensor_spatial_indices, physical, augmented, sigma, shared.velocity_rms, shared.asset_path, shared.asset_sha256)


def entropy(weights: torch.Tensor) -> torch.Tensor:
    safe = weights.clamp_min(1.0e-300)
    return -(safe * safe.log()).sum()


def entropy_project(weights: torch.Tensor, target: float) -> torch.Tensor:
    weights = weights / weights.sum().clamp_min(1e-300)
    if float(entropy(weights)) >= target:
        return weights
    uniform = torch.full_like(weights, 1.0 / weights.numel())
    low, high = 0.0, 1.0
    for _ in range(45):
        middle = 0.5 * (low + high)
        mixed = (1.0 - middle) * weights + middle * uniform
        if float(entropy(mixed)) < target:
            low = middle
        else:
            high = middle
    return ((1.0 - high) * weights + high * uniform).clamp_min(1e-300).div(((1.0 - high) * weights + high * uniform).sum())


def covariance(scenario: Scenario) -> torch.Tensor:
    return scenario.obs_sigma**2 * torch.eye(scenario.config.observed_points, dtype=torch.float64, device=scenario.truth.device)


def weighted_denkf(state: torch.Tensor, weights: torch.Tensor, observation: torch.Tensor, operator: Any, obs_cov: torch.Tensor, loc: torch.Tensor) -> torch.Tensor:
    weights = weights.to(state).clamp_min(1e-300)
    weights = weights / weights.sum()
    predicted = operator(state)
    mean = (weights[:, None] * state).sum(dim=0)
    zmean = (weights[:, None] * predicted).sum(dim=0)
    xa = state - mean
    za = predicted - zmean
    denom = (1.0 - weights.square().sum()).clamp_min(torch.finfo(state.dtype).eps)
    cross = xa.mT @ (weights[:, None] * za) / denom
    innov = za.mT @ (weights[:, None] * za) / denom + obs_cov
    factor = stable_cholesky(innov)
    gain = torch.cholesky_solve(cross.mT, factor).mT * loc
    updated_mean = mean + gain @ (observation - zmean)
    updated_anom = xa - 0.5 * (za @ gain.mT)
    return updated_mean.unsqueeze(0) + updated_anom


def alpha_aug_analysis(state: torch.Tensor, alpha: torch.Tensor, obs: torch.Tensor, operator: Any, cov: torch.Tensor, config: KOL64VelocityConfig, loc: torch.Tensor, system: KolmogorovVelocitySystem) -> tuple[torch.Tensor, torch.Tensor]:
    augmented = torch.cat((state, alpha[:, None]), dim=-1)
    updated = denkf_analysis(augmented, obs, operator, cov, localization=loc)
    return system.project(updated[:, :-1]), updated[:, -1].clamp(config.alpha_min, config.alpha_max)


class VelocityMetrics:
    def __init__(self) -> None:
        self.se = 0.0
        self.te = 0.0
        self.points = 0
        self.crps: list[float] = []
        self.coverage: list[float] = []
        self.width: list[float] = []

    def add(self, ensemble: torch.Tensor, truth: torch.Tensor, weights: torch.Tensor, estimate: torch.Tensor) -> None:
        w = weights / weights.sum().clamp_min(1e-300)
        self.se += float((estimate - truth).square().sum().detach().cpu())
        self.te += float(truth.square().sum().detach().cpu())
        self.points += int(truth.numel())
        self.crps.append(float(weighted_ensemble_crps(ensemble, truth, w).detach().cpu()))
        cov, width = weighted_central_interval_coverage_width(ensemble, truth, w, level=0.90)
        self.coverage.append(float(cov.detach().cpu()))
        self.width.append(float(width.detach().cpu()))

    def finalize(self) -> dict[str, float]:
        return {
            "nrmse": math.sqrt(self.se / max(self.te, 1e-30)),
            "rmse": math.sqrt(self.se / max(self.points, 1)),
            "crps": float(np.mean(self.crps)),
            "coverage_90": float(np.mean(self.coverage)),
            "coverage_90_error": abs(float(np.mean(self.coverage)) - 0.90),
            "interval_width_90": float(np.mean(self.width)),
        }


def evidence_score(predicted: torch.Tensor, observation: torch.Tensor, config: KOL64VelocityConfig, obs_sigma: float, dimension_weights: torch.Tensor | None) -> torch.Tensor:
    mean = predicted.mean(dim=0)
    residual = observation - mean
    anomalies = predicted - mean
    matrix = anomalies.mT @ anomalies / max(predicted.shape[0] - 1, 1)
    matrix = (1.0 - config.evidence_shrinkage) * matrix + config.evidence_shrinkage * torch.diag(torch.diagonal(matrix))
    matrix = matrix + (obs_sigma**2 + 1e-8) * torch.eye(observation.numel(), dtype=observation.dtype, device=observation.device)
    if dimension_weights is not None:
        weights = observation.numel() * dimension_weights / dimension_weights.sum().clamp_min(1e-12)
        variance = torch.diagonal(matrix).clamp_min(1e-12)
        marginal = residual.square() / variance + variance.log() + math.log(2.0 * math.pi)
        return -0.5 * (weights * marginal).sum()
    factor = stable_cholesky(matrix)
    solved = torch.cholesky_solve(residual[:, None], factor).squeeze(-1)
    logdet = 2.0 * torch.log(torch.diagonal(factor)).sum()
    return -0.5 * (residual @ solved + logdet + observation.numel() * math.log(2.0 * math.pi))


def interpolate_paths(old_grid: torch.Tensor, values: torch.Tensor, new_grid: torch.Tensor) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for alpha in new_grid:
        if float(alpha) <= float(old_grid[0]):
            rows.append(values[0])
        elif float(alpha) >= float(old_grid[-1]):
            rows.append(values[-1])
        else:
            right = int(torch.searchsorted(old_grid, alpha).detach().cpu())
            left = right - 1
            frac = (alpha - old_grid[left]) / (old_grid[right] - old_grid[left])
            rows.append((1.0 - frac) * values[left] + frac * values[right])
    return torch.stack(rows)


def local_grid(grid: torch.Tensor, scores: torch.Tensor, config: KOL64VelocityConfig) -> torch.Tensor:
    center = grid[int(torch.argmax(scores))]
    left = max(config.alpha_min, float(center) - config.local_grid_radius)
    right = min(config.alpha_max, float(center) + config.local_grid_radius)
    result = torch.linspace(left, right, config.local_grid_points, dtype=grid.dtype, device=grid.device)
    result[int(torch.argmin((result - center).abs()))] = center
    return result


def update_metrics(metrics: VelocityMetrics, ensemble: torch.Tensor, truth: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    estimate = (weights[:, None] * ensemble).sum(dim=0)
    metrics.add(ensemble, truth, weights, estimate)
    return estimate


def start_runtime(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    return time.perf_counter()


def finish_runtime(started: float, device: torch.device) -> tuple[float, float]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return time.perf_counter() - started, torch.cuda.max_memory_allocated(device) / 1024**2
    return time.perf_counter() - started, 0.0


def run_aug_enkf(scenario: Scenario, record_trace: bool) -> dict[str, Any]:
    cfg, device = scenario.config, scenario.truth.device
    system = KolmogorovVelocitySystem(cfg, device)
    operator = lambda state: state.index_select(-1, scenario.sensor_indices)
    cov = covariance(scenario)
    gen = make_generator(device, cfg.seed + 401)
    alpha = torch.linspace(cfg.alpha_min, cfg.alpha_max, cfg.ensemble_size, dtype=torch.float64, device=device)
    alpha = (alpha + 0.02 * torch.randn(alpha.shape, dtype=alpha.dtype, device=device, generator=gen)).clamp(cfg.alpha_min, cfg.alpha_max)
    state = scenario.initial_ensemble.clone()
    weights = torch.full((cfg.ensemble_size,), 1.0 / cfg.ensemble_size, dtype=torch.float64, device=device)
    metrics = VelocityMetrics(); trace: list[torch.Tensor] = []; alpha_hist: list[float] = []
    started = start_runtime(device)
    for step in range(cfg.steps + 1):
        if step % cfg.observation_interval == 0:
            updated = denkf_analysis(torch.cat((state, alpha[:, None]), dim=-1), scenario.observations[step], operator, cov, localization=scenario.augmented_localization)
            state = system.project(updated[:, :-1])
            alpha = updated[:, -1].clamp(cfg.alpha_min, cfg.alpha_max)
        estimate = update_metrics(metrics, state, scenario.truth[step], weights)
        if record_trace: trace.append(estimate.detach().cpu()); alpha_hist.append(float(alpha.mean().detach().cpu()))
        if step == cfg.steps: break
        alpha = (alpha + cfg.alpha_random_walk_std * torch.randn(alpha.shape, dtype=alpha.dtype, device=device, generator=gen)).clamp(cfg.alpha_min, cfg.alpha_max)
        state = system.step(state, alpha, scenario.forecast_noise[step])
    elapsed, memory = finish_runtime(started, device)
    result = metrics.finalize(); alpha_est = float(alpha.mean().detach().cpu())
    result.update(runtime_seconds=elapsed, peak_gpu_memory_mb=memory, alpha_estimate=alpha_est, alpha_absolute_error=abs(alpha_est - cfg.alpha_true), reynolds_estimate=float(system.reynolds_from_alpha(alpha.mean(), torch.float64, device).detach().cpu()), max_abs_state=float(state.abs().max().detach().cpu()))
    if record_trace: result.update(mean_states=torch.stack(trace).numpy(), alpha_mean_history=np.asarray(alpha_hist))
    return result


def run_bma(scenario: Scenario, record_trace: bool) -> dict[str, Any]:
    cfg, device = scenario.config, scenario.truth.device
    system = KolmogorovVelocitySystem(cfg, device); operator = lambda state: state.index_select(-1, scenario.sensor_indices); cov = covariance(scenario)
    grid = torch.linspace(cfg.alpha_min, cfg.alpha_max, cfg.bma_alpha_grid_size, dtype=torch.float64, device=device)
    branches = scenario.initial_ensemble.unsqueeze(0).repeat(grid.numel(), 1, 1); shadow = branches.clone()
    logw = torch.zeros(grid.numel(), dtype=torch.float64, device=device); metrics = VelocityMetrics(); trace: list[torch.Tensor] = []; w_hist: list[torch.Tensor] = []
    started = start_runtime(device)
    for step in range(cfg.steps + 1):
        if step % cfg.observation_interval == 0:
            predicted = torch.stack([operator(shadow[i]) for i in range(grid.numel())])
            score = torch.stack([evidence_score(predicted[i], scenario.observations[step], cfg, scenario.obs_sigma, None) for i in range(grid.numel())])
            logw = logw + score - score.mean(); weights = torch.softmax(logw, dim=0)
            for i in range(grid.numel()): branches[i] = system.project(denkf_analysis(branches[i], scenario.observations[step], operator, cov, localization=scenario.localization))
        flat = branches.reshape(-1, cfg.state_dim); flatw = weights[:, None].expand(-1, cfg.ensemble_size).reshape(-1) / cfg.ensemble_size
        estimate = update_metrics(metrics, flat, scenario.truth[step], flatw)
        if record_trace: trace.append(estimate.detach().cpu()); w_hist.append(weights.detach().cpu())
        if step == cfg.steps: break
        branches = system.step(branches, grid[:, None], scenario.forecast_noise[step].unsqueeze(0).expand(grid.numel(), -1, -1))
        shadow = system.step(shadow, grid[:, None], scenario.forecast_noise[step].unsqueeze(0).expand(grid.numel(), -1, -1))
    elapsed, memory = finish_runtime(started, device); alpha_est = float((weights * grid).sum().detach().cpu())
    result = metrics.finalize(); result.update(runtime_seconds=elapsed, peak_gpu_memory_mb=memory, alpha_estimate=alpha_est, alpha_absolute_error=abs(alpha_est - cfg.alpha_true), reynolds_estimate=float(system.reynolds_from_alpha(torch.tensor(alpha_est, device=device), torch.float64, device).detach().cpu()), alpha_final_entropy=float(entropy(weights).detach().cpu()), max_abs_state=float(branches.abs().max().detach().cpu()))
    if record_trace: result.update(mean_states=torch.stack(trace).numpy(), alpha_weight_history=np.stack(w_hist))
    return result


def run_pce_apce(scenario: Scenario, method: Literal["pce", "apce"], record_trace: bool) -> dict[str, Any]:
    cfg, device = scenario.config, scenario.truth.device
    system = KolmogorovVelocitySystem(cfg, device); operator = lambda state: state.index_select(-1, scenario.sensor_indices); cov = covariance(scenario)
    grid = torch.as_tensor(cfg.coarse_alpha_grid, dtype=torch.float64, device=device); bounds = (cfg.alpha_min, cfg.alpha_max); paths = grid.numel()
    branches = scenario.initial_ensemble.unsqueeze(0).repeat(paths, 1, 1); shadow = branches.clone()
    gen = make_generator(device, cfg.seed + (701 if method == "pce" else 1701))
    alpha_members = grid[:, None].expand(-1, cfg.ensemble_size).clone() + cfg.branch_member_alpha_jitter * torch.randn((paths, cfg.ensemble_size), dtype=torch.float64, device=device, generator=gen)
    alpha_members = alpha_members.clamp(cfg.alpha_min, cfg.alpha_max); log_scores = torch.zeros(paths, dtype=torch.float64, device=device)
    alpha_weights = torch.softmax(log_scores, dim=0); state_weights = alpha_weights.clone(); metrics = VelocityMetrics(); trace: list[torch.Tensor] = []; whist: list[torch.Tensor] = []; ghist: list[torch.Tensor] = []; ahist: list[float] = []; regrid_count = 0
    started = start_runtime(device)
    for step in range(cfg.steps + 1):
        flat = branches.reshape(-1, cfg.state_dim); flatw = state_weights[:, None].expand(-1, cfg.ensemble_size).reshape(-1) / cfg.ensemble_size
        estimate = update_metrics(metrics, flat, scenario.truth[step], flatw)
        if record_trace: trace.append(estimate.detach().cpu()); whist.append(state_weights.detach().cpu()); ghist.append(grid.detach().cpu()); ahist.append(float((alpha_weights * grid).sum().detach().cpu()))
        if step == cfg.steps: break
        branches = torch.stack([system.step(branches[i], alpha_members[i], scenario.forecast_noise[step]) for i in range(grid.numel())])
        shadow = torch.stack([system.step(shadow[i], alpha_members[i], scenario.forecast_noise[step]) for i in range(grid.numel())])
        if (step + 1) % cfg.observation_interval == 0:
            observation = scenario.observations[step + 1]
            shadow_pred = torch.stack([operator(shadow[i]) for i in range(grid.numel())])
            if method == "apce":
                between = shadow_pred.mean(dim=1).var(dim=0, unbiased=True)
                dimw = cfg.dimension_weight_floor + cfg.dimension_weight_gain * between / between.max().clamp_min(1e-12)
            else:
                dimw = None
            evidence = torch.stack([evidence_score(shadow_pred[i], observation, cfg, scenario.obs_sigma, dimw) for i in range(grid.numel())]); centered = evidence - evidence.mean()
            if method == "pce":
                log_scores = log_scores + cfg.pce_temperature * centered
                alpha_weights = torch.softmax(log_scores, dim=0)
            else:
                temperature = max(cfg.apce_min_temperature, cfg.apce_temperature * float((entropy(alpha_weights) / math.log(grid.numel())) ** 0.75))
                log_scores = cfg.apce_forgetting * log_scores + temperature * centered
                alpha_weights = torch.softmax(log_scores, dim=0); alpha_weights = entropy_project(alpha_weights, cfg.apce_entropy_floor)
            state_weights = alpha_weights.clone()
            refined = local_grid(grid, log_scores, cfg)
            if refined.shape != grid.shape or not torch.allclose(refined, grid):
                branches = interpolate_paths(grid, branches, refined); shadow = interpolate_paths(grid, shadow, refined); alpha_members = interpolate_paths(grid, alpha_members, refined); log_scores = interpolate_paths(grid, log_scores, refined); grid = refined; alpha_weights = torch.softmax(log_scores, dim=0); state_weights = entropy_project(alpha_weights, cfg.apce_entropy_floor) if method == "apce" else alpha_weights; regrid_count += 1
            forecast_branches = branches
            local_branches = torch.empty_like(forecast_branches)
            for i in range(grid.numel()):
                local_branches[i] = system.project(denkf_analysis(forecast_branches[i], observation, operator, cov, localization=scenario.localization))
            flat = forecast_branches.reshape(-1, cfg.state_dim); flatw = state_weights[:, None].expand(-1, cfg.ensemble_size).reshape(-1) / cfg.ensemble_size
            if cfg.global_analysis_strength > 0:
                global_state = weighted_denkf(flat, flatw, observation, operator, cov, scenario.localization).reshape_as(branches)
                branches = (1.0 - cfg.global_analysis_strength) * local_branches + cfg.global_analysis_strength * global_state
            else:
                branches = local_branches
            branches = system.project(branches)
            if cfg.branch_augmented_alpha_analysis_strength > 0:
                joint_states = torch.empty_like(branches); joint_alpha = torch.empty_like(alpha_members)
                aug_loc = scenario.augmented_localization
                for i in range(grid.numel()): joint_states[i], joint_alpha[i] = alpha_aug_analysis(branches[i], alpha_members[i], observation, operator, cov, cfg, aug_loc, system)
                s = cfg.branch_augmented_alpha_analysis_strength; branches = (1.0 - s) * branches + s * joint_states; alpha_members = (1.0 - s) * alpha_members + s * joint_alpha
            if cfg.global_augmented_alpha_analysis_strength > 0:
                flat = branches.reshape(-1, cfg.state_dim); flat_alpha = alpha_members.reshape(-1); flatw = state_weights[:, None].expand(-1, cfg.ensemble_size).reshape(-1) / cfg.ensemble_size
                joint = weighted_denkf(torch.cat((flat, flat_alpha[:, None]), dim=-1), flatw, observation, operator, cov, scenario.augmented_localization)
                s = cfg.global_augmented_alpha_analysis_strength; branches = (1.0 - s) * branches + s * system.project(joint[:, :-1].reshape_as(branches)); alpha_members = (1.0 - s) * alpha_members + s * joint[:, -1].reshape_as(alpha_members).clamp(cfg.alpha_min, cfg.alpha_max)
            branches = system.project(branches); alpha_members = alpha_members.clamp(cfg.alpha_min, cfg.alpha_max)
    elapsed, memory = finish_runtime(started, device); alpha_est = float((state_weights * grid).sum().detach().cpu()); result = metrics.finalize(); result.update(runtime_seconds=elapsed, peak_gpu_memory_mb=memory, alpha_estimate=alpha_est, alpha_absolute_error=abs(alpha_est - cfg.alpha_true), reynolds_estimate=float(system.reynolds_from_alpha(torch.tensor(alpha_est, device=device), torch.float64, device).detach().cpu()), alpha_final_entropy=float(entropy(state_weights).detach().cpu()), alpha_evidence_entropy=float(entropy(alpha_weights).detach().cpu()), alpha_regrid_count=regrid_count, final_grid_points=int(grid.numel()), max_abs_state=float(branches.abs().max().detach().cpu()))
    if record_trace:
        result.update(
            mean_states=torch.stack(trace).numpy(),
            alpha_weight_history=np.asarray([item.numpy() for item in whist], dtype=object),
            alpha_grid_history=np.asarray([item.numpy() for item in ghist], dtype=object),
            alpha_estimate_history=np.asarray(ahist),
        )
    return result


def run_method(scenario: Scenario, method: MethodName, record_trace: bool) -> dict[str, Any]:
    if method == "aug_enkf": return run_aug_enkf(scenario, record_trace)
    if method == "bma_static": return run_bma(scenario, record_trace)
    return run_pce_apce(scenario, method, record_trace)


def divergence_rms(state: torch.Tensor, config: KOL64VelocityConfig, device: torch.device) -> float:
    system = KolmogorovVelocitySystem(config, device); field = system._reshape(state); uh = torch.fft.fft2(field[..., 0], dim=(-2, -1)); vh = torch.fft.fft2(field[..., 1], dim=(-2, -1)); div = torch.fft.ifft2(1j * system.kx * uh + 1j * system.ky * vh, dim=(-2, -1)).real; return float(div.square().mean().sqrt().detach().cpu())


def numerical_status(result: dict[str, Any], scenario: Scenario) -> tuple[str, str]:
    required = ("nrmse", "crps", "coverage_90", "interval_width_90", "max_abs_state")
    if not all(math.isfinite(float(result.get(k, float("nan")))) for k in required): return "nonfinite", "nonfinite_metric"
    truth_amp = float(scenario.truth.abs().max().detach().cpu())
    if truth_amp > 0 and float(result["max_abs_state"]) > scenario.config.max_valid_amplitude_ratio * truth_amp: return "diverged", "amplitude_divergence"
    return "valid", ""


def trace_path(output: Path, config: KOL64VelocityConfig, method: MethodName) -> Path:
    return output / "artifacts" / "method_traces" / f"sensor{config.sensor_grid}" / method / f"seed_{config.seed}.npz"


def save_trace(output: Path, scenario: Scenario, method: MethodName, result: dict[str, Any]) -> str:
    path = trace_path(output, scenario.config, method); path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"truth": scenario.truth.detach().cpu().numpy(), "observations": np.stack([scenario.observations[i].detach().cpu().numpy() for i in range(scenario.config.steps + 1)]), "sensor_indices": scenario.sensor_indices.detach().cpu().numpy(), "sensor_spatial_indices": scenario.sensor_spatial_indices.detach().cpu().numpy(), "times": np.arange(scenario.config.steps + 1) * scenario.config.save_dt, "alpha_true": np.asarray(scenario.config.alpha_true)}
    for key in ("mean_states", "alpha_mean_history", "alpha_weight_history", "alpha_grid_history", "alpha_estimate_history"):
        if key in result: payload[key] = result[key]
    np.savez_compressed(path, **payload); return str(path)


def run_json_path(output: Path, config: KOL64VelocityConfig, method: MethodName) -> Path:
    return output / "artifacts" / "run_json" / f"sensor{config.sensor_grid}" / method / f"seed_{config.seed}.json"


def run_one(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cuda": torch.cuda.set_device(device)
    cfg = KOL64VelocityConfig(
        seed=args.seed,
        sensor_grid=args.sensor_grid,
        window_start=args.window_start,
        data_path=str(args.data_path),
        reynolds=args.reynolds,
        forcing_wavenumber=args.forcing_wavenumber,
        observation_interval=args.observation_interval,
        alpha_true=alpha_for_re(args.reynolds),
    )
    asset_root = args.output / "artifacts" / "common_assets"
    shared = load_shared_assets(cfg, asset_root, device); scenario = materialize(shared, device)
    started_total = time.perf_counter(); status = "completed"; numerical = "valid"; error_type = ""; error_text = ""; result: dict[str, Any] = {}; trace = ""
    try:
        result = run_method(scenario, args.method, args.record_trace)
        if "reynolds_estimate" in result:
            result["reynolds_relative_error"] = abs(float(result["reynolds_estimate"]) - cfg.reynolds) / cfg.reynolds
        numerical, error_type = numerical_status(result, scenario)
        if args.record_trace: trace = save_trace(args.output, scenario, args.method, result)
    except Exception as exc:
        status = "failed"; numerical = "failed"; error_type = type(exc).__name__; error_text = str(exc); traceback.print_exc()
    payload: dict[str, Any] = {"run_id": f"kolmogorov64_velocityobs_re{int(cfg.reynolds)}_k{cfg.forcing_wavenumber}_sensor{cfg.sensor_grid}_t{cfg.observation_interval}_{args.method}_seed{cfg.seed}", "status": status, "numerical_status": numerical, "error_type": error_type, "error": error_text, "legacy_status": "old_vorticity_observation_runner_invalid", "protocol_version": "kolmogorov64_velocity_observation_smoke_v2_weighted_aug_temporal_sweep", "case": "kolmogorov64_velocityobs", "method": args.method, "label": METHOD_LABELS[args.method], "seed": cfg.seed, "window_start": cfg.window_start, "window_end": cfg.window_start + cfg.steps, "reynolds": cfg.reynolds, "forcing_wavenumber": cfg.forcing_wavenumber, "alpha_true": cfg.alpha_true, "sensor_grid": cfg.sensor_grid, "sensor_count_locations": cfg.sensor_grid * cfg.sensor_grid, "observed_points_uv": cfg.observed_points, "spatial_downsampling_factor_per_axis": cfg.nx // cfg.sensor_grid, "obs_interval": cfg.observation_interval, "steps": cfg.steps, "save_dt": cfg.save_dt, "solver_substeps": cfg.solver_substeps, "state_dim": cfg.state_dim, "obs_noise_fraction": cfg.obs_noise_fraction, "observation_noise_definition": "independent Gaussian velocity-component noise scaled by window velocity RMS", "initial_spread_fraction": cfg.initial_spread_fraction, "process_noise_fraction": cfg.process_noise_fraction, "state_process_noise_definition": "2/3-bandlimited divergence-free Gaussian field scaled by window velocity RMS", "pce_apce_core": "shadow-analysis separation; parallel local/global weighted analysis; branch-local augmented alpha; global weighted augmented alpha; APCE entropy calibration", "re_log_mapping": {"Re_min": RE_MIN, "Re_max": RE_MAX, "alpha_min": ALPHA_MIN, "alpha_max": ALPHA_MAX, "log_a": RE_LOG_A, "log_b": RE_LOG_B}, "alpha_parameterization": "global Liu-quantile log-Re map fixed by Re=50 and Re=1500", "velocity_rms": scenario.velocity_rms, "obs_sigma": scenario.obs_sigma, "data_path": cfg.data_path, "asset_npz": str(scenario.asset_path), "asset_sha256": scenario.asset_sha256, "runner": str(Path(__file__).resolve()), "runner_sha256": file_sha256(Path(__file__).resolve()), "trace_npz": trace, "runtime_total_seconds": time.perf_counter() - started_total, **{k: v for k, v in result.items() if isinstance(v, (int, float, str))}}
    path = run_json_path(args.output, cfg, args.method); write_json(path, clean_json(payload)); print(json.dumps(clean_json(payload), ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Figure 4 KOL-64 sparse velocity-observation worker")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sensor-grid", type=int, choices=(8, 16), required=True)
    parser.add_argument("--window-start", type=int, required=True)
    parser.add_argument("--reynolds", type=float, choices=RE_LEVELS, default=575.0)
    parser.add_argument("--forcing-wavenumber", type=int, choices=K_LEVELS, default=4)
    parser.add_argument("--observation-interval", type=int, choices=(1, 2, 4, 6, 8), default=1)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--record-trace", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_one(parse_args())
