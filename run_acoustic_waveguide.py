from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from run_hybrid_wave import (
    branch_compatibility,
    liu_normal_inverse,
    logsumexp,
    mixture_score_and_denoised,
    softmax,
)


@dataclass(frozen=True)
class WaveguideConfig:
    seed: int = 20260803
    length: float = 1.5
    diameter: float = 0.13
    nx_pressure: int = 101
    density: float = 1.21
    sound_speed: float = 343.0
    damping: float = 8.0
    dt: float = 2.0e-5
    t_end: float = 0.012

    primary_frequency: float = 300.0
    uncertain_frequency: float = 450.0
    primary_source_x: float = 0.25
    uncertain_source_x: float = 1.02
    primary_source_rate: float = 1800.0
    uncertain_source_rate: float = 900.0

    ensemble_size: int = 18
    n_alpha: int = 7
    alpha_min: float = 0.08
    alpha_max: float = 0.92
    alpha_true: float = 0.78
    epistemic_scale: float = 1.0

    process_noise_pressure: float = 0.0015
    obs_noise_pressure: float = 0.020
    n_pressure_sensors: int = 8
    obs_interval: int = 20

    reverse_steps: int = 8
    sigma_min: float = 0.015
    sigma_max: float = 0.70
    guidance: float = 0.75
    reverse_noise_scale: float = 0.45
    score_clip: float = 40.0
    scale_floor_pressure: float = 0.035
    credibility_rate: float = 0.55
    regression_ridge: float = 1.0e-5


@dataclass(frozen=True)
class PairedWaveguideConfig:
    """Configuration for deterministic direct/reverberant endpoint pairs."""

    length: float = 1.5
    nx_pressure: int = 101
    density: float = 1.21
    sound_speed: float = 343.0
    damping: float = 8.0
    dt: float = 2.0e-5
    t_end: float = 0.012
    sponge_cells: int = 8
    sponge_strength: float = 1800.0
    source_width: float = 0.025
    pulse_center: float = 0.0015
    pulse_width: float = 0.00075
    source_x_min: float = 0.15
    source_x_max: float = 1.35
    frequency_min: float = 300.0
    frequency_max: float = 1200.0
    amplitude_min: float = 900.0
    amplitude_max: float = 1800.0
    phase_min: float = 0.0
    phase_max: float = 2.0 * math.pi
    time_stride: int = 4
    n_sensors: int = 9
    observation_noise_fraction: float = 0.01
    train_size: int = 1024
    val_size: int = 256
    test_size: int = 256
    train_seed: int = 20260811
    val_seed: int = 20260812
    test_seed: int = 20260813
    batch_size: int = 32


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def paired_grids(cfg: PairedWaveguideConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = cfg.length / cfg.nx_pressure
    x_pressure = (np.arange(cfg.nx_pressure) + 0.5) * dx
    x_velocity = np.arange(cfg.nx_pressure + 1) * dx
    n_steps = int(round(cfg.t_end / cfg.dt))
    time = np.arange(0, n_steps + 1, cfg.time_stride) * cfg.dt
    return x_pressure, x_velocity, time


def latin_hypercube_source_params(
    n_samples: int, seed: int, cfg: PairedWaveguideConfig
) -> np.ndarray:
    """Independent four-dimensional Latin-hypercube samples."""

    rng = np.random.default_rng(seed)
    unit = np.empty((n_samples, 4), dtype=np.float64)
    for column in range(unit.shape[1]):
        strata = (np.arange(n_samples) + rng.random(n_samples)) / n_samples
        unit[:, column] = strata[rng.permutation(n_samples)]
    lower = np.array(
        [cfg.source_x_min, cfg.frequency_min, cfg.amplitude_min, cfg.phase_min]
    )
    upper = np.array(
        [cfg.source_x_max, cfg.frequency_max, cfg.amplitude_max, cfg.phase_max]
    )
    return lower + unit * (upper - lower)


def build_sparse_operators(
    nx_pressure: int = 101, n_sensors: int = 9
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact selection and piecewise-linear backprojection operators."""

    if n_sensors < 2 or n_sensors > nx_pressure:
        raise ValueError("n_sensors must be between 2 and nx_pressure.")
    sensor_indices = np.rint(
        np.linspace(0.1 * (nx_pressure - 1), 0.9 * (nx_pressure - 1), n_sensors)
    ).astype(np.int64)
    if np.unique(sensor_indices).size != n_sensors:
        raise ValueError("Sensor indices are not unique.")
    h_select = np.zeros((n_sensors, nx_pressure), dtype=np.float64)
    h_select[np.arange(n_sensors), sensor_indices] = 1.0
    e_interp = np.empty((nx_pressure, n_sensors), dtype=np.float64)
    sensor_coordinate = sensor_indices.astype(np.float64)
    for grid_index in range(nx_pressure):
        e_interp[grid_index] = np.array(
            [
                np.interp(
                    float(grid_index), sensor_coordinate, np.eye(n_sensors)[basis_index]
                )
                for basis_index in range(n_sensors)
            ]
        )
    if not np.allclose(h_select @ e_interp, np.eye(n_sensors), atol=1.0e-12):
        raise RuntimeError("Sparse operators violate H_select @ E_interp = I.")
    return h_select, e_interp, sensor_indices


def _pulse_source_rate(
    x_pressure: np.ndarray,
    t: float,
    source_params: np.ndarray,
    cfg: PairedWaveguideConfig,
) -> np.ndarray:
    source_x = source_params[:, 0:1]
    frequency = source_params[:, 1:2]
    amplitude = source_params[:, 2:3]
    phase = source_params[:, 3:4]
    spatial = np.exp(-((x_pressure[None, :] - source_x) / cfg.source_width) ** 2)
    envelope = math.exp(-((t - cfg.pulse_center) / cfg.pulse_width) ** 2)
    carrier = np.sin(2.0 * math.pi * frequency * t + phase)
    return amplitude * envelope * carrier * spatial


def _sponge_profile(cfg: PairedWaveguideConfig) -> np.ndarray:
    profile = np.zeros(cfg.nx_pressure, dtype=np.float64)
    cells = min(cfg.sponge_cells, cfg.nx_pressure // 2)
    if cells:
        ramp = ((np.arange(cells, 0, -1) / cells) ** 2) * cfg.sponge_strength
        profile[:cells] = ramp
        profile[-cells:] = ramp[::-1]
    return profile


def simulate_pulse_batch(
    source_params: np.ndarray,
    cfg: PairedWaveguideConfig,
    boundary: Literal["rigid", "absorbing"],
) -> np.ndarray:
    """Simulate p and impedance-scaled axial velocity on pressure centers.

    Returns an array with shape [batch, 2, time, space].
    """

    if boundary not in ("rigid", "absorbing"):
        raise ValueError(f"Unknown boundary mode: {boundary}")
    source_params = np.asarray(source_params, dtype=np.float64)
    if source_params.ndim != 2 or source_params.shape[1] != 4:
        raise ValueError("source_params must have shape [batch, 4].")
    x_pressure, _, output_time = paired_grids(cfg)
    n_p = cfg.nx_pressure
    n_steps = int(round(cfg.t_end / cfg.dt))
    dx = cfg.length / n_p
    cfl = cfg.sound_speed * cfg.dt / dx
    if cfl >= 0.95:
        raise ValueError(f"Acoustic CFL={cfl:.3f} is too large.")

    pressure = np.zeros((source_params.shape[0], n_p), dtype=np.float64)
    velocity = np.zeros((source_params.shape[0], n_p + 1), dtype=np.float64)
    result = np.empty(
        (source_params.shape[0], 2, output_time.size, n_p), dtype=np.float32
    )
    sponge = _sponge_profile(cfg) if boundary == "absorbing" else np.zeros(n_p)
    sponge_factor_p = np.exp(-sponge * cfg.dt)
    sponge_faces = np.zeros(n_p + 1, dtype=np.float64)
    sponge_faces[1:-1] = 0.5 * (sponge[:-1] + sponge[1:])
    sponge_faces[0], sponge_faces[-1] = sponge[0], sponge[-1]
    sponge_factor_u = np.exp(-sponge_faces * cfg.dt)

    def record(output_index: int) -> None:
        velocity_center = 0.5 * (velocity[:, :-1] + velocity[:, 1:])
        result[:, 0, output_index] = pressure
        result[:, 1, output_index] = cfg.density * cfg.sound_speed * velocity_center

    record(0)
    output_index = 1
    for step in range(1, n_steps + 1):
        t = (step - 1) * cfg.dt
        pressure_gradient = (pressure[:, 1:] - pressure[:, :-1]) / dx
        velocity[:, 1:-1] = (
            velocity[:, 1:-1]
            - cfg.dt / cfg.density * pressure_gradient
            - cfg.damping * cfg.dt * velocity[:, 1:-1]
        )
        if boundary == "rigid":
            velocity[:, 0] = 0.0
            velocity[:, -1] = 0.0
        else:
            impedance = cfg.density * cfg.sound_speed
            velocity[:, 0] = -pressure[:, 0] / impedance
            velocity[:, -1] = pressure[:, -1] / impedance
            velocity *= sponge_factor_u[None, :]
        divergence = (velocity[:, 1:] - velocity[:, :-1]) / dx
        pressure = (
            pressure
            - cfg.density * cfg.sound_speed**2 * cfg.dt * divergence
            - cfg.damping * cfg.dt * pressure
            + cfg.dt * _pulse_source_rate(x_pressure, t, source_params, cfg)
        )
        if boundary == "absorbing":
            pressure *= sponge_factor_p[None, :]
        if step % cfg.time_stride == 0:
            record(output_index)
            output_index += 1
    if output_index != output_time.size:
        raise RuntimeError("Unexpected number of downsampled time frames.")
    return result


def _create_split_datasets(group, n: int, n_time: int, n_space: int, n_sensors: int):
    field_chunks = (1, 2, n_time, n_space)
    sparse_chunks = (min(16, n), 2, n_time, n_sensors)
    kwargs = {"compression": "lzf", "shuffle": True}
    group.create_dataset("X0_direct", (n, 2, n_time, n_space), dtype="f4", chunks=field_chunks, **kwargs)
    group.create_dataset("X1_reverb_full", (n, 2, n_time, n_space), dtype="f4", chunks=field_chunks, **kwargs)
    group.create_dataset("Y_sparse", (n, 2, n_time, n_sensors), dtype="f4", chunks=sparse_chunks, **kwargs)
    group.create_dataset("X1_condition", (n, 2, n_time, n_space), dtype="f4", chunks=field_chunks, **kwargs)
    group.create_dataset("source_params", (n, 4), dtype="f8")
    group.create_dataset("branch_id", (n,), dtype="i1")


def generate_paired_hdf5(
    output_path: Path, cfg: PairedWaveguideConfig
) -> dict[str, object]:
    """Generate train/validation/test endpoint pairs without time-frame leakage."""

    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("paired-data mode requires h5py") from exc

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_pressure, _, time = paired_grids(cfg)
    h_select, e_interp, sensor_indices = build_sparse_operators(
        cfg.nx_pressure, cfg.n_sensors
    )
    split_specs = {
        "train": (cfg.train_size, cfg.train_seed),
        "val": (cfg.val_size, cfg.val_seed),
        "test": (cfg.test_size, cfg.test_seed),
    }
    script_path = Path(__file__).resolve()
    source_sha = _sha256(script_path)
    clean_observation_sumsq = np.zeros(2, dtype=np.float64)
    clean_observation_count = 0

    with h5py.File(output_path, "w") as h5:
        h5.attrs["schema_version"] = "triadbridge-waveguide-pairs-v1"
        h5.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        h5.attrs["config_json"] = json.dumps(asdict(cfg), sort_keys=True)
        h5.attrs["units_json"] = json.dumps(
            {
                "X_channels": ["pressure_Pa", "rho_c_times_axial_velocity_Pa"],
                "source_params": ["position_m", "frequency_Hz", "amplitude_Pa_per_s", "phase_rad"],
                "time": "s",
                "space": "m",
            },
            sort_keys=True,
        )
        h5.attrs["source_script"] = str(script_path)
        h5.attrs["source_script_sha256"] = source_sha
        h5.attrs["python_version"] = platform.python_version()
        h5.attrs["numpy_version"] = np.__version__
        operator = h5.create_group("operator")
        operator.create_dataset("H_select", data=h_select)
        operator.create_dataset("E_interp", data=e_interp)
        operator.create_dataset("sensor_indices", data=sensor_indices)
        grid = h5.create_group("grid")
        grid.create_dataset("x", data=x_pressure)
        grid.create_dataset("time", data=time)
        normalization = h5.create_group("normalization")
        channel_mean_ds = normalization.create_dataset("channel_mean", (2,), dtype="f8")
        channel_std_ds = normalization.create_dataset("channel_std", (2,), dtype="f8")
        noise_rms_ds = normalization.create_dataset("observation_clean_rms", (2,), dtype="f8")

        for split, (n_samples, seed) in split_specs.items():
            group = h5.create_group(split)
            _create_split_datasets(group, n_samples, time.size, cfg.nx_pressure, cfg.n_sensors)
            params = latin_hypercube_source_params(n_samples, seed, cfg)
            group["source_params"][:] = params
            group["branch_id"][:] = (params[:, 0] >= 0.5 * cfg.length).astype(np.int8)
            for start in range(0, n_samples, cfg.batch_size):
                stop = min(start + cfg.batch_size, n_samples)
                batch_params = params[start:stop]
                direct = simulate_pulse_batch(batch_params, cfg, "absorbing")
                reverb = simulate_pulse_batch(batch_params, cfg, "rigid")
                clean_y = np.einsum("bcix,sx->bcis", reverb, h_select, optimize=True)
                group["X0_direct"][start:stop] = direct
                group["X1_reverb_full"][start:stop] = reverb
                group["Y_sparse"][start:stop] = clean_y
                if split == "train":
                    clean_observation_sumsq += np.square(clean_y, dtype=np.float64).sum(axis=(0, 2, 3))
                    clean_observation_count += clean_y.shape[0] * clean_y.shape[2] * clean_y.shape[3]

        observation_rms = np.sqrt(clean_observation_sumsq / clean_observation_count)
        noise_rms_ds[:] = observation_rms
        total_sum = np.zeros(2, dtype=np.float64)
        total_sumsq = np.zeros(2, dtype=np.float64)
        total_count = 0
        for split, (n_samples, seed) in split_specs.items():
            group = h5[split]
            noise_rng = np.random.default_rng(seed + 1000003)
            for start in range(0, n_samples, cfg.batch_size):
                stop = min(start + cfg.batch_size, n_samples)
                clean_y = group["Y_sparse"][start:stop]
                noise = noise_rng.normal(size=clean_y.shape).astype(np.float32)
                noisy_y = clean_y + (
                    cfg.observation_noise_fraction * observation_rms[None, :, None, None] * noise
                )
                condition = np.einsum("bcis,xs->bcix", noisy_y, e_interp, optimize=True).astype(np.float32)
                group["Y_sparse"][start:stop] = noisy_y
                group["X1_condition"][start:stop] = condition
                if split == "train":
                    direct = group["X0_direct"][start:stop]
                    for array in (direct, condition):
                        total_sum += array.sum(axis=(0, 2, 3), dtype=np.float64)
                        total_sumsq += np.square(array, dtype=np.float64).sum(axis=(0, 2, 3))
                        total_count += array.shape[0] * array.shape[2] * array.shape[3]
        channel_mean = total_sum / total_count
        channel_var = np.maximum(total_sumsq / total_count - channel_mean**2, 1.0e-16)
        channel_mean_ds[:] = channel_mean
        channel_std_ds[:] = np.sqrt(channel_var)
        h5.flush()

    file_sha = _sha256(output_path)
    manifest = {
        "schema_version": "triadbridge-waveguide-pairs-v1",
        "hdf5": str(output_path),
        "hdf5_sha256": file_sha,
        "source_script": str(script_path),
        "source_script_sha256": source_sha,
        "config": asdict(cfg),
        "splits": {name: {"size": size, "seed": seed} for name, (size, seed) in split_specs.items()},
        "shape": {"field": [2, int(time.size), cfg.nx_pressure], "sparse": [2, int(time.size), cfg.n_sensors]},
        "normalization": {"channel_mean": channel_mean.tolist(), "channel_std": np.sqrt(channel_var).tolist()},
        "observation_clean_rms": observation_rms.tolist(),
        "operator_identity_max_abs": float(np.max(np.abs(h_select @ e_interp - np.eye(cfg.n_sensors)))),
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
    return manifest


def smooth_pressure_noise(z: np.ndarray) -> np.ndarray:
    z = z.copy()
    z[..., 1:-1] = 0.25 * z[..., :-2] + 0.50 * z[..., 1:-1] + 0.25 * z[..., 2:]
    return z


def source_rate(x: np.ndarray, t: float, theta: float, cfg: WaveguideConfig) -> np.ndarray:
    primary_profile = np.exp(-((x - cfg.primary_source_x) / 0.035) ** 2)
    uncertain_profile = np.exp(-((x - cfg.uncertain_source_x) / 0.045) ** 2)

    primary_envelope = np.exp(-((t - 0.0028) / 0.0018) ** 2)
    uncertain_envelope = np.exp(-((t - 0.0058) / 0.0025) ** 2)

    primary = (
        cfg.primary_source_rate
        * primary_envelope
        * np.sin(2.0 * math.pi * cfg.primary_frequency * t)
        * primary_profile
    )
    uncertain = (
        theta
        * cfg.uncertain_source_rate
        * uncertain_envelope
        * np.sin(2.0 * math.pi * cfg.uncertain_frequency * t + 0.35)
        * uncertain_profile
    )
    return primary + uncertain


def initial_state(cfg: WaveguideConfig) -> np.ndarray:
    # Pressure at cell centers, particle velocity at faces.
    return np.zeros(cfg.nx_pressure + cfg.nx_pressure + 1, dtype=float)


def propagate_batch(
    states: np.ndarray,
    theta: float,
    t: float,
    cfg: WaveguideConfig,
    rng: np.random.Generator,
    stochastic: bool = True,
    common_noise: np.ndarray | None = None,
) -> np.ndarray:
    n_p = cfg.nx_pressure
    n_u = n_p + 1
    dx = cfg.length / n_p
    cfl = cfg.sound_speed * cfg.dt / dx
    if cfl >= 0.95:
        raise ValueError(f"Acoustic CFL={cfl:.3f} is too large.")

    pressure = states[:, :n_p]
    velocity = states[:, n_p : n_p + n_u]

    # Staggered-grid symplectic Euler update.
    velocity_new = velocity.copy()
    pressure_gradient = (pressure[:, 1:] - pressure[:, :-1]) / dx
    velocity_new[:, 1:-1] = (
        velocity[:, 1:-1]
        - cfg.dt / cfg.density * pressure_gradient
        - cfg.damping * cfg.dt * velocity[:, 1:-1]
    )
    # Rigid terminations: normal particle velocity is zero.
    velocity_new[:, 0] = 0.0
    velocity_new[:, -1] = 0.0

    divergence = (velocity_new[:, 1:] - velocity_new[:, :-1]) / dx
    x_p = (np.arange(n_p) + 0.5) * dx
    pressure_new = (
        pressure
        - cfg.density * cfg.sound_speed**2 * cfg.dt * divergence
        - cfg.damping * cfg.dt * pressure
        + cfg.dt * source_rate(x_p, t, theta, cfg)[None, :]
    )

    if stochastic:
        noise = smooth_pressure_noise(
            rng.normal(size=pressure.shape) if common_noise is None else common_noise
        )
        pressure_new += cfg.process_noise_pressure * noise

    return np.concatenate([pressure_new, velocity_new], axis=1)


def observed_ensf_update(
    prior_observed: np.ndarray,
    observation: np.ndarray,
    cfg: WaveguideConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    mean = prior_observed.mean(axis=0)
    scale = np.maximum(prior_observed.std(axis=0, ddof=1), cfg.scale_floor_pressure)
    clean = (prior_observed - mean) / scale
    ratio = cfg.sigma_max / cfg.sigma_min
    log_ratio = math.log(ratio)
    z = clean + cfg.sigma_max * rng.normal(size=clean.shape)
    pseudo_times = np.linspace(1.0, 0.0, cfg.reverse_steps + 1)

    for k in range(cfg.reverse_steps):
        tau = max(pseudo_times[k], 1.0e-4)
        tau_next = max(pseudo_times[k + 1], 0.0)
        d_tau = tau_next - tau
        sigma = cfg.sigma_min * ratio**tau
        diffusion = sigma * math.sqrt(2.0 * log_ratio)
        prior_score, denoised = mixture_score_and_denoised(z, clean, sigma)
        predicted = mean[None, :] + scale[None, :] * denoised
        likelihood_score = (
            scale[None, :]
            * (observation[None, :] - predicted)
            / cfg.obs_noise_pressure**2
        )
        posterior_score = prior_score + cfg.guidance * (1.0 - tau) * likelihood_score
        posterior_score = np.clip(posterior_score, -cfg.score_clip, cfg.score_clip)
        z = (
            z
            + (-diffusion**2 * posterior_score) * d_tau
            + cfg.reverse_noise_scale
            * diffusion
            * math.sqrt(-d_tau)
            * rng.normal(size=z.shape)
        )
    return mean + scale * z


def ensf_lr_update(
    prior: np.ndarray,
    observation: np.ndarray,
    observation_indices: np.ndarray,
    cfg: WaveguideConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    updated_observed = observed_ensf_update(
        prior[:, observation_indices], observation, cfg, rng
    )
    increment = updated_observed - prior[:, observation_indices]
    all_indices = np.arange(prior.shape[1])
    unobserved = np.setdiff1d(all_indices, observation_indices)
    updated = prior.copy()
    updated[:, observation_indices] = updated_observed

    anomalies_o = prior[:, observation_indices] - prior[:, observation_indices].mean(axis=0)
    anomalies_u = prior[:, unobserved] - prior[:, unobserved].mean(axis=0)
    denom = max(prior.shape[0] - 1, 1)
    cov_oo = anomalies_o.T @ anomalies_o / denom
    cov_uo = anomalies_u.T @ anomalies_o / denom
    ridge = cfg.regression_ridge * (np.trace(cov_oo) / max(cov_oo.shape[0], 1) + 1.0)
    regression = cov_uo @ np.linalg.pinv(cov_oo + ridge * np.eye(cov_oo.shape[0]))
    updated[:, unobserved] = prior[:, unobserved] + increment @ regression.T

    n_p = cfg.nx_pressure
    updated[:, n_p] = 0.0
    updated[:, -1] = 0.0
    return updated


def run(cfg: WaveguideConfig, output_dir: Path) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    n_p = cfg.nx_pressure
    n_u = n_p + 1
    dx = cfg.length / n_p
    x_p = (np.arange(n_p) + 0.5) * dx
    x_u = np.arange(n_u) * dx
    n_steps = int(round(cfg.t_end / cfg.dt))
    times = np.arange(n_steps + 1) * cfg.dt

    alpha_grid = np.linspace(cfg.alpha_min, cfg.alpha_max, cfg.n_alpha)
    theta_grid = cfg.epistemic_scale * liu_normal_inverse(alpha_grid)
    theta_true = float(cfg.epistemic_scale * liu_normal_inverse(cfg.alpha_true))

    sensor_indices = np.linspace(5, n_p - 6, cfg.n_pressure_sensors, dtype=int)
    observation_indices = sensor_indices.copy()

    x0 = initial_state(cfg)
    truth = x0[None, :].copy()
    baseline = x0[None, :].copy()
    branches = np.repeat(x0[None, None, :], cfg.n_alpha, axis=0)
    branches = np.repeat(branches, cfg.ensemble_size, axis=1)
    branches[:, :, :n_p] += 0.02 * smooth_pressure_noise(
        rng.normal(size=(cfg.n_alpha, cfg.ensemble_size, n_p))
    )
    branches[:, :, n_p + 1 : -1] += 3.0e-5 * rng.normal(
        size=(cfg.n_alpha, cfg.ensemble_size, n_u - 2)
    )

    log_weights = np.zeros(cfg.n_alpha)
    weights = softmax(log_weights)

    truth_pressure = np.zeros((n_steps + 1, n_p))
    estimate_pressure = np.zeros_like(truth_pressure)
    baseline_pressure = np.zeros_like(truth_pressure)
    truth_velocity = np.zeros((n_steps + 1, n_u))
    estimate_velocity = np.zeros_like(truth_velocity)
    rmse_hybrid = np.zeros(n_steps + 1)
    rmse_baseline = np.zeros(n_steps + 1)
    credibility_times: list[float] = []
    credibility_history: list[np.ndarray] = []

    def summarize(step: int) -> None:
        branch_means = branches.mean(axis=1)
        estimate = np.sum(weights[:, None] * branch_means, axis=0)
        truth_pressure[step] = truth[0, :n_p]
        estimate_pressure[step] = estimate[:n_p]
        baseline_pressure[step] = baseline[0, :n_p]
        truth_velocity[step] = truth[0, n_p : n_p + n_u]
        estimate_velocity[step] = estimate[n_p : n_p + n_u]
        rmse_hybrid[step] = float(np.sqrt(np.mean((estimate_pressure[step] - truth_pressure[step]) ** 2)))
        rmse_baseline[step] = float(np.sqrt(np.mean((baseline_pressure[step] - truth_pressure[step]) ** 2)))

    summarize(0)
    for step in range(1, n_steps + 1):
        t = times[step - 1]
        truth = propagate_batch(truth, theta_true, t, cfg, rng, stochastic=True)
        baseline = propagate_batch(baseline, 0.0, t, cfg, rng, stochastic=False)
        common_noise = rng.normal(size=(cfg.ensemble_size, n_p))
        for q, theta in enumerate(theta_grid):
            branches[q] = propagate_batch(
                branches[q], float(theta), t, cfg, rng,
                stochastic=True, common_noise=common_noise,
            )

        if step % cfg.obs_interval == 0:
            observation = (
                truth[0, observation_indices]
                + cfg.obs_noise_pressure * rng.normal(size=observation_indices.size)
            )
            compatibility = np.empty(cfg.n_alpha)
            for q in range(cfg.n_alpha):
                compatibility[q] = branch_compatibility(
                    branches[q][:, observation_indices],
                    observation,
                    cfg.obs_noise_pressure,
                )
                branches[q] = ensf_lr_update(
                    branches[q], observation, observation_indices, cfg, rng
                )
            log_weights += cfg.credibility_rate * np.log(np.clip(compatibility, 1.0e-8, 1.0))
            log_weights -= np.max(log_weights)
            weights = softmax(log_weights)
            credibility_times.append(times[step])
            credibility_history.append(weights.copy())
        summarize(step)

    alpha_best = float(alpha_grid[int(np.argmax(weights))])
    metrics = {
        "mean_rmse_hybrid_pa": float(rmse_hybrid.mean()),
        "mean_rmse_baseline_pa": float(rmse_baseline.mean()),
        "relative_rmse_reduction_percent": float(100.0 * (1.0 - rmse_hybrid.mean() / max(rmse_baseline.mean(), 1.0e-15))),
        "final_rmse_hybrid_pa": float(rmse_hybrid[-1]),
        "final_rmse_baseline_pa": float(rmse_baseline[-1]),
        "alpha_true": cfg.alpha_true,
        "alpha_best_final": alpha_best,
        "theta_true": theta_true,
        "plane_wave_cutoff_hz": float(1.841 * cfg.sound_speed / (math.pi * cfg.diameter)),
        "cfl": float(cfg.sound_speed * cfg.dt / dx),
    }

    # Geometry schematic
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.plot([0, cfg.length], [0, 0], linewidth=12, alpha=0.25)
    ax.scatter([cfg.primary_source_x], [0], marker="*", s=180, label="known primary source")
    ax.scatter([cfg.uncertain_source_x], [0], marker="D", s=80, label="epistemically uncertain disturbance")
    ax.scatter(x_p[sensor_indices], np.zeros_like(sensor_indices), marker="x", s=55, label="pressure sensors")
    ax.axvline(0.0, linewidth=2)
    ax.axvline(cfg.length, linewidth=2)
    ax.set_xlim(-0.05, cfg.length + 0.05)
    ax.set_ylim(-0.25, 0.35)
    ax.set_yticks([])
    ax.set_xlabel("axial position x (m)")
    ax.set_title("One-dimensional rigid-termination acoustic waveguide")
    ax.legend(ncol=3, loc="upper center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "17_waveguide_geometry.png", dpi=180)
    plt.close(fig)

    max_amp = max(np.abs(truth_pressure).max(), np.abs(estimate_pressure).max())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True, sharey=True)
    arrays = [truth_pressure, estimate_pressure, np.abs(truth_pressure - estimate_pressure)]
    titles = ["Pressure truth", "Pressure estimate", "Absolute pressure error"]
    for i, (ax, array, title) in enumerate(zip(axes, arrays, titles)):
        kwargs = dict(origin="lower", aspect="auto", extent=[0.0, cfg.length, 0.0, 1000.0 * cfg.t_end])
        if i < 2:
            kwargs.update(vmin=-max_amp, vmax=max_amp)
        image = ax.imshow(array, **kwargs)
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("time (ms)")
    fig.tight_layout()
    fig.savefig(output_dir / "18_waveguide_spacetime.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_p, truth_pressure[-1], linewidth=2.0, label="Truth")
    ax.plot(x_p, estimate_pressure[-1], linewidth=1.8, label="Hybrid EnSF-LR")
    ax.plot(x_p, baseline_pressure[-1], linestyle="--", linewidth=1.5, label="Deterministic baseline")
    ax.scatter(x_p[sensor_indices], truth_pressure[-1, sensor_indices], marker="x", s=50, label="Sensors")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("sound pressure (Pa)")
    ax.set_title("Final pressure field in the acoustic waveguide")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "19_waveguide_final_pressure.png", dpi=180)
    plt.close(fig)

    phase_p_index = int(np.argmin(np.abs(x_p - 1.10)))
    phase_u_index = int(np.argmin(np.abs(x_u - x_p[phase_p_index])))
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(1000.0 * times, truth_pressure[:, phase_p_index], truth_velocity[:, phase_u_index], label="Truth")
    ax.plot(1000.0 * times, estimate_pressure[:, phase_p_index], estimate_velocity[:, phase_u_index], label="Estimate")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("pressure (Pa)")
    ax.set_zlabel("particle velocity (m/s)")
    ax.set_title(f"Pressure-velocity trajectory near x={x_p[phase_p_index]:.2f} m")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "20_waveguide_pressure_velocity_trajectory_3d.png", dpi=180)
    plt.close(fig)

    if credibility_history:
        credibility_array = np.asarray(credibility_history)
        fig, ax = plt.subplots(figsize=(9, 5))
        for q, alpha in enumerate(alpha_grid):
            ax.plot(1000.0 * np.asarray(credibility_times), credibility_array[:, q], label=fr"$\alpha={alpha:.2f}$")
        ax.set_xlabel("assimilation time (ms)")
        ax.set_ylabel("normalized path weight")
        ax.set_title("Alpha-path identification in the acoustic waveguide")
        ax.grid(alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "21_waveguide_alpha_paths.png", dpi=180)
        plt.close(fig)

    with (output_dir / "waveguide_metrics.json").open("w", encoding="utf-8") as file:
        json.dump({"config": asdict(cfg), "metrics": metrics}, file, ensure_ascii=False, indent=2)
    np.savez_compressed(
        output_dir / "waveguide_data.npz",
        x_pressure=x_p,
        x_velocity=x_u,
        times=times,
        truth_pressure=truth_pressure,
        estimate_pressure=estimate_pressure,
        baseline_pressure=baseline_pressure,
        truth_velocity=truth_velocity,
        estimate_velocity=estimate_velocity,
        alpha_grid=alpha_grid,
        final_weights=weights,
        credibility_times=np.asarray(credibility_times),
        credibility_history=np.asarray(credibility_history),
        rmse_hybrid=rmse_hybrid,
        rmse_baseline=rmse_baseline,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Physical 1D acoustic waveguide demonstration")
    parser.add_argument(
        "--task", choices=["hilda", "paired-data"], default="hilda",
        help="Keep the historical HILDA run as default; paired-data writes TriadBridge endpoints.",
    )
    parser.add_argument("--output", default="results_acoustic_waveguide")
    parser.add_argument("--paired-output", default="waveguide_pairs.h5")
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--train-size", type=int, default=1024)
    parser.add_argument("--val-size", type=int, default=256)
    parser.add_argument("--test-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.task == "paired-data":
        cfg = PairedWaveguideConfig(
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            batch_size=args.batch_size,
        )
        manifest = generate_paired_hdf5(Path(args.paired_output), cfg)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    cfg = WaveguideConfig(seed=args.seed)
    metrics = run(cfg, Path(args.output))
    print("\nAcoustic waveguide simulation finished:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print(f"\nOutput: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
