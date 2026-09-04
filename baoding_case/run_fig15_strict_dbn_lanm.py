#!/usr/bin/env python3
"""Strict, auditable reconstruction of the DBN-LA-NM Fig. 15 field run.

This runner implements Eqs. (21), (24), (26), (44), and (45) and Algorithms
1--2 from Zhang et al. (2022).  It deliberately excludes grid search, GPS
correction after initialization, trust regions, trajectory alignment, and
post-hoc target relabeling.  Protocol ambiguities are command-line switches
and are written to the result manifest instead of being silently tuned.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

PAPER_NODES = (1, 3, 5, 6, 7, 8, 11, 13)
NODE_TO_IP = {1: 47, 3: 40, 5: 54, 6: 43, 7: 49, 8: 61, 11: 5, 13: 46}
CHANNEL_GROUPS = {
    "z": (19, 18, 17, 13, 14, 15, 16),
    "x": (9, 8, 7, 1, 2, 3),
    "y": (12, 11, 10, 4, 5, 6),
}
PAPER_STATE_UUDV = (
    (38614853.4, 25.85033, 4337388.27, 23.00194),
    (38615012.2, -41.09208, 4336467.20, 6.67753),
    (38615647.2, 3.20862, 4337215.10, -41.49795),
)
GPS_FILES = {1: "GPS1_plane1.gps", 2: "GPS3_plane2.gps", 3: "GPS4_plane2to3.gps"}
FREQUENCIES_HZ = tuple(range(3, 1498, 3))
FS_HZ = 3000
SNAPSHOT = 2048
SOUND_SPEED = 340.0
PROCESSING_HEIGHT_M = 230.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def hms_seconds(value: str | float | int) -> float:
    number = float(value)
    integer = int(number)
    text = str(integer).zfill(6)
    return int(text[:2]) * 3600 + int(text[2:4]) * 60 + int(text[4:]) + number - integer


def parse_nod(path: Path) -> dict[int, dict[str, float | int]]:
    nodes: dict[int, dict[str, float | int]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 7:
            continue
        node = int(fields[2])
        nodes[node] = {
            "ip": int(fields[0].split(".")[-1]),
            "x": float(fields[3]),
            "y": float(fields[4]),
            "z": float(fields[5]),
            "azimuth_offset_deg": float(fields[7]) if len(fields) > 7 else 0.0,
            "elevation_offset_deg": float(fields[8]) if len(fields) > 8 else 0.0,
            "horizontal_reverse": int(float(fields[9])) if len(fields) > 9 else 0,
            "vertical_reverse": int(float(fields[10])) if len(fields) > 10 else 0,
        }
    return nodes


def read_gps(path: Path) -> list[tuple[float, float, float, float]]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 8:
            continue
        try:
            rows.append((hms_seconds(fields[7]), float(fields[4]), float(fields[5]), float(fields[6])))
        except ValueError:
            pass
    if not rows:
        raise RuntimeError(f"no GPS rows in {path}")
    return rows


def nearest_gps(rows: list[tuple[float, float, float, float]], time_s: float) -> tuple[float, float, float]:
    _, x, y, z = min(rows, key=lambda row: abs(row[0] - time_s))
    return x, y, z


def interpolate_gps(rows: list[tuple[float, float, float, float]], time_s: float) -> tuple[float, float, float]:
    """Linearly interpolate GPS for sub-second evaluation frames.

    The raw GPS stream is an offline scoring reference, so interpolation does
    not enter the filter state or any source/precision update.
    """
    times = np.asarray([row[0] for row in rows], dtype=np.float64)
    coordinates = np.asarray([[row[1], row[2], row[3]] for row in rows], dtype=np.float64)
    order = np.argsort(times, kind="stable")
    times = times[order]
    coordinates = coordinates[order]
    unique, indices = np.unique(times, return_index=True)
    coordinates = coordinates[indices]
    return tuple(float(np.interp(time_s, unique, coordinates[:, column])) for column in range(3))


def sensor_geometry(
    nodes: dict[int, dict[str, float | int]], profile: str, orientation_mode: str
) -> np.ndarray:
    all_sensors = []
    xy_offsets = np.asarray((-3.0, -2.0, -1.0, 1.0, 2.0, 3.0)) * 0.5
    if profile == "field_nonuniform_z":
        z_offsets = np.asarray((-2.13, -1.53, -0.93, 0.0, 1.0, 2.0, 3.0)) * 0.5
    elif profile == "paper_uniform_cross":
        z_offsets = np.asarray((-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)) * 0.5
    else:
        raise ValueError(profile)
    for node_id in PAPER_NODES:
        node = nodes[node_id]
        reverse_horizontal = int(node["horizontal_reverse"])
        reverse_vertical = int(node["vertical_reverse"])
        if orientation_mode == "ignore":
            x_direction = np.asarray((1.0, 0.0))
            y_direction = np.asarray((0.0, 1.0))
            vertical_sign = 1.0
        elif orientation_mode == "legacy_all_axis_sign":
            horizontal_sign = -1.0 if reverse_horizontal else 1.0
            x_direction = np.asarray((horizontal_sign, 0.0))
            y_direction = np.asarray((0.0, horizontal_sign))
            vertical_sign = -1.0 if reverse_vertical else 1.0
        elif orientation_mode == "nod_azimuth_convention":
            angular_sign = -1.0 if reverse_horizontal else 1.0
            theta = np.deg2rad(angular_sign * float(node["azimuth_offset_deg"]))
            x_direction = np.asarray((np.cos(theta), np.sin(theta)))
            y_direction = angular_sign * np.asarray((-np.sin(theta), np.cos(theta)))
            vertical_sign = -1.0 if reverse_vertical else 1.0
        else:
            raise ValueError(orientation_mode)
        x0, y0, z0 = float(node["x"]), float(node["y"]), float(node["z"])
        positions = np.zeros((19, 3), dtype=np.float64)
        for channel, offset in zip(CHANNEL_GROUPS["x"], xy_offsets):
            positions[channel - 1] = (x0 + x_direction[0] * offset, y0 + x_direction[1] * offset, z0)
        for channel, offset in zip(CHANNEL_GROUPS["y"], xy_offsets):
            positions[channel - 1] = (x0 + y_direction[0] * offset, y0 + y_direction[1] * offset, z0)
        for channel, offset in zip(CHANNEL_GROUPS["z"], z_offsets):
            positions[channel - 1] = (x0, y0, z0 + vertical_sign * offset)
        all_sensors.append(positions)
    return np.vstack(all_sensors)


def state_transition(dt: float, q_accel: float, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    f = torch.tensor(
        [[1.0, dt, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, dt], [0.0, 0.0, 0.0, 1.0]],
        device=device, dtype=dtype,
    )
    g = torch.tensor(
        [[dt * dt / 2.0, 0.0], [dt, 0.0], [0.0, dt * dt / 2.0], [0.0, dt]],
        device=device, dtype=dtype,
    )
    q = g @ (torch.eye(2, device=device, dtype=dtype) * q_accel) @ g.T
    return f, q


def steering(
    xy: torch.Tensor, sensors: torch.Tensor, frequencies: torch.Tensor, height_model: str
) -> torch.Tensor:
    """Return F x P x M near-field steering matrices."""
    delta_xy = xy[:, None, :] - sensors[None, :, :2]
    horizontal2 = torch.sum(delta_xy * delta_xy, dim=-1)
    if height_model == "paper230":
        vertical2 = torch.full_like(horizontal2, PROCESSING_HEIGHT_M**2)
    elif height_model == "sensor_z":
        vertical2 = (PROCESSING_HEIGHT_M - sensors[None, :, 2]) ** 2
    elif height_model == "planar":
        # Eq. (5) and the delivered Tracking.py use the 2-D horizontal
        # distance; altitude is not a state variable in the field model.
        vertical2 = torch.zeros_like(horizontal2)
    else:
        raise ValueError(height_model)
    distance = torch.sqrt(horizontal2 + vertical2).T  # P x M
    gain = torch.reciprocal(torch.clamp(distance, min=1e-9))
    phase = torch.exp(-1j * 2.0 * torch.pi * frequencies[:, None, None] * distance[None, :, :] / SOUND_SPEED)
    return phase * gain[None, :, :]


def expected_steering_stats(
    mean: torch.Tensor, covariance: torch.Tensor, sensors: torch.Tensor, frequencies: torch.Tensor,
    particle_count: int, height_model: str, generator: torch.Generator, frequency_chunk: int,
    moment_mode: str = "paper_full",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Monte-Carlo E[A] and E[A^H A] from Eq. (22)/(25)."""
    target_count = mean.shape[0]
    freq_count = len(frequencies)
    sensor_count = sensors.shape[0]
    complex_dtype = torch.complex128 if mean.dtype == torch.float64 else torch.complex64
    expected_a = torch.empty((freq_count, sensor_count, target_count), device=mean.device, dtype=complex_dtype)
    diagonal_power = torch.empty((freq_count, target_count), device=mean.device, dtype=mean.dtype)
    coordinate_indices = torch.tensor((0, 2), device=mean.device, dtype=torch.long)
    for target in range(target_count):
        cov_xy = covariance[target].index_select(0, coordinate_indices).index_select(1, coordinate_indices)
        cov_xy = (cov_xy + cov_xy.T) * 0.5
        position_mean = mean[target].index_select(0, coordinate_indices)
        if moment_mode == "author_diagonal":
            # RandomState.multivariate_normal in the delivered Tracking.py
            # uses an SVD factor and only warns for a non-PSD covariance. Its
            # singular values therefore allow sampling to continue after an
            # indefinite Newton inverse. Reproduce that runtime behavior only
            # in the literal author-code branch.
            _, singular_values, vh = torch.linalg.svd(cov_xy)
            factor = torch.diag(torch.sqrt(singular_values)) @ vh
        else:
            factor = torch.linalg.cholesky(cov_xy).T
        if moment_mode == "paper_full":
            standard = torch.randn((particle_count, 2), generator=generator, device=mean.device, dtype=mean.dtype)
            samples = position_mean[None, :] + standard @ factor
            delta = samples[:, None, :] - sensors[None, :, :2]
            horizontal2 = torch.sum(delta * delta, dim=-1)
            if height_model == "paper230":
                vertical2 = torch.full_like(horizontal2, PROCESSING_HEIGHT_M**2)
            elif height_model == "sensor_z":
                vertical2 = (PROCESSING_HEIGHT_M - sensors[None, :, 2]) ** 2
            elif height_model == "planar":
                vertical2 = torch.zeros_like(horizontal2)
            else:
                raise ValueError(height_model)
            distance = torch.sqrt(horizontal2 + vertical2)
            gain = torch.reciprocal(torch.clamp(distance, min=1e-9))
            diagonal_power[:, target] = torch.mean(torch.sum(gain * gain, dim=1))
            for start in range(0, freq_count, frequency_chunk):
                stop = min(freq_count, start + frequency_chunk)
                phase = torch.exp(
                    -1j * 2.0 * torch.pi * frequencies[start:stop, None, None] * distance[None, :, :] / SOUND_SPEED
                )
                expected_a[start:stop, :, target] = torch.mean(phase * gain[None, :, :], dim=1)
        elif moment_mode == "author_diagonal":
            for start in range(0, freq_count, frequency_chunk):
                stop = min(freq_count, start + frequency_chunk)
                count = stop - start
                standard = torch.randn(
                    (count, particle_count, 2), generator=generator, device=mean.device, dtype=mean.dtype
                )
                samples = position_mean[None, None, :] + standard @ factor
                delta = samples[:, :, None, :] - sensors[None, None, :, :2]
                horizontal2 = torch.sum(delta * delta, dim=-1)
                if height_model == "paper230":
                    vertical2 = torch.full_like(horizontal2, PROCESSING_HEIGHT_M**2)
                elif height_model == "sensor_z":
                    vertical2 = (PROCESSING_HEIGHT_M - sensors[None, None, :, 2]) ** 2
                elif height_model == "planar":
                    vertical2 = torch.zeros_like(horizontal2)
                else:
                    raise ValueError(height_model)
                distance = torch.sqrt(horizontal2 + vertical2)
                gain = torch.reciprocal(torch.clamp(distance, min=1e-9))
                phase = torch.exp(
                    -1j * 2.0 * torch.pi * frequencies[start:stop, None, None]
                    * distance / SOUND_SPEED
                )
                expected_a[start:stop, :, target] = torch.mean(phase * gain, dim=1)
                diagonal_power[start:stop, target] = torch.mean(torch.sum(gain * gain, dim=2), dim=1)
        else:
            raise ValueError(moment_mode)
    if moment_mode == "paper_full":
        # Independent target-state factors make the off-diagonal moments the
        # products of the corresponding steering-vector expectations.
        expected_aha = torch.einsum("fpm,fpn->fmn", torch.conj(expected_a), expected_a)
    elif moment_mode == "author_diagonal":
        # Tracking.py initializes EX_AHA with zeros and only writes its
        # diagonal. Keep that delivered-code behavior isolated.
        expected_aha = torch.zeros(
            (freq_count, target_count, target_count), device=mean.device, dtype=complex_dtype
        )
    else:
        raise ValueError(moment_mode)
    indices = torch.arange(target_count, device=mean.device)
    expected_aha[:, indices, indices] = diagonal_power.to(expected_aha.dtype)
    return expected_a, expected_aha


def _as_snapshot_tensor(z: torch.Tensor) -> torch.Tensor:
    """Normalize a frequency-domain observation to F x L x P."""
    if z.ndim == 2:
        return z[:, None, :]
    if z.ndim != 3:
        raise ValueError(f"expected z with shape F x P or F x L x P, got {tuple(z.shape)}")
    return z


def _validate_snapshot_count(z: torch.Tensor, snapshot_count_l: float | None) -> torch.Tensor:
    snapshots = _as_snapshot_tensor(z)
    if snapshot_count_l is not None and not math.isclose(
        float(snapshot_count_l), float(snapshots.shape[1]), rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError(
            f"snapshot_count_l={snapshot_count_l} disagrees with observation L={snapshots.shape[1]}"
        )
    return snapshots


def eq44_eq45_initialization(
    a: torch.Tensor, z: torch.Tensor, snapshot_count_l: float | None = None
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Initialize the noise precision and source coefficients in Eqs. (44)-(45).

    Equation (44) defines ``R_z = Z Z^H / L``.  Consequently, repeated
    identical snapshots do not spuriously increase the initial precision: the
    projection residual is averaged over the physical snapshots.
    """
    snapshots = _validate_snapshot_count(z, snapshot_count_l)
    snapshot_count = snapshots.shape[1]
    gram = torch.einsum("fpm,fpn->fmn", torch.conj(a), a)
    zbar = torch.mean(snapshots, dim=1)
    rhs = torch.einsum("fpm,fp->fm", torch.conj(a), zbar)
    eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)[None, :, :]
    regularized_gram = gram + eye * 1e-12
    source = torch.linalg.solve(regularized_gram, rhs)
    # Eq. (44) is tr(P_A^perp R_z), not a residual around one source
    # coefficient shared by all snapshots. Project every snapshot separately;
    # otherwise legitimate within-frame source-amplitude variation is counted
    # as sensor noise.
    snapshot_rhs = torch.einsum("fpm,flp->flm", torch.conj(a), snapshots)
    snapshot_source = torch.linalg.solve(
        regularized_gram[:, None, :, :], snapshot_rhs[..., None]
    )[..., 0]
    projected_residual = snapshots - torch.einsum("fpm,flm->flp", a, snapshot_source)
    residual_power = torch.mean(
        torch.sum(torch.abs(projected_residual) ** 2, dim=2), dim=1
    ).real
    precision = (a.shape[1] - a.shape[2]) / torch.clamp(
        residual_power, min=1e-20
    )
    diagnostics = {
        "eq44_snapshot_count_l": int(snapshot_count),
        "projected_residual_power_min": float(torch.min(residual_power).item()),
        "projected_residual_power_median": float(torch.median(residual_power).item()),
        "projected_residual_power_max": float(torch.max(residual_power).item()),
        "lambda_min": float(torch.min(precision).item()),
        "lambda_median": float(torch.median(precision).item()),
        "lambda_max": float(torch.max(precision).item()),
        "source_norm_min": float(torch.min(torch.linalg.vector_norm(source, dim=1)).item()),
        "source_norm_median": float(torch.median(torch.linalg.vector_norm(source, dim=1)).item()),
        "source_norm_max": float(torch.max(torch.linalg.vector_norm(source, dim=1)).item()),
    }
    return precision, source, diagnostics


def gamma_prior(precision: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "author_literal":
        return precision.clone(), torch.ones_like(precision)
    if mode == "unit_shape":
        return torch.ones_like(precision), torch.reciprocal(torch.clamp(precision, min=1e-20))
    raise ValueError(mode)


def expected_residual_components(
    z: torch.Tensor, expected_a: torch.Tensor, expected_aha: torch.Tensor,
    source_mean: torch.Tensor, source_cov: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the two expectations in Eq. (21).

    Expanding E[||z - A s||^2] is necessary because, for uncertain target
    states, E[A^H A] is not E[A]^H E[A].  Writing the first term as
    ||z - E[A] E[s]||^2 drops that state-uncertainty contribution.
    """
    snapshots = _as_snapshot_tensor(z)
    zbar = torch.mean(snapshots, dim=1)
    z_power = torch.mean(torch.sum(torch.abs(snapshots) ** 2, dim=2), dim=1).real
    predicted_mean = torch.einsum("fpm,fm->fp", expected_a, source_mean)
    cross = 2.0 * torch.sum(torch.conj(zbar) * predicted_mean, dim=1).real
    mean_quadratic = torch.einsum(
        "fm,fmn,fn->f", torch.conj(source_mean), expected_aha, source_mean
    ).real
    covariance_quadratic = torch.einsum("fmn,fnm->f", expected_aha, source_cov).real
    mean_residual_power = torch.clamp(z_power - cross + mean_quadratic, min=1e-20)
    return mean_residual_power, torch.clamp(covariance_quadratic, min=0.0)


def expected_residual_power(
    z: torch.Tensor, expected_a: torch.Tensor, expected_aha: torch.Tensor,
    source_mean: torch.Tensor, source_cov: torch.Tensor,
) -> torch.Tensor:
    """Return complete expected residual power for L=1 compatibility."""
    mean_residual_power, covariance_quadratic = expected_residual_components(
        z, expected_a, expected_aha, source_mean, source_cov
    )
    return torch.clamp(mean_residual_power + covariance_quadratic, min=1e-20)


def update_hidden(
    z: torch.Tensor, expected_a: torch.Tensor, expected_aha: torch.Tensor,
    alpha_prior: torch.Tensor, beta_prior: torch.Tensor, source_prior_mean: torch.Tensor,
    source_prior_cov: torch.Tensor, source_mean: torch.Tensor, source_cov: torch.Tensor,
    snapshot_count_l: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    alpha, beta, expected_precision, residual_power = update_noise_precision(
        z, expected_a, expected_aha, alpha_prior, beta_prior, source_mean, source_cov,
        snapshot_count_l,
    )
    prior_precision = torch.linalg.inv(source_prior_cov)
    posterior_precision = expected_precision[:, None, None].to(expected_aha.dtype) * expected_aha + prior_precision
    source_cov_new = torch.linalg.inv(posterior_precision)
    zbar = torch.mean(_validate_snapshot_count(z, snapshot_count_l), dim=1)
    rhs = (
        expected_precision[:, None].to(expected_a.dtype)
        * torch.einsum("fpm,fp->fm", torch.conj(expected_a), zbar)
        + torch.einsum("fmn,fn->fm", prior_precision, source_prior_mean)
    )
    source_mean_new = torch.einsum("fmn,fn->fm", source_cov_new, rhs)
    diagnostics = {
        "lambda_min": float(torch.min(expected_precision).item()),
        "lambda_median": float(torch.median(expected_precision).item()),
        "lambda_max": float(torch.max(expected_precision).item()),
        "residual_power_median": float(torch.median(residual_power).item()),
        "source_norm_median": float(torch.median(torch.linalg.vector_norm(source_mean_new, dim=1)).item()),
    }
    return alpha, beta, source_mean_new, source_cov_new, diagnostics


def update_noise_precision(
    z: torch.Tensor, expected_a: torch.Tensor, expected_aha: torch.Tensor,
    alpha_prior: torch.Tensor, beta_prior: torch.Tensor,
    source_mean: torch.Tensor, source_cov: torch.Tensor, snapshot_count_l: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update only q(lambda_f) in Eq. (21)."""
    _validate_snapshot_count(z, snapshot_count_l)
    mean_residual_power, covariance_quadratic = expected_residual_components(
        z, expected_a, expected_aha, source_mean, source_cov
    )
    alpha = alpha_prior + z.shape[-1] / 2.0
    # Eq. (21) averages the data residual over L received samples. The source
    # posterior covariance term is already an expectation over q(x_k).
    residual_power = mean_residual_power + covariance_quadratic
    beta = beta_prior + 0.5 * residual_power
    return alpha, beta, alpha / beta, residual_power


def update_author_hierarchical_source(
    z: torch.Tensor, expected_a: torch.Tensor, expected_aha: torch.Tensor,
    expected_noise_precision: torch.Tensor, source_mean: torch.Tensor, source_cov: torch.Tensor,
    hyper_mean: torch.Tensor, hyper_variance: torch.Tensor,
    precision_shape: torch.Tensor, precision_rate: torch.Tensor,
    quadratic_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Reproduce the UpdateLambda/UpdateMu/UpdateSk hierarchy in Tracking.py.

    The delivered author code places a Gaussian-Gamma hierarchy above each
    complex source coefficient.  This layer is not present in Eqs. (21)/(24),
    so it is exposed as a separate, auditable mode.  The complex quadratic is
    evaluated as a modulus squared to keep the Gamma rate real and positive.
    """
    posterior_variance = torch.diagonal(source_cov, dim1=-2, dim2=-1)
    precision_shape_new = precision_shape + 0.5
    if quadratic_mode == "modulus":
        quadratic = torch.abs(source_mean - hyper_mean) ** 2
        posterior_variance = posterior_variance.real
        hyper_variance = hyper_variance.real
        precision_rate_new = precision_rate.real + 0.5 * (
            quadratic + posterior_variance + hyper_variance
        )
        source_prior_precision = precision_shape_new.real / torch.clamp(precision_rate_new, min=1e-20)
        hyper_precision = torch.reciprocal(torch.clamp(hyper_variance, min=1e-20))
    elif quadratic_mode == "author_complex_square":
        precision_rate_new = precision_rate + 0.5 * (
            (source_mean - hyper_mean) ** 2 + torch.reciprocal(
                torch.diagonal(torch.linalg.inv(source_cov), dim1=-2, dim2=-1)
            ) + hyper_variance
        )
        source_prior_precision = precision_shape_new / precision_rate_new
        hyper_precision = torch.reciprocal(hyper_variance)
    else:
        raise ValueError(quadratic_mode)
    hyper_variance_new = torch.reciprocal(hyper_precision + source_prior_precision)
    hyper_mean_new = hyper_variance_new.to(source_mean.dtype) * (
        hyper_precision.to(source_mean.dtype) * hyper_mean
        + source_prior_precision.to(source_mean.dtype) * source_mean
    )

    prior_precision_matrix = torch.diag_embed(source_prior_precision).to(expected_aha.dtype)
    zbar = torch.mean(_as_snapshot_tensor(z), dim=1)
    posterior_precision = (
        expected_noise_precision[:, None, None].to(expected_aha.dtype) * expected_aha
        + prior_precision_matrix
    )
    source_cov_full = torch.linalg.inv(posterior_precision)
    rhs = (
        expected_noise_precision[:, None].to(expected_a.dtype)
        * torch.einsum("fpm,fp->fm", torch.conj(expected_a), zbar)
        + source_prior_precision.to(source_mean.dtype) * hyper_mean_new
    )
    source_mean_new = torch.einsum("fmn,fn->fm", source_cov_full, rhs)
    # Tracking.py retains only the diagonal of the source posterior precision
    # as sk.s.lam. Reconstruct the corresponding diagonal covariance for the
    # next noise-precision and source-hyperparameter updates.
    source_diag_precision = torch.diagonal(posterior_precision, dim1=-2, dim2=-1)
    if quadratic_mode == "modulus":
        source_diag_precision = source_diag_precision.real
        source_variance = torch.reciprocal(torch.clamp(source_diag_precision, min=1e-20))
    else:
        source_variance = torch.reciprocal(source_diag_precision)
    source_cov_new = torch.diag_embed(source_variance).to(source_cov_full.dtype)
    diagnostics = {
        "source_precision_abs_min": float(torch.min(torch.abs(source_prior_precision)).item()),
        "source_precision_abs_median": float(torch.median(torch.abs(source_prior_precision)).item()),
        "source_precision_abs_max": float(torch.max(torch.abs(source_prior_precision)).item()),
        "source_hyper_variance_abs_median": float(torch.median(torch.abs(hyper_variance_new)).item()),
        "source_norm_median": float(torch.median(torch.linalg.vector_norm(source_mean_new, dim=1)).item()),
    }
    return (
        source_mean_new, source_cov_new, hyper_mean_new, hyper_variance_new,
        precision_shape_new, precision_rate_new, diagnostics,
    )


def log_posterior(
    states: torch.Tensor, predicted_mean: torch.Tensor, predicted_covariance: torch.Tensor, z: torch.Tensor,
    source_mean: torch.Tensor, source_cov: torch.Tensor, expected_precision: torch.Tensor,
    sensors: torch.Tensor, frequencies: torch.Tensor, height_model: str, objective_mode: str,
) -> torch.Tensor:
    prior_value = torch.zeros((), device=states.device, dtype=states.dtype)
    for target in range(states.shape[0]):
        delta = states[target] - predicted_mean[target]
        prior_value = prior_value - 0.5 * delta @ torch.linalg.solve(predicted_covariance[target], delta)
    a = steering(states[:, [0, 2]], sensors, frequencies, height_model)
    if objective_mode in ("paper_eq26_eq21_consistent", "paper_eq26_literal"):
        snapshots = _as_snapshot_tensor(z)
        residual = snapshots - torch.einsum("fpm,fm->fp", a, source_mean)[:, None, :]
        residual_power = torch.mean(torch.sum(torch.abs(residual) ** 2, dim=2), dim=1).real
        gram = torch.einsum("fpm,fpn->fmn", torch.conj(a), a)
        trace_term = torch.einsum("fmn,fnm->f", gram, source_cov).real
        if objective_mode == "paper_eq26_literal":
            # Printed Eq. (26) places both the sum over L residuals and the
            # single covariance trace inside 1/(F L).  This differs from
            # Eq. (21), where only the residual is averaged over L.
            trace_term = trace_term / snapshots.shape[1]
        observation = -0.5 * torch.mean(expected_precision * (residual_power + trace_term))
    elif objective_mode == "author_tracking_py":
        # This scalar objective has the same target-wise derivatives as
        # Tracking.py: frequency terms are summed, lambda_0 is omitted, and
        # cross-target products are absent.
        source_variance = torch.diagonal(source_cov, dim1=-2, dim2=-1).real
        source_power = torch.abs(source_mean) ** 2 + source_variance
        steering_power = torch.sum(torch.abs(a) ** 2, dim=1)
        zbar = torch.mean(_as_snapshot_tensor(z), dim=1)
        matched = torch.einsum("fm,fpm,fp->fm", torch.conj(source_mean), torch.conj(a), zbar).real
        observation = torch.sum(matched - 0.5 * source_power * steering_power)
    else:
        raise ValueError(objective_mode)
    return prior_value + observation


def newton_update(
    states: torch.Tensor, predicted_mean: torch.Tensor, predicted_covariance: torch.Tensor, z: torch.Tensor,
    source_mean: torch.Tensor, source_cov: torch.Tensor, expected_precision: torch.Tensor,
    sensors: torch.Tensor, frequencies: torch.Tensor, height_model: str, max_iterations: int, threshold: float,
    allow_indefinite_hessian: bool, objective_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    current = states.clone()
    final_covariances = predicted_covariance.clone()
    diagnostics = []
    for inner in range(max_iterations):
        before = current.clone()
        for target in range(current.shape[0]):
            base = current.detach().clone()
            variable = base[target].detach().clone().requires_grad_(True)

            def objective(value: torch.Tensor) -> torch.Tensor:
                candidate = torch.cat((base[:target], value[None, :], base[target + 1 :]), dim=0)
                return log_posterior(
                    candidate, predicted_mean, predicted_covariance, z, source_mean, source_cov,
                    expected_precision, sensors, frequencies, height_model, objective_mode,
                )

            value = objective(variable)
            gradient = torch.autograd.grad(value, variable, create_graph=True)[0]
            hessian = torch.autograd.functional.hessian(objective, variable, vectorize=True)
            hessian = (hessian + hessian.T) * 0.5
            eigenvalues = torch.linalg.eigvalsh(hessian)
            if not torch.isfinite(hessian).all() or not torch.isfinite(gradient).all():
                raise FloatingPointError(f"non-finite Newton derivatives inner={inner} target={target + 1}")
            hessian_negative_definite = bool(float(torch.max(eigenvalues).item()) < 0.0)
            if not hessian_negative_definite and not allow_indefinite_hessian:
                raise RuntimeError(
                    f"log-posterior Hessian is not negative definite: inner={inner} target={target + 1} "
                    f"eigenvalues={eigenvalues.detach().cpu().tolist()}"
                )
            step = torch.linalg.solve(hessian, gradient)
            updated = variable.detach() - step.detach()
            posterior_cov = -torch.linalg.inv(hessian.detach())
            current[target] = updated
            final_covariances[target] = (posterior_cov + posterior_cov.T) * 0.5
            diagnostics.append(
                {
                    "inner_iteration": inner + 1,
                    "target": target + 1,
                    "objective": float(value.detach().item()),
                    "gradient_norm": float(torch.linalg.vector_norm(gradient.detach()).item()),
                    "step_norm": float(torch.linalg.vector_norm(step.detach()).item()),
                    "hessian_negative_definite": hessian_negative_definite,
                    "hessian_eigenvalues": [float(x) for x in eigenvalues.detach().cpu().tolist()],
                }
            )
        if float(torch.sum(torch.abs(current - before)).item()) < threshold:
            break
    return current, final_covariances, diagnostics


def frame_ospa(estimated: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    fixed = math.sqrt(float(np.mean(np.sum((estimated - truth) ** 2, axis=1))))
    assignment = min(
        math.sqrt(float(np.mean([np.sum((estimated[i] - truth[j]) ** 2) for i, j in enumerate(order)])))
        for order in itertools.permutations(range(3))
    )
    return fixed, assignment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync-root", type=Path, required=True)
    parser.add_argument(
        "--spectrum-root", type=Path, default=None,
        help="Optional native-clock spectrum bundle containing spectrum_real.npy and spectrum_node_shift.npy.",
    )
    parser.add_argument("--nod", type=Path, required=True)
    parser.add_argument("--gps-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument(
        "--sample-rate-hz", type=float, default=FS_HZ,
        help="Effective acoustic sample rate. Paper nominal is 3000 Hz; archived packet clock is about 3050 Hz.",
    )
    parser.add_argument(
        "--snapshot-samples", type=int, default=SNAPSHOT,
        help="Total time-domain samples covered by one processing frame.",
    )
    parser.add_argument(
        "--snapshot-count", type=int, default=1,
        help="Number L of physical DFT snapshots per processing frame.",
    )
    parser.add_argument(
        "--dft-length", type=int, default=None,
        help="DFT length of each snapshot. Defaults to snapshot-samples for L=1.",
    )
    parser.add_argument(
        "--snapshot-stride", type=int, default=None,
        help="Sample stride between snapshots. Defaults to dft-length.",
    )
    parser.add_argument(
        "--eq44-snapshot-count-l",
        type=float,
        default=None,
        help=(
            "Optional consistency assertion for L in Eq. (44). The physical "
            "snapshot count is taken from --snapshot-count."
        ),
    )
    parser.add_argument("--start-hhmmss", type=float, default=132754.0)
    parser.add_argument("--hop-samples", type=int, default=3000)
    parser.add_argument("--state-dt", type=float, default=1.0)
    parser.add_argument("--outer-iterations", type=int, default=8)
    parser.add_argument("--newton-iterations", type=int, default=4)
    parser.add_argument("--particles", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument(
        "--allow-indefinite-hessian",
        action="store_true",
        help="Continue with the paper/public-code Newton inverse when the Hessian is not negative definite; no covariance repair is applied.",
    )
    parser.add_argument("--initial-covariance", type=float, default=0.001)
    parser.add_argument("--process-acceleration-variance", type=float, default=1.0)
    parser.add_argument("--source-prior-covariance", type=float, default=0.1)
    parser.add_argument(
        "--implementation-mode",
        choices=("paper_equations", "author_hierarchy_corrected", "author_tracking_literal"),
        default="paper_equations",
    )
    parser.add_argument("--lambda-hyper-mode", choices=("author_literal", "unit_shape"), default="author_literal")
    parser.add_argument("--vb-update-mode", choices=("paper_fixed_prior", "author_cumulative"), default="paper_fixed_prior")
    parser.add_argument("--hidden-temporal-mode", choices=("carry", "reset"), default="carry")
    parser.add_argument("--geometry-profile", choices=("field_nonuniform_z", "paper_uniform_cross"), default="field_nonuniform_z")
    parser.add_argument("--height-model", choices=("paper230", "sensor_z", "planar"), default="paper230")
    parser.add_argument("--nod-direction-flags", action="store_true")
    parser.add_argument(
        "--array-orientation-mode",
        choices=("ignore", "nod_azimuth_convention", "legacy_all_axis_sign"),
        default="ignore",
    )
    parser.add_argument(
        "--dft-sign", choices=("negative", "positive"), default="negative",
        help="DFT bin convention: author code uses the negative-frequency index.",
    )
    parser.add_argument("--input-mode", choices=("real", "zero", "node_shift"), default="real")
    parser.add_argument(
        "--eq26-scaling",
        choices=("eq21_consistent", "paper_literal"),
        default="eq21_consistent",
        help=(
            "Resolve the printed Eq. (21)/Eq. (26) L-scaling ambiguity. "
            "eq21_consistent leaves the covariance trace outside the L average; "
            "paper_literal divides the single Eq. (26) trace term by L."
        ),
    )
    parser.add_argument("--raw-scale", type=float, default=1.0)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float64")
    parser.add_argument("--frequency-chunk", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2026082901)
    args = parser.parse_args()

    if args.frames <= 0 or args.particles <= 0 or args.snapshot_count <= 0 or args.sample_rate_hz <= 0:
        raise ValueError("frames, particles, and snapshot-count must be positive")
    dft_length = args.dft_length or args.snapshot_samples
    snapshot_stride = args.snapshot_stride or dft_length
    frame_span = (args.snapshot_count - 1) * snapshot_stride + dft_length
    if dft_length <= 0 or snapshot_stride <= 0 or args.snapshot_samples < frame_span:
        raise ValueError(
            "snapshot-samples must cover (snapshot-count - 1) * snapshot-stride + dft-length"
        )
    if args.eq44_snapshot_count_l is not None and not math.isclose(
        args.eq44_snapshot_count_l, args.snapshot_count, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError("eq44-snapshot-count-l must match the physical snapshot-count")
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    dtype = torch.float64 if args.precision == "float64" else torch.float32
    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    generator = torch.Generator(device=device).manual_seed(args.seed)
    nodes = parse_nod(args.nod)
    missing = sorted(set(PAPER_NODES) - set(nodes))
    if missing:
        raise RuntimeError(f"missing paper nodes: {missing}")
    orientation_mode = args.array_orientation_mode
    if args.nod_direction_flags:
        if orientation_mode != "ignore":
            raise ValueError("use either --nod-direction-flags or --array-orientation-mode, not both")
        orientation_mode = "legacy_all_axis_sign"
    sensors_np = sensor_geometry(nodes, args.geometry_profile, orientation_mode)
    center = np.mean(np.asarray([[float(nodes[n]["x"]), float(nodes[n]["y"])] for n in PAPER_NODES]), axis=0)
    sensors_np[:, :2] -= center[None, :]
    sensors = torch.as_tensor(sensors_np, device=device, dtype=dtype)
    frequencies = torch.as_tensor(FREQUENCIES_HZ, device=device, dtype=dtype)
    positive_bins = torch.round(frequencies / args.sample_rate_hz * dft_length).to(torch.long)
    if args.dft_sign == "negative":
        bins = torch.remainder(dft_length - positive_bins, dft_length)
        actual_frequencies = torch.remainder(-bins, dft_length).to(dtype) * args.sample_rate_hz / dft_length
    else:
        bins = positive_bins
        actual_frequencies = bins.to(dtype) * args.sample_rate_hz / dft_length
    max_frequency_mismatch_hz = float(torch.max(torch.abs(actual_frequencies - frequencies)).item())

    streams = []
    source_hashes = {}
    spectrum_stream = None
    if args.spectrum_root is not None:
        spectrum_name = "spectrum_node_shift.npy" if args.input_mode == "node_shift" else "spectrum_real.npy"
        spectrum_path = args.spectrum_root / spectrum_name
        spectrum_stream = np.load(spectrum_path, mmap_mode="r")
        expected_shape = (args.frames, len(FREQUENCIES_HZ), len(PAPER_NODES) * 19)
        if spectrum_stream.ndim != 3 or any(
            spectrum_stream.shape[index] < expected_shape[index] for index in range(3)
        ):
            raise RuntimeError(f"invalid native spectrum {spectrum_path}: {spectrum_stream.shape}, need {expected_shape}")
        source_hashes["native_spectrum"] = sha256(spectrum_path)
        manifest_path = args.spectrum_root / "sync_manifest.json"
        if manifest_path.exists():
            source_hashes["native_spectrum_manifest"] = sha256(manifest_path)
        max_frequency_mismatch_hz = 0.0
    else:
        for node in PAPER_NODES:
            path = args.sync_root / f"node{node}_ip{NODE_TO_IP[node]}_3khz.npy"
            stream = np.load(path, mmap_mode="r")
            needed = (args.frames - 1) * args.hop_samples + frame_span
            if stream.shape[0] != 19 or stream.shape[1] < needed:
                raise RuntimeError(f"invalid synchronized stream {path}: {stream.shape}, need {needed}")
            streams.append(stream)
            source_hashes[str(node)] = sha256(path)

    gps = {target: read_gps(args.gps_root / filename) for target, filename in GPS_FILES.items()}
    initial = np.asarray(PAPER_STATE_UUDV, dtype=np.float64)
    initial[:, 0] -= center[0]
    initial[:, 2] -= center[1]
    states = torch.as_tensor(initial, device=device, dtype=dtype)
    covariance = torch.eye(4, device=device, dtype=dtype)[None, :, :].repeat(3, 1, 1) * args.initial_covariance
    transition, process_cov = state_transition(args.state_dt, args.process_acceleration_variance, device, dtype)
    source_prior_cov_base = torch.eye(3, device=device, dtype=complex_dtype)[None, :, :].repeat(len(frequencies), 1, 1) * args.source_prior_covariance
    alpha_carry = beta_carry = source_mean_carry = source_cov_carry = None
    hyper_mean_carry = hyper_variance_carry = precision_shape_carry = precision_rate_carry = None
    records = []
    frame_diagnostics = []
    initialization_diagnostics = None
    started = time.perf_counter()
    failure = None

    try:
        for frame_index in range(args.frames):
            if spectrum_stream is not None:
                spectrum = torch.as_tensor(
                    np.asarray(spectrum_stream[frame_index]), device=device, dtype=complex_dtype
                )[:, None, :]
                if args.input_mode == "zero":
                    spectrum = torch.zeros_like(spectrum)
            else:
                sample_start = frame_index * args.hop_samples
                raw = np.vstack(
                    [stream[:, sample_start : sample_start + frame_span] for stream in streams]
                ).astype(np.float64, copy=False)
                raw /= args.raw_scale
                if args.input_mode == "zero":
                    raw.fill(0.0)
                elif args.input_mode == "node_shift":
                    raw = raw.reshape(8, 19, frame_span)
                    raw = np.stack(
                        [np.roll(raw[node], shift=node * 73, axis=1) for node in range(8)], axis=0
                    ).reshape(152, frame_span)
                chunks = np.stack(
                    [raw[:, offset : offset + dft_length] for offset in range(0, (args.snapshot_count - 1) * snapshot_stride + 1, snapshot_stride)],
                    axis=1,
                )
                frame_tensor = torch.as_tensor(chunks, device=device, dtype=dtype)
                spectrum = (
                    torch.fft.fft(frame_tensor, dim=2)[:, :, bins].permute(2, 1, 0).to(complex_dtype)
                    / dft_length
                )

            # The paper's initial GPS state is the state at the first acoustic
            # frame.  Apply the transition only after that frame; predicting
            # frame 0 would introduce an artificial one-step time offset.
            if frame_index == 0:
                predicted_mean = states.clone()
                predicted_covariance = covariance.clone()
            else:
                predicted_mean = states @ transition.T
                predicted_covariance = transition[None, :, :] @ covariance @ transition.T[None, :, :] + process_cov[None, :, :]
            states = predicted_mean.clone()
            covariance = predicted_covariance.clone()
            deterministic_a = steering(states[:, [0, 2]], sensors, frequencies, args.height_model)
            lambda0, source0, init_diag = eq44_eq45_initialization(
                deterministic_a, spectrum, args.snapshot_count
            )
            if frame_index == 0:
                initialization_diagnostics = init_diag
            alpha0, beta0 = gamma_prior(lambda0, args.lambda_hyper_mode)
            source_prior_mean = source0.clone()
            source_prior_cov = source_prior_cov_base.clone()
            if frame_index > 0 and args.hidden_temporal_mode == "carry" and alpha_carry is not None:
                alpha0, beta0 = alpha_carry.clone(), beta_carry.clone()
                source_prior_mean, source_prior_cov = source_mean_carry.clone(), source_cov_carry.clone()
            alpha, beta = alpha0.clone(), beta0.clone()
            source_mean, source_cov = source_prior_mean.clone(), source_prior_cov.clone()
            hyper_mean = source_prior_mean.clone()
            hyper_variance = torch.full_like(source_mean.real, args.source_prior_covariance)
            precision_shape = torch.full_like(source_mean.real, 1.0 / args.source_prior_covariance)
            precision_rate = torch.ones_like(source_mean.real)
            if (
                frame_index > 0
                and args.hidden_temporal_mode == "carry"
                and hyper_mean_carry is not None
            ):
                hyper_mean = hyper_mean_carry.clone()
                hyper_variance = hyper_variance_carry.clone()
                precision_shape = precision_shape_carry.clone()
                precision_rate = precision_rate_carry.clone()
            outer_rows = []
            for outer in range(args.outer_iterations):
                previous_states = states.clone()
                moment_mode = (
                    "author_diagonal"
                    if args.implementation_mode == "author_tracking_literal"
                    else "paper_full"
                )
                expected_a, expected_aha = expected_steering_stats(
                    states, covariance, sensors, frequencies, args.particles, args.height_model,
                    generator, args.frequency_chunk, moment_mode,
                )
                cumulative = (
                    args.implementation_mode != "paper_equations"
                    or args.vb_update_mode == "author_cumulative"
                )
                update_alpha = alpha if cumulative else alpha0
                update_beta = beta if cumulative else beta0
                if args.implementation_mode == "paper_equations":
                    update_source_prior_mean = source_mean if cumulative else source_prior_mean
                    update_source_prior_cov = source_cov if cumulative else source_prior_cov
                    alpha, beta, source_mean, source_cov, hidden_diag = update_hidden(
                        spectrum, expected_a, expected_aha, update_alpha, update_beta,
                        update_source_prior_mean, update_source_prior_cov, source_mean, source_cov,
                        args.snapshot_count,
                    )
                else:
                    alpha, beta, noise_precision, residual_power = update_noise_precision(
                        spectrum, expected_a, expected_aha, update_alpha, update_beta,
                        source_mean, source_cov,
                    )
                    quadratic_mode = (
                        "author_complex_square"
                        if args.implementation_mode == "author_tracking_literal"
                        else "modulus"
                    )
                    (
                        source_mean, source_cov, hyper_mean, hyper_variance,
                        precision_shape, precision_rate, source_diag,
                    ) = update_author_hierarchical_source(
                        spectrum, expected_a, expected_aha, noise_precision,
                        source_mean, source_cov, hyper_mean, hyper_variance,
                        precision_shape, precision_rate, quadratic_mode,
                    )
                    hidden_diag = {
                        "lambda_min": float(torch.min(noise_precision).item()),
                        "lambda_median": float(torch.median(noise_precision).item()),
                        "lambda_max": float(torch.max(noise_precision).item()),
                        "residual_power_median": float(torch.median(residual_power).item()),
                        **source_diag,
                    }
                if args.implementation_mode == "author_tracking_literal":
                    objective_mode = "author_tracking_py"
                else:
                    objective_mode = (
                        "paper_eq26_eq21_consistent"
                        if args.eq26_scaling == "eq21_consistent"
                        else "paper_eq26_literal"
                    )
                states, covariance, newton_diag = newton_update(
                    states, predicted_mean, predicted_covariance, spectrum, source_mean, source_cov, alpha / beta,
                    sensors, frequencies, args.height_model, args.newton_iterations, args.threshold,
                    args.allow_indefinite_hessian, objective_mode,
                )
                state_change = float(torch.sum(torch.abs(states - previous_states)).item())
                outer_rows.append(
                    {
                        "outer_iteration": outer + 1,
                        "state_change_l1": state_change,
                        "hidden": hidden_diag,
                        "newton": newton_diag,
                    }
                )
                if state_change < args.threshold:
                    break
            alpha_carry, beta_carry = alpha.detach().clone(), beta.detach().clone()
            source_mean_carry, source_cov_carry = source_mean.detach().clone(), source_cov.detach().clone()
            hyper_mean_carry = hyper_mean.detach().clone()
            hyper_variance_carry = hyper_variance.detach().clone()
            precision_shape_carry = precision_shape.detach().clone()
            precision_rate_carry = precision_rate.detach().clone()
            state_np = states.detach().cpu().numpy()
            cov_np = covariance.detach().cpu().numpy()
            time_s = hms_seconds(args.start_hhmmss) + frame_index * args.hop_samples / args.sample_rate_hz
            truth_reader = interpolate_gps if args.hop_samples < args.sample_rate_hz else nearest_gps
            truth = np.asarray([truth_reader(gps[target], time_s)[:2] for target in range(1, 4)], dtype=np.float64)
            estimated = np.column_stack((state_np[:, 0] + center[0], state_np[:, 2] + center[1]))
            fixed_ospa, assignment_ospa = frame_ospa(estimated, truth)
            for target in range(3):
                error = float(np.linalg.norm(estimated[target] - truth[target]))
                records.append(
                    {
                        "frame_index": frame_index,
                        "time_s": time_s,
                        "target": target + 1,
                        "estimated_x": float(estimated[target, 0]),
                        "estimated_y": float(estimated[target, 1]),
                        "estimated_vx": float(state_np[target, 1]),
                        "estimated_vy": float(state_np[target, 3]),
                        "truth_x": float(truth[target, 0]),
                        "truth_y": float(truth[target, 1]),
                        "position_error_m": error,
                        "covariance": cov_np[target].tolist(),
                    }
                )
            frame_diagnostics.append(
                {
                    "frame_index": frame_index,
                    "time_s": time_s,
                    "fixed_label_ospa_order2_m": fixed_ospa,
                    "assignment_ospa_order2_m": assignment_ospa,
                    "outer_iterations_executed": len(outer_rows),
                    "outer": outer_rows,
                }
            )
            print(json.dumps({"frame": frame_index, "fixed_ospa": fixed_ospa, "assignment_ospa": assignment_ospa, "outer": len(outer_rows)}, ensure_ascii=False), flush=True)
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}

    runtime = time.perf_counter() - started
    fixed_values = [row["fixed_label_ospa_order2_m"] for row in frame_diagnostics]
    assignment_values = [row["assignment_ospa_order2_m"] for row in frame_diagnostics]
    payload = {
        "claim_status": "fig15_strict_dbn_lanm_reconstruction",
        "completion_status": "failed" if failure else "completed",
        "failure": failure,
        "paper": "Zhang et al., Wideband Multitarget Tracking Based on Dynamic Bayesian Network Learning in an Acoustic Sensor Array Network, 2022",
        "implemented_equations": [21, 22, 24, 25, 26, 44, 45],
        "implemented_algorithms": ["Algorithm 1", "Algorithm 2"],
        "excluded_repairs": ["GPS runtime correction", "grid search", "trust region", "state damping", "covariance eigenvalue floor", "post-hoc alignment", "truth-based relabeling"],
        "hessian_policy": "allow_indefinite_without_repair" if args.allow_indefinite_hessian else "abort_on_non_negative_eigenvalue",
        "config": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "paper_constants": {
            "nodes": list(PAPER_NODES), "node_to_ip": NODE_TO_IP, "channels_per_node": 19,
            "sensor_count": 152, "sample_rate_hz": args.sample_rate_hz,
            "snapshot_samples": args.snapshot_samples,
            "dft_length": dft_length,
            "snapshot_count_l": args.snapshot_count,
            "snapshot_stride": snapshot_stride,
            "max_dft_frequency_mismatch_hz": max_frequency_mismatch_hz,
            "frequency_grid_hz": "3,6,...,1497", "frequency_count": len(FREQUENCIES_HZ),
            "processing_height_m": PROCESSING_HEIGHT_M, "sound_speed_mps": SOUND_SPEED,
        },
        "protocol_ambiguities_exposed_as_switches": [
            "hop_samples/state_dt", "initial covariance", "process acceleration variance",
            "Gamma hyperparameter mapping from Eq. (44)", "fixed-prior vs author-cumulative VB update",
            "hidden-variable carry vs reset", "uniform vs field nonuniform z-arm",
            "paper fixed-height range vs sensor-z range", "node direction flags",
            "Eq. (44) snapshot count L",
            "printed Eq. (26) trace scaling versus Eq. (21)-consistent scaling",
            "common resampled clock vs node-native packet-second DFT",
        ],
        "device": {"gpu": args.gpu, "name": torch.cuda.get_device_name(args.gpu), "precision": args.precision},
        "source_hashes": source_hashes,
        "nod_sha256": sha256(args.nod),
        "gps_sha256": {str(target): sha256(args.gps_root / filename) for target, filename in GPS_FILES.items()},
        "initialization": initialization_diagnostics,
        "frames_completed": len(frame_diagnostics),
        "mean_fixed_label_ospa_order2_m": float(np.mean(fixed_values)) if fixed_values else None,
        "mean_assignment_ospa_order2_m": float(np.mean(assignment_values)) if assignment_values else None,
        "runtime_s": runtime,
        "records": records,
        "frame_diagnostics": frame_diagnostics,
        "warning": "Independent equation-level reconstruction; not an undisclosed author Fig. 15 program.",
    }
    write_json(args.output, payload)
    print(json.dumps({key: payload[key] for key in ("completion_status", "failure", "frames_completed", "mean_fixed_label_ospa_order2_m", "runtime_s")}, ensure_ascii=False, indent=2))
    if failure:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
