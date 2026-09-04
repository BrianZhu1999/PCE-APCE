from __future__ import annotations

import numpy as np
from scipy.interpolate import RBFInterpolator, griddata


SHAPE_ZYX = (9, 21, 21)


def full_grid() -> tuple[np.ndarray, np.ndarray]:
    z = np.linspace(-0.2, 0.2, SHAPE_ZYX[0])
    y = np.linspace(-0.5, 0.5, SHAPE_ZYX[1])
    x = np.linspace(-0.5, 0.5, SHAPE_ZYX[2])
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    return np.column_stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]), np.stack([xx, yy, zz], axis=-1)


def boundary_mask() -> np.ndarray:
    iz, iy, ix = np.indices(SHAPE_ZYX)
    return (iz == 0) | (iz == SHAPE_ZYX[0] - 1) | (iy == 0) | (iy == SHAPE_ZYX[1] - 1) | (ix == 0) | (ix == SHAPE_ZYX[2] - 1)


def _farthest_with_seeds(points: np.ndarray, count: int, seed_indices: list[int]) -> np.ndarray:
    if count < len(seed_indices) or count > len(points):
        raise ValueError("invalid farthest-point sampling count")
    selected = list(dict.fromkeys(int(i) for i in seed_indices))
    distances = np.full(len(points), np.inf, dtype=float)
    for index in selected:
        distances = np.minimum(distances, np.linalg.norm(points - points[index], axis=1))
    while len(selected) < count:
        candidate = int(np.argmax(distances))
        selected.append(candidate)
        distances = np.minimum(distances, np.linalg.norm(points - points[candidate], axis=1))
    return np.asarray(selected, dtype=int)


def _nearest_seed(points: np.ndarray, target: np.ndarray) -> int:
    return int(np.argmin(np.linalg.norm(points - np.asarray(target, dtype=float), axis=1)))


def sample_192() -> dict[str, np.ndarray]:
    """Deterministic geometry-only 128-boundary/64-interior sampling."""
    points, _ = full_grid()
    boundary = boundary_mask().reshape(-1)
    boundary_points = points[boundary]
    interior_points = points[~boundary]

    zmin, zmax = points[:, 2].min(), points[:, 2].max()
    xmin, xmax = points[:, 0].min(), points[:, 0].max()
    ymin, ymax = points[:, 1].min(), points[:, 1].max()
    face_centres = np.asarray([
        [0.0, 0.0, zmin], [0.0, 0.0, zmax],
        [xmin, 0.0, 0.0], [xmax, 0.0, 0.0],
        [0.0, ymin, 0.0], [0.0, ymax, 0.0],
    ])
    boundary_seeds = [_nearest_seed(boundary_points, target) for target in face_centres]
    boundary_local = _farthest_with_seeds(boundary_points, 128, boundary_seeds)
    interior_seeds = [_nearest_seed(interior_points, [0.0, 0.0, z]) for z in np.linspace(-0.15, 0.15, 7)]
    interior_local = _farthest_with_seeds(interior_points, 64, interior_seeds)

    boundary_flat = np.flatnonzero(boundary)[boundary_local]
    interior_flat = np.flatnonzero(~boundary)[interior_local]
    if len(np.intersect1d(boundary_flat, interior_flat)) or len(np.union1d(boundary_flat, interior_flat)) != 192:
        raise RuntimeError("192-point sampling is not disjoint")
    return {
        "points": points,
        "boundary_flat": boundary_flat,
        "interior_flat": interior_flat,
        "boundary_all_flat": np.flatnonzero(boundary),
        "interior_all_flat": np.flatnonzero(~boundary),
    }


def to_grid(field_tm: np.ndarray) -> np.ndarray:
    return np.asarray(field_tm).reshape((len(field_tm),) + SHAPE_ZYX)


def boundary_series_from_sparse(truth: np.ndarray, boundary_flat: np.ndarray) -> np.ndarray:
    """Causally usable boundary completion from only selected boundary points."""
    time_count = len(truth)
    grid_points, _ = full_grid()
    boundary_all = np.flatnonzero(boundary_mask().reshape(-1))
    source_points = grid_points[boundary_flat]
    target_points = grid_points[boundary_all]
    values = truth.reshape(time_count, -1)[:, boundary_flat]
    completed = np.zeros((time_count, np.prod(SHAPE_ZYX)), dtype=np.float32)
    for time_index in range(time_count):
        linear = griddata(source_points, values[time_index], target_points, method="linear")
        nearest = griddata(source_points, values[time_index], target_points, method="nearest")
        linear = np.where(np.isfinite(linear), linear, nearest)
        completed[time_index, boundary_all] = linear.astype(np.float32)
    return completed.reshape((time_count,) + SHAPE_ZYX)


def boundary_candidate_series_from_sparse(
    truth: np.ndarray,
    boundary_flat: np.ndarray,
    scales_m: list[float],
) -> np.ndarray:
    """Boundary closure candidates from selected boundary measurements only.

    Candidate 0 is piecewise-linear scattered interpolation. The remaining
    candidates are Gaussian spatial smoothers. The measured boundary samples
    are reinserted exactly for every candidate, so candidates differ only at
    unmeasured boundary locations.
    """
    time_count = len(truth)
    grid_points, _ = full_grid()
    boundary_all = np.flatnonzero(boundary_mask().reshape(-1))
    source_points = grid_points[boundary_flat]
    target_points = grid_points[boundary_all]
    values = truth.reshape(time_count, -1)[:, boundary_flat]
    output = np.zeros((time_count, len(scales_m) + 1, np.prod(SHAPE_ZYX)), dtype=np.float32)
    linear_series = boundary_series_from_sparse(truth, boundary_flat).reshape(time_count, -1)
    output[:, 0, boundary_all] = linear_series[:, boundary_all]
    distances2 = np.sum((target_points[:, None, :] - source_points[None, :, :]) ** 2, axis=2)
    local_lookup = {int(flat): i for i, flat in enumerate(boundary_all)}
    exact_rows = np.asarray([local_lookup[int(flat)] for flat in boundary_flat], dtype=int)
    for scale_index, scale in enumerate(scales_m):
        length = max(float(scale), 1e-6)
        weights = np.exp(-0.5 * distances2 / (length * length))
        weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-30)
        completed = values @ weights.T
        completed[:, exact_rows] = values
        output[:, scale_index + 1, boundary_all] = completed.astype(np.float32)
    return output.reshape((time_count, len(scales_m) + 1) + SHAPE_ZYX)


def _causal_wavefront_completion(
    values: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_position: np.ndarray,
    linear: np.ndarray,
    sample_rate: float,
    sound_speed: float,
    neighbours: int,
    spatial_scale_m: float,
) -> np.ndarray:
    """Complete a boundary using only observations available at or before t.

    For a target farther from the source than an observed boundary point, the
    observed waveform is delayed so both values correspond to the same source
    emission time. Targets without a causal neighbour fall back to the
    instantaneous linear completion.
    """
    time_count = len(values)
    source_distance = np.linalg.norm(source_points - source_position[None], axis=1)
    target_distance = np.linalg.norm(target_points - source_position[None], axis=1)
    spatial_distance2 = np.sum((target_points[:, None] - source_points[None]) ** 2, axis=2)
    delay = np.rint(
        (target_distance[:, None] - source_distance[None, :])
        * float(sample_rate) / float(sound_speed)
    ).astype(int)
    output = np.asarray(linear, dtype=np.float32).copy()
    neighbour_count = min(int(neighbours), len(source_points))
    scale2 = max(float(spatial_scale_m) ** 2, 1e-12)
    for target in range(len(target_points)):
        causal = np.flatnonzero(delay[target] >= 0)
        if not len(causal):
            continue
        order = causal[np.argsort(spatial_distance2[target, causal])[:neighbour_count]]
        weights = np.exp(-0.5 * spatial_distance2[target, order] / scale2)
        distance_gain = source_distance[order] / max(target_distance[target], 1e-6)
        weights = weights / np.maximum(weights.sum(), 1e-30)
        aligned = np.zeros(time_count, dtype=np.float64)
        available = np.zeros(time_count, dtype=np.float64)
        for weight, gain, source_index in zip(weights, distance_gain, order):
            lag = int(delay[target, source_index])
            scaled_weight = float(weight * np.clip(gain, 0.5, 2.0))
            if lag == 0:
                aligned += scaled_weight * values[:, source_index]
                available += scaled_weight
            elif lag < time_count:
                aligned[lag:] += scaled_weight * values[:-lag, source_index]
                available[lag:] += scaled_weight
        valid = available > 1e-12
        output[valid, target] = (aligned[valid] / available[valid]).astype(np.float32)
    return output


def _phase_compensate(series: np.ndarray, strength: float) -> np.ndarray:
    output = np.asarray(series, dtype=np.float32).copy()
    output[1:] += float(strength) * (series[1:] - series[:-1])
    return output


def causal_boundary_candidate_series_from_sparse(
    truth: np.ndarray,
    boundary_flat: np.ndarray,
    grid_positions: np.ndarray,
    source_position: np.ndarray,
    sample_rate: float,
    sound_speed: float,
) -> tuple[np.ndarray, list[str]]:
    """Build eight causal boundary-closure candidates.

    Candidate zero remains the original piecewise-linear closure. The other
    candidates combine source-delay-aligned boundary observations with the
    linear closure or apply a small causal phase correction. No held-out field
    value is read when constructing any candidate.
    """
    time_count = len(truth)
    boundary_all = np.flatnonzero(boundary_mask().reshape(-1))
    source_points = np.asarray(grid_positions, dtype=float)[boundary_flat]
    target_points = np.asarray(grid_positions, dtype=float)[boundary_all]
    values = truth.reshape(time_count, -1)[:, boundary_flat]
    linear_grid = boundary_series_from_sparse(truth, boundary_flat).reshape(time_count, -1)
    linear = linear_grid[:, boundary_all]
    aligned4 = _causal_wavefront_completion(
        values, source_points, target_points, np.asarray(source_position, dtype=float), linear,
        sample_rate, sound_speed, neighbours=4, spatial_scale_m=0.18,
    )
    aligned8 = _causal_wavefront_completion(
        values, source_points, target_points, np.asarray(source_position, dtype=float), linear,
        sample_rate, sound_speed, neighbours=8, spatial_scale_m=0.24,
    )
    blend25 = 0.75 * linear + 0.25 * aligned8
    blend50 = 0.50 * linear + 0.50 * aligned8
    blend75 = 0.25 * linear + 0.75 * aligned8
    candidates = [
        linear,
        aligned4,
        aligned8,
        blend25,
        blend50,
        blend75,
        _phase_compensate(linear, 0.20),
        _phase_compensate(blend50, 0.20),
    ]
    labels = [
        "linear",
        "wavefront_k4",
        "wavefront_k8",
        "linear75_wavefront25",
        "linear50_wavefront50",
        "linear25_wavefront75",
        "linear_phase_lead_0.20",
        "blend50_phase_lead_0.20",
    ]
    local_lookup = {int(flat): index for index, flat in enumerate(boundary_all)}
    exact_rows = np.asarray([local_lookup[int(flat)] for flat in boundary_flat], dtype=int)
    output = np.zeros((time_count, len(candidates), np.prod(SHAPE_ZYX)), dtype=np.float32)
    for candidate_index, candidate in enumerate(candidates):
        candidate = np.asarray(candidate, dtype=np.float32).copy()
        candidate[:, exact_rows] = values
        output[:, candidate_index, boundary_all] = candidate
    return output.reshape((time_count, len(candidates)) + SHAPE_ZYX), labels


def rbf_boundary_candidate_series_from_sparse(
    truth: np.ndarray,
    boundary_flat: np.ndarray,
    grid_positions: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Build boundary candidates selected by observed-boundary cross-validation.

    The first candidate is the original linear completion. Other candidates
    are spatial RBF closures and convex blends with the best Gaussian RBF
    closure. All measured boundary samples are reinserted exactly.
    """
    time_count = len(truth)
    boundary_all = np.flatnonzero(boundary_mask().reshape(-1))
    source_points = np.asarray(grid_positions, dtype=float)[boundary_flat]
    target_points = np.asarray(grid_positions, dtype=float)[boundary_all]
    values = truth.reshape(time_count, -1)[:, boundary_flat]
    linear = boundary_series_from_sparse(truth, boundary_flat).reshape(time_count, -1)[:, boundary_all]
    exact_rows = np.asarray([np.flatnonzero(boundary_all == int(flat))[0] for flat in boundary_flat], dtype=int)

    specifications = [
        ("linear", None, None),
        ("rbf_gaussian_e4", "gaussian", 4.0),
        ("rbf_gaussian_e2", "gaussian", 2.0),
        ("rbf_inverse_multiquadric_e4", "inverse_multiquadric", 4.0),
        ("rbf_cubic", "cubic", None),
    ]
    candidate_fields: list[np.ndarray] = [linear]
    for label, kernel, epsilon in specifications[1:]:
        kwargs: dict[str, object] = {"kernel": kernel, "smoothing": 1e-4}
        if epsilon is not None:
            kwargs["epsilon"] = epsilon
        interpolator = RBFInterpolator(source_points, np.eye(len(source_points)), **kwargs)
        weights = np.asarray(interpolator(target_points), dtype=np.float64)
        field = values @ weights.T
        field[:, exact_rows] = values
        candidate_fields.append(field.astype(np.float32))

    best_rbf = candidate_fields[1]
    candidate_fields.extend([
        (0.75 * linear + 0.25 * best_rbf).astype(np.float32),
        (0.50 * linear + 0.50 * best_rbf).astype(np.float32),
        (0.25 * linear + 0.75 * best_rbf).astype(np.float32),
    ])
    labels = [name for name, _, _ in specifications] + [
        "linear75_rbf25",
        "linear50_rbf50",
        "linear25_rbf75",
    ]
    output = np.zeros((time_count, len(candidate_fields), np.prod(SHAPE_ZYX)), dtype=np.float32)
    for index, candidate in enumerate(candidate_fields):
        candidate = np.asarray(candidate, dtype=np.float32).copy()
        candidate[:, exact_rows] = values
        output[:, index, boundary_all] = candidate
    return output.reshape((time_count, len(candidate_fields)) + SHAPE_ZYX), labels


def scattered_series_from_sparse(truth: np.ndarray, sparse_flat: np.ndarray) -> np.ndarray:
    """Static spatial baseline using only the 192 measured points at each time."""
    time_count = len(truth)
    grid_points, _ = full_grid()
    source_points = grid_points[sparse_flat]
    values = truth.reshape(time_count, -1)[:, sparse_flat]
    output = np.zeros((time_count, np.prod(SHAPE_ZYX)), dtype=np.float32)
    for time_index in range(time_count):
        linear = griddata(source_points, values[time_index], grid_points, method="linear")
        nearest = griddata(source_points, values[time_index], grid_points, method="nearest")
        output[time_index] = np.where(np.isfinite(linear), linear, nearest).astype(np.float32)
    return output.reshape((time_count,) + SHAPE_ZYX)
