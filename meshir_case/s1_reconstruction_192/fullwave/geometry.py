from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def grid_mapping(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(positions, dtype=float)
    x = np.unique(np.round(positions[:, 0], 6))
    y = np.unique(np.round(positions[:, 1], 6))
    z = np.unique(np.round(positions[:, 2], 6))
    if (len(x), len(y), len(z)) != (21, 21, 9):
        raise ValueError(f"unexpected grid {(len(x), len(y), len(z))}")
    mapping = np.empty((len(z), len(y), len(x)), dtype=int)
    for microphone, point in enumerate(positions):
        ix = int(np.argmin(np.abs(x - point[0])))
        iy = int(np.argmin(np.abs(y - point[1])))
        iz = int(np.argmin(np.abs(z - point[2])))
        mapping[iz, iy, ix] = microphone
    if len(np.unique(mapping)) != len(positions):
        raise ValueError("grid mapping is not bijective")
    return mapping, x, y, z


def sparse_axis_indices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.rint(np.linspace(0, 20, 7)).astype(int)
    y = np.rint(np.linspace(0, 20, 7)).astype(int)
    z = np.rint(np.linspace(0, 8, 3)).astype(int)
    return x, y, z


def sparse_flat_indices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, z = sparse_axis_indices()
    triples = np.asarray([(iz, iy, ix) for iz in z for iy in y for ix in x], dtype=int)
    flat = np.ravel_multi_index(triples.T, (9, 21, 21))
    boundary = np.any((triples == np.asarray([0, 0, 0])) | (triples == np.asarray([8, 20, 20])), axis=1)
    return flat, flat[boundary], flat[~boundary]


def to_grid(field_tm: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    return np.asarray(field_tm)[:, mapping]


def interpolate_sparse_series(field_grid: np.ndarray) -> np.ndarray:
    x_idx, y_idx, z_idx = sparse_axis_indices()
    sparse = field_grid[:, z_idx][:, :, y_idx][:, :, :, x_idx]
    values = np.transpose(sparse, (1, 2, 3, 0))
    interpolator = RegularGridInterpolator(
        (z_idx.astype(float), y_idx.astype(float), x_idx.astype(float)),
        values, method="linear", bounds_error=True,
    )
    zz, yy, xx = np.meshgrid(np.arange(9), np.arange(21), np.arange(21), indexing="ij")
    points = np.column_stack([zz.reshape(-1), yy.reshape(-1), xx.reshape(-1)])
    output = interpolator(points).T.reshape(len(field_grid), 9, 21, 21)
    return output.astype(np.float32)


def boundary_mask() -> np.ndarray:
    mask = np.zeros((9, 21, 21), dtype=bool)
    mask[[0, -1], :, :] = True
    mask[:, [0, -1], :] = True
    mask[:, :, [0, -1]] = True
    return mask
