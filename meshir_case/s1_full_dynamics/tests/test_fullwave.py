from __future__ import annotations

import numpy as np
import torch

from fullwave.geometry import grid_mapping, sparse_flat_indices
from fullwave.model import cfl_number, laplacian


def test_sparse_layout_counts() -> None:
    observed, boundary, interior = sparse_flat_indices()
    assert len(observed) == 147
    assert len(boundary) == 122
    assert len(interior) == 25
    assert len(np.intersect1d(boundary, interior)) == 0


def test_cfl_is_stable() -> None:
    assert cfl_number(354.0, 1.0 / 16000.0, 0.05) < 1.0


def test_constant_field_has_zero_laplacian() -> None:
    values = torch.ones((2, 3, 9, 21, 21))
    assert torch.allclose(laplacian(values), torch.zeros_like(values))


def test_grid_mapping_is_bijective() -> None:
    x = np.linspace(-0.5, 0.5, 21)
    y = np.linspace(-0.5, 0.5, 21)
    z = np.linspace(-0.2, 0.2, 9)
    positions = np.asarray([[xx, yy, zz] for zz in z for yy in y for xx in x])
    mapping, *_ = grid_mapping(positions)
    assert mapping.shape == (9, 21, 21)
    assert len(np.unique(mapping)) == 3969
