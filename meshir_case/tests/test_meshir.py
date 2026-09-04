from __future__ import annotations

import numpy as np

from meshir.data import farthest_point_sampling, s1_spatial_folds
from meshir.evidence import entropy_project
from meshir.geometry import direct_toa
from meshir.rom import candidate_paths, fit_pod


def test_pod_shapes_and_candidate_paths() -> None:
    rng = np.random.default_rng(2)
    field = rng.standard_normal((128, 40))
    basis, coefficients, mean = fit_pod(field, 8)
    assert basis.shape == (40, 8)
    assert coefficients.shape == (128, 8)
    assert mean.shape == (40,)
    paths, params = candidate_paths(coefficients, 1000.0, [0.98, 1.0], [0.95, 1.05], 1.0)
    assert paths.shape == (2, 128, 8)
    assert params.shape == (2, 2)
    assert np.isfinite(paths).all()


def test_spatial_folds_and_fps_are_disjoint() -> None:
    x = np.linspace(-0.5, 0.5, 4)
    positions = np.asarray([[a, b, c] for a in x for b in x for c in [-0.2, 0.0, 0.2]])
    folds = s1_spatial_folds(positions)
    assert len(folds) == 4
    assert len(np.unique(np.concatenate(folds))) == len(positions)
    selected = farthest_point_sampling(positions, 12)
    assert len(np.unique(selected)) == 12


def test_entropy_projection_is_normalized() -> None:
    projected = entropy_project(np.asarray([0.9, 0.08, 0.02]), 0.9)
    assert np.isclose(projected.sum(), 1.0)
    assert np.all(projected >= 0)


def test_direct_toa_is_geometric() -> None:
    source = np.asarray([1.0, 2.0, 3.0])
    receivers = np.asarray([[1.0, 2.0, 3.0], [1.0, 2.0, 4.0]])
    toa = direct_toa(source, receivers, 2.0)
    assert np.allclose(toa, [0.0, 0.5])
