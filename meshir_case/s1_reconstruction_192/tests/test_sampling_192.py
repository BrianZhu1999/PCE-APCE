from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geometry_192 import (
    causal_boundary_candidate_series_from_sparse,
    rbf_boundary_candidate_series_from_sparse,
    sample_192,
)


def test_sampling_counts_and_disjointness() -> None:
    sample = sample_192()
    assert len(sample["points"]) == 3969
    assert len(sample["boundary_all_flat"]) == 1442
    assert len(sample["interior_all_flat"]) == 2527
    assert len(sample["boundary_flat"]) == 128
    assert len(sample["interior_flat"]) == 64
    assert len(np.intersect1d(sample["boundary_flat"], sample["interior_flat"])) == 0


def test_interior_sampling_covers_all_interior_z_layers() -> None:
    sample = sample_192()
    iz = np.asarray(np.unravel_index(sample["interior_flat"], (9, 21, 21))[0])
    assert set(iz.tolist()) == set(range(1, 8))


def test_causal_candidates_preserve_measurements_and_ignore_future() -> None:
    sample = sample_192()
    rng = np.random.default_rng(17)
    truth = rng.standard_normal((8, 9, 21, 21)).astype(np.float32)
    candidates, labels = causal_boundary_candidate_series_from_sparse(
        truth,
        sample["boundary_flat"],
        sample["points"],
        np.asarray([0.0, -2.0, 0.0]),
        sample_rate=16000.0,
        sound_speed=343.0,
    )
    assert candidates.shape == (8, 8, 9, 21, 21)
    assert len(labels) == 8
    candidate_flat = candidates.reshape(8, 8, -1)
    truth_flat = truth.reshape(8, -1)
    np.testing.assert_allclose(
        candidate_flat[:, :, sample["boundary_flat"]],
        np.repeat(truth_flat[:, None, sample["boundary_flat"]], 8, axis=1),
    )

    changed = truth.copy().reshape(8, -1)
    changed[5:, sample["boundary_flat"]] += 100.0
    changed_candidates, _ = causal_boundary_candidate_series_from_sparse(
        changed.reshape(truth.shape),
        sample["boundary_flat"],
        sample["points"],
        np.asarray([0.0, -2.0, 0.0]),
        sample_rate=16000.0,
        sound_speed=343.0,
    )
    np.testing.assert_allclose(candidates[:5], changed_candidates[:5])


def test_rbf_candidates_preserve_observed_boundary_values() -> None:
    sample = sample_192()
    rng = np.random.default_rng(23)
    truth = rng.standard_normal((5, 9, 21, 21)).astype(np.float32)
    candidates, labels = rbf_boundary_candidate_series_from_sparse(
        truth, sample["boundary_flat"], sample["points"]
    )
    assert candidates.shape == (5, 8, 9, 21, 21)
    assert labels[0] == "linear"
    candidate_flat = candidates.reshape(5, 8, -1)
    truth_flat = truth.reshape(5, -1)
    np.testing.assert_allclose(
        candidate_flat[:, :, sample["boundary_flat"]],
        np.repeat(truth_flat[:, None, sample["boundary_flat"]], 8, axis=1),
        rtol=1e-6,
        atol=1e-6,
    )
