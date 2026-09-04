import math

import torch

from baoding_case import ALPHA_GRID, CHANNEL_GROUPS, NODE_IDS
from baoding_case.nearfield_audit import decimal_hms_seconds
from baoding_case.nearfield_tracker import denkf_update
from baoding_case.run_baoding import angle_residual, entropy_project, parse_nod, propagate, ray_direction, score_weights
from baoding_case.shuangyuan_dual_association import transform_pair


def test_protocol_constants() -> None:
    assert NODE_IDS == (1, 2, 3, 5, 6, 7, 8, 11, 13)
    assert ALPHA_GRID == (0.08, 0.22, 0.36, 0.50, 0.64, 0.78, 0.92)
    assert sorted(sum(CHANNEL_GROUPS.values(), ())) == list(range(1, 20))


def test_angle_residual_wraps() -> None:
    pred = torch.tensor([math.radians(359.0), math.radians(1.0)])
    obs = torch.tensor([math.radians(1.0), math.radians(359.0)])
    assert torch.allclose(angle_residual(pred, obs), torch.tensor([math.radians(-2.0), math.radians(2.0)]), atol=1e-6)


def test_decimal_hms_seconds() -> None:
    assert abs(decimal_hms_seconds(124929.080) - (12 * 3600 + 49 * 60 + 29.080)) < 1e-6


def test_dual_association_applies_frozen_azimuth_offset() -> None:
    transformed = transform_pair(((10.0, 30.0), (350.0, 40.0)), -1.0, 15.0, 1.0, -2.0)
    assert transformed == [(5.0, 28.0), (25.0, 38.0)]


def test_alpha_propagation_is_finite_and_monotonic_noise_scale() -> None:
    state = torch.zeros((2, 6), dtype=torch.float64)
    noise = torch.ones((2, 6), dtype=torch.float64)
    low = propagate(state, torch.full((2,), 0.08), 1.0, noise, 0.5, 12.0)
    high = propagate(state, torch.full((2,), 0.92), 1.0, noise, 0.5, 12.0)
    assert torch.isfinite(high).all()
    assert float(high[0, 3]) > float(low[0, 3])


def test_entropy_projection_preserves_normalization_and_floor() -> None:
    weights = entropy_project(torch.tensor([0.999, 0.001], dtype=torch.float64), 0.12)
    entropy = -(weights * weights.clamp_min(1e-12).log()).sum()
    assert torch.isclose(weights.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert float(entropy) >= 0.12 - 1e-6


def test_pce_apce_score_weights_are_finite() -> None:
    predicted = torch.zeros((len(ALPHA_GRID), 4, 4), dtype=torch.float64)
    observation = torch.zeros(4, dtype=torch.float64)
    for method in ("pce", "apce"):
        logits, weights = score_weights(method, torch.zeros(len(ALPHA_GRID), dtype=torch.float64), predicted, observation, 8.0)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(weights).all()
        assert torch.isclose(weights.sum(), torch.tensor(1.0, dtype=torch.float64))


def test_circular_denkf_update_is_finite() -> None:
    states = torch.tensor([
        [100.0, -1.0, 10.0, 0.0, 0.0, 0.0],
        [100.0, 1.0, 10.0, 0.0, 0.0, 0.0],
        [101.0, 0.0, 11.0, 0.0, 0.0, 0.0],
    ], dtype=torch.float64)
    nodes = {1: torch.zeros(3, dtype=torch.float64)}
    observation = torch.tensor([0.0, math.atan2(10.0, 100.0)], dtype=torch.float64)
    updated = denkf_update(states, observation, torch.tensor([0.01, 0.01], dtype=torch.float64), nodes, [1])
    assert updated.shape == states.shape
    assert torch.isfinite(updated).all()
