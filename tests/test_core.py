import math

import torch

from pce_assimilation.config import AlphaConfig
from pce_assimilation.evidence import AlphaEvidenceTracker, liu_quantile


def test_liu_quantile_is_symmetric() -> None:
    alpha = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
    quantile = liu_quantile(alpha)
    assert torch.allclose(quantile, -quantile.flip(0), atol=1.0e-12)
    assert float(quantile[1]) == 0.0


def test_cumulative_evidence_weights_are_normalized() -> None:
    tracker = AlphaEvidenceTracker.create(
        AlphaConfig(initial_nodes=3, max_nodes=3, evidence_decay=1.0),
        dtype=torch.float64,
    )
    first = tracker.update(torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64))
    second = tracker.update(torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64))

    assert math.isclose(float(second.sum()), 1.0, abs_tol=1.0e-12)
    assert torch.all(second > 0.0)
    assert float(second[-1]) > float(first[-1]) > float(first[0])
