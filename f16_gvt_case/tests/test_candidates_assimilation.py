from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from f16_gvt.assimilation import run_two_pass
from f16_gvt.candidates import ModalCandidateFamily, liu_quantile
from f16_gvt.identification import IdentifiedModel


HERE = Path(__file__).resolve().parents[1]


def synthetic_model() -> IdentifiedModel:
    rate = 100.0
    radius = 0.995
    angle = 2.0 * np.pi * 7.3 / rate
    a = radius * np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return IdentifiedModel(
        a=a,
        b=np.asarray([[0.05], [0.02]]),
        c=np.asarray([[1.0, 0.0], [0.6, 0.3], [0.2, 0.9]]),
        d=np.zeros((3, 1)),
        q=np.eye(2) * 1e-4,
        r=np.eye(3) * 1e-3,
        s=np.zeros((2, 3)),
        input_scale=1.0,
        output_scale=np.ones(3),
        sample_rate_hz=rate,
        order=2,
    )


def family(model: IdentifiedModel) -> ModalCandidateFamily:
    path = {
        "mean": [-0.03, 0.10],
        "principal_direction": [0.7, 0.714142842],
        "principal_scale": 0.02,
    }
    return ModalCandidateFamily(model, path, 32, 0.35, 1.5)


def test_liu_quantile_direction() -> None:
    assert liu_quantile(0.05) < 0.0 < liu_quantile(0.95)
    assert abs(liu_quantile(0.5)) < 1e-12


def test_candidate_grid_is_real_and_stable() -> None:
    model = synthetic_model()
    candidate = family(model)
    for alpha in (0.05, 0.5, 0.95):
        matrix = candidate.matrix(alpha, 1.0)
        assert np.isrealobj(matrix)
        assert np.max(np.abs(np.linalg.eigvals(matrix))) < 1.0


def test_apce_blackout_freezes_weight_updates() -> None:
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    config["candidates"]["coarse_alpha"] = [0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95]
    config["candidates"]["local_points"] = 5
    config["assimilation"]["ensemble_size_smoke"] = 8
    model = synthetic_model()
    candidate = family(model)
    count = 160
    time = np.arange(count) / model.sample_rate_hz
    force = np.sin(2.0 * np.pi * 7.3 * time)
    state = np.zeros((count, 2))
    for index in range(1, count):
        state[index] = model.a @ state[index - 1] + model.b[:, 0] * force[index]
    acceleration = state @ model.c.T
    rng = np.random.default_rng(42)
    initial = rng.standard_normal((7, 8, 2)).astype(np.float32)
    noise = rng.standard_normal((count, 7, 8, 2)).astype(np.float32)
    result = run_two_pass(
        "APCE", model, candidate, force, acceleration, [0, 1], config,
        "cpu", initial, noise, "blackout",
    ).local
    start, end = config["assimilation"]["blackout_fraction"]
    left, right = start * time[-1], end * time[-1]
    assert not np.any((result.weight_time >= left) & (result.weight_time <= right))
    assert np.isfinite(result.final_weights).all()
    assert np.all(result.final_weights >= 0.0)
    assert np.isclose(result.final_weights.sum(), 1.0)
