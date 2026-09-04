from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from f16_gvt.data import FullMSineLevel
from f16_gvt.preprocess import causal_preprocess


HERE = Path(__file__).resolve().parents[1]


def test_causal_preprocess_retains_eight_periods_at_100_hz() -> None:
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    count = int(config["period_samples_native"]) * int(config["period_count"])
    time = np.arange(count) / float(config["native_rate_hz"])
    force = np.sin(2.0 * np.pi * 7.3 * time)
    acceleration = np.column_stack([
        np.sin(2.0 * np.pi * 7.3 * time + phase) for phase in (0.1, 0.3, 0.5)
    ])
    level = FullMSineLevel(1, force, force * 0.01, acceleration, 400.0, "synthetic")
    payload, audit = causal_preprocess(level, config)
    assert payload["force"].shape == (8, 2048)
    assert payload["acceleration"].shape == (8, 2048, 3)
    assert audit["causal"] is True
    assert audit["zero_phase_filtering_used"] is False
    assert audit["discarded_periods"] == 1
    assert np.isfinite(payload["acceleration"]).all()


def test_preprocess_rejects_wrong_length() -> None:
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    level = FullMSineLevel(1, np.zeros(100), np.zeros(100), np.zeros((100, 3)), 400.0, "synthetic")
    try:
        causal_preprocess(level, config)
    except ValueError as error:
        assert "expected" in str(error)
    else:
        raise AssertionError("wrong-length data was accepted")
