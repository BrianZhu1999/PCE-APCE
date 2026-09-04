from __future__ import annotations

import numpy as np
from scipy import signal

from .data import FullMSineLevel


def causal_preprocess(level: FullMSineLevel, config: dict) -> tuple[dict[str, np.ndarray], dict]:
    native_rate = float(config["native_rate_hz"])
    processed_rate = float(config["processed_rate_hz"])
    period_samples = int(config["period_samples_native"])
    period_count = int(config["period_count"])
    discard_periods = int(config["discard_periods"])
    if not np.isclose(level.sample_rate_hz, native_rate):
        raise ValueError(f"Level {level.level}: sample rate {level.sample_rate_hz} != {native_rate}")
    expected = period_samples * period_count
    if level.force.size != expected:
        raise ValueError(f"Level {level.level}: expected {expected} rows, found {level.force.size}")
    stride = int(round(native_rate / processed_rate))
    if not np.isclose(native_rate / processed_rate, stride):
        raise ValueError("native/processed sample-rate ratio must be an integer")
    matrix = np.column_stack([level.force, level.voltage, level.acceleration])
    baseline = matrix[:period_samples].mean(axis=0, keepdims=True)
    centered = matrix - baseline
    sos = signal.butter(
        int(config["filter_order"]),
        tuple(float(value) for value in config["model_filter_band_hz"]),
        btype="bandpass",
        fs=native_rate,
        output="sos",
    )
    filtered = signal.sosfilt(sos, centered, axis=0)
    start = discard_periods * period_samples
    retained = filtered[start::stride]
    samples_per_period = period_samples // stride
    retained_periods = period_count - discard_periods
    retained = retained.reshape(retained_periods, samples_per_period, matrix.shape[1])
    time = np.arange(samples_per_period, dtype=np.float64) / processed_rate
    force = retained[:, :, 0]
    voltage = retained[:, :, 1]
    acceleration = retained[:, :, 2:5]
    audit = {
        "level": level.level,
        "source_member": level.source_member,
        "method": "causal_model_bandpass_then_stride_decimation",
        "causal": True,
        "zero_phase_filtering_used": False,
        "model_filter_band_hz": list(config["model_filter_band_hz"]),
        "score_filter_band_hz": list(config["filter_band_hz"]),
        "filter_order": int(config["filter_order"]),
        "native_rate_hz": native_rate,
        "processed_rate_hz": processed_rate,
        "discarded_periods": discard_periods,
        "discarded_seconds": discard_periods * period_samples / native_rate,
        "required_filter_transient_seconds": float(config["filter_transient_seconds"]),
        "retained_periods": retained_periods,
        "samples_per_period": samples_per_period,
        "all_values_finite": bool(np.isfinite(retained).all()),
        "force_rms_by_period": np.sqrt(np.mean(force ** 2, axis=1)).tolist(),
        "acceleration_rms_by_period": np.sqrt(np.mean(acceleration ** 2, axis=1)).tolist(),
    }
    return {"time": time, "force": force, "voltage": voltage, "acceleration": acceleration}, audit
