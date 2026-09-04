from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .identification import IdentifiedModel, identify_models, target_mode


def liu_quantile(alpha: float) -> float:
    alpha = float(np.clip(alpha, 1e-8, 1.0 - 1e-8))
    return math.sqrt(3.0) / math.pi * math.log(alpha / (1.0 - alpha))


def causal_force_envelope(force_scaled: np.ndarray, sample_rate_hz: float, window_seconds: float) -> np.ndarray:
    window = max(1, int(round(sample_rate_hz * window_seconds)))
    squared = np.asarray(force_scaled, dtype=float) ** 2
    cumulative = np.cumsum(np.concatenate([[0.0], squared]))
    moving = (cumulative[window:] - cumulative[:-window]) / window
    prefix = np.asarray([np.mean(squared[:index + 1]) for index in range(window - 1)])
    rms = np.sqrt(np.concatenate([prefix, moving]))
    reference = max(float(np.percentile(rms, 95)), 1e-12)
    return np.clip(rms / reference, 0.0, 1.0)


def fit_bootstrap_path(
    model: IdentifiedModel,
    level_payloads: dict[int, dict[str, np.ndarray]],
    config: dict,
    seed: int = 20260817,
) -> dict:
    identification = config["identification"]
    rng = np.random.default_rng(seed)
    levels = [3, 5, 7]
    force_rms = np.asarray([np.sqrt(np.mean(level_payloads[level]["force"] ** 2)) for level in levels])
    envelope = force_rms / max(float(force_rms.max()), 1e-12)
    base = target_mode(model, float(identification["target_mode_hz"]), float(identification["target_mode_tolerance_hz"]))
    def period_modal_estimates(payload: dict[str, np.ndarray]) -> np.ndarray:
        force = payload["force"]
        acceleration = payload["acceleration"]
        window = np.hanning(force.shape[1])
        frequencies = np.fft.rfftfreq(force.shape[1], 1.0 / model.sample_rate_hz)
        band = (frequencies >= float(config["filter_band_hz"][0])) & (frequencies <= float(config["filter_band_hz"][1]))
        output = []
        for period in range(force.shape[0]):
            u = np.fft.rfft(force[period] * window)
            y = np.fft.rfft(acceleration[period] * window[:, None], axis=0)
            h1 = y * np.conj(u)[:, None] / np.maximum(np.abs(u)[:, None] ** 2, 1e-12)
            magnitude = np.mean(np.abs(h1), axis=1)
            band_indices = np.flatnonzero(band)
            peak_index = int(band_indices[np.argmax(magnitude[band_indices])])
            peak_value = float(magnitude[peak_index])
            half = peak_value / math.sqrt(2.0)
            left = peak_index
            while left > band_indices[0] and magnitude[left] >= half:
                left -= 1
            right = peak_index
            while right < band_indices[-1] and magnitude[right] >= half:
                right += 1
            bandwidth = max(float(frequencies[right] - frequencies[left]), float(frequencies[1] - frequencies[0]))
            frequency = float(frequencies[peak_index])
            damping = bandwidth / max(2.0 * frequency, 1e-12)
            output.append([frequency, max(damping, 1e-5)])
        return np.asarray(output, dtype=float)

    rows = []
    period_estimates = {level: period_modal_estimates(level_payloads[level]) for level in levels}
    replicates = int(identification["bootstrap_replicates"])
    for replicate in range(replicates):
        frequencies = []
        damping = []
        valid = True
        for level in levels:
            payload = level_payloads[level]
            period_count = payload["force"].shape[0]
            count = max(3, int(math.ceil(float(identification["bootstrap_period_fraction"]) * period_count)))
            selected = rng.choice(period_count, size=count, replace=True)
            selected_parameters = period_estimates[level][selected]
            frequencies.append(float(np.mean(selected_parameters[:, 0])))
            damping.append(float(np.mean(selected_parameters[:, 1])))
        if not valid:
            continue
        x = np.concatenate([[0.0], envelope])
        frequency_values = np.asarray([base["frequency_hz"], *frequencies])
        damping_values = np.asarray([max(base["damping_ratio"], 1e-5), *damping])
        frequency_slope = float(np.polyfit(x, frequency_values / base["frequency_hz"] - 1.0, 1)[0])
        damping_slope = float(np.polyfit(x, np.log(damping_values / max(base["damping_ratio"], 1e-5)), 1)[0])
        rows.append([frequency_slope, damping_slope])
    samples = np.asarray(rows, dtype=float)
    if samples.shape[0] < max(8, replicates // 2):
        raise RuntimeError(f"only {samples.shape[0]}/{replicates} bootstrap modal fits succeeded")
    mean = samples.mean(axis=0)
    covariance = np.cov(samples.T)
    values, vectors = np.linalg.eigh(covariance)
    index = int(np.argmax(values))
    direction = vectors[:, index]
    if direction[0] < 0.0:
        direction = -direction
    scale = math.sqrt(max(float(values[index]), 1e-12))
    return {
        "parameter_names": ["fractional_frequency_slope", "log_damping_slope"],
        "bootstrap_seed": seed,
        "requested_replicates": replicates,
        "successful_replicates": int(samples.shape[0]),
        "levels": levels,
        "force_rms": force_rms.tolist(),
        "normalized_envelope": envelope.tolist(),
        "base_target_mode": base,
        "bootstrap_samples": samples.tolist(),
        "mean": mean.tolist(),
        "covariance": covariance.tolist(),
        "principal_direction": direction.tolist(),
        "principal_scale": scale,
        "explained_fraction": float(values[index] / max(values.sum(), 1e-12)),
    }


@dataclass
class ModalCandidateFamily:
    model: IdentifiedModel
    path: dict
    quantization_bins: int
    maximum_frequency_scale: float
    maximum_damping_log_scale: float
    _values: np.ndarray = field(init=False, repr=False)
    _vectors: np.ndarray = field(init=False, repr=False)
    _inverse: np.ndarray = field(init=False, repr=False)
    _target_indices: tuple[int, int] = field(init=False, repr=False)
    _cache: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        values, vectors = np.linalg.eig(self.model.a)
        if np.linalg.cond(vectors) > 1e8:
            raise RuntimeError("base model eigenvectors are ill-conditioned")
        dt = 1.0 / self.model.sample_rate_hz
        frequencies = np.abs(np.imag(np.log(values) / dt)) / (2.0 * np.pi)
        positive = np.flatnonzero(np.imag(values) > 1e-9)
        first = int(positive[np.argmin(np.abs(frequencies[positive] - 7.3))])
        second = int(np.argmin(np.abs(values - np.conj(values[first]))))
        self._values = values
        self._vectors = vectors
        self._inverse = np.linalg.inv(vectors)
        self._target_indices = (first, second)

    def parameters(self, alpha: float) -> np.ndarray:
        mean = np.asarray(self.path["mean"], dtype=float)
        direction = np.asarray(self.path["principal_direction"], dtype=float)
        values = mean + float(self.path["principal_scale"]) * direction * liu_quantile(alpha)
        values[0] = np.clip(values[0], -self.maximum_frequency_scale, self.maximum_frequency_scale)
        values[1] = np.clip(values[1], -self.maximum_damping_log_scale, self.maximum_damping_log_scale)
        return values

    def matrix(self, alpha: float, envelope: float) -> np.ndarray:
        envelope_bin = int(np.clip(round(envelope * (self.quantization_bins - 1)), 0, self.quantization_bins - 1))
        key = (round(float(alpha), 10), envelope_bin)
        if key in self._cache:
            return self._cache[key]
        effective_envelope = envelope_bin / max(self.quantization_bins - 1, 1)
        frequency_slope, damping_slope = self.parameters(alpha)
        transformed = self._values.copy()
        dt = 1.0 / self.model.sample_rate_hz
        for index in self._target_indices:
            value = self._values[index]
            pole = np.log(value) / dt
            base_natural = abs(pole)
            base_damping = -np.real(pole) / max(base_natural, 1e-12)
            frequency_factor = max(0.5, 1.0 + frequency_slope * effective_envelope)
            damping_factor = math.exp(damping_slope * effective_envelope)
            target_imag = np.imag(pole) * frequency_factor
            target_real = -max(base_damping * damping_factor, 1e-5) * base_natural * frequency_factor
            transformed[index] = np.exp((target_real + 1j * target_imag) * dt)
        matrix = self._vectors @ np.diag(transformed) @ self._inverse
        if np.max(np.abs(np.imag(matrix))) > 1e-7:
            raise RuntimeError("candidate transform did not preserve a real matrix")
        result = np.real(matrix)
        if np.max(np.abs(np.linalg.eigvals(result))) >= 1.0:
            raise RuntimeError("candidate transform is unstable")
        self._cache[key] = result
        return result

    def audit_grid(self, grid: np.ndarray) -> list[dict]:
        rows = []
        for alpha in np.asarray(grid, dtype=float):
            parameters = self.parameters(float(alpha))
            radii = [float(np.max(np.abs(np.linalg.eigvals(self.matrix(float(alpha), envelope))))) for envelope in (0.0, 0.5, 1.0)]
            rows.append({
                "alpha": float(alpha),
                "frequency_slope": float(parameters[0]),
                "damping_log_slope": float(parameters[1]),
                "maximum_spectral_radius": max(radii),
                "stable": bool(max(radii) < 1.0),
            })
        return rows
