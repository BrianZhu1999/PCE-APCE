from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy import signal


def load_npy_zip(archive: Path, prefix: str, name: str) -> np.ndarray:
    with ZipFile(archive) as handle:
        return np.load(BytesIO(handle.read(f"{prefix}/{name}")))


def list_ir_names(archive: Path, prefix: str) -> list[str]:
    with ZipFile(archive) as handle:
        return sorted(
            name for name in handle.namelist()
            if name.startswith(f"{prefix}/ir_") and name.endswith(".npy")
        )


def load_ir_matrix(archive: Path, prefix: str, mic_count: int, source_count: int) -> np.ndarray:
    output = None
    with ZipFile(archive) as handle:
        names = [
            name for name in handle.namelist()
            if name.startswith(f"{prefix}/ir_") and name.endswith(".npy")
        ]
        if len(names) != mic_count:
            raise ValueError(f"expected {mic_count} IR files, found {len(names)}")
        for name in names:
            index = int(Path(name).stem.split("_")[-1])
            values = np.load(BytesIO(handle.read(name))).astype(np.float32)
            if values.shape != (source_count, 32768):
                raise ValueError(f"unexpected {name} shape {values.shape}")
            if output is None:
                output = np.zeros((source_count, mic_count, values.shape[1]), dtype=np.float32)
            output[:, index, :] = values
    if output is None or not np.isfinite(output).all():
        raise ValueError("empty or non-finite IR matrix")
    return output


def causal_downsample(rir: np.ndarray, native_rate: int, processed_rate: int, taps: int, cutoff: float) -> np.ndarray:
    if native_rate % processed_rate:
        raise ValueError("processed rate must divide native rate")
    factor = native_rate // processed_rate
    kernel = signal.firwin(taps, cutoff, fs=native_rate)
    filtered = signal.lfilter(kernel, [1.0], rir, axis=-1)
    return filtered[..., ::factor].astype(np.float32)


def causal_log_energy(rir: np.ndarray, sample_rate: float, tau: float, floor: float) -> np.ndarray:
    decay = np.exp(-1.0 / max(sample_rate * tau, 1.0))
    energy = signal.lfilter([1.0 - decay], [1.0, -decay], rir ** 2, axis=-1)
    return (10.0 * np.log10(energy + floor)).astype(np.float32)


def s1_spatial_folds(positions: np.ndarray, count: int = 4) -> list[np.ndarray]:
    if count != 4:
        raise ValueError("the pilot uses four fixed spatial folds")
    x_mid = float(np.median(positions[:, 0]))
    y_mid = float(np.median(positions[:, 1]))
    labels = (positions[:, 0] >= x_mid).astype(int) + 2 * (positions[:, 1] >= y_mid).astype(int)
    return [np.flatnonzero(labels == index) for index in range(4)]


def farthest_point_sampling(positions: np.ndarray, count: int, seed_index: int | None = None) -> np.ndarray:
    if count <= 0 or count > len(positions):
        raise ValueError("invalid sampling count")
    selected = [int(np.argmin(np.linalg.norm(positions - positions.mean(axis=0), axis=1))) if seed_index is None else seed_index]
    distance = np.linalg.norm(positions - positions[selected[0]], axis=1)
    while len(selected) < count:
        candidate = int(np.argmax(distance))
        selected.append(candidate)
        distance = np.minimum(distance, np.linalg.norm(positions - positions[candidate], axis=1))
    return np.asarray(selected, dtype=int)
