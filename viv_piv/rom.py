from __future__ import annotations

import dataclasses
import math
import pathlib
from collections.abc import Iterable

import numpy as np
import torch

from .io import VIVCase


@dataclasses.dataclass
class PODModel:
    mean: np.ndarray
    basis: np.ndarray
    singular_values: np.ndarray
    explained_fraction: np.ndarray
    reference_x_mm: np.ndarray
    reference_y_mm: np.ndarray
    evaluation_flat_indices: np.ndarray

    @property
    def rank(self) -> int:
        return int(self.basis.shape[1])

    @property
    def state_dim(self) -> int:
        return int(self.basis.shape[0])

    def observation_matrix(self, flat_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        flat_indices = np.asarray(flat_indices, dtype=np.int64)
        return self.mean[flat_indices], self.basis[flat_indices]

    def save(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            mean=self.mean.astype(np.float32),
            basis=self.basis.astype(np.float32),
            singular_values=self.singular_values.astype(np.float64),
            explained_fraction=self.explained_fraction.astype(np.float64),
            reference_x_mm=self.reference_x_mm.astype(np.float32),
            reference_y_mm=self.reference_y_mm.astype(np.float32),
            evaluation_flat_indices=self.evaluation_flat_indices.astype(np.int64),
        )

    @classmethod
    def load(cls, path: pathlib.Path) -> "PODModel":
        data = np.load(path, allow_pickle=False)
        return cls(**{key: data[key] for key in cls.__dataclass_fields__})


@dataclasses.dataclass
class DMDCCandidate:
    case_id: str
    reduced_velocity: float
    a: np.ndarray
    b: np.ndarray
    q_diag: np.ndarray
    residual_rms: float
    spectral_radius: float

    def save(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            case_id=np.asarray(self.case_id),
            reduced_velocity=np.asarray(self.reduced_velocity),
            a=self.a.astype(np.float64),
            b=self.b.astype(np.float64),
            q_diag=self.q_diag.astype(np.float64),
            residual_rms=np.asarray(self.residual_rms),
            spectral_radius=np.asarray(self.spectral_radius),
        )

    @classmethod
    def load(cls, path: pathlib.Path) -> "DMDCCandidate":
        data = np.load(path, allow_pickle=False)
        return cls(
            case_id=str(data["case_id"].item()),
            reduced_velocity=float(data["reduced_velocity"]),
            a=np.asarray(data["a"], dtype=np.float64),
            b=np.asarray(data["b"], dtype=np.float64),
            q_diag=np.asarray(data["q_diag"], dtype=np.float64),
            residual_rms=float(data["residual_rms"]),
            spectral_radius=float(data["spectral_radius"]),
        )


def evaluation_indices(reference: VIVCase, config: dict[str, object], count: int) -> np.ndarray:
    diameter_mm = float(config["cylinder_diameter_m"]) * 1000.0
    candidates: list[int] = []
    for iy, y in enumerate(reference.y_mm):
        if not -2.2 <= y / diameter_mm <= 2.2:
            continue
        for ix, x in enumerate(reference.x_mm):
            if not 0.75 <= x / diameter_mm <= 8.5:
                continue
            for component in (0, 1):
                flat = (iy * reference.x_mm.size + ix) * 2 + component
                candidates.append(flat)
    if count > len(candidates):
        raise ValueError("Requested more evaluation dimensions than eligible wake dimensions")
    positions = np.linspace(0, len(candidates) - 1, count, dtype=int)
    return np.asarray([candidates[index] for index in positions], dtype=np.int64)


def training_mean(cases: Iterable[VIVCase], *, block: int = 32, frame_stride: int = 1) -> tuple[np.ndarray, np.ndarray, int]:
    cases = list(cases)
    state_dim = int(np.prod(cases[0].velocities.shape[1:]))
    total = np.zeros(state_dim, dtype=np.float64)
    counts = np.zeros(state_dim, dtype=np.int64)
    sample_count = 0
    for case in cases:
        for _start, values, valid in case.iter_physical(block=block):
            values = values[::frame_stride]
            valid = valid[::frame_stride]
            total += np.sum(np.where(valid, values, 0.0), axis=0, dtype=np.float64)
            counts += np.sum(valid, axis=0, dtype=np.int64)
            sample_count += values.shape[0]
    mean = np.divide(total, counts, out=np.zeros_like(total), where=counts > 0).astype(np.float32)
    return mean, counts, sample_count


def _centered_blocks(cases: list[VIVCase], mean: np.ndarray, block: int, frame_stride: int):
    for case in cases:
        for start, values, valid in case.iter_physical(block=block):
            local = np.arange(values.shape[0]) + start
            keep = local % frame_stride == 0
            centered = np.where(valid[keep], values[keep] - mean[None, :], 0.0).astype(np.float32)
            if centered.size:
                yield case.case_id, local[keep], centered


def randomized_pod(
    cases: list[VIVCase],
    mean: np.ndarray,
    rank: int,
    *,
    frame_stride: int,
    block: int,
    seed: int,
    device: torch.device,
    oversample: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_count = sum(math.ceil(case.time_s.size / frame_stride) for case in cases)
    width = min(rank + oversample, sample_count)
    generator = np.random.default_rng(seed)
    omega = generator.normal(size=(sample_count, width)).astype(np.float32) / math.sqrt(width)
    state_dim = mean.size
    y = torch.zeros((state_dim, width), dtype=torch.float32, device=device)
    cursor = 0
    for _case_id, _indices, centered in _centered_blocks(cases, mean, block, frame_stride):
        count = centered.shape[0]
        x = torch.as_tensor(centered, dtype=torch.float32, device=device)
        om = torch.as_tensor(omega[cursor : cursor + count], dtype=torch.float32, device=device)
        y.add_(x.mT @ om)
        cursor += count
    if cursor != sample_count:
        raise RuntimeError(f"Randomized POD sample mismatch: {cursor} != {sample_count}")
    q, _ = torch.linalg.qr(y, mode="reduced")
    b = torch.empty((width, sample_count), dtype=torch.float32, device=device)
    cursor = 0
    for _case_id, _indices, centered in _centered_blocks(cases, mean, block, frame_stride):
        count = centered.shape[0]
        x = torch.as_tensor(centered, dtype=torch.float32, device=device)
        b[:, cursor : cursor + count] = q.mT @ x.mT
        cursor += count
    u_small, singular, _vh = torch.linalg.svd(b, full_matrices=False)
    basis = q @ u_small[:, :rank]
    singular_np = singular.detach().cpu().numpy().astype(np.float64)
    explained = np.cumsum(singular_np**2) / max(float(np.sum(singular_np**2)), 1e-30)
    return basis.detach().cpu().numpy().astype(np.float32), singular_np, explained


def project_case(case: VIVCase, pod: PODModel, *, block: int, device: torch.device) -> np.ndarray:
    basis = torch.as_tensor(pod.basis, dtype=torch.float32, device=device)
    output = np.empty((case.time_s.size, pod.rank), dtype=np.float32)
    for start, values, valid in case.iter_physical(block=block):
        centered = np.where(valid, values - pod.mean[None, :], 0.0).astype(np.float32)
        projection = torch.as_tensor(centered, dtype=torch.float32, device=device) @ basis
        output[start : start + centered.shape[0]] = projection.detach().cpu().numpy()
    return output


def control_inputs(case: VIVCase, diameter_m: float) -> np.ndarray:
    displacement = case.cyl_displ_m / diameter_m
    velocity = np.gradient(displacement, case.time_s)
    return np.column_stack([np.ones_like(displacement), displacement, velocity]).astype(np.float64)


def fit_dmdc(case: VIVCase, coefficients: np.ndarray, diameter_m: float, ridge: float) -> DMDCCandidate:
    z = np.asarray(coefficients, dtype=np.float64)
    control = control_inputs(case, diameter_m)
    design = np.column_stack([z[:-1], control[:-1]])
    target = z[1:]
    gram = design.T @ design
    penalty = ridge * np.trace(gram) / max(gram.shape[0], 1)
    coefficients_matrix = np.linalg.solve(gram + penalty * np.eye(gram.shape[0]), design.T @ target)
    rank = z.shape[1]
    a = coefficients_matrix[:rank].T
    b = coefficients_matrix[rank:].T
    residual = target - design @ coefficients_matrix
    q_diag = np.var(residual, axis=0, ddof=1).clip(1e-12)
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(a))))
    return DMDCCandidate(
        case_id=case.case_id,
        reduced_velocity=case.reduced_velocity,
        a=a,
        b=b,
        q_diag=q_diag,
        residual_rms=float(np.sqrt(np.mean(residual**2))),
        spectral_radius=spectral_radius,
    )
