from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from nfoursid.nfoursid import NFourSID


def psd(matrix: np.ndarray, floor: float) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    scale = max(float(np.max(np.abs(values))), 1.0)
    values = np.maximum(values, floor * scale)
    return (vectors * values[None, :]) @ vectors.T


def stabilize_near_unit_poles(matrix: np.ndarray, target_radius: float = 0.9995) -> tuple[np.ndarray, dict]:
    values, vectors = np.linalg.eig(np.asarray(matrix, dtype=float))
    transformed = values.copy()
    projected = []
    for index, value in enumerate(values):
        radius = abs(value)
        if radius >= 1.0:
            transformed[index] = value / radius * target_radius
            projected.append({"index": index, "original_radius": float(radius), "new_radius": target_radius})
    if not projected:
        return np.asarray(matrix, dtype=float), {"projected": []}
    result = vectors @ np.diag(transformed) @ np.linalg.inv(vectors)
    if np.max(np.abs(np.imag(result))) > 1e-7:
        raise RuntimeError("near-unit stabilization did not preserve a real matrix")
    return np.real(result), {"projected": projected}


@dataclass
class IdentifiedModel:
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    d: np.ndarray
    q: np.ndarray
    r: np.ndarray
    s: np.ndarray
    input_scale: float
    output_scale: np.ndarray
    sample_rate_hz: float
    order: int

    def scaled_input(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) / self.input_scale

    def scaled_output(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) / self.output_scale

    def physical_output(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.output_scale

    def modes(self) -> list[dict]:
        output = []
        dt = 1.0 / self.sample_rate_hz
        for value in np.linalg.eigvals(self.a):
            if np.imag(value) <= 1e-9:
                continue
            pole = np.log(value) / dt
            natural = abs(pole)
            output.append({
                "frequency_hz": abs(float(np.imag(pole))) / (2.0 * np.pi),
                "damping_ratio": -float(np.real(pole)) / max(natural, 1e-12),
                "magnitude": float(abs(value)),
                "stable": bool(abs(value) < 1.0),
            })
        return sorted(output, key=lambda row: row["frequency_hz"])


def target_mode(model: IdentifiedModel, target_hz: float, tolerance_hz: float) -> dict:
    modes = model.modes()
    if not modes:
        raise RuntimeError("identified model has no oscillatory mode")
    selected = min(modes, key=lambda row: abs(row["frequency_hz"] - target_hz))
    if abs(selected["frequency_hz"] - target_hz) > tolerance_hz:
        raise RuntimeError(f"nearest mode {selected['frequency_hz']:.3f} Hz is outside target tolerance")
    return selected


def _frame(force: np.ndarray, acceleration: np.ndarray) -> tuple[pd.DataFrame, float, np.ndarray]:
    input_scale = max(float(np.std(force)), 1e-12)
    output_scale = np.maximum(np.std(acceleration, axis=0), 1e-12)
    data = {"force": force / input_scale}
    data.update({f"acc{index + 1}": acceleration[:, index] / output_scale[index] for index in range(3)})
    return pd.DataFrame(data), input_scale, output_scale


def identify_models(
    force: np.ndarray,
    acceleration: np.ndarray,
    orders: list[int],
    block_rows: int,
    covariance_floor: float,
    sample_rate_hz: float,
) -> dict[int, IdentifiedModel]:
    frame, input_scale, output_scale = _frame(force, acceleration)
    identifier = NFourSID(
        frame,
        output_columns=["acc1", "acc2", "acc3"],
        input_columns=["force"],
        num_block_rows=block_rows,
    )
    identifier.subspace_identification()
    models = {}
    for order in orders:
        state_space, covariance = identifier.system_identification(rank=int(order))
        covariance = np.asarray(covariance, dtype=float)
        output_dim = 3
        r = psd(covariance[:output_dim, :output_dim], covariance_floor)
        q = psd(covariance[output_dim:, output_dim:], covariance_floor)
        stabilized_a, stabilization = stabilize_near_unit_poles(np.asarray(state_space.a, dtype=float))
        models[int(order)] = IdentifiedModel(
            a=stabilized_a,
            b=np.asarray(state_space.b, dtype=float),
            c=np.asarray(state_space.c, dtype=float),
            d=np.asarray(state_space.d, dtype=float),
            q=q,
            r=r,
            s=np.asarray(covariance[output_dim:, :output_dim], dtype=float),
            input_scale=input_scale,
            output_scale=output_scale,
            sample_rate_hz=sample_rate_hz,
            order=int(order),
        )
    return models


def kalman_prediction(model: IdentifiedModel, force: np.ndarray, acceleration: np.ndarray, warmup: int) -> dict:
    u = model.scaled_input(force)
    y = model.scaled_output(acceleration)
    x = np.zeros(model.order)
    p = np.eye(model.order)
    identity = np.eye(model.order)
    prediction = np.zeros_like(y)
    innovations = np.zeros_like(y)
    for index in range(len(y)):
        y_pred = model.c @ x + model.d[:, 0] * u[index]
        prediction[index] = y_pred
        innovations[index] = y[index] - y_pred
        covariance = model.c @ p @ model.c.T + model.r
        gain = p @ model.c.T @ np.linalg.pinv(covariance)
        filtered = x + gain @ innovations[index]
        left = identity - gain @ model.c
        p_filtered = left @ p @ left.T + gain @ model.r @ gain.T
        x = model.a @ filtered + model.b[:, 0] * u[index]
        p = psd(model.a @ p_filtered @ model.a.T + model.q, 1e-12)
    physical = model.physical_output(prediction)
    target = acceleration[warmup:]
    estimate = physical[warmup:]
    nrmse = float(np.linalg.norm(estimate - target) / max(np.linalg.norm(target), 1e-12))
    return {
        "prediction": physical,
        "nrmse": nrmse,
        "per_channel_nrmse": (
            np.sqrt(np.mean((estimate - target) ** 2, axis=0))
            / np.maximum(np.sqrt(np.mean(target ** 2, axis=0)), 1e-12)
        ).tolist(),
        "innovation_covariance_scaled": np.cov(innovations[warmup:].T).tolist(),
    }


def evaluate_periods(model: IdentifiedModel, force_periods: np.ndarray, acceleration_periods: np.ndarray, warmup: int) -> dict:
    rows = []
    for period in range(force_periods.shape[0]):
        result = kalman_prediction(model, force_periods[period], acceleration_periods[period], warmup)
        rows.append({"period": period, "nrmse": result["nrmse"], "per_channel_nrmse": result["per_channel_nrmse"]})
    return {"periods": rows, "mean_nrmse": float(np.mean([row["nrmse"] for row in rows]))}


def fit_select_refit(payload: dict[str, np.ndarray], config: dict) -> tuple[IdentifiedModel, dict]:
    identification = config["identification"]
    force_periods = payload["force"]
    acceleration_periods = payload["acceleration"]
    train = np.asarray(identification["train_periods"], dtype=int)
    validation = np.asarray(identification["validation_periods"], dtype=int)
    test = np.asarray(identification["test_periods"], dtype=int)
    train_models = identify_models(
        force_periods[train].reshape(-1),
        acceleration_periods[train].reshape(-1, 3),
        [int(value) for value in identification["orders"]],
        int(identification["num_block_rows"]),
        float(identification["covariance_floor"]),
        float(config["processed_rate_hz"]),
    )
    warmup = int(round(float(identification["warmup_seconds"]) * float(config["processed_rate_hz"])))
    diagnostics = []
    for order, model in train_models.items():
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(model.a))))
        validation_metrics = evaluate_periods(model, force_periods[validation], acceleration_periods[validation], warmup)
        test_metrics = evaluate_periods(model, force_periods[test], acceleration_periods[test], warmup)
        mode = None
        try:
            mode = target_mode(model, float(identification["target_mode_hz"]), float(identification["target_mode_tolerance_hz"]))
        except RuntimeError:
            pass
        diagnostics.append({
            "order": order,
            "spectral_radius": spectral_radius,
            "stable": bool(spectral_radius < 1.0),
            "target_mode": mode,
            "validation": validation_metrics,
            "test": test_metrics,
            "modes": model.modes(),
        })
    admitted = [row for row in diagnostics if row["stable"] and row["target_mode"] is not None and np.isfinite(row["validation"]["mean_nrmse"])]
    if not admitted:
        raise RuntimeError("no stable N4SID model contains the target 7.3 Hz mode")
    selected = min(admitted, key=lambda row: (row["validation"]["mean_nrmse"], row["order"]))
    final = identify_models(
        force_periods.reshape(-1),
        acceleration_periods.reshape(-1, 3),
        [int(selected["order"])],
        int(identification["num_block_rows"]),
        float(identification["covariance_floor"]),
        float(config["processed_rate_hz"]),
    )[int(selected["order"])]
    summary = {
        "selection_rule": "minimum validation-period prediction nRMSE among stable models containing the 7.3 Hz mode",
        "selected_order": int(selected["order"]),
        "order_diagnostics": diagnostics,
        "final_modes": final.modes(),
        "final_target_mode": target_mode(final, float(identification["target_mode_hz"]), float(identification["target_mode_tolerance_hz"])),
        "input_scale": final.input_scale,
        "output_scale": final.output_scale.tolist(),
    }
    return final, summary


def save_model(model: IdentifiedModel, summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "level1_base_model.npz",
        a=model.a, b=model.b, c=model.c, d=model.d,
        q=model.q, r=model.r, s=model.s,
        input_scale=np.asarray(model.input_scale),
        output_scale=model.output_scale,
        sample_rate_hz=np.asarray(model.sample_rate_hz),
        order=np.asarray(model.order),
    )
    (output_dir / "level1_identification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def load_model(output_dir: Path) -> IdentifiedModel:
    data = np.load(output_dir / "level1_base_model.npz")
    return IdentifiedModel(
        a=data["a"], b=data["b"], c=data["c"], d=data["d"],
        q=data["q"], r=data["r"], s=data["s"],
        input_scale=float(data["input_scale"]), output_scale=data["output_scale"],
        sample_rate_hz=float(data["sample_rate_hz"]), order=int(data["order"]),
    )
