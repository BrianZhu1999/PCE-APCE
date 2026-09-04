"""Independent physical-plausibility audit for decoded VIV-PIV predictions.

This is an audit-only tool. It never feeds test fields back into the model and
never changes a trace. It checks decoded fields against the valid PIV domain,
while reporting the cylinder-interior pixels separately because they are
masked and are not a physical evaluation region.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import numpy as np
import torch

from .common import load_config
from .io import VIVCase, list_cases
from .rom import PODModel


CASES = ("0463", "0556", "0679", "0803", "1359")
METHODS = ("pce", "apce")


def _trace_path(
    root: pathlib.Path,
    case_id: str,
    method: str,
    seed: int,
    layout: str,
    *,
    include_ensemble_suffix: bool,
) -> pathlib.Path:
    ensemble_suffix = "_ens064" if include_ensemble_suffix else ""
    name = f"viv_{case_id}_{method}_seed{seed:03d}_layout{layout}{ensemble_suffix}_covfull_shr050.npz"
    return root / name


def _run_path(
    root: pathlib.Path,
    case_id: str,
    method: str,
    seed: int,
    layout: str,
    *,
    include_ensemble_suffix: bool,
) -> pathlib.Path:
    ensemble_suffix = "_ens064" if include_ensemble_suffix else ""
    name = f"viv_{case_id}_{method}_seed{seed:03d}_layout{layout}{ensemble_suffix}_covfull_shr050.json"
    return root / name


def _masked_extrema(values: np.ndarray, valid: np.ndarray) -> tuple[float, float, float, int]:
    selected = values[valid]
    if selected.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    return float(np.min(selected)), float(np.max(selected)), float(np.max(np.abs(selected))), int(selected.size)


def _percentile_sample(samples: list[np.ndarray], limit: int = 3_000_000) -> dict[str, float]:
    if not samples:
        return {"p001": float("nan"), "p01": float("nan"), "p50": float("nan"), "p99": float("nan"), "p999": float("nan")}
    merged = np.concatenate(samples)
    if merged.size > limit:
        indices = np.linspace(0, merged.size - 1, limit, dtype=np.int64)
        merged = merged[indices]
    quantiles = np.percentile(merged, [0.1, 1.0, 50.0, 99.0, 99.9])
    return {key: float(value) for key, value in zip(("p001", "p01", "p50", "p99", "p999"), quantiles)}


def _field_stats(
    field: np.ndarray,
    valid_pair: np.ndarray,
    dx: float,
    dy: float,
) -> dict[str, Any]:
    """Calculate finite-field, temporal and spatial diagnostics for one block."""
    valid = np.asarray(valid_pair, dtype=bool)
    finite = np.isfinite(field).all(axis=-1)
    good = valid & finite
    gradient_good = good.copy()
    gradient_good[:, :, 0] = False
    gradient_good[:, :, -1] = False
    gradient_good[:, 0, :] = False
    gradient_good[:, -1, :] = False
    gradient_good &= np.roll(good, 1, axis=1) & np.roll(good, -1, axis=1)
    gradient_good &= np.roll(good, 1, axis=2) & np.roll(good, -1, axis=2)
    u = field[..., 0]
    v = field[..., 1]
    speed = np.sqrt(u * u + v * v)
    kinetic = 0.5 * (u * u + v * v)
    gx_u = np.gradient(u, dx, axis=2)
    gy_u = np.gradient(u, dy, axis=1)
    gx_v = np.gradient(v, dx, axis=2)
    gy_v = np.gradient(v, dy, axis=1)
    divergence = gx_u + gy_v
    vorticity = gx_v - gy_u
    gradients = np.sqrt(gx_u * gx_u + gy_u * gy_u + gx_v * gx_v + gy_v * gy_v)
    out: dict[str, Any] = {}
    for name, values in (("u", u), ("v", v), ("speed", speed), ("kinetic_energy", kinetic),
                         ("divergence", divergence), ("vorticity", vorticity), ("gradient_norm", gradients)):
        selected = values[gradient_good if name in {"divergence", "vorticity", "gradient_norm"} else good]
        finite_selected = selected[np.isfinite(selected)]
        out[name] = {
            "min": float(np.min(finite_selected)) if finite_selected.size else float("nan"),
            "max": float(np.max(finite_selected)) if finite_selected.size else float("nan"),
            "mean": float(np.mean(finite_selected)) if finite_selected.size else float("nan"),
            "rms": float(np.sqrt(np.mean(finite_selected * finite_selected))) if finite_selected.size else float("nan"),
            "nonfinite_count": int(selected.size - finite_selected.size),
            "negative_count": int(np.count_nonzero(finite_selected < 0.0)) if name in {"speed", "kinetic_energy"} else 0,
        }
    out["valid_count"] = int(np.count_nonzero(good))
    return out


def audit_case(
    case: VIVCase,
    pod: PODModel,
    latent: np.ndarray,
    training_min: np.ndarray,
    training_max: np.ndarray,
    sensor_flat: np.ndarray,
    device: torch.device,
    block: int = 16,
) -> dict[str, Any]:
    height = case.y_mm.size
    width = case.x_mm.size
    dx = float(np.median(np.diff(case.x_mm))) / 1000.0
    dy = float(np.median(np.diff(case.y_mm))) / 1000.0
    latent_finite = bool(np.isfinite(latent).all())
    physical: dict[str, Any] = {
        "latent_shape": list(latent.shape),
        "latent_finite": latent_finite,
        "valid_frames": int(case.time_s.size),
        "dt_s": float(case.dt_s),
        "time_start_s": float(case.time_s[0]),
        "time_end_s": float(case.time_s[-1]),
        "physical_extrema": {
            label: {
                "min": {"u": float("inf"), "v": float("inf"), "speed": float("inf"), "kinetic_energy": float("inf")},
                "max": {"u": float("-inf"), "v": float("-inf"), "speed": float("-inf"), "kinetic_energy": float("-inf")},
            }
            for label in ("truth", "prediction")
        },
        "training_range_exceedance": {"u": 0, "v": 0, "scalar_total": 0},
        "valid_scalar_total": 0,
        "valid_prediction_nonfinite": 0,
        "invalid_region_prediction_nonfinite": 0,
        "invalid_region_prediction_finite": 0,
        "error_partition": {
            "observed_error_square": 0.0,
            "observed_truth_square": 0.0,
            "observed_count": 0,
            "unobserved_error_square": 0.0,
            "unobserved_truth_square": 0.0,
            "unobserved_count": 0,
        },
        "spatial": {"truth": [], "prediction": []},
        "temporal": {"truth_max_jump_m_per_s": 0.0, "prediction_max_jump_m_per_s": 0.0,
                      "truth_rms_jump_m_per_s": 0.0, "prediction_rms_jump_m_per_s": 0.0},
        "samples": {"truth_u": [], "truth_v": [], "prediction_u": [], "prediction_v": [],
                    "truth_speed": [], "prediction_speed": [], "prediction_divergence": [], "truth_divergence": []},
    }
    jump_truth_sum = 0.0
    jump_pred_sum = 0.0
    jump_count = 0
    previous_truth: np.ndarray | None = None
    previous_prediction: np.ndarray | None = None
    previous_valid: np.ndarray | None = None
    basis_t = torch.as_tensor(pod.basis, dtype=torch.float32, device=device)
    mean_t = torch.as_tensor(pod.mean, dtype=torch.float32, device=device)
    for start, values, valid_flat in case.iter_physical(block=block):
        stop = start + values.shape[0]
        truth = values.reshape(values.shape[0], height, width, 2).astype(np.float64)
        valid = valid_flat.reshape(values.shape[0], height, width, 2)[..., 0]
        prediction_flat = (
            torch.as_tensor(latent[start:stop], dtype=torch.float32, device=device) @ basis_t.mT + mean_t[None, :]
        ).detach().cpu().numpy().astype(np.float64)
        prediction = prediction_flat.reshape(values.shape[0], height, width, 2)
        prediction_finite = np.isfinite(prediction).all(axis=-1)
        physical["valid_prediction_nonfinite"] += int(np.count_nonzero(valid & ~prediction_finite))
        physical["invalid_region_prediction_nonfinite"] += int(np.count_nonzero(~valid & ~prediction_finite))
        physical["invalid_region_prediction_finite"] += int(np.count_nonzero(~valid & prediction_finite))

        for label, field in (("truth", truth), ("prediction", prediction)):
            stats = _field_stats(field, valid, dx, dy)
            physical["spatial"][label].append(stats)
            for component, index in (("u", 0), ("v", 1)):
                selected = field[..., index][valid & np.isfinite(field[..., index])]
                if selected.size:
                    physical["physical_extrema"][label]["min"][component] = min(physical["physical_extrema"][label]["min"][component], float(selected.min()))
                    physical["physical_extrema"][label]["max"][component] = max(physical["physical_extrema"][label]["max"][component], float(selected.max()))
                    physical["samples"][f"{label}_{component}"].append(selected[::max(1, selected.size // 50000)])
            speed = np.linalg.norm(field, axis=-1)
            kinetic = 0.5 * np.sum(field * field, axis=-1)
            for name, array in (("speed", speed), ("kinetic_energy", kinetic)):
                selected = array[valid & np.isfinite(array)]
                if selected.size:
                    physical["physical_extrema"][label]["min"][name] = min(physical["physical_extrema"][label]["min"][name], float(selected.min()))
                    physical["physical_extrema"][label]["max"][name] = max(physical["physical_extrema"][label]["max"][name], float(selected.max()))
                    if name == "speed":
                        physical["samples"][f"{label}_speed"].append(selected[::max(1, selected.size // 50000)])
            divergence = stats["divergence"]
            if label == "truth":
                physical["samples"]["truth_divergence"].append(np.asarray([divergence["rms"]]))
            else:
                physical["samples"]["prediction_divergence"].append(np.asarray([divergence["rms"]]))

        valid_scalar = np.repeat(valid[..., None], 2, axis=-1).reshape(values.shape[0], -1)
        prediction_flat = prediction.reshape(values.shape[0], -1)
        truth_flat = truth.reshape(values.shape[0], -1)
        for component in (0, 1):
            component_valid = valid[..., None] & np.asarray([component == 0, component == 1])[None, None, None, :]
            selected = prediction[component_valid]
            key = "u" if component == 0 else "v"
            physical["training_range_exceedance"][key] += int(
                np.count_nonzero((selected < training_min[component]) | (selected > training_max[component]))
            )
        physical["valid_scalar_total"] += int(np.count_nonzero(valid_scalar))
        observed_valid = valid_scalar[:, sensor_flat]
        observed_residual = prediction_flat[:, sensor_flat] - truth_flat[:, sensor_flat]
        partition = physical["error_partition"]
        partition["observed_error_square"] += float(np.sum(observed_residual[observed_valid] ** 2))
        partition["observed_truth_square"] += float(np.sum(truth_flat[:, sensor_flat][observed_valid] ** 2))
        partition["observed_count"] += int(np.count_nonzero(observed_valid))
        unobserved_valid = valid_scalar.copy()
        unobserved_valid[:, sensor_flat] = False
        residual_flat = prediction_flat - truth_flat
        partition["unobserved_error_square"] += float(np.sum(residual_flat[unobserved_valid] ** 2))
        partition["unobserved_truth_square"] += float(np.sum(truth_flat[unobserved_valid] ** 2))
        partition["unobserved_count"] += int(np.count_nonzero(unobserved_valid))

        if previous_truth is not None and previous_valid is not None and previous_prediction is not None:
            truth_sequence = np.concatenate((previous_truth[None, ...], truth), axis=0)
            prediction_sequence = np.concatenate((previous_prediction[None, ...], prediction), axis=0)
            valid_sequence = np.concatenate((previous_valid[None, ...], valid), axis=0)
        else:
            truth_sequence = truth
            prediction_sequence = prediction
            valid_sequence = valid
        if truth_sequence.shape[0] > 1:
            common = valid_sequence[1:] & valid_sequence[:-1]
            common &= np.isfinite(truth_sequence[1:]).all(axis=-1) & np.isfinite(truth_sequence[:-1]).all(axis=-1)
            common &= np.isfinite(prediction_sequence[1:]).all(axis=-1) & np.isfinite(prediction_sequence[:-1]).all(axis=-1)
            truth_jump = np.linalg.norm(np.diff(truth_sequence, axis=0), axis=-1)[common]
            pred_jump = np.linalg.norm(np.diff(prediction_sequence, axis=0), axis=-1)[common]
            if truth_jump.size:
                physical["temporal"]["truth_max_jump_m_per_s"] = max(physical["temporal"]["truth_max_jump_m_per_s"], float(truth_jump.max() / case.dt_s))
                physical["temporal"]["prediction_max_jump_m_per_s"] = max(physical["temporal"]["prediction_max_jump_m_per_s"], float(pred_jump.max() / case.dt_s))
                jump_truth_sum += float(np.sum((truth_jump / case.dt_s) ** 2))
                jump_pred_sum += float(np.sum((pred_jump / case.dt_s) ** 2))
                jump_count += int(truth_jump.size)
        previous_truth = truth[-1]
        previous_prediction = prediction[-1]
        previous_valid = valid[-1]

    physical["temporal"]["truth_rms_jump_m_per_s"] = float(np.sqrt(jump_truth_sum / max(jump_count, 1)))
    physical["temporal"]["prediction_rms_jump_m_per_s"] = float(np.sqrt(jump_pred_sum / max(jump_count, 1)))
    for key, values in physical["samples"].items():
        physical["samples"][key] = _percentile_sample(values)
    physical["training_range_exceedance"]["fraction"] = float(
        (physical["training_range_exceedance"]["u"] + physical["training_range_exceedance"]["v"])
        / max(physical["valid_scalar_total"], 1)
    )
    physical["training_range_exceedance"]["scalar_total"] = int(physical["valid_scalar_total"])
    partition = physical["error_partition"]
    partition["observed_nrmse"] = float(np.sqrt(partition["observed_error_square"] / max(partition["observed_truth_square"], 1e-30)))
    partition["unobserved_nrmse"] = float(np.sqrt(partition["unobserved_error_square"] / max(partition["unobserved_truth_square"], 1e-30)))
    physical["spatial_summary"] = {}
    for label in ("truth", "prediction"):
        rows = physical["spatial"][label]
        for name in ("divergence", "vorticity", "gradient_norm"):
            values = [float(row[name]["rms"]) for row in rows]
            physical["spatial_summary"].setdefault(label, {})[name] = {
                "mean_block_rms": float(np.mean(values)),
                "max_block_rms": float(np.max(values)),
            }
    del physical["spatial"]
    return physical


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit physical plausibility of VIV-PIV decoded predictions.")
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--variant", default="rank256_stride1")
    parser.add_argument("--layout", default="adaptive_fullfield_valid")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", choices=METHODS, action="append", dest="methods")
    parser.add_argument(
        "--omit-ensemble-suffix",
        action="store_true",
        help="Use run ids without the explicit _ens064 token.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = pathlib.Path(config["data_root"])
    result_root = pathlib.Path(config["output_root"])
    model_root = result_root / "models" / args.variant
    run_root = result_root / "runs" / args.variant
    cases = list_cases(root)
    pod = PODModel.load(model_root / "pod_model.npz")
    selected_methods = tuple(args.methods) if args.methods else METHODS
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    training_min = np.full(2, np.inf, dtype=np.float64)
    training_max = np.full(2, -np.inf, dtype=np.float64)
    training_count = 0
    for case_id in config["train_cases"]:
        case = VIVCase.open(cases[str(case_id)])
        for _start, values, valid in case.iter_physical(block=32):
            shaped = values.reshape(values.shape[0], case.y_mm.size, case.x_mm.size, 2)
            valid_grid = valid.reshape(values.shape[0], case.y_mm.size, case.x_mm.size, 2)[..., 0]
            for component in range(2):
                selected = shaped[..., component][valid_grid]
                training_min[component] = min(training_min[component], float(selected.min()))
                training_max[component] = max(training_max[component], float(selected.max()))
                training_count += int(selected.size)
    layout_dir = model_root / "sensor_layouts" / args.layout
    first_layout = np.load(layout_dir / f"case_{config['test_cases'][0]}.npz", allow_pickle=False)
    sensor_flat = np.asarray(first_layout["sensor_flat_indices"], dtype=np.int64)
    output: dict[str, Any] = {
        "protocol": {
            "test_cases": list(CASES),
            "methods": list(selected_methods),
            "variant": args.variant,
            "layout": args.layout,
            "seed": args.seed,
            "training_physical_range": {"u": [float(training_min[0]), float(training_max[0])], "v": [float(training_min[1]), float(training_max[1])]},
            "training_valid_scalar_count": training_count,
            "note": "Values inside the cylinder mask are reported separately and are not treated as physical evaluation points.",
        },
        "runs": {},
    }
    for case_id in CASES:
        case = VIVCase.open(cases[case_id])
        output["runs"][case_id] = {}
        for method in selected_methods:
            trace = np.load(
                _trace_path(
                    run_root / "traces",
                    case_id,
                    method,
                    args.seed,
                    args.layout,
                    include_ensemble_suffix=not args.omit_ensemble_suffix,
                ),
                allow_pickle=False,
            )
            latent = np.asarray(trace["latent_estimate"], dtype=np.float64)
            output["runs"][case_id][method] = audit_case(case, pod, latent, training_min, training_max, sensor_flat, device)
            numeric_keys = [key for key in trace.files if np.issubdtype(np.asarray(trace[key]).dtype, np.number)]
            trace_finite = all(np.isfinite(np.asarray(trace[key])).all() for key in numeric_keys)
            weights = np.asarray(trace["weights"], dtype=np.float64) if "weights" in trace.files else np.empty((0, 0))
            output["runs"][case_id][method]["trace_integrity"] = {
                "all_numeric_arrays_finite": bool(trace_finite),
                "weight_min": float(weights.min()) if weights.size else None,
                "weight_max_sum_error": float(np.max(np.abs(weights.sum(axis=1) - 1.0))) if weights.size else None,
            }
            run_path = _run_path(
                run_root / "runs",
                case_id,
                method,
                args.seed,
                args.layout,
                include_ensemble_suffix=not args.omit_ensemble_suffix,
            )
            if run_path.exists():
                payload = json.loads(run_path.read_text(encoding="utf-8"))
                output["runs"][case_id][method]["reported_metrics"] = {
                    key: payload.get(key) for key in ("evaluation_nrmse", "unobserved_full_field_physical_nrmse", "kinetic_energy_nrmse", "kinetic_energy_correlation", "coverage_90", "normalized_crps", "blackout_mean_nrmse")
                }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
