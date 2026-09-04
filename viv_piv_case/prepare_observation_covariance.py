"""Estimate a shrinkage full observation covariance from training residuals."""
from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
import torch

from .common import load_config, software_environment, write_json
from .io import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare full VIV-PIV observation covariance.")
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--layout", required=True)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_config(args.config)
    variant = args.variant or f"rank{int(config['rank'])}_stride1"
    model_root = pathlib.Path(config["output_root"]) / "models" / variant
    layout_root = model_root / "sensor_layouts" / args.layout
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()

    observations: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    for case_id in config["train_cases"]:
        sensor = np.load(layout_root / f"case_{case_id}.npz", allow_pickle=False)
        latent = np.load(model_root / "coefficients" / f"case_{case_id}.npz", allow_pickle=False)
        values = np.asarray(sensor["sensor_observations"], dtype=np.float32)
        mean = np.asarray(sensor["sensor_mean"], dtype=np.float32)
        basis = np.asarray(sensor["sensor_basis"], dtype=np.float32)
        coefficients = np.asarray(latent["coefficients"], dtype=np.float32)
        observations.append(values)
        residuals.append(values - (mean[None, :] + coefficients @ basis.T))

    observation = torch.as_tensor(np.concatenate(observations), dtype=torch.float32, device=device)
    residual = torch.as_tensor(np.concatenate(residuals), dtype=torch.float32, device=device)
    observation = observation - observation.mean(dim=0, keepdim=True)
    residual = residual - residual.mean(dim=0, keepdim=True)
    count = residual.shape[0]
    empirical = residual.mT @ residual / max(count - 1, 1)
    residual_std = torch.sqrt(torch.diagonal(empirical).clamp_min(1e-14))
    signal_std = torch.sqrt(torch.sum(observation.square(), dim=0) / max(count - 1, 1)).clamp_min(1e-7)
    target_std = torch.maximum(
        residual_std,
        float(config["observation_noise_fraction"]) * signal_std,
    ).clamp_min(1e-7)
    correlation = empirical / (residual_std[:, None] * residual_std[None, :]).clamp_min(1e-14)
    correlation = 0.5 * (correlation + correlation.mT)
    correlation.fill_diagonal_(1.0)
    shrinkage = float(config["observation_covariance_shrinkage"])
    covariance = (1.0 - shrinkage) * correlation * (target_std[:, None] * target_std[None, :])
    covariance = covariance + shrinkage * torch.diag(target_std.square())
    covariance = 0.5 * (covariance + covariance.mT)
    factor, info = torch.linalg.cholesky_ex(covariance)
    if int(info.max()) != 0:
        raise RuntimeError("Shrinkage observation covariance is not positive definite")

    output_path = layout_root / "observation_covariance_full.npz"
    covariance_np = covariance.detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(
        output_path,
        covariance=covariance_np,
        observation_std=target_std.detach().cpu().numpy().astype(np.float32),
        residual_std=residual_std.detach().cpu().numpy().astype(np.float32),
        signal_std=signal_std.detach().cpu().numpy().astype(np.float32),
        shrinkage=np.asarray(shrinkage),
        sample_count=np.asarray(count),
        training_cases=np.asarray(config["train_cases"]),
    )
    diagonal = torch.diagonal(covariance)
    off_diagonal = correlation[~torch.eye(correlation.shape[0], dtype=torch.bool, device=device)]
    manifest = {
        "layout": args.layout,
        "variant": variant,
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "dimensions": int(covariance.shape[0]),
        "sample_count": int(count),
        "training_cases": list(config["train_cases"]),
        "test_cases_used": [],
        "shrinkage": shrinkage,
        "diagonal_min": float(diagonal.min()),
        "diagonal_max": float(diagonal.max()),
        "correlation_abs_median": float(off_diagonal.abs().median()),
        "correlation_abs_q95": float(torch.quantile(off_diagonal.abs(), 0.95)),
        "cholesky_finite": bool(torch.isfinite(factor).all()),
        "wall_seconds": time.perf_counter() - started,
        "environment": software_environment(),
    }
    write_json(layout_root / "observation_covariance_full_manifest.json", manifest)
    print(layout_root / "observation_covariance_full_manifest.json")


if __name__ == "__main__":
    main()
