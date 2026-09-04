from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from run_figure4_kolmogorov64_velocityobs import (
    ALPHA_TRUE_RE575,
    KOL64VelocityConfig,
    KolmogorovVelocitySystem,
    WINDOW_STARTS,
    load_truth,
    sensor_indices,
)


def state_from_velocity(velocity: np.ndarray, device: torch.device) -> torch.Tensor:
    field = torch.as_tensor(velocity, dtype=torch.float64, device=device).permute(0, 3, 1, 2)
    return torch.cat((field[:, 0].reshape(field.shape[0], -1), field[:, 1].reshape(field.shape[0], -1)), dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the corrected KOL velocity-observation protocol")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    cfg = KOL64VelocityConfig(seed=2026081600, sensor_grid=16, window_start=10, data_path=str(args.data_path))
    system = KolmogorovVelocitySystem(cfg, device)
    truth = state_from_velocity(load_truth(cfg), device)
    projected = system.project(truth)
    projection_nrmse = float(torch.sqrt(torch.mean((projected - truth).square())) / torch.sqrt(torch.mean(truth.square())))
    one_step: list[float] = []
    twenty_step: list[float] = []
    for seed, start in WINDOW_STARTS.items():
        local_cfg = KOL64VelocityConfig(seed=seed, sensor_grid=16, window_start=start, data_path=str(args.data_path))
        local_system = KolmogorovVelocitySystem(local_cfg, device)
        local_truth = state_from_velocity(load_truth(local_cfg), device)
        alpha = torch.tensor(ALPHA_TRUE_RE575, dtype=torch.float64, device=device)
        pred = local_system.step(local_truth[0], alpha, None)
        one_step.append(float(torch.sqrt(torch.mean((pred - local_truth[1]).square())) / torch.sqrt(torch.mean(local_truth[1].square()))))
        free = local_truth[0]
        for _ in range(20):
            free = local_system.step(free, alpha, None)
        twenty_step.append(float(torch.sqrt(torch.mean((free - local_truth[20]).square())) / torch.sqrt(torch.mean(local_truth[20].square()))))
    _, s16 = sensor_indices(cfg, device)
    cfg8 = KOL64VelocityConfig(seed=2026081600, sensor_grid=8, window_start=10, data_path=str(args.data_path))
    _, s8 = sensor_indices(cfg8, device)
    subset_ok = bool(set(s8.detach().cpu().tolist()).issubset(set(s16.detach().cpu().tolist())))
    report = {
        "protocol": "kolmogorov64_velocity_observation_v1",
        "axis_order": "source array [time, x, y, (u_x, u_y)]",
        "projection_nrmse": projection_nrmse,
        "oracle_one_step_mean_nrmse": float(np.mean(one_step)),
        "oracle_one_step_by_window": one_step,
        "oracle_20_step_mean_nrmse": float(np.mean(twenty_step)),
        "oracle_20_step_by_window": twenty_step,
        "sensor_8x8_subset_of_16x16": subset_ok,
        "alpha_true_re575": ALPHA_TRUE_RE575,
        "thresholds": {"projection_nrmse_lt": 0.005, "one_step_mean_nrmse_lt": 0.03, "twenty_step_mean_nrmse_lt": 0.15},
    }
    report["pass"] = bool(
        projection_nrmse < 0.005
        and np.mean(one_step) < 0.03
        and np.mean(twenty_step) < 0.15
        and subset_ok
        and all(math.isfinite(v) for v in one_step + twenty_step)
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
