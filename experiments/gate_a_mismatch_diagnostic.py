from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_benchmark_v3 as v3


def displacement_nrmse(estimate: np.ndarray, truth: np.ndarray, nx: int) -> float:
    scale = np.sqrt(np.mean(truth[:, :nx] ** 2))
    return float(np.sqrt(np.mean((estimate[:, :nx] - truth[:, :nx]) ** 2)) / max(scale, 1.0e-12))


def free_trajectory(cfg: v3.Config, alpha: float, times: np.ndarray) -> np.ndarray:
    state = v3.initial_state(np.linspace(0.0, cfg.length, cfg.nx))[None, :]
    trajectory = [state[0].copy()]
    theta = v3.alpha_to_theta(alpha, cfg)
    rng = np.random.default_rng(cfg.seed + 991)
    for step in range(times.size - 1):
        state = v3.propagate_batch(
            state, theta, times[step], cfg, rng, stochastic=False
        )
        trajectory.append(state[0].copy())
    return np.stack(trajectory)


def main() -> None:
    values = (0.12, 0.30, 0.50, 0.70, 0.88)
    print("ALPHA_TRUE WRONG_ALPHA WRONG_NRMSE ORACLE_NRMSE ORACLE_GAP")
    for index, alpha_true in enumerate(values):
        cfg = dataclasses.replace(
            v3.make_config("quick"),
            seed=2026080610 + index,
            nx=41,
            t_end=1.0,
            dt=0.0025,
            process_noise=0.0,
            alpha_true=alpha_true,
        )
        scenario = v3.generate_scenario(cfg)
        wrong = free_trajectory(cfg, 0.50, scenario.times)
        oracle = free_trajectory(cfg, alpha_true, scenario.times)
        wrong_error = displacement_nrmse(wrong, scenario.truth_states, cfg.nx)
        oracle_error = displacement_nrmse(oracle, scenario.truth_states, cfg.nx)
        print(
            f"{alpha_true:.2f} 0.50 {wrong_error:.8g} {oracle_error:.8g} "
            f"{wrong_error - oracle_error:.8g}"
        )


if __name__ == "__main__":
    main()
