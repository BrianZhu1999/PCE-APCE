from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import run_benchmark_v3 as v3
from run_benchmark_v4 import run_ablation


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def run_grid(
    mode: str,
    n_seeds: int,
    base_seed: int,
    alpha_values: list[float],
    noise_values: list[float],
    sensor_values: list[int],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    methods = ["A6_pce", "A7_apce"]

    total = len(alpha_values) * len(noise_values) * len(sensor_values) * n_seeds
    count = 0
    for alpha_true in alpha_values:
        for obs_noise in noise_values:
            for n_sensors in sensor_values:
                for seed_index in range(n_seeds):
                    cfg = replace(
                        v3.make_config(mode),
                        seed=base_seed + seed_index,
                        alpha_true=alpha_true,
                        obs_noise=obs_noise,
                        n_sensors=n_sensors,
                        filter_variant="lr",
                    )
                    scenario = v3.generate_scenario(cfg)
                    for method in methods:
                        result = run_ablation(scenario, method)  # type: ignore[arg-type]
                        records.append(
                            {
                                "alpha_true": alpha_true,
                                "obs_noise": obs_noise,
                                "n_sensors": n_sensors,
                                "seed": cfg.seed,
                                "method": method,
                                "mean_rmse": result["mean_rmse"],
                                "final_rmse": result["final_rmse"],
                                "alpha_best": result["alpha_best"],
                                "alpha_continuous": result["alpha_continuous"],
                                "alpha_abs_error": result["alpha_abs_error"],
                                "alpha_top1_correct": result["alpha_top1_correct"],
                                "alpha_entropy": result["alpha_entropy"],
                                "off_grid": bool(
                                    np.min(np.abs(scenario.alpha_grid - alpha_true)) > 1.0e-12
                                ),
                            }
                        )
                    count += 1
                    print(f"[{count}/{total}] alpha={alpha_true:.3f}, noise={obs_noise:.3f}, sensors={n_sensors}, seed={cfg.seed}")

    fields = sorted({key for record in records for key in record})
    with (output_dir / "generalization_runs.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    summary: list[dict[str, Any]] = []
    for alpha_true in alpha_values:
        for obs_noise in noise_values:
            for n_sensors in sensor_values:
                for method in methods:
                    subset = [
                        record for record in records
                        if record["alpha_true"] == alpha_true
                        and record["obs_noise"] == obs_noise
                        and record["n_sensors"] == n_sensors
                        and record["method"] == method
                    ]
                    summary.append(
                        {
                            "alpha_true": alpha_true,
                            "obs_noise": obs_noise,
                            "n_sensors": n_sensors,
                            "method": method,
                            "off_grid": subset[0]["off_grid"],
                            "mean_rmse": float(np.mean([item["mean_rmse"] for item in subset])),
                            "final_rmse": float(np.mean([item["final_rmse"] for item in subset])),
                            "alpha_mae": float(np.mean([item["alpha_abs_error"] for item in subset])),
                            "top1_accuracy_percent": float(100.0 * np.mean([item["alpha_top1_correct"] for item in subset])),
                            "alpha_entropy": float(np.mean([item["alpha_entropy"] for item in subset])),
                        }
                    )

    payload = {
        "mode": mode,
        "n_seeds_per_condition": n_seeds,
        "base_seed": base_seed,
        "alpha_values": alpha_values,
        "obs_noise_values": noise_values,
        "sensor_values": sensor_values,
        "summary": summary,
    }
    with (output_dir / "generalization_summary.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    make_figures(summary, alpha_values, noise_values, sensor_values, output_dir)
    return payload


def make_figures(
    summary: list[dict[str, Any]],
    alpha_values: list[float],
    noise_values: list[float],
    sensor_values: list[int],
    output_dir: Path,
) -> None:
    for method in ("A6_pce", "A7_apce"):
        selected_noise = min(noise_values, key=lambda value: abs(value - 0.02))
        selected_sensors = min(sensor_values, key=lambda value: abs(value - 6))
        subset = [
            item for item in summary
            if item["method"] == method
            and item["obs_noise"] == selected_noise
            and item["n_sensors"] == selected_sensors
        ]
        subset.sort(key=lambda item: item["alpha_true"])
        figure, axis_left = plt.subplots(figsize=(9, 5.5))
        axis_left.plot(
            [item["alpha_true"] for item in subset],
            [item["alpha_mae"] for item in subset],
            marker="o",
            label="continuous alpha MAE",
        )
        axis_left.set_xlabel("true alpha")
        axis_left.set_ylabel("continuous alpha MAE")
        axis_left.grid(alpha=0.25)
        axis_right = axis_left.twinx()
        axis_right.plot(
            [item["alpha_true"] for item in subset],
            [item["mean_rmse"] for item in subset],
            marker="s",
            linestyle="--",
            label="state RMSE",
        )
        axis_right.set_ylabel("time-mean state RMSE")
        axis_left.set_title(f"{method}: on-grid and off-grid generalization")
        handles_left, labels_left = axis_left.get_legend_handles_labels()
        handles_right, labels_right = axis_right.get_legend_handles_labels()
        axis_left.legend(handles_left + handles_right, labels_left + labels_right)
        figure.tight_layout()
        figure.savefig(output_dir / f"{method}_offgrid_alpha.png", dpi=180)
        plt.close(figure)

    method = "A7_apce"
    alpha_reference = min(alpha_values, key=lambda value: abs(value - 0.78))
    matrix = np.zeros((len(noise_values), len(sensor_values)), dtype=float)
    for i, noise in enumerate(noise_values):
        for j, sensors in enumerate(sensor_values):
            item = next(
                row for row in summary
                if row["method"] == method
                and row["alpha_true"] == alpha_reference
                and row["obs_noise"] == noise
                and row["n_sensors"] == sensors
            )
            matrix[i, j] = item["mean_rmse"]
    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    image = axis.imshow(matrix, origin="lower", aspect="auto")
    axis.set_xticks(range(len(sensor_values)), [str(value) for value in sensor_values])
    axis.set_yticks(range(len(noise_values)), [f"{value:.3f}" for value in noise_values])
    axis.set_xlabel("number of sensors")
    axis.set_ylabel("observation noise standard deviation")
    axis.set_title("APCE state RMSE sensitivity map")
    figure.colorbar(image, ax=axis, label="time-mean RMSE")
    figure.tight_layout()
    figure.savefig(output_dir / "apce_noise_sensor_map.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="V4 off-grid alpha and observation generalization")
    parser.add_argument("--mode", choices=["quick", "balanced", "large"], default="quick")
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=20260803)
    parser.add_argument("--alpha-values", default="0.18,0.35,0.50,0.70,0.78,0.86")
    parser.add_argument("--obs-noise-values", default="0.01,0.02,0.05")
    parser.add_argument("--sensor-values", default="3,6,9")
    parser.add_argument("--output", default="results_generalization_v4")
    args = parser.parse_args()
    if args.n_seeds < 3:
        raise ValueError("Use at least three seeds per condition.")
    payload = run_grid(
        args.mode,
        args.n_seeds,
        args.base_seed,
        parse_float_list(args.alpha_values),
        parse_float_list(args.obs_noise_values),
        parse_int_list(args.sensor_values),
        Path(args.output),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
