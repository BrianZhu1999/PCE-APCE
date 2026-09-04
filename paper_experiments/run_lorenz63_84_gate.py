from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hilda_da.alpha import liu_quantile
from hilda_da.baselines import denkf_analysis, letkf_analysis
from hilda_da.metrics import weighted_central_interval_coverage_width, weighted_ensemble_crps
from hilda_da.observations import SparseObservation


CaseName = Literal["lorenz63", "lorenz84"]
MethodName = Literal["denkf", "letkf", "oracle_alpha", "pce", "apce"]

METHODS: tuple[MethodName, ...] = ("denkf", "letkf", "oracle_alpha", "pce", "apce")
PLOT_METHODS: tuple[MethodName, ...] = ("denkf", "letkf", "pce", "apce")
LABELS = {
    "denkf": "DEnKF",
    "letkf": "LETKF",
    "oracle_alpha": "Oracle-alpha",
    "pce": "PCE",
    "apce": "APCE",
}


@dataclass(frozen=True)
class CaseConfig:
    name: CaseName
    seed: int
    steps: int
    dt: float
    obs_interval: int
    ensemble_size: int
    obs_noise: float
    alpha_true: float = 0.12
    fixed_alpha: float = 0.50
    alpha_grid: tuple[float, ...] = (0.08, 0.22, 0.36, 0.50, 0.64, 0.78, 0.92)
    pce_temperature: float = 0.40
    apce_temperature: float = 0.52
    apce_min_temperature: float = 0.14
    apce_forgetting: float = 0.985
    apce_entropy_floor: float = 0.72
    evidence_shrinkage: float = 0.20


@dataclass
class Scenario:
    config: CaseConfig
    times: torch.Tensor
    truth: torch.Tensor
    observations: dict[int, torch.Tensor]
    observation_indices: torch.Tensor
    initial_ensemble: torch.Tensor
    forecast_noise: torch.Tensor
    alpha_grid: torch.Tensor


class Lorenz63:
    state_dim = 3

    def __init__(self) -> None:
        self.sigma = 10.0
        self.beta = 8.0 / 3.0
        self.rho_base = 28.0
        self.epistemic_scale = 7.5
        self.stochastic_scale = torch.tensor([0.55, 0.70, 0.65], dtype=torch.float64)

    def drift(self, state: torch.Tensor, alpha_quantile: torch.Tensor) -> torch.Tensor:
        rho = self.rho_base + self.epistemic_scale * alpha_quantile
        x, y, z = state.unbind(dim=-1)
        return torch.stack(
            [
                self.sigma * (y - x),
                x * (rho - z) - y,
                x * y - self.beta * z,
            ],
            dim=-1,
        )

    def diffusion(self, state: torch.Tensor) -> torch.Tensor:
        return self.stochastic_scale.to(dtype=state.dtype, device=state.device).expand_as(state)

    def project(self, state: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(state, nan=0.0, posinf=80.0, neginf=-80.0).clamp(-80.0, 80.0)


class Lorenz84:
    state_dim = 3

    def __init__(self) -> None:
        self.a = 0.25
        self.b = 4.0
        self.f_base = 8.0
        self.g_base = 1.0
        self.f_epistemic_scale = 1.8
        self.g_epistemic_scale = 0.45
        self.stochastic_scale = torch.tensor([0.08, 0.08, 0.08], dtype=torch.float64)

    def drift(self, state: torch.Tensor, alpha_quantile: torch.Tensor) -> torch.Tensor:
        forcing = self.f_base + self.f_epistemic_scale * alpha_quantile
        g_term = self.g_base + self.g_epistemic_scale * alpha_quantile
        x, y, z = state.unbind(dim=-1)
        return torch.stack(
            [
                -(y.square()) - z.square() - self.a * x + self.a * forcing,
                x * y - self.b * x * z - y + g_term,
                self.b * x * y + x * z - z,
            ],
            dim=-1,
        )

    def diffusion(self, state: torch.Tensor) -> torch.Tensor:
        return self.stochastic_scale.to(dtype=state.dtype, device=state.device).expand_as(state)

    def project(self, state: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(state, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)


class Metrics:
    def __init__(self) -> None:
        self.se = 0.0
        self.ts = 0.0
        self.points = 0
        self.crps: list[float] = []
        self.coverage: list[float] = []
        self.width: list[float] = []

    def add(self, ensemble: torch.Tensor, truth: torch.Tensor, weights: torch.Tensor) -> None:
        weights = weights / weights.sum()
        estimate = (weights.unsqueeze(-1) * ensemble).sum(dim=0)
        self.se += float((estimate - truth).square().sum())
        self.ts += float(truth.square().sum())
        self.points += int(truth.numel())
        self.crps.append(float(weighted_ensemble_crps(ensemble, truth, weights)))
        coverage, width = weighted_central_interval_coverage_width(ensemble, truth, weights, level=0.90)
        self.coverage.append(float(coverage))
        self.width.append(float(width))

    def finalize(self) -> dict[str, float]:
        return {
            "nrmse": math.sqrt(self.se / max(self.ts, 1.0e-30)),
            "rmse": math.sqrt(self.se / max(self.points, 1)),
            "crps": float(np.mean(self.crps)),
            "coverage_90": float(np.mean(self.coverage)),
            "interval_width_90": float(np.mean(self.width)),
        }


def config_for_case(name: CaseName, seed: int) -> CaseConfig:
    if name == "lorenz63":
        return CaseConfig(name=name, seed=seed, steps=900, dt=0.005, obs_interval=5, ensemble_size=36, obs_noise=1.15)
    if name == "lorenz84":
        return CaseConfig(
            name=name,
            seed=seed,
            steps=1100,
            dt=0.010,
            obs_interval=5,
            ensemble_size=36,
            obs_noise=0.12,
            pce_temperature=0.36,
            apce_temperature=0.50,
            evidence_shrinkage=0.18,
        )
    raise ValueError(name)


def make_system(config: CaseConfig) -> Lorenz63 | Lorenz84:
    return Lorenz63() if config.name == "lorenz63" else Lorenz84()


def generator(device: torch.device, seed: int) -> torch.Generator:
    gen = torch.Generator(device=device if device.type == "cuda" else "cpu")
    gen.manual_seed(seed)
    return gen


def rk4(system: Lorenz63 | Lorenz84, state: torch.Tensor, dt: float, alpha: float) -> torch.Tensor:
    q = liu_quantile(torch.tensor(alpha, dtype=state.dtype, device=state.device))
    k1 = system.drift(state, q)
    k2 = system.drift(state + 0.5 * dt * k1, q)
    k3 = system.drift(state + 0.5 * dt * k2, q)
    k4 = system.drift(state + dt * k3, q)
    return system.project(state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0)


def step(system: Lorenz63 | Lorenz84, state: torch.Tensor, dt: float, alpha: float, noise: torch.Tensor) -> torch.Tensor:
    deterministic = rk4(system, state, dt, alpha)
    return system.project(deterministic + math.sqrt(dt) * system.diffusion(state) * noise)


def observation_indices(config: CaseConfig, device: torch.device) -> torch.Tensor:
    if config.name == "lorenz63":
        return torch.tensor([0, 2], dtype=torch.int64, device=device)
    return torch.tensor([0, 1], dtype=torch.int64, device=device)


def spinup(system: Lorenz63 | Lorenz84, config: CaseConfig, gen: torch.Generator, device: torch.device) -> torch.Tensor:
    dtype = torch.float64
    if config.name == "lorenz63":
        state = torch.tensor([-8.0, 8.0, 25.0], dtype=dtype, device=device)
        state = state + torch.randn(3, dtype=dtype, device=device, generator=gen)
        spin_steps = 1600
    else:
        state = torch.tensor([1.2, 0.1, 0.2], dtype=dtype, device=device)
        state = state + 0.08 * torch.randn(3, dtype=dtype, device=device, generator=gen)
        spin_steps = 2500
    zero = torch.zeros(3, dtype=dtype, device=device)
    for _ in range(spin_steps):
        state = step(system, state, config.dt, config.alpha_true, zero)
    return state.detach()


def generate_scenario(config: CaseConfig, device: torch.device) -> Scenario:
    dtype = torch.float64
    system = make_system(config)
    gen = generator(device, config.seed)
    state0 = spinup(system, config, gen, device)
    obs_idx = observation_indices(config, device)
    truth_noise = torch.randn((config.steps, 3), dtype=dtype, device=device, generator=gen)
    forecast_noise = torch.randn((config.steps, config.ensemble_size, 3), dtype=dtype, device=device, generator=gen)
    initial_noise = torch.randn((config.ensemble_size, 3), dtype=dtype, device=device, generator=gen)
    obs_noise = torch.randn((config.steps // config.obs_interval + 1, obs_idx.numel()), dtype=dtype, device=device, generator=gen)
    initial_scale = 1.55 if config.name == "lorenz63" else 0.22
    initial_ensemble = system.project(state0.unsqueeze(0) + initial_scale * initial_noise)
    truth = torch.empty((config.steps + 1, 3), dtype=dtype, device=device)
    truth[0] = state0
    for i in range(config.steps):
        truth[i + 1] = step(system, truth[i], config.dt, config.alpha_true, truth_noise[i])
    observations: dict[int, torch.Tensor] = {}
    row = 0
    for i in range(config.obs_interval, config.steps + 1, config.obs_interval):
        observations[i] = truth[i, obs_idx] + config.obs_noise * obs_noise[row]
        row += 1
    return Scenario(
        config=config,
        times=torch.arange(config.steps + 1, dtype=dtype, device=device) * config.dt,
        truth=truth,
        observations=observations,
        observation_indices=obs_idx,
        initial_ensemble=initial_ensemble,
        forecast_noise=forecast_noise,
        alpha_grid=torch.tensor(config.alpha_grid, dtype=dtype, device=device),
    )


def evidence_score(ensemble_observation: torch.Tensor, observation: torch.Tensor, obs_noise: float, shrinkage: float) -> torch.Tensor:
    mean = ensemble_observation.mean(dim=0)
    anomalies = ensemble_observation - mean
    covariance = anomalies.mT @ anomalies / max(ensemble_observation.shape[0] - 1, 1)
    covariance = (1.0 - shrinkage) * covariance + shrinkage * torch.diag(torch.diagonal(covariance))
    covariance = covariance + (obs_noise**2 + 1.0e-8) * torch.eye(observation.numel(), dtype=observation.dtype, device=observation.device)
    residual = observation - mean
    solve = torch.linalg.solve(covariance, residual)
    _, log_det = torch.linalg.slogdet(covariance)
    return -0.5 * (residual @ solve + log_det + observation.numel() * math.log(2.0 * math.pi))


def entropy(weights: torch.Tensor) -> torch.Tensor:
    safe = weights.clamp_min(1.0e-12)
    return -(safe * safe.log()).sum() / math.log(weights.numel())


def run_fixed(scenario: Scenario, method: MethodName, device: torch.device, record: bool = False) -> dict[str, Any]:
    config = scenario.config
    system = make_system(config)
    ensemble = scenario.initial_ensemble.clone()
    obs = SparseObservation(scenario.observation_indices)
    covariance = config.obs_noise**2 * torch.eye(scenario.observation_indices.numel(), dtype=ensemble.dtype, device=device)
    weights = torch.full((config.ensemble_size,), 1.0 / config.ensemble_size, dtype=ensemble.dtype, device=device)
    alpha = config.alpha_true if method == "oracle_alpha" else config.fixed_alpha
    metrics = Metrics()
    trace = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for i in range(config.steps + 1):
        if record:
            trace.append(ensemble.mean(dim=0).detach().cpu())
        metrics.add(ensemble, scenario.truth[i], weights)
        if i == config.steps:
            break
        ensemble = step(system, ensemble, config.dt, alpha, scenario.forecast_noise[i])
        if i + 1 not in scenario.observations:
            continue
        y = scenario.observations[i + 1]
        if method == "denkf":
            ensemble = denkf_analysis(ensemble, y, obs, covariance)
        elif method == "letkf":
            ensemble = letkf_analysis(ensemble, y, obs, covariance)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = metrics.finalize()
    result.update(
        runtime_seconds=float(time.perf_counter() - started),
        forward_member_steps=config.steps * config.ensemble_size,
        peak_gpu_memory_mb=float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
        alpha_estimate=float(alpha),
        alpha_absolute_error=abs(float(alpha) - config.alpha_true),
    )
    if record:
        result["mean_states"] = torch.stack(trace).numpy()
    return result


def run_pce(scenario: Scenario, method: MethodName, device: torch.device, record: bool = False) -> dict[str, Any]:
    config = scenario.config
    system = make_system(config)
    obs = SparseObservation(scenario.observation_indices)
    path_count = scenario.alpha_grid.numel()
    branches = scenario.initial_ensemble.unsqueeze(0).repeat(path_count, 1, 1)
    shadow = branches.clone()
    log_weights = torch.zeros(path_count, dtype=branches.dtype, device=device)
    covariance = config.obs_noise**2 * torch.eye(scenario.observation_indices.numel(), dtype=branches.dtype, device=device)
    metrics = Metrics()
    trace = []
    weight_trace = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for i in range(config.steps + 1):
        weights = torch.softmax(log_weights, dim=0)
        flat = branches.reshape(path_count * config.ensemble_size, 3)
        flat_weights = weights.unsqueeze(1).expand(-1, config.ensemble_size).reshape(-1) / config.ensemble_size
        if record:
            trace.append((flat_weights.unsqueeze(-1) * flat).sum(dim=0).detach().cpu())
            weight_trace.append(weights.detach().cpu())
        metrics.add(flat, scenario.truth[i], flat_weights)
        if i == config.steps:
            break
        for j, alpha in enumerate(scenario.alpha_grid):
            branches[j] = step(system, branches[j], config.dt, float(alpha), scenario.forecast_noise[i])
            shadow[j] = step(system, shadow[j], config.dt, float(alpha), scenario.forecast_noise[i])
        if i + 1 not in scenario.observations:
            continue
        y = scenario.observations[i + 1]
        shadow_obs = torch.stack([obs(branch) for branch in shadow])
        scores = torch.stack([evidence_score(shadow_obs[j], y, config.obs_noise, config.evidence_shrinkage) for j in range(path_count)])
        centered = scores - scores.max()
        temp = config.pce_temperature
        if method == "apce":
            progress = (i + 1) / max(config.steps, 1)
            temp = max(config.apce_min_temperature, config.apce_temperature * (1.0 - 0.62 * progress))
            log_weights = config.apce_forgetting * log_weights + centered / temp
            weights = torch.softmax(log_weights, dim=0)
            if float(entropy(weights)) < config.apce_entropy_floor:
                uniform = torch.full_like(weights, 1.0 / weights.numel())
                weights = 0.78 * weights + 0.22 * uniform
                log_weights = weights.log()
        else:
            log_weights = log_weights + centered / temp
        for j in range(path_count):
            branches[j] = denkf_analysis(branches[j], y, obs, covariance)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    weights = torch.softmax(log_weights, dim=0)
    result = metrics.finalize()
    result.update(
        runtime_seconds=float(time.perf_counter() - started),
        forward_member_steps=2 * config.steps * path_count * config.ensemble_size,
        peak_gpu_memory_mb=float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
        alpha_estimate=float((scenario.alpha_grid * weights).sum()),
        alpha_absolute_error=abs(float((scenario.alpha_grid * weights).sum()) - config.alpha_true),
        alpha_final_entropy=float(entropy(weights)),
    )
    if record:
        result["mean_states"] = torch.stack(trace).numpy()
        result["alpha_weight_history"] = torch.stack(weight_trace).numpy()
    return result


def run_method(scenario: Scenario, method: MethodName, device: torch.device, record: bool = False) -> dict[str, Any]:
    return run_pce(scenario, method, device, record=record) if method in {"pce", "apce"} else run_fixed(scenario, method, device, record=record)


def bootstrap_ci(values: np.ndarray, seed: int, n_boot: int = 10_000) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, values.size, size=(n_boot, values.size))].mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def summarize(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    summary = []
    for case in ("lorenz63", "lorenz84"):
        for method in METHODS:
            subset = [row for row in records if row["case"] == case and row["method"] == method]
            item: dict[str, Any] = {"case": case, "method": method, "label": LABELS[method], "n_seeds": len(subset)}
            for key in ("nrmse", "rmse", "crps", "coverage_90", "interval_width_90", "alpha_absolute_error", "runtime_seconds"):
                values = np.asarray([float(row[key]) for row in subset], dtype=float)
                item[key] = float(values.mean())
                item[f"{key}_ci95"] = bootstrap_ci(values, seed + len(summary))
            summary.append(item)
    return summary


def decisions(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for case in ("lorenz63", "lorenz84"):
        rows = {row["method"]: row for row in summary if row["case"] == case}
        for method in ("pce", "apce"):
            nrmse_win = all(rows[method]["nrmse"] < rows[base]["nrmse"] for base in ("denkf", "letkf"))
            crps_win = all(rows[method]["crps"] < rows[base]["crps"] for base in ("denkf", "letkf"))
            out.append(
                {
                    "case": case,
                    "method": method,
                    "wins_fixed_baselines_on_nrmse": bool(nrmse_win),
                    "wins_fixed_baselines_on_crps": bool(crps_win),
                    "quick_gate_pass": bool(nrmse_win and crps_win),
                    "nrmse_excess_over_oracle": float(rows[method]["nrmse"] - rows["oracle_alpha"]["nrmse"]),
                }
            )
    return out


def export_representative(output: Path, base_seed: int, device: torch.device) -> None:
    source = output / "representative_source"
    source.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for case in ("lorenz63", "lorenz84"):
        config = config_for_case(case, base_seed)
        scenario = generate_scenario(config, device)
        arrays = {
            "times": scenario.times.detach().cpu().numpy(),
            "truth_states": scenario.truth.detach().cpu().numpy(),
            "observation_indices": scenario.observation_indices.detach().cpu().numpy(),
        }
        for method in PLOT_METHODS:
            result = run_method(scenario, method, device, record=True)
            arrays[f"{method}_mean_states"] = np.asarray(result["mean_states"])
        path = source / f"{case}_representative_seed_{base_seed}.npz"
        np.savez_compressed(path, **arrays)
        manifest[case] = {"source": path.name, **asdict(config)}
    (source / "representative_source_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def run_suite(n_seeds: int, base_seed: int, output: Path, device: torch.device) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    records = []
    total = 2 * n_seeds * len(METHODS)
    done = 0
    for case in ("lorenz63", "lorenz84"):
        for seed_index in range(n_seeds):
            config = config_for_case(case, base_seed + seed_index)
            scenario = generate_scenario(config, device)
            for method in METHODS:
                result = run_method(scenario, method, device)
                row = {"case": case, "seed": config.seed, "method": method, "label": LABELS[method], **result}
                records.append(row)
                done += 1
                print(f"[{done}/{total}] case={case} seed={config.seed} method={method} nrmse={row['nrmse']:.4%} crps={row['crps']:.4e}", flush=True)
    with (output / "run_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in records for k in row}))
        writer.writeheader()
        writer.writerows(records)
    summary = summarize(records, base_seed)
    gate = decisions(summary)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in summary for k in row}))
        writer.writeheader()
        writer.writerows(summary)
    payload = {"n_seeds": n_seeds, "base_seed": base_seed, "summary": summary, "decisions": gate}
    (output / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    export_representative(output, base_seed, device)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Lorenz63/Lorenz84 APCE-PCE quick gate.")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=2026080610)
    parser.add_argument("--output", type=Path, default=Path("results_lorenz63_84_gate_5seeds"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    payload = run_suite(args.n_seeds, args.base_seed, args.output, torch.device(args.device))
    print(json.dumps({"decisions": payload["decisions"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
