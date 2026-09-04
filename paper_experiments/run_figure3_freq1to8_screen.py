from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path("<HILDA_RESULTS_ROOT>/code")
OBS_RUNNER = Path(
    "<HILDA_RESULTS_ROOT>/results/"
    "figure3_selected6_observation_insufficient_screen_5seeds_20260813/"
    "run_obs_insufficient.py"
)


def load_obsrun():
    spec = importlib.util.spec_from_file_location("obsrun", OBS_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load observation runner from {OBS_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["obsrun"] = module
    spec.loader.exec_module(module)
    for factor in range(1, 9):
        module.SCENARIOS[f"freq{factor}"] = {
            "obs_interval_factor": factor,
            "obs_noise_factor": 1.0,
        }
    return module


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_scenario(args: argparse.Namespace) -> None:
    obsrun = load_obsrun()
    scenario = args.scenario
    if scenario not in {f"freq{i}" for i in range(1, 9)}:
        raise ValueError(f"Unsupported scenario {scenario}; expected freq1..freq8")

    cases = [item.strip() for item in args.cases.split(",") if item.strip()]
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    output = Path(args.output) / scenario
    profile = obsrun.load_tuning_profile(Path(args.tuning_matrix), args.tuning_profile)
    device = torch.device(args.device)

    (output / "runs").mkdir(parents=True, exist_ok=True)
    (output / "source_data").mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    total = len(cases) * len(methods) * args.n_seeds
    completed = 0
    for case in cases:
        for seed_offset in range(args.n_seeds):
            seed = args.base_seed + seed_offset
            for method in methods:
                completed += 1
                path = output / "runs" / f"fig3obs_{scenario}_{case}_{method}_s{seed}.json"
                row = obsrun.run_json_path(
                    path,
                    case,
                    method,
                    seed,
                    scenario,
                    device,
                    profile,
                    args.record_all_traces,
                )
                records.append(row)
                print(
                    f"[{completed}/{total}] scenario={scenario} case={case} "
                    f"seed={seed} method={method} device={device} "
                    f"status={row.get('numerical_status')} "
                    f"nrmse={float(row.get('nrmse', float('nan'))):.5g} "
                    f"alpha={float(row.get('alpha_absolute_error', float('nan'))):.5g}",
                    flush=True,
                )

    source_dir = output / "source_data"
    write_csv(source_dir / "figure3_freq_sweep_run_source_data.csv", records)

    summary = obsrun.summarize(records, args.base_seed)
    for row in summary:
        row["observation_scenario"] = scenario
        row["obs_interval_factor"] = obsrun.SCENARIOS[scenario]["obs_interval_factor"]
        row["obs_noise_factor"] = obsrun.SCENARIOS[scenario]["obs_noise_factor"]
    write_csv(source_dir / "figure3_freq_sweep_method_summary.csv", summary)

    comparisons = obsrun.paired_comparisons(records, args.base_seed + 10000)
    for row in comparisons:
        row["observation_scenario"] = scenario
        row["obs_interval_factor"] = obsrun.SCENARIOS[scenario]["obs_interval_factor"]
        row["obs_noise_factor"] = obsrun.SCENARIOS[scenario]["obs_noise_factor"]
    write_csv(source_dir / "figure3_freq_sweep_paired_comparisons.csv", comparisons)

    trace_rows = []
    for npz in sorted((output / "runs").glob("*.npz")):
        run_json = npz.with_suffix(".json")
        meta = {}
        if run_json.is_file():
            meta = json.loads(run_json.read_text(encoding="utf-8"))
        trace_rows.append(
            {
                "observation_scenario": scenario,
                "case": meta.get("case", ""),
                "method": meta.get("method", ""),
                "seed": meta.get("seed", ""),
                "numerical_status": meta.get("numerical_status", ""),
                "trace_npz_path": str(npz),
                "run_json_path": str(run_json),
            }
        )
    write_csv(source_dir / "figure3_freq_sweep_trace_index.csv", trace_rows)

    files = {}
    for path in [
        PROJECT_ROOT / "paper_experiments/run_figure3_applied_ode.py",
        PROJECT_ROOT / "hilda_da/systems/applied_odes.py",
        PROJECT_ROOT / "hilda_da/alpha_refinement.py",
        OBS_RUNNER,
        Path(__file__),
        Path(args.tuning_matrix),
    ]:
        files[str(path)] = sha256_file(path) if path.is_file() else "MISSING"

    manifest = {
        "protocol": "figure3-selected5-freq1to8-screen-5seed",
        "scenario": scenario,
        "scenario_config": obsrun.SCENARIOS[scenario],
        "cases": cases,
        "methods": methods,
        "n_seeds": args.n_seeds,
        "base_seed": args.base_seed,
        "tuning_profile": args.tuning_profile,
        "record_all_traces": bool(args.record_all_traces),
        "records": len(records),
        "valid_records": sum(1 for row in records if row.get("numerical_status") == "valid"),
        "trace_npz_count": len(trace_rows),
        "source_hash": hashlib.sha256(
            json.dumps(files, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
        "source_files": files,
    }
    (output / "figure3_freq_sweep_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument(
        "--cases",
        default="pk_infusion,chemical,pendulum,fhn,robertson",
    )
    parser.add_argument("--methods", default="aug_enkf,bma_static,pce,apce")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=2026081200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        default="<HILDA_RESULTS_ROOT>/results/"
        "figure3_selected5_freq1to8_screen_5seeds_20260813",
    )
    parser.add_argument(
        "--tuning-matrix",
        default="<HILDA_RESULTS_ROOT>/results/"
        "figure3_selected6_targeted_auglocal_tuning_5seeds_20260813/"
        "tuning_matrix_targeted_20260813.json",
    )
    parser.add_argument("--tuning-profile", default="v52_lagcases_aug_global100")
    parser.add_argument("--record-all-traces", action="store_true")
    run_scenario(parser.parse_args())


if __name__ == "__main__":
    main()
