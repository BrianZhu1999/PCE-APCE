from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch


SCRIPT = Path(
    "<HILDA_RESULTS_ROOT>/results/"
    "figure3_selected6_observation_insufficient_screen_5seeds_20260813/"
    "run_obs_insufficient.py"
)


def load_obsrun():
    spec = importlib.util.spec_from_file_location("obsrun", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load observation runner from {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["obsrun"] = module
    spec.loader.exec_module(module)
    module.SCENARIOS["full"] = {"obs_interval_factor": 1, "obs_noise_factor": 1.0}
    return module


def write_csv(path: Path, rows):
    rows = list(rows)
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


def main() -> None:
    obsrun = load_obsrun()

    out = Path(
        "<HILDA_RESULTS_ROOT>/results/"
        "figure3_selected5_observation_trend_full_5seeds_20260813"
    )
    cases = ["pk_infusion", "chemical", "pendulum", "fhn", "robertson"]
    methods = ["aug_enkf", "bma_static", "pce", "apce"]
    base_seed = 2026081200
    n_seeds = 5
    profile_name = "v52_lagcases_aug_global100"
    tuning_matrix = Path(
        "<HILDA_RESULTS_ROOT>/results/"
        "figure3_selected6_targeted_auglocal_tuning_5seeds_20260813/"
        "tuning_matrix_targeted_20260813.json"
    )
    profile = obsrun.load_tuning_profile(tuning_matrix, profile_name)

    (out / "runs").mkdir(parents=True, exist_ok=True)
    (out / "source_data").mkdir(parents=True, exist_ok=True)

    devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    if not devices:
        devices = [torch.device("cpu")]
    print("devices=" + ",".join(str(device) for device in devices), flush=True)

    records = []
    total = len(cases) * len(methods) * n_seeds
    idx = 0
    for case_index, case in enumerate(cases):
        for seed_offset in range(n_seeds):
            seed = base_seed + seed_offset
            for method_index, method in enumerate(methods):
                idx += 1
                device = devices[(case_index + seed_offset + method_index) % len(devices)]
                path = out / "runs" / f"fig3obs_full_{case}_{method}_s{seed}.json"
                row = obsrun.run_json_path(
                    path, case, method, seed, "full", device, profile, False
                )
                records.append(row)
                status = row.get("numerical_status")
                nrmse = float(row.get("nrmse", float("nan")))
                alpha = float(row.get("alpha_absolute_error", float("nan")))
                print(
                    f"[{idx}/{total}] full case={case} seed={seed} "
                    f"method={method} device={device} status={status} "
                    f"nrmse={nrmse:.5g} alpha={alpha:.5g}",
                    flush=True,
                )

    write_csv(out / "source_data" / "figure3_obs_full5_run_source_data.csv", records)

    summary = obsrun.summarize(records, base_seed)
    for row in summary:
        row["observation_scenario"] = "full"
    write_csv(out / "source_data" / "figure3_obs_full5_method_summary.csv", summary)

    files = {}
    for path in [
        Path("<HILDA_RESULTS_ROOT>/code/paper_experiments/run_figure3_applied_ode.py"),
        Path("<HILDA_RESULTS_ROOT>/code/hilda_da/systems/applied_odes.py"),
        SCRIPT,
        tuning_matrix,
    ]:
        files[str(path)] = sha256_file(path)

    manifest = {
        "protocol": "figure3-selected5-observation-trend-full-5seed",
        "scenario": "full",
        "scenario_config": obsrun.SCENARIOS["full"],
        "cases": cases,
        "methods": methods,
        "n_seeds": n_seeds,
        "base_seed": base_seed,
        "tuning_profile": profile_name,
        "records": len(records),
        "valid_records": sum(
            1 for row in records if row.get("numerical_status") == "valid"
        ),
        "source_hash": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()
        ).hexdigest(),
        "source_files": files,
    }
    (out / "figure3_obs_full5_config_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
