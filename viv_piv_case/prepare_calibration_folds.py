"""Build strict leave-one-training-case-out model folds for calibration."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import subprocess
import sys
from typing import Any

from .common import load_config, write_json


def _parse_list(value: str | None, fallback: list[str]) -> list[str]:
    if value is None:
        return list(fallback)
    result = [item.strip().replace(",", "").zfill(4)[-4:] for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("At least one holdout case is required")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build strict leave-one-training-case-out POD/DMDc and 20x40 sensor-layout folds."
    )
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--layout", default="20x40")
    parser.add_argument("--holdouts", default=None, help="Comma-separated training case ids; defaults to all twelve.")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    training_cases = list(config["train_cases"])
    holdouts = _parse_list(args.holdouts, training_cases)
    unknown = sorted(set(holdouts) - set(training_cases))
    if unknown:
        raise ValueError(f"Calibration holdouts must be configured training cases: {unknown}")
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("At least one GPU id is required")

    root = pathlib.Path(config["output_root"]) / "calibration" / "uncertainty" / "folds"
    root.mkdir(parents=True, exist_ok=True)

    def worker(holdout: str, gpu: str) -> dict[str, Any]:
        active = [case_id for case_id in training_cases if case_id != holdout]
        variant = f"rank{int(config['rank'])}_stride1_calibration_holdout{holdout}"
        model_manifest = pathlib.Path(config["output_root"]) / "models" / variant / "model_manifest.json"
        layout_manifest = pathlib.Path(config["output_root"]) / "models" / variant / "sensor_layouts" / args.layout / "manifest.json"
        if args.skip_existing and model_manifest.exists() and layout_manifest.exists():
            return {"holdout": holdout, "gpu": gpu, "status": "skipped_completed", "variant": variant}
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        common = [sys.executable, "-m"]
        prepare = [
            *common, "viv_piv_case.prepare", "--config", str(args.config), "--variant", variant,
            "--train-cases", ",".join(active), "--projection-cases", ",".join([*active, holdout]),
            "--device", "cpu",
        ]
        layout = [
            *common, "viv_piv_case.prepare_sensor_layout", "--config", str(args.config), "--variant", variant,
            "--layout", args.layout, "--train-cases", ",".join(active), "--case-ids", ",".join([*active, holdout]),
        ]
        if args.force:
            prepare.append("--force")
            layout.append("--force")
        log_path = root / f"holdout_{holdout}.log"
        with log_path.open("w", encoding="utf-8") as log:
            first = subprocess.run(prepare, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
            second = subprocess.run(layout, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False) if first.returncode == 0 else None
        return {
            "holdout": holdout,
            "gpu": gpu,
            "variant": variant,
            "status": "completed" if first.returncode == 0 and second is not None and second.returncode == 0 else "failed",
            "prepare_returncode": first.returncode,
            "layout_returncode": None if second is None else second.returncode,
            "model_manifest": str(model_manifest),
            "layout_manifest": str(layout_manifest),
            "log": str(log_path),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(worker, holdout, gpus[index % len(gpus)]) for index, holdout in enumerate(holdouts)]
        rows = [future.result() for future in futures]
    failed = [row for row in rows if row["status"] == "failed"]
    manifest = {
        "protocol": "strict leave-one-training-case-out model preparation",
        "rank": int(config["rank"]),
        "layout": args.layout,
        "training_pool": training_cases,
        "holdouts": holdouts,
        "gpus": gpus,
        "rows": rows,
        "failed_count": len(failed),
    }
    manifest_path = root / "fold_preparation_manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "failed_count": len(failed)}, ensure_ascii=False))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
