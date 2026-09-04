from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
import torch

from .common import load_config, write_json
from .io import VIVCase, list_cases
from .rom import DMDCCandidate, PODModel, evaluation_indices, fit_dmdc, project_case, randomized_pod, training_mean


def _parse_cases(value: str | None) -> list[str] | None:
    if value is None:
        return None
    result = [item.strip().replace(",", "").zfill(4)[-4:] for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("At least one case id is required")
    if len(set(result)) != len(result):
        raise ValueError(f"Duplicate case ids: {value}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a training-only POD-DMDc candidate library for VIV-PIV.")
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--block", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026081400)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variant", default=None, help="Output model variant; defaults to rank{rank}_stride{stride}.")
    parser.add_argument("--train-cases", default=None, help="Comma-separated model-fitting cases; defaults to config train cases.")
    parser.add_argument("--projection-cases", default=None, help="Comma-separated cases to project after fitting; defaults to all configured cases.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    rank = int(args.rank or config["rank"])
    data_root = pathlib.Path(config["data_root"])
    variant = args.variant or f"rank{rank}_stride{args.frame_stride}"
    output_root = pathlib.Path(config["output_root"]) / "models" / variant
    manifest_path = output_root / "model_manifest.json"
    if manifest_path.exists() and not args.force:
        print(f"model already exists: {manifest_path}")
        return
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    paths = list_cases(data_root)
    train_ids = _parse_cases(args.train_cases) or list(config["train_cases"])
    projection_ids = _parse_cases(args.projection_cases) or [*config["train_cases"], *config["test_cases"]]
    projection_ids = list(dict.fromkeys([*train_ids, *projection_ids]))
    missing = sorted(set([*train_ids, *projection_ids]) - set(paths))
    if missing:
        raise ValueError(f"Requested cases are not available under {data_root}: {missing}")
    train_cases = [VIVCase.open(paths[case_id]) for case_id in train_ids]
    projection_cases = [VIVCase.open(paths[case_id]) for case_id in projection_ids]
    started = time.perf_counter()
    mean, counts, sample_count = training_mean(train_cases, block=args.block, frame_stride=args.frame_stride)
    basis, singular, explained = randomized_pod(
        train_cases,
        mean,
        rank,
        frame_stride=args.frame_stride,
        block=args.block,
        seed=args.seed,
        device=device,
    )
    reference = train_cases[0]
    evaluation = evaluation_indices(reference, config, int(config["evaluation_points"]))
    pod = PODModel(
        mean=mean,
        basis=basis,
        singular_values=singular,
        explained_fraction=explained,
        reference_x_mm=reference.x_mm,
        reference_y_mm=reference.y_mm,
        evaluation_flat_indices=evaluation,
    )
    pod_path = output_root / "pod_model.npz"
    pod.save(pod_path)
    coefficient_root = output_root / "coefficients"
    coefficient_root.mkdir(parents=True, exist_ok=True)
    candidates: list[DMDCCandidate] = []
    projection_rows: list[dict[str, object]] = []
    for case in projection_cases:
        coefficients = project_case(case, pod, block=args.block, device=device)
        eval_truth = np.empty((case.time_s.size, evaluation.size), dtype=np.float32)
        for start, values, _valid in case.iter_physical(block=args.block):
            eval_truth[start : start + values.shape[0]] = values[:, evaluation]
        payload = {
            "coefficients": coefficients,
            "evaluation_truth": eval_truth,
            "evaluation_flat_indices": evaluation,
            "time_s": case.time_s,
            "cyl_displ_m": case.cyl_displ_m,
            "split": np.asarray("train" if case.case_id in train_ids else "test"),
        }
        coefficient_path = coefficient_root / f"case_{case.case_id}.npz"
        np.savez_compressed(coefficient_path, **payload)
        projection_rows.append({
            "case_id": case.case_id,
            "split": str(payload["split"].item()),
            "coefficient_path": str(coefficient_path),
        })
        if case.case_id in train_ids:
            candidate = fit_dmdc(
                case,
                coefficients,
                float(config["cylinder_diameter_m"]),
                float(config["dmdc_ridge"]),
            )
            candidate.save(output_root / "candidates" / f"candidate_{case.case_id}.npz")
            candidates.append(candidate)
    candidates.sort(key=lambda item: item.reduced_velocity)
    manifest = {
        "variant": variant,
        "rank": rank,
        "frame_stride": args.frame_stride,
        "sample_count": sample_count,
        "train_cases": train_ids,
        "projected_nontraining_cases": [case_id for case_id in projection_ids if case_id not in train_ids],
        "test_data_used_for_model_fit": False,
        "test_data_use": "projection and sealed evaluation summaries only",
        "valid_count_min": int(counts.min()),
        "valid_count_max": int(counts.max()),
        "pod_path": str(pod_path),
        "pod_explained_fraction_at_rank": float(explained[min(rank, explained.size) - 1]),
        "candidate_grid": [candidate.reduced_velocity for candidate in candidates],
        "candidate_cases": [candidate.case_id for candidate in candidates],
        "candidate_spectral_radius": [candidate.spectral_radius for candidate in candidates],
        "candidate_residual_rms": [candidate.residual_rms for candidate in candidates],
        "projection_records": projection_rows,
        "device": str(device),
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(manifest_path, manifest)
    print(f"prepared={variant} candidates={len(candidates)} output={output_root}")


if __name__ == "__main__":
    main()
