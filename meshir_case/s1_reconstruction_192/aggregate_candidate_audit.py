#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def nrmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    denominator = max(float(np.sum(truth ** 2)), 1e-20)
    return float(np.sqrt(np.sum((prediction - truth) ** 2) / denominator))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, float | int]] = []
    summaries: list[dict[str, float | int | list[float]]] = []
    for seed_dir in sorted(path for path in args.root.glob("seed_*") if path.is_dir()):
        seed = int(seed_dir.name.split("_")[-1])
        with np.load(seed_dir / "pce.npz") as data:
            required = {"truth", "mean", "candidate_mean", "heldout_interior", "final_weights", "score_history"}
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"{seed_dir}: missing {sorted(missing)}")
            truth = np.asarray(data["truth"], dtype=np.float32)
            mixture = np.asarray(data["mean"], dtype=np.float32)
            candidate_mean = np.asarray(data["candidate_mean"], dtype=np.float32)
            heldout = np.asarray(data["heldout_interior"], dtype=int)
            weights = np.asarray(data["final_weights"], dtype=float)
            scores = np.asarray(data["score_history"], dtype=float)

        analysis_end = min(1024, len(truth))
        truth_region = truth[:analysis_end].reshape(analysis_end, -1)[:, heldout]
        candidate_errors = []
        for candidate in range(len(candidate_mean)):
            prediction = candidate_mean[candidate, :analysis_end].reshape(analysis_end, -1)[:, heldout]
            error = nrmse(truth_region, prediction)
            candidate_errors.append(error)
            rows.append({
                "seed": seed,
                "candidate": candidate,
                "analysis_nrmse": error,
                "final_weight": float(weights[candidate]),
                "cumulative_centered_score": float(np.sum(scores[:, candidate] - scores.mean(axis=1))),
            })

        candidate_errors_np = np.asarray(candidate_errors)
        oracle = int(np.argmin(candidate_errors_np))
        selected = int(np.argmax(weights))
        mixture_error = nrmse(
            truth_region,
            mixture[:analysis_end].reshape(analysis_end, -1)[:, heldout],
        )
        selected_prediction = candidate_mean[selected, :analysis_end].reshape(analysis_end, -1)[:, heldout]
        selected_error = nrmse(truth_region, selected_prediction)
        rank_correlation = float(spearmanr(-candidate_errors_np, weights).statistic)
        summaries.append({
            "seed": seed,
            "oracle_candidate": oracle,
            "oracle_nrmse": float(candidate_errors_np[oracle]),
            "selected_candidate": selected,
            "selected_nrmse": selected_error,
            "mixture_nrmse": mixture_error,
            "selection_regret": selected_error - float(candidate_errors_np[oracle]),
            "mixing_penalty": mixture_error - selected_error,
            "evidence_error_rank_correlation": rank_correlation,
            "candidate_nrmse": candidate_errors,
            "final_weights": weights.tolist(),
        })

    if not summaries:
        raise ValueError(f"no seed directories under {args.root}")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "candidate_level_source_data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "seed_level_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [key for key in summaries[0] if key not in {"candidate_nrmse", "final_weights"}]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fieldnames} for row in summaries])

    oracle_gain = [row["candidate_nrmse"][0] - row["oracle_nrmse"] for row in summaries]
    report = {
        "seeds": len(summaries),
        "mean_oracle_gain_over_linear": float(np.mean(oracle_gain)),
        "mean_selection_regret": float(np.mean([row["selection_regret"] for row in summaries])),
        "mean_mixing_penalty": float(np.mean([row["mixing_penalty"] for row in summaries])),
        "mean_evidence_error_rank_correlation": float(np.mean([row["evidence_error_rank_correlation"] for row in summaries])),
        "seed_results": summaries,
    }
    (args.output / "candidate_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
