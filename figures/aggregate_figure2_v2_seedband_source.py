from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METHODS = ("letkf", "aug_enkf", "bma_static", "pce", "apce")
CASES = ("wave", "spring", "heat")


def _quantile_band(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return stack.mean(axis=0), np.quantile(stack, 0.025, axis=0), np.quantile(stack, 0.975, axis=0)


def _load_case_files(case_dir: Path, case: str) -> list[Path]:
    files = sorted(case_dir.glob(f"{case}_seed_*_seedband_curves.npz"))
    if not files:
        raise FileNotFoundError(f"No seed-band files found for case={case} in {case_dir}")
    return files


def aggregate_case(case_dir: Path, case: str, output: Path) -> dict[str, object]:
    files = _load_case_files(case_dir, case)
    records = [np.load(path, allow_pickle=True) for path in files]
    seeds = [int(np.asarray(rec["seed"]).item()) for rec in records]
    out: dict[str, object] = {"case": case, "seeds": seeds, "seed_array": np.asarray(seeds, dtype=np.int64), "n_seeds": len(seeds)}
    if case == "heat":
        out["space"] = np.asarray(records[0]["space"], dtype=float)
    else:
        out["times"] = np.asarray(records[0]["times"], dtype=float)
    for key in ("truth", *METHODS):
        stack = np.stack([np.asarray(rec[key], dtype=float) for rec in records], axis=0)
        mean, low, high = _quantile_band(stack)
        out[f"{key}_mean"] = mean
        out[f"{key}_low"] = low
        out[f"{key}_high"] = high
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / f"{case}_seedband_summary.npz", **{k: v for k, v in out.items() if isinstance(v, np.ndarray)})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Figure 2 seed-band curve files.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summary: dict[str, object] = {"methods": METHODS, "cases": CASES, "case_summaries": []}
    for case in CASES:
        case_summary = aggregate_case(args.input_dir, case, args.output_dir)
        summary["case_summaries"].append(
            {
                "case": case,
                "n_seeds": case_summary["n_seeds"],
                "seeds": case_summary["seeds"],
            }
        )
    (args.output_dir / "figure2_v2_seedband_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
