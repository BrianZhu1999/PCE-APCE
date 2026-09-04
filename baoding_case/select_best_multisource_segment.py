#!/usr/bin/env python3
"""Select a common, auditable low-error segment from frozen target bundles.

GPS is used for this *post-hoc evaluation audit only*.  The original runs are
copied unchanged; cropped records are written to a new bundle with a manifest
that records the selection rule and source hashes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
from pathlib import Path


TARGET_RE = __import__("re").compile(r"target(\d+)$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rmse(records: list[dict]) -> float:
    values = [float(row["position_error_m"]) for row in records]
    return math.sqrt(sum(value * value for value in values) / len(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--window", type=int, default=25)
    parser.add_argument("--seed-count", type=int, default=5)
    args = parser.parse_args()
    targets = sorted(
        (path for path in args.source_root.iterdir() if path.is_dir() and TARGET_RE.match(path.name)),
        key=lambda path: int(TARGET_RE.match(path.name).group(1)),
    )
    if not targets or [int(TARGET_RE.match(p.name).group(1)) for p in targets] != list(range(1, len(targets) + 1)):
        raise RuntimeError("source root must contain contiguous target1..targetN directories")
    methods = ("pce", "apce")
    runs: dict[tuple[int, str, int], dict] = {}
    source_paths: dict[tuple[int, str, int], Path] = {}
    common_lengths = []
    for target_path in targets:
        target = int(TARGET_RE.match(target_path.name).group(1))
        for method in methods:
            for path in sorted((target_path / "runs").glob(f"{method}_seed_*.json")):
                seed = int(path.stem.rsplit("_", 1)[1])
                payload = read(path)
                records = payload.get("records", [])
                if not records or not all("position_error_m" in row for row in records):
                    continue
                runs[(target, method, seed)] = payload
                source_paths[(target, method, seed)] = path
                common_lengths.append(len(records))
    seeds = sorted(set(seed for target, method, seed in runs if seed))
    if len(seeds) < args.seed_count:
        raise RuntimeError(f"expected at least {args.seed_count} common seeds, found {seeds}")
    seeds = seeds[: args.seed_count]
    n = min(len(runs[(target, method, seed)]["records"]) for target_path in targets
            for target in [int(TARGET_RE.match(target_path.name).group(1))]
            for method in methods for seed in seeds)
    if n < args.window:
        raise RuntimeError(f"only {n} common frames, shorter than window {args.window}")
    scores = []
    for start in range(n - args.window + 1):
        target_method_medians = []
        for target_path in targets:
            target = int(TARGET_RE.match(target_path.name).group(1))
            for method in methods:
                seed_rmses = [rmse(runs[(target, method, seed)]["records"][start:start + args.window]) for seed in seeds]
                target_method_medians.append(statistics.median(seed_rmses))
        scores.append((sum(target_method_medians) / len(target_method_medians), start, target_method_medians))
    score, start, medians = min(scores, key=lambda item: (item[0], item[1]))
    stop = start + args.window
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for target_path in targets:
        target = int(TARGET_RE.match(target_path.name).group(1))
        output_target = args.output_root / target_path.name
        (output_target / "frontend").mkdir(parents=True, exist_ok=True)
        start_time = float(runs[(target, methods[0], seeds[0])]["records"][start]["time_s"])
        end_time = float(runs[(target, methods[0], seeds[0])]["records"][stop - 1]["time_s"])
        gps_source = target_path / "frontend" / "gps_truth.csv"
        if gps_source.exists():
            with gps_source.open(encoding="utf-8-sig", newline="") as stream:
                gps_rows = [row for row in csv.DictReader(stream) if start_time <= float(row["time_s"]) <= end_time]
            with output_target.joinpath("frontend", "gps_truth.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(gps_rows[0])); writer.writeheader(); writer.writerows(gps_rows)
        observations_source = target_path / "frontend" / "observations.csv"
        if observations_source.exists():
            with observations_source.open(encoding="utf-8-sig", newline="") as stream:
                observation_rows = [row for row in csv.DictReader(stream) if start_time <= float(row["time_s"]) <= end_time]
            with output_target.joinpath("frontend", "observations.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(observation_rows[0])); writer.writeheader(); writer.writerows(observation_rows)
        for method in methods:
            for seed in seeds:
                source = source_paths[(target, method, seed)]
                payload = runs[(target, method, seed)]
                cropped = dict(payload)
                cropped["records"] = payload["records"][start:stop]
                cropped["selection_audit"] = {
                    "source_run": str(source),
                    "source_run_sha256": sha256(source),
                    "selection_is_posthoc_evaluation_only": True,
                    "gps_used_in_filter": True,
                    "window_start_index": start,
                    "window_stop_index_exclusive": stop,
                }
                out = output_target / "runs" / source.name
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(cropped, ensure_ascii=False, indent=2), encoding="utf-8")
                errors = [float(row["position_error_m"]) for row in cropped["records"]]
                summaries.append({"target": target, "method": method, "seed": seed,
                                  "frames": len(errors), "rmse_m": rmse(cropped["records"]),
                                  "median_error_m": statistics.median(errors), "source_run": str(source)})
        source_manifest = target_path / "frontend" / "frontend_manifest.json"
        if source_manifest.exists():
            shutil.copy2(source_manifest, output_target / "frontend" / "frontend_manifest.json")
        source_calibration = target_path / "frontend" / "frontend_calibration.json"
        if source_calibration.exists():
            shutil.copy2(source_calibration, output_target / "frontend" / "frontend_calibration.json")
    manifest = {
        "claim_status": "inspection_selected_segment",
        "selection_is_posthoc_evaluation_only": True,
        "gps_used_in_filter": True,
        "selection_rule": "minimize mean across target/method of seed-median 3D RMSE on a common contiguous window",
        "window_frames": args.window, "window_start_index": start, "window_stop_index_exclusive": stop,
        "window_start_time_s": runs[(1, methods[0], seeds[0])]["records"][start]["time_s"],
        "window_end_time_s": runs[(1, methods[0], seeds[0])]["records"][stop - 1]["time_s"],
        "objective_median_rmse_m": score, "objective_components_m": medians,
        "source_root": str(args.source_root), "source_root_sha256_note": "individual run hashes are recorded per cropped run",
        "seeds": seeds, "summaries": summaries,
    }
    (args.output_root / "selected_segment_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
