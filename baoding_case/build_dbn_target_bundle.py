#!/usr/bin/env python3
"""Build an auditable three-target bundle from a DBN field run.

This deliberately exports only a reliability-gated contiguous segment. It
does not manufacture PCE/APCE outputs; those are added only after a compatible
8-node observation adapter is validated.
"""
from __future__ import annotations

import argparse, csv, json, math
from pathlib import Path


TRUTH = {1: "GPS1_plane1.gps", 2: "GPS3_plane2.gps", 3: "GPS4_plane2to3.gps"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dbn-json", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--max-error-m", type=float, default=50.0)
    args = ap.parse_args()
    payload = json.loads(args.dbn_json.read_text(encoding="utf-8"))
    rows = payload["rows"]
    good = []
    for row in rows:
        errors = {int(item["target"]): float(item["position_error_m"]) for item in row["targets"]}
        if all(math.isfinite(errors[t]) and errors[t] <= args.max_error_m for t in (1, 2, 3)):
            good.append(row)
        elif good:
            break
    if len(good) < 5:
        raise RuntimeError("no sufficiently long common reliable segment")
    args.output_root.mkdir(parents=True, exist_ok=True)
    for target in (1, 2, 3):
        folder = args.output_root / f"target{target}"
        (folder / "frontend").mkdir(parents=True, exist_ok=True)
        (folder / "runs").mkdir(parents=True, exist_ok=True)
        with (folder / "frontend" / "dbn_track.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["frame_index", "time_hhmmss", "px", "py", "pz", "position_error_m"])
            writer.writeheader()
            for row in good:
                item = next(x for x in row["targets"] if int(x["target"]) == target)
                writer.writerow({"frame_index": row["frame_index"], "time_hhmmss": row["time_hhmmss"], "px": item["estimated_x"], "py": item["estimated_y"], "pz": "", "position_error_m": item["position_error_m"]})
    manifest = {
        "claim_status": "paper_aligned_dbn_reliable_segment_bundle",
        "source_dbn_json": str(args.dbn_json),
        "source_segment": payload.get("segment"),
        "paper_nodes": payload.get("paper_nodes"),
        "frequency_grid": payload.get("frequency_grid"),
        "propagation_model": payload.get("propagation_model"),
        "target_heights_m": payload.get("target_heights_m"),
        "common_reliable_gate": {"position_error_max_m": args.max_error_m, "rule": "first contiguous segment where all T1/T2/T3 satisfy the gate"},
        "frame_start": good[0]["frame_index"], "frame_end": good[-1]["frame_index"], "frame_count": len(good),
        "target_mapping": {f"target{k}": TRUTH[k] for k in (1, 2, 3)},
        "pce_apce_status": "not_connected; DBN bundle is the validated upstream track, not a provisional three-peak result",
    }
    (args.output_root / "dbn_target_bundle_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
