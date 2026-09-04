#!/usr/bin/env python3
"""Summarize the paper-aligned DBN raw smoke as an auditable baseline."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.smoke_json.read_text(encoding="utf-8"))
    rows = payload["rows"]
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for target in (1, 2, 3):
        target_rows = []
        for frame in rows:
            item = next(value for value in frame["targets"] if value["target"] == target)
            target_rows.append({"frame_index": frame["frame_index"], "time_hhmmss": frame["time_hhmmss"], **item})
        with (args.output_root / f"target{target}_dbn_lanm_baseline.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(target_rows[0])); writer.writeheader(); writer.writerows(target_rows)
        errors = [float(row["position_error_m"]) for row in target_rows]
        summary[str(target)] = {"frames": len(errors), "median_error_m": statistics.median(errors), "p90_error_m": sorted(errors)[min(len(errors)-1, int(0.90*len(errors)))], "max_error_m": max(errors), "valid_under_100m_fraction": sum(value < 100.0 for value in errors) / len(errors)}
    manifest = {
        "claim_status": "paper_aligned_dbn_raw_baseline",
        "source_smoke_json": str(args.smoke_json),
        "paper_nodes": payload["paper_nodes"],
        "excluded_node": payload["excluded_node"],
        "segment": payload["segment"],
        "target_mapping": {"target1": "GPS1_plane1.gps", "target2": "GPS3_plane2.gps", "target3": "GPS4_plane2to3.gps"},
        "snapshot_len": payload["snapshot_len"], "sample_rate_hz": payload["sample_rate_hz"], "frequency_count": payload["frequency_count"],
        "adapter_warning": payload["warning"],
        "summary": summary,
    }
    (args.output_root / "paper_dbn_baseline_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
