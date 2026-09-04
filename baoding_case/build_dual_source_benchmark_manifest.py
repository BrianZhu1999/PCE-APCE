#!/usr/bin/env python3
"""Build a provenance manifest for the admitted Baoding dual-source bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--association", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(path for path in args.root.rglob("*") if path.is_file() and (path.name.endswith(".json") or path.name.endswith(".csv")))
    manifest = {
        "claim_status": "baoding_dual_source_real_acoustic_benchmark",
        "root": str(args.root),
        "association_gate": str(args.association),
        "association_gate_sha256": sha256(args.association),
        "source_hashes": {str(path.relative_to(args.root)): sha256(path) for path in files},
        "targets": [1, 2],
        "nodes": [1, 2, 3, 5, 6, 7, 8, 11, 13],
        "frame_count": 956,
        "frame_dt_s": 0.2098360655737705,
        "methods": ["pce", "apce"],
        "seed_count": 5,
        "gps_runtime_observation": False,
        "upstream": "historical dual-source MUSIC/DOA plus GPS-free global association and robust triangulation",
        "dbn_status": "not yet a complete DBN-LA-NM field reproduction; benchmark is admitted as real-acoustic dual-source PCE/APCE baseline",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
