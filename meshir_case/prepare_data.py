#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from meshir.data import causal_downsample, load_ir_matrix, load_npy_zip


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1-archive", type=Path, required=True)
    parser.add_argument("--s32-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    s1_pos = load_npy_zip(args.s1_archive, "S1-M3969_npy", "pos_mic.npy")
    s1_src = load_npy_zip(args.s1_archive, "S1-M3969_npy", "pos_src.npy")
    s1_raw = load_ir_matrix(args.s1_archive, "S1-M3969_npy", 3969, 1)
    s32_pos = load_npy_zip(args.s32_archive, "S32-M441_npy", "pos_mic.npy")
    s32_src = load_npy_zip(args.s32_archive, "S32-M441_npy", "pos_src.npy")
    s32_raw = load_ir_matrix(args.s32_archive, "S32-M441_npy", 441, 32)
    s1 = causal_downsample(s1_raw, 48000, 16000, 127, 7200.0)[0].T
    s32 = causal_downsample(s32_raw, 48000, 16000, 127, 7200.0).transpose(0, 2, 1)
    np.save(args.output / "s1_rir_16k.npy", s1)
    np.save(args.output / "s32_rir_16k.npy", s32)
    np.savez_compressed(args.output / "geometry.npz", s1_positions=s1_pos, s1_source=s1_src, s32_positions=s32_pos, s32_sources=s32_src)
    manifest = {
        "case": "meshir_s1_s32_pilot",
        "native_rate_hz": 48000,
        "processed_rate_hz": 16000,
        "s1_shape": list(s1.shape),
        "s32_shape": list(s32.shape),
        "s1_grid_shape": [21, 21, 9],
        "s1_source_count": int(len(s1_src)),
        "s32_source_count": int(len(s32_src)),
        "s1_archive": {"name": s1_raw.shape, "size": args.s1_archive.stat().st_size, "sha256": sha256(args.s1_archive)},
        "s32_archive": {"name": s32_raw.shape, "size": args.s32_archive.stat().st_size, "sha256": sha256(args.s32_archive)},
        "future_data_used": False,
    }
    (args.output / "data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
