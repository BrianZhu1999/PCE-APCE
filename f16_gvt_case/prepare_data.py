#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from f16_gvt.data import audit_archive, load_fullmsine_level
from f16_gvt.preprocess import causal_preprocess


HERE = Path(__file__).resolve().parent


def frf_diagnostic(payload: dict[str, np.ndarray], rate: float, band: tuple[float, float]) -> dict:
    force = payload["force"]
    acceleration = payload["acceleration"]
    spectra_u = np.fft.rfft(force, axis=1)
    spectra_y = np.fft.rfft(acceleration, axis=1)
    frequency = np.fft.rfftfreq(force.shape[1], 1.0 / rate)
    denominator = np.mean(np.abs(spectra_u) ** 2, axis=0)
    h1 = np.mean(spectra_y * np.conj(spectra_u)[:, :, None], axis=0) / np.maximum(denominator[:, None], 1e-12)
    mask = (frequency >= band[0]) & (frequency <= band[1])
    peaks = []
    for channel in range(3):
        local = np.flatnonzero(mask)[np.argmax(np.abs(h1[mask, channel]))]
        peaks.append({
            "channel": channel + 1,
            "peak_frequency_hz": float(frequency[local]),
            "peak_magnitude": float(abs(h1[local, channel])),
        })
    return {"frequency": frequency, "h1": h1, "peaks": peaks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=HERE / "cache")
    args = parser.parse_args()
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    archive_audit = audit_archive(args.archive)
    archive_audit["dataset_doi"] = config["dataset_doi"]
    archive_audit["source_archive_read_only"] = True
    if archive_audit["missing_required_members"]:
        raise RuntimeError(f"missing FullMSine data: {archive_audit['missing_required_members']}")
    (args.output_root / "archive_audit.json").write_text(json.dumps(archive_audit, indent=2), encoding="utf-8")
    audits = []
    payloads = {}
    for level in range(1, 8):
        raw = load_fullmsine_level(args.archive, level)
        payload, audit = causal_preprocess(raw, config)
        np.savez_compressed(args.output_root / f"fullmsine_level{level}_processed.npz", **payload)
        (args.output_root / f"fullmsine_level{level}_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        payloads[level] = payload
        audits.append(audit)
    diagnostic = frf_diagnostic(payloads[1], float(config["processed_rate_hz"]), tuple(config["filter_band_hz"]))
    np.savez_compressed(args.output_root / "level1_frf_diagnostic.npz", frequency=diagnostic.pop("frequency"), h1=diagnostic.pop("h1"))
    (args.output_root / "level1_frf_diagnostic.json").write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
    manifest = {
        "levels": list(range(1, 8)),
        "background_levels": config["levels"]["background"],
        "estimation_levels": config["levels"]["estimation"],
        "validation_levels": config["levels"]["validation"],
        "processed_rate_hz": config["processed_rate_hz"],
        "filter_band_hz": config["filter_band_hz"],
        "all_levels_finite": all(row["all_values_finite"] for row in audits),
        "validation_used_for_selection": False,
    }
    (args.output_root / "preprocess_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
