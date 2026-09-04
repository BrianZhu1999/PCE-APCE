"""Audit MUSIC spectral quality for the real three-source Baoding recording.

The previous association used the peak height only.  This audit recomputes a
small neighbourhood of each stored MUSIC peak from the raw WAVFM prefix and
records peak prominence, half-prominence width, local curvature, and a simple
two-band stability score.  GPS is used only after the acoustic quantities are
computed, to label the offline audit; it is never used to choose a peak.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np

from audit_three_source_raw_music import (
    MARKER,
    frequency_bins,
    frequency_decompose,
    music_spectrum,
    music_xy,
    set_channel,
)


def read_wavfm_prefix(path: Path, n_samples: int) -> np.ndarray:
    """Read only the first n_samples framed samples, not the full recording."""
    words_needed = (n_samples + 4) * 22
    raw = np.fromfile(path, dtype="<i2", count=words_needed)
    marker_idx = np.flatnonzero(raw == MARKER)
    if marker_idx.size == 0:
        raise ValueError(f"no frame marker in {path}")
    start = int(marker_idx[0])
    usable = ((raw.size - start) // 22) * 22
    frames = raw[start : start + usable].reshape(-1, 22)
    if np.mean(frames[:, 0] == MARKER) < 0.99:
        raise ValueError(f"frame-marker integrity failed for {path}")
    if frames.shape[0] < n_samples:
        raise ValueError(f"{path} contains only {frames.shape[0]} framed samples")
    return frames[:n_samples, 1:20].T.astype(np.float64)


def circular_distance_deg(a: np.ndarray, b: float) -> np.ndarray:
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def interp_peak_quality(angles: np.ndarray, spectrum: np.ndarray, peak_angle: float, circular: bool) -> dict:
    """Return robust local quality metrics around one peak."""
    values = np.asarray(spectrum, dtype=float)
    finite = np.isfinite(values) & (values > 0)
    if not np.any(finite):
        return {"prominence_db": -np.inf, "width_deg": np.inf, "curvature": 0.0, "local_snr": 0.0}
    if circular:
        dist = circular_distance_deg(angles, peak_angle)
    else:
        dist = np.abs(angles - peak_angle)
    local = dist <= 10.0
    if not np.any(local):
        local = finite
    floor = float(np.median(values[finite]))
    peak_idx = int(np.nanargmax(np.where(local, values, -np.inf)))
    peak = max(float(values[peak_idx]), np.finfo(float).tiny)
    prominence_db = 10.0 * math.log10(peak / max(floor, np.finfo(float).tiny))
    half = floor + 0.5 * (peak - floor)
    above = (values >= half) & local
    width = float(np.sum(above) * (abs(float(angles[1] - angles[0])) if len(angles) > 1 else 1.0))
    # Log-spectrum curvature at the peak is a local sharpness proxy.  It is
    # intentionally clipped; a broad/flat peak should receive larger variance.
    curvature = 0.0
    if 0 < peak_idx < len(values) - 1:
        y0, y1, y2 = np.log(np.maximum(values[peak_idx - 1 : peak_idx + 2], 1e-30))
        curvature = float(max(0.0, -(y2 - 2.0 * y1 + y0)))
    local_snr = max(0.0, 10.0 ** (prominence_db / 10.0) - 1.0)
    return {
        "prominence_db": float(prominence_db),
        "width_deg": float(max(width, 1.0)),
        "curvature": float(curvature),
        "local_snr": float(local_snr),
    }


def band_peak(
    xz: np.ndarray,
    xx: np.ndarray,
    xy: np.ndarray,
    bins: np.ndarray,
    freqs: np.ndarray,
    source_angles: list[tuple[float, float]],
    k: int,
    fsnap: int,
    c: float,
    band_name: str,
) -> list[dict]:
    """Recompute narrow MUSIC spectra and quality for one frequency band."""
    if band_name == "low":
        select = freqs <= np.median(freqs)
    elif band_name == "high":
        select = freqs > np.median(freqs)
    else:
        select = np.ones(len(freqs), dtype=bool)
    band_bins = bins[select]
    band_freqs = freqs[select]
    xz_f = frequency_decompose(xz, band_bins, 512, fsnap)
    xx_f = frequency_decompose(xx, band_bins, 512, fsnap)
    xy_f = frequency_decompose(xy, band_bins, 512, fsnap)
    out = []
    for azimuth, zenith in source_angles:
        az_grid = np.arange(azimuth - 10.0, azimuth + 10.1, 1.0) % 360.0
        el_grid = np.arange(max(0.0, zenith - 10.0), min(90.0, zenith + 10.1), 1.0)
        az_spec = music_xy(k, xx_f, xy_f, band_freqs, c, az_grid)
        el_spec = music_spectrum(k, xz_f, np.asarray([-2.13, -1.53, -0.93, 0.0, 1.0, 2.0, 3.0]) * 0.5, band_freqs, c, el_grid)
        az_q = interp_peak_quality(az_grid, az_spec, azimuth, circular=True)
        el_q = interp_peak_quality(el_grid, el_spec, zenith, circular=False)
        out.append({"azimuth": az_q, "zenith": el_q})
    return out


def audit_node(raw_path: Path, json_path: Path, args: argparse.Namespace) -> dict:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    frames = payload["frames"]
    samples_per_update = args.nfft * args.fsnap
    data = read_wavfm_prefix(raw_path, (args.end_frame + 1) * samples_per_update)
    bins, freqs = frequency_bins(args.fs, args.nfft, args.fl, args.fh)
    rows = []
    for item in frames:
        frame = int(item["frame"])
        if frame < args.start_frame or frame > args.end_frame:
            continue
        lo = frame * samples_per_update
        hi = lo + samples_per_update
        xz, xx, xy = set_channel(data[:, lo:hi])
        sources = list(zip(item["azimuth_deg"], item["elevation_deg"]))
        full = band_peak(xz, xx, xy, bins, freqs, sources, args.k, args.fsnap, args.c, "full")
        low = band_peak(xz, xx, xy, bins, freqs, sources, args.k, args.fsnap, args.c, "low")
        high = band_peak(xz, xx, xy, bins, freqs, sources, args.k, args.fsnap, args.c, "high")
        source_rows = []
        for source in range(args.k):
            f = full[source]
            l = low[source]
            h = high[source]
            az_prom = float(f["azimuth"]["prominence_db"])
            el_prom = float(f["zenith"]["prominence_db"])
            # Angle variance in degrees^2: broad peaks and low prominence are
            # mapped to larger uncertainty, with finite audit bounds.
            az_sigma = np.clip(f["azimuth"]["width_deg"] / max(math.sqrt(2.0 * max(f["azimuth"]["curvature"], 1e-3)), 0.2), 1.0, 45.0)
            el_sigma = np.clip(f["zenith"]["width_deg"] / max(math.sqrt(2.0 * max(f["zenith"]["curvature"], 1e-3)), 0.2), 1.0, 45.0)
            band_delta = abs(l["azimuth"]["prominence_db"] - h["azimuth"]["prominence_db"]) + abs(l["zenith"]["prominence_db"] - h["zenith"]["prominence_db"])
            source_rows.append({
                "source": source,
                "azimuth_deg": float(sources[source][0]),
                "zenith_deg": float(sources[source][1]),
                "full": f,
                "low": l,
                "high": h,
                "band_prominence_delta_db": float(band_delta),
                "azimuth_sigma_deg": float(az_sigma),
                "zenith_sigma_deg": float(el_sigma),
            })
        rows.append({"frame": frame, "sources": source_rows})
    return {"node": json_path.stem.split("_")[0].replace("node", ""), "raw_path": str(raw_path), "rows": rows}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--json-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--nodes", nargs="+", required=True)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--end-frame", type=int, default=81)
    p.add_argument("--pattern", default="node{node}_k3_paper512x4_82.json")
    p.add_argument("--raw-pattern", default="*_{node}_19.wavfm")
    p.add_argument("--fs", type=float, default=3050.0)
    p.add_argument("--nfft", type=int, default=512)
    p.add_argument("--fsnap", type=int, default=4)
    p.add_argument("--fl", type=float, default=100.0)
    p.add_argument("--fh", type=float, default=500.0)
    p.add_argument("--c", type=float, default=340.0)
    p.add_argument("--k", type=int, default=3)
    args = p.parse_args()
    nodes = {}
    for node in args.nodes:
        raw_matches = glob.glob(str(args.raw_root / args.raw_pattern.format(node=node)))
        if len(raw_matches) != 1:
            raise RuntimeError(f"expected one raw file for node {node}, got {raw_matches}")
        nodes[node] = audit_node(Path(raw_matches[0]), args.json_dir / args.pattern.format(node=node), args)
        print(json.dumps({"node": node, "frames": len(nodes[node]["rows"])}, ensure_ascii=True), flush=True)
    output = {
        "protocol": {
            "estimator": "MUSIC local spectral quality audit",
            "raw_gps_free": True,
            "nodes": args.nodes,
            "frames": [args.start_frame, args.end_frame],
            "nfft": args.nfft,
            "fsnap": args.fsnap,
            "frequency_band_hz": [args.fl, args.fh],
            "quality_to_covariance": "bounded width/curvature-derived azimuth and zenith sigma; GPS only for later scoring",
        },
        "nodes": nodes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "nodes": len(nodes)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
