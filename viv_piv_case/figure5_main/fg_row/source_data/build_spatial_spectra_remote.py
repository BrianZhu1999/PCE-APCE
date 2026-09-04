"""Build radial spatial kinetic-energy spectra on the Super-Server.

The source VIV-PIV archives contain full 2-D velocity fields.  Truth is read
from the public archive and PCE/APCE are reconstructed from the formal5 latent
traces and POD basis on the same 201 x 416 grid.  The output stores an average
radial 2-D FFT energy spectrum in nondimensional spatial wavenumber kD.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


CASES = ["0463", "0556", "0679", "0803", "1359"]


def radial_spectrum(field: np.ndarray, valid: np.ndarray, k_radius: np.ndarray, edges: np.ndarray, window: np.ndarray) -> np.ndarray:
    field = np.asarray(field, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    field = np.where(valid[..., None], field, 0.0)
    field = field - np.mean(field, axis=(1, 2), keepdims=True)
    field = field * window[None, ..., None]
    transformed = np.fft.fftshift(np.fft.fft2(field, axes=(1, 2)), axes=(1, 2))
    power = 0.5 * np.sum(np.abs(transformed) ** 2, axis=3)
    flat_k = k_radius.ravel()
    bin_ids = np.searchsorted(edges, flat_k, side="right") - 1
    keep = (bin_ids >= 0) & (bin_ids < len(edges) - 1)
    spectra = np.zeros((field.shape[0], len(edges) - 1), dtype=np.float64)
    for index in range(len(edges) - 1):
        pixels = keep & (bin_ids == index)
        if np.any(pixels):
            spectra[:, index] = power[:, :, :].reshape(field.shape[0], -1)[:, pixels].mean(axis=1)
    return spectra.mean(axis=0)


def normalise(lam: np.ndarray, energy: np.ndarray) -> np.ndarray:
    integral = float(np.trapezoid(energy, lam))
    return energy / max(integral, 1e-30)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    sys.path.insert(0, str(args.code_root))
    from hybrid_uncertain_wave.viv_piv_case.io import VIVCase

    pod_path = args.result_root / "models" / "rank256_stride1" / "pod_model.npz"
    pod = np.load(pod_path, allow_pickle=False)
    mean = np.asarray(pod["mean"], dtype=np.float32)
    basis = np.asarray(pod["basis"], dtype=np.float32)
    y_count, x_count = 201, 416
    diameter_m = 0.05
    d_x = float(np.median(np.diff(np.asarray(pod["reference_x_mm"], dtype=np.float64)))) / 1000.0
    d_y = float(np.median(np.diff(np.asarray(pod["reference_y_mm"], dtype=np.float64)))) / 1000.0
    kx = np.fft.fftshift(np.fft.fftfreq(x_count, d=d_x)) * diameter_m
    ky = np.fft.fftshift(np.fft.fftfreq(y_count, d=d_y)) * diameter_m
    k_radius = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    max_k = float(np.max(k_radius))
    edges = np.linspace(0.0, max_k, 129)
    lam = 0.5 * (edges[:-1] + edges[1:])
    window = np.outer(np.hanning(y_count), np.hanning(x_count)).astype(np.float32)

    output: dict[str, np.ndarray] = {"wavenumber_lambda": lam.astype(np.float64)}
    provenance: dict[str, object] = {
        "kind": "radial spatial kinetic-energy spectrum",
        "coordinate": "lambda = k * D, with k in cycles per metre and D=0.05 m",
        "grid": {"ny": y_count, "nx": x_count, "dx_m": d_x, "dy_m": d_y},
        "processing": "masked invalid pixels set to zero, per-frame spatial mean removed, 2-D Hann window, radial bin mean, temporal mean, integral normalisation",
        "warmup_frames": int(args.warmup),
        "sources": {},
    }

    for case_id in CASES:
        archive = args.data_root / f"reduced_velocity_{case_id}.npz"
        case = VIVCase.open(archive)
        mask = np.asarray(case.mask[0] > 0.5, dtype=bool)
        trace_dir = args.result_root / "runs" / "rank256_stride1" / "traces"
        traces = {
            "pce": trace_dir / f"viv_{case_id}_pce_seed000_layoutadaptive_fullfield_valid_ens064_covfull_shr050.npz",
            "apce": trace_dir / f"viv_{case_id}_apce_seed000_layoutadaptive_fullfield_valid_ens064_covfull_shr050.npz",
        }
        provenance["sources"][case_id] = {
            "truth_archive": str(archive),
            "pod_model": str(pod_path),
            "traces": {method: str(path) for method, path in traces.items()},
        }
        accum = {"truth": [], "pce": [], "apce": []}
        for start in range(args.warmup, case.time_s.size, args.batch):
            stop = min(start + args.batch, case.time_s.size)
            truth, valid = case.physical_frames(start, stop)
            valid = np.asarray(valid > 0.5, dtype=bool)
            accum["truth"].append(radial_spectrum(truth, valid, k_radius, edges, window))
            for method, path in traces.items():
                with np.load(path, allow_pickle=False) as trace:
                    latent = np.asarray(trace["latent_estimate"][start:stop], dtype=np.float32)
                prediction = (mean[None, :] + latent @ basis.T).reshape(stop - start, y_count, x_count, 2)
                accum[method].append(radial_spectrum(prediction, mask[None, ...].repeat(stop - start, axis=0), k_radius, edges, window))
        for method, chunks in accum.items():
            output[f"{case_id}_{method}"] = normalise(lam, np.mean(np.stack(chunks), axis=0))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    provenance["output"] = str(args.output)
    args.output.with_suffix(".json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
