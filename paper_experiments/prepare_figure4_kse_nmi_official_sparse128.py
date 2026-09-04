from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_INPUT = Path("<HILDA_RESULTS_ROOT>/external/S3GM_NMI_2024/KSE_test.npy")
DEFAULT_OUTPUT = Path(
    "<HILDA_RESULTS_ROOT>/results/figure4_kse_nmi_official_sparse128_20260814/source_data"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def spectral_roughness(field: np.ndarray) -> float:
    """Rank a trajectory by high-wavenumber energy for visual selection only."""
    # field: time x space
    demeaned = field - field.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(demeaned, axis=1)
    power = np.abs(spectrum) ** 2
    if power.shape[1] <= 5:
        return 0.0
    k = np.arange(power.shape[1], dtype=float)
    total = float(power[:, 1:].sum())
    if total <= 0.0:
        return 0.0
    return float((power[:, 1:] * k[1:][None, :]).sum() / total)


def periodic_spectral_upsample(samples: np.ndarray, target_n: int) -> np.ndarray:
    """Upsample uniformly spaced periodic samples by zero-padding Fourier modes."""
    if samples.ndim != 2:
        raise ValueError(f"Expected samples as time x observed_space, got {samples.shape}.")
    time_n, observed_n = samples.shape
    if target_n % observed_n != 0:
        raise ValueError("target_n must be an integer multiple of observed_n for this check.")
    observed_spec = np.fft.rfft(samples, axis=1)
    target_spec = np.zeros((time_n, target_n // 2 + 1), dtype=np.complex128)
    copy_n = min(observed_spec.shape[1], target_spec.shape[1])
    target_spec[:, :copy_n] = observed_spec[:, :copy_n]
    recon = np.fft.irfft(target_spec, n=target_n, axis=1) * (target_n / observed_n)
    return recon.astype(np.float64, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare official NMI/S3GM KSE test source-data for a 1024-point, "
            "128-sensor Figure 4 admission/visualization draft."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--observed-points", type=int, default=128)
    parser.add_argument("--n-frames", type=int, default=100)
    parser.add_argument("--dt-saved", type=float, default=0.5)
    parser.add_argument("--select", choices=("roughest", "first"), default="roughest")
    args = parser.parse_args()

    started = time.perf_counter()
    input_path = args.input
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = np.load(input_path)
    if raw.ndim != 4 or raw.shape[-1] != 1:
        raise ValueError(f"Expected official KSE_test.npy shape (n, t, x, 1), got {raw.shape}.")
    data = np.asarray(raw[..., 0], dtype=np.float64)
    n_samples, n_time, n_space = data.shape
    if n_space != 1024:
        raise ValueError(f"Expected 1024 spatial points, got {n_space}.")
    if n_time < args.n_frames:
        raise ValueError(f"Requested {args.n_frames} frames, but file only has {n_time}.")
    if n_space % args.observed_points != 0:
        raise ValueError("1024 space points must be divisible by observed-points.")

    frames = np.arange(args.n_frames, dtype=np.int64)
    sensor_stride = n_space // args.observed_points
    sensor_indices = np.arange(0, n_space, sensor_stride, dtype=np.int64)
    if sensor_indices.size != args.observed_points:
        raise RuntimeError(f"Expected {args.observed_points} sensors, got {sensor_indices.size}.")

    visible = data[:, frames, :]
    roughness = np.asarray([spectral_roughness(visible[i]) for i in range(n_samples)], dtype=float)
    if args.select == "first":
        selected = 0
    else:
        selected = int(np.argmax(roughness))

    truth = visible[selected]
    observed = truth[:, sensor_indices]
    recon = periodic_spectral_upsample(observed, n_space)
    abs_error = np.abs(recon - truth)

    # Paper text states x in [0, pi/8], saved test data uses 0.5 time spacing after
    # temporal downsampling. We store coordinates explicitly for traceability.
    x = np.linspace(0.0, math.pi / 8.0, n_space, endpoint=False, dtype=np.float64)
    x_obs = x[sensor_indices]
    t = frames.astype(np.float64) * float(args.dt_saved)

    npz_path = output_dir / "kse_nmi_official_sparse128_source_data.npz"
    np.savez_compressed(
        npz_path,
        all_truth=visible,
        selected_truth=truth,
        selected_observations=observed,
        selected_reconstruction_spectral_interp=recon,
        selected_abs_error=abs_error,
        x=x,
        x_observed=x_obs,
        t=t,
        frame_indices=frames,
        sensor_indices=sensor_indices,
        sample_roughness=roughness,
        selected_sample_index=np.asarray(selected, dtype=np.int64),
    )

    summary_rows = []
    for i in range(n_samples):
        local_truth = visible[i]
        local_obs = local_truth[:, sensor_indices]
        local_recon = periodic_spectral_upsample(local_obs, n_space)
        numerator = float(np.square(local_recon - local_truth).sum())
        denominator = float(np.square(local_truth).sum())
        summary_rows.append(
            {
                "sample_index": i,
                "roughness": float(roughness[i]),
                "spectral_interp_nrmse": math.sqrt(numerator / max(denominator, 1.0e-30)),
                "truth_min": float(local_truth.min()),
                "truth_max": float(local_truth.max()),
                "truth_std": float(local_truth.std()),
            }
        )
    csv_path = output_dir / "kse_nmi_official_sparse128_sample_summary.csv"
    with csv_path.open("w", encoding="utf-8") as handle:
        keys = list(summary_rows[0])
        handle.write(",".join(keys) + "\n")
        for row in summary_rows:
            handle.write(",".join(str(row[k]) for k in keys) + "\n")

    manifest = {
        "created_at_unix": time.time(),
        "runtime_seconds": time.perf_counter() - started,
        "input_path": str(input_path),
        "input_sha256": file_sha256(input_path),
        "source_npz": str(npz_path),
        "source_npz_sha256": file_sha256(npz_path),
        "summary_csv": str(csv_path),
        "summary_csv_sha256": file_sha256(csv_path),
        "source": {
            "paper": "Li et al., Learning spatiotemporal dynamics with a pretrained generative model, Nature Machine Intelligence, 2024",
            "zenodo_record": "https://zenodo.org/records/14607274",
            "file": "KSE_test.npy",
            "protocol_from_supplement": {
                "system": "Kuramoto-Sivashinsky dynamics",
                "space_points": 1024,
                "saved_dt": 0.5,
                "periodic_boundary": True,
                "etdrk4_spectral_solver_reported": True,
                "mu_values_reported_for_dataset": "[1.0, 5.0], 21 values in training; test follows same generation with unseen initial conditions",
            },
        },
        "figure4_admission_geometry": {
            "full_spatial_points": int(n_space),
            "observed_spatial_points": int(args.observed_points),
            "spatial_downsampling_factor": int(sensor_stride),
            "frames_used": int(args.n_frames),
            "time_extent_saved_units": [float(t[0]), float(t[-1])],
            "x_extent_paper_units": [0.0, float(math.pi / 8.0)],
        },
        "selection": {
            "policy": args.select,
            "selected_sample_index": int(selected),
            "note": "Selection is for visualization only; all 9 official test sequences are archived in the source-data.",
        },
        "caveats": [
            "KSE_test.npy does not include an explicit per-sample mu metadata array in the Zenodo record.",
            "The spectral interpolation reconstruction is a data/visual sanity check, not a PCE/APCE performance result.",
        ],
    }
    manifest_path = output_dir / "manifest_kse_nmi_official_sparse128.json"
    manifest_path.write_text(json.dumps(json_ready(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(json_ready(manifest), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
