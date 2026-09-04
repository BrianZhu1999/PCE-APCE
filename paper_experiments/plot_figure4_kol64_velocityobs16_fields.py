from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Nature figure backend contract: editable SVG text and consistent sans-serif typography.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 8.0
plt.rcParams["axes.linewidth"] = 0.7
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["xtick.major.width"] = 0.6
plt.rcParams["ytick.major.width"] = 0.6


METHODS = ("Reference", "Aug-EnKF", "BMA", "PCE", "APCE")
TRACE_FILES = {
    "Aug-EnKF": "aug_enkf.npz",
    "BMA": "bma_static.npz",
    "PCE": "pce.npz",
    "APCE": "apce.npz",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_traces(trace_dir: Path) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    loaded: dict[str, np.lib.npyio.NpzFile] = {}
    try:
        for label, filename in TRACE_FILES.items():
            path = trace_dir / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            loaded[label] = np.load(path, allow_pickle=False)
        truth = np.asarray(loaded["Aug-EnKF"]["truth"], dtype=np.float64)
        times = np.asarray(loaded["Aug-EnKF"]["times"], dtype=np.float64)
        mean_states: dict[str, np.ndarray] = {
            label: np.asarray(data["mean_states"], dtype=np.float64)
            for label, data in loaded.items()
            if label != "Reference"
        }
        if truth.shape != (59, 8192):
            raise ValueError(f"Expected truth shape (59, 8192), got {truth.shape}")
        for label, state in mean_states.items():
            if state.shape != truth.shape:
                raise ValueError(f"{label} shape {state.shape} does not match truth {truth.shape}")
        for label in ("BMA", "PCE", "APCE"):
            if not np.array_equal(truth, np.asarray(loaded[label]["truth"], dtype=np.float64)):
                raise ValueError("Truth arrays differ across methods; paired comparison is invalid")
        return truth, mean_states, times
    finally:
        for data in loaded.values():
            data.close()


def state_to_field(states: np.ndarray) -> np.ndarray:
    return np.asarray(states, dtype=np.float64).reshape(states.shape[0], 2, 64, 64)


def vorticity(states: np.ndarray) -> np.ndarray:
    fields = state_to_field(states)
    nx = ny = 64
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=2.0 * np.pi / nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=2.0 * np.pi / ny)
    u_hat = np.fft.fft2(fields[:, 0], axes=(-2, -1))
    v_hat = np.fft.fft2(fields[:, 1], axes=(-2, -1))
    dv_dx = np.fft.ifft2(1j * kx[None, :, None] * v_hat, axes=(-2, -1)).real
    du_dy = np.fft.ifft2(1j * ky[None, None, :] * u_hat, axes=(-2, -1)).real
    return dv_dx - du_dy


def configure_axis(ax: plt.Axes, show_x: bool, show_y: bool) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(0.0, 2.0 * np.pi)
    ax.set_ylim(0.0, 2.0 * np.pi)
    ax.set_xticks([0.0, np.pi, 2.0 * np.pi])
    ax.set_xticklabels(["0", r"$\pi$", r"$2\pi$"] if show_x else [])
    ax.set_yticks([0.0, np.pi, 2.0 * np.pi])
    ax.set_yticklabels(["0", r"$\pi$", r"$2\pi$"] if show_y else [])
    ax.tick_params(length=2.5, pad=2, labelsize=7)
    if not show_x:
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    if not show_y:
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)


def save_figure(fig: plt.Figure, base: Path) -> list[str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return [str(base.with_suffix(ext)) for ext in (".svg", ".pdf", ".png", ".tiff")]


def plot_velocity(fields: dict[str, np.ndarray], time_index: int, time_value: float, output: Path) -> list[str]:
    arrays = [fields[label][time_index] for label in METHODS]
    limit = max(float(np.max(np.abs(array))) for array in arrays)
    limit = max(limit, 1e-12)
    fig, axes = plt.subplots(2, 5, figsize=(10.4, 4.35), squeeze=False)
    extent = (0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi)
    for col, label in enumerate(METHODS):
        for row, component in enumerate((0, 1)):
            ax = axes[row, col]
            image = ax.imshow(
                fields[label][time_index, component].T,
                origin="lower",
                extent=extent,
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                interpolation="nearest",
            )
            configure_axis(ax, show_x=row == 1, show_y=col == 0)
            if row == 0:
                ax.set_title(label, fontsize=8.5, pad=5)
            if col == 0:
                ax.set_ylabel(r"$u_x$" if component == 0 else r"$u_y$", fontsize=8.5, labelpad=3)
    cbar = fig.colorbar(image, ax=axes, fraction=0.018, pad=0.018, aspect=28)
    cbar.ax.tick_params(labelsize=7, length=2)
    cbar.set_label("velocity", fontsize=8, labelpad=4)
    fig.text(0.5, 0.005, f"x, t = {time_value:g}", ha="center", va="bottom", fontsize=8)
    fig.subplots_adjust(left=0.055, right=0.93, bottom=0.085, top=0.88, wspace=0.08, hspace=0.10)
    return save_figure(fig, output)


def plot_vorticity(omega: dict[str, np.ndarray], time_index: int, time_value: float, output: Path) -> list[str]:
    arrays = [omega[label][time_index] for label in METHODS]
    limit = max(float(np.max(np.abs(array))) for array in arrays)
    limit = max(limit, 1e-12)
    fig, axes = plt.subplots(1, 5, figsize=(10.4, 2.35), squeeze=False)
    extent = (0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi)
    for col, label in enumerate(METHODS):
        ax = axes[0, col]
        image = ax.imshow(
            omega[label][time_index].T,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        configure_axis(ax, show_x=True, show_y=col == 0)
        ax.set_title(label, fontsize=8.5, pad=5)
        if col == 0:
            ax.set_ylabel(r"$\omega$", fontsize=8.5, labelpad=3)
    cbar = fig.colorbar(image, ax=axes, fraction=0.018, pad=0.018, aspect=28)
    cbar.ax.tick_params(labelsize=7, length=2)
    cbar.set_label("vorticity", fontsize=8, labelpad=4)
    fig.text(0.5, 0.005, f"x, t = {time_value:g}", ha="center", va="bottom", fontsize=8)
    fig.subplots_adjust(left=0.055, right=0.93, bottom=0.12, top=0.80, wspace=0.08)
    return save_figure(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot KOL-64 16x16 velocity and derived-vorticity comparisons")
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026081603)
    parser.add_argument("--time-index", type=int, default=58)
    args = parser.parse_args()
    truth, mean_states, times = load_traces(args.trace_dir)
    if not 0 <= args.time_index < truth.shape[0]:
        raise ValueError(f"time-index must be in [0, {truth.shape[0] - 1}]")
    fields: dict[str, np.ndarray] = {"Reference": state_to_field(truth)}
    fields.update({label: state_to_field(states) for label, states in mean_states.items()})
    omega = {label: vorticity(state) for label, state in [("Reference", truth), *mean_states.items()]}
    output_dir = args.output_dir
    velocity_paths = plot_velocity(fields, args.time_index, float(times[args.time_index]), output_dir / f"figure4_kol64_velocityobs16_velocity_seed{args.seed}_t{args.time_index}")
    vorticity_paths = plot_vorticity(omega, args.time_index, float(times[args.time_index]), output_dir / f"figure4_kol64_velocityobs16_vorticity_seed{args.seed}_t{args.time_index}")
    manifest = {
        "case": "kolmogorov64_velocityobs",
        "sensor_grid": "16x16",
        "seed": args.seed,
        "time_index": args.time_index,
        "time": float(times[args.time_index]),
        "state_observation": "direct ux, uy only; vorticity is derived diagnostically",
        "velocity_color_scale": "shared symmetric scale across Reference, Aug-EnKF, BMA, PCE, APCE and ux/uy",
        "vorticity_color_scale": "shared symmetric scale across Reference, Aug-EnKF, BMA, PCE, APCE",
        "trace_files": {label: str((args.trace_dir / filename).resolve()) for label, filename in TRACE_FILES.items()},
        "trace_sha256": {label: sha256(args.trace_dir / filename) for label, filename in TRACE_FILES.items()},
        "outputs": {"velocity": velocity_paths, "vorticity": vorticity_paths},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"figure4_kol64_velocityobs16_seed{args.seed}_t{args.time_index}_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
