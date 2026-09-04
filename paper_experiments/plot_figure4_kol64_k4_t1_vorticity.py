from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


# Nature figure contract: editable SVG/PDF text and one consistent sans-serif font.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 8.0
plt.rcParams["axes.linewidth"] = 0.7
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


METHOD_FILES = {
    "Truth": "apce.npz",
    "APCE": "apce.npz",
    "PCE": "pce.npz",
    "Aug-EnKF": "aug_enkf.npz",
    "BMA": "bma_static.npz",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vorticity(states: np.ndarray) -> np.ndarray:
    fields = np.asarray(states, dtype=np.float64).reshape(states.shape[0], 2, 64, 64)
    kx = 2.0 * np.pi * np.fft.fftfreq(64, d=2.0 * np.pi / 64.0)
    ky = 2.0 * np.pi * np.fft.fftfreq(64, d=2.0 * np.pi / 64.0)
    u_hat = np.fft.fft2(fields[:, 0], axes=(-2, -1))
    v_hat = np.fft.fft2(fields[:, 1], axes=(-2, -1))
    dv_dx = np.fft.ifft2(1j * kx[None, :, None] * v_hat, axes=(-2, -1)).real
    du_dy = np.fft.ifft2(1j * ky[None, None, :] * u_hat, axes=(-2, -1)).real
    return dv_dx - du_dy


def load_fields(trace_dir: Path) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, str]]:
    fields: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    times: np.ndarray | None = None
    for label, filename in METHOD_FILES.items():
        path = trace_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[label] = sha256(path)
        with np.load(path, allow_pickle=False) as data:
            if label == "Truth":
                states = np.asarray(data["truth"], dtype=np.float64)
                times = np.asarray(data["times"], dtype=np.float64)
            else:
                states = np.asarray(data["mean_states"], dtype=np.float64)
        fields[label] = vorticity(states)
    assert times is not None
    shape = fields["Truth"].shape
    if shape != (59, 64, 64):
        raise ValueError(f"Expected vorticity shape (59, 64, 64), got {shape}")
    for label, array in fields.items():
        if array.shape != shape:
            raise ValueError(f"{label} shape {array.shape} does not match {shape}")
    return fields, times, hashes


def configure_axis(ax: plt.Axes, show_y: bool) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(0.0, 2.0 * np.pi)
    ax.set_ylim(0.0, 2.0 * np.pi)
    ax.set_xticks([0.0, np.pi, 2.0 * np.pi])
    ax.set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
    ax.set_yticks([0.0, np.pi, 2.0 * np.pi])
    ax.set_yticklabels(["0", r"$\pi$", r"$2\pi$"] if show_y else [])
    ax.tick_params(length=2.4, width=0.6, pad=2, labelsize=7)
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


def plot_vorticity(fields: dict[str, np.ndarray], times: np.ndarray, time_index: int, output: Path) -> list[str]:
    labels = tuple(METHOD_FILES)
    frame = np.stack([fields[label][time_index] for label in labels])
    limit = float(np.max(np.abs(frame)))
    limit = max(1.0, float(np.ceil(limit)))
    fig, axes = plt.subplots(1, 5, figsize=(10.8, 2.55), squeeze=False)
    extent = (0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi)
    image = None
    for col, label in enumerate(labels):
        ax = axes[0, col]
        image = ax.imshow(
            fields[label][time_index].T,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            norm=Normalize(vmin=-limit, vmax=limit),
            interpolation="nearest",
        )
        configure_axis(ax, show_y=col == 0)
        ax.set_title(label, fontsize=9.0, pad=5)
        if col == 0:
            ax.set_ylabel(r"$\omega$", fontsize=9.0, labelpad=3)
        ax.set_xlabel("x", fontsize=8.0, labelpad=2)
    assert image is not None
    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.018, pad=0.02, aspect=28)
    cbar.ax.tick_params(labelsize=7, length=2)
    cbar.set_label("Vorticity", fontsize=8.0, labelpad=4)
    fig.subplots_adjust(left=0.055, right=0.93, bottom=0.17, top=0.80, wspace=0.08)
    return save_figure(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser(description="KOL-64 Re=1500 derived-vorticity comparison")
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--time-index", type=int, default=58)
    parser.add_argument("--forcing-wavenumber", type=int, default=4)
    parser.add_argument("--observation-interval", type=int, default=1)
    args = parser.parse_args()
    fields, times, hashes = load_fields(args.trace_dir)
    if not 0 <= args.time_index < len(times):
        raise ValueError(f"time-index must be in [0, {len(times) - 1}]")
    outputs = plot_vorticity(fields, times, args.time_index, args.output)
    frame = np.stack([fields[label][args.time_index] for label in METHOD_FILES])
    limit = max(1.0, float(np.ceil(np.max(np.abs(frame)))))
    manifest = {
        "case": "kolmogorov64_velocityobs",
        "reynolds": 1500,
        "forcing_wavenumber": args.forcing_wavenumber,
        "sensor_grid": "16x16",
        "observation_interval": args.observation_interval,
        "seed": args.seed,
        "time_index": args.time_index,
        "time": float(times[args.time_index]),
        "method_order": list(METHOD_FILES),
        "state_observation": "direct ux, uy only; vorticity is derived diagnostically",
        "shared_symmetric_color_scale": [-limit, limit],
        "trace_sha256": hashes,
        "outputs": outputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_name(args.output.name + "_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
