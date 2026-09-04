from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
SEED_SOURCE = ROOT / "figures" / "source_data" / "figure4_lorenz96_1024_obs128_t8_seed2026080601"
SWEEP_SOURCE = ROOT / "figures" / "source_data" / "figure4_lorenz96_1024_obs128_time1to8"
OUTPUT_DIR = ROOT / "figures" / "figure4_l96_1024_candidate_modules"

FONT_PANEL = 22
FONT_TITLE = 14
FONT_AXIS = 13
FONT_TICK = 11
FONT_LEGEND = 14

METHODS = ["aug_enkf", "bma_static", "pce", "apce"]
LABELS = {"aug_enkf": "Aug-EnKF", "bma_static": "BMA", "pce": "PCE", "apce": "APCE"}
COLORS = {
    "aug_enkf": "#9B59B6",
    "bma_static": "#F39C12",
    "pce": "#1F77B4",
    "apce": "#2ECC71",
    "truth": "#E84A3A",
    "sensor": "#2AA6B8",
    "ring": "#B9BEC4",
    "text": "#282828",
}


# Figure contract
# Core conclusion: Lorenz-96 is better shown as high-dimensional chaotic
# geometry, sparse ring sensing, and spectral/statistical preservation than as
# a dense Hovmoller plate.
# Archetype: quantitative/image hybrid module set.
# Evidence chain: PCA/delay embedding shows trajectory geometry, ring schematic
# shows observation geometry, interval sweep shows robustness trend, low-mode
# spectrum/correlation shows statistical structure.


def clean_number(value: float, _position: int | None = None) -> str:
    return f"{value:g}"


def configure_matplotlib() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "font.size": FONT_AXIS,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_all(fig: plt.Figure, base: Path) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix, kwargs in {
        ".svg": {},
        ".pdf": {},
        ".png": {"dpi": 650},
        ".tiff": {"dpi": 650},
    }.items():
        path = base.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", pad_inches=0.035, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def load_seed_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(SEED_SOURCE / "shared_asset.npz", allow_pickle=False) as asset:
        truth = np.asarray(asset["truth"], dtype=float)
        sensors = np.asarray(asset["observation_indices"], dtype=int)
    with np.load(SEED_SOURCE / "apce_trace.npz", allow_pickle=False) as trace:
        apce = np.asarray(trace["mean_states"], dtype=float)
    return truth, apce, sensors


def pca_project(truth: np.ndarray, apce: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    combined = np.vstack([truth, apce])
    mean = combined.mean(axis=0, keepdims=True)
    centered = combined - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:3].T
    return (truth - mean) @ basis, (apce - mean) @ basis, basis


def draw_pca_delay_embedding(truth: np.ndarray, apce: np.ndarray) -> list[Path]:
    truth_p, apce_p, _ = pca_project(truth, apce)
    fig = plt.figure(figsize=(4.85, 4.45))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=25, azim=-54)
    ax.set_box_aspect((1.0, 1.0, 0.82))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("#F4F6F7")
        axis.pane.set_edgecolor("#D4D9DC")
        axis.line.set_color("#A8AEB2")
        axis.line.set_linewidth(0.7)
        axis.set_major_formatter(FuncFormatter(clean_number))
    ax.grid(False)
    ax.plot(truth_p[:, 0], truth_p[:, 1], truth_p[:, 2], color=COLORS["truth"], lw=2.0, alpha=0.80, label="Truth")
    ax.plot(apce_p[:, 0], apce_p[:, 1], apce_p[:, 2], color=COLORS["apce"], lw=1.65, alpha=0.96, label="APCE")
    ax.scatter(truth_p[0, 0], truth_p[0, 1], truth_p[0, 2], s=30, color=COLORS["truth"], depthshade=False)
    ax.scatter(apce_p[-1, 0], apce_p[-1, 1], apce_p[-1, 2], s=34, color=COLORS["apce"], marker=">", depthshade=False)
    ax.set_xlabel("PC1", fontsize=FONT_AXIS, labelpad=4)
    ax.set_ylabel("PC2", fontsize=FONT_AXIS, labelpad=4)
    ax.set_zlabel("PC3", fontsize=FONT_AXIS, labelpad=2)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), fontsize=FONT_LEGEND, handlelength=1.9)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.90)
    return save_all(fig, OUTPUT_DIR / "figure4_l96_module_pca_embedding_v1")


def draw_sensor_ring(sensors: np.ndarray, state_dim: int = 1024) -> list[Path]:
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    ax.set_aspect("equal")
    ax.axis("off")
    theta = np.linspace(0, 2 * np.pi, state_dim, endpoint=False)
    ax.add_patch(Circle((0, 0), 1.0, fill=False, lw=2.1, ec=COLORS["ring"], alpha=0.95))
    ax.add_patch(Circle((0, 0), 0.78, fill=False, lw=0.9, ec="#E1E4E6", alpha=0.95))
    sensor_theta = theta[sensors]
    xs, ys = np.cos(sensor_theta), np.sin(sensor_theta)
    ax.scatter(xs, ys, s=18, color=COLORS["sensor"], edgecolor="white", linewidth=0.35, zorder=5)
    for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        ax.plot([0.82 * np.cos(angle), 1.03 * np.cos(angle)], [0.82 * np.sin(angle), 1.03 * np.sin(angle)], color="#C9CDD1", lw=0.8)
    ax.text(0.0, 0.06, "1024 states", ha="center", va="center", fontsize=FONT_TITLE, color=COLORS["text"])
    ax.text(0.0, -0.10, "128 sensors", ha="center", va="center", fontsize=FONT_TITLE, color=COLORS["sensor"])
    ax.text(0.0, -1.23, "uniform ring observations", ha="center", va="top", fontsize=FONT_AXIS, color=COLORS["text"])
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.28, 1.18)
    return save_all(fig, OUTPUT_DIR / "figure4_l96_module_sensor_ring_v1")


def read_method_summary(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def draw_interval_sweep() -> list[Path]:
    rows = read_method_summary(SWEEP_SOURCE / "method_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.9, 3.25), sharex=True)
    for ax, metric, ylabel in zip(axes, ["nrmse_mean", "crps_mean"], ["nRMSE", "CRPS"], strict=True):
        for method in METHODS:
            selected = sorted(
                (row for row in rows if row["method"] == method),
                key=lambda row: int(row["obs_interval"]),
            )
            x = np.asarray([int(row["obs_interval"]) for row in selected], dtype=int)
            y = np.asarray([float(row[metric]) for row in selected], dtype=float)
            ax.plot(
                x,
                y,
                color=COLORS[method],
                lw=2.15 if method in {"pce", "apce"} else 1.55,
                marker="o",
                ms=5.0,
                label=LABELS[method],
                alpha=0.95,
            )
        ax.set_xlabel("Observation interval", fontsize=FONT_AXIS)
        ax.set_ylabel(ylabel, fontsize=FONT_AXIS)
        ax.set_xticks(np.arange(1, 9))
        ax.yaxis.set_major_formatter(FuncFormatter(clean_number))
        ax.tick_params(labelsize=FONT_TICK)
        ax.grid(False)
    axes[0].legend(loc="upper left", fontsize=FONT_LEGEND, handlelength=1.8)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.20, top=0.94, wspace=0.33)
    return save_all(fig, OUTPUT_DIR / "figure4_l96_module_interval_sweep_v1")


def mean_low_mode_spectrum(field: np.ndarray, n_modes: int = 32) -> tuple[np.ndarray, np.ndarray]:
    centered = field - field.mean(axis=1, keepdims=True)
    spec = np.abs(np.fft.rfft(centered, axis=1)) ** 2
    mean_spec = spec.mean(axis=0)
    modes = np.arange(mean_spec.shape[0])
    return modes[1 : n_modes + 1], mean_spec[1 : n_modes + 1] / max(float(mean_spec[1]), 1.0e-12)


def spatial_autocorrelation(field: np.ndarray, max_lag: int = 128) -> tuple[np.ndarray, np.ndarray]:
    centered = field - field.mean(axis=1, keepdims=True)
    denom = np.mean(centered * centered)
    lags = np.arange(max_lag + 1)
    corr = np.asarray([np.mean(centered * np.roll(centered, -lag, axis=1)) / max(denom, 1.0e-12) for lag in lags])
    return lags, corr


def draw_spectrum_correlation(truth: np.ndarray, apce: np.ndarray) -> list[Path]:
    modes_t, spec_t = mean_low_mode_spectrum(truth)
    modes_a, spec_a = mean_low_mode_spectrum(apce)
    lags_t, corr_t = spatial_autocorrelation(truth)
    lags_a, corr_a = spatial_autocorrelation(apce)
    fig, axes = plt.subplots(1, 2, figsize=(7.9, 3.25))
    axes[0].plot(modes_t, spec_t, color=COLORS["truth"], lw=2.25, label="Truth")
    axes[0].plot(modes_a, spec_a, color=COLORS["apce"], lw=1.95, label="APCE")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Fourier mode", fontsize=FONT_AXIS)
    axes[0].set_ylabel("Normalized energy", fontsize=FONT_AXIS)
    axes[0].legend(loc="upper right", fontsize=FONT_LEGEND, handlelength=1.8)
    axes[1].plot(lags_t, corr_t, color=COLORS["truth"], lw=2.25, label="Truth")
    axes[1].plot(lags_a, corr_a, color=COLORS["apce"], lw=1.95, label="APCE")
    axes[1].set_xlabel("Spatial lag", fontsize=FONT_AXIS)
    axes[1].set_ylabel("Correlation", fontsize=FONT_AXIS)
    axes[1].set_ylim(-0.35, 1.04)
    for ax in axes:
        ax.tick_params(labelsize=FONT_TICK)
        ax.xaxis.set_major_formatter(FuncFormatter(clean_number))
        ax.yaxis.set_major_formatter(FuncFormatter(clean_number))
        ax.grid(False)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.20, top=0.94, wspace=0.33)
    return save_all(fig, OUTPUT_DIR / "figure4_l96_module_spectrum_correlation_v1")


def main() -> None:
    configure_matplotlib()
    truth, apce, sensors = load_seed_data()
    outputs = {
        "pca_embedding": [str(path) for path in draw_pca_delay_embedding(truth, apce)],
        "sensor_ring": [str(path) for path in draw_sensor_ring(sensors, truth.shape[1])],
        "interval_sweep": [str(path) for path in draw_interval_sweep()],
        "spectrum_correlation": [str(path) for path in draw_spectrum_correlation(truth, apce)],
    }
    manifest = {
        "case": "Lorenz-96 D=1024",
        "seed": 2026080601,
        "apce_trace": str(SEED_SOURCE / "apce_trace.npz"),
        "shared_asset": str(SEED_SOURCE / "shared_asset.npz"),
        "sweep_source": str(SWEEP_SOURCE / "method_summary.csv"),
        "note": "Interval sweep uses the earlier baseline obs128 time1-8 smoke; PCA and statistical modules use the tuned APCE t8 representative run.",
        "outputs": outputs,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "qa_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
