from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "figures" / "source_data" / "figure4_lorenz96_1024_obs128_t8_seed2026080601"
ASSET_PATH = SOURCE / "shared_asset.npz"
TRACE_PATH = SOURCE / "apce_trace.npz"
OUTPUT_DIR = ROOT / "figures" / "figure4_l96_1024_obs128_t8_apce_preview"

FONT_TITLE = 14
FONT_AXIS = 13
FONT_TICK = 11
FONT_LEGEND = 14

COLORS = {
    "truth": "#E84A3A",
    "apce": "#2ECC71",
    "axis": "#343434",
    "wall": "#F3F4F4",
}


# Figure contract
# Core conclusion: in the D=1024, 128-sensor, time-8 setting, APCE preserves
# the evolving Lorenz-96 field and its low-dimensional trajectory geometry.
# Archetype: image plate + phase-space validation.
# Evidence hierarchy: field reconstruction (primary), pointwise relative error
# (localization), shared-coordinate 3D phase portrait (dynamical validation).
# Reviewer guardrail: this preview is explicitly traceable to one frozen APCE
# run; the accompanying five-seed metrics remain the quantitative evidence.


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
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.frameon": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def load_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(ASSET_PATH, allow_pickle=False) as asset:
        truth = np.asarray(asset["truth"], dtype=float)
        sensor_indices = np.asarray(asset["observation_indices"], dtype=int)
        observation_noise = np.asarray(asset["observation_noise"], dtype=float)
    with np.load(TRACE_PATH, allow_pickle=False) as trace:
        reconstruction = np.asarray(trace["mean_states"], dtype=float)
        times = np.asarray(trace["times"], dtype=float)
    if truth.shape != reconstruction.shape:
        raise ValueError(f"Truth/reconstruction shape mismatch: {truth.shape} versus {reconstruction.shape}")
    return truth, reconstruction, sensor_indices, observation_noise


def field_cmap() -> LinearSegmentedColormap:
    # Blue maps to negative values and red to positive values.
    return LinearSegmentedColormap.from_list(
        "l96_field",
        ["#1F5A89", "#E9F1F4", "#FAE8D1", "#9A1F2A"],
        N=256,
    )


def error_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "l96_relative_error",
        ["#F7FBFF", "#B8DAE6", "#4B99BE", "#123A67"],
        N=256,
    )


def observations_canvas(
    truth: np.ndarray,
    sensors: np.ndarray,
    observation_noise: np.ndarray,
    *,
    observation_interval: int,
    observation_sd: float,
) -> np.ndarray:
    canvas = np.full_like(truth, np.nan, dtype=float)
    for step in range(observation_interval, truth.shape[0], observation_interval):
        canvas[step, sensors] = truth[step, sensors] + observation_sd * observation_noise[step - 1]
    return canvas


def setup_field_axes(ax: plt.Axes, *, with_y: bool) -> None:
    ax.set_xlabel("$t$", fontsize=FONT_AXIS, labelpad=2)
    if with_y:
        ax.set_ylabel("State index", fontsize=FONT_AXIS, labelpad=3)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks([0, 512, 1024])
    ax.xaxis.set_major_formatter(FuncFormatter(clean_number))
    ax.yaxis.set_major_formatter(FuncFormatter(clean_number))
    ax.tick_params(labelsize=FONT_TICK, pad=1)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def save_all(fig: plt.Figure, output_base: Path) -> list[Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for suffix, kwargs in {
        ".svg": {},
        ".pdf": {},
        ".png": {"dpi": 650},
        ".tiff": {"dpi": 650},
    }.items():
        target = output_base.with_suffix(suffix)
        fig.savefig(target, bbox_inches="tight", pad_inches=0.035, **kwargs)
        saved.append(target)
    return saved


def make_spacetime_figure(
    truth: np.ndarray,
    reconstruction: np.ndarray,
    sensors: np.ndarray,
    observation_noise: np.ndarray,
) -> list[Path]:
    times = np.linspace(0.0, 3.0, truth.shape[0])
    observed = observations_canvas(
        truth,
        sensors,
        observation_noise,
        observation_interval=8,
        observation_sd=0.45,
    )
    rms_truth = float(np.sqrt(np.mean(np.square(truth))))
    relative_error = np.abs(reconstruction - truth) / max(rms_truth, 1.0e-12)

    fig = plt.figure(figsize=(10.2, 3.15))
    grid = fig.add_gridspec(
        1,
        4,
        left=0.055,
        right=0.985,
        bottom=0.23,
        top=0.87,
        wspace=0.15,
        width_ratios=[1.0, 1.0, 1.0, 1.0],
    )
    axes = [fig.add_subplot(grid[0, index]) for index in range(4)]
    vlimit = 12.0
    field_norm = Normalize(vmin=-vlimit, vmax=vlimit)
    error_norm = Normalize(vmin=0.0, vmax=0.40)
    cmap = field_cmap()

    observation_image = axes[0].imshow(
        observed.T,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], 0, truth.shape[1]],
        interpolation="nearest",
        cmap=cmap,
        norm=field_norm,
    )
    truth_image = axes[1].imshow(
        truth.T,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], 0, truth.shape[1]],
        interpolation="nearest",
        cmap=cmap,
        norm=field_norm,
    )
    reconstruction_image = axes[2].imshow(
        reconstruction.T,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], 0, truth.shape[1]],
        interpolation="nearest",
        cmap=cmap,
        norm=field_norm,
    )
    error_image = axes[3].imshow(
        relative_error.T,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], 0, truth.shape[1]],
        interpolation="nearest",
        cmap=error_cmap(),
        norm=error_norm,
    )
    for ax, title, show_y in zip(
        axes,
        ["Observations", "Truth", "APCE", "Relative error"],
        [True, False, False, False],
        strict=True,
    ):
        ax.set_title(title, fontsize=FONT_TITLE, pad=7)
        setup_field_axes(ax, with_y=show_y)

    field_bar = fig.colorbar(
        truth_image,
        ax=axes[:3],
        orientation="horizontal",
        fraction=0.070,
        pad=0.22,
        aspect=42,
    )
    field_bar.set_ticks([-12, 0, 12])
    field_bar.ax.xaxis.set_major_formatter(FuncFormatter(clean_number))
    field_bar.ax.tick_params(labelsize=FONT_TICK, length=0, pad=5)
    field_bar.outline.set_visible(False)
    field_bar.set_label("$x_i$", fontsize=FONT_AXIS, labelpad=2)

    error_bar = fig.colorbar(
        error_image,
        ax=axes[3],
        orientation="horizontal",
        fraction=0.070,
        pad=0.22,
        aspect=42,
    )
    error_bar.set_ticks([0, 0.2, 0.4])
    error_bar.ax.xaxis.set_major_formatter(FuncFormatter(clean_number))
    error_bar.ax.tick_params(labelsize=FONT_TICK, length=0, pad=5)
    error_bar.outline.set_visible(False)
    error_bar.set_label("Relative error", fontsize=FONT_AXIS, labelpad=2)

    return save_all(fig, OUTPUT_DIR / "figure4_l96_1024_obs128_t8_apce_spacetime_v1")


def phase_coordinates(truth: np.ndarray, reconstruction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Use a fixed adjacent triplet. It retains the native ring-coupled state
    # geometry rather than adding a learned display projection.
    state_indices = np.asarray([0, 1, 2], dtype=int)
    return truth[:, state_indices], reconstruction[:, state_indices]


def style_phase_axes(ax: plt.Axes) -> None:
    ax.view_init(elev=23, azim=-58)
    ax.set_box_aspect((1.0, 1.0, 0.82))
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(COLORS["wall"])
        axis.pane.set_edgecolor("#D3D7D8")
        axis.line.set_color("#A5AAAC")
        axis.line.set_linewidth(0.7)
    ax.tick_params(labelsize=FONT_TICK, pad=1)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_formatter(FuncFormatter(clean_number))


def make_phase_figure(truth: np.ndarray, reconstruction: np.ndarray) -> list[Path]:
    truth_phase, apce_phase = phase_coordinates(truth, reconstruction)
    fig = plt.figure(figsize=(4.65, 4.35))
    ax = fig.add_subplot(111, projection="3d")
    style_phase_axes(ax)
    ax.plot(
        truth_phase[:, 0],
        truth_phase[:, 1],
        truth_phase[:, 2],
        color=COLORS["truth"],
        linewidth=2.00,
        alpha=0.82,
        label="Truth",
        zorder=2,
    )
    ax.plot(
        apce_phase[:, 0],
        apce_phase[:, 1],
        apce_phase[:, 2],
        color=COLORS["apce"],
        linewidth=1.55,
        alpha=0.95,
        label="APCE",
        zorder=3,
    )
    for values, color, marker in (
        (truth_phase, COLORS["truth"], "o"),
        (apce_phase, COLORS["apce"], "o"),
    ):
        ax.scatter(values[0, 0], values[0, 1], values[0, 2], s=30, color=color, marker=marker, depthshade=False)
        ax.scatter(values[-1, 0], values[-1, 1], values[-1, 2], s=34, color=color, marker=">", depthshade=False)
    ax.set_xlabel("$x_0$", fontsize=FONT_AXIS, labelpad=4)
    ax.set_ylabel("$x_1$", fontsize=FONT_AXIS, labelpad=4)
    ax.set_zlabel("$x_2$", fontsize=FONT_AXIS, labelpad=2)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.00, 1.03),
        fontsize=FONT_LEGEND,
        handlelength=1.9,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.90)
    return save_all(fig, OUTPUT_DIR / "figure4_l96_1024_obs128_t8_apce_phase_v1")


def write_qa_manifest(spacetime_outputs: list[Path], phase_outputs: list[Path]) -> None:
    manifest = {
        "case": "Lorenz-96",
        "state_dim": 1024,
        "observed_points": 128,
        "spatial_downsampling_factor": 8,
        "obs_interval_steps": 8,
        "seed": 2026080601,
        "method": "APCE",
        "tuning_profile": "apce_floor_045+alpha_conservative",
        "selection": "lowest APCE nRMSE among the frozen five-seed smoke, for visual inspection only",
        "spacetime_source": str(ASSET_PATH),
        "trace_source": str(TRACE_PATH),
        "image_adjustments": {
            "truth_and_apce": "shared fixed display range [-12, 12]",
            "relative_error": "fixed range [0, 0.40]",
            "cropping": "none",
        },
        "outputs": {
            "spacetime": [str(path) for path in spacetime_outputs],
            "phase": [str(path) for path in phase_outputs],
        },
    }
    (OUTPUT_DIR / "qa_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    configure_matplotlib()
    truth, reconstruction, sensors, observation_noise = load_inputs()
    spacetime_outputs = make_spacetime_figure(truth, reconstruction, sensors, observation_noise)
    phase_outputs = make_phase_figure(truth, reconstruction)
    write_qa_manifest(spacetime_outputs, phase_outputs)
    print("\n".join(str(path) for path in [*spacetime_outputs, *phase_outputs]))


if __name__ == "__main__":
    main()
