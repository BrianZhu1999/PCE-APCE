from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FormatStrFormatter

from figure4_v62_style_helpers import KOL_BACKGROUND_SOFT


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
FIGURES_DIR = PROJECT_ROOT / "CLEAN_MANUSCRIPT" / "figures"

RECON_TRACE = (
    HERE
    / "source_data"
    / "figure4_kolmogorov64_re1500_k2_s16_t4_seed2026081612"
    / "apce_reconstruction.npz"
)
FORECAST_TRACE = (
    PROJECT_ROOT
    / "audit"
    / "figure4_kolmogorov64_velocityobs_re1500_k2_s16_t4_blackout40_formal_50seeds_20260816_2gpu"
    / "traces"
    / "kol64_re1500_k2_s16_t4_blackout40_apce_seed2026081637.npz"
)
STRESS_SWEEP_ROOT = (
    HERE
    / "source_data"
    / "figure4_kolmogorov64_re1500_k468_s16_temporal_sweep_smoke_20260816"
)
STRESS_REP_TRACE = STRESS_SWEEP_ROOT / "k8_t8_representative_trace" / "seed_2026081604.npz"

OUT_RECON = HERE / "supp_figure4_kol_velocity_reconstruction_atlas_v2"
OUT_FORECAST = HERE / "supp_figure4_kol_velocity_blackout_atlas_v2"
OUT_STRESS = HERE / "supp_figure4_kol_highk_boundary_audit_v2"


METHOD_ORDER = ("Reference", "APCE")
METHOD_COLORS = {
    "aug_enkf": "#5A9BD5",
    "bma_static": "#9A9A9A",
    "pce": "#F08A4B",
    "apce": "#D84A3A",
}
METHOD_DISPLAY = {
    "aug_enkf": "Aug-EnKF",
    "bma_static": "BMA",
    "pce": "PCE",
    "apce": "APCE",
}
BOX_ORDER = ("aug_enkf", "bma_static", "pce", "apce")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_all(fig: plt.Figure, base: Path, *, dpi: int = 650) -> dict[str, str]:
    outputs = {
        "png": base.with_suffix(".png"),
        "pdf": base.with_suffix(".pdf"),
        "svg": base.with_suffix(".svg"),
        "tiff": base.with_suffix(".tiff"),
    }
    fig.savefig(outputs["png"], dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(outputs["pdf"], transparent=True, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(outputs["svg"], transparent=True, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(outputs["tiff"], dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0.02)
    return {name: str(path) for name, path in outputs.items()}


def make_field_cmap() -> LinearSegmentedColormap:
    # Red denotes positive field values and blue denotes negative values,
    # matching the final Figure 4 semantic convention.
    return LinearSegmentedColormap.from_list(
        "kol_field",
        ["#214c78", "#f4f2ee", "#9d3c35"],
        N=256,
    )


def load_reconstruction_trace(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as trace:
        truth = np.asarray(trace["truth"], dtype=np.float64)
        mean_states = np.asarray(trace["mean_states"], dtype=np.float64)
        observations = np.asarray(trace["observations"], dtype=np.float64)
        sensor_indices = np.asarray(trace["sensor_indices"], dtype=np.int64)
        times = np.asarray(trace["times"], dtype=np.float64)
    if truth.shape != mean_states.shape:
        raise ValueError(f"Reconstruction truth/mean shape mismatch: {truth.shape} vs {mean_states.shape}")
    return truth, mean_states, observations, sensor_indices, times


def load_forecast_trace(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    with np.load(path, allow_pickle=True) as trace:
        truth = np.asarray(trace["truth"], dtype=np.float64)
        mean_states = np.asarray(trace["mean_states"], dtype=np.float64)
        observations = np.asarray(trace["assimilated_observations"], dtype=np.float64)
        sensor_indices = np.asarray(trace["sensor_indices"], dtype=np.int64)
        times = np.asarray(trace["times"], dtype=np.float64)
        blackout_start_step = int(np.asarray(trace["blackout_start_step"]).item())
    if truth.shape != mean_states.shape:
        raise ValueError(f"Forecast truth/mean shape mismatch: {truth.shape} vs {mean_states.shape}")
    return truth, mean_states, observations, sensor_indices, times, blackout_start_step


def load_stress_trace(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as trace:
        truth = np.asarray(trace["truth"], dtype=np.float64)
        mean_states = np.asarray(trace["mean_states"], dtype=np.float64)
        observations = np.asarray(trace["observations"], dtype=np.float64)
        sensor_indices = np.asarray(trace["sensor_indices"], dtype=np.int64)
        times = np.asarray(trace["times"], dtype=np.float64)
    if truth.shape != mean_states.shape:
        raise ValueError(f"Stress truth/mean shape mismatch: {truth.shape} vs {mean_states.shape}")
    return truth, mean_states, observations, sensor_indices, times


def state_to_components(states: np.ndarray) -> np.ndarray:
    return np.asarray(states, dtype=np.float64).reshape(states.shape[0], 2, 64, 64)


def spectral_derivative(field: np.ndarray, axis: int) -> np.ndarray:
    """Periodic derivative on [0, 2π) using integer Fourier modes."""
    n = field.shape[axis]
    modes = np.fft.fftfreq(n, d=1.0 / n)
    shape = [1] * field.ndim
    shape[axis] = n
    multiplier = (1j * modes).reshape(shape)
    return np.fft.ifft(multiplier * np.fft.fft(field, axis=axis), axis=axis).real


def vorticity_from_components(components: np.ndarray) -> np.ndarray:
    """Return ω = ∂u_y/∂x − ∂u_x/∂y for components shaped (..., 2, 64, 64)."""
    ux = components[..., 0, :, :]
    uy = components[..., 1, :, :]
    return spectral_derivative(uy, axis=-2) - spectral_derivative(ux, axis=-1)


def component_limits(truth: np.ndarray, mean: np.ndarray, steps: list[int], component: int) -> float:
    comp_truth = state_to_components(truth)[steps, component]
    comp_mean = state_to_components(mean)[steps, component]
    limit = float(np.max(np.abs(np.concatenate([comp_truth.ravel(), comp_mean.ravel()]))))
    return max(limit, 1e-12)


def plot_velocity_atlas(
    truth: np.ndarray,
    mean: np.ndarray,
    times: np.ndarray,
    *,
    steps: list[int],
    title: str,
    output_base: Path,
    source_trace: Path,
    source_sha256: str,
    seed: int,
    config_text: str,
) -> dict[str, str]:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(9.2, 3.13), facecolor="white")
    gs = fig.add_gridspec(
        4,
        11,
        left=0.095,
        right=0.975,
        bottom=0.065,
        top=0.905,
        wspace=0.02,
        hspace=0.08,
        width_ratios=[1.0] * 10 + [0.11],
    )
    cmap = make_field_cmap()
    limits = {
        0: component_limits(truth, mean, steps, 0),
        1: component_limits(truth, mean, steps, 1),
    }
    shared_limit = max(limits.values())
    rows = (
        (truth, 0, r"Ref. $u_x$"),
        (mean, 0, r"APCE $u_x$"),
        (truth, 1, r"Ref. $u_y$"),
        (mean, 1, r"APCE $u_y$"),
    )
    images: dict[int, matplotlib.image.AxesImage] = {}
    for row_index, (source, component, row_label) in enumerate(rows):
        for c, step in enumerate(steps):
            ax = fig.add_subplot(gs[row_index, c])
            field = state_to_components(source)[step, component]
            image = ax.imshow(
                field.T,
                origin="lower",
                extent=(0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi),
                cmap=cmap,
                vmin=-shared_limit,
                vmax=shared_limit,
                interpolation="nearest",
                rasterized=True,
            )
            images[component] = image
            ax.set_facecolor(KOL_BACKGROUND_SOFT)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_index == 0:
                ax.set_title(rf"$t={times[step]:.1f}$", fontsize=11.5, pad=3)

    row_y = [0.805, 0.592, 0.374, 0.160]
    for y, (_, _, row_label) in zip(row_y, rows, strict=True):
        fig.text(0.080, y, row_label, ha="right", va="center", fontsize=11.2)

    if set(images) != {0, 1}:
        raise RuntimeError("Velocity atlas did not render any images")

    cax = fig.add_subplot(gs[:, 10])
    cbar = fig.colorbar(images[0], cax=cax, orientation="vertical")
    cbar.ax.tick_params(labelsize=10.5, length=0, pad=2)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    cbar.outline.set_visible(False)

    outputs = save_all(fig, output_base)
    plt.close(fig)
    qa = {
        "backend": "Python/matplotlib",
        "figure": title,
        "panel_letters": [],
        "source_trace": str(source_trace),
        "source_trace_sha256": source_sha256,
        "seed": seed,
        "steps": steps,
        "times": [float(times[s]) for s in steps],
        "components": ["u_x", "u_y"],
        "figure_size_inches": [9.2, 3.13],
        "row_labels": [item[2] for item in rows],
        "title": title,
        "config_text": config_text,
        "limits": {"u_x": limits[0], "u_y": limits[1], "shared": shared_limit},
        "outputs": outputs,
    }
    output_base.with_suffix(".qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    return outputs


def load_method_summary_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_stress_sweep(root: Path) -> dict[tuple[int, int, str], dict[str, float]]:
    records: dict[tuple[int, int, str], dict[str, float]] = {}
    for k in (4, 6, 8):
        for interval in (1, 2, 4, 6, 8):
            summary = root / f"k{k}_t{interval}" / "kolmogorov64_velocityobs_method_summary.csv"
            if not summary.exists():
                raise FileNotFoundError(summary)
            for row in load_method_summary_rows(summary):
                method = row["method"]
                if method not in BOX_ORDER:
                    continue
                records[(k, interval, method)] = {
                    "nrmse": float(row["nrmse_mean"]),
                    "crps": float(row["crps_mean"]),
                    "n": float(row["n"]),
                }
    return records


def plot_highk_boundary_audit(
    truth: np.ndarray,
    mean: np.ndarray,
    times: np.ndarray,
    sweep_root: Path,
    *,
    output_base: Path,
    source_trace: Path,
    source_sha256: str,
    seed: int,
) -> dict[str, str]:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    sweep = load_stress_sweep(sweep_root)
    fig = plt.figure(figsize=(14.6, 6.2), facecolor="white")
    outer = fig.add_gridspec(
        1,
        2,
        left=0.045,
        right=0.975,
        bottom=0.115,
        top=0.880,
        wspace=0.34,
        width_ratios=[1.25, 1.0],
    )
    cmap = make_field_cmap()
    step = len(times) - 1
    truth_comp = state_to_components(truth[[step]])[0]
    mean_comp = state_to_components(mean[[step]])[0]
    truth_vort = vorticity_from_components(truth_comp[None, ...])[0]
    mean_vort = vorticity_from_components(mean_comp[None, ...])[0]

    velocity_limit = float(np.max(np.abs(np.stack([truth_comp, mean_comp]))))
    vort_display = np.stack([truth_vort, mean_vort])
    vort_limit = float(np.percentile(np.abs(vort_display), 99))
    velocity_limit = max(velocity_limit, 1e-12)
    vort_limit = max(vort_limit, 1e-12)

    field_grid = outer[0, 0].subgridspec(
        2,
        4,
        width_ratios=[1.0, 1.0, 1.0, 0.055],
        wspace=0.075,
        hspace=0.14,
    )
    fields = [
        (truth_comp[0], truth_comp[1], truth_vort),
        (mean_comp[0], mean_comp[1], mean_vort),
    ]
    row_labels = ["Ref.", "APCE"]
    col_labels = [r"$u_x$", r"$u_y$", r"$\omega$"]
    velocity_images = []
    vort_images = []
    for r in range(2):
        for c in range(3):
            ax = fig.add_subplot(field_grid[r, c])
            limit = velocity_limit if c < 2 else vort_limit
            im = ax.imshow(
                fields[r][c].T,
                origin="lower",
                cmap=cmap,
                norm=Normalize(vmin=-limit, vmax=limit),
                interpolation="nearest",
                rasterized=True,
            )
            if c < 2:
                velocity_images.append(im)
            else:
                vort_images.append(im)
            ax.set_facecolor(KOL_BACKGROUND_SOFT)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(col_labels[c], fontsize=10.5, pad=3)
            if c == 0:
                ax.text(
                    0.03,
                    0.92,
                    row_labels[r],
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9.5,
                    color="#111111",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, boxstyle="round,pad=0.12"),
                )

    cax_vel = fig.add_subplot(field_grid[0, 3])
    cb_vel = fig.colorbar(velocity_images[-1], cax=cax_vel)
    cb_vel.ax.tick_params(labelsize=8.5, length=0, pad=2)
    cb_vel.ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    cb_vel.outline.set_visible(False)
    cb_vel.ax.set_title("vel.", fontsize=8.5, pad=3)
    cax_vort = fig.add_subplot(field_grid[1, 3])
    cb_vort = fig.colorbar(vort_images[-1], cax=cax_vort)
    cb_vort.ax.tick_params(labelsize=8.5, length=0, pad=2)
    cb_vort.ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    cb_vort.outline.set_visible(False)
    cb_vort.ax.set_title(r"$\omega$", fontsize=9.5, pad=3)

    right = outer[0, 1].subgridspec(3, 1, hspace=0.55, height_ratios=[1.0, 1.0, 1.05])
    ax_nrmse = fig.add_subplot(right[0, 0])
    ax_crps = fig.add_subplot(right[1, 0])
    ax_heat = fig.add_subplot(right[2, 0])

    intervals = [1, 2, 4, 6, 8]
    for ax, metric, ylabel in ((ax_nrmse, "nrmse", "nRMSE"), (ax_crps, "crps", "CRPS")):
        for method in BOX_ORDER:
            y = [sweep[(8, interval, method)][metric] for interval in intervals]
            ax.plot(
                intervals,
                y,
                marker="o",
                markersize=3.6,
                linewidth=1.45,
                color=METHOD_COLORS[method],
                label=METHOD_DISPLAY[method],
            )
        if metric == "nrmse":
            ax.axhline(0.25, color="#777777", linestyle="--", linewidth=0.9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(intervals)
        ax.tick_params(labelsize=8.5, length=2)
        ax.set_facecolor(KOL_BACKGROUND_SOFT)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        if metric == "nrmse":
            ax.set_title("k=8 temporal-sparsity sweep", fontsize=10.2, pad=3)
        if metric == "crps":
            ax.set_xlabel("Temporal observation interval", fontsize=10)
        else:
            ax.set_xticklabels([])
            ax.legend(loc="upper left", ncol=2, fontsize=8.2, frameon=False, handlelength=1.2, columnspacing=0.8)

    heat = np.array([[sweep[(k, interval, "apce")]["nrmse"] for interval in intervals] for k in (4, 6, 8)])
    im_heat = ax_heat.imshow(heat, origin="lower", cmap="magma_r", vmin=0.0, vmax=max(0.75, float(np.max(heat))))
    ax_heat.set_xticks(range(len(intervals)))
    ax_heat.set_xticklabels([str(v) for v in intervals], fontsize=8.5)
    ax_heat.set_yticks(range(3))
    ax_heat.set_yticklabels(["4", "6", "8"], fontsize=8.5)
    ax_heat.set_xlabel("Temporal interval", fontsize=10)
    ax_heat.set_ylabel(r"Forcing $k$", fontsize=10)
    ax_heat.set_title("APCE mean nRMSE boundary", fontsize=10.2, pad=3)
    for r in range(heat.shape[0]):
        for c in range(heat.shape[1]):
            ax_heat.text(c, r, f"{heat[r, c]:.2f}", ha="center", va="center", fontsize=7.5, color="white" if heat[r, c] > 0.42 else "#111111")
    cbar_heat = fig.colorbar(im_heat, ax=ax_heat, fraction=0.046, pad=0.025)
    cbar_heat.ax.tick_params(labelsize=8, length=0, pad=2)
    cbar_heat.outline.set_visible(False)

    outputs = save_all(fig, output_base)
    plt.close(fig)
    qa = {
        "backend": "Python/matplotlib",
        "figure": "KOL high-k and temporal-sparsity boundary audit",
        "panel_letters": [],
        "source_trace": str(source_trace),
        "source_trace_sha256": source_sha256,
        "seed": seed,
        "representative_step": int(step),
        "representative_time": float(times[step]),
        "statistical_source_root": str(sweep_root),
        "remote_statistical_source_roots": [
            f"<HILDA_RESULTS_ROOT>/results/figure4_kolmogorov64_velocityobs_re1500_k{k}_s16_t{interval}_smoke_5seeds_20260816_2gpu"
            for k in (4, 6, 8)
            for interval in (1, 2, 4, 6, 8)
        ],
        "remote_representative_trace": "<HILDA_RESULTS_ROOT>/results/figure4_kolmogorov64_velocityobs_re1500_k8_s16_t8_smoke_5seeds_20260816_2gpu/artifacts/method_traces/sensor16/apce/seed_2026081604.npz",
        "stress_condition": {"Re": 1500, "k": 8, "sensor_grid": "16x16", "temporal_interval": 8},
        "methods": BOX_ORDER,
        "figure_size_inches": [14.6, 6.2],
        "velocity_color_limit": velocity_limit,
        "vorticity_color_limit": vort_limit,
        "vorticity_color_limit_rule": "symmetric 99th percentile across displayed Ref./APCE vorticity fields",
        "outputs": outputs,
    }
    output_base.with_suffix(".qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    return outputs


def main() -> None:
    recon_truth, recon_mean, recon_obs, recon_sensor, recon_times = load_reconstruction_trace(RECON_TRACE)
    forecast_truth, forecast_mean, forecast_obs, forecast_sensor, forecast_times, blackout_start = load_forecast_trace(FORECAST_TRACE)
    stress_truth, stress_mean, stress_obs, stress_sensor, stress_times = load_stress_trace(STRESS_REP_TRACE)

    plot_velocity_atlas(
        recon_truth,
        recon_mean,
        recon_times,
        steps=[0, 6, 12, 18, 24, 30, 36, 42, 50, 58],
        title="KOL velocity reconstruction atlas",
        output_base=OUT_RECON,
        source_trace=RECON_TRACE,
        source_sha256=sha256(RECON_TRACE),
        seed=2026081612,
        config_text=r"k=2, Re=1500, 16×16 sensors, temporal interval t=4",
    )

    plot_velocity_atlas(
        forecast_truth,
        forecast_mean,
        forecast_times,
        steps=[41, 43, 45, 47, 49, 51, 53, 55, 57, 58],
        title="KOL blackout forecast atlas",
        output_base=OUT_FORECAST,
        source_trace=FORECAST_TRACE,
        source_sha256=sha256(FORECAST_TRACE),
        seed=2026081637,
        config_text=r"blackout after step 40; 10 forecast snapshots shown",
    )

    plot_highk_boundary_audit(
        stress_truth,
        stress_mean,
        stress_times,
        STRESS_SWEEP_ROOT,
        output_base=OUT_STRESS,
        source_trace=STRESS_REP_TRACE,
        source_sha256=sha256(STRESS_REP_TRACE),
        seed=2026081604,
    )


if __name__ == "__main__":
    main()
