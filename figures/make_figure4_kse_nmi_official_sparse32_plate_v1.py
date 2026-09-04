from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_DIR = ROOT / "figures" / "kse_32x_tracefix_assets"
DEFAULT_SOURCE = DEFAULT_ASSET_DIR / "kse_nmi_official_sparse128_source_data.npz"
DEFAULT_PCE_TRACE = DEFAULT_ASSET_DIR / "kse_nmi32x_pce_seed02_sample02.npz"
DEFAULT_APCE_TRACE = DEFAULT_ASSET_DIR / "kse_nmi32x_apce_seed02_sample02.npz"
DEFAULT_OUTPUT = ROOT / "figures" / "figure4_kse_nmi_official_sparse32_plate_v1"


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.0,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "legend.frameon": False,
        }
    )


def save_all(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=700, bbox_inches="tight", pad_inches=0.018)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.018)
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.018)
    fig.savefig(output_base.with_suffix(".tiff"), dpi=700, bbox_inches="tight", pad_inches=0.018)


def make_field_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "kse_field",
        ["#214c78", "#f4f2ee", "#9d3c35"],
        N=256,
    )


def make_error_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "kse_error",
        ["#f7fbff", "#d5e7f5", "#7daed1", "#1e4c7d"],
        N=256,
    )


def sparse_observation_map(truth: np.ndarray, observations: np.ndarray, sensor_indices: np.ndarray) -> np.ndarray:
    obs = np.full_like(truth, np.nan, dtype=float)
    obs[:, sensor_indices] = observations
    return obs


def strip_labels(ax: plt.Axes, *, xlabel: bool = False, ylabel: bool = False) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    if not xlabel:
        ax.set_xlabel("")
    if not ylabel:
        ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_colorbar_strip(
    fig: plt.Figure,
    left: float,
    bottom: float,
    width: float,
    height: float,
    im: mpl.image.AxesImage,
    *,
    label: str,
    ticks: list[float] | None = None,
    ticklabels: list[str] | None = None,
    label_size: float = 7.0,
    tick_size: float = 6.4,
) -> None:
    cax = fig.add_axes([left, bottom, width, height])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    if ticks is not None:
        cb.set_ticks(ticks)
    if ticklabels is not None:
        cb.set_ticklabels(ticklabels)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=tick_size, length=0, pad=0.5)
    cax.text(0.5, -1.15, label, transform=cax.transAxes, ha="center", va="top", fontsize=label_size)


def read_trace(path: Path) -> dict[str, np.ndarray]:
    arr = np.load(path, allow_pickle=True)
    return {
        "mean_states": np.asarray(arr["mean_states"], dtype=float),
        "mu_estimate_history": np.asarray(arr["mu_estimate_history"], dtype=float),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw a Nature-style KSE sparse-observation hero plate for Figure 4.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--pce-trace", type=Path, default=DEFAULT_PCE_TRACE)
    parser.add_argument("--apce-trace", type=Path, default=DEFAULT_APCE_TRACE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-index", type=int, default=2)
    parser.add_argument("--downsampling-factor", type=int, default=32)
    args = parser.parse_args()

    configure_matplotlib()

    source = np.load(args.source)
    truth_all = np.asarray(source["all_truth"], dtype=float)
    x = np.asarray(source["x"], dtype=float)
    t = np.asarray(source["t"], dtype=float)
    sensor_indices = np.asarray(source["sensor_indices"], dtype=int)
    selected_sample = int(np.asarray(source["selected_sample_index"]).item())

    sample_index = int(args.sample_index)
    truth = np.asarray(truth_all[sample_index], dtype=float)
    obs = np.asarray(source["selected_observations"], dtype=float)
    # If the selected sample in the source asset differs, reconstruct observations from truth.
    if selected_sample != sample_index:
        obs = truth[:, sensor_indices]
    obs_map = sparse_observation_map(truth, obs, sensor_indices)

    pce = read_trace(args.pce_trace)
    apce = read_trace(args.apce_trace)
    pce_mean = np.asarray(pce["mean_states"], dtype=float)
    apce_mean = np.asarray(apce["mean_states"], dtype=float)

    pce_err = np.abs(pce_mean - truth)
    apce_err = np.abs(apce_mean - truth)

    field_abs = np.abs(truth)
    field_abs = field_abs[np.isfinite(field_abs)]
    vmax = float(np.nanpercentile(field_abs, 99.4))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(np.abs(truth)))
    emax = float(np.nanpercentile(np.concatenate([pce_err.ravel(), apce_err.ravel()]), 99.4))
    if not np.isfinite(emax) or emax <= 0:
        emax = float(np.nanmax(np.concatenate([pce_err.ravel(), apce_err.ravel()])))

    field_cmap = make_field_cmap()
    err_cmap = make_error_cmap()
    field_cmap.set_bad("white", alpha=1.0)

    fig = plt.figure(figsize=(11.6, 2.95))
    gs = fig.add_gridspec(
        1,
        6,
        width_ratios=[0.98, 1.02, 1.02, 1.02, 1.02, 1.02],
        left=0.035,
        right=0.995,
        top=0.78,
        bottom=0.20,
        wspace=0.12,
    )

    axes = [fig.add_subplot(gs[0, i]) for i in range(6)]
    titles = [
        "Sparse observations",
        "Truth",
        "PCE reconstruction",
        "APCE reconstruction",
        "PCE absolute error",
        "APCE absolute error",
    ]
    letter_colors = {
        0: "#223f60",
        1: "#111111",
        2: "#234f86",
        3: "#a85c18",
        4: "#65758a",
        5: "#8b4c2f",
    }

    im0 = axes[0].imshow(
        np.ma.masked_invalid(obs_map.T),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=field_cmap,
        vmin=-vmax,
        vmax=vmax,
        rasterized=True,
    )
    im1 = axes[1].imshow(
        truth.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=field_cmap,
        vmin=-vmax,
        vmax=vmax,
        rasterized=True,
    )
    im2 = axes[2].imshow(
        pce_mean.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=field_cmap,
        vmin=-vmax,
        vmax=vmax,
        rasterized=True,
    )
    im3 = axes[3].imshow(
        apce_mean.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=field_cmap,
        vmin=-vmax,
        vmax=vmax,
        rasterized=True,
    )
    im4 = axes[4].imshow(
        pce_err.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=err_cmap,
        vmin=0.0,
        vmax=emax,
        rasterized=True,
    )
    im5 = axes[5].imshow(
        apce_err.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=err_cmap,
        vmin=0.0,
        vmax=emax,
        rasterized=True,
    )

    for ax, title, idx in zip(axes, titles, range(6)):
        strip_labels(ax)
        ax.set_title(title, fontsize=8.0, pad=4.0, color=letter_colors[idx])
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[0].set_xlabel("$t$", fontsize=7.8, labelpad=1)
    axes[0].set_ylabel("$x$", fontsize=7.8, labelpad=1)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].text(
        0.50,
        0.02,
        f"{args.downsampling_factor}× downsampling\n{len(sensor_indices)} of {len(x)} sensors",
        transform=axes[0].transAxes,
        ha="center",
        va="bottom",
        fontsize=6.8,
        color="#4a4a4a",
    )

    # Small parameter-estimate readout, so the reconstructed panels carry the alpha story too.
    pce_mu = float(np.asarray(pce["mu_estimate_history"], dtype=float)[-1])
    apce_mu = float(np.asarray(apce["mu_estimate_history"], dtype=float)[-1])
    axes[2].set_title(
        f"PCE reconstruction\n$\\hat\\mu={pce_mu:.2f}$",
        fontsize=8.0,
        pad=4.0,
        color=letter_colors[2],
    )
    axes[3].set_title(
        f"APCE reconstruction\n$\\hat\\mu={apce_mu:.2f}$",
        fontsize=8.0,
        pad=4.0,
        color=letter_colors[3],
    )

    fig.text(0.035, 0.965, "a", fontsize=11.2, fontweight="bold", ha="left", va="top")
    fig.text(0.072, 0.965, "Kuramoto–Sivashinsky", fontsize=9.0, ha="left", va="top")
    fig.text(0.072, 0.905, "Official NMI/S3GM test field; 1024 grid points, 32 regular sensors", fontsize=6.9, ha="left", va="top", color="#555555")
    true_mu = (1.1, 2.5, 3.2)[sample_index % 3]
    fig.text(0.995, 0.965, f"sample {sample_index} / true μ = {true_mu:.1f}", fontsize=6.4, ha="right", va="top", color="#777777")

    # Colorbar strips
    field_left = axes[0].get_position().x0
    field_right = axes[3].get_position().x1
    err_left = axes[4].get_position().x0
    err_right = axes[5].get_position().x1
    strip_y = 0.095
    strip_h = 0.028
    add_colorbar_strip(
        fig,
        field_left,
        strip_y,
        field_right - field_left,
        strip_h,
        im1,
        label="$u$",
        ticks=[-vmax, vmax],
        ticklabels=[f"{-vmax:.1f}", f"{vmax:.1f}"],
    )
    add_colorbar_strip(
        fig,
        err_left,
        strip_y,
        err_right - err_left,
        strip_h,
        im4,
        label="$|u-\\hat u|$",
        ticks=[0.0, emax],
        ticklabels=["0", f"{emax:.1e}"],
    )

    save_all(fig, args.output)

    qa = {
        "source": str(args.source),
        "pce_trace": str(args.pce_trace),
        "apce_trace": str(args.apce_trace),
        "output_base": str(args.output),
        "sample_index": sample_index,
        "downsampling_factor": int(args.downsampling_factor),
        "truth_shape": list(truth.shape),
        "observations_shape": list(obs.shape),
        "pce_nrmse_vs_truth": float(np.sqrt(np.square(pce_mean - truth).sum() / max(np.square(truth).sum(), 1.0e-30))),
        "apce_nrmse_vs_truth": float(np.sqrt(np.square(apce_mean - truth).sum() / max(np.square(truth).sum(), 1.0e-30))),
        "pce_mu_estimate": pce_mu,
        "apce_mu_estimate": apce_mu,
        "note": "Hero plate uses sparse observations, truth, PCE/APCE reconstructions, and absolute error panels.",
    }
    (args.output.with_suffix(".json")).write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(qa, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
