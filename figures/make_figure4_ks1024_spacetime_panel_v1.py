from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = Path(r"<LOCAL_PATH>图4绘制\ks1024_sparse128_smoke_20260814")
OUT_DIR = Path(r"<LOCAL_PATH>图4绘制")
OUT_STEM = OUT_DIR / "figure4_ks1024_sparse128_spacetime_v2"


def load_npz(name: str) -> dict[str, np.ndarray]:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.linewidth": 0.75,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
        }
    )


def add_heatmap(
    ax: plt.Axes,
    field: np.ndarray,
    times: np.ndarray,
    x: np.ndarray,
    title: str,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    ylabel: bool = False,
) -> mpl.image.AxesImage:
    im = ax.imshow(
        field.T,
        origin="lower",
        aspect="auto",
        extent=(times[0], times[-1], x[0], x[-1]),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        rasterized=True,
    )
    ax.set_title(title, fontsize=7.8, pad=2.4)
    ax.set_xlabel(r"$t$", fontsize=7.0, labelpad=1.3)
    if ylabel:
        ax.set_ylabel(r"$x$", fontsize=7.0, labelpad=1.3)
    else:
        ax.set_yticklabels([])
        ax.set_ylabel("")
    ax.set_xticks([0, 6, 12])
    ax.set_yticks([0, 11, 22])
    ax.tick_params(labelsize=6.2, pad=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.72)
        spine.set_color("#202020")
    return im


def main() -> None:
    style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    common = load_npz("ks1024_sparse128_common_assets.npz")
    pce = load_npz("pce_seed_2026081400.npz")
    apce = load_npz("apce_seed_2026081400.npz")

    times = common["times"]
    x = common["coordinates"]
    truth = common["truth_states"]
    pce_field = pce["mean_states"]
    apce_field = apce["mean_states"]

    obs_steps = common["observation_steps"].astype(int)
    obs_indices = common["observation_indices"].astype(int)
    observations = common["observations"]
    obs_times = np.repeat(times[obs_steps], obs_indices.size)
    obs_x = np.tile(x[obs_indices], obs_steps.size)
    obs_values = observations.reshape(-1)

    state_stack = np.concatenate([truth.ravel(), pce_field.ravel(), apce_field.ravel()])
    vmax = float(np.nanpercentile(np.abs(state_stack), 99.5))
    vmin = -vmax
    err = np.abs(apce_field - truth)
    err_vmax = float(np.nanpercentile(err, 99.4))

    # Nature/NMI-like field plate: compact horizontal heatmaps, sparse panel as
    # colored sensor samples on a black physical-space canvas.
    fig = plt.figure(figsize=(7.35, 1.82), facecolor="white")
    gs = fig.add_gridspec(
        1,
        7,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 1.0, 0.050, 0.050],
        left=0.055,
        right=0.985,
        bottom=0.250,
        top=0.800,
        wspace=0.135,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(5)]
    cax_state = fig.add_subplot(gs[0, 5])
    cax_err = fig.add_subplot(gs[0, 6])

    state_cmap = mpl.colormaps["RdBu_r"].copy()
    err_cmap = mpl.colormaps["magma"].copy()

    im0 = add_heatmap(axes[0], truth, times, x, "Truth", cmap=state_cmap, vmin=vmin, vmax=vmax, ylabel=True)

    axes[1].set_facecolor("#080808")
    axes[1].scatter(
        obs_times,
        obs_x,
        c=obs_values,
        cmap=state_cmap,
        vmin=vmin,
        vmax=vmax,
        marker="s",
        s=0.16,
        linewidths=0,
        alpha=0.96,
        rasterized=True,
    )
    axes[1].set_title("Sparse sensors", fontsize=7.8, pad=2.4)
    axes[1].set_xlim(times[0], times[-1])
    axes[1].set_ylim(x[0], x[-1])
    axes[1].set_xlabel(r"$t$", fontsize=7.0, labelpad=1.3)
    axes[1].set_yticklabels([])
    axes[1].set_xticks([0, 6, 12])
    axes[1].set_yticks([0, 11, 22])
    axes[1].tick_params(labelsize=6.2, pad=1.2)
    for spine in axes[1].spines.values():
        spine.set_linewidth(0.72)
        spine.set_color("#202020")

    add_heatmap(axes[2], pce_field, times, x, "PCE", cmap=state_cmap, vmin=vmin, vmax=vmax)
    add_heatmap(axes[3], apce_field, times, x, "APCE", cmap=state_cmap, vmin=vmin, vmax=vmax)
    im4 = add_heatmap(axes[4], err, times, x, r"$|$APCE$-$Truth$|$", cmap=err_cmap, vmin=0.0, vmax=err_vmax)

    cb0 = fig.colorbar(im0, cax=cax_state)
    cb0.set_label(r"$u(x,t)$", fontsize=6.7, labelpad=1.6)
    cb0.ax.tick_params(labelsize=5.8, length=2, pad=1)
    cb1 = fig.colorbar(im4, cax=cax_err)
    cb1.set_label("error", fontsize=6.7, labelpad=1.6)
    cb1.ax.tick_params(labelsize=5.8, length=2, pad=1)

    fig.text(0.010, 0.955, "a", fontsize=11.5, fontweight="bold", ha="left", va="top")
    fig.text(
        0.055,
        0.942,
        "KS field reconstruction from 128 of 1024 spatial sensors",
        fontsize=7.4,
        ha="left",
        va="top",
        color="#202020",
    )

    for suffix, kwargs in {
        ".png": {"dpi": 600},
        ".tiff": {"dpi": 600},
        ".pdf": {},
        ".svg": {},
    }.items():
        fig.savefig(OUT_STEM.with_suffix(suffix), bbox_inches="tight", pad_inches=0.025, **kwargs)

    manifest = {
        "figure": "figure4_ks1024_sparse128_spacetime_v2",
        "seed": 2026081400,
        "source_data_dir": str(DATA_DIR),
        "panels": ["Truth", "Sparse sensors", "PCE", "APCE", "|APCE-Truth|"],
        "state_dim": int(truth.shape[1]),
        "time_steps": int(truth.shape[0]),
        "observed_points": int(obs_indices.size),
        "observation_snapshots": int(obs_steps.size),
        "state_vmin": vmin,
        "state_vmax": vmax,
        "error_vmax": err_vmax,
    }
    OUT_STEM.with_suffix(".json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    plt.close(fig)


if __name__ == "__main__":
    main()
