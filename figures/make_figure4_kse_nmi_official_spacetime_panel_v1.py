from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SOURCE = Path(
    r"<LOCAL_PATH>图4绘制\S3GM_NMI_2024\source_data\kse_nmi_official_sparse128_source_data.npz"
)
DEFAULT_OUTPUT = Path(r"<LOCAL_PATH>图4绘制\figure4_kse_nmi_official_sparse128_plate_v1")


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.0,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "legend.frameon": False,
        }
    )


def add_image(ax: plt.Axes, arr: np.ndarray, *, extent: tuple[float, float, float, float], cmap: str, vmin: float, vmax: float) -> mpl.image.AxesImage:
    im = ax.imshow(
        arr.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        rasterized=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def add_colorbar_strip(fig: plt.Figure, ax: plt.Axes, im: mpl.image.AxesImage, label: str, tick_left: str, tick_right: str) -> None:
    bbox = ax.get_position()
    cax = fig.add_axes([bbox.x0, bbox.y0 - 0.055, bbox.width, 0.030])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([im.norm.vmin, im.norm.vmax])
    cb.set_ticklabels([tick_left, tick_right])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.5, length=0, pad=1)
    cax.text(0.5, -1.15, label, transform=cax.transAxes, ha="center", va="top", fontsize=6.8, fontstyle="italic")


def save_all(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=650, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(output_base.with_suffix(".tiff"), dpi=650, bbox_inches="tight", pad_inches=0.025)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw an NMI-style KSE sparse-observation image plate.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--title", default="Kuramoto–Sivashinsky")
    args = parser.parse_args()

    configure_matplotlib()
    src = np.load(args.source)
    truth = np.asarray(src["selected_truth"], dtype=float)
    obs = np.asarray(src["selected_observations"], dtype=float)
    recon = np.asarray(src["selected_reconstruction_spectral_interp"], dtype=float)
    err = np.asarray(src["selected_abs_error"], dtype=float)
    t = np.asarray(src["t"], dtype=float)
    x = np.asarray(src["x"], dtype=float)
    selected = int(np.asarray(src["selected_sample_index"]).item())

    # Match the NMI plate: time horizontal, space vertical, compact white page.
    extent_full = (float(t[0]), float(t[-1]), float(x[0]), float(x[-1]))
    extent_obs = (float(t[0]), float(t[-1]), float(0), float(obs.shape[1] - 1))
    v = float(np.nanpercentile(np.abs(truth), 99.2))
    emax = float(np.nanpercentile(err, 99.0))

    fig = plt.figure(figsize=(7.15, 2.08))
    gs = fig.add_gridspec(
        1,
        5,
        width_ratios=[0.90, 0.12, 1.00, 1.00, 1.00],
        left=0.035,
        right=0.992,
        top=0.70,
        bottom=0.27,
        wspace=0.24,
    )
    ax_obs = fig.add_subplot(gs[0, 0])
    ax_arrow = fig.add_subplot(gs[0, 1])
    ax_ref = fig.add_subplot(gs[0, 2])
    ax_rec = fig.add_subplot(gs[0, 3])
    ax_err = fig.add_subplot(gs[0, 4])

    im_obs = add_image(ax_obs, obs, extent=extent_obs, cmap="RdBu_r", vmin=-v, vmax=v)
    im_ref = add_image(ax_ref, truth, extent=extent_full, cmap="RdBu_r", vmin=-v, vmax=v)
    im_rec = add_image(ax_rec, recon, extent=extent_full, cmap="RdBu_r", vmin=-v, vmax=v)
    im_err = add_image(ax_err, err, extent=extent_full, cmap="mako" if "mako" in plt.colormaps() else "Blues", vmin=0.0, vmax=emax)

    ax_obs.set_title("Sparse observations", fontsize=7.8, pad=3.5)
    ax_ref.set_title("Ref.", fontsize=7.8, pad=3.5)
    ax_rec.set_title("Spectral interp.", fontsize=7.8, pad=3.5)
    ax_err.set_title("Absolute error", fontsize=7.8, pad=3.5)

    ax_obs.set_xlabel("$t$", fontsize=7.5, labelpad=1)
    ax_obs.set_ylabel("$x$", fontsize=7.5, rotation=0, labelpad=7)
    ax_obs.text(0.5, -0.18, "8× downsampling\n128 of 1024 sensors", transform=ax_obs.transAxes, ha="center", va="top", fontsize=6.8)

    ax_arrow.set_axis_off()
    ax_arrow.annotate(
        "",
        xy=(0.95, 0.50),
        xytext=(0.05, 0.50),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>,head_width=0.9,head_length=1.0", lw=0, color="#66bdc5", mutation_scale=18),
    )

    fig.text(0.035, 0.965, "a", fontsize=10.8, fontweight="bold", ha="left", va="top")
    fig.text(0.075, 0.965, args.title, fontsize=8.8, ha="left", va="top")
    fig.text(0.075, 0.905, "Official NMI/S3GM KSE test field; 1024 spatial grid, 128 regular sensors", fontsize=6.9, ha="left", va="top", color="#555555")
    fig.text(0.992, 0.965, f"visual sample {selected}", fontsize=6.5, ha="right", va="top", color="#777777")

    add_colorbar_strip(fig, ax_ref, im_ref, "$u$", f"{-v:.1f}", f"{v:.1f}")
    add_colorbar_strip(fig, ax_err, im_err, "|error|", "0", f"{emax:.1e}")

    save_all(fig, args.output)

    qa = {
        "source": str(args.source),
        "output_base": str(args.output),
        "selected_sample_index": selected,
        "truth_shape": list(truth.shape),
        "observations_shape": list(obs.shape),
        "nrmse_spectral_interp": float(np.sqrt(np.square(recon - truth).sum() / max(np.square(truth).sum(), 1.0e-30))),
        "note": "Spectral interpolation is a visual/data sanity check, not a PCE/APCE result.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(qa, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
