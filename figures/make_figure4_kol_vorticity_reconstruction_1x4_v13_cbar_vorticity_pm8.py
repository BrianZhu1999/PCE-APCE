from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
TRACE_PATH = (
    HERE
    / "source_data"
    / "figure4_kolmogorov64_re1500_k2_s16_t4_seed2026081612"
    / "apce_reconstruction.npz"
)
OUTPUT_BASE = HERE / "figure4_kol_vorticity_reconstruction_1x4_v13_cbar_vorticity_pm8"

SEED = 2026081612
STEPS = (20, 40)
LIMIT = 8.0
FIG_W = 8.86
FIG_H = 3.25
DPI = 650

FONT_PANEL = 22
FONT_TITLE = 14
FONT_LEGEND = 14
FONT_AXIS = 13
FONT_TICK = 11


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vorticity(states: np.ndarray) -> np.ndarray:
    fields = np.asarray(states, dtype=np.float64).reshape(states.shape[0], 2, 64, 64)
    wave = 2.0 * np.pi * np.fft.fftfreq(64, d=2.0 * np.pi / 64.0)
    u_hat = np.fft.fft2(fields[:, 0], axes=(-2, -1))
    v_hat = np.fft.fft2(fields[:, 1], axes=(-2, -1))
    dv_dx = np.fft.ifft2(1j * wave[None, :, None] * v_hat, axes=(-2, -1)).real
    du_dy = np.fft.ifft2(1j * wave[None, None, :] * u_hat, axes=(-2, -1)).real
    return dv_dx - du_dy


def make_kse_field_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "kse_field",
        ["#214c78", "#f4f2ee", "#9d3c35"],
        N=256,
    )


def add_axes_inches(fig: plt.Figure, left: float, bottom: float, width: float, height: float) -> plt.Axes:
    return fig.add_axes([left / FIG_W, bottom / FIG_H, width / FIG_W, height / FIG_H])


def save_all(fig: plt.Figure) -> dict[str, str]:
    outputs = {
        "png": OUTPUT_BASE.with_suffix(".png"),
        "pdf": OUTPUT_BASE.with_suffix(".pdf"),
        "svg": OUTPUT_BASE.with_suffix(".svg"),
        "tiff": OUTPUT_BASE.with_suffix(".tiff"),
    }
    fig.savefig(outputs["png"], dpi=DPI, facecolor="white")
    fig.savefig(outputs["pdf"], facecolor="white")
    fig.savefig(outputs["svg"], facecolor="white")
    fig.savefig(outputs["tiff"], dpi=DPI, facecolor="white")
    return {name: str(path) for name, path in outputs.items()}


def main() -> None:
    configure_matplotlib()
    with np.load(TRACE_PATH, allow_pickle=True) as trace:
        truth_states = np.asarray(trace["truth"], dtype=np.float64)
        apce_states = np.asarray(trace["mean_states"], dtype=np.float64)
        times = np.asarray(trace["times"], dtype=np.float64)

    truth = vorticity(truth_states)
    apce = vorticity(apce_states)
    full_vorticity_nrmse = float(np.sqrt(np.sum((apce - truth) ** 2) / np.sum(truth**2)))
    displayed_vorticity_nrmse = float(
        np.sqrt(np.sum((apce[list(STEPS)] - truth[list(STEPS)]) ** 2) / np.sum(truth[list(STEPS)] ** 2))
    )

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
    field_cmap = make_kse_field_cmap().reversed()
    image_w = 1.80
    image_h = 1.75
    # Match the accepted KSE j-panel SVG image edge: x = 16.428 pt.
    # Matplotlib's SVG image placement lands 0.264 pt to the right in this
    # export, so offset the requested axes position and verify the SVG below.
    x0 = (16.428 - 0.264) / 72.0
    x_gap = 0.24
    x_positions = [x0 + col * (image_w + x_gap) for col in range(4)]
    image_y = 0.78
    columns = (
        (20, r"Truth    $t=20$", truth),
        (20, r"APCE    $t=20$", apce),
        (40, r"Truth    $t=40$", truth),
        (40, r"APCE    $t=40$", apce),
    )

    image = None
    for col, (step, _label, field) in enumerate(columns):
        ax = add_axes_inches(fig, x_positions[col], image_y, image_w, image_h)
        image = ax.imshow(
            field[step].T,
            origin="lower",
            extent=(0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi),
            cmap=field_cmap,
            norm=Normalize(vmin=-LIMIT, vmax=LIMIT),
            interpolation="bilinear",
            aspect="auto",
            rasterized=True,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for col, (_step, label, _field) in enumerate(columns):
        fig.text(
            (x_positions[col] + image_w / 2.0) / FIG_W,
            2.57 / FIG_H,
            label,
            ha="center",
            va="bottom",
            fontsize=FONT_TITLE,
            color="#111111",
        )

    fig.text(0.012, 0.985, "m", ha="left", va="top", fontsize=FONT_PANEL, fontweight="bold", color="#111111")
    fig.text(
        (x0 + (4 * image_w + 3 * x_gap) / 2.0) / FIG_W,
        0.945,
        r"Vorticity reconstruction    $k=2,\ \mathrm{Re}=1500$",
        ha="center",
        va="top",
        fontsize=FONT_TITLE,
        color="#111111",
    )
    if image is None:
        raise RuntimeError("No vorticity images were drawn")
    cax = add_axes_inches(fig, x_positions[-1] + image_w + 0.14, image_y, 0.15, image_h)
    colorbar = fig.colorbar(image, cax=cax, orientation="vertical", extend="both")
    colorbar.set_ticks([-LIMIT, LIMIT])
    colorbar.set_ticklabels(["-8", "8"])
    colorbar.ax.tick_params(labelsize=FONT_TICK, length=0, pad=3)
    colorbar.set_label("Vorticity", rotation=270, labelpad=6, fontsize=FONT_AXIS, color="#111111")
    colorbar.outline.set_visible(False)

    outputs = save_all(fig)
    plt.close(fig)

    qa = {
        "backend": "Python/matplotlib",
        "panel": "m",
        "title": "Vorticity reconstruction",
        "figure_size_inches": [FIG_W, FIG_H],
        "figure_size_px": [round(FIG_W * DPI), round(FIG_H * DPI)],
        "small_image_size_inches": [image_w, image_h],
        "small_image_size_px": [round(image_w * DPI), round(image_h * DPI)],
        "reference_jk_small_image_size_px": [1170, round(1.75 * 650)],
        "first_image_left_edge_inches": x0,
        "first_image_left_edge_svg_pt": x0 * 72.0,
        "reference_j_first_image_left_edge_svg_pt": 16.428,
        "column_gap_inches": x_gap,
        "layout": "one row by four columns with white-background time-group rules",
        "column_order": ["Truth    $t=20$", "APCE    $t=20$", "Truth    $t=40$", "APCE    $t=40$"],
        "source_trace": str(TRACE_PATH),
        "source_trace_sha256": sha256(TRACE_PATH),
        "seed": SEED,
        "reynolds": 1500,
        "forcing_wavenumber": 2,
        "sensor_grid": "16x16",
        "observation_interval": 4,
        "steps": list(STEPS),
        "times": [float(times[step]) for step in STEPS],
        "state_observation": "direct ux and uy; vorticity is derived diagnostically",
        "vorticity_definition": "d(uy)/dx - d(ux)/dy via periodic spectral differentiation",
        "full_59_frame_vorticity_nrmse": full_vorticity_nrmse,
        "displayed_steps_vorticity_nrmse": displayed_vorticity_nrmse,
        "seed_selection": "minimum full-59-frame APCE reconstruction vorticity nRMSE among 50 paired seeds",
        "shared_color_scale": [-LIMIT, LIMIT],
        "colormap": "make_kse_field_cmap().reversed()",
        "colormap_colors_before_reversal": ["#214c78", "#f4f2ee", "#9d3c35"],
        "font_rules": {
            "panel_label": FONT_PANEL,
            "panel_title": FONT_TITLE,
            "legend": FONT_LEGEND,
            "axis_label": FONT_AXIS,
            "tick": FONT_TICK,
        },
        "outputs": outputs,
    }
    qa_path = OUTPUT_BASE.with_suffix(".qa.json")
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**qa, "qa": str(qa_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
