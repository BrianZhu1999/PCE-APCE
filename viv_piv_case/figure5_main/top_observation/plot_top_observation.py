"""Draw the standalone top observation/model diagram for Figure 5."""
from __future__ import annotations

import hashlib
import json
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

mpl.use("Agg")


HERE = pathlib.Path(__file__).resolve().parent
SOURCE = HERE.parent / "source_data" / "figure5_viv_piv_compact_source.npz"
OUT = HERE / "outputs"
FIGURE = "figure5_top_observation"
DPI = 650
FIG_W_PX = 11532
FIG_H_PX = 2800
FIG_W = FIG_W_PX / DPI
FIG_H = FIG_H_PX / DPI

FONT_PANEL = 22
FONT_TITLE = 14
FONT_LEGEND = 14
FONT_AXIS = 13
FONT_TICK = 11


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": FONT_TICK,
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "legend.frameon": False,
        }
    )


def add_box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, text: str, *, face: str, edge: str = "#303030", fontsize: int = FONT_TITLE) -> None:
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.025,rounding_size=0.035",
            facecolor=face,
            edgecolor=edge,
            linewidth=0.9,
        )
    )
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize, fontweight="normal")


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, color: str = "#4E79A7", lw: float = 1.8, style: str = "-|>") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=16,
            linewidth=lw,
            color=color,
            shrinkA=5,
            shrinkB=5,
            connectionstyle="arc3,rad=0.0",
        )
    )


def add_axes_inches(fig: plt.Figure, left: float, bottom: float, width: float, height: float) -> plt.Axes:
    return fig.add_axes([left / FIG_W, bottom / FIG_H, width / FIG_W, height / FIG_H])


def main() -> None:
    configure()
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(SOURCE, allow_pickle=False) as source:
        x = np.asarray(source["x_over_d"], dtype=float)
        y = np.asarray(source["y_over_d"], dtype=float)
        speed = np.asarray(source["truth_origin_speed"], dtype=float)
        sensor_x = np.asarray(source["sensor_x_over_d"], dtype=float)
        sensor_y = np.asarray(source["sensor_y_over_d"], dtype=float)
        time_s = np.asarray(source["cylinder_time_s"], dtype=float)
        displacement = np.asarray(source["cylinder_displacement_over_d"], dtype=float)
        origin_time = float(source["origin_time_s"])
        final_time = float(source["final_time_s"])
        cylinder_y = float(source["cylinder_origin_y_over_d"])

    speed_values = speed[np.isfinite(speed)]
    speed_vmax = float(np.percentile(speed_values, 99.5))
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")

    # Left block: real observation field plus the reduced-order model interface.
    model_ax = add_axes_inches(fig, 0.25, 0.45, 10.25, 3.35)
    model_ax.set_xlim(0, 10.25)
    model_ax.set_ylim(0, 3.35)
    model_ax.axis("off")
    model_ax.add_patch(
        FancyBboxPatch(
            (0.02, 0.02),
            10.20,
            3.25,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor="#FBFBFB",
            edgecolor="#2F2F2F",
            linewidth=1.0,
        )
    )
    model_ax.text(0.25, 3.05, "Reduced-order dynamical system", fontsize=FONT_PANEL, ha="left", va="center", fontweight="normal")

    field_ax = add_axes_inches(fig, 0.65, 1.10, 3.55, 1.72)
    field_mesh = field_ax.pcolormesh(x, y, speed, shading="auto", cmap="viridis", vmin=0.0, vmax=speed_vmax, rasterized=True)
    field_ax.scatter(sensor_x, sensor_y, s=3.0, facecolors="none", edgecolors="white", linewidths=0.25, alpha=0.9, zorder=5)
    field_ax.set_aspect("equal", adjustable="box")
    field_ax.set_xlim(float(x.min()), float(x.max()))
    field_ax.set_ylim(float(y.min()), float(y.max()))
    field_ax.tick_params(labelsize=FONT_TICK - 1, length=2.0, width=0.55, pad=1)
    field_ax.set_xlabel(r"$x/D$", fontsize=FONT_AXIS)
    field_ax.set_ylabel(r"$y/D$", fontsize=FONT_AXIS)
    field_ax.add_patch(plt.Circle((0.0, cylinder_y), 0.5, facecolor="white", edgecolor="#202020", linewidth=0.8, zorder=6))
    model_ax.text(0.65, 0.91, r"Full PIV field $v(x,y,t)$", fontsize=FONT_TITLE, ha="left", va="center", fontweight="normal")
    cb_ax = add_axes_inches(fig, 1.05, 0.72, 2.75, 0.10)
    cb = fig.colorbar(field_mesh, cax=cb_ax, orientation="horizontal")
    cb.set_label(r"speed $|v|$ (m s$^{-1}$)", fontsize=FONT_TICK)
    cb.ax.tick_params(labelsize=FONT_TICK - 1, length=2, pad=1)

    add_box(model_ax, (4.55, 1.58), 1.25, 0.80, "POD\nr = 256", face="#EAF2F8")
    add_box(model_ax, (6.20, 1.58), 2.15, 0.80, "DMDc candidate library\n12 training regimes", face="#F6EFE3")
    add_box(model_ax, (8.75, 1.58), 1.18, 0.80, "PCE / APCE\nshadow evidence", face="#FCEBD6")
    arrow(model_ax, (4.20, 1.98), (4.52, 1.98))
    arrow(model_ax, (5.84, 1.98), (6.17, 1.98))
    arrow(model_ax, (8.38, 1.98), (8.72, 1.98))
    model_ax.text(4.55, 1.30, "field projection", fontsize=FONT_LEGEND, ha="left", va="center", color="#555555")
    model_ax.text(6.20, 1.30, r"$z_{t+1}=A_jz_t+B_j[1,y_c,\dot y_c]^\mathsf{T}+\epsilon_j$", fontsize=FONT_LEGEND, ha="left", va="center", color="#303030")
    model_ax.text(8.75, 1.30, "analysis + shadow branches", fontsize=FONT_LEGEND, ha="left", va="center", color="#555555")

    # Right top: known external displacement input, not an assimilation observation.
    disp_ax = add_axes_inches(fig, 11.20, 2.15, 6.10, 1.48)
    disp_ax.plot(time_s, displacement, color="#202020", linewidth=1.2)
    disp_ax.axvline(origin_time, color="#F28E2B", linewidth=1.0, linestyle=(0, (4, 2)))
    disp_ax.scatter([origin_time], [np.interp(origin_time, time_s, displacement)], s=25, color="#F28E2B", edgecolor="white", linewidth=0.6, zorder=4)
    disp_ax.set_title("Known cylinder displacement input", fontsize=FONT_TITLE, pad=4, fontweight="normal")
    disp_ax.set_xlabel("Time (s)", fontsize=FONT_AXIS)
    disp_ax.set_ylabel(r"$y_c/D$", fontsize=FONT_AXIS)
    disp_ax.tick_params(labelsize=FONT_TICK, length=3, width=0.7)
    disp_ax.text(0.98, 0.88, "retained during blackout", transform=disp_ax.transAxes, ha="right", va="top", fontsize=FONT_LEGEND, color="#555555")
    disp_ax.set_xlim(float(time_s.min()), float(time_s.max()))

    # Right bottom: exact sparse-observation count and coverage.
    cov_ax = add_axes_inches(fig, 11.20, 0.45, 6.10, 1.48)
    cov_ax.set_title("Sparse PIV velocity observations", fontsize=FONT_TITLE, pad=4, fontweight="normal")
    total_positions = 201 * 416
    observed_positions = 745
    unobserved_positions = total_positions - observed_positions
    observed_fraction = observed_positions / total_positions
    unobserved_fraction = unobserved_positions / total_positions
    cov_ax.barh([1], [unobserved_fraction], color="#E3E3E3", height=0.32, edgecolor="none")
    cov_ax.barh([1], [observed_fraction], color="#4E79A7", height=0.32, edgecolor="none")
    cov_ax.text(0.03, 1.22, "0.89% observed (745)", va="center", ha="left", fontsize=FONT_LEGEND, color="#4E79A7")
    cov_ax.text(0.97, 0.78, "99.11% unobserved (82,871)", va="center", ha="right", fontsize=FONT_LEGEND, color="#555555")
    cov_ax.set_xlim(0, 1.0)
    cov_ax.set_ylim(0.35, 1.55)
    cov_ax.set_yticks([])
    cov_ax.set_xlabel("Fraction of spatial grid positions (201 × 416 = 83,616)", fontsize=FONT_AXIS)
    cov_ax.set_xticks([0.0, 0.5, 1.0], ["0", "0.5", "1.0"])
    cov_ax.tick_params(axis="x", labelsize=FONT_TICK, length=3, width=0.7)
    cov_ax.spines["top"].set_visible(False)
    cov_ax.spines["right"].set_visible(False)
    cov_ax.spines["left"].set_visible(False)

    # A left-pointing connector keeps the logic of the user's draft: inputs feed the model.
    fig_ax = add_axes_inches(fig, 0.0, 0.0, FIG_W, FIG_H)
    fig_ax.set_xlim(0, FIG_W)
    fig_ax.set_ylim(0, FIG_H)
    fig_ax.axis("off")
    arrow(fig_ax, (11.05, 2.15), (10.58, 2.15), color="#4E79A7", lw=2.4)
    fig_ax.text(10.82, 2.48, "inputs", fontsize=FONT_LEGEND, ha="center", va="bottom", color="#4E79A7")

    outputs = {}
    for extension, kwargs in (("svg", {}), ("pdf", {}), ("png", {"dpi": DPI}), ("tiff", {"dpi": DPI})):
        path = OUT / f"{FIGURE}.{extension}"
        fig.savefig(path, facecolor="white", **kwargs)
        outputs[extension] = str(path)
    plt.close(fig)

    metadata = {
        "figure": FIGURE,
        "role": "standalone top observation/model diagram for Figure 5",
        "case_id": "0679",
        "reduced_velocity": 6.79,
        "source_local": str(SOURCE),
        "source_sha256": sha256_file(SOURCE),
        "remote_source_bundle": "<HILDA_RESULTS_ROOT>/results/viv_piv_pce_apce_adaptive_valid_formal5/",
        "canvas_px": [FIG_W_PX, FIG_H_PX],
        "dpi": DPI,
        "fonts": {"panel": FONT_PANEL, "title": FONT_TITLE, "legend": FONT_LEGEND, "axis": FONT_AXIS, "tick": FONT_TICK, "weight": "regular"},
        "observation_definition": {
            "known_input": "cylinder displacement y_c(t), retained during blackout",
            "piv_valid_positions": observed_positions,
            "piv_scalar_dimensions": 1490,
            "full_grid_positions": total_positions,
            "observed_fraction": observed_positions / total_positions,
            "unobserved_fraction": unobserved_positions / total_positions,
        },
        "highlighted_blackout_origin_time_s": origin_time,
        "highlighted_blackout_final_time_s": final_time,
        "outputs": outputs,
    }
    metadata_path = OUT / f"{FIGURE}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
