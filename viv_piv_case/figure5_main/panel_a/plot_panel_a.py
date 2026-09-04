"""Draw Figure 5a: sparse VIV-PIV observations and shadow-evidence protocol."""
from __future__ import annotations

import hashlib
import json
import pathlib

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


HERE = pathlib.Path(__file__).resolve().parent
SOURCE = HERE.parent / "source_data" / "figure5_viv_piv_compact_source.npz"
OUT_DIR = HERE / "outputs"
STEM = "figure5a_viv_piv_observation_protocol"

DPI = 650
FIG_W_PX = 11532
FIG_H_PX = 3250
FIG_W = FIG_W_PX / DPI
FIG_H = FIG_H_PX / DPI

FONT_PANEL = 22
FONT_TITLE = 14
FONT_LEGEND = 14
FONT_AXIS = 13
FONT_TICK = 11

BLACK = "#202020"
BLUE = "#4E79A7"
ORANGE = "#F28E2B"
GRAY = "#7F8C8D"
LIGHT_GRAY = "#F3F3F3"


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
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


def add_axes_inches(fig: plt.Figure, left: float, bottom: float, width: float, height: float) -> plt.Axes:
    return fig.add_axes([left / FIG_W, bottom / FIG_H, width / FIG_W, height / FIG_H])


def box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str = "#444444",
    linestyle: str = "-",
    fontsize: int = FONT_TITLE,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.018,rounding_size=0.035",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=0.9,
            linestyle=linestyle,
        )
    )
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize, fontweight="normal")


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = BLUE,
    linestyle: str = "-",
    linewidth: float = 1.5,
    rad: float = 0.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=4,
            shrinkB=4,
        )
    )


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    configure()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with np.load(SOURCE, allow_pickle=False) as source:
        x = np.asarray(source["x_over_d"], dtype=float)
        y = np.asarray(source["y_over_d"], dtype=float)
        speed = np.asarray(source["truth_origin_speed"], dtype=float)
        sensor_x = np.asarray(source["sensor_x_over_d"], dtype=float)
        sensor_y = np.asarray(source["sensor_y_over_d"], dtype=float)
        cylinder_y = float(source["cylinder_origin_y_over_d"])
        time_s = np.asarray(source["cylinder_time_s"], dtype=float)
        displacement = np.asarray(source["cylinder_displacement_over_d"], dtype=float)
        blackout_time = float(source["origin_time_s"])

    speed_vmax = float(np.percentile(speed[np.isfinite(speed)], 99.5))
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
    canvas = add_axes_inches(fig, 0, 0, FIG_W, FIG_H)
    canvas.set_xlim(0, FIG_W)
    canvas.set_ylim(0, FIG_H)
    canvas.axis("off")

    # One panel label and one concise title, aligned on the same baseline.
    canvas.text(0.12, 4.78, "a", fontsize=FONT_PANEL, ha="left", va="center", fontweight="normal")
    canvas.text(0.46, 4.78, "Sparse observations and shadow-evidence assimilation", fontsize=FONT_TITLE, ha="left", va="center", fontweight="normal")

    # Left: held-out reference field only for evaluation; open circles are the actual inputs.
    field_ax = add_axes_inches(fig, 0.45, 1.08, 4.35, 2.12)
    field = field_ax.pcolormesh(x, y, speed, shading="auto", cmap="viridis", vmin=0, vmax=speed_vmax, rasterized=True)
    field_ax.scatter(sensor_x, sensor_y, s=3.0, facecolors="none", edgecolors="white", linewidths=0.24, alpha=0.92, zorder=5)
    field_ax.add_patch(plt.Circle((0.0, cylinder_y), 0.5, facecolor="white", edgecolor=BLACK, linewidth=0.8, zorder=6))
    field_ax.set_aspect("equal", adjustable="box")
    field_ax.set_xlim(float(x.min()), float(x.max()))
    field_ax.set_ylim(float(y.min()), float(y.max()))
    field_ax.set_xlabel(r"$x/D$", fontsize=FONT_AXIS)
    field_ax.set_ylabel(r"$y/D$", fontsize=FONT_AXIS)
    field_ax.tick_params(labelsize=FONT_TICK, pad=1.5)
    field_ax.set_title(r"Held-out $U_r=6.79$ reference and PIV geometry", fontsize=FONT_TITLE, pad=4, fontweight="normal")
    field_ax.text(0.02, 0.96, "reference field: evaluation only", transform=field_ax.transAxes, fontsize=FONT_LEGEND, ha="left", va="top", color="white")
    field_ax.text(0.02, 0.05, "745 locations = 0.89% of the grid", transform=field_ax.transAxes, fontsize=FONT_LEGEND, ha="left", va="bottom", color="white")
    cax = add_axes_inches(fig, 0.85, 0.65, 3.55, 0.10)
    cb = fig.colorbar(field, cax=cax, orientation="horizontal")
    cb.set_label(r"speed $|v|$ (m s$^{-1}$)", fontsize=FONT_AXIS)
    cb.ax.tick_params(labelsize=FONT_TICK, length=2, pad=1)

    # A single observation arrow avoids implying that the full held-out field enters the model.
    arrow(canvas, (4.92, 2.12), (5.32, 2.12), color=BLUE, linewidth=1.8)
    canvas.text(5.10, 2.30, "sampled u,v only", fontsize=FONT_LEGEND, color=BLUE, ha="center", va="bottom")

    # Centre upper lane: candidate construction from training conditions only.
    canvas.add_patch(FancyBboxPatch((5.28, 3.05), 6.38, 1.02, boxstyle="round,pad=0.02,rounding_size=0.04", facecolor="#F7F7F7", edgecolor="#C5C5C5", linewidth=0.8))
    canvas.text(5.46, 3.92, "Offline construction (training conditions only)", fontsize=FONT_TITLE, ha="left", va="center", color="#555555")
    box(canvas, 5.48, 3.25, 1.35, 0.48, "12 training\nfull fields", facecolor="#ECECEC")
    box(canvas, 7.20, 3.25, 1.25, 0.48, "POD basis\nr = 256", facecolor="#EAF2F8")
    box(canvas, 8.82, 3.25, 2.34, 0.48, r"12 DMDc models  $\mathcal{M}_1,\ldots,\mathcal{M}_{12}$", facecolor="#F6EFE3")
    arrow(canvas, (6.84, 3.49), (7.17, 3.49), color=GRAY, linewidth=1.2)
    arrow(canvas, (8.46, 3.49), (8.79, 3.49), color=GRAY, linewidth=1.2)

    # Centre lower lane: held-out online assimilation with explicitly separated branches.
    canvas.add_patch(FancyBboxPatch((5.28, 0.55), 6.38, 2.25, boxstyle="round,pad=0.02,rounding_size=0.04", facecolor="#FCFDFE", edgecolor="#AFC6DC", linewidth=0.9))
    canvas.text(5.46, 2.64, "Online held-out test condition", fontsize=FONT_TITLE, ha="left", va="center", color=BLUE)
    box(canvas, 5.48, 1.67, 1.45, 0.58, r"Sparse PIV  $y_t$\n1,490 scalars", facecolor="#EAF2F8", edgecolor=BLUE)
    box(canvas, 5.48, 0.83, 1.45, 0.58, r"Known input  $c_t$\n$[1,y_c/D,\dot y_c/D]^T$", facecolor="#FFF2E3", edgecolor=ORANGE)
    box(canvas, 7.38, 1.67, 1.48, 0.58, "analysis branches\nPIV-updated", facecolor="#EAF2F8", edgecolor=BLUE)
    box(canvas, 7.38, 0.83, 1.48, 0.58, "shadow branches\nno analysis update", facecolor="#FFF8EF", edgecolor=ORANGE, linestyle="--")
    box(canvas, 9.30, 0.83, 1.32, 0.58, "shadow predictive\nevidence", facecolor="#FFF2E3", edgecolor=ORANGE)
    box(canvas, 10.88, 0.83, 0.58, 0.58, r"$w_j(t)$", facecolor="#FFF2E3", edgecolor=ORANGE)
    box(canvas, 9.30, 1.67, 1.32, 0.58, "weighted analysis\nmixture", facecolor="#EDF4FA", edgecolor=BLUE)
    box(canvas, 10.88, 1.67, 0.58, 0.58, r"$\hat v_t$", facecolor="#EDF4FA", edgecolor=BLUE)
    canvas.text(7.40, 2.43, r"$z_{t+1}^{(j)}=A_jz_t^{(j)}+B_jc_t+\epsilon_{j,t}$", fontsize=FONT_LEGEND, ha="left", va="center")

    arrow(canvas, (6.95, 1.96), (7.35, 1.96), color=BLUE)
    arrow(canvas, (6.95, 1.12), (7.35, 1.12), color=ORANGE)
    arrow(canvas, (8.88, 1.12), (9.27, 1.12), color=ORANGE, linestyle="--")
    arrow(canvas, (10.64, 1.12), (10.85, 1.12), color=ORANGE)
    arrow(canvas, (8.88, 1.96), (9.27, 1.96), color=BLUE)
    arrow(canvas, (10.64, 1.96), (10.85, 1.96), color=BLUE)
    arrow(canvas, (11.16, 1.44), (10.24, 1.65), color=ORANGE, rad=0.24)
    arrow(canvas, (10.93, 3.23), (8.22, 2.27), color=GRAY, linestyle="--", linewidth=1.0, rad=0.12)
    arrow(canvas, (10.93, 3.23), (8.22, 1.43), color=GRAY, linestyle="--", linewidth=1.0, rad=0.18)

    # Right: one blackout time on a local real displacement window.
    displacement_ax = add_axes_inches(fig, 12.15, 3.05, 5.10, 1.02)
    relative_time = time_s - blackout_time
    keep = (relative_time >= -6.0) & (relative_time <= 4.0)
    displacement_ax.plot(relative_time[keep], displacement[keep], color=BLACK, linewidth=1.15)
    displacement_ax.axvline(0.0, color=ORANGE, linestyle=(0, (4, 2)), linewidth=1.0)
    displacement_ax.scatter([0.0], [np.interp(blackout_time, time_s, displacement)], s=22, color=ORANGE, edgecolor="white", linewidth=0.5, zorder=4)
    displacement_ax.set_xlim(-6.0, 4.0)
    displacement_ax.set_ylabel(r"$y_c/D$", fontsize=FONT_AXIS)
    displacement_ax.set_xlabel(r"$t-t_b$ (s)", fontsize=FONT_AXIS)
    displacement_ax.set_title("Known cylinder displacement input", fontsize=FONT_TITLE, pad=3, fontweight="normal")
    displacement_ax.tick_params(labelsize=FONT_TICK, pad=1.5)
    displacement_ax.spines["top"].set_visible(False)
    displacement_ax.spines["right"].set_visible(False)

    # Protocol timeline shares the same t_b and leaves space for all four forecast horizons.
    tx0, tx1 = 12.20, 17.22
    tbx = tx0 + 0.60 * (tx1 - tx0)
    canvas.plot([tx0, tbx], [2.55, 2.55], color=BLUE, linewidth=2.0)
    canvas.plot([tbx, tx1], [2.55, 2.55], color=GRAY, linewidth=1.4, linestyle=(0, (4, 3)))
    canvas.text((tx0 + tbx) / 2, 2.70, "PIV available", fontsize=FONT_LEGEND, color=BLUE, ha="center", va="bottom")
    canvas.text((tbx + tx1) / 2, 2.70, "PIV unavailable", fontsize=FONT_LEGEND, color=GRAY, ha="center", va="bottom")
    canvas.plot([tx0, tx1], [2.00, 2.00], color=BLACK, linewidth=1.5)
    canvas.text((tx0 + tx1) / 2, 2.10, r"known $y_c(t)$ retained", fontsize=FONT_LEGEND, color=BLACK, ha="center", va="bottom")
    canvas.plot([tx0, tbx], [1.34, 1.34], color=BLUE, linewidth=2.0)
    canvas.plot([tbx, tx1], [1.34, 1.34], color=ORANGE, linewidth=2.0, linestyle=(0, (4, 2)))
    canvas.text((tx0 + tbx) / 2, 1.08, "assimilation", fontsize=FONT_LEGEND, color=BLUE, ha="center", va="top")
    canvas.text((tbx + tx1) / 2, 1.08, "conditional forecast", fontsize=FONT_LEGEND, color=ORANGE, ha="center", va="top")
    canvas.plot([tbx, tbx], [0.80, 2.85], color=ORANGE, linewidth=1.1, linestyle=(0, (4, 2)))
    canvas.text(tbx, 0.69, r"$t=t_b$", fontsize=FONT_TITLE, ha="center", va="top")
    canvas.text(tbx + 0.12, 2.88, "PIV off; weights frozen", fontsize=FONT_LEGEND, color=ORANGE, ha="left", va="bottom")
    for horizon, xpos in zip((0.5, 1.0, 2.0, 4.0), np.linspace(tbx + 0.28, tx1 - 0.10, 4)):
        canvas.plot(xpos, 1.34, marker="o", ms=5.5, color=ORANGE, markeredgecolor="white", markeredgewidth=0.5)
        canvas.text(xpos, 1.55, f"{horizon:g} s", fontsize=FONT_TICK, color=ORANGE, ha="center", va="bottom")
    canvas.text((tx0 + tbx) / 2, 0.67, r"reconstruction  $\hat v(t)$", fontsize=FONT_LEGEND, color=BLUE, ha="center", va="top")
    canvas.text((tbx + tx1) / 2, 0.67, r"forecast  $\hat v(t_b+\tau)$", fontsize=FONT_LEGEND, color=ORANGE, ha="center", va="top")

    outputs: dict[str, str] = {}
    for extension, kwargs in (
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": DPI}),
        ("tiff", {"dpi": DPI}),
    ):
        path = OUT_DIR / f"{STEM}.{extension}"
        fig.savefig(path, facecolor="white", **kwargs)
        outputs[extension] = str(path)
    plt.close(fig)

    metadata = {
        "figure": STEM,
        "panel": "a",
        "core_conclusion": "The held-out VIV test uses only sparse PIV samples and a known cylinder-displacement input; shadow forecasts provide candidate evidence while analysis branches reconstruct the field before and after observation blackout.",
        "canvas_px": [FIG_W_PX, FIG_H_PX],
        "dpi": DPI,
        "font_sizes_pt": {"panel": FONT_PANEL, "title": FONT_TITLE, "legend": FONT_LEGEND, "axis": FONT_AXIS, "tick": FONT_TICK},
        "source_local": str(SOURCE),
        "source_sha256": sha256_file(SOURCE),
        "source_remote": "<HILDA_RESULTS_ROOT>/results/viv_piv_pce_apce_adaptive_valid_formal5/",
        "representative_case": {"case_id": "0679", "reduced_velocity": 6.79, "blackout_time_s": blackout_time},
        "observation_protocol": {"valid_spatial_positions": 745, "scalar_uv_observations": 1490, "full_grid_positions": 83616, "spatial_coverage": 745 / 83616},
        "integrity_notes": {
            "reference_field_role": "evaluation visualization only; not an online model input",
            "offline_training_separated": True,
            "analysis_shadow_split_shown": True,
            "single_blackout_time_symbol": "t_b",
            "displacement_retained_during_blackout": True,
        },
        "outputs": outputs,
    }
    metadata_path = OUT_DIR / f"{STEM}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
