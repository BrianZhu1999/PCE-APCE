"""Compact Figure 5a: real sparse observations, paired branches and blackout."""
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
SOURCE = HERE.parent / "source_data" / "figure5_viv_piv_compact_source_x40y20.npz"
OUT = HERE / "outputs_x40y20"
STEM = "figure5a_viv_piv_observation_protocol_v2"
DPI = 650
FIG_W_PX, FIG_H_PX = 11532, 3600
FIG_W, FIG_H = FIG_W_PX / DPI, FIG_H_PX / DPI

FP, FT, FL, FA, FK = 22, 14, 14, 13, 11
BLACK, BLUE, ORANGE, GRAY = "#202020", "#4E79A7", "#F28E2B", "#7F8C8D"


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
        }
    )


def axes_in(fig: plt.Figure, left: float, bottom: float, width: float, height: float) -> plt.Axes:
    return fig.add_axes([left / FIG_W, bottom / FIG_H, width / FIG_W, height / FIG_H])


def rect(ax: plt.Axes, x: float, y: float, w: float, h: float, text: str, face: str, edge: str, *, ls: str = "-", fs: int = FL) -> None:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.018,rounding_size=0.028", facecolor=face, edgecolor=edge, linewidth=0.85, linestyle=ls))
    ax.text(x + w / 2, y + h / 2, text, fontsize=fs, ha="center", va="center", fontweight="normal")


def arr(ax: plt.Axes, a: tuple[float, float], b: tuple[float, float], color: str, *, ls: str = "-", rad: float = 0.0) -> None:
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color=color, linewidth=1.3, linestyle=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=4, shrinkB=4))


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    configure()
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(SOURCE, allow_pickle=False) as z:
        x, y = z["x_over_d"], z["y_over_d"]
        speed = np.asarray(z["truth_origin_speed"], float)
        sx, sy = z["sensor_x_over_d"], z["sensor_y_over_d"]
        cy = float(z["cylinder_origin_y_over_d"])
        t, yc = np.asarray(z["cylinder_time_s"], float), np.asarray(z["cylinder_displacement_over_d"], float)
        tb = float(z["origin_time_s"])

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
    cv = axes_in(fig, 0, 0, FIG_W, FIG_H)
    cv.set_xlim(0, FIG_W)
    cv.set_ylim(0, FIG_H)
    cv.axis("off")

    cv.text(0.12, 5.30, "a", fontsize=FP, ha="left", va="center", fontweight="normal")
    cv.text(0.46, 5.30, "Sparse observations and shadow-evidence assimilation", fontsize=FT, ha="left", va="center", fontweight="normal")

    # Observation geometry: the coloured field is explicitly evaluation-only.
    cv.text(0.48, 4.62, "Sparse PIV observations", fontsize=FT, ha="left", va="center")
    cv.text(0.48, 4.31, "751 / 83,616 locations (0.90%); 1,502 u/v scalars", fontsize=FL, color=BLUE, ha="left", va="center")
    cv.text(0.48, 4.04, "Reference field shown only to register geometry and evaluate predictions", fontsize=FK, color=GRAY, ha="left", va="center")
    fax = axes_in(fig, 0.48, 1.05, 4.45, 2.55)
    vmax = float(np.percentile(speed[np.isfinite(speed)], 99.5))
    mesh = fax.pcolormesh(x, y, speed, shading="auto", cmap="viridis", vmin=0, vmax=vmax, rasterized=True)
    fax.scatter(sx, sy, s=3.1, facecolors="none", edgecolors="white", linewidths=0.25, alpha=0.92, zorder=5)
    fax.add_patch(plt.Circle((0.0, cy), 0.5, facecolor="white", edgecolor=BLACK, linewidth=0.8, zorder=6))
    fax.set_aspect("equal", adjustable="box")
    fax.set_xlim(float(x.min()), float(x.max()))
    fax.set_ylim(float(y.min()), float(y.max()))
    fax.set_xlabel(r"$x/D$", fontsize=FA)
    fax.set_ylabel(r"$y/D$", fontsize=FA)
    fax.tick_params(labelsize=FK, pad=1)
    fax.set_title(r"Held-out $U_r=6.79$", fontsize=FT, pad=3, fontweight="normal")
    cax = axes_in(fig, 0.88, 0.67, 3.62, 0.10)
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cb.set_label(r"speed $|v|$ (m s$^{-1}$)", fontsize=FA)
    cb.ax.tick_params(labelsize=FK, length=2, pad=1)

    # One arrow carries only sampled u,v into the online lane.
    arr(cv, (5.03, 2.30), (5.40, 2.30), BLUE)
    cv.text(5.22, 2.50, "sampled u,v", fontsize=FK, color=BLUE, ha="center", va="bottom")

    # Offline lane: no held-out field is used here.
    cv.add_patch(FancyBboxPatch((5.38, 3.40), 6.08, 1.10, boxstyle="round,pad=0.02,rounding_size=0.04", facecolor="#F7F7F7", edgecolor="#C7C7C7", linewidth=0.8))
    cv.text(5.56, 4.27, "Offline construction: training conditions only", fontsize=FT, color=GRAY, ha="left", va="center")
    rect(cv, 5.58, 3.58, 1.32, 0.48, "Training fields", "#ECECEC", GRAY)
    rect(cv, 7.30, 3.58, 1.25, 0.48, "POD\nr = 256", "#EAF2F8", GRAY)
    rect(cv, 8.96, 3.58, 2.05, 0.48, r"DMDc candidates  $\mathcal{M}_{1:12}$", "#F6EFE3", GRAY)
    arr(cv, (6.91, 3.82), (7.28, 3.82), GRAY)
    arr(cv, (8.56, 3.82), (8.94, 3.82), GRAY)

    # Online lane: analysis and shadow roles are visually separate.
    cv.add_patch(FancyBboxPatch((5.38, 0.68), 6.08, 2.42, boxstyle="round,pad=0.02,rounding_size=0.04", facecolor="#FCFDFE", edgecolor="#AFC6DC", linewidth=0.9))
    cv.text(5.56, 2.90, "Online held-out test", fontsize=FT, color=BLUE, ha="left", va="center")
    cv.text(7.15, 2.90, r"$z_{t+1}^{(j)}=A_jz_t^{(j)}+B_jc_t+\epsilon_{j,t}$", fontsize=FL, ha="left", va="center")
    rect(cv, 5.58, 1.70, 1.42, 0.62, r"PIV $y_t$\n1,502 scalars", "#EAF2F8", BLUE)
    rect(cv, 5.58, 0.92, 1.42, 0.55, r"Input $c_t$", "#FFF2E3", ORANGE)
    rect(cv, 7.38, 1.70, 1.50, 0.62, "Analysis", "#EAF2F8", BLUE)
    rect(cv, 7.38, 0.92, 1.50, 0.55, "Shadow", "#FFF8EF", ORANGE, ls="--")
    rect(cv, 9.25, 0.92, 1.28, 0.55, "Evidence", "#FFF2E3", ORANGE)
    rect(cv, 10.78, 0.92, 0.48, 0.55, r"$w_j$", "#FFF2E3", ORANGE)
    rect(cv, 9.25, 1.70, 1.28, 0.62, "Mixture", "#EAF2F8", BLUE)
    rect(cv, 10.78, 1.70, 0.48, 0.62, r"$\hat v_t$", "#EAF2F8", BLUE)
    arr(cv, (7.02, 2.01), (7.36, 2.01), BLUE)
    arr(cv, (7.02, 1.20), (7.36, 1.20), ORANGE)
    arr(cv, (8.90, 1.20), (9.23, 1.20), ORANGE, ls="--")
    arr(cv, (10.55, 1.20), (10.76, 1.20), ORANGE)
    arr(cv, (8.90, 2.01), (9.23, 2.01), BLUE)
    arr(cv, (10.55, 2.01), (10.76, 2.01), BLUE)
    arr(cv, (11.02, 1.49), (9.89, 1.68), ORANGE, rad=0.20)
    arr(cv, (10.70, 3.55), (8.15, 2.34), GRAY, ls="--", rad=0.10)
    arr(cv, (10.70, 3.55), (8.15, 1.50), GRAY, ls="--", rad=0.16)
    cv.text(8.13, 1.58, "PIV-updated", fontsize=FK, color=BLUE, ha="center", va="bottom")
    cv.text(8.13, 0.83, "no analysis update", fontsize=FK, color=ORANGE, ha="center", va="top")
    cv.text(6.29, 0.83, r"$[1,y_c/D,\dot y_c/D]^T$", fontsize=FK, color=ORANGE, ha="center", va="top")

    # Displacement and blackout protocol use a local real time window around one t_b.
    dax = axes_in(fig, 12.05, 3.42, 5.25, 1.08)
    tr = t - tb
    keep = (tr >= -6.0) & (tr <= 4.0)
    dax.plot(tr[keep], yc[keep], color=BLACK, lw=1.15)
    dax.axvline(0, color=ORANGE, lw=1.0, ls=(0, (4, 2)))
    dax.scatter([0], [np.interp(tb, t, yc)], s=22, color=ORANGE, edgecolor="white", lw=0.5, zorder=4)
    dax.set_xlim(-6, 4)
    dax.set_xlabel(r"$t-t_b$ (s)", fontsize=FA)
    dax.set_ylabel(r"$y_c/D$", fontsize=FA)
    dax.set_title("Known cylinder displacement input", fontsize=FT, pad=3, fontweight="normal")
    dax.tick_params(labelsize=FK, pad=1)
    dax.spines["top"].set_visible(False)
    dax.spines["right"].set_visible(False)

    x0, x1 = 12.12, 17.25
    xb = x0 + 0.60 * (x1 - x0)
    cv.plot([x0, xb], [2.70, 2.70], color=BLUE, lw=2.0)
    cv.plot([xb, x1], [2.70, 2.70], color=GRAY, lw=1.4, ls=(0, (4, 3)))
    cv.text((x0 + xb) / 2, 2.86, "PIV available", fontsize=FL, color=BLUE, ha="center", va="bottom")
    cv.text((xb + x1) / 2, 2.86, "PIV unavailable", fontsize=FL, color=GRAY, ha="center", va="bottom")
    cv.plot([x0, x1], [2.18, 2.18], color=BLACK, lw=1.5)
    cv.text((x0 + x1) / 2, 2.29, r"known $y_c(t)$ retained", fontsize=FL, ha="center", va="bottom")
    cv.plot([x0, xb], [1.52, 1.52], color=BLUE, lw=2.0)
    cv.plot([xb, x1], [1.52, 1.52], color=ORANGE, lw=2.0, ls=(0, (4, 2)))
    cv.text((x0 + xb) / 2, 1.27, "assimilation", fontsize=FL, color=BLUE, ha="center", va="top")
    cv.text((xb + x1) / 2, 1.27, "conditional forecast", fontsize=FL, color=ORANGE, ha="center", va="top")
    cv.plot([xb, xb], [0.88, 3.02], color=ORANGE, lw=1.1, ls=(0, (4, 2)))
    cv.text(xb, 0.78, r"$t=t_b$", fontsize=FT, ha="center", va="top")
    cv.text(xb + 0.10, 3.05, "PIV off; weights frozen", fontsize=FK, color=ORANGE, ha="left", va="bottom")
    for horizon, xpos in zip((0.5, 1.0, 2.0, 4.0), np.linspace(xb + 0.26, x1 - 0.08, 4)):
        cv.plot(xpos, 1.52, marker="o", ms=5.2, color=ORANGE, mec="white", mew=0.5)
        cv.text(xpos, 1.73, f"{horizon:g} s", fontsize=FK, color=ORANGE, ha="center", va="bottom")
    cv.text((x0 + xb) / 2, 0.86, r"reconstruction  $\hat v(t)$", fontsize=FL, color=BLUE, ha="center", va="top")
    cv.text((xb + x1) / 2, 0.86, r"forecast  $\hat v(t_b+\tau)$", fontsize=FL, color=ORANGE, ha="center", va="top")

    outputs = {}
    for ext, kwargs in (("svg", {}), ("pdf", {}), ("png", {"dpi": DPI}), ("tiff", {"dpi": DPI})):
        path = OUT / f"{STEM}.{ext}"
        fig.savefig(path, facecolor="white", **kwargs)
        outputs[ext] = str(path)
    plt.close(fig)

    meta = {
        "figure": STEM,
        "panel": "a",
        "source": str(SOURCE),
        "source_sha256": sha(SOURCE),
        "remote_source": "<HILDA_RESULTS_ROOT>/results/viv_piv_pce_apce_adaptive_valid_x40y20_formal5/",
        "canvas_px": [FIG_W_PX, FIG_H_PX],
        "dpi": DPI,
        "font_sizes_pt": {"panel": FP, "title": FT, "legend": FL, "axis": FA, "tick": FK},
        "facts": {"case": "0679", "U_r": 6.79, "layout": "x40-y20 mask-aware", "PIV_locations": 751, "scalar_observations": 1502, "full_grid_positions": 83616, "blackout_time_s": tb},
        "integrity": {"full_reference_field_used_online": False, "test_conditions_used_for_model_fit": False, "shadow_receives_analysis_update": False, "single_blackout_time": True},
        "outputs": outputs,
    }
    (OUT / f"{STEM}_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
