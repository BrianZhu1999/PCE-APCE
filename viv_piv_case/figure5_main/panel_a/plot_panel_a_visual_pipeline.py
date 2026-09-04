"""Image-led Figure 5a pipeline for the x40-y20 VIV-PIV experiment."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.ticker import MaxNLocator


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREVIEW = ROOT.parent / "results_preview" / "x40y20_formal5" / "excel_source"
FIELD = PREVIEW / "field_reconstruction_frame_0200.csv"
LAYOUT = PREVIEW / "layout_points_0679.csv"
DISPLACEMENT = PREVIEW / "displacement_timeseries.csv"
TRACE = HERE / "source_x40y20_apce_trace.npz"
BLACKOUT = HERE / "blackout_x40y20_source_0679.npz"
OUT = HERE / "outputs_visual_pipeline_x40y20"
OUT.mkdir(parents=True, exist_ok=True)
STEM = "figure5a_viv_piv_visual_pipeline_x40y20"

DPI = 650
FIG_W_PX, FIG_H_PX = 10553, 2200
FIG_W, FIG_H = FIG_W_PX / DPI, FIG_H_PX / DPI
BLACK = "#202020"
BLUE = "#4C78A8"
ORANGE = "#F28E2B"
GRAY = "#8A9698"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.weight": "normal",
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "axes.linewidth": 0.75,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_field() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    with FIELD.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ny = max(int(row["y_index"]) for row in rows) + 1
    nx = max(int(row["x_index"]) for row in rows) + 1
    x = np.empty((ny, nx), dtype=float)
    y = np.empty((ny, nx), dtype=float)
    speed = np.full((ny, nx), np.nan, dtype=float)
    for row in rows:
        iy, ix = int(row["y_index"]), int(row["x_index"])
        x[iy, ix] = float(row["x_over_d"])
        y[iy, ix] = float(row["y_over_d"])
        if row["valid_fluid"].lower() == "true":
            u = float(row["reference_u_m_s"])
            v = float(row["reference_v_m_s"])
            speed[iy, ix] = np.hypot(u, v)
    return x, y, speed, float(rows[0]["time_s"])


def load_layout() -> tuple[np.ndarray, np.ndarray]:
    with LAYOUT.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return (
        np.asarray([float(row["x_over_d"]) for row in rows]),
        np.asarray([float(row["y_over_d"]) for row in rows]),
    )


def load_displacement() -> tuple[np.ndarray, np.ndarray]:
    with DISPLACEMENT.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["case_id"] == "0679"]
    return (
        np.asarray([float(row["time_s"]) for row in rows]),
        np.asarray([float(row["displacement_over_d"]) for row in rows]),
    )


def pca2(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centred = np.asarray(values, dtype=float) - np.mean(values, axis=0, keepdims=True)
    _, _, vectors = np.linalg.svd(centred, full_matrices=False)
    return centred @ vectors[:2].T, vectors[0], vectors[1]


def flow_arrow(fig: plt.Figure, start: tuple[float, float], end: tuple[float, float], color: str) -> None:
    fig.add_artist(FancyArrowPatch(
        start, end, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=13,
        linewidth=1.3, color=color, shrinkA=5, shrinkB=5,
    ))


def main() -> None:
    x, y, speed, field_time = load_field()
    sensor_x, sensor_y = load_layout()
    time_s, displacement = load_displacement()
    with np.load(TRACE, allow_pickle=False) as trace:
        latent = np.asarray(trace["latent_estimate"], dtype=float)
        trace_time = np.asarray(trace["time_s"], dtype=float)
        candidate_grid = np.asarray(trace["candidate_grid"], dtype=float)
        weights = np.asarray(trace["weights"], dtype=float)
    with np.load(BLACKOUT, allow_pickle=False) as blackout:
        blackout_time = float(blackout["origin_time_s"])
        blackout_latent = np.asarray(blackout["latent_estimate"], dtype=float)

    latent_2d, pc1, pc2 = pca2(latent)
    blackout_2d = (blackout_latent - np.mean(latent, axis=0, keepdims=True)) @ np.vstack([pc1, pc2]).T
    t_weights = trace_time[1:]
    weighted_coordinate = weights @ candidate_grid

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor="white")
    # Only the panel letter is textual; the pipeline is carried by scientific visual objects.
    fig.text(0.018, 0.945, "a", fontsize=22, fontweight="bold", va="center", color=BLACK)

    field_ax = fig.add_axes([0.055, 0.205, 0.325, 0.61])
    disp_ax = fig.add_axes([0.435, 0.575, 0.235, 0.235])
    latent_ax = fig.add_axes([0.435, 0.205, 0.235, 0.255])
    evidence_ax = fig.add_axes([0.735, 0.205, 0.225, 0.61])

    vmax = float(np.nanpercentile(speed, 99.5))
    field_ax.pcolormesh(x, y, speed, shading="auto", cmap="viridis", vmin=0.0, vmax=vmax, rasterized=True)
    field_ax.scatter(sensor_x, sensor_y, s=5.2, facecolors="none", edgecolors="white",
                     linewidths=0.35, alpha=0.96, zorder=5)
    cylinder_y = float(np.interp(field_time, time_s, displacement))
    field_ax.add_patch(Circle((0.0, cylinder_y), 0.5, facecolor="white", edgecolor=BLACK, linewidth=0.9, zorder=6))
    field_ax.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))
    field_ax.set_ylim(float(np.nanmin(y)), float(np.nanmax(y)))
    field_ax.set_aspect("equal", adjustable="box")
    field_ax.set_xlabel(r"$x/D$", fontsize=13, labelpad=1)
    field_ax.set_ylabel(r"$y/D$", fontsize=13, labelpad=1)
    field_ax.tick_params(labelsize=11, width=0.65, length=2.6, pad=1.5)
    field_ax.set_xticks([-1, 2, 5, 8])
    field_ax.set_yticks([-2, 0, 2])

    rel_t = time_s - blackout_time
    disp_ax.plot(rel_t, displacement, color=BLACK, lw=1.35, zorder=3)
    disp_ax.axvspan(0.0, 4.0, color=ORANGE, alpha=0.10, zorder=0)
    disp_ax.axvline(0.0, color=ORANGE, lw=1.25, ls="--", zorder=4)
    disp_ax.scatter([0.0], [np.interp(blackout_time, time_s, displacement)], s=22,
                    color=ORANGE, edgecolor="white", linewidth=0.55, zorder=5)
    disp_ax.set_xlim(-8.0, 6.0)
    disp_ax.set_xlabel(r"$t-t_b$", fontsize=13, labelpad=1)
    disp_ax.set_ylabel(r"$y_c/D$", fontsize=13, labelpad=1)
    disp_ax.tick_params(labelsize=11, width=0.65, length=2.6, pad=1.5)
    disp_ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    disp_ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    colour = np.linspace(0.0, 1.0, latent_2d.shape[0])
    latent_ax.plot(latent_2d[:, 0], latent_2d[:, 1], color="#C8D0D2", lw=0.8, zorder=1)
    latent_ax.scatter(latent_2d[:, 0], latent_2d[:, 1], c=colour, cmap="PuBu", s=4.5,
                      linewidths=0, zorder=2)
    latent_ax.plot(blackout_2d[:, 0], blackout_2d[:, 1], color=ORANGE, lw=1.35, zorder=4)
    latent_ax.scatter(blackout_2d[0, 0], blackout_2d[0, 1], s=28, color=ORANGE,
                      edgecolor="white", linewidth=0.6, zorder=5)
    latent_ax.set_xlabel(r"$z_1$", fontsize=13, labelpad=1)
    latent_ax.set_ylabel(r"$z_2$", fontsize=13, labelpad=1)
    latent_ax.set_xticks([])
    latent_ax.set_yticks([])

    evidence_ax.pcolormesh(t_weights - t_weights[0], candidate_grid, weights.T, shading="nearest",
                            cmap="PuBu", vmin=0.0, vmax=float(np.percentile(weights, 99.5)), rasterized=True)
    evidence_ax.plot(t_weights - t_weights[0], weighted_coordinate, color=ORANGE, lw=1.4, zorder=4)
    evidence_ax.axhline(6.79, color=BLACK, lw=0.9, ls="--", zorder=4)
    evidence_ax.axvline(blackout_time - t_weights[0], color=ORANGE, lw=1.0, ls="--", zorder=5)
    evidence_ax.set_xlim(0.0, min(100.0, float(t_weights[-1] - t_weights[0])))
    evidence_ax.set_xlabel(r"$t$", fontsize=13, labelpad=1)
    evidence_ax.set_ylabel(r"candidate $U_r$", fontsize=13, labelpad=1)
    evidence_ax.tick_params(labelsize=11, width=0.65, length=2.6, pad=1.5)
    evidence_ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    # The arrows are intentionally graphical rather than explanatory text.
    flow_arrow(fig, (0.382, 0.505), (0.426, 0.335), BLUE)
    flow_arrow(fig, (0.670, 0.335), (0.726, 0.505), ORANGE)
    flow_arrow(fig, (0.548, 0.570), (0.548, 0.470), ORANGE)

    stem = OUT / STEM
    for ext, kwargs in ((".png", {"dpi": DPI}), (".tiff", {"dpi": DPI}), (".pdf", {}), (".svg", {})):
        fig.savefig(stem.with_suffix(ext), facecolor="white", pad_inches=0, **kwargs)
    plt.close(fig)

    outputs = {ext: str(stem.with_suffix(ext)) for ext in (".png", ".tiff", ".pdf", ".svg")}
    metadata = {
        "figure": STEM,
        "panel": "a",
        "core_conclusion": "Image-led protocol view: 751 mask-aware observations, known cylinder motion, reduced latent dynamics and shadow predictive evidence support blackout forecasting in held-out VIV-PIV conditions.",
        "backend": "Python/matplotlib only",
        "canvas": {"width_px": FIG_W_PX, "height_px": FIG_H_PX, "dpi": DPI},
        "typography_pt": {"panel": 22, "axis": 13, "tick": 11},
        "observation_layout": {"x_points": 40, "y_points": 20, "nominal_points": 800, "effective_points": 751, "scalar_observations": 1502},
        "visual_sources": {
            "field": str(FIELD),
            "layout": str(LAYOUT),
            "displacement": str(DISPLACEMENT),
            "x40y20_apce_trace": str(TRACE),
            "x40y20_blackout_source": str(BLACKOUT),
            "remote_result_root": "<HILDA_RESULTS_ROOT>/results/viv_piv_pce_apce_adaptive_valid_x40y20_formal5",
        },
        "integrity": {"test_cases_used_for_model_fit": False, "shadow_receives_analysis_update": False, "weights_are_not_bayesian_posteriors": True},
        "outputs": outputs,
    }
    metadata["output_sha256"] = {ext: sha256(stem.with_suffix(ext)) for ext in outputs}
    (OUT / f"{STEM}_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
