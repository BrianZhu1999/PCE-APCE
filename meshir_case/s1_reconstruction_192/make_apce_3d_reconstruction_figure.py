#!/usr/bin/env python3
"""Create a Nature-style APCE 3D reconstruction image plate."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# Mandatory editable-text settings for the Python figure workflow.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update(
    {
        "font.size": 6.5,
        "axes.linewidth": 0.55,
        "axes.edgecolor": "#6D757C",
        "xtick.major.width": 0.45,
        "ytick.major.width": 0.45,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
    }
)


SHAPE = (9, 21, 21)
X = np.linspace(-0.5, 0.5, SHAPE[2])
Y = np.linspace(-0.5, 0.5, SHAPE[1])
Z = np.linspace(-0.2, 0.2, SHAPE[0])


def draw_domain_box(ax: mpl.axes.Axes) -> None:
    vertices = np.asarray(
        [
            [-0.5, -0.5, -0.2], [0.5, -0.5, -0.2], [0.5, 0.5, -0.2], [-0.5, 0.5, -0.2],
            [-0.5, -0.5, 0.2], [0.5, -0.5, 0.2], [0.5, 0.5, 0.2], [-0.5, 0.5, 0.2],
        ]
    )
    edges = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7))
    for first, second in edges:
        ax.plot(
            vertices[[first, second], 0],
            vertices[[first, second], 1],
            vertices[[first, second], 2],
            color="#AEB6BD",
            lw=0.38,
            alpha=0.72,
            zorder=1,
        )


def style_3d_axis(ax: mpl.axes.Axes, show_labels: bool) -> None:
    ax.set_xlim(-0.53, 0.53)
    ax.set_ylim(-0.53, 0.53)
    ax.set_zlim(-0.22, 0.22)
    ax.set_box_aspect((1.0, 1.0, 0.42))
    ax.view_init(elev=20, azim=-58)
    ax.set_xticks([-0.5, 0.0, 0.5])
    ax.set_yticks([-0.5, 0.0, 0.5])
    ax.set_zticks([-0.2, 0.0, 0.2])
    ax.tick_params(axis="both", which="major", labelsize=4.2, pad=-2, length=1.8)
    ax.tick_params(axis="z", which="major", labelsize=4.2, pad=-1, length=1.8)
    if show_labels:
        ax.set_xlabel("x (m)", labelpad=-5, fontsize=5.0)
        ax.set_ylabel("y (m)", labelpad=-5, fontsize=5.0)
        ax.set_zlabel("z (m)", labelpad=-5, fontsize=5.0)
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#E2E5E8")
        axis.line.set_color("#AEB6BD")


def draw_scalar_planes(ax: mpl.axes.Axes, field: np.ndarray, norm: colors.Normalize, cmap) -> None:
    z_mid, y_mid, x_mid = 4, 10, 10
    xx, yy = np.meshgrid(X, Y, indexing="xy")
    xy_z = np.full_like(xx, Z[z_mid])
    ax.plot_surface(
        xx, yy, xy_z,
        facecolors=cmap(norm(field[z_mid])),
        rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False, alpha=0.90,
    )

    xx_xz, zz_xz = np.meshgrid(X, Z, indexing="xy")
    xz_y = np.full_like(xx_xz, Y[y_mid])
    ax.plot_surface(
        xx_xz, xz_y, zz_xz,
        facecolors=cmap(norm(field[:, y_mid, :])),
        rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False, alpha=0.90,
    )

    yy_yz, zz_yz = np.meshgrid(Y, Z, indexing="xy")
    yz_x = np.full_like(yy_yz, X[x_mid])
    ax.plot_surface(
        yz_x, yy_yz, zz_yz,
        facecolors=cmap(norm(field[:, :, x_mid])),
        rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False, alpha=0.90,
    )


def draw_sparse_observations(
    ax: mpl.axes.Axes,
    field: np.ndarray,
    positions: np.ndarray,
    observed_flat: np.ndarray,
    norm: colors.Normalize,
    cmap,
) -> None:
    draw_domain_box(ax)
    values = field.reshape(-1)[observed_flat]
    points = positions[observed_flat]
    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=values, cmap=cmap, norm=norm, s=6.2,
        edgecolors="white", linewidths=0.16, alpha=0.95, depthshade=False,
    )


def add_row_band(fig: plt.Figure, y_center: float, label: str, color: str) -> None:
    fig.text(
        0.018, y_center, label,
        rotation=90, ha="center", va="center", fontsize=6.4,
        color="#27313A", fontweight="bold",
        bbox={"boxstyle": "square,pad=0.20", "facecolor": color, "edgecolor": "none", "alpha": 0.78},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with np.load(args.npz) as data:
        truth = np.asarray(data["truth"], dtype=np.float32)
        apce = np.asarray(data["mean"], dtype=np.float32)
        positions = np.asarray(data["positions"], dtype=float)
        mapping = np.asarray(data["mapping"], dtype=int)
        boundary_flat = np.asarray(data["boundary_flat"], dtype=int)
        interior_flat = np.asarray(data["interior_flat"], dtype=int)

    grid_positions = positions[mapping.reshape(-1)]
    observed_flat = np.sort(np.union1d(boundary_flat, interior_flat))
    snapshot_indices = np.asarray([256, 512, 768, 1023], dtype=int)
    if snapshot_indices[-1] >= len(truth):
        raise ValueError("requested snapshot exceeds result length")
    times_ms = snapshot_indices / 16.0

    field_values = np.concatenate([truth[snapshot_indices].ravel(), apce[snapshot_indices].ravel()])
    field_vmax = float(np.quantile(np.abs(field_values), 0.995))
    if not np.isfinite(field_vmax) or field_vmax <= 0:
        raise ValueError("invalid field colour scale")
    error = apce[snapshot_indices] - truth[snapshot_indices]
    error_vmax = float(np.quantile(np.abs(error), 0.995))
    if not np.isfinite(error_vmax) or error_vmax <= 0:
        error_vmax = field_vmax * 0.25

    field_norm = colors.Normalize(vmin=-field_vmax, vmax=field_vmax)
    error_norm = colors.TwoSlopeNorm(vmin=-error_vmax, vcenter=0.0, vmax=error_vmax)
    field_cmap = mpl.colormaps["RdBu_r"]
    error_cmap = mpl.colormaps["RdBu_r"]

    fig = plt.figure(figsize=(7.2, 6.75), facecolor="white")
    grid = fig.add_gridspec(
        4, 4,
        left=0.075, right=0.985, top=0.935, bottom=0.105,
        wspace=0.015, hspace=0.018,
    )
    axes: list[mpl.axes.Axes] = []
    row_labels = ["sparse obs.", "APCE", "reference", "signed error"]
    row_colors = ["#E8EEF4", "#FCEBDD", "#F0F1F2", "#F5E9E9"]

    for row in range(4):
        for col in range(4):
            ax = fig.add_subplot(grid[row, col], projection="3d")
            axes.append(ax)
            style_3d_axis(ax, show_labels=(row == 3 or col == 0))
            if row == 0:
                draw_sparse_observations(
                    ax, truth[snapshot_indices[col]], grid_positions, observed_flat, field_norm, field_cmap
                )
            elif row == 1:
                draw_domain_box(ax)
                draw_scalar_planes(ax, apce[snapshot_indices[col]], field_norm, field_cmap)
            elif row == 2:
                draw_domain_box(ax)
                draw_scalar_planes(ax, truth[snapshot_indices[col]], field_norm, field_cmap)
            else:
                draw_domain_box(ax)
                draw_scalar_planes(ax, apce[snapshot_indices[col]] - truth[snapshot_indices[col]], error_norm, error_cmap)

    for col, time_ms in enumerate(times_ms):
        fig.text(
            0.185 + col * 0.227, 0.952, f"{time_ms:.0f} ms",
            ha="center", va="bottom", fontsize=6.8, color="#27313A",
        )
    row_centers = [0.82, 0.615, 0.405, 0.20]
    for center, label, color in zip(row_centers, row_labels, row_colors):
        add_row_band(fig, center, label, color)

    # Row-level panel labels keep the grid citable without adding text to every cell.
    for center, label in zip([0.925, 0.72, 0.51, 0.30], "abcd"):
        fig.text(0.068, center, label, fontsize=9.5, fontweight="bold", color="#27313A", va="top")

    field_sm = ScalarMappable(norm=field_norm, cmap=field_cmap)
    field_sm.set_array([])
    error_sm = ScalarMappable(norm=error_norm, cmap=error_cmap)
    error_sm.set_array([])
    cax_field = fig.add_axes([0.28, 0.045, 0.27, 0.012])
    cax_error = fig.add_axes([0.64, 0.045, 0.22, 0.012])
    cbar_field = fig.colorbar(field_sm, cax=cax_field, orientation="horizontal")
    cbar_error = fig.colorbar(error_sm, cax=cax_error, orientation="horizontal")
    cbar_field.set_label("pressure amplitude", fontsize=5.8, labelpad=1)
    cbar_error.set_label("APCE − reference", fontsize=5.8, labelpad=1)
    cbar_field.ax.tick_params(labelsize=4.8, pad=1, length=1.5)
    cbar_error.ax.tick_params(labelsize=4.8, pad=1, length=1.5)

    fig.text(
        0.075, 0.012,
        "Central x–y, x–z and y–z planes; sparse row shows the 192 measured points.  Analysis window: 0–64 ms.",
        ha="left", va="bottom", fontsize=5.7, color="#59636B",
    )

    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / "s1_apce_3d_reconstruction_v4"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    source_data = args.output / "s1_apce_3d_reconstruction_v4_source_data.csv"
    with source_data.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["seed", "row", "time_ms", "plane", "x_m", "y_m", "z_m", "value"],
        )
        writer.writeheader()
        for col, index in enumerate(snapshot_indices):
            time_ms = float(times_ms[col])
            for row, row_data in (
                ("APCE", apce[index]),
                ("reference", truth[index]),
                ("signed_error", apce[index] - truth[index]),
            ):
                for plane, values, coords in (
                    ("xy_z0", row_data[4], [(x, y, 0.0) for y in Y for x in X]),
                    ("xz_y0", row_data[:, 10, :], [(x, 0.0, z) for z in Z for x in X]),
                    ("yz_x0", row_data[:, :, 10], [(0.0, y, z) for z in Z for y in Y]),
                ):
                    for (x, y, z), value in zip(coords, np.asarray(values).reshape(-1)):
                        writer.writerow({
                            "seed": args.seed, "row": row, "time_ms": time_ms, "plane": plane,
                            "x_m": x, "y_m": y, "z_m": z, "value": float(value),
                        })
            sparse_values = truth[index].reshape(-1)[observed_flat]
            for flat, value in zip(observed_flat, sparse_values):
                x, y, z = grid_positions[flat]
                writer.writerow({
                    "seed": args.seed, "row": "sparse_obs", "time_ms": time_ms, "plane": "points",
                    "x_m": float(x), "y_m": float(y), "z_m": float(z), "value": float(value),
                })

    registry = {
        "figure": "s1_apce_3d_reconstruction_v4",
        "backend": "Python/matplotlib",
        "archetype": "image_plate_plus_quantitative_error",
        "core_conclusion": "APCE reconstructs the three-dimensional pressure structure from 192 sparse measurements during the 0-64 ms analysis window.",
        "remote_authoritative_bundle": "<HILDA_RESULTS_ROOT>/experiments/meshir_s1_reconstruction_192_20260819/stage_d_rbf_v4_shared_noise/seed_0/apce.npz",
        "local_source_npz": str(args.npz),
        "seed": int(args.seed),
        "snapshot_indices": snapshot_indices.tolist(),
        "snapshot_times_ms": times_ms.tolist(),
        "grid_shape_zyx": list(SHAPE),
        "observed_count": int(len(observed_flat)),
        "panels": {
            "a": {"row": "sparse observations", "source": "boundary_flat + interior_flat", "role": "measurement geometry and sampled field values"},
            "b": {"row": "APCE", "source": "mean", "role": "three-dimensional reconstruction"},
            "c": {"row": "reference", "source": "truth", "role": "full-field reference"},
            "d": {"row": "signed error", "source": "mean - truth", "role": "spatial reconstruction error"},
        },
        "color_scale": {
            "field_vmax_99_5_abs": field_vmax,
            "error_vmax_99_5_abs": error_vmax,
            "field_colormap": "RdBu_r",
            "error_colormap": "RdBu_r",
        },
        "statistics": {
            "n_definition": "192 measured grid points for the sparse row; full 3969-grid field for APCE/reference/error rows",
            "replicates": "one representative paired seed for the image plate; three-seed summary remains in the formal source-data bundle",
            "metric": "visual field comparison; quantitative nRMSE and coverage are not encoded in this plate",
        },
        "outputs": [str(stem.with_suffix(s)) for s in (".svg", ".pdf", ".png", ".tiff")],
        "source_data": str(source_data),
        "image_integrity": {
            "operation": "global symmetric colour normalization within field and error families; no local contrast enhancement or selective retouching",
            "planes": "central x-y, x-z and y-z planes",
        },
    }
    (args.output / "s1_apce_3d_reconstruction_v4.provenance.json").write_text(
        json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    qa = {
        "finite_inputs": bool(np.isfinite(truth).all() and np.isfinite(apce).all()),
        "field_vmax_positive": bool(field_vmax > 0),
        "error_vmax_positive": bool(error_vmax > 0),
        "observed_count": int(len(observed_flat)),
        "snapshot_count": int(len(snapshot_indices)),
        "svg_text_editable": True,
        "pdf_fonttype": 42,
        "raster_dpi": 600,
        "manuscript_modified": False,
    }
    (args.output / "s1_apce_3d_reconstruction_v4.qa.json").write_text(
        json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"figure": str(stem), "source_data": str(source_data), "qa": qa}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
