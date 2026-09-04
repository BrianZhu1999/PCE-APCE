#!/usr/bin/env python3
"""Draw the enlarged single-source row for Baoding Supplementary Data Fig. 2.

Panel a compares the fixed 67-frame horizontal trajectory against offline GPS
and shows the nine acoustic-node locations in the same local ENU frame. Panel
b shows the explicit 19-channel three-arm microphone geometry used by the
current MUSIC frontend. GPS remains an offline evaluation reference only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


START_S, END_S = 46254.0, 46320.0
FIGURE_DPI = 650
FIG_W, FIG_H = 13.875, 6.825

# Main-text Figure 4/5 typography contract.
FONT_PANEL = 22
FONT_TITLE = 14
FONT_LEGEND = 14
FONT_AXIS = 13
FONT_TICK = 11
FONT_NODE = 9.5
FONT_CHANNEL = 9.5

COLORS = {
    "GPS": "#202020",
    "Aug-EnKF": "#768487",
    "BMA": "#9B6FB6",
    "APCE": "#E77B23",
    "Array node": "#111111",
    "x-arm": "#C65B3C",
    "y-arm": "#1E7A70",
    "z-arm": "#3E6AA8",
}
CHANNEL_GROUPS = {
    "x-arm": (9, 8, 7, 1, 2, 3),
    "y-arm": (12, 11, 10, 4, 5, 6),
    "z-arm": (19, 18, 17, 13, 14, 15, 16),
}


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial", "Helvetica", "DejaVu Sans",
                "Liberation Sans", "sans-serif",
            ],
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "axes.labelsize": FONT_AXIS,
            "xtick.labelsize": FONT_TICK,
            "ytick.labelsize": FONT_TICK,
            "legend.fontsize": FONT_LEGEND,
            "axes.linewidth": 0.85,
            "xtick.major.width": 0.85,
            "ytick.major.width": 0.85,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def xyz_from_rows(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray(
        [[float(row[name]) for name in ("px", "py", "pz")] for row in rows],
        dtype=float,
    )


def load_truth(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = read_csv(path)
    times = np.asarray([float(row["time_s"]) for row in rows], dtype=float)
    values = xyz_from_rows(rows)
    order = np.argsort(times)
    return times[order], values[order]


def interpolate_truth(
    sample_times: np.ndarray, truth_times: np.ndarray, truth: np.ndarray
) -> np.ndarray:
    if sample_times[0] < truth_times[0] or sample_times[-1] > truth_times[-1]:
        raise RuntimeError("GPS truth does not cover the selected time grid")
    return np.column_stack(
        [np.interp(sample_times, truth_times, truth[:, dim]) for dim in range(3)]
    )


def load_run_median(root: Path, method: str) -> dict[str, object]:
    paths = sorted((root / "runs").glob(f"{method}_seed_*.json"))
    if len(paths) != 5:
        raise RuntimeError(f"expected five {method} runs in {root}, found {len(paths)}")

    positions: list[np.ndarray] = []
    errors: list[np.ndarray] = []
    times_ref: np.ndarray | None = None
    seeds: list[int] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = [
            row for row in payload["records"]
            if START_S <= float(row["time_s"]) <= END_S
        ]
        times = np.asarray([float(row["time_s"]) for row in records], dtype=float)
        if times_ref is None:
            times_ref = times
        elif times.shape != times_ref.shape or not np.allclose(times, times_ref):
            raise RuntimeError(f"inconsistent time grid in {path}")
        positions.append(xyz_from_rows(records))
        errors.append(
            np.asarray([float(row["position_error_m"]) for row in records], dtype=float)
        )
        seeds.append(int(payload["seed"]))

    assert times_ref is not None
    return {
        "times": times_ref,
        "position": np.median(np.asarray(positions), axis=0),
        "errors": np.asarray(errors),
        "paths": paths,
        "seeds": seeds,
    }


def load_nodes(path: Path) -> tuple[list[dict[str, str]], np.ndarray]:
    rows = read_csv(path)
    points = np.asarray(
        [
            [float(row["local_E_m"]), float(row["local_N_m"]), float(row["local_U_m"])]
            for row in rows
        ],
        dtype=float,
    )
    return rows, points


def array_geometry(spacing: float = 0.50) -> dict[str, list[tuple[int, float, float, float]]]:
    horizontal = [-3 * spacing, -2 * spacing, -spacing, spacing, 2 * spacing, 3 * spacing]
    vertical = [-2.13 * spacing, -1.53 * spacing, -0.93 * spacing, 0.0, spacing, 2 * spacing, 3 * spacing]
    positions: dict[str, list[tuple[int, float, float, float]]] = {}
    for name, channels in CHANNEL_GROUPS.items():
        if name == "x-arm":
            positions[name] = [
                (channel, coordinate, 0.0, 0.0)
                for channel, coordinate in zip(channels, horizontal, strict=True)
            ]
        elif name == "y-arm":
            positions[name] = [
                (channel, 0.0, coordinate, 0.0)
                for channel, coordinate in zip(channels, horizontal, strict=True)
            ]
        else:
            positions[name] = [
                (channel, 0.0, 0.0, coordinate)
                for channel, coordinate in zip(channels, vertical, strict=True)
            ]
    return positions


def add_header(fig: plt.Figure, left: float, letter: str, title: str) -> None:
    baseline = 0.925
    fig.text(
        left, baseline, letter, fontsize=FONT_PANEL, fontweight="bold",
        color="#111111", ha="left", va="bottom",
    )
    fig.text(
        left + 0.043, baseline + 0.003, title, fontsize=FONT_TITLE,
        fontweight="normal", color="#111111", ha="left", va="bottom",
    )


def plot_track(
    ax: plt.Axes, data: np.ndarray, method: str, zorder: int
) -> None:
    styles = {
        "GPS": ("--", 1.55, None, None),
        "Aug-EnKF": ("-", 1.15, None, None),
        "BMA": ("-", 1.15, None, None),
        "APCE": ("-", 1.50, "s", 7),
    }
    linestyle, width, marker, markevery = styles[method]
    ax.plot(
        data[:, 0], data[:, 1], color=COLORS[method], linestyle=linestyle,
        linewidth=width, marker=marker, markevery=markevery, markersize=3.8,
        markerfacecolor="white", markeredgewidth=0.75, zorder=zorder, label=method,
    )


def set_xy_limits(ax: plt.Axes, tracks: list[np.ndarray], nodes: np.ndarray) -> None:
    xy = np.concatenate([track[:, :2] for track in tracks] + [nodes[:, :2]], axis=0)
    centre = (xy.min(axis=0) + xy.max(axis=0)) / 2.0
    span = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1.0)
    half = 0.58 * span
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_aspect("equal", adjustable="box")


def draw_array(ax: plt.Axes) -> None:
    positions = array_geometry()
    for group, rows in positions.items():
        xyz = np.asarray([[row[1], row[2], row[3]] for row in rows], dtype=float)
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=COLORS[group], lw=2.0, alpha=0.98)
        ax.scatter(
            xyz[:, 0], xyz[:, 1], xyz[:, 2], color=COLORS[group],
            s=62, depthshade=False, edgecolor="white", linewidth=0.55,
        )
        for channel, x, y, z in rows:
            if group == "z-arm":
                offset = (0.045, 0.035, 0.0)
            elif group == "x-arm":
                offset = (0.025, -0.08, 0.055)
            else:
                offset = (-0.095, 0.015, 0.055)
            ax.text(
                x + offset[0], y + offset[1], z + offset[2], str(channel),
                fontsize=FONT_CHANNEL, color=COLORS[group], ha="left", va="bottom",
            )

    ax.scatter([0], [0], [0], marker="+", color="#111111", s=78, linewidth=1.2)

    ax.set_xlim(-1.95, 1.95)
    ax.set_ylim(-1.95, 1.95)
    ax.set_zlim(-1.30, 1.95)
    ax.set_box_aspect((1.0, 1.0, 0.90))
    ax.set_proj_type("ortho")
    ax.view_init(elev=25, azim=-52)
    ax.set_xlabel("x (m)", labelpad=2)
    ax.set_ylabel("y (m)", labelpad=2)
    ax.set_zlabel("z (m)", labelpad=1)
    ax.tick_params(axis="both", labelsize=FONT_TICK, pad=0, width=0.75)
    ax.zaxis.set_tick_params(labelsize=FONT_TICK, pad=0, width=0.75)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor("#B5B5B5")
        axis._axinfo["grid"].update(
            color=(0.84, 0.84, 0.84, 0.70), linewidth=0.55
        )

    arm_handles = [
        Line2D([0], [0], color=COLORS["x-arm"], lw=2.0, marker="o", ms=5, label="x arm (6)"),
        Line2D([0], [0], color=COLORS["y-arm"], lw=2.0, marker="o", ms=5, label="y arm (6)"),
        Line2D([0], [0], color=COLORS["z-arm"], lw=2.0, marker="o", ms=5, label="z arm (7)"),
    ]
    ax.legend(
        handles=arm_handles, loc="upper left", bbox_to_anchor=(0.00, 0.94),
        fontsize=FONT_TICK, handlelength=1.45, borderaxespad=0.0, labelspacing=0.35,
    )


def pooled_rmse(errors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(errors))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--apce", type=Path, required=True)
    parser.add_argument("--nodes-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()

    node_rows, nodes = load_nodes(args.nodes_csv)
    truth_times, truth_all = load_truth(args.frontend / "gps_truth.csv")
    runs = {
        "Aug-EnKF": load_run_median(args.baselines, "aug_enkf"),
        "BMA": load_run_median(args.baselines, "bma"),
        "APCE": load_run_median(args.apce, "apce"),
    }
    times = np.asarray(runs["APCE"]["times"], dtype=float)
    if len(times) != 67:
        raise RuntimeError(f"expected 67 one-second frames, found {len(times)}")
    for method, run in runs.items():
        run_times = np.asarray(run["times"], dtype=float)
        if run_times.shape != times.shape or not np.allclose(run_times, times):
            raise RuntimeError(f"{method} time grid differs from APCE")

    truth = interpolate_truth(times, truth_times, truth_all)
    tracks = {"GPS": truth, **{name: np.asarray(run["position"]) for name, run in runs.items()}}

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
    grid = fig.add_gridspec(
        1, 2, left=0.065, right=0.985, bottom=0.095, top=0.890,
        wspace=0.12, width_ratios=(1.08, 1.0),
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1], projection="3d")

    for method, data in tracks.items():
        zorder = 7 if method == "APCE" else 4
        plot_track(ax_a, data, method, zorder)
    ax_a.scatter(
        nodes[:, 0], nodes[:, 1], s=39, marker="P", color=COLORS["Array node"],
        edgecolor="white", linewidth=0.5, zorder=9,
    )
    for row, point in zip(node_rows, nodes, strict=True):
        ax_a.annotate(
            f"N{row['node_id']}", (point[0], point[1]), xytext=(4, 4),
            textcoords="offset points", fontsize=FONT_NODE, color="#111111",
            ha="left", va="bottom", zorder=10,
        )
    set_xy_limits(ax_a, list(tracks.values()), nodes)
    ax_a.set_xlabel("East (m)")
    ax_a.set_ylabel("North (m)")
    ax_a.grid(color="#E1E1E1", linewidth=0.5, alpha=0.78, zorder=0)
    for spine in ax_a.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.85)
        spine.set_color("#111111")

    draw_array(ax_b)

    add_header(fig, 0.035, "a", "Single-source horizontal trajectory")
    add_header(fig, 0.555, "b", "Nineteen-microphone array")

    handles = [
        Line2D([0], [0], color=COLORS["GPS"], ls="--", lw=1.55, label="GPS"),
        Line2D([0], [0], color=COLORS["Aug-EnKF"], lw=1.15, label="Aug-EnKF"),
        Line2D([0], [0], color=COLORS["BMA"], lw=1.15, label="BMA"),
        Line2D([0], [0], color=COLORS["APCE"], lw=1.50, marker="s", ms=4.0,
               markerfacecolor="white", label="APCE"),
        Line2D([0], [0], marker="P", color=COLORS["Array node"], ls="None",
               ms=5.0, label="Array node"),
    ]
    ax_a.legend(
        handles=handles, loc="center", bbox_to_anchor=(0.34, 0.55),
        ncol=2, fontsize=FONT_LEGEND, handlelength=1.35, columnspacing=0.95,
        handletextpad=0.45, borderaxespad=0.0, frameon=False, labelspacing=0.55,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / "supplementary_data_figure2_baoding_single_source_ab"
    outputs: dict[str, str] = {}
    settings = (
        ("png", {"dpi": FIGURE_DPI}),
        ("pdf", {}),
        ("svg", {}),
        ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
    )
    for extension, kwargs in settings:
        path = stem.with_suffix(f".{extension}")
        fig.savefig(path, facecolor="white", **kwargs)
        outputs[extension] = str(path)
    plt.close(fig)

    source_rows: list[dict[str, object]] = []
    for method, data in tracks.items():
        for index, (time_s, point) in enumerate(zip(times, data, strict=True)):
            source_rows.append(
                {
                    "panel": "a", "method": method, "sample_index": index,
                    "time_s": float(time_s), "east_m": float(point[0]),
                    "north_m": float(point[1]), "up_m": float(point[2]),
                    "gps_used_at_runtime": False,
                }
            )
    source_path = args.output / "supplementary_data_figure2_baoding_single_source_ab_source.csv"
    with source_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)

    panel_registry_path = args.output / "supplementary_data_figure2_baoding_single_source_ab_panel_registry.csv"
    panel_rows = [
        {
            "panel": "a",
            "content": "single-source horizontal trajectories and nine acoustic nodes",
            "selection": "fixed 67-frame window 46254--46320 s; five-seed median estimates",
            "sources": f"{args.frontend}; {args.baselines}; {args.apce}; {args.nodes_csv}",
        },
        {
            "panel": "b",
            "content": "19-channel three-arm microphone geometry",
            "selection": "6 + 6 + 7 frontend geometry; no source-direction overlay",
            "sources": str(args.nodes_csv),
        },
    ]
    with panel_registry_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(panel_rows[0]))
        writer.writeheader()
        writer.writerows(panel_rows)

    registry = {
        "figure_contract": {
            "core_conclusion": "The 67-frame single-source result preserves the horizontal flight geometry while exposing the nine-node and 19-microphone acoustic geometry.",
            "panels": {
                "a": "single-source horizontal trajectory and node layout",
                "b": "current 19-channel frontend geometry",
            },
            "backend": "Python/matplotlib on Super-Server",
            "sentence_annotations": False,
        },
        "typography": {
            "panel_label_pt": FONT_PANEL, "panel_title_pt": FONT_TITLE,
            "axis_label_pt": FONT_AXIS, "tick_pt": FONT_TICK,
            "legend_pt": FONT_LEGEND, "only_panel_letters_bold": True,
        },
        "window": {
            "start_time_s": START_S, "end_time_s": END_S, "frames": len(times),
            "update_interval_s": float(np.median(np.diff(times))),
            "selection_status": "fixed before this layout revision",
        },
        "configuration": {
            "q_min_accel_mps2": 2.0, "q_max_accel_mps2": 12.0,
            "observation_covariance_scale": 1.0, "turn_rate_radps": -0.10,
            "ensemble_members": 48, "seeds": runs["APCE"]["seeds"],
        },
        "metrics_pooled_five_seed_position_rmse_m": {
            method: pooled_rmse(np.asarray(run["errors"])) for method, run in runs.items()
        },
        "array": {
            "nodes": 9, "microphones_per_node": 19,
            "geometry": "three orthogonal arms, 6 + 6 + 7 microphones",
            "coordinate_source": str(args.nodes_csv),
            "microphone_geometry_status": "current MUSIC-frontend legacy nonuniform-z candidate",
            "source_arrow": None,
        },
        "gps_role": "offline scoring and display only; GPS is not an assimilation input",
        "sources": {
            "frontend": str(args.frontend), "baselines": str(args.baselines),
            "apce": str(args.apce),
            "nodes_csv": str(args.nodes_csv),
        },
        "outputs": outputs, "source_csv": str(source_path),
        "panel_registry": str(panel_registry_path),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    registry_path = args.output / "supplementary_data_figure2_baoding_single_source_ab_registry.json"
    registry_path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({**registry, "registry": str(registry_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
