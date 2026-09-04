#!/usr/bin/env python3
"""Draw the compact Baoding Supplementary Data Figure 2 (a--d).

Panels a/c are horizontal trajectory comparisons. Panels b/d show the
explicit 19-microphone three-arm manifold used by the current MUSIC frontend,
with one or two source-direction arrows. GPS is an offline reference only.
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


START_SINGLE, END_SINGLE = 46254.0, 46320.0
FIGURE_DPI = 650
FIG_W, FIG_H = 9.25, 6.55

# Figure 4 master typography inherited by Figure 5.
FONT_PANEL = 22
FONT_TITLE = 14
FONT_LEGEND = 14
FONT_AXIS = 13
FONT_TICK = 11

COLORS = {
    "GPS": "#202020",
    "Acoustic tri.": "#1E7A70",
    "Aug-EnKF": "#7F8C8D",
    "BMA": "#A77BBE",
    "PCE": "#4C78A8",
    "APCE": "#F28E2B",
    "Array nodes": "#111111",
    "x-arm": "#C65B3C",
    "y-arm": "#1E7A70",
    "z-arm": "#3E6AA8",
}
NODE_CENTER = np.array([38615217.16891715, 4337060.422857108, 22.885555555555555])
CHANNEL_GROUPS = {
    "x-arm": (9, 8, 7, 1, 2, 3),
    "y-arm": (12, 11, 10, 4, 5, 6),
    "z-arm": (19, 18, 17, 13, 14, 15, 16),
}


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "axes.labelsize": FONT_AXIS,
            "xtick.labelsize": FONT_TICK,
            "ytick.labelsize": FONT_TICK,
            "legend.fontsize": FONT_LEGEND,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
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


def xyz_from_rows(rows: list[dict[str, str]], names=("px", "py", "pz")) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in names] for row in rows], dtype=float)


def load_truth(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = read_csv(path)
    times = np.asarray([float(row["time_s"]) for row in rows], dtype=float)
    values = xyz_from_rows(rows)
    order = np.argsort(times)
    return times[order], values[order]


def interp_truth(times: np.ndarray, truth_times: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(times, truth_times, truth[:, dim]) for dim in range(3)])


def load_run_median(root: Path, method: str, start: float | None = None, end: float | None = None):
    paths = sorted((root / "runs").glob(f"{method}_seed_*.json"))
    if len(paths) != 5:
        raise RuntimeError(f"expected five {method} runs in {root}, found {len(paths)}")
    values: list[np.ndarray] = []
    errors: list[np.ndarray] = []
    widths: list[np.ndarray] = []
    times_ref: np.ndarray | None = None
    seeds: list[int] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload["records"]
        if start is not None:
            records = [row for row in records if start <= float(row["time_s"]) <= float(end)]
        times = np.asarray([float(row["time_s"]) for row in records], dtype=float)
        if times_ref is None:
            times_ref = times
        elif times.shape != times_ref.shape or not np.allclose(times, times_ref):
            raise RuntimeError(f"inconsistent time grid in {path}")
        values.append(xyz_from_rows(records))
        errors.append(np.asarray([float(row["position_error_m"]) for row in records], dtype=float))
        widths.append(np.asarray([float(row.get("interval_width_m", np.nan)) for row in records], dtype=float))
        seeds.append(int(payload["seed"]))
    assert times_ref is not None
    return {
        "times": times_ref,
        "position": np.median(np.asarray(values), axis=0),
        "errors": np.asarray(errors),
        "width": np.median(np.asarray(widths), axis=0),
        "paths": paths,
        "seeds": seeds,
    }


def load_nodes(path: Path) -> np.ndarray:
    rows = read_csv(path)
    return np.asarray([[float(row["local_E_m"]), float(row["local_N_m"]), float(row["local_U_m"])] for row in rows], dtype=float)


def array_geometry(spacing: float = 0.50) -> dict[str, list[tuple[int, float, float, float]]]:
    horizontal = [-3 * spacing, -2 * spacing, -spacing, spacing, 2 * spacing, 3 * spacing]
    vertical = [-2.13 * spacing, -1.53 * spacing, -0.93 * spacing, 0.0, spacing, 2 * spacing, 3 * spacing]
    positions: dict[str, list[tuple[int, float, float, float]]] = {}
    for name, channels in CHANNEL_GROUPS.items():
        if name == "x-arm":
            positions[name] = [(channel, coordinate, 0.0, 0.0) for channel, coordinate in zip(channels, horizontal)]
        elif name == "y-arm":
            positions[name] = [(channel, 0.0, coordinate, 0.0) for channel, coordinate in zip(channels, horizontal)]
        else:
            positions[name] = [(channel, 0.0, 0.0, coordinate) for channel, coordinate in zip(channels, vertical)]
    return positions


def add_header(fig: plt.Figure, left: float, bottom: float, letter: str, title: str) -> None:
    fig.text(left, bottom, letter, fontsize=FONT_PANEL, fontweight="bold", color="#111111", ha="left", va="bottom")
    fig.text(left + 0.032, bottom + 0.002, title, fontsize=FONT_TITLE, fontweight="normal", color="#111111", ha="left", va="bottom")


def draw_nodes(ax: plt.Axes, nodes: np.ndarray) -> None:
    ax.scatter(nodes[:, 0], nodes[:, 1], s=25, marker="P", color=COLORS["Array nodes"], edgecolor="white", linewidth=0.45, zorder=8)


def set_xy_limits(ax: plt.Axes, tracks: list[np.ndarray], nodes: np.ndarray) -> None:
    xy = np.concatenate([track[:, :2] for track in tracks] + [nodes[:, :2]], axis=0)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    span = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1.0)
    centre = (lo + hi) / 2.0
    pad = 0.08 * span
    half = 0.5 * span + pad
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_aspect("equal", adjustable="box")


def plot_track(ax: plt.Axes, data: np.ndarray, method: str, marker: str, zorder: int = 4) -> None:
    step = max(1, len(data) // 12)
    styles = {
        "GPS": ("--", 1.35), "Acoustic tri.": (":", 1.15), "Aug-EnKF": ("-", 1.05),
        "BMA": ("-", 1.05), "PCE": ("-", 1.20), "APCE": ("-", 1.30),
    }
    linestyle, width = styles[method]
    ax.plot(data[:, 0], data[:, 1], color=COLORS[method], linestyle=linestyle, linewidth=width,
            marker=marker, markevery=step, markersize=3.0, markerfacecolor="white",
            markeredgewidth=0.65, zorder=zorder, label=method)


def draw_array(ax, source_vectors: list[np.ndarray], source_labels: list[str], source_colors: list[str]) -> None:
    positions = array_geometry()
    for group, rows in positions.items():
        xyz = np.asarray([[row[1], row[2], row[3]] for row in rows], dtype=float)
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=COLORS[group], lw=1.0, alpha=0.9)
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=COLORS[group], s=25, depthshade=False)
        for channel, x, y, z in rows:
            ax.text(x, y, z, str(channel), fontsize=9.0, color=COLORS[group], ha="left", va="bottom")
    ax.scatter([0], [0], [0], marker="+", color="#111111", s=45, linewidth=0.9)
    for vector, label, color in zip(source_vectors, source_labels, source_colors):
        direction = np.asarray(vector[:2], dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            continue
        direction = direction / norm * 2.25
        arrow_z = 1.78
        ax.quiver(0, 0, arrow_z, direction[0], direction[1], 0.0, color=color, linewidth=1.5, arrow_length_ratio=0.16)
        ax.text(direction[0] * 1.05, direction[1] * 1.05, arrow_z + 0.08, label, color="#111111", fontsize=FONT_TICK, ha="center", va="bottom")
    ax.set_xlim(-1.9, 1.9); ax.set_ylim(-1.9, 1.9); ax.set_zlim(-1.35, 2.25)
    ax.set_box_aspect((1.0, 1.0, 0.82))
    ax.view_init(elev=22, azim=35)
    ax.set_xlabel("x (m)", labelpad=1); ax.set_ylabel("y (m)", labelpad=1); ax.set_zlabel("z (m)", labelpad=1)
    ax.tick_params(axis="both", labelsize=FONT_TICK, pad=0, width=0.65)
    ax.zaxis.set_tick_params(labelsize=FONT_TICK, pad=0, width=0.65)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor("#B5B5B5")
        axis._axinfo["grid"].update(color=(0.84, 0.84, 0.84, 0.70), linewidth=0.5)


def load_dual(root: Path, source_csv: Path):
    source_rows = read_csv(source_csv)
    tracks: dict[int, dict[str, np.ndarray]] = {}
    for target in (1, 2):
        target_rows = [row for row in source_rows if int(row["target"]) == target]
        times = np.asarray([float(row["time_s"]) for row in target_rows if row["source"] == "GPS truth (offline)"], dtype=float)
        truth = np.asarray([[float(row[key]) for key in ("east_m", "north_m", "up_m")] for row in target_rows if row["source"] == "GPS truth (offline)"], dtype=float)
        tri = np.asarray([[float(row[key]) for key in ("east_m", "north_m", "up_m")] for row in target_rows if row["source"] == "Acoustic triangulation"], dtype=float)
        pce = load_run_median(root / f"target{target}", "pce")["position"]
        apce = load_run_median(root / f"target{target}", "apce")["position"]
        if not (len(times) == len(tri) == len(pce) == len(apce) == 25):
            raise RuntimeError(f"dual target {target} trajectory lengths are inconsistent")
        tracks[target] = {"times": times, "GPS": truth, "Acoustic tri.": tri, "PCE": pce, "APCE": apce}
    return tracks


def pooled_rmse(errors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(errors))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-frontend", type=Path, required=True)
    parser.add_argument("--single-baselines", type=Path, required=True)
    parser.add_argument("--single-pce", type=Path, required=True)
    parser.add_argument("--single-apce", type=Path, required=True)
    parser.add_argument("--dual-root", type=Path, required=True)
    parser.add_argument("--dual-source-csv", type=Path, required=True)
    parser.add_argument("--nodes-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()

    nodes = load_nodes(args.nodes_csv)
    single_truth_times, single_truth_all = load_truth(args.single_frontend / "gps_truth.csv")
    single_pce = load_run_median(args.single_pce, "pce", START_SINGLE, END_SINGLE)
    single_apce = load_run_median(args.single_apce, "apce", START_SINGLE, END_SINGLE)
    single_aug = load_run_median(args.single_baselines, "aug_enkf", START_SINGLE, END_SINGLE)
    single_bma = load_run_median(args.single_baselines, "bma", START_SINGLE, END_SINGLE)
    single_times = single_pce["times"]
    single_truth = interp_truth(single_times, single_truth_times, single_truth_all)
    single_tracks = {
        "GPS": single_truth, "Aug-EnKF": single_aug["position"], "BMA": single_bma["position"],
        "PCE": single_pce["position"], "APCE": single_apce["position"],
    }

    dual_tracks = load_dual(args.dual_root, args.dual_source_csv)
    dual_all = [track[method] for target in dual_tracks.values() for method in ("GPS", "Acoustic tri.", "PCE", "APCE") for track in [target]]

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
    grid = fig.add_gridspec(2, 2, left=0.070, right=0.985, bottom=0.165, top=0.925, wspace=0.20, hspace=0.47, width_ratios=(1.0, 1.0), height_ratios=(1.0, 1.0))
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1], projection="3d")
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1], projection="3d")

    marker_single = "o"
    for method, data in single_tracks.items():
        plot_track(ax_a, data, method, marker_single, zorder=4 if method == "APCE" else 3)
    draw_nodes(ax_a, nodes)
    for node, point in zip(read_csv(args.nodes_csv), nodes):
        ax_a.annotate(f"N{node['node_id']}", (point[0], point[1]), xytext=(3, 3), textcoords="offset points", fontsize=7.0, color="#111111")
    set_xy_limits(ax_a, list(single_tracks.values()), nodes)
    ax_a.set_xlabel("East (m)"); ax_a.set_ylabel("North (m)")
    ax_a.grid(color="#E1E1E1", linewidth=0.45, alpha=0.75)

    single_source = np.mean(single_truth, axis=0)
    draw_array(ax_b, [single_source], ["S1"], [COLORS["APCE"]])

    target_colors = {1: "#202020", 2: "#202020"}
    for target, target_data in dual_tracks.items():
        marker = "o" if target == 1 else "s"
        for method in ("GPS", "Acoustic tri.", "PCE", "APCE"):
            plot_track(ax_c, target_data[method], method, marker, zorder=5 if method == "APCE" else 3)
    draw_nodes(ax_c, nodes)
    for node, point in zip(read_csv(args.nodes_csv), nodes):
        ax_c.annotate(f"N{node['node_id']}", (point[0], point[1]), xytext=(3, 3), textcoords="offset points", fontsize=7.0, color="#111111")
    set_xy_limits(ax_c, dual_all, nodes)
    ax_c.set_xlabel("East (m)"); ax_c.set_ylabel("North (m)")
    ax_c.grid(color="#E1E1E1", linewidth=0.45, alpha=0.75)

    dual_sources = [np.mean(dual_tracks[target]["GPS"], axis=0) for target in (1, 2)]
    draw_array(ax_d, dual_sources, ["T1", "T2"], [COLORS["PCE"], COLORS["APCE"]])

    add_header(fig, 0.075, 0.935, "a", "Single-source trajectory")
    add_header(fig, 0.535, 0.935, "b", "Single-source array")
    add_header(fig, 0.075, 0.500, "c", "Dual-source trajectories")
    add_header(fig, 0.535, 0.500, "d", "Dual-source array")

    line_handles = [
        Line2D([0], [0], color=COLORS["GPS"], ls="--", lw=1.35, label="GPS"),
        Line2D([0], [0], color=COLORS["Acoustic tri."], ls=":", lw=1.15, label="Acoustic tri."),
        Line2D([0], [0], color=COLORS["Aug-EnKF"], lw=1.05, label="Aug-EnKF"),
        Line2D([0], [0], color=COLORS["BMA"], lw=1.05, label="BMA"),
        Line2D([0], [0], color=COLORS["PCE"], lw=1.2, label="PCE"),
        Line2D([0], [0], color=COLORS["APCE"], lw=1.3, label="APCE"),
        Line2D([0], [0], marker="o", color="#111111", ls="None", ms=4, label="T1 / single"),
        Line2D([0], [0], marker="s", color="#111111", ls="None", ms=4, label="T2"),
        Line2D([0], [0], marker="P", color="#111111", ls="None", ms=4, label="Array node"),
    ]
    fig.legend(handles=line_handles, loc="lower center", bbox_to_anchor=(0.5, 0.022), ncol=5, fontsize=FONT_LEGEND, handlelength=1.45, columnspacing=0.85, labelspacing=0.45, borderaxespad=0.0)

    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / "supplementary_data_figure2_baoding_abcd"
    outputs = {}
    for ext, kwargs in (("png", {"dpi": FIGURE_DPI}), ("pdf", {}), ("svg", {}), ("tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}})):
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, facecolor="white", **kwargs)
        outputs[ext] = str(path)
    plt.close(fig)

    source_rows = []
    for panel, tracks, target_values in (("a", single_tracks, {"target": "single"}),):
        for method, data in tracks.items():
            for idx, (time_s, point) in enumerate(zip(single_times, data, strict=True)):
                source_rows.append({"panel": panel, "target": target_values["target"], "method": method, "sample_index": idx, "time_s": float(time_s), "east_m": float(point[0]), "north_m": float(point[1]), "up_m": float(point[2]), "gps_used_at_runtime": False})
    for target, tracks in dual_tracks.items():
        for method, data in tracks.items():
            for idx, (time_s, point) in enumerate(zip(tracks["times"], data, strict=True)) if method != "times" else []:
                source_rows.append({"panel": "c", "target": f"T{target}", "method": method, "sample_index": idx, "time_s": float(time_s), "east_m": float(point[0]), "north_m": float(point[1]), "up_m": float(point[2]), "gps_used_at_runtime": False})
    source_path = args.output / "supplementary_data_figure2_baoding_abcd_source.csv"
    with source_path.open("w", encoding="utf-8-sig", newline="") as stream:
        fields = list(source_rows[0])
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(source_rows)

    panel_registry_path = args.output / "supplementary_data_figure2_baoding_abcd_panel_registry.csv"
    panel_rows = [
        {"panel": "a", "content": "single-source horizontal trajectories", "selection": "fixed 67-frame window 46254--46320 s", "sources": f"{args.single_frontend}; {args.single_baselines}; {args.single_pce}; {args.single_apce}"},
        {"panel": "b", "content": "single-source 19-microphone array manifold and one source-direction arrow", "selection": "same node and microphone geometry as panel a", "sources": str(args.nodes_csv)},
        {"panel": "c", "content": "dual-source horizontal trajectories", "selection": "GPS-free acoustic-quality-selected 25-frame window", "sources": f"{args.dual_root}; {args.dual_source_csv}"},
        {"panel": "d", "content": "dual-source 19-microphone array manifold and two source-direction arrows", "selection": "same node and microphone geometry as panel c", "sources": str(args.nodes_csv)},
    ]
    with panel_registry_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(panel_rows[0])); writer.writeheader(); writer.writerows(panel_rows)

    registry = {
        "figure_contract": {
            "core_conclusion": "The compact supplement contrasts a single-source 67-s frame with a GPS-free-selected dual-source acoustic window and makes the shared nine-node/19-microphone geometry explicit.",
            "panels": {"a": "single-source 2D trajectory", "b": "single-source array", "c": "dual-source 2D trajectories", "d": "dual-source array"},
            "backend": "Python/matplotlib only", "sentence_annotations": False,
        },
        "typography": {"panel_label_pt": FONT_PANEL, "panel_title_pt": FONT_TITLE, "axis_label_pt": FONT_AXIS, "tick_pt": FONT_TICK, "legend_pt": FONT_LEGEND, "only_panel_letters_bold": True},
        "single_source": {
            "window": {"start_time_s": START_SINGLE, "end_time_s": END_SINGLE, "frames": len(single_times), "update_interval_s": float(np.median(np.diff(single_times)))},
            "configuration": {"q_min_accel_mps2": 2.0, "q_max_accel_mps2": 12.0, "observation_covariance_scale": 1.0, "turn_rate_radps": -0.10, "ensemble_members": 48, "seeds": single_pce["seeds"]},
            "metrics_pooled_rmse_m": {"Aug-EnKF": pooled_rmse(single_aug["errors"]), "BMA": pooled_rmse(single_bma["errors"]), "PCE": pooled_rmse(single_pce["errors"]), "APCE": pooled_rmse(single_apce["errors"])},
        },
        "dual_source": {"window_frames": 25, "window_selection": "acoustic-quality selection; GPS error and GPS runtime excluded", "seeds": [2026082500, 2026082501, 2026082502, 2026082503, 2026082504], "targets": 2, "nodes": 9, "pce_apce_display": "five-seed median trajectories", "array_geometry": "three orthogonal arms, 6 + 6 + 7 microphones; x/y nominal 0.5 m, retained nonuniform z-arm convention"},
        "array": {"node_coordinates_csv": str(args.nodes_csv), "array_origin": "shared local ENU array centre", "microphone_geometry_source": "current MUSIC frontend explicit legacy nonuniform-z candidate"},
        "gps_role": "offline scoring/reference only; not used at runtime, for window selection, or for array placement",
        "outputs": outputs, "source_csv": str(source_path), "panel_registry": str(panel_registry_path),
        "script": str(Path(__file__).resolve()), "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    registry_path = args.output / "supplementary_data_figure2_baoding_abcd_registry.json"
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**registry, "registry": str(registry_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
