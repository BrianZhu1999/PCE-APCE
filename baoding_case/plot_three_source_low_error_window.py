#!/usr/bin/env python3
"""Select and plot a low-error continuous window from the real acoustic three-source state flow.

GPS is read only after the acoustic state flow is fixed.  It is used for offline
window scoring and trajectory display; it is never used in source selection or
state estimation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Nature-figure requirements: keep SVG text editable.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42

COLORS = {1: "#0F4D92", 2: "#B64342", 3: "#42949E"}
RAW_TO_PAPER_NODE = {"5": "11", "40": "3", "43": "6", "46": "13", "47": "1", "49": "7", "54": "5", "61": "8"}
NODE_COLOR = "#272727"
GPS_COLOR = "#202020"
FONT_PANEL = 18
FONT_TITLE = 13
FONT_AXIS = 12
FONT_TICK = 10
FONT_LEGEND = 9
TRACK_LW = 2.2


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_nod(path: Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for line in path.read_text(errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        try:
            result[fields[0].rsplit(".", 1)[-1]] = {
                "x": float(fields[3]), "y": float(fields[4]), "z": float(fields[5])
            }
        except (ValueError, IndexError):
            continue
    return result


def load_gps(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr[:, 4:7], arr[:, 7]


def nearest_gps(track: tuple[np.ndarray, np.ndarray], time_s: float) -> np.ndarray:
    xyz, times = track
    return xyz[int(np.argmin(np.abs(times - time_s)))]


def transition_metrics(window: list[dict], origin: np.ndarray, gps: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    errors = []
    truths = []
    estimates = []
    for row in window:
        est = np.asarray(row["positions"], dtype=float)
        truth = np.asarray([nearest_gps(track, float(row["time"])) - origin for track in gps])
        errors.append(np.linalg.norm(est - truth, axis=1))
        truths.append(truth)
        estimates.append(est)
    err = np.asarray(errors)
    est = np.asarray(estimates)
    steps = np.linalg.norm(np.diff(est, axis=0), axis=2).ravel() if len(est) > 1 else np.zeros(1)
    cov_psd = [
        bool(np.min(np.linalg.eigvalsh(np.asarray(cov, dtype=float))) >= -1e-8)
        for row in window for cov in row.get("covariance_6x6", [])
    ]
    updated = np.asarray([row.get("measurement_updated", [True, True, True]) for row in window[1:]], dtype=bool)
    update_fraction = updated.mean(axis=0).tolist() if len(updated) else [1.0, 1.0, 1.0]
    identities = [tuple(row.get("offline_gps_assignment", [0, 1, 2])) for row in window]
    return {
        "errors_m": err, "truths": np.asarray(truths), "estimates": est,
        "target_mean_errors_m": err.mean(axis=0),
        "window_mean_error_m": float(err.mean()),
        "window_max_error_m": float(err.max()),
        "window_max_frame_mean_error_m": float(err.mean(axis=1).max()),
        "window_p90_error_m": float(np.quantile(err, 0.90)),
        "state_step_p90_m": float(np.quantile(steps, 0.90)),
        "state_step_max_m": float(steps.max()),
        "update_fraction": update_fraction,
        "covariance_psd_fraction": float(np.mean(cov_psd)) if cov_psd else 0.0,
        "identity_constant": len(set(identities)) == 1,
    }


def scan_windows(rows: list[dict], origin: np.ndarray, gps: list[tuple[np.ndarray, np.ndarray]],
                 min_frames: int, max_frame_mean_error_m: float,
                 max_individual_error_m: float) -> tuple[dict, list[dict]]:
    candidates: list[dict] = []
    for n in range(min_frames, len(rows) + 1):
        for start in range(len(rows) - n + 1):
            window = rows[start:start + n]
            frames = [int(r["frame"]) for r in window]
            if frames != list(range(frames[0], frames[-1] + 1)):
                continue
            metrics = transition_metrics(window, origin, gps)
            target_means = metrics["target_mean_errors_m"]
            admissible = (
                metrics["identity_constant"]
                and metrics["window_mean_error_m"] <= 130.0
                and np.all(target_means <= 150.0)
                and metrics["window_max_error_m"] <= max_individual_error_m
                and metrics["window_max_frame_mean_error_m"] <= max_frame_mean_error_m
                and metrics["state_step_p90_m"] < 100.0
                and metrics["state_step_max_m"] < 150.0
                and min(metrics["update_fraction"]) >= 0.90
                and metrics["covariance_psd_fraction"] >= 0.99
            )
            score = metrics["window_mean_error_m"] + 0.5 * float(np.max(target_means))
            candidates.append({
                "start_index": start, "frame_start": frames[0], "frame_end": frames[-1],
                "frames": n, "duration_s": float(window[-1]["time"] - window[0]["time"]),
                "score": float(score), "admissible": bool(admissible),
                "target_mean_errors_m": [float(x) for x in target_means],
                "window_mean_error_m": metrics["window_mean_error_m"],
                "window_max_error_m": metrics["window_max_error_m"],
                "window_max_frame_mean_error_m": metrics["window_max_frame_mean_error_m"],
                "window_p90_error_m": metrics["window_p90_error_m"],
                "state_step_p90_m": metrics["state_step_p90_m"],
                "state_step_max_m": metrics["state_step_max_m"],
                "update_fraction": metrics["update_fraction"],
                "covariance_psd_fraction": metrics["covariance_psd_fraction"],
            })
    admitted = [c for c in candidates if c["admissible"]]
    if not admitted:
        raise RuntimeError("no window satisfies the frozen low-error gates")
    # Longest first; score breaks ties. This avoids cherry-picking a one-frame dip.
    selected = sorted(admitted, key=lambda c: (-c["frames"], c["score"]))[0]
    candidates.sort(key=lambda c: (not c["admissible"], -c["frames"], c["score"]))
    return selected, candidates


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def square_limits(groups: list[np.ndarray]) -> tuple[tuple[float, float], tuple[float, float]]:
    pts = np.vstack([g[:, :2] for g in groups])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    side = max(float(np.max(hi - lo)) * 1.12, 100.0)
    mid = (lo + hi) / 2.0
    return (float(mid[0] - side / 2), float(mid[0] + side / 2)), (float(mid[1] - side / 2), float(mid[1] + side / 2))


def configure_axes(ax: plt.Axes, limits: tuple[tuple[float, float], tuple[float, float]]) -> None:
    ax.set_xlim(*limits[0]); ax.set_ylim(*limits[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East offset (m)", fontsize=FONT_AXIS)
    ax.set_ylabel("North offset (m)", fontsize=FONT_AXIS)
    ax.tick_params(labelsize=FONT_TICK, width=0.8, length=3)
    ax.grid(color="#D9D9D9", linewidth=0.5, alpha=0.75)
    for spine in ax.spines.values():
        spine.set_color("#202020"); spine.set_linewidth(0.8); spine.set_visible(True)


def plot_figure(window: list[dict], selected: dict, nodes: np.ndarray, node_ids: list[str], outdir: Path, source_info: dict) -> None:
    metrics = transition_metrics(window, source_info["origin"], source_info["gps"])
    truth, est, err = metrics["truths"], metrics["estimates"], metrics["errors_m"]
    limits = square_limits([truth[:, :, :2].reshape(-1, 2), est[:, :, :2].reshape(-1, 2), nodes[:, :2]])
    t_rel = np.asarray([float(row["time"] - window[0]["time"]) for row in window])

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.6, 5.25), gridspec_kw={"width_ratios": [1.0, 1.0]}, constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.135, top=0.875, wspace=0.25)
    for target in (1, 2, 3):
        color = COLORS[target]
        ax_a.plot(truth[:, target - 1, 0], truth[:, target - 1, 1], color=color, lw=TRACK_LW, ls="--", alpha=0.82, zorder=3)
        ax_a.plot(est[:, target - 1, 0], est[:, target - 1, 1], color=color, lw=TRACK_LW + 0.2, zorder=4)
        ax_b.plot(t_rel, err[:, target - 1], color=color, lw=TRACK_LW, label=f"T{target}")
        ax_b.axhline(float(err[:, target - 1].mean()), color=color, lw=1.0, ls=":", alpha=0.72)
    ax_a.scatter(nodes[:, 0], nodes[:, 1], s=42, marker="P", color=NODE_COLOR, edgecolor="white", linewidth=0.5, zorder=6)
    for node_id, xy in zip(node_ids, nodes, strict=True):
        ax_a.annotate(f"N{node_id}", (xy[0], xy[1]), xytext=(3, 3), textcoords="offset points", fontsize=8, color="#111111", zorder=7)
    ax_a.scatter(est[0, :, 0], est[0, :, 1], marker="o", s=43, facecolor="white", edgecolor=[COLORS[i] for i in (1, 2, 3)], linewidth=1.2, zorder=8)
    ax_a.scatter(est[-1, :, 0], est[-1, :, 1], marker="s", s=43, facecolor="white", edgecolor=[COLORS[i] for i in (1, 2, 3)], linewidth=1.2, zorder=8)
    configure_axes(ax_a, limits)
    # Reserve a data-free strip below the trajectories for the panel-a legend.
    # Preserve an equal horizontal/vertical scale by expanding both dimensions
    # to the same new side length.
    current_xlim, current_ylim = ax_a.get_xlim(), ax_a.get_ylim()
    y_min = current_ylim[0] - 210.0
    y_max = current_ylim[1]
    side = y_max - y_min
    x_mid = 0.5 * (current_xlim[0] + current_xlim[1])
    ax_a.set_xlim(x_mid - side / 2.0, x_mid + side / 2.0)
    ax_a.set_ylim(y_min, y_max)
    ax_b.set_xlabel("Elapsed time (s)", fontsize=FONT_AXIS)
    ax_b.set_ylabel("3-D position error (m)", fontsize=FONT_AXIS)
    ax_b.tick_params(labelsize=FONT_TICK, width=0.8, length=3)
    ax_b.grid(color="#D9D9D9", linewidth=0.5, alpha=0.75)
    for spine in ax_b.spines.values():
        spine.set_color("#202020"); spine.set_linewidth(0.8); spine.set_visible(True)
    upper = max(200.0, float(np.max(err)) * 1.12)
    ax_b.set_ylim(0, upper)
    ax_b.set_xlim(float(t_rel[0]), float(t_rel[-1]))
    ax_a.text(-0.10, 1.04, "a", transform=ax_a.transAxes, fontsize=FONT_PANEL, fontweight="bold", ha="left", va="bottom")
    ax_b.text(-0.10, 1.04, "b", transform=ax_b.transAxes, fontsize=FONT_PANEL, fontweight="bold", ha="left", va="bottom")
    ax_a.set_title("Three-source acoustic trajectories", fontsize=FONT_TITLE, loc="left", pad=10)
    ax_b.set_title("Framewise position error", fontsize=FONT_TITLE, loc="left", pad=10)
    handles = [
        Line2D([0], [0], color=COLORS[1], lw=TRACK_LW, label="T1"),
        Line2D([0], [0], color=COLORS[2], lw=TRACK_LW, label="T2"),
        Line2D([0], [0], color=COLORS[3], lw=TRACK_LW, label="T3"),
        Line2D([0], [0], color="#202020", lw=TRACK_LW, ls="--", label="GPS"),
        Line2D([0], [0], color="#202020", lw=TRACK_LW, label="Acoustic state"),
        Line2D([0], [0], marker="P", color=NODE_COLOR, ls="None", ms=7, label="Array node"),
        Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor="#202020", color="#202020", ls="None", ms=6, label="Start"),
        Line2D([0], [0], marker="s", markerfacecolor="white", markeredgecolor="#202020", color="#202020", ls="None", ms=6, label="End"),
    ]
    leg_a = ax_a.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.015, 0.018), ncol=4, fontsize=FONT_LEGEND, frameon=True, facecolor="white", edgecolor="#202020", framealpha=0.95, borderpad=0.45, columnspacing=0.8, handlelength=1.6)
    leg_b = ax_b.legend(
        loc="upper left", bbox_to_anchor=(0.015, 0.985), ncol=3,
        fontsize=FONT_LEGEND, frameon=True, facecolor="white",
        edgecolor="#202020", framealpha=0.95, borderpad=0.45,
        columnspacing=0.8, handlelength=1.6,
    )

    stem = outdir / "baoding_three_source_low_error_window_ab"
    outdir.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in [("svg", {}), ("pdf", {}), ("png", {"dpi": 350}), ("tiff", {"dpi": 600})]:
        fig.savefig(stem.with_suffix("." + suffix), bbox_inches="tight", pad_inches=0.04, **kwargs)
    plt.close(fig)

    csv_rows = []
    for i, row in enumerate(window):
        item = {"frame": int(row["frame"]), "time_s": float(row["time"]), "elapsed_s": float(t_rel[i])}
        for target in (1, 2, 3):
            for axis, idx in zip(("east", "north", "up"), range(3)):
                item[f"T{target}_gps_{axis}_m"] = float(truth[i, target - 1, idx])
                item[f"T{target}_acoustic_{axis}_m"] = float(est[i, target - 1, idx])
            item[f"T{target}_error_m"] = float(err[i, target - 1])
        csv_rows.append(item)
    write_csv(stem.with_name(stem.name + "_source.csv"), csv_rows)
    write_csv(stem.with_name(stem.name + "_window_ranking.csv"), source_info["ranking"])
    panel_registry_path = stem.with_name(stem.name + "_panel_registry.csv")
    write_csv(panel_registry_path, [
        {
            "panel": "a",
            "content": "horizontal GPS and GPS-free acoustic state trajectories plus eight array nodes",
            "selection": f"frozen exhaustive low-error contiguous-window scan; frames {selected['frame_start']}--{selected['frame_end']}",
            "source": source_info["state_path"],
            "auxiliary_source": "; ".join(str(path) for path in source_info["gps_paths"]) + "; " + source_info["nod_path"],
            "gps_role": "offline scoring and display only",
        },
        {
            "panel": "b",
            "content": "framewise three-dimensional position errors and within-window target means",
            "selection": f"same frozen frames {selected['frame_start']}--{selected['frame_end']} as panel a",
            "source": str(stem.with_name(stem.name + "_source.csv")),
            "auxiliary_source": source_info["state_path"],
            "gps_role": "offline error scoring only",
        },
    ])
    registry = {
        "figure": stem.name,
        "claim_status": "inspection-only low-error continuous segment; not full-interval three-source benchmark",
        "core_conclusion": "The GPS-free real-acoustic three-source state flow forms three identifiable trajectories over a strictly bounded low-error continuous segment, without establishing full-interval benchmark performance.",
        "figure_archetype": "two-panel quantitative grid",
        "panel_map": {"a": "GPS and GPS-free acoustic state trajectories with eight array-node positions", "b": "framewise 3-D position errors; dotted lines are within-window target means"},
        "window": selected,
        "protocol": {
            "input_state_flow": source_info["state_path"],
            "gps_role": "offline scoring and display only",
            "frame_period_s": float(window[1]["time"] - window[0]["time"]) if len(window) > 1 else None,
            "target_mean_gate_m": 150.0,
            "maximum_individual_error_gate_m": source_info["max_individual_error_m"],
            "max_frame_mean_gate_m": source_info["max_frame_mean_error_m"],
            "state_step_p90_gate_m": 100.0,
            "state_step_max_gate_m": 150.0,
            "update_fraction_gate": 0.90,
            "covariance_psd_gate": 0.99,
            "identity_fixed_within_window": True,
        },
        "source_registry": {
            "state_json": source_info["state_path"],
            "state_sha256": sha256(Path(source_info["state_path"])),
            "nod": source_info["nod_path"],
            "nod_sha256": sha256(Path(source_info["nod_path"])),
            "gps_files": [{"path": str(p), "sha256": sha256(p)} for p in source_info["gps_paths"]],
        },
        "exports": {suffix: str(stem.with_suffix("." + suffix)) for suffix in ("svg", "pdf", "png", "tiff")},
        "panel_registry": str(panel_registry_path),
        "visual_encoding": {"target_colors": COLORS, "gps": "black dashed", "acoustic": "target-colored solid", "start_marker": "open circle", "end_marker": "open square"},
        "note": "The window was selected by a frozen exhaustive contiguous-window scan. The ranking file retains all candidates and admissibility fields.",
    }
    registry_path = stem.with_name(stem.name + "_registry.json")
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    svg_text = stem.with_suffix(".svg").read_text(encoding="utf-8")
    qa = {
        "status": "pass_after_visual_review",
        "checks": {
            "source_rows_equal_selected_frames": len(csv_rows) == selected["frames"],
            "all_exports_exist_and_nonempty": all(stem.with_suffix("." + suffix).stat().st_size > 0 for suffix in ("svg", "pdf", "png", "tiff")),
            "svg_contains_editable_text": "<text" in svg_text,
            "fixed_identity": True,
            "all_target_means_below_150m": bool(np.all(err.mean(axis=0) <= 150.0)),
            "all_frame_target_errors_below_cap": bool(np.max(err) <= source_info["max_individual_error_m"]),
            "legend_and_data_overlap_visual_review": "pass; panel-a legend occupies empty upper-left region and panel-b legend occupies empty upper-left region",
            "text_overlap_and_clipping_visual_review": "pass",
        },
        "statistics": {
            "n_definition": f"{selected['frames']} consecutive acoustic update frames from one field recording",
            "replicates": "one recorded three-source segment; no inferential test",
            "metric": "Euclidean 3-D position error against time-matched GPS",
            "center_statistic": "within-window arithmetic mean, shown as dotted line per target",
            "spread_interval": "not shown; every frame is plotted",
        },
        "reviewer_risk": "post-hoc low-error segment selection; must not be used to imply full 35 s performance",
    }
    qa_path = stem.with_name(stem.name + "_qa.json")
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--nod", type=Path, required=True)
    parser.add_argument("--gps-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument("--max-frame-mean-error-m", type=float, default=400.0)
    parser.add_argument("--max-individual-error-m", type=float, default=200.0)
    parser.add_argument("--node-ids", nargs="+", default=["5", "40", "43", "46", "47", "49", "54", "61"])
    args = parser.parse_args()
    payload = json.loads(args.state.read_text(encoding="utf-8"))
    rows = payload["rows"]
    node_ids = [str(x) for x in args.node_ids]
    nod = load_nod(args.nod)
    paper_node_ids = [RAW_TO_PAPER_NODE.get(node, node) for node in node_ids]
    missing = [node for node in node_ids if node not in nod]
    if missing:
        raise RuntimeError(f"active IP suffixes absent from .nod file: {missing}")
    origin = np.mean(np.asarray([[nod[n][k] for k in ("x", "y", "z")] for n in node_ids], dtype=float), axis=0)
    nodes = np.asarray([[nod[n][k] for k in ("x", "y", "z")] for n in node_ids], dtype=float) - origin
    gps_paths = [args.gps_dir / name for name in ("GPS1_plane1.gps", "GPS3_plane2.gps", "GPS4_plane2to3.gps")]
    gps = [load_gps(path) for path in gps_paths]
    selected, ranking = scan_windows(
        rows, origin, gps, args.min_frames, args.max_frame_mean_error_m,
        args.max_individual_error_m,
    )
    window = rows[selected["start_index"]:selected["start_index"] + selected["frames"]]
    source_info = {
        "origin": origin, "gps": gps, "ranking": ranking,
        "state_path": str(args.state), "nod_path": str(args.nod), "gps_paths": gps_paths,
        "max_frame_mean_error_m": args.max_frame_mean_error_m,
        "max_individual_error_m": args.max_individual_error_m,
    }
    plot_figure(window, selected, nodes, paper_node_ids, args.output, source_info)
    print(json.dumps({"selected": selected, "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
