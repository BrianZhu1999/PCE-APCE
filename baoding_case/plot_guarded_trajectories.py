#!/usr/bin/env python3
"""Render segmented PCE/APCE trajectories without bridging low-quality gaps."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"pce": "#1D6F8A", "apce": "#C75A3C", "truth": "#20262C", "guard": "#B4363E"}


def read_records(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["records"]


def numeric(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def segment_spans(times: np.ndarray, segment_ids: np.ndarray) -> list[tuple[int, np.ndarray]]:
    spans = []
    if len(times) == 0:
        return spans
    start = 0
    for index in range(1, len(times)):
        if segment_ids[index] != segment_ids[index - 1]:
            spans.append((int(segment_ids[start]), np.arange(start, index)))
            start = index
    spans.append((int(segment_ids[start]), np.arange(start, len(times))))
    return spans


def plot_seed(seed: int, result: Path, output: Path) -> dict:
    pce = read_records(result / "runs" / f"pce_seed_{seed}.json")
    apce = read_records(result / "runs" / f"apce_seed_{seed}.json")
    if [row["time_s"] for row in pce] != [row["time_s"] for row in apce]:
        raise RuntimeError(f"PCE/APCE time mismatch for seed {seed}")

    truth_global = np.c_[numeric(pce, "truth_x"), numeric(pce, "truth_y"), numeric(pce, "truth_z")]
    pce_global = np.c_[numeric(pce, "px"), numeric(pce, "py"), numeric(pce, "pz")]
    apce_global = np.c_[numeric(apce, "px"), numeric(apce, "py"), numeric(apce, "pz")]
    anchor_global = np.c_[numeric(pce, "anchor_x"), numeric(pce, "anchor_y"), numeric(pce, "anchor_z")]
    time_s = numeric(pce, "time_s")
    segment_ids = numeric(pce, "observation_segment_id").astype(int)
    accepted = np.asarray([bool(row.get("accepted_acoustic_frame", True)) for row in pce], dtype=bool)
    relocalized = np.asarray([bool(row["relocalized"]) for row in pce]) | np.asarray([bool(row["relocalized"]) for row in apce])
    origin = truth_global.mean(axis=0)
    truth = truth_global - origin
    pce_xyz = pce_global - origin
    apce_xyz = apce_global - origin
    anchor = anchor_global - origin
    segments = segment_spans(time_s - time_s[0], segment_ids)

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.75,
        "legend.frameon": False,
    })
    figure = plt.figure(figsize=(7.2, 5.3), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 0.85])
    axis_3d = figure.add_subplot(grid[:, 0], projection="3d")
    axis_plan = figure.add_subplot(grid[0, 1])
    axis_error = figure.add_subplot(grid[1, 1])

    axis_3d.plot(truth[:, 0], truth[:, 1], truth[:, 2], color=COLORS["truth"], lw=1.8, label="GPS truth", zorder=4)
    for segment_index, (_, indices) in enumerate(segments):
        label_pce = "PCE" if segment_index == 0 else None
        label_apce = "APCE" if segment_index == 0 else None
        axis_3d.plot(pce_xyz[indices, 0], pce_xyz[indices, 1], pce_xyz[indices, 2], color=COLORS["pce"], lw=1.15, label=label_pce, zorder=3)
        axis_3d.plot(apce_xyz[indices, 0], apce_xyz[indices, 1], apce_xyz[indices, 2], color=COLORS["apce"], lw=1.15, label=label_apce, zorder=2)
        axis_3d.scatter(pce_xyz[indices[0], 0], pce_xyz[indices[0], 1], pce_xyz[indices[0], 2], marker="o", s=18, facecolors="none", edgecolors=COLORS["pce"], linewidths=0.8, zorder=5)
        axis_3d.scatter(apce_xyz[indices[0], 0], apce_xyz[indices[0], 1], apce_xyz[indices[0], 2], marker="o", s=18, facecolors="none", edgecolors=COLORS["apce"], linewidths=0.8, zorder=5)
    if relocalized.any():
        axis_3d.scatter(pce_xyz[relocalized, 0], pce_xyz[relocalized, 1], pce_xyz[relocalized, 2], marker="x", s=16, lw=0.9, color=COLORS["guard"], label="re-localized", zorder=5)
    axis_3d.set_xlabel(r"$\Delta X$ from GPS centroid (m)", labelpad=4)
    axis_3d.set_ylabel(r"$\Delta Y$ from GPS centroid (m)", labelpad=4)
    axis_3d.set_zlabel(r"$\Delta Z$ from GPS centroid (m)", labelpad=4)
    axis_3d.view_init(elev=24, azim=-58)
    span = np.ptp(np.r_[truth, pce_xyz, apce_xyz], axis=0)
    axis_3d.set_box_aspect(tuple(np.maximum(span, 1.0)))
    axis_3d.set_title("3D trajectory", loc="left", pad=7, weight="bold")
    axis_3d.legend(loc="upper left", fontsize=6)

    axis_plan.plot(truth[:, 0], truth[:, 1], color=COLORS["truth"], lw=1.5, label="GPS truth")
    for segment_index, (_, indices) in enumerate(segments):
        label_pce = "PCE" if segment_index == 0 else None
        label_apce = "APCE" if segment_index == 0 else None
        axis_plan.plot(pce_xyz[indices, 0], pce_xyz[indices, 1], color=COLORS["pce"], lw=1.0, label=label_pce)
        axis_plan.plot(apce_xyz[indices, 0], apce_xyz[indices, 1], color=COLORS["apce"], lw=1.0, label=label_apce)
        axis_plan.scatter(pce_xyz[indices[0], 0], pce_xyz[indices[0], 1], s=14, facecolors="none", edgecolors=COLORS["pce"], linewidths=0.8)
        axis_plan.scatter(apce_xyz[indices[0], 0], apce_xyz[indices[0], 1], s=14, facecolors="none", edgecolors=COLORS["apce"], linewidths=0.8)
    if relocalized.any():
        axis_plan.scatter(anchor[relocalized, 0], anchor[relocalized, 1], s=10, color=COLORS["guard"], marker="x", linewidths=0.8, label="re-localized")
    axis_plan.set_aspect("equal", adjustable="box")
    axis_plan.set_xlabel(r"$\Delta X$ from GPS centroid (m)")
    axis_plan.set_ylabel(r"$\Delta Y$ from GPS centroid (m)")
    axis_plan.set_title("Horizontal projection", loc="left", weight="bold")
    axis_plan.legend(loc="best", fontsize=6)

    pce_error = numeric(pce, "position_error_m")
    apce_error = numeric(apce, "position_error_m")
    time_rel = time_s - time_s[0]
    for segment_index, (_, indices) in enumerate(segments):
        label_p = "PCE error" if segment_index == 0 else None
        label_a = "APCE error" if segment_index == 0 else None
        axis_error.plot(time_rel[indices], pce_error[indices], color=COLORS["pce"], lw=1.0, label=label_p)
        axis_error.plot(time_rel[indices], apce_error[indices], color=COLORS["apce"], lw=1.0, label=label_a)
    for (_, left_indices), (_, right_indices) in zip(segments, segments[1:]):
        gap_start = float(time_rel[left_indices][-1])
        gap_end = float(time_rel[right_indices][0])
        if gap_end > gap_start:
            axis_error.axvspan(gap_start, gap_end, color="#D7DCE0", alpha=0.45, lw=0, zorder=0)
    if relocalized.any():
        axis_error.scatter(time_rel[relocalized], np.maximum(pce_error[relocalized], apce_error[relocalized]), color=COLORS["guard"], marker="x", s=15, lw=0.8, label="re-localized")
    axis_error.set_xlabel("seconds after evaluation start")
    axis_error.set_ylabel("3D error (m)")
    axis_error.set_title("Error and segmented recovery", loc="left", weight="bold")
    axis_error.legend(loc="upper right", fontsize=6)
    axis_error.set_ylim(bottom=0)
    figure.suptitle(f"2017 Baoding near-field held-out trajectory, seed {seed}", x=0.02, ha="left", fontsize=10, weight="bold")

    output.mkdir(parents=True, exist_ok=True)
    stem = output / f"baoding_nearfield_pce_apce_trajectory_seed_{seed}"
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)

    combined = []
    for index, (pce_row, apce_row) in enumerate(zip(pce, apce)):
        combined.append({
            "time_s": pce_row["time_s"],
            "origin_x": origin[0],
            "origin_y": origin[1],
            "origin_z": origin[2],
            "truth_x": truth[index, 0],
            "truth_y": truth[index, 1],
            "truth_z": truth[index, 2],
            "pce_x": pce_xyz[index, 0],
            "pce_y": pce_xyz[index, 1],
            "pce_z": pce_xyz[index, 2],
            "pce_error_m": pce_row["position_error_m"],
            "pce_alpha": pce_row["alpha_estimate"],
            "pce_entropy": pce_row["evidence_entropy"],
            "apce_x": apce_xyz[index, 0],
            "apce_y": apce_xyz[index, 1],
            "apce_z": apce_xyz[index, 2],
            "apce_error_m": apce_row["position_error_m"],
            "apce_alpha": apce_row["alpha_estimate"],
            "apce_entropy": apce_row["evidence_entropy"],
            "anchor_x": anchor[index, 0],
            "anchor_y": anchor[index, 1],
            "anchor_z": anchor[index, 2],
            "relocalized": bool(relocalized[index]),
            "observation_segment_id": int(segment_ids[index]),
            "accepted_acoustic_frame": bool(accepted[index]),
            "inlier_nodes": pce_row["inlier_nodes"],
        })
    source = output / f"baoding_nearfield_pce_apce_trajectory_seed_{seed}_source.csv"
    write_csv(source, combined)
    return {
        "seed": seed,
        "figure": str(stem.with_suffix(".pdf")),
        "source_data": str(source),
        "origin_global_projected_m": origin.tolist(),
        "relocalization_count": int(relocalized.sum()),
        "segment_count": len(segments),
        "accepted_frame_count": int(accepted.sum()),
        "pce_rmse_m": float(np.sqrt(np.mean(pce_error ** 2))),
        "apce_rmse_m": float(np.sqrt(np.mean(apce_error ** 2))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pce_paths = sorted((args.result / "runs").glob("pce_seed_*.json"))
    seeds = [int(path.stem.split("_")[-1]) for path in pce_paths]
    registry = [plot_seed(seed, args.result, args.output) for seed in seeds]
    (args.output / "figure_registry.json").write_text(
        json.dumps(
            {
                "claim": "Observation-gated segmented reconstruction omits low-quality intervals rather than bridging them.",
                "backend": "python/matplotlib",
                "source_result": str(args.result),
                "seeds": registry,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
