#!/usr/bin/env python3
"""Plot the lowest-error APCE observation segment against GPS truth."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def select_best(source_paths: list[Path], min_frames: int) -> tuple[Path, int, list[dict], float]:
    candidates = []
    for path in source_paths:
        rows = read_rows(path)
        by_segment: dict[int, list[dict]] = {}
        for row in rows:
            by_segment.setdefault(int(row["observation_segment_id"]), []).append(row)
        for segment_id, segment_rows in by_segment.items():
            if len(segment_rows) < min_frames:
                continue
            errors = np.asarray([float(row["apce_error_m"]) for row in segment_rows], dtype=float)
            candidates.append((float(np.sqrt(np.mean(errors ** 2))), path, segment_id, segment_rows))
    if not candidates:
        raise RuntimeError(f"No observation segment has at least {min_frames} frames")
    rmse, path, segment_id, rows = min(candidates, key=lambda value: value[0])
    return path, segment_id, rows, rmse


def plot_segment(source_paths: list[Path], output: Path, min_frames: int) -> dict:
    source, segment_id, rows, rmse = select_best(source_paths, min_frames)
    truth = np.asarray([[float(row["truth_x"]), float(row["truth_y"]), float(row["truth_z"])] for row in rows])
    apce = np.asarray([[float(row["apce_x"]), float(row["apce_y"]), float(row["apce_z"])] for row in rows])
    time_s = np.asarray([float(row["time_s"]) for row in rows]) - float(rows[0]["time_s"])
    seed = int(source.stem.split("_")[-2])

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 9,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    })
    figure = plt.figure(figsize=(7.0, 5.6), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(truth[:, 0], truth[:, 1], truth[:, 2], color="#20262C", lw=2.4, label="GPS truth")
    axis.plot(apce[:, 0], apce[:, 1], apce[:, 2], color="#C75A3C", lw=1.8, label="APCE")
    axis.scatter(truth[0, 0], truth[0, 1], truth[0, 2], color="#20262C", s=28, marker="o", label="segment start")
    axis.scatter(truth[-1, 0], truth[-1, 1], truth[-1, 2], color="#20262C", s=34, marker="s", label="segment end")
    axis.set_xlabel(r"$\Delta X$ from GPS centroid (m)", labelpad=8)
    axis.set_ylabel(r"$\Delta Y$ from GPS centroid (m)", labelpad=8)
    axis.set_zlabel(r"$\Delta Z$ from GPS centroid (m)", labelpad=8)
    axis.view_init(elev=25, azim=-58)
    span = np.ptp(np.vstack([truth, apce]), axis=0)
    axis.set_box_aspect(tuple(np.maximum(span, 1.0)))
    axis.legend(loc="upper left", fontsize=8)
    axis.set_title(
        f"Lowest-error APCE observation segment\n"
        f"seed {seed}, segment {segment_id}, {len(rows)} frames, RMSE={rmse:.2f} m",
        loc="left",
        pad=14,
        weight="bold",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)
    metadata = {
        "selection": "minimum APCE 3-D position RMSE among observation segments",
        "minimum_frames": min_frames,
        "source_csv": str(source),
        "seed": seed,
        "observation_segment_id": segment_id,
        "frame_count": len(rows),
        "start_time_s": float(rows[0]["time_s"]),
        "end_time_s": float(rows[-1]["time_s"]),
        "duration_s": float(time_s[-1] - time_s[0]),
        "apce_rmse_m": rmse,
        "figure": str(output.with_suffix(".pdf")),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-glob", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-frames", type=int, default=10)
    args = parser.parse_args()
    paths = [Path(path) for path in sorted(glob.glob(args.source_glob))]
    if not paths:
        raise SystemExit("No source CSV files matched")
    print(json.dumps(plot_segment(paths, args.output, args.min_frames), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
