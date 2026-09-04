#!/usr/bin/env python3
"""Render 2-D and 3-D trajectories for the fixed real-acoustic quality window.

The window must be selected by the accompanying manifest, whose rule excludes
GPS errors. GPS is read here solely as an offline reference. The figure shows
the GPS-free global-association/triangulation frontend and a fixed, predeclared
PCE/APCE seed; it is not an author-implementation DBN reproduction.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


NODE_CENTER_M = np.array([38615217.16891715, 4337060.422857108, 22.885555555555555])
COLORS = {
    "truth": "#272727",
    "triangulation": "#0F4D92",
    "pce": "#B64342",
    "apce": "#9A4D8E",
}
MARKERS = {1: "o", 2: "s"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def numeric_xyz(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([[float(row[key]) for key in ("px", "py", "pz")] for row in rows])


def records_xyz(payload: dict) -> np.ndarray:
    return np.asarray(
        [[float(row[key]) for key in ("px", "py", "pz")] for row in payload["records"]],
        dtype=float,
    )


def select_triangulation(
    rows: list[dict[str, str]], start_frame: int, end_frame: int
) -> tuple[np.ndarray, np.ndarray]:
    chosen = [
        row
        for row in rows
        if start_frame <= int(row["frame_index"]) <= end_frame and row["valid"].lower() == "true"
    ]
    expected = end_frame - start_frame + 1
    if len(chosen) != expected:
        raise RuntimeError(f"expected {expected} valid triangulation rows, found {len(chosen)}")
    return (
        np.asarray([float(row["time_s"]) for row in chosen], dtype=float),
        numeric_xyz(chosen) - NODE_CENTER_M,
    )


def configure_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["legend.frameon"] = False


def plot_track_2d(ax, xyz: np.ndarray, *, label: str, color: str, marker: str, style: str, width: float) -> None:
    marker_every = max(1, len(xyz) // 5)
    ax.plot(
        xyz[:, 0],
        xyz[:, 1],
        color=color,
        linestyle=style,
        linewidth=width,
        marker=marker,
        markevery=marker_every,
        markersize=3.4,
        label=label,
    )


def plot_track_3d(ax, xyz: np.ndarray, *, color: str, marker: str, style: str, width: float) -> None:
    marker_every = max(1, len(xyz) // 5)
    ax.plot(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        color=color,
        linestyle=style,
        linewidth=width,
        marker=marker,
        markevery=marker_every,
        markersize=3.4,
    )


def limits(tracks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    all_xyz = np.concatenate(tracks, axis=0)
    low, high = all_xyz.min(axis=0), all_xyz.max(axis=0)
    center = 0.5 * (low + high)
    span = np.maximum(high - low, 1.0)
    # Use a common horizontal scale. The vertical range remains physical instead
    # of expanding it to horizontal extent and hiding the height evolution.
    horizontal = max(float(span[0]), float(span[1])) * 0.56
    return (
        np.array([center[0] - horizontal, center[1] - horizontal, low[2] - 0.08 * span[2] - 1.0]),
        np.array([center[0] + horizontal, center[1] + horizontal, high[2] + 0.08 * span[2] + 1.0]),
    )


def flatten_source_rows(
    target: int, times: np.ndarray, tracks: dict[str, np.ndarray], source_rows: list[dict[str, object]]
) -> None:
    for source, xyz in tracks.items():
        for index, (time_s, point) in enumerate(zip(times, xyz, strict=True)):
            source_rows.append(
                {
                    "target": target,
                    "source": source,
                    "sample_index": index,
                    "time_s": float(time_s),
                    "east_m": float(point[0]),
                    "north_m": float(point[1]),
                    "up_m": float(point[2]),
                    "gps_used_at_runtime": False,
                }
            )


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["target", "source", "sample_index", "time_s", "east_m", "north_m", "up_m", "gps_used_at_runtime"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-manifest", type=Path, required=True)
    parser.add_argument("--window-root", type=Path, required=True)
    parser.add_argument("--association-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026082502)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_style()

    manifest = read_json(args.window_manifest)
    rule = manifest["selection_rule"]
    if rule["uses_gps_error"] or rule["uses_gps_runtime"]:
        raise RuntimeError("window manifest does not satisfy GPS-free selection requirement")
    selected = manifest["selected"]
    start_frame, end_frame = int(selected["start_frame"]), int(selected["end_frame"])
    args.output.mkdir(parents=True, exist_ok=True)

    tracks_by_target: dict[int, dict[str, np.ndarray]] = {}
    times_by_target: dict[int, np.ndarray] = {}
    source_rows: list[dict[str, object]] = []
    for target in (1, 2):
        target_root = args.window_root / f"target{target}"
        truth_rows = read_csv(target_root / "frontend" / "gps_truth.csv")
        times = np.asarray([float(row["time_s"]) for row in truth_rows], dtype=float)
        truth = numeric_xyz(truth_rows)
        pce = records_xyz(read_json(target_root / "runs" / f"pce_seed_{args.seed}.json"))
        apce = records_xyz(read_json(target_root / "runs" / f"apce_seed_{args.seed}.json"))
        tri_times, triangulation = select_triangulation(
            read_csv(args.association_root / f"target{target}_triangulation_global.csv"), start_frame, end_frame
        )
        if not (len(times) == len(tri_times) == len(truth) == len(pce) == len(apce)):
            raise RuntimeError(f"trajectory length mismatch for target {target}")
        if not np.allclose(times, tri_times, rtol=0.0, atol=1e-6):
            raise RuntimeError(f"timestamp mismatch for target {target}")
        tracks = {"GPS truth (offline)": truth, "Acoustic triangulation": triangulation, "PCE": pce, "APCE": apce}
        tracks_by_target[target] = tracks
        times_by_target[target] = times
        flatten_source_rows(target, times, tracks, source_rows)

    all_tracks = [track for target_tracks in tracks_by_target.values() for track in target_tracks.values()]
    low, high = limits(all_tracks)
    fig = plt.figure(figsize=(12.2, 5.2), constrained_layout=True)
    ax2d = fig.add_subplot(1, 2, 1)
    ax3d = fig.add_subplot(1, 2, 2, projection="3d")
    method_styles = {
        "GPS truth (offline)": (COLORS["truth"], "-", 1.7),
        "Acoustic triangulation": (COLORS["triangulation"], "--", 1.25),
        "PCE": (COLORS["pce"], "-.", 1.1),
        "APCE": (COLORS["apce"], ":", 1.5),
    }
    for target in (1, 2):
        for method, xyz in tracks_by_target[target].items():
            color, style, width = method_styles[method]
            label = f"T{target} {method}"
            plot_track_2d(ax2d, xyz, label=label, color=color, marker=MARKERS[target], style=style, width=width)
            plot_track_3d(ax3d, xyz, color=color, marker=MARKERS[target], style=style, width=width)
        start = tracks_by_target[target]["GPS truth (offline)"][0]
        ax2d.annotate(f"T{target} start", (start[0], start[1]), xytext=(4, 4), textcoords="offset points", fontsize=7)

    ax2d.set_xlabel("East relative to array centre (m)")
    ax2d.set_ylabel("North relative to array centre (m)")
    ax2d.set_aspect("equal", adjustable="box")
    ax2d.set_xlim(low[0], high[0]); ax2d.set_ylim(low[1], high[1])
    ax2d.grid(color="#D9D9D9", linewidth=0.6)
    ax2d.set_title("a  Horizontal trajectories", loc="left", fontweight="bold")
    ax2d.legend(loc="upper left", bbox_to_anchor=(0.0, -0.20), ncol=2, fontsize=6.6, columnspacing=1.1, handlelength=2.2)

    ax3d.set_xlim(low[0], high[0]); ax3d.set_ylim(low[1], high[1]); ax3d.set_zlim(low[2], high[2])
    ax3d.set_box_aspect((1.0, 1.0, max((high[2] - low[2]) / (high[0] - low[0]), 0.22)))
    ax3d.view_init(elev=22, azim=-58)
    ax3d.set_xlabel("East (m)", labelpad=8)
    ax3d.set_ylabel("North (m)", labelpad=8)
    ax3d.set_zlabel("Up (m)", labelpad=8)
    ax3d.set_title("b  Three-dimensional trajectories", loc="left", fontweight="bold")
    fig.suptitle("Real-acoustic dual-source quality window: frames 924–948 (5.04 s)", x=0.5, y=1.02, fontsize=10, fontweight="bold")
    fig.text(
        0.5,
        0.005,
        "Window selected from association cost and triangulation geometry only; GPS is evaluation-only. "
        "Lines: method; markers: T1 circle, T2 square; displayed PCE/APCE seed: 2026082502.",
        ha="center",
        fontsize=7,
        color="#4D4D4D",
    )
    stem = args.output / "dual_source_quality_window_2d_3d_trajectories"
    for extension, kwargs in (("png", {"dpi": 350}), ("pdf", {}), ("svg", {})):
        fig.savefig(stem.with_suffix(f".{extension}"), bbox_inches="tight", **kwargs)
    plt.close(fig)

    save_csv(args.output / "dual_source_quality_window_trajectory_source.csv", source_rows)
    registry = {
        "figure_contract": {
            "core_conclusion": "Within the fixed real-acoustic quality window, the GPS-free dual-source triangulation is trajectory-coherent, whereas the current PCE/APCE angle reinitialization does not preserve that upstream positional accuracy.",
            "archetype": "quantitative grid",
            "backend": "Python/Matplotlib",
            "panels": {"a": "2-D East-North trajectory comparison", "b": "3-D East-North-Up trajectory comparison"},
            "review_risk": "GPS is evaluation-only and cannot be interpreted as an online observation; the selected window does not establish full-record performance or author-method reproduction.",
        },
        "selection": {
            "manifest": str(args.window_manifest),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frames": end_frame - start_frame + 1,
            "uses_gps_error": rule["uses_gps_error"],
            "uses_gps_runtime": rule["uses_gps_runtime"],
        },
        "sources": {
            "window_result_root": str(args.window_root),
            "association_root": str(args.association_root),
            "truth": "target{1,2}/frontend/gps_truth.csv (offline evaluation only)",
            "triangulation": "target{1,2}_triangulation_global.csv (GPS-free global association plus robust triangulation)",
            "pce_apce": f"target{{1,2}}/runs/{{pce,apce}}_seed_{args.seed}.json",
        },
        "coordinate_system": "centred local East-North-Up metres; global triangulation is translated by the documented array centre only",
        "representative_seed": args.seed,
        "outputs": {
            "png": str(stem.with_suffix(".png")),
            "pdf": str(stem.with_suffix(".pdf")),
            "svg": str(stem.with_suffix(".svg")),
            "source_data": str(args.output / "dual_source_quality_window_trajectory_source.csv"),
        },
    }
    (args.output / "dual_source_quality_window_trajectory_registry.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(registry, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
