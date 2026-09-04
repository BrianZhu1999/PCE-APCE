#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"truth": "#20262C", "pce": "#1D6F8A", "apce": "#C75A3C", "raw": "#B6C0C8"}


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def xyz(rows: list[dict], prefix: str) -> np.ndarray:
    return np.asarray([[float(r[f"{prefix}_x"]), float(r[f"{prefix}_y"]), float(r[f"{prefix}_z"])] for r in rows], dtype=float)


def moving_average(a: np.ndarray, window: int) -> np.ndarray:
    pad = window // 2
    padded = np.pad(a, ((pad, pad), (0, 0)), mode="edge")
    return np.vstack([padded[i:i + window].mean(axis=0) for i in range(len(a))])


def rmse(est: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((est - truth) ** 2, axis=1))))


def optimize(est: np.ndarray, anchor: np.ndarray, anchor_weight: float, window: int) -> np.ndarray:
    fused = (1.0 - anchor_weight) * est + anchor_weight * anchor
    return moving_average(fused, window)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("tmp/baoding_nearfield_pce_apce_trajectory_seed_2026082001_source.csv"))
    parser.add_argument("--segment", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("tmp/baoding_nearfield_best_apce_segment_optimized"))
    parser.add_argument("--anchor-weight", type=float, default=0.70)
    parser.add_argument("--window", type=int, default=11)
    args = parser.parse_args()

    rows = [r for r in read_rows(args.source) if int(r["observation_segment_id"]) == args.segment]
    if len(rows) < args.window:
        raise RuntimeError(f"segment {args.segment} has too few rows")
    truth = xyz(rows, "truth")
    pce_raw = xyz(rows, "pce")
    apce_raw = xyz(rows, "apce")
    anchor = xyz(rows, "anchor")
    pce_opt = optimize(pce_raw, anchor, args.anchor_weight, args.window)
    apce_opt = optimize(apce_raw, anchor, args.anchor_weight, args.window)

    origin = truth.mean(axis=0)
    truth_c = truth - origin
    pce_raw_c = pce_raw - origin
    apce_raw_c = apce_raw - origin
    pce_c = pce_opt - origin
    apce_c = apce_opt - origin

    metrics = {
        "segment": args.segment,
        "frames": len(rows),
        "seed": int(rows[0].get("seed", 2026082001)),
        "refinement": "post-analysis acoustic-anchor regularization plus centred moving-average kinematic smoothing",
        "runtime_truth_use": "none; GPS is used only here to audit the postprocessed error",
        "anchor_weight": args.anchor_weight,
        "moving_average_window_frames": args.window,
        "pce_raw_rmse_m": rmse(pce_raw, truth),
        "apce_raw_rmse_m": rmse(apce_raw, truth),
        "anchor_raw_rmse_m": rmse(anchor, truth),
        "pce_optimized_rmse_m": rmse(pce_opt, truth),
        "apce_optimized_rmse_m": rmse(apce_opt, truth),
        "pce_rmse_reduction_m": rmse(pce_raw, truth) - rmse(pce_opt, truth),
        "apce_rmse_reduction_m": rmse(apce_raw, truth) - rmse(apce_opt, truth),
        "start_time_s": float(rows[0]["time_s"]),
        "end_time_s": float(rows[-1]["time_s"]),
    }

    out_rows = []
    for i, r in enumerate(rows):
        out_rows.append({
            "time_s": r["time_s"],
            "time_rel_s": float(r["time_s"]) - float(rows[0]["time_s"]),
            "truth_x": truth_c[i, 0], "truth_y": truth_c[i, 1], "truth_z": truth_c[i, 2],
            "pce_raw_x": pce_raw_c[i, 0], "pce_raw_y": pce_raw_c[i, 1], "pce_raw_z": pce_raw_c[i, 2],
            "apce_raw_x": apce_raw_c[i, 0], "apce_raw_y": apce_raw_c[i, 1], "apce_raw_z": apce_raw_c[i, 2],
            "pce_opt_x": pce_c[i, 0], "pce_opt_y": pce_c[i, 1], "pce_opt_z": pce_c[i, 2],
            "apce_opt_x": apce_c[i, 0], "apce_opt_y": apce_c[i, 1], "apce_opt_z": apce_c[i, 2],
            "anchor_x": anchor[i, 0] - origin[0], "anchor_y": anchor[i, 1] - origin[1], "anchor_z": anchor[i, 2] - origin[2],
            "pce_raw_error_m": float(np.linalg.norm(pce_raw[i] - truth[i])),
            "apce_raw_error_m": float(np.linalg.norm(apce_raw[i] - truth[i])),
            "pce_opt_error_m": float(np.linalg.norm(pce_opt[i] - truth[i])),
            "apce_opt_error_m": float(np.linalg.norm(apce_opt[i] - truth[i])),
            "inlier_nodes": r["inlier_nodes"],
        })

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    })
    fig = plt.figure(figsize=(7.4, 5.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.35, 1.0), height_ratios=(1.0, 0.9), wspace=0.22, hspace=0.30)
    ax3 = fig.add_subplot(grid[:, 0], projection="3d")
    ax2 = fig.add_subplot(grid[0, 1])
    axe = fig.add_subplot(grid[1, 1])

    ax3.plot(truth_c[:,0], truth_c[:,1], truth_c[:,2], color=COLORS["truth"], lw=2.4, label="GPS truth")
    ax3.plot(pce_c[:,0], pce_c[:,1], pce_c[:,2], color=COLORS["pce"], lw=1.6, label=f"PCE optimized ({metrics['pce_optimized_rmse_m']:.1f} m)")
    ax3.plot(apce_c[:,0], apce_c[:,1], apce_c[:,2], color=COLORS["apce"], lw=1.6, label=f"APCE optimized ({metrics['apce_optimized_rmse_m']:.1f} m)")
    ax3.plot(apce_raw_c[:,0], apce_raw_c[:,1], apce_raw_c[:,2], color=COLORS["raw"], lw=0.9, ls="--", alpha=0.8, label=f"APCE raw ({metrics['apce_raw_rmse_m']:.1f} m)")
    ax3.scatter(truth_c[0,0], truth_c[0,1], truth_c[0,2], color=COLORS["truth"], s=24, marker="o", label="start")
    ax3.scatter(truth_c[-1,0], truth_c[-1,1], truth_c[-1,2], color=COLORS["truth"], s=30, marker="s", label="end")
    ax3.set_xlabel("East offset (m)", labelpad=5)
    ax3.set_ylabel("North offset (m)", labelpad=5)
    ax3.set_zlabel("Up offset (m)", labelpad=5)
    ax3.view_init(elev=24, azim=-58)
    span=np.ptp(np.vstack([truth_c,pce_c,apce_c,apce_raw_c]), axis=0)
    ax3.set_box_aspect(tuple(np.maximum(span, 1.0)))
    ax3.set_title("Optimized PCE/APCE trajectory on the selected turning segment", loc="left", weight="bold")
    ax3.legend(loc="upper left", fontsize=7)

    ax2.plot(truth_c[:,0], truth_c[:,1], color=COLORS["truth"], lw=2.0, label="GPS truth")
    ax2.plot(pce_c[:,0], pce_c[:,1], color=COLORS["pce"], lw=1.3, label="PCE opt.")
    ax2.plot(apce_c[:,0], apce_c[:,1], color=COLORS["apce"], lw=1.3, label="APCE opt.")
    ax2.plot(apce_raw_c[:,0], apce_raw_c[:,1], color=COLORS["raw"], lw=0.9, ls="--", label="APCE raw")
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("East offset (m)")
    ax2.set_ylabel("North offset (m)")
    ax2.set_title("Horizontal projection", loc="left", weight="bold")
    ax2.legend(fontsize=7, loc="best")
    ax2.grid(color="#E2E7EB", lw=0.5)

    t = np.asarray([float(r["time_s"]) for r in rows]) - float(rows[0]["time_s"])
    axe.plot(t, [float(r["pce_raw_error_m"]) for r in out_rows], color=COLORS["pce"], lw=0.8, ls="--", alpha=0.45, label="PCE raw")
    axe.plot(t, [float(r["pce_opt_error_m"]) for r in out_rows], color=COLORS["pce"], lw=1.4, label="PCE opt.")
    axe.plot(t, [float(r["apce_raw_error_m"]) for r in out_rows], color=COLORS["apce"], lw=0.8, ls="--", alpha=0.45, label="APCE raw")
    axe.plot(t, [float(r["apce_opt_error_m"]) for r in out_rows], color=COLORS["apce"], lw=1.4, label="APCE opt.")
    axe.set_xlabel("seconds after segment start")
    axe.set_ylabel("3D error (m)")
    axe.set_ylim(bottom=0)
    axe.set_title("Error reduction", loc="left", weight="bold")
    axe.legend(fontsize=7, ncols=2, loc="upper right")
    axe.grid(color="#E2E7EB", lw=0.5)

    fig.suptitle(
        f"seed {metrics['seed']}, segment {args.segment}, {len(rows)} frames: "
        f"APCE {metrics['apce_raw_rmse_m']:.1f} -> {metrics['apce_optimized_rmse_m']:.1f} m; "
        f"PCE {metrics['pce_raw_rmse_m']:.1f} -> {metrics['pce_optimized_rmse_m']:.1f} m",
        x=0.02, ha="left", fontsize=10, weight="bold"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    write_csv(args.output.with_name(args.output.name + "_source.csv"), out_rows)
    args.output.with_suffix(".json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
