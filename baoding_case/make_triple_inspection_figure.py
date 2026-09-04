#!/usr/bin/env python3
"""Render a provisional three-rank triangulation audit for Baoding.

The three ranks are MUSIC peak ranks, not target labels.  This figure is an
inspection artifact only and intentionally has no PCE/APCE panel or truth
comparison.  It is useful for deciding whether a future three-target
association is worth implementing.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import shuangyuan_dual_association as base

NODES = (1, 2, 3, 5, 6, 7, 8, 11, 13)
COLORS = ("#1D6F8A", "#C75A3C", "#6C5B9B")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--nod", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    candidate = audit["top_candidates"][0]
    start, stop = candidate["start_index"], candidate["stop_index_exclusive"]
    rows = {}
    for node in NODES:
        path = args.input_root / f"node{node}" / f"triple_doa_node_{node}_132614.csv"
        with path.open(encoding="utf-8", newline="") as stream:
            rows[node] = list(csv.DictReader(stream))[start:stop]
    nodes = base.parse_nod(args.nod)
    positions = {rank: [] for rank in (1, 2, 3)}
    diagnostics = []
    for index in range(stop - start):
        for rank in (1, 2, 3):
            observations = {
                node: (float(rows[node][index][f"azimuth_{rank}_deg"]), float(rows[node][index][f"zenith_{rank}_deg"]))
                for node in NODES
            }
            position, inliers, condition = base.robust_triangulate(observations, nodes)
            if position is not None:
                positions[rank].append(position)
                diagnostics.append({"frame_index": start + index, "rank": rank, "inlier_nodes": len(inliers), "condition_number": condition, "x": position[0], "y": position[1], "z": position[2]})
            else:
                positions[rank].append((float("nan"),) * 3)
    origin = np.nanmedian(np.vstack([np.asarray(values) for values in positions.values()]), axis=0)
    fig = plt.figure(figsize=(9.0, 4.7), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.35, 1.0))
    ax3 = fig.add_subplot(grid[0, 0], projection="3d")
    ax2 = fig.add_subplot(grid[0, 1])
    for rank, color in zip((1, 2, 3), COLORS):
        values = np.asarray(positions[rank], dtype=float) - origin
        valid = np.isfinite(values).all(axis=1)
        ax3.plot(values[valid, 0], values[valid, 1], values[valid, 2], color=color, lw=1.8, label=f"MUSIC rank {rank}")
        ax2.plot(values[valid, 0], values[valid, 1], color=color, lw=1.4, label=f"rank {rank}")
    ax3.set_xlabel("East offset (m)"); ax3.set_ylabel("North offset (m)"); ax3.set_zlabel("Up offset (m)")
    ax3.set_title("a  Provisional three-rank triangulation", loc="left", weight="bold")
    ax3.legend(fontsize=7, loc="best")
    ax2.set_aspect("equal", adjustable="box"); ax2.grid(color="#E2E7EB", lw=0.5)
    ax2.set_xlabel("East offset (m)"); ax2.set_ylabel("North offset (m)")
    ax2.set_title("b  Horizontal projection", loc="left", weight="bold"); ax2.legend(fontsize=7)
    fig.suptitle(
        f"Baoding sanyuan_tongxinyuan_6 | inspection only | {candidate['start_time_s']:.3f}--{candidate['end_time_s']:.3f} s\n"
        "Ranks are provisional MUSIC peaks; no target identity, GPS truth, PCE/APCE, or gate is claimed",
        x=0.01, ha="left", fontsize=9, weight="bold",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    out = {
        "claim_status": "frontend_inspection_only",
        "candidate_window": candidate,
        "origin_xyz": origin.tolist(),
        "rank_semantics": "MUSIC strength rank, not target identity",
        "triangulation": "robust three-ray audit using node geometry; GPS not used",
        "diagnostics": diagnostics,
    }
    args.output.with_suffix(".json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.output.with_name(args.output.name + "_source.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(diagnostics[0]) if diagnostics else ["frame_index"])
        writer.writeheader(); writer.writerows(diagnostics)
    print(json.dumps({"output": str(args.output), "candidate_window": candidate, "claim_status": out["claim_status"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
