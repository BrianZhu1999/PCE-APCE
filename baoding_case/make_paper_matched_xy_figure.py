#!/usr/bin/env python3
"""Render a paper-matched 2-D x-y trajectory figure for the Baoding bridge.

The reference field figure uses x-y coordinates in metres, GPS truth, two
tracker outputs, array-node markers, and start/end markers. This renderer
follows that visual grammar while preserving the DBN-derived observation
provenance of the PCE/APCE bridge.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


PAPER_NODES = (1, 3, 5, 6, 7, 8, 11, 13)
# Fig. 15 assigns colour to the physical target and line/marker semantics to
# the tracker.  Keep that grammar instead of assigning colour to the method.
TARGET_COLORS = {1: "#1F77B4", 2: "#FF7F0E", 3: "#2CA02C"}
METHOD_STYLE = {
    "truth": {"linestyle": "-", "marker": None, "linewidth": 1.7},
    "pce": {"linestyle": "--", "marker": "o", "linewidth": 1.05},
    "apce": {"linestyle": "-.", "marker": "D", "linewidth": 1.05},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260822)
    ap.add_argument("--remote-source-root", required=True)
    args = ap.parse_args()

    targets = sorted(
        int(path.name.replace("target", ""))
        for path in args.result_root.iterdir()
        if path.is_dir() and path.name.startswith("target")
    )
    if targets != [1, 2, 3]:
        raise RuntimeError(f"expected target1/2/3, found {targets}")

    fig, ax = plt.subplots(figsize=(7.0, 6.2), constrained_layout=True)
    source_registry = []
    all_xy = []
    for target in targets:
        root = args.result_root / f"target{target}"
        color = TARGET_COLORS[target]
        truth_rows = read_csv(root / "frontend" / "gps_truth.csv")
        truth = np.asarray([[float(r["px"]), float(r["py"])] for r in truth_rows])
        all_xy.append(truth)
        truth_style = METHOD_STYLE["truth"]
        ax.plot(
            truth[:, 0],
            truth[:, 1],
            color=color,
            lw=truth_style["linewidth"],
            linestyle=truth_style["linestyle"],
            zorder=2,
        )
        # Fig. 15 uses one start square and one end star per physical target.
        ax.scatter(truth[0, 0], truth[0, 1], marker="s", s=33, color=color, edgecolor="black", linewidth=0.35, zorder=7)
        ax.scatter(truth[-1, 0], truth[-1, 1], marker="*", s=48, color=color, edgecolor="black", linewidth=0.35, zorder=7)
        source_registry.append(
            {
                "target": target,
                "truth_csv": str(root / "frontend" / "gps_truth.csv"),
                "truth_sha256": sha256(root / "frontend" / "gps_truth.csv"),
            }
        )
        for method in ("pce", "apce"):
            run_path = root / "runs" / f"{method}_seed_{args.seed}.json"
            payload = read_json(run_path)
            rows = [r for r in payload["records"] if r.get("position_error_m") is not None]
            xy = np.asarray([[float(r["px"]), float(r["py"])] for r in rows])
            all_xy.append(xy)
            style = METHOD_STYLE[method]
            ax.plot(
                xy[:, 0],
                xy[:, 1],
                color=color,
                lw=style["linewidth"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                ms=2.3,
                alpha=0.95,
                markevery=max(1, len(xy) // 10),
                zorder=3 if method == "pce" else 4,
            )
            source_registry.append(
                {
                    "target": target,
                    "method": method,
                    "run_json": str(run_path),
                    "run_sha256": sha256(run_path),
                    "seed": args.seed,
                }
            )

    nod_path = args.result_root / "nod_8paper.csv"
    if nod_path.is_file():
        nodes = read_csv(nod_path)
        for row in nodes:
            ax.scatter(float(row["x"]), float(row["y"]), marker="^", s=28, color="#D62728", zorder=6)
            ax.text(float(row["x"]) + 2, float(row["y"]) + 2, f"IP{row['ip']}", fontsize=6, color="#8B1A1A")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Three-target tracking trajectories (8-node paper protocol)", fontsize=10.5)
    ax.grid(True, ls="--", lw=0.45, color="#AAB2B8", alpha=0.7)
    ax.set_aspect("equal", adjustable="box")
    # Split the legend into target colours and method semantics, as in Fig. 15.
    target_handles = [
        Line2D([0], [0], color=TARGET_COLORS[target], lw=2.0, label=f"Target {target}")
        for target in targets
    ]
    method_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            lw=1.3,
            linestyle=METHOD_STYLE[method]["linestyle"],
            marker=METHOD_STYLE[method]["marker"],
            markersize=4,
            label={"truth": "GPS truth", "pce": "PCE", "apce": "APCE"}[method],
        )
        for method in ("truth", "pce", "apce")
    ]
    node_handle = Line2D([0], [0], marker="^", color="w", markerfacecolor="#D62728", markersize=6, label="Array nodes")
    start_handle = Line2D([0], [0], marker="s", color="black", linestyle="None", markersize=5, label="Start")
    end_handle = Line2D([0], [0], marker="*", color="black", linestyle="None", markersize=7, label="End")
    legend1 = ax.legend(handles=target_handles, loc="upper left", fontsize=7, frameon=True, title="Target", title_fontsize=7)
    ax.add_artist(legend1)
    ax.legend(
        handles=method_handles + [node_handle, start_handle, end_handle],
        loc="lower right",
        fontsize=7,
        frameon=True,
        title="Encoding",
        title_fontsize=7,
    )
    ax.text(
        0.01,
        0.01,
        "8 paper nodes; PCE/APCE use DBN-derived line-of-sight observations; "
        "not an independent acoustic end-to-end benchmark.",
        transform=ax.transAxes,
        fontsize=6.2,
        color="#4D5961",
        va="bottom",
    )

    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / "baoding_paper_matched_xy_3source_trajectory"
    exports = {}
    for suffix in (".pdf", ".png", ".svg"):
        path = stem.with_suffix(suffix)
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.025}
        if suffix == ".png":
            kwargs["dpi"] = 300
        fig.savefig(path, **kwargs)
        exports[suffix[1:]] = str(path)
    plt.close(fig)

    registry = {
        "figure": stem.name,
        "reference_visual_grammar": "Zhang et al. 2022 IEEE IoT J. Fig. 15",
        "claim_status": "inspection",
        "panel": "single 2-D x-y trajectory panel",
        "coordinate_system": "centered local ENU, metres",
        "targets": targets,
        "methods": ["GPS truth", "PCE", "APCE"],
        "visual_encoding": {
            "target_colour": {"T1": TARGET_COLORS[1], "T2": TARGET_COLORS[2], "T3": TARGET_COLORS[3]},
            "method_line": {"GPS truth": "-", "PCE": "--", "APCE": "-."},
            "method_marker": {"GPS truth": "none", "PCE": "o", "APCE": "D"},
            "start_marker": "square",
            "end_marker": "star",
            "array_node_marker": "red triangle",
        },
        "representative_seed": args.seed,
        "paper_nodes": list(PAPER_NODES),
        "remote_authoritative_source_root": args.remote_source_root,
        "source_registry": source_registry,
        "exports": exports,
        "provenance_note": "PCE/APCE observations are deterministically derived from upstream DBN Cartesian tracks.",
    }
    write_json(stem.with_name(stem.name + "_registry.json"), registry)
    print(json.dumps({"exports": exports, "registry": str(stem) + "_registry.json"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
