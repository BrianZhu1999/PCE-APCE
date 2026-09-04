#!/usr/bin/env python3
"""Render a clean Baoding three-source 2-D trajectory figure.

This figure is inspection-only.  It uses a centered local x-y view, one panel,
and a minimal legend so the trajectories stay readable.
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


TARGET_COLORS = {1: "#1F77B4", 2: "#FF7F0E", 3: "#2CA02C"}
METHOD_STYLE = {
    "truth": {"linestyle": "-", "marker": None, "linewidth": 1.8},
    "pce": {"linestyle": "--", "marker": "o", "linewidth": 1.15},
    "apce": {"linestyle": "-.", "marker": "D", "linewidth": 1.15},
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
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--remote-source-root", required=True)
    ap.add_argument("--title", default="Baoding three-source trajectories")
    args = ap.parse_args()

    targets = sorted(
        int(path.name.replace("target", ""))
        for path in args.result_root.iterdir()
        if path.is_dir() and path.name.startswith("target")
    )
    if targets != [1, 2, 3]:
        raise RuntimeError(f"expected target1/2/3, found {targets}")

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
        }
    )

    fig, ax = plt.subplots(figsize=(6.1, 5.2), constrained_layout=True)
    all_xy = []
    registry_rows = []
    for target in targets:
        root = args.result_root / f"target{target}"
        color = TARGET_COLORS[target]
        truth_rows = read_csv(root / "frontend" / "gps_truth.csv")
        truth = np.asarray([[float(r["px"]), float(r["py"])] for r in truth_rows], dtype=float)
        all_xy.append(truth)
        ax.plot(truth[:, 0], truth[:, 1], color=color, lw=METHOD_STYLE["truth"]["linewidth"], zorder=2)
        ax.scatter(truth[0, 0], truth[0, 1], marker="s", s=34, color=color, edgecolor="black", linewidth=0.35, zorder=6)
        ax.scatter(truth[-1, 0], truth[-1, 1], marker="*", s=52, color=color, edgecolor="black", linewidth=0.35, zorder=6)
        registry_rows.append(
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
            xy = np.asarray([[float(r["px"]), float(r["py"])] for r in rows], dtype=float)
            all_xy.append(xy)
            style = METHOD_STYLE[method]
            ax.plot(
                xy[:, 0],
                xy[:, 1],
                color=color,
                lw=style["linewidth"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=2.4 if method == "pce" else 2.6,
                markevery=max(1, len(xy) // 12),
                alpha=0.96,
                zorder=3 if method == "pce" else 4,
            )
            registry_rows.append(
                {
                    "target": target,
                    "method": method,
                    "run_json": str(run_path),
                    "run_sha256": sha256(run_path),
                    "seed": args.seed,
                }
            )

    all_xy_arr = np.vstack(all_xy)
    min_xy = all_xy_arr.min(axis=0)
    max_xy = all_xy_arr.max(axis=0)
    pad = np.maximum((max_xy - min_xy) * 0.08, 20.0)
    ax.set_xlim(min_xy[0] - pad[0], max_xy[0] + pad[0])
    ax.set_ylim(min_xy[1] - pad[1], max_xy[1] + pad[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, ls="--", lw=0.45, color="#B8C1C7", alpha=0.75)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(args.title, fontsize=10.5)

    target_handles = [
        Line2D([0], [0], color=TARGET_COLORS[target], lw=2.0, label=f"Target {target}")
        for target in targets
    ]
    method_handles = [
        Line2D([0], [0], color="#333333", lw=1.4, linestyle=METHOD_STYLE[name]["linestyle"], marker=METHOD_STYLE[name]["marker"], markersize=4, label=label)
        for name, label in (("truth", "GPS truth"), ("pce", "PCE"), ("apce", "APCE"))
    ]
    start_handle = Line2D([0], [0], marker="s", color="black", linestyle="None", markersize=5, label="Start")
    end_handle = Line2D([0], [0], marker="*", color="black", linestyle="None", markersize=7, label="End")
    legend1 = ax.legend(handles=target_handles, loc="upper left", fontsize=7, frameon=True, title="Target", title_fontsize=7)
    ax.add_artist(legend1)
    ax.legend(handles=method_handles + [start_handle, end_handle], loc="lower right", fontsize=7, frameon=True, title="Encoding", title_fontsize=7)

    ax.text(
        0.01,
        0.01,
        "Inspection-only figure from the independent acoustic observation frontend.",
        transform=ax.transAxes,
        fontsize=6.2,
        color="#4D5961",
        va="bottom",
    )

    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / "baoding_three_source_xy_selected_window"
    exports = {}
    for suffix in (".pdf", ".png", ".svg"):
        path = stem.with_suffix(suffix)
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.03}
        if suffix == ".png":
            kwargs["dpi"] = 300
        fig.savefig(path, **kwargs)
        exports[suffix[1:]] = str(path)
    plt.close(fig)

    registry = {
        "figure": stem.name,
        "claim_status": "inspection",
        "panel": "single 2-D x-y trajectory panel",
        "coordinate_system": "centered local ENU, metres",
        "targets": targets,
        "methods": ["GPS truth", "PCE", "APCE"],
        "representative_seed": args.seed,
        "remote_authoritative_source_root": args.remote_source_root,
        "source_registry": registry_rows,
        "exports": exports,
        "note": "This replaces the broken 3-D composite with a plain paper-style trajectory view.",
    }
    write_json(stem.with_name(stem.name + "_registry.json"), registry)
    print(json.dumps({"exports": exports, "registry": str(stem) + "_registry.json"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
