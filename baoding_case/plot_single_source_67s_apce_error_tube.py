#!/usr/bin/env python3
"""Prototype: GPS/APCE trajectory with a posterior interval-width proxy tube."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


START_S, END_S = 46254.0, 46320.0


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def truth_map(path):
    return {float(r["time_s"]): np.asarray([float(r[k]) for k in ("px", "py", "pz")], dtype=float) for r in csv_rows(path)}


def apce_payload(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {float(r["time_s"]): (np.asarray([float(r[k]) for k in ("px", "py", "pz")], dtype=float), float(r["interval_width_m"])) for r in payload["records"]}


def nearest(mapping, t):
    key = min(mapping, key=lambda value: abs(value - t))
    return mapping[key] if abs(key - t) <= 2.0 else None


def tube_surface(points, radii, sides=18):
    theta = np.linspace(0, 2 * np.pi, sides, endpoint=True)
    verts = np.zeros((len(points), sides, 3), dtype=float)
    for i, point in enumerate(points):
        if i == 0:
            tangent = points[1] - points[0]
        elif i == len(points) - 1:
            tangent = points[-1] - points[-2]
        else:
            tangent = points[i + 1] - points[i - 1]
        tangent /= max(np.linalg.norm(tangent), 1e-9)
        reference = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(tangent, reference)) > 0.92:
            reference = np.array([0.0, 1.0, 0.0])
        normal = np.cross(tangent, reference); normal /= max(np.linalg.norm(normal), 1e-9)
        binormal = np.cross(tangent, normal); binormal /= max(np.linalg.norm(binormal), 1e-9)
        verts[i] = point + radii[i] * (np.cos(theta)[:, None] * normal + np.sin(theta)[:, None] * binormal)
    return verts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend", type=Path, required=True)
    parser.add_argument("--apce-runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gps = truth_map(args.frontend / "gps_truth.csv")
    maps = [apce_payload(path) for path in sorted(args.apce_runs.glob("apce_seed_*.json"))]
    times = sorted(t for t in gps if START_S <= t <= END_S)
    truth = np.asarray([gps[t] for t in times])
    est_stack, width_stack = [], []
    for mapping in maps:
        values = [nearest(mapping, t) for t in times]
        est_stack.append(np.asarray([item[0] for item in values]))
        width_stack.append(np.asarray([item[1] for item in values]))
    apce = np.median(np.asarray(est_stack), axis=0)
    # The run JSON stores only a scalar mean position interval width.  The
    # radius is therefore an isotropic visual proxy, not a full covariance tube.
    radius = np.median(np.asarray(width_stack), axis=0) / 2.0
    origin = truth.mean(axis=0)
    truth -= origin; apce -= origin
    surface = tube_surface(apce, radius)

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8, "svg.fonttype": "none", "pdf.fonttype": 42, "axes.spines.right": False, "axes.spines.top": False, "legend.frameon": False})
    fig = plt.figure(figsize=(8.4, 4.6), constrained_layout=False)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.15, 0.85))
    ax3 = fig.add_subplot(grid[0, 0], projection="3d")
    ax2 = fig.add_subplot(grid[0, 1])
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.20, top=0.82, wspace=0.18)
    truth_color, apce_color = "#20262C", "#C75A3C"
    ax3.plot(truth[:, 0], truth[:, 1], truth[:, 2], color=truth_color, lw=2.0, label="GPS truth")
    ax3.plot(apce[:, 0], apce[:, 1], apce[:, 2], color=apce_color, lw=1.8, label="APCE median")
    ax3.plot_surface(surface[:, :, 0], surface[:, :, 1], surface[:, :, 2], color=apce_color, alpha=0.16, linewidth=0, antialiased=True, shade=False, label="APCE interval-width proxy")
    ax3.scatter(*truth[0], color=truth_color, s=22, marker="o"); ax3.scatter(*truth[-1], color=truth_color, s=26, marker="s")
    ax3.set_xlabel("East offset (m)"); ax3.set_ylabel("North offset (m)"); ax3.set_zlabel("Up offset (m)")
    ax3.set_title("A  3D trajectory with error tube", loc="left", fontweight="bold")
    ax3.view_init(elev=23, azim=-58); ax3.set_box_aspect(tuple(np.maximum(np.ptp(np.vstack((truth, apce)), axis=0), 1.0)))
    ax3.legend(loc="upper left", fontsize=7)
    ax2.plot(truth[:, 0], truth[:, 1], color=truth_color, lw=2.0, label="GPS truth")
    ax2.plot(apce[:, 0], apce[:, 1], color=apce_color, lw=1.8, label="APCE median")
    ax2.set_aspect("equal", adjustable="box"); ax2.set_xlabel("East offset (m)"); ax2.set_ylabel("North offset (m)")
    ax2.set_title("B  Horizontal projection", loc="left", fontweight="bold"); ax2.grid(color="#E2E7EB", lw=0.5); ax2.legend(fontsize=7, loc="best")
    fig.suptitle("Baoding single-source near-full-circle window | APCE posterior interval-width proxy", x=0.02, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.5, 0.01, "The translucent tube is a post-processing isotropic proxy from APCE 90% interval width; it is not a full covariance ellipsoid.", ha="center", fontsize=7, color="#4D4D4D")
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / "baoding_single_source_67s_apce_error_tube"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    registry = {"backend": "Python/matplotlib", "window": {"start_time_s": START_S, "end_time_s": END_S, "frames": len(times)}, "tube_definition": "isotropic radius = median over five seeds of scalar APCE interval_width_m / 2; visual proxy only", "gps_role": "offline evaluation only", "sources": {"frontend": str(args.frontend), "apce_runs": [str(p) for p in sorted(args.apce_runs.glob("apce_seed_*.json"))]}, "outputs": {"png": str(stem.with_suffix(".png")), "pdf": str(stem.with_suffix(".pdf")), "svg": str(stem.with_suffix(".svg"))}}
    (args.output / "baoding_single_source_67s_apce_error_tube_registry.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(registry, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
