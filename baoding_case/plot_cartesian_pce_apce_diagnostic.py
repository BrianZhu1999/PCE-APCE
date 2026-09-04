#!/usr/bin/env python3
"""Plot the held-out Cartesian frontend/PCE/APCE diagnostic in Python."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})


COLORS = {
    "truth": "#2f3437",
    "triangulation": "#1464a5",
    "pce": "#c45b3c",
    "apce": "#238b72",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_truth(path: Path) -> dict[float, tuple[float, float, float]]:
    with path.open(encoding="utf-8") as stream:
        return {float(row["time_s"]): tuple(float(row[name]) for name in ("px", "py", "pz")) for row in csv.DictReader(stream)}


def read_tri(path: Path) -> dict[float, tuple[float, float, float]]:
    output = {}
    with path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["segment"] == "danyuan_panxuan_3" and row["valid"] == "True":
                output[float(row["time_s"])] = tuple(float(row[name]) for name in ("y_E", "y_N", "y_U"))
    return output


def read_run(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["records"]


def nearest(truth: dict[float, tuple[float, float, float]], time_s: float) -> tuple[float, float, float] | None:
    key = min(truth, key=lambda value: abs(value - time_s))
    return truth[key] if abs(key - time_s) <= 2.0 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    truth = read_truth(args.frontend / "gps_truth.csv")
    triangulation = read_tri(args.frontend / "observations_cartesian.csv")
    runs = {
        "pce": read_run(args.results / "runs/pce_seed_2026082501.json"),
        "apce": read_run(args.results / "runs/apce_seed_2026082501.json"),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.25, 4.15), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.15, 1.0), height_ratios=(1.0, 0.82))
    ax_traj = fig.add_subplot(grid[:, 0])
    ax_err = fig.add_subplot(grid[0, 1])
    ax_unc = fig.add_subplot(grid[1, 1])

    truth_times = sorted(truth)
    ax_traj.plot([truth[t][0] for t in truth_times], [truth[t][1] for t in truth_times], color=COLORS["truth"], lw=1.6, label="GPS truth (offline)")
    tri_times = sorted(triangulation)
    ax_traj.plot([triangulation[t][0] for t in tri_times], [triangulation[t][1] for t in tri_times], color=COLORS["triangulation"], lw=1.0, label="Acoustic triangulation")
    for method in ("pce", "apce"):
        rows = runs[method]
        ax_traj.plot([row["px"] for row in rows], [row["py"] for row in rows], color=COLORS[method], lw=0.9, alpha=0.95, label=method.upper())
    ax_traj.set_xlabel("East (m)")
    ax_traj.set_ylabel("North (m)")
    ax_traj.set_title("Held-out trajectory: Cartesian observation interface", loc="left", fontweight="bold")
    ax_traj.legend(loc="best", fontsize=7)
    ax_traj.set_aspect("equal", adjustable="box")

    for method in ("triangulation", "pce", "apce"):
        if method == "triangulation":
            times = sorted(triangulation)
            errors = []
            for time_s in times:
                target = nearest(truth, time_s)
                if target is not None:
                    estimate = triangulation[time_s]
                    errors.append(((estimate[0] - target[0]) ** 2 + (estimate[1] - target[1]) ** 2 + (estimate[2] - target[2]) ** 2) ** 0.5)
        else:
            times = [float(row["time_s"]) for row in runs[method]]
            errors = [float(row["position_error_m"]) for row in runs[method]]
        ax_err.plot([time - times[0] for time in times], errors, lw=0.9, color=COLORS[method], label=method.upper() if method != "triangulation" else "Acoustic")
    ax_err.set_xlabel("Elapsed time (s)")
    ax_err.set_ylabel("3D error (m)")
    ax_err.set_title("Point-estimate error", loc="left", fontweight="bold")
    ax_err.legend(fontsize=7, ncol=2)

    for method in ("pce", "apce"):
        rows = runs[method]
        times = [float(row["time_s"]) for row in rows]
        coverage = []
        widths = []
        for index in range(len(rows)):
            start = max(0, index - 24)
            coverage.append(sum(float(row["coverage_90"]) for row in rows[start:index + 1]) / (index - start + 1))
            widths.append(float(rows[index]["interval_width_m"]))
        ax_unc.plot([time - times[0] for time in times], coverage, lw=0.9, color=COLORS[method], label=f"{method.upper()} coverage")
        ax_unc.plot([time - times[0] for time in times], [min(1.0, width / 250.0) for width in widths], lw=0.7, ls="--", color=COLORS[method], alpha=0.6, label=f"{method.upper()} width/250")
    ax_unc.axhspan(0.80, 0.98, color="#dfeee8", alpha=0.55, zorder=0)
    ax_unc.set_ylim(0.0, 1.05)
    ax_unc.set_xlabel("Elapsed time (s)")
    ax_unc.set_ylabel("Coverage / scaled width")
    ax_unc.set_title("Uncertainty diagnostic", loc="left", fontweight="bold")
    ax_unc.legend(fontsize=6.5, ncol=2)
    fig.suptitle("Baoding single-source Cartesian PCE/APCE audit", x=0.02, ha="left", fontsize=10.5, fontweight="bold")
    stem = args.output / "baoding_single_cartesian_pce_apce_diagnostic"
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)
    registry = {
        "figure": str(stem),
        "core_conclusion": "The GPS-free Cartesian frontend preserves a coherent trajectory, but the current direct PCE/APCE update does not yet match the upstream acoustic point estimate or coverage gate.",
        "backend": "Python/matplotlib",
        "panels": {
            "trajectory": {"source": str(args.frontend / "observations_cartesian.csv"), "role": "held-out trajectory; GPS truth offline only"},
            "point_error": {"source": str(args.frontend / "observations_cartesian.csv"), "methods": [str(args.results / "runs/pce_seed_2026082501.json"), str(args.results / "runs/apce_seed_2026082501.json")], "role": "point-error comparison"},
            "uncertainty": {"source": str(args.results / "runs"), "role": "rolling 90% coverage and interval-width diagnostic"},
        },
        "source_sha256": {"frontend_manifest": sha256(args.frontend / "frontend_manifest.json"), "plot_script": sha256(Path(__file__))},
        "gps_role": "offline evaluation only",
    }
    (args.output / "baoding_single_cartesian_pce_apce_diagnostic_registry.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
