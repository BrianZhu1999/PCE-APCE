#!/usr/bin/env python3
"""Plot and audit the frozen-parameter 87-frame dual-source full-circle run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


TARGETS = (1, 2)
METHODS = ("pce", "apce")
COLORS = {1: "#3F6B8F", 2: "#D97932"}
GPS_COLOR = "#202020"
TEXT_COLOR = "#111111"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
        "font.size": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 0.9,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def load_truth(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = read_csv(path)
    times = np.asarray([int(round(float(row["time_s"]))) for row in rows], dtype=int)
    xyz = np.asarray([[float(row[key]) for key in ("px", "py", "pz")] for row in rows])
    return times, xyz


def load_method(root: Path, method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = sorted((root / "runs").glob(f"{method}_seed_*.json"))
    if len(paths) != 5:
        raise RuntimeError(f"expected five {method} runs under {root}, found {len(paths)}")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(payload.get("status") != "valid" for payload in payloads):
        raise RuntimeError(f"invalid {method} payload under {root}")
    record_maps = [
        {int(round(float(row["time_s"]))): row for row in payload["records"]}
        for payload in payloads
    ]
    times = np.asarray(sorted(set.intersection(*(set(records) for records in record_maps))), dtype=int)
    states = np.asarray([
        [[float(records[int(time)][key]) for key in ("px", "py", "pz")] for time in times]
        for records in record_maps
    ])
    widths = np.asarray([
        [float(records[int(time)]["interval_width_m"]) for time in times]
        for records in record_maps
    ])
    return times, np.median(states, axis=0), np.median(widths, axis=0)


def position_metrics(estimate: np.ndarray, truth: np.ndarray, times: np.ndarray) -> dict[str, float | int]:
    error = np.linalg.norm(estimate - truth, axis=1)
    steps = np.linalg.norm(np.diff(estimate, axis=0), axis=1)
    maximum_step_index = int(np.argmax(steps)) + 1
    return {
        "rmse_m": float(np.sqrt(np.mean(np.square(error)))),
        "median_error_m": float(np.median(error)),
        "p90_error_m": float(np.percentile(error, 90.0)),
        "maximum_step_m": float(np.max(steps)),
        "maximum_step_end_time_s": int(times[maximum_step_index]),
        "start_end_distance_m": float(np.linalg.norm(estimate[-1] - estimate[0])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend-root", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    parser.add_argument("--nodes-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    args.output.mkdir(parents=True, exist_ok=True)

    nodes_rows = read_csv(args.nodes_csv)
    nodes = np.asarray([[float(row["local_E_m"]), float(row["local_N_m"])] for row in nodes_rows])
    node_ids = [int(row["node_id"]) for row in nodes_rows]
    truth: dict[int, np.ndarray] = {}
    estimates: dict[int, dict[str, np.ndarray]] = {}
    widths: dict[int, np.ndarray] = {}
    times: np.ndarray | None = None
    metrics: dict[str, dict[str, float]] = {}
    source_rows: list[dict[str, object]] = []

    for target in TARGETS:
        truth_times, truth_xyz = load_truth(args.frontend_root / f"target{target}" / "frontend" / "gps_truth.csv")
        estimates[target] = {}
        for method in METHODS:
            method_times, median_state, median_width = load_method(args.formal_root / f"target{target}", method)
            if not np.array_equal(method_times, truth_times):
                raise RuntimeError(f"time mismatch for target {target} {method}")
            estimates[target][method] = median_state
            if method == "apce":
                widths[target] = median_width
            metrics[f"target{target}_{method}"] = position_metrics(median_state, truth_xyz, truth_times)
        truth[target] = truth_xyz
        times = truth_times

    assert times is not None
    direct_cost = np.linalg.norm(estimates[1]["apce"] - truth[1], axis=1) + np.linalg.norm(estimates[2]["apce"] - truth[2], axis=1)
    swapped_cost = np.linalg.norm(estimates[1]["apce"] - truth[2], axis=1) + np.linalg.norm(estimates[2]["apce"] - truth[1], axis=1)
    identity_fraction = float(np.mean(direct_cost <= swapped_cost))

    for index, time in enumerate(times):
        row: dict[str, object] = {"time_s": int(time), "elapsed_s": int(time - times[0])}
        for target in TARGETS:
            for dimension, label in enumerate(("east", "north", "up")):
                row[f"target{target}_gps_{label}_m"] = float(truth[target][index, dimension])
                for method in METHODS:
                    row[f"target{target}_{method}_{label}_median_5seeds_m"] = float(estimates[target][method][index, dimension])
            row[f"target{target}_apce_error_m"] = float(np.linalg.norm(estimates[target]["apce"][index] - truth[target][index]))
            row[f"target{target}_apce_median_interval_width_m"] = float(widths[target][index])
        source_rows.append(row)
    source_path = args.output / "dual_full_circle_current_params_source.csv"
    write_csv(source_path, source_rows)

    all_xy = np.concatenate([nodes, *(truth[target][:, :2] for target in TARGETS), *(estimates[target]["apce"][:, :2] for target in TARGETS)])
    lower = np.min(all_xy, axis=0)
    upper = np.max(all_xy, axis=0)
    center = 0.5 * (lower + upper)
    span = max(float(np.max(upper - lower)), 1.0)
    limits = ((center[0] - 0.56 * span, center[0] + 0.56 * span), (center[1] - 0.56 * span, center[1] + 0.56 * span))

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 6.1), constrained_layout=True)
    for ax, target, letter in zip(axes, TARGETS, ("a", "b"), strict=True):
        color = COLORS[target]
        ax.plot(truth[target][:, 0], truth[target][:, 1], color=GPS_COLOR, lw=2.5, ls="--", label="GPS")
        ax.plot(estimates[target]["apce"][:, 0], estimates[target]["apce"][:, 1], color=color, lw=2.6, label="APCE")
        ax.scatter(nodes[:, 0], nodes[:, 1], s=35, marker="P", color=TEXT_COLOR, edgecolor="white", linewidth=0.6, zorder=5)
        for node_id, point in zip(node_ids, nodes, strict=True):
            ax.annotate(f"N{node_id}", point, xytext=(4, 4), textcoords="offset points", fontsize=8, color=TEXT_COLOR)
        ax.scatter(*estimates[target]["apce"][0, :2], s=50, marker="o", facecolor="white", edgecolor=color, linewidth=1.6, zorder=6)
        ax.scatter(*estimates[target]["apce"][-1, :2], s=50, marker="s", facecolor="white", edgecolor=color, linewidth=1.6, zorder=6)
        ax.set_xlim(*limits[0]); ax.set_ylim(*limits[1]); ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("East offset (m)"); ax.set_ylabel("North offset (m)")
        ax.grid(color="#DEDEDE", linewidth=0.6, alpha=0.8)
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_linewidth(0.9); spine.set_color(TEXT_COLOR)
        ax.set_title(f"{letter}   Target {target}: 87-frame complete circle", loc="left", fontsize=15, fontweight="normal", pad=10)
        handles = [
            Line2D([0], [0], color=GPS_COLOR, lw=2.5, ls="--", label="GPS"),
            Line2D([0], [0], color=color, lw=2.6, label=f"APCE (RMSE {metrics[f'target{target}_apce']['rmse_m']:.1f} m)"),
            Line2D([0], [0], marker="P", color=TEXT_COLOR, ls="None", label="Array node"),
            Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor=color, color="none", label="Start"),
            Line2D([0], [0], marker="s", markerfacecolor="white", markeredgecolor=color, color="none", label="End"),
        ]
        ax.legend(handles=handles, loc="lower left", ncol=2, frameon=True, facecolor="white", edgecolor="#222222", framealpha=0.94)

    stem = args.output / "dual_full_circle_current_params_apce"
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=350)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600)
    plt.close(fig)

    geometry = json.loads(args.geometry_manifest.read_text(encoding="utf-8"))
    registry = {
        "figure_contract": {
            "core_conclusion": "Under the frozen 25-s configuration, five-seed median APCE retains both target identities over a GPS-selected common complete-circle window; long-window error and jump diagnostics determine manuscript eligibility.",
            "archetype": "two-panel quantitative trajectory audit",
            "backend": "Python/matplotlib on Super-Server",
            "panels": {"a": "target 1 horizontal trajectory", "b": "target 2 horizontal trajectory"},
            "reviewer_risks": ["GPS geometry selected the showcase window", "the standard 200 m maximum-step gate is evaluated separately", "GPS never enters APCE state updates"],
        },
        "window": geometry["selected"],
        "configuration": {"q_min_accel_mps2": 2.0, "q_max_accel_mps2": 12.0, "observation_covariance_scale": 1.0, "turn_rate_radps": 0.2, "ensemble_members": 48, "seeds": 5},
        "metrics": metrics,
        "identity_match_fraction": identity_fraction,
        "standard_maximum_step_gate_m": 200.0,
        "standard_gate_passed": bool(identity_fraction >= 0.90 and max(metrics["target1_apce"]["maximum_step_m"], metrics["target2_apce"]["maximum_step_m"]) <= 200.0),
        "gps_role": "offline geometric window selection and evaluation only; no GPS enters A6 or PCE/APCE updates",
        "sources": {
            "frontend_root": str(args.frontend_root), "formal_root": str(args.formal_root),
            "geometry_manifest": str(args.geometry_manifest), "nodes_csv": str(args.nodes_csv),
            "source_csv": str(source_path),
        },
        "source_hashes": {
            str(args.geometry_manifest): sha256(args.geometry_manifest),
            str(args.nodes_csv): sha256(args.nodes_csv),
            str(source_path): sha256(source_path),
        },
        "exports": {suffix: str(stem.with_suffix(f".{suffix}")) for suffix in ("png", "pdf", "svg", "tiff")},
    }
    registry_path = args.output / "dual_full_circle_current_params_registry.json"
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": metrics, "identity_match_fraction": identity_fraction, "standard_gate_passed": registry["standard_gate_passed"], "registry": str(registry_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
