#!/usr/bin/env python3
"""Aggregate and render Supplementary Data Figure 2 for Baoding.

This figure is a bounded real-data transfer check on one held-out helicopter
trajectory. Panel a shows a predeclared representative PCE/APCE trajectory.
Panel b reports five-method numerical sensitivity over the same 20 paired
seeds. The script does not rerun tracking and does not alter any run JSON.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


METHODS = ("denkf", "aug_enkf", "bma", "pce", "apce")
SEEDS = tuple(range(2026082000, 2026082020))
LABELS = {"denkf": "DEnKF", "aug_enkf": "Aug-EnKF", "bma": "BMA", "pce": "PCE", "apce": "APCE"}
COLORS = {
    "truth": "#1F252B",
    "denkf": "#7D8992",
    "aug_enkf": "#A6B0B8",
    "bma": "#C9A95E",
    "pce": "#1D6F8A",
    "apce": "#C75A3C",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def valid_records(payload: dict) -> list[dict]:
    return [
        row for row in payload.get("records", [])
        if row.get("position_error_m") is not None
        and finite(row.get("position_error_m"))
        and all(key in row for key in ("truth_x", "truth_y", "truth_z", "px", "py", "pz"))
    ]


def trajectory_scale(records: list[dict]) -> float:
    truth = np.asarray(
        [[float(row["truth_x"]), float(row["truth_y"]), float(row["truth_z"])] for row in records],
        dtype=float,
    )
    centred = truth - truth.mean(axis=0, keepdims=True)
    scale = float(np.sqrt(np.mean(np.sum(centred**2, axis=1))))
    return max(scale, 1e-12)


def summarize_run(payload: dict) -> dict:
    records = valid_records(payload)
    if payload.get("status") != "valid" or not records:
        raise RuntimeError(f"invalid run {payload.get('method')} seed {payload.get('seed')}")
    errors = np.asarray([float(row["position_error_m"]) for row in records], dtype=float)
    rmse = float(np.sqrt(np.mean(errors**2)))
    scale = trajectory_scale(records)
    coverage = float(np.mean([float(row["coverage_90"]) for row in records]))
    return {
        "method": payload["method"],
        "seed": int(payload["seed"]),
        "status": payload["status"],
        "frames": len(records),
        "position_rmse_m": rmse,
        "nrmse_3d": rmse / scale,
        "trajectory_scale_m": scale,
        "position_median_error_m": float(np.median(errors)),
        "position_p90_error_m": float(np.quantile(errors, 0.90, method="higher")),
        "crps_position_m": float(np.mean([float(row["crps_position_m"]) for row in records])),
        "coverage_90": coverage,
        "coverage_error_90": abs(coverage - 0.90),
        "interval_width_m": float(np.mean([float(row["interval_width_m"]) for row in records])),
        "accepted_frame_fraction": float(payload.get("accepted_frame_fraction", 1.0)),
        "observation_segment_count": int(payload.get("observation_segment_count", 0)),
        "runtime_s": float(payload.get("runtime_s", float("nan"))),
        "runner_sha256": str(payload.get("runner_sha256", "")),
        "anchor_fused": bool(payload.get("anchor_fused", False)),
        "observation_only": bool(payload.get("observation_only", False)),
    }


def load_matrix(result: Path) -> tuple[list[dict], dict[tuple[str, int], dict]]:
    rows: list[dict] = []
    payloads: dict[tuple[str, int], dict] = {}
    missing: list[str] = []
    for method in METHODS:
        for seed in SEEDS:
            path = result / "runs" / f"{method}_seed_{seed}.json"
            if not path.is_file():
                missing.append(str(path))
                continue
            payload = read_json(path)
            payloads[(method, seed)] = payload
            rows.append(summarize_run(payload))
    if missing:
        raise RuntimeError(f"missing {len(missing)} frozen runs; first: {missing[0]}")
    expected = len(METHODS) * len(SEEDS)
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} rows, found {len(rows)}")
    runner_hashes = sorted({row["runner_sha256"] for row in rows})
    if len(runner_hashes) != 1:
        raise RuntimeError(f"mixed runner hashes in 20-seed bundle: {runner_hashes}")
    return rows, payloads


def paired_deltas(rows: list[dict]) -> list[dict]:
    by_key = {(row["method"], row["seed"]): row for row in rows}
    output = []
    for seed in SEEDS:
        pce = by_key[("pce", seed)]
        apce = by_key[("apce", seed)]
        best_method = "pce" if pce["position_rmse_m"] <= apce["position_rmse_m"] else "apce"
        best = by_key[(best_method, seed)]
        for baseline in ("denkf", "aug_enkf", "bma"):
            base = by_key[(baseline, seed)]
            output.append({
                "seed": seed,
                "best_pce_apce_method": best_method,
                "baseline": baseline,
                "delta_nrmse_baseline_minus_best_pce_apce": base["nrmse_3d"] - best["nrmse_3d"],
                "delta_crps_baseline_minus_best_pce_apce_m": base["crps_position_m"] - best["crps_position_m"],
                "delta_coverage_error_baseline_minus_best_pce_apce": base["coverage_error_90"] - best["coverage_error_90"],
                "delta_interval_width_baseline_minus_best_pce_apce_m": base["interval_width_m"] - best["interval_width_m"],
            })
    return output


def method_summary(rows: list[dict]) -> list[dict]:
    summary: list[dict] = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        entry = {"method": method, "label": LABELS[method], "runs": len(selected)}
        for metric in (
            "position_rmse_m", "nrmse_3d", "crps_position_m", "coverage_90",
            "coverage_error_90", "interval_width_m", "accepted_frame_fraction", "runtime_s",
        ):
            values = [float(row[metric]) for row in selected]
            entry[f"{metric}_mean"] = float(statistics.mean(values))
            entry[f"{metric}_sd"] = float(statistics.stdev(values))
            entry[f"{metric}_median"] = float(statistics.median(values))
            entry[f"{metric}_min"] = float(min(values))
            entry[f"{metric}_max"] = float(max(values))
        summary.append(entry)
    return summary


def representative_seed(rows: list[dict]) -> dict:
    by_key = {(row["method"], row["seed"]): row for row in rows}
    candidates = []
    for seed in SEEDS:
        pce = by_key[("pce", seed)]
        apce = by_key[("apce", seed)]
        candidates.append({
            "seed": seed,
            "joint_pce_apce_rmse_m": 0.5 * (pce["position_rmse_m"] + apce["position_rmse_m"]),
            "pce_rmse_m": pce["position_rmse_m"],
            "apce_rmse_m": apce["position_rmse_m"],
            "pce_nrmse": pce["nrmse_3d"],
            "apce_nrmse": apce["nrmse_3d"],
        })
    median_value = float(np.median([item["joint_pce_apce_rmse_m"] for item in candidates]))
    chosen = min(
        candidates,
        key=lambda item: (abs(item["joint_pce_apce_rmse_m"] - median_value), item["seed"]),
    )
    ordered = sorted(candidates, key=lambda item: (item["joint_pce_apce_rmse_m"], item["seed"]))
    chosen["rank_by_joint_rmse"] = 1 + [item["seed"] for item in ordered].index(chosen["seed"])
    chosen["median_joint_pce_apce_rmse_m"] = median_value
    chosen["selection_rule"] = (
        "seed with PCE/APCE mean RMSE closest to the median of the 20 paired seeds; "
        "ties resolved by smaller seed"
    )
    return chosen


def aggregate(result: Path, output: Path, base_smoke: Path) -> tuple[list[dict], dict[tuple[str, int], dict], dict, dict]:
    rows, payloads = load_matrix(result)
    summary = method_summary(rows)
    deltas = paired_deltas(rows)
    rep = representative_seed(rows)
    aggregate_dir = output / "aggregate"
    write_csv(aggregate_dir / "baoding_anchor_fused_20seed_run_source_data.csv", rows)
    write_csv(aggregate_dir / "baoding_anchor_fused_20seed_method_summary.csv", summary)
    write_csv(aggregate_dir / "baoding_anchor_fused_20seed_paired_deltas.csv", deltas)
    runner_hash = sorted({row["runner_sha256"] for row in rows})[0]
    manifest = {
        "task": "Supplementary Data Figure 2: Baoding single-turn trajectory and 20-seed statistics",
        "core_conclusion": (
            "On one held-out Baoding single-helicopter turning trajectory, the anchor-fused acoustic reconstruction "
            "yields continuous auditable PCE/APCE tracks and a five-method numerical-sensitivity comparison of point "
            "accuracy, distributional score and calibration trade-offs."
        ),
        "result_root": str(result),
        "base_smoke_protocol": str(base_smoke),
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "expected_runs": len(METHODS) * len(SEEDS),
        "valid_runs": len(rows),
        "runner_sha256": runner_hash,
        "representative_seed": rep,
        "seeds_role": "numerical sensitivity on one held-out physical trajectory; not independent field experiments",
        "formal_admission": False,
        "display_policy": "panel a uses the predeclared median joint PCE/APCE RMSE seed; panel b uses all 20 paired seeds",
        "source_data": {
            "run_level": str(aggregate_dir / "baoding_anchor_fused_20seed_run_source_data.csv"),
            "method_summary": str(aggregate_dir / "baoding_anchor_fused_20seed_method_summary.csv"),
            "paired_deltas": str(aggregate_dir / "baoding_anchor_fused_20seed_paired_deltas.csv"),
        },
    }
    write_json(aggregate_dir / "baoding_anchor_fused_20seed_manifest.json", manifest)
    return rows, payloads, rep, manifest


def setup_style() -> None:
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
        "axes.formatter.useoffset": False,
    })


def contiguous_spans(records: list[dict]) -> list[np.ndarray]:
    if not records:
        return []
    times = np.asarray([float(row["time_s"]) for row in records], dtype=float)
    accepted = np.asarray([bool(row.get("accepted_acoustic_frame", True)) for row in records], dtype=bool)
    segment_ids = np.asarray([int(row.get("observation_segment_id", 0)) for row in records], dtype=int)
    indices = np.flatnonzero(accepted & np.isfinite(times))
    if len(indices) == 0:
        return []
    gaps = np.diff(times[indices])
    typical = float(np.median(gaps)) if len(gaps) else 1.0
    max_gap = max(1.5 * typical, typical + 0.5, 1.5)
    spans: list[list[int]] = [[int(indices[0])]]
    for left, right in zip(indices[:-1], indices[1:]):
        if segment_ids[right] != segment_ids[left] or times[right] - times[left] > max_gap:
            spans.append([])
        spans[-1].append(int(right))
    return [np.asarray(span, dtype=int) for span in spans if span]


def trajectory_source(payloads: dict[tuple[str, int], dict], seed: int, output: Path) -> tuple[list[dict], dict]:
    pce_records = valid_records(payloads[("pce", seed)])
    apce_records = valid_records(payloads[("apce", seed)])
    if [row["time_s"] for row in pce_records] != [row["time_s"] for row in apce_records]:
        raise RuntimeError(f"PCE/APCE time mismatch for representative seed {seed}")
    truth_global = np.asarray(
        [[float(row["truth_x"]), float(row["truth_y"]), float(row["truth_z"])] for row in pce_records],
        dtype=float,
    )
    origin = truth_global.mean(axis=0)
    rows: list[dict] = []
    for index, (pce, apce) in enumerate(zip(pce_records, apce_records)):
        truth = (truth_global[index] - origin) / 1000.0
        pce_xyz = (np.asarray([float(pce["px"]), float(pce["py"]), float(pce["pz"])]) - origin) / 1000.0
        apce_xyz = (np.asarray([float(apce["px"]), float(apce["py"]), float(apce["pz"])]) - origin) / 1000.0
        rows.append({
            "panel": "a",
            "seed": seed,
            "time_s": pce["time_s"],
            "time_rel_s": float(pce["time_s"]) - float(pce_records[0]["time_s"]),
            "origin_global_x_m": origin[0],
            "origin_global_y_m": origin[1],
            "origin_global_z_m": origin[2],
            "truth_east_km": truth[0],
            "truth_north_km": truth[1],
            "truth_up_km": truth[2],
            "pce_east_km": pce_xyz[0],
            "pce_north_km": pce_xyz[1],
            "pce_up_km": pce_xyz[2],
            "apce_east_km": apce_xyz[0],
            "apce_north_km": apce_xyz[1],
            "apce_up_km": apce_xyz[2],
            "pce_error_m": pce["position_error_m"],
            "apce_error_m": apce["position_error_m"],
            "accepted_acoustic_frame": bool(pce.get("accepted_acoustic_frame", True)),
            "observation_segment_id": int(pce.get("observation_segment_id", 0)),
            "inlier_nodes": int(pce.get("inlier_nodes", 0)),
            "pce_relocalized": bool(pce.get("relocalized", False)),
            "apce_relocalized": bool(apce.get("relocalized", False)),
        })
    source_path = output / "figures" / "supplementary_data_figure2_trajectory_source.csv"
    write_csv(source_path, rows)
    meta = {
        "source": str(source_path),
        "origin_global_projected_m": origin.tolist(),
        "coordinate_display": "local ENU offsets from GPS-truth centroid, divided by 1000 and labelled in km",
        "accepted_frame_count": int(sum(bool(row["accepted_acoustic_frame"]) for row in rows)),
        "frame_count": len(rows),
        "gap_count": max(0, len(contiguous_spans(pce_records)) - 1),
    }
    return rows, meta


def bootstrap_ci(values: np.ndarray, seed: int, repeats: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(repeats, dtype=float)
    for index in range(repeats):
        means[index] = values[rng.integers(0, n, size=n)].mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def stripplot(axis: plt.Axes, rows: list[dict], metric: str, title: str, xlabel: str, reference: float | None = None) -> None:
    rng = np.random.default_rng(20260820 + sum(ord(char) for char in metric))
    for index, method in enumerate(METHODS):
        values = np.asarray([row[metric] for row in rows if row["method"] == method], dtype=float)
        y = np.full(len(values), index, dtype=float) + rng.uniform(-0.12, 0.12, size=len(values))
        axis.scatter(values, y, s=9.5, color=COLORS[method], alpha=0.48, edgecolors="none", zorder=2)
        mean = float(values.mean())
        lo, hi = bootstrap_ci(values, 20260820 + index * 101 + len(metric))
        axis.errorbar(
            mean,
            index,
            xerr=np.asarray([[mean - lo], [hi - mean]]),
            fmt="o",
            ms=3.6,
            color=COLORS[method],
            capsize=2.0,
            lw=1.15,
            zorder=3,
        )
    if reference is not None:
        axis.axvline(reference, color="#4E5961", lw=0.8, ls=(0, (2.5, 2.5)), zorder=1)
    axis.set_yticks(range(len(METHODS)), [LABELS[method] for method in METHODS])
    axis.invert_yaxis()
    axis.grid(axis="x", color="#DCE2E6", lw=0.5, zorder=0)
    axis.set_axisbelow(True)
    axis.set_xlabel(xlabel)
    axis.set_title(title, loc="left", fontsize=7.0, weight="bold", pad=3)
    axis.ticklabel_format(axis="x", style="plain", useOffset=False)


def make_figure(rows: list[dict], payloads: dict[tuple[str, int], dict], rep: dict, output: Path, manifest: dict) -> dict:
    setup_style()
    seed = int(rep["seed"])
    pce_records = valid_records(payloads[("pce", seed)])
    trajectory_rows, trajectory_meta = trajectory_source(payloads, seed, output)
    truth = np.asarray([[row["truth_east_km"], row["truth_north_km"], row["truth_up_km"]] for row in trajectory_rows], dtype=float)
    pce = np.asarray([[row["pce_east_km"], row["pce_north_km"], row["pce_up_km"]] for row in trajectory_rows], dtype=float)
    apce = np.asarray([[row["apce_east_km"], row["apce_north_km"], row["apce_up_km"]] for row in trajectory_rows], dtype=float)
    spans = contiguous_spans(pce_records)

    fig = plt.figure(figsize=(7.25, 4.05), constrained_layout=True)
    outer = fig.add_gridspec(1, 2, width_ratios=(1.13, 1.0), wspace=0.18)
    ax3d = fig.add_subplot(outer[0], projection="3d")
    right = outer[1].subgridspec(2, 2, wspace=0.34, hspace=0.42)
    stat_axes = [fig.add_subplot(right[0, 0]), fig.add_subplot(right[0, 1]), fig.add_subplot(right[1, 0]), fig.add_subplot(right[1, 1])]

    ax3d.plot(truth[:, 0], truth[:, 1], truth[:, 2], color=COLORS["truth"], lw=1.75, label="GPS truth", zorder=4)
    for span_index, indices in enumerate(spans):
        ax3d.plot(pce[indices, 0], pce[indices, 1], pce[indices, 2], color=COLORS["pce"], lw=1.15, label="PCE" if span_index == 0 else None, zorder=3)
        ax3d.plot(apce[indices, 0], apce[indices, 1], apce[indices, 2], color=COLORS["apce"], lw=1.15, label="APCE" if span_index == 0 else None, zorder=2)
    ax3d.scatter(truth[0, 0], truth[0, 1], truth[0, 2], color=COLORS["truth"], marker="o", s=22, label="start", zorder=5)
    ax3d.scatter(truth[-1, 0], truth[-1, 1], truth[-1, 2], color=COLORS["truth"], marker="s", s=24, label="end", zorder=5)
    ax3d.set_xlabel("East offset (km)", labelpad=2)
    ax3d.set_ylabel("North offset (km)", labelpad=2)
    ax3d.set_zlabel("Up offset (km)", labelpad=0)
    ax3d.ticklabel_format(style="plain", useOffset=False)
    ax3d.tick_params(labelsize=5.7, pad=0)
    ax3d.view_init(elev=24, azim=-58)
    span = np.ptp(np.vstack([truth, pce, apce]), axis=0)
    ax3d.set_box_aspect(tuple(np.maximum(span, 0.001)))
    z_min, z_max = float(np.min(np.vstack([truth, pce, apce])[:, 2])), float(np.max(np.vstack([truth, pce, apce])[:, 2]))
    ax3d.set_zticks(np.linspace(z_min, z_max, 4))
    ax3d.set_title("Held-out single-turn trajectory", loc="left", fontsize=7.6, weight="bold", pad=4)
    ax3d.legend(loc="upper left", bbox_to_anchor=(-0.03, 0.98), fontsize=5.8, ncols=1, handlelength=2.4)

    ax_in = inset_axes(ax3d, width="32%", height="25%", loc="upper right", borderpad=0.78)
    ax_in.set_facecolor((1, 1, 1, 0.92))
    ax_in.plot(truth[:, 0], truth[:, 1], color=COLORS["truth"], lw=1.1)
    for indices in spans:
        ax_in.plot(pce[indices, 0], pce[indices, 1], color=COLORS["pce"], lw=0.9)
        ax_in.plot(apce[indices, 0], apce[indices, 1], color=COLORS["apce"], lw=0.9)
    ax_in.scatter(truth[0, 0], truth[0, 1], color=COLORS["truth"], marker="o", s=10)
    ax_in.scatter(truth[-1, 0], truth[-1, 1], color=COLORS["truth"], marker="s", s=10)
    ax_in.set_aspect("equal", adjustable="box")
    ax_in.set_title("horizontal view", fontsize=5.4, loc="left", pad=1)
    ax_in.tick_params(labelsize=5.0, length=1.8)
    ax_in.ticklabel_format(style="plain", useOffset=False)
    ax_in.grid(color="#E5EAEE", lw=0.35)

    stripplot(stat_axes[0], rows, "nrmse_3d", "nRMSE", "dimensionless")
    stripplot(stat_axes[1], rows, "crps_position_m", "Position CRPS", "m")
    stripplot(stat_axes[2], rows, "coverage_error_90", "90% coverage error", "|coverage - 0.90|", reference=0.0)
    stripplot(stat_axes[3], rows, "interval_width_m", "90% interval width", "m")
    for axis in (stat_axes[1], stat_axes[3]):
        axis.tick_params(axis="y", left=False, labelleft=False)

    fig.text(0.006, 0.990, "a", ha="left", va="top", fontsize=9.5, fontweight="bold")
    fig.text(0.522, 0.990, "b", ha="left", va="top", fontsize=9.5, fontweight="bold")
    fig.text(
        0.012,
        1.004,
        f"Panel a uses the predeclared representative seed {seed}; panel b shows 20 paired numerical-sensitivity seeds per method.",
        ha="left",
        va="top",
        fontsize=6.1,
        color="#4D5961",
    )

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    stem = figures / "supplementary_data_figure2_baoding_anchor_fused_20seeds"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    stats_source = figures / "supplementary_data_figure2_statistics_source.csv"
    write_csv(stats_source, [{**row, "panel": "b"} for row in rows])
    figure_source = figures / "supplementary_data_figure2_source.csv"
    write_csv(figure_source, trajectory_rows + [{**row, "panel": "b"} for row in rows])

    registry = {
        "figure": "Supplementary Data Figure 2",
        "claim": manifest["core_conclusion"],
        "backend": "python/matplotlib",
        "archetype": "asymmetric mixed-modality figure",
        "final_size": "183 mm wide equivalent; reader preview PNG at 300 dpi",
        "panels": {
            "a": {
                "conclusion": "A representative numerical-sensitivity seed forms continuous PCE/APCE tracks on the held-out single-turn trajectory.",
                "source": trajectory_meta["source"],
                "seed_selection": rep,
                "coordinate_definition": trajectory_meta["coordinate_display"],
                "gap_policy": "method trajectories are split at rejected acoustic frames or time/segment discontinuities; no interpolation is used",
                "frame_count": trajectory_meta["frame_count"],
                "accepted_frame_count": trajectory_meta["accepted_frame_count"],
                "gap_count": trajectory_meta["gap_count"],
            },
            "b": {
                "conclusion": "Five methods are compared on the same 20 paired numerical-sensitivity seeds using point, distributional and calibration metrics.",
                "metrics": ["nrmse_3d", "crps_position_m", "coverage_error_90", "interval_width_m"],
                "source": str(stats_source),
                "seed_selection": "all 20 frozen paired seeds for each method",
                "summary_mark": "mean with seed-bootstrap 95% CI of the mean; dots are individual seeds",
            },
        },
        "provenance": {
            "remote_result_bundle": manifest["result_root"],
            "base_smoke_protocol": manifest["base_smoke_protocol"],
            "runner_sha256": manifest["runner_sha256"],
            "run_level_source": manifest["source_data"]["run_level"],
            "method_summary": manifest["source_data"]["method_summary"],
            "paired_deltas": manifest["source_data"]["paired_deltas"],
        },
        "exports": {suffix[1:]: str(stem.with_suffix(suffix)) for suffix in (".png", ".pdf", ".svg", ".tiff")},
        "figure_source": str(figure_source),
    }
    write_json(figures / "supplementary_data_figure2_registry.json", registry)
    write_json(figures / "supplementary_data_figure2_qa.json", {
        "core_conclusion": manifest["core_conclusion"],
        "backend_exclusive": "Python/matplotlib for plot, preview, and exports",
        "run_count": manifest["valid_runs"],
        "complete_5_by_20_matrix": manifest["valid_runs"] == 100,
        "representative_seed": rep,
        "seed_definition": manifest["seeds_role"],
        "statistics": "individual seed dots plus mean and bootstrap 95% CI; no independent-field-experiment inference",
        "source_data": {
            **manifest["source_data"],
            "trajectory": trajectory_meta["source"],
            "statistics": str(stats_source),
            "figure_source": str(figure_source),
        },
        "image_integrity": "No scientific-content editing in vector exports; figure generated from run-level JSON by this script.",
        "checks": {
            "single_runner_hash": True,
            "complete_5_by_20_matrix": manifest["valid_runs"] == 100,
            "panel_a_local_km_coordinates": True,
            "no_scientific_axis_offset": True,
            "panel_a_representative_seed_rule": rep["selection_rule"],
            "panel_b_seed_points_match_run_source": len(rows) == 100,
        },
    })
    return registry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-smoke", type=Path, required=True)
    args = parser.parse_args()
    rows, payloads, rep, manifest = aggregate(args.result, args.output, args.base_smoke)
    registry = make_figure(rows, payloads, rep, args.output, manifest)
    print(json.dumps({
        "valid_runs": manifest["valid_runs"],
        "representative_seed": rep,
        "figure_png": registry["exports"]["png"],
        "figure_pdf": registry["exports"]["pdf"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
