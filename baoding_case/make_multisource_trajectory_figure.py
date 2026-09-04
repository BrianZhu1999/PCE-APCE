#!/usr/bin/env python3
"""Render Baoding dual-/triple-source trajectories in the single-source a/b style.

The script is post-processing only.  It reads frozen target-labelled bundles::

    RESULT_ROOT/target1/frontend/gps_truth.csv
    RESULT_ROOT/target1/runs/pce_seed_<seed>.json
    RESULT_ROOT/target1/runs/apce_seed_<seed>.json

and any additional ``target<N>`` directories.  Panel a is a shared local-ENU
3-D trajectory view with a horizontal inset.  Panel b contains target-stratified
numerical-sensitivity summaries.  The default output is explicitly inspection
only; gate admission is recorded as provenance and is never inferred here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


DEFAULT_METHODS = ("pce", "apce")
METHOD_LABELS = {
    "denkf": "DEnKF",
    "aug_enkf": "Aug-EnKF",
    "bma": "BMA",
    "pce": "PCE",
    "apce": "APCE",
}
METHOD_COLORS = {
    "truth": "#1F252B",
    "denkf": "#7D8992",
    "aug_enkf": "#A6B0B8",
    "bma": "#C9A95E",
    "pce": "#1D6F8A",
    "apce": "#C75A3C",
}
FALLBACK_COLORS = ("#7C4D9B", "#4E8A55", "#AD6C2F", "#506C8F")
TARGET_LINESTYLES = (
    "-",
    (0, (5.0, 2.0)),
    (0, (1.4, 1.4)),
    (0, (4.0, 1.3, 1.2, 1.3)),
)
TARGET_MARKERS = ("o", "s", "^", "D", "v", "P")
RUN_RE = re.compile(r"^(?P<method>.+)_seed_(?P<seed>-?\d+)$")
TARGET_RE = re.compile(r"^target(?P<target>\d+)$", re.IGNORECASE)


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "axes.formatter.useoffset": False,
            "legend.frameon": False,
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def target_number(path: Path) -> int:
    match = TARGET_RE.match(path.name)
    if match is None:
        raise ValueError(f"not a target directory: {path}")
    return int(match.group("target"))


def discover_targets(result_root: Path) -> dict[int, Path]:
    targets = {
        target_number(path): path
        for path in result_root.iterdir()
        if path.is_dir() and TARGET_RE.match(path.name)
    }
    if not targets:
        raise RuntimeError(f"no target<N> directories found below {result_root}")
    expected = list(range(1, len(targets) + 1))
    if sorted(targets) != expected:
        raise RuntimeError(f"target IDs must be contiguous from 1; found {sorted(targets)}")
    return dict(sorted(targets.items()))


def discover_run_files(target_root: Path, methods: tuple[str, ...]) -> dict[tuple[str, int], Path]:
    runs = target_root / "runs"
    if not runs.is_dir():
        raise RuntimeError(f"missing run directory: {runs}")
    output: dict[tuple[str, int], Path] = {}
    for path in sorted(runs.glob("*_seed_*.json")):
        match = RUN_RE.match(path.stem)
        if match is None:
            continue
        method = match.group("method")
        if method not in methods:
            continue
        key = (method, int(match.group("seed")))
        if key in output:
            raise RuntimeError(f"duplicate run for {target_root.name} {key}")
        output[key] = path
    return output


def common_seeds(
    targets: dict[int, Path], methods: tuple[str, ...]
) -> tuple[list[int], dict[int, dict[tuple[str, int], Path]]]:
    files_by_target = {
        target: discover_run_files(target_root, methods)
        for target, target_root in targets.items()
    }
    seed_sets = []
    for target, files in files_by_target.items():
        for method in methods:
            seeds = {seed for run_method, seed in files if run_method == method}
            if not seeds:
                raise RuntimeError(f"no {method} seeds found for target {target}")
            seed_sets.append(seeds)
    paired = sorted(set.intersection(*seed_sets))
    if not paired:
        raise RuntimeError("no seed is common to every target and requested method")
    return paired, files_by_target


def valid_records(payload: dict) -> list[dict]:
    """Keep finite position records; retain rejected-frame metadata for gap splitting."""
    output = []
    for row in payload.get("records", []):
        if not finite(row.get("time_s")):
            continue
        if not all(finite(row.get(key)) for key in ("px", "py", "pz")):
            continue
        if not finite(row.get("position_error_m")):
            continue
        output.append(row)
    output.sort(key=lambda row: float(row["time_s"]))
    return output


def truth_xyz(records: list[dict], fallback_truth: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    """Return truth times/positions from run records or the frontend GPS CSV."""
    if records and all(
        all(finite(row.get(key)) for key in ("truth_x", "truth_y", "truth_z"))
        for row in records
    ):
        times = np.asarray([float(row["time_s"]) for row in records], dtype=float)
        truth = np.asarray(
            [[float(row["truth_x"]), float(row["truth_y"]), float(row["truth_z"])] for row in records],
            dtype=float,
        )
        return times, truth
    gps = {
        round(float(row["time_s"]), 6): np.asarray(
            [float(row["px"]), float(row["py"]), float(row["pz"])], dtype=float
        )
        for row in fallback_truth
        if all(finite(row.get(key)) for key in ("time_s", "px", "py", "pz"))
    }
    selected_times = []
    selected_truth = []
    for row in records:
        key = round(float(row["time_s"]), 6)
        if key in gps:
            selected_times.append(float(row["time_s"]))
            selected_truth.append(gps[key])
    if not selected_truth:
        raise RuntimeError("run records do not carry truth_xyz and cannot be joined to gps_truth.csv")
    return np.asarray(selected_times, dtype=float), np.asarray(selected_truth, dtype=float)


def trajectory_scale(records: list[dict], fallback_truth: list[dict[str, str]]) -> float:
    _, truth = truth_xyz(records, fallback_truth)
    centred = truth - truth.mean(axis=0, keepdims=True)
    scale = float(np.sqrt(np.mean(np.sum(centred**2, axis=1))))
    if not math.isfinite(scale) or scale <= 0:
        raise RuntimeError("invalid truth trajectory scale")
    return scale


def mean_field(records: list[dict], key: str) -> float:
    values = [float(row[key]) for row in records if finite(row.get(key))]
    return float(np.mean(values)) if values else float("nan")


def summarize_run(
    target: int,
    method: str,
    seed: int,
    path: Path,
    fallback_truth: list[dict[str, str]],
) -> tuple[dict, dict, list[dict]]:
    payload = read_json(path)
    records = valid_records(payload)
    if not records:
        raise RuntimeError(f"no finite position records in {path}")
    errors = np.asarray([float(row["position_error_m"]) for row in records], dtype=float)
    scale = trajectory_scale(records, fallback_truth)
    coverage = mean_field(records, "coverage_90")
    summary = {
        "panel": "b",
        "target": target,
        "method": method,
        "method_label": METHOD_LABELS.get(method, method),
        "seed": seed,
        "status": str(payload.get("status", "unspecified")),
        "frames": len(records),
        "position_rmse_m": float(np.sqrt(np.mean(errors**2))),
        "nrmse_3d": float(np.sqrt(np.mean(errors**2))) / scale,
        "trajectory_scale_m": scale,
        "position_crps_m": mean_field(records, "crps_position_m"),
        "coverage_90": coverage,
        "coverage_error_90": abs(coverage - 0.90) if math.isfinite(coverage) else float("nan"),
        "interval_width_m": mean_field(records, "interval_width_m"),
        "run_file": str(path),
        "run_sha256": sha256(path),
    }
    return summary, payload, records


def load_all_runs(
    targets: dict[int, Path],
    methods: tuple[str, ...],
    seeds: list[int],
    files_by_target: dict[int, dict[tuple[str, int], Path]],
) -> tuple[list[dict], dict[tuple[int, str, int], tuple[dict, list[dict]]], dict[int, list[dict[str, str]]]]:
    summaries: list[dict] = []
    payloads: dict[tuple[int, str, int], tuple[dict, list[dict]]] = {}
    truth_by_target: dict[int, list[dict[str, str]]] = {}
    for target, target_root in targets.items():
        truth_path = target_root / "frontend" / "gps_truth.csv"
        if not truth_path.is_file():
            raise RuntimeError(f"missing GPS truth: {truth_path}")
        truth_rows = read_csv(truth_path)
        if not truth_rows:
            raise RuntimeError(f"empty GPS truth: {truth_path}")
        truth_by_target[target] = truth_rows
        for method in methods:
            for seed in seeds:
                path = files_by_target[target][(method, seed)]
                summary, payload, records = summarize_run(target, method, seed, path, truth_rows)
                summaries.append(summary)
                payloads[(target, method, seed)] = (payload, records)
    return summaries, payloads, truth_by_target


def choose_representative_seed(summaries: list[dict], seeds: list[int], explicit: int | None) -> dict:
    by_seed: dict[int, list[float]] = {seed: [] for seed in seeds}
    for row in summaries:
        by_seed[int(row["seed"])].append(float(row["position_rmse_m"]))
    candidates = [
        {"seed": seed, "joint_target_method_rmse_m": float(np.mean(by_seed[seed]))}
        for seed in seeds
    ]
    median_score = float(np.median([item["joint_target_method_rmse_m"] for item in candidates]))
    if explicit is not None:
        if explicit not in by_seed:
            raise RuntimeError(f"requested representative seed {explicit} is not in paired seeds {seeds}")
        chosen = next(dict(item) for item in candidates if item["seed"] == explicit)
        rule = "explicit command-line seed"
    else:
        chosen = dict(
            min(
                candidates,
                key=lambda item: (abs(item["joint_target_method_rmse_m"] - median_score), item["seed"]),
            )
        )
        rule = (
            "common seed whose mean RMSE across every target and plotted method is closest to the "
            "median joint score; ties resolved by smaller seed"
        )
    ordered = sorted(candidates, key=lambda item: (item["joint_target_method_rmse_m"], item["seed"]))
    chosen.update(
        {
            "median_joint_target_method_rmse_m": median_score,
            "rank_by_joint_rmse": 1 + [item["seed"] for item in ordered].index(chosen["seed"]),
            "selection_rule": rule,
        }
    )
    return chosen


def time_key(value: object) -> float:
    return round(float(value), 6)


def aligned_trajectory(
    target: int,
    seed: int,
    methods: tuple[str, ...],
    payloads: dict[tuple[int, str, int], tuple[dict, list[dict]]],
    fallback_truth: list[dict[str, str]],
) -> dict:
    records_by_method = {
        method: payloads[(target, method, seed)][1]
        for method in methods
    }
    maps = {
        method: {time_key(row["time_s"]): row for row in records}
        for method, records in records_by_method.items()
    }
    shared_keys = sorted(set.intersection(*(set(rows) for rows in maps.values())))
    if not shared_keys:
        raise RuntimeError(f"target {target} representative runs share no timestamps")
    first_method = methods[0]
    gps_map = {
        time_key(row["time_s"]): np.asarray(
            [float(row["px"]), float(row["py"]), float(row["pz"])], dtype=float
        )
        for row in fallback_truth
        if all(finite(row.get(key)) for key in ("time_s", "px", "py", "pz"))
    }
    truth = []
    retained_keys = []
    for key in shared_keys:
        row = maps[first_method][key]
        if all(finite(row.get(name)) for name in ("truth_x", "truth_y", "truth_z")):
            truth.append([float(row["truth_x"]), float(row["truth_y"]), float(row["truth_z"])])
            retained_keys.append(key)
        elif key in gps_map:
            truth.append(gps_map[key].tolist())
            retained_keys.append(key)
    if not retained_keys:
        raise RuntimeError(f"target {target} has no aligned truth for representative seed {seed}")
    method_xyz = {
        method: np.asarray(
            [
                [
                    float(maps[method][key]["px"]),
                    float(maps[method][key]["py"]),
                    float(maps[method][key]["pz"]),
                ]
                for key in retained_keys
            ],
            dtype=float,
        )
        for method in methods
    }
    method_rows = {
        method: [maps[method][key] for key in retained_keys]
        for method in methods
    }
    return {
        "target": target,
        "seed": seed,
        "times": np.asarray(retained_keys, dtype=float),
        "truth": np.asarray(truth, dtype=float),
        "method_xyz": method_xyz,
        "method_rows": method_rows,
    }


def contiguous_spans(times: np.ndarray, records: list[dict]) -> list[np.ndarray]:
    """Split accepted tracks at rejected frames, segment changes or time gaps."""
    if len(times) != len(records):
        raise ValueError("time/record length mismatch")
    if len(times) == 0:
        return []
    accepted = np.asarray(
        [bool(row.get("accepted_acoustic_frame", True)) for row in records], dtype=bool
    )
    segment_ids = np.asarray(
        [int(row.get("observation_segment_id", 0)) for row in records], dtype=int
    )
    positive_dt = np.diff(times)
    positive_dt = positive_dt[positive_dt > 0]
    typical_dt = float(np.median(positive_dt)) if len(positive_dt) else 1.0
    maximum_dt = max(1.5 * typical_dt, typical_dt + 1.0e-6)
    spans: list[np.ndarray] = []
    current: list[int] = []
    for index in range(len(times)):
        if not accepted[index]:
            if current:
                spans.append(np.asarray(current, dtype=int))
                current = []
            continue
        if current:
            previous = current[-1]
            discontinuous = (
                segment_ids[index] != segment_ids[previous]
                or times[index] - times[previous] > maximum_dt
                or times[index] <= times[previous]
            )
            if discontinuous:
                spans.append(np.asarray(current, dtype=int))
                current = []
        current.append(index)
    if current:
        spans.append(np.asarray(current, dtype=int))
    return spans


def method_color(method: str, index: int) -> str:
    return METHOD_COLORS.get(method, FALLBACK_COLORS[index % len(FALLBACK_COLORS)])


def target_linestyle(index: int):
    return TARGET_LINESTYLES[index % len(TARGET_LINESTYLES)]


def bootstrap_ci(values: np.ndarray, seed: int, samples: int = 5000) -> tuple[float, float]:
    if len(values) <= 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def stripplot(
    axis: plt.Axes,
    summaries: list[dict],
    targets: list[int],
    methods: tuple[str, ...],
    metric: str,
    title: str,
    xlabel: str,
    show_labels: bool,
) -> None:
    categories = [(target, method) for target in targets for method in methods]
    for category_index, (target, method) in enumerate(categories):
        values = np.asarray(
            [
                float(row[metric])
                for row in summaries
                if int(row["target"]) == target
                and row["method"] == method
                and finite(row.get(metric))
            ],
            dtype=float,
        )
        if not len(values):
            continue
        rng = np.random.default_rng(20260821 + 101 * target + 17 * category_index + len(metric))
        jitter = rng.normal(0.0, 0.045, size=len(values))
        color = method_color(method, methods.index(method))
        marker = TARGET_MARKERS[(target - 1) % len(TARGET_MARKERS)]
        axis.scatter(
            values,
            category_index + jitter,
            s=14,
            alpha=0.45,
            color=color,
            marker=marker,
            linewidths=0,
            zorder=2,
        )
        mean = float(values.mean())
        lo, hi = bootstrap_ci(
            values, 20260821 + 1009 * target + 53 * category_index + len(metric)
        )
        axis.errorbar(
            mean,
            category_index,
            xerr=np.asarray([[mean - lo], [hi - mean]]),
            fmt=marker,
            ms=4.1,
            color=color,
            capsize=2.0,
            lw=1.1,
            zorder=3,
        )
    axis.set_yticks(range(len(categories)))
    if show_labels:
        axis.set_yticklabels(
            [f"T{target} {METHOD_LABELS.get(method, method)}" for target, method in categories]
        )
    else:
        axis.set_yticklabels([])
        axis.tick_params(axis="y", length=0)
    axis.invert_yaxis()
    axis.grid(axis="x", color="#DCE2E6", lw=0.5, zorder=0)
    axis.set_axisbelow(True)
    axis.set_xlabel(xlabel)
    axis.set_title(title, loc="left", fontsize=7.0, weight="bold", pad=3)
    axis.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:g}"))


def source_count_label(count: int) -> str:
    return {
        1: "single-source",
        2: "dual-source",
        3: "triple-source",
    }.get(count, f"{count}-source")


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return slug or "baoding_multisource"


def plot_figure(
    trajectories: list[dict],
    summaries: list[dict],
    methods: tuple[str, ...],
    representative: dict,
    scenario_label: str,
    claim_status: str,
    output: Path,
) -> tuple[Path, list[dict], dict]:
    setup_style()
    targets = [int(item["target"]) for item in trajectories]
    count_label = source_count_label(len(targets))
    height = 4.20 if len(targets) <= 2 else 4.55
    fig = plt.figure(figsize=(7.25, height), constrained_layout=True)
    outer = fig.add_gridspec(1, 2, width_ratios=(1.15, 1.0), wspace=0.18)
    ax3d = fig.add_subplot(outer[0], projection="3d")
    right = outer[1].subgridspec(2, 2, wspace=0.36, hspace=0.44)
    stat_axes = [
        fig.add_subplot(right[0, 0]),
        fig.add_subplot(right[0, 1]),
        fig.add_subplot(right[1, 0]),
        fig.add_subplot(right[1, 1]),
    ]

    all_arrays = []
    source_rows: list[dict] = []
    gap_counts: dict[str, int] = {}
    for target_index, trajectory in enumerate(trajectories):
        target = int(trajectory["target"])
        linestyle = target_linestyle(target_index)
        truth_km = trajectory["truth"] / 1000.0
        all_arrays.append(truth_km)
        ax3d.plot(
            truth_km[:, 0],
            truth_km[:, 1],
            truth_km[:, 2],
            color=METHOD_COLORS["truth"],
            lw=1.8,
            ls=linestyle,
            zorder=4,
        )
        ax3d.scatter(*truth_km[0], color=METHOD_COLORS["truth"], marker="o", s=18, zorder=5)
        ax3d.scatter(*truth_km[-1], color=METHOD_COLORS["truth"], marker="s", s=19, zorder=5)
        for index, time_s in enumerate(trajectory["times"]):
            source_rows.append(
                {
                    "panel": "a",
                    "scenario": scenario_label,
                    "target": target,
                    "seed": int(representative["seed"]),
                    "series": "GPS truth",
                    "time_s": float(time_s),
                    "east_km": float(truth_km[index, 0]),
                    "north_km": float(truth_km[index, 1]),
                    "up_km": float(truth_km[index, 2]),
                }
            )
        for method_index, method in enumerate(methods):
            xyz_km = trajectory["method_xyz"][method] / 1000.0
            all_arrays.append(xyz_km)
            spans = contiguous_spans(trajectory["times"], trajectory["method_rows"][method])
            gap_counts[f"target{target}_{method}"] = max(len(spans) - 1, 0)
            for span in spans:
                ax3d.plot(
                    xyz_km[span, 0],
                    xyz_km[span, 1],
                    xyz_km[span, 2],
                    color=method_color(method, method_index),
                    lw=1.1,
                    ls=linestyle,
                    zorder=3 - min(method_index, 1),
                )
            for index, (time_s, row) in enumerate(
                zip(trajectory["times"], trajectory["method_rows"][method])
            ):
                source_rows.append(
                    {
                        "panel": "a",
                        "scenario": scenario_label,
                        "target": target,
                        "seed": int(representative["seed"]),
                        "series": METHOD_LABELS.get(method, method),
                        "method": method,
                        "time_s": float(time_s),
                        "east_km": float(xyz_km[index, 0]),
                        "north_km": float(xyz_km[index, 1]),
                        "up_km": float(xyz_km[index, 2]),
                        "accepted_acoustic_frame": bool(row.get("accepted_acoustic_frame", True)),
                        "observation_segment_id": int(row.get("observation_segment_id", 0)),
                    }
                )

    all_xyz = np.vstack(all_arrays)
    ax3d.set_xlabel("East offset (km)", labelpad=2)
    ax3d.set_ylabel("North offset (km)", labelpad=2)
    ax3d.set_zlabel("Up offset (km)", labelpad=0)
    ax3d.tick_params(labelsize=5.5, pad=0)
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.set_major_formatter(mticker.StrMethodFormatter("{x:g}"))
    ax3d.view_init(elev=24, azim=-58)
    span = np.ptp(all_xyz, axis=0)
    ax3d.set_box_aspect(tuple(np.maximum(span, 0.001)))
    z_min, z_max = float(np.min(all_xyz[:, 2])), float(np.max(all_xyz[:, 2]))
    if z_max > z_min:
        ax3d.set_zticks(np.linspace(z_min, z_max, 4))
    ax3d.set_title(
        f"{count_label.capitalize()} associated trajectories",
        loc="left",
        fontsize=7.6,
        weight="bold",
        pad=4,
    )

    method_handles = [
        Line2D([0], [0], color=METHOD_COLORS["truth"], lw=1.8, label="GPS truth")
    ] + [
        Line2D(
            [0], [0], color=method_color(method, index), lw=1.15,
            label=METHOD_LABELS.get(method, method)
        )
        for index, method in enumerate(methods)
    ]
    method_handles.extend(
        [
            Line2D([0], [0], color=METHOD_COLORS["truth"], marker="o", lw=0, ms=4, label="start"),
            Line2D([0], [0], color=METHOD_COLORS["truth"], marker="s", lw=0, ms=4, label="end"),
        ]
    )
    method_legend = ax3d.legend(
        handles=method_handles,
        loc="upper left",
        bbox_to_anchor=(-0.03, 0.98),
        fontsize=5.5,
        handlelength=2.3,
    )
    ax3d.add_artist(method_legend)
    if len(targets) > 1:
        target_handles = [
            Line2D(
                [0], [0], color="#59636B", lw=1.25,
                ls=target_linestyle(index), label=f"target {target}"
            )
            for index, target in enumerate(targets)
        ]
        ax3d.legend(
            handles=target_handles,
            loc="lower left",
            bbox_to_anchor=(-0.03, 0.02),
            fontsize=5.4,
            handlelength=2.5,
        )

    ax_in = inset_axes(ax3d, width="32%", height="25%", loc="upper right", borderpad=0.78)
    ax_in.set_facecolor((1, 1, 1, 0.92))
    for target_index, trajectory in enumerate(trajectories):
        linestyle = target_linestyle(target_index)
        truth_km = trajectory["truth"] / 1000.0
        ax_in.plot(truth_km[:, 0], truth_km[:, 1], color=METHOD_COLORS["truth"], lw=1.05, ls=linestyle)
        for method_index, method in enumerate(methods):
            xyz_km = trajectory["method_xyz"][method] / 1000.0
            for indices in contiguous_spans(trajectory["times"], trajectory["method_rows"][method]):
                ax_in.plot(
                    xyz_km[indices, 0], xyz_km[indices, 1],
                    color=method_color(method, method_index), lw=0.8, ls=linestyle
                )
    ax_in.set_aspect("equal", adjustable="box")
    ax_in.set_title("horizontal view", fontsize=5.3, loc="left", pad=1)
    ax_in.tick_params(labelsize=4.8, length=1.6)
    ax_in.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:g}"))
    ax_in.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:g}"))
    ax_in.grid(color="#E5EAEE", lw=0.35)

    stripplot(stat_axes[0], summaries, targets, methods, "nrmse_3d", "nRMSE", "dimensionless", True)
    stripplot(stat_axes[1], summaries, targets, methods, "position_crps_m", "Position CRPS", "m", False)
    stripplot(
        stat_axes[2], summaries, targets, methods, "coverage_error_90",
        "90% coverage error", r"$|\mathrm{coverage}-0.90|$", True
    )
    stripplot(stat_axes[3], summaries, targets, methods, "interval_width_m", "90% interval width", "m", False)

    fig.text(0.006, 0.990, "a", ha="left", va="top", fontsize=9.5, fontweight="bold")
    fig.text(0.515, 0.990, "b", ha="left", va="top", fontsize=9.5, fontweight="bold")
    status_text = "gate-admitted" if claim_status == "admitted" else "inspection only"
    fig.text(
        0.012,
        1.004,
        (
            f"{scenario_label}: panel a uses representative seed {representative['seed']}; "
            f"panel b shows {len({int(row['seed']) for row in summaries})} paired numerical-sensitivity seeds "
            f"per target/method ({status_text})."
        ),
        ha="left",
        va="top",
        fontsize=5.8,
        color="#4D5961",
    )

    output.mkdir(parents=True, exist_ok=True)
    stem = output / f"baoding_{slugify(scenario_label)}_{len(targets)}source_trajectory_ab"
    exports = {}
    for suffix in (".svg", ".pdf", ".png", ".tiff"):
        path = stem.with_suffix(suffix)
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.025}
        if suffix in (".png", ".tiff"):
            kwargs["dpi"] = 600 if suffix == ".tiff" else 300
        fig.savefig(path, **kwargs)
        exports[suffix[1:]] = str(path)
    plt.close(fig)
    return stem, source_rows, {"exports": exports, "gap_counts": gap_counts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario-label")
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--representative-seed", type=int)
    parser.add_argument("--expected-targets", type=int)
    parser.add_argument("--gate-json", type=Path)
    parser.add_argument("--remote-source-root")
    parser.add_argument("--claim-status", choices=("inspection", "admitted"), default="inspection")
    args = parser.parse_args()

    result_root = args.result_root.resolve()
    output = args.output.resolve()
    methods = tuple(dict.fromkeys(args.methods))
    if not methods:
        raise SystemExit("at least one method is required")
    targets = discover_targets(result_root)
    if args.expected_targets is not None and len(targets) != args.expected_targets:
        raise RuntimeError(f"expected {args.expected_targets} targets, found {len(targets)}")
    scenario_label = args.scenario_label or result_root.name
    seeds, files_by_target = common_seeds(targets, methods)
    summaries, payloads, truth_by_target = load_all_runs(targets, methods, seeds, files_by_target)
    representative = choose_representative_seed(summaries, seeds, args.representative_seed)
    trajectories = [
        aligned_trajectory(
            target,
            int(representative["seed"]),
            methods,
            payloads,
            truth_by_target[target],
        )
        for target in targets
    ]
    stem, trajectory_rows, figure_meta = plot_figure(
        trajectories,
        summaries,
        methods,
        representative,
        scenario_label,
        args.claim_status,
        output,
    )

    trajectory_source = stem.with_name(stem.name + "_trajectory_source.csv")
    statistics_source = stem.with_name(stem.name + "_statistics_source.csv")
    combined_source = stem.with_name(stem.name + "_source.csv")
    write_csv(trajectory_source, trajectory_rows)
    write_csv(statistics_source, summaries)
    write_csv(combined_source, trajectory_rows + summaries)

    gate = None
    if args.gate_json is not None:
        gate_path = args.gate_json.resolve()
        if not gate_path.is_file():
            raise RuntimeError(f"gate JSON does not exist: {gate_path}")
        gate = {"path": str(gate_path), "sha256": sha256(gate_path)}
    registry = {
        "figure": stem.name,
        "claim_status": args.claim_status,
        "claim": (
            "Frozen target-labelled acoustic associations yield auditable multi-target PCE/APCE trajectories and "
            "target-stratified numerical-sensitivity summaries."
            if args.claim_status == "admitted"
            else "Inspection-only visualization of frozen target-labelled multi-source trajectories; no superiority claim."
        ),
        "backend": "python/matplotlib",
        "archetype": "asymmetric a/b mixed-modality trajectory figure",
        "scenario": scenario_label,
        "source_count": len(targets),
        "targets": list(targets),
        "methods": list(methods),
        "paired_seeds": seeds,
        "representative_seed": representative,
        "panels": {
            "a": {
                "conclusion": "Representative target-labelled 3-D tracks in centered local ENU coordinates.",
                "source": str(trajectory_source),
                "coordinate_definition": "centered local ENU in source metres; displayed in kilometres",
                "encoding": "method by colour; target identity by line style; circle=start; square=end",
                "gap_policy": "method trajectories are split at rejected frames, segment changes and time discontinuities; no interpolation",
                "gap_counts": figure_meta["gap_counts"],
            },
            "b": {
                "conclusion": "Target-stratified numerical sensitivity over the paired seed set.",
                "source": str(statistics_source),
                "metrics": ["nrmse_3d", "position_crps_m", "coverage_error_90", "interval_width_m"],
                "summary_mark": "individual seeds plus mean and seed-bootstrap 95% CI of the mean",
            },
        },
        "provenance": {
            "local_or_remote_result_root": str(result_root),
            "authoritative_remote_source_root": args.remote_source_root,
            "gate": gate,
        },
        "exports": figure_meta["exports"],
        "source_data": {
            "trajectory": str(trajectory_source),
            "statistics": str(statistics_source),
            "combined": str(combined_source),
        },
    }
    registry_path = stem.with_name(stem.name + "_registry.json")
    write_json(registry_path, registry)
    qa = {
        "figure": stem.name,
        "claim_status": args.claim_status,
        "checks": {
            "target_count_matches_request": args.expected_targets is None or len(targets) == args.expected_targets,
            "contiguous_target_ids": list(targets) == list(range(1, len(targets) + 1)),
            "common_paired_seed_count": len(seeds),
            "complete_target_method_seed_matrix": len(summaries) == len(targets) * len(methods) * len(seeds),
            "finite_representative_coordinates": all(
                np.isfinite(trajectory["truth"]).all()
                and all(np.isfinite(values).all() for values in trajectory["method_xyz"].values())
                for trajectory in trajectories
            ),
            "panel_labels": ["a", "b"],
            "local_enu_axes_in_km": True,
            "no_scientific_axis_offset": True,
            "no_trajectory_interpolation": True,
            "gate_provenance_supplied": gate is not None,
            "authoritative_remote_source_supplied": bool(args.remote_source_root),
        },
        "representative_seed": representative,
        "exports": figure_meta["exports"],
        "source_data": registry["source_data"],
        "integrity_note": "The script reads frozen run files only and does not rerun tracking or edit exported scientific content.",
    }
    qa_path = stem.with_name(stem.name + "_qa.json")
    write_json(qa_path, qa)
    print(
        json.dumps(
            {
                "figure_pdf": figure_meta["exports"]["pdf"],
                "figure_png": figure_meta["exports"]["png"],
                "registry": str(registry_path),
                "qa": str(qa_path),
                "targets": list(targets),
                "paired_seeds": seeds,
                "representative_seed": representative,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
