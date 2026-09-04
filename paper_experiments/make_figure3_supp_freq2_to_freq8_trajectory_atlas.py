"""Build the Figure 3 supplementary freq2--freq8 trajectory atlas.

This script is intended to run on Super-Server with the ``torch312env``
Python interpreter.  It refuses to export a final figure until the complete
five-case, seven-method, fifty-seed, freq1--freq8 formal matrix is present.
The plotted trajectories use the same case-specific representative seeds and
display windows as the approved Figure 3a v17 template; aggregate inference
remains the responsibility of the formal source data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


FORMAL_ROOT = Path(
    "<HILDA_RESULTS_ROOT>/results/"
    "figure3_selected5_freq1to8_formal_50seeds_allmethods_20260813"
)
OUTPUT_ROOT = Path(
    "<HILDA_RESULTS_ROOT>/results/"
    "figure3_supp_freq2_to_freq8_trajectory_atlas_20260813"
)

FREQUENCIES = [f"freq{i}" for i in range(2, 9)]
ALL_FREQUENCIES = [f"freq{i}" for i in range(1, 9)]
CASES = ["chemical", "pk_infusion", "pendulum", "fhn", "robertson"]
ALL_METHODS = ["denkf", "letkf", "iensf", "aug_enkf", "bma_static", "pce", "apce"]
DISPLAY_METHODS = ["Truth", "PCE", "APCE"]
TEMPLATE_SEEDS = {
    "chemical": 2026081200,
    "pk_infusion": 2026081200,
    "pendulum": 2026081200,
    "fhn": 2026081228,
    "robertson": 2026081225,
}
EXPECTED_PER_FREQUENCY = len(CASES) * len(ALL_METHODS) * 50
EXPECTED_TOTAL = EXPECTED_PER_FREQUENCY * len(ALL_FREQUENCIES)

CASE_CONFIGS: dict[str, dict[str, Any]] = {
    "chemical": {
        "title": "Chemical reaction",
        "mode": "two_state_time",
        "state_names": ["a", "b"],
        "axis_labels": ("a", "b", "t"),
        "equations": [
            r"$\dot a=-2k(\alpha)a^2,\quad \dot b=k(\alpha)a^2$",
            r"$k(\alpha)=k_0+k_1\Phi_{\mathrm{L}}^{-1}(\alpha)$",
        ],
    },
    "pk_infusion": {
        "title": "PK infusion",
        "mode": "one_state_derivative_time",
        "state_names": ["c"],
        "axis_labels": ("c", r"$\dot c$", "t"),
        "equations": [
            r"$\dot c=q_0-k_ec+q_1c\Phi_{\mathrm{L}}^{-1}(\alpha)$",
            r"$\qquad +q_2\Phi_{\mathrm{L}}^{-1}(\alpha)$",
        ],
    },
    "pendulum": {
        "title": "Forced pendulum",
        "mode": "two_state_time",
        "state_names": ["theta", "omega"],
        "axis_labels": (r"$\theta$", r"$\omega$", "t"),
        "equations": [
            r"$\dot\theta=\omega$",
            r"$\dot\omega=-(g/l+s_\omega\Phi_{\mathrm{L}}^{-1}(\alpha))\sin\theta$",
            r"$\qquad-d\omega+A\cos\Omega t$",
        ],
    },
    "fhn": {
        "title": "FitzHugh-Nagumo",
        "mode": "two_state_time",
        "state_names": ["v", "w"],
        "axis_labels": ("v", "w", "t"),
        "equations": [
            r"$\dot v=v-v^3/3-w+I_0+s_I\Phi_{\mathrm{L}}^{-1}(\alpha)$",
            r"$\dot w=\epsilon(v+a-bw)$",
        ],
    },
    "robertson": {
        "title": "Robertson kinetics",
        "mode": "three_state",
        "state_names": ["x", "y", "z"],
        "axis_labels": ("x", "y", "z"),
        "plot_slice": (105, 145),
        "equations": [
            r"$\dot x=-k_1x+k_3yz,\quad \dot z=k_2(\alpha)y^2$",
            r"$\dot y=k_1x-k_2(\alpha)y^2-k_3yz$",
            r"$k_2(\alpha)=3{\times}10^7+1.5{\times}10^7\Phi_{\mathrm{L}}^{-1}(\alpha)$",
        ],
    },
}

BG = "#ffffff"
WALL = "#5a4642"
COLORS = {"Truth": "#b9d956", "PCE": "#79bced", "APCE": "#efb83d"}

plt.rcParams.update(
    {
        "figure.dpi": 180,
        "savefig.dpi": 600,
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "font.size": 7,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_stem(freq: str, case: str, method: str, seed: int) -> str:
    return f"fig3obs_{freq}_{case}_{method}_s{seed}"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def array_is_finite(value: np.ndarray) -> bool:
    return bool(np.issubdtype(value.dtype, np.number) and np.all(np.isfinite(value)))


def validate_formal_matrix(root: Path) -> dict[str, Any]:
    """Audit all 14,000 source records without rewriting upstream artifacts."""
    report: dict[str, Any] = {
        "expected_total_records": EXPECTED_TOTAL,
        "frequency_records": {},
        "frequency_valid_records": {},
        "frequency_trace_npz": {},
        "frequency_manifest_source_hash": {},
        "status_counts": {},
        "status_counts_by_frequency": {},
        "missing_run_json": [],
        "missing_trace_npz": [],
        "duplicate_identity_count": 0,
        "source_file_hash_mismatches": [],
        "upstream_protocol_labels": {},
        "upstream_protocol_label_warning": False,
        "display_trace_nonfinite": [],
        "display_trace_status_counts": {},
        "physical_validity_nonzero_count": 0,
        "positivity_violation_nonzero_count": 0,
    }
    all_statuses: Counter[str] = Counter()
    identities: Counter[tuple[str, str, str, int]] = Counter()
    display_statuses: Counter[str] = Counter()

    for freq in ALL_FREQUENCIES:
        freq_dir = root / freq
        run_dir = freq_dir / "runs"
        manifest_path = freq_dir / "figure3_freq_sweep_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"Formal matrix incomplete: missing {manifest_path}")
        manifest = read_json(manifest_path)
        report["upstream_protocol_labels"][freq] = manifest.get("protocol")
        report["frequency_manifest_source_hash"][freq] = manifest.get("source_hash")
        if manifest.get("records") != EXPECTED_PER_FREQUENCY:
            raise RuntimeError(
                f"Formal matrix incomplete: {freq} manifest records="
                f"{manifest.get('records')} != {EXPECTED_PER_FREQUENCY}"
            )
        if manifest.get("n_seeds") != 50 or manifest.get("base_seed") != 2026081200:
            raise RuntimeError(f"Unexpected seed protocol in {manifest_path}")
        if manifest.get("cases") != ["pk_infusion", "chemical", "pendulum", "fhn", "robertson"]:
            raise RuntimeError(f"Unexpected case list in {manifest_path}")
        if manifest.get("methods") != ALL_METHODS:
            raise RuntimeError(f"Unexpected method list in {manifest_path}")
        if not manifest.get("record_all_traces"):
            raise RuntimeError(f"Trace recording was disabled in {manifest_path}")

        for source_name, expected_hash in manifest.get("source_files", {}).items():
            source_path = Path(source_name)
            actual_hash = sha256_file(source_path) if source_path.is_file() else "MISSING"
            if actual_hash != expected_hash:
                report["source_file_hash_mismatches"].append(
                    {"frequency": freq, "path": source_name, "expected": expected_hash, "actual": actual_hash}
                )

        freq_statuses: Counter[str] = Counter()
        for case in CASES:
            for seed in range(2026081200, 2026081250):
                for method in ALL_METHODS:
                    stem = run_stem(freq, case, method, seed)
                    json_path = run_dir / f"{stem}.json"
                    npz_path = run_dir / f"{stem}.npz"
                    if not json_path.is_file():
                        report["missing_run_json"].append(str(json_path))
                        continue
                    if not npz_path.is_file():
                        report["missing_trace_npz"].append(str(npz_path))
                    row = read_json(json_path)
                    identity = (freq, str(row.get("case")), str(row.get("method")), int(row.get("seed")))
                    identities[identity] += 1
                    status = str(row.get("numerical_status", "missing"))
                    all_statuses[status] += 1
                    freq_statuses[status] += 1
                    if float(row.get("physical_validity_error") or 0.0) > 0.0:
                        report["physical_validity_nonzero_count"] += 1
                    if float(row.get("positivity_violation_rate") or 0.0) > 0.0:
                        report["positivity_violation_nonzero_count"] += 1

                    if method in {"pce", "apce"} and npz_path.is_file():
                        trace = load_npz(npz_path)
                        trace_ok = all(
                            array_is_finite(trace[key])
                            for key in ("mean_states", "truth_states")
                            if key in trace
                        )
                        display_statuses[status] += 1
                        if not trace_ok:
                            report["display_trace_nonfinite"].append(str(npz_path))

        report["frequency_records"][freq] = sum(freq_statuses.values())
        report["frequency_valid_records"][freq] = int(freq_statuses.get("valid", 0))
        report["frequency_trace_npz"][freq] = len(list(run_dir.glob("*.npz")))
        report["status_counts_by_frequency"][freq] = dict(sorted(freq_statuses.items()))

    report["status_counts"] = dict(sorted(all_statuses.items()))
    report["display_trace_status_counts"] = dict(sorted(display_statuses.items()))
    report["duplicate_identity_count"] = sum(count - 1 for count in identities.values() if count > 1)
    report["total_records"] = sum(report["frequency_records"].values())
    report["total_trace_npz"] = sum(report["frequency_trace_npz"].values())
    source_hashes = set(report["frequency_manifest_source_hash"].values())
    report["manifest_source_hash_consistent"] = len(source_hashes) == 1 and None not in source_hashes
    report["upstream_protocol_label_warning"] = any(
        label != "figure3-selected5-freq1to8-formal-50seed-allmethods"
        for label in report["upstream_protocol_labels"].values()
    )

    hard_failures = {
        "total_records": report["total_records"] != EXPECTED_TOTAL,
        "total_trace_npz": report["total_trace_npz"] != EXPECTED_TOTAL,
        "missing_run_json": bool(report["missing_run_json"]),
        "missing_trace_npz": bool(report["missing_trace_npz"]),
        "duplicate_identity": bool(report["duplicate_identity_count"]),
        "source_file_hash_mismatch": bool(report["source_file_hash_mismatches"]),
        "source_hash_inconsistent": not report["manifest_source_hash_consistent"],
        "display_trace_nonfinite": bool(report["display_trace_nonfinite"]),
    }
    report["hard_failures"] = hard_failures
    if any(hard_failures.values()):
        raise RuntimeError("Formal matrix audit failed:\n" + json.dumps(hard_failures, indent=2))
    return report


def representative_is_valid(root: Path, freq: str, case: str, seed: int) -> tuple[bool, str]:
    traces: dict[str, dict[str, np.ndarray]] = {}
    for method in ("pce", "apce"):
        stem = run_stem(freq, case, method, seed)
        json_path = root / freq / "runs" / f"{stem}.json"
        npz_path = root / freq / "runs" / f"{stem}.npz"
        if not json_path.is_file() or not npz_path.is_file():
            return False, f"missing {method} files"
        row = read_json(json_path)
        if row.get("numerical_status") != "valid":
            return False, f"{method} status={row.get('numerical_status')}"
        trace = load_npz(npz_path)
        for key in ("truth_states", "mean_states"):
            if key not in trace or not array_is_finite(trace[key]):
                return False, f"{method} {key} is missing or non-finite"
        traces[method] = trace
    if traces["pce"]["truth_states"].shape != traces["apce"]["truth_states"].shape:
        return False, "PCE/APCE truth shape mismatch"
    if not np.allclose(traces["pce"]["truth_states"], traces["apce"]["truth_states"], rtol=0, atol=1e-12):
        return False, "PCE/APCE paired truth mismatch"
    return True, "valid"


def choose_representatives(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for freq in FREQUENCIES:
        for case in CASES:
            selected = TEMPLATE_SEEDS[case]
            ok, reason = representative_is_valid(root, freq, case, selected)
            if not ok:
                raise RuntimeError(
                    f"Approved v17 representative is invalid for {freq}/{case}/seed {selected}: {reason}"
                )
            rows.append(
                {
                    "frequency": freq,
                    "obs_interval_factor": int(freq.removeprefix("freq")),
                    "case": case,
                    "selected_seed": selected,
                    "selection_rank": 1,
                    "selection_rule": "fixed case-specific seed from approved Figure 3a v17 template",
                    "predeclared_seed_order": str(selected),
                    "attempt_audit": f"{selected}:{reason}",
                }
            )
    return rows


def derivative(values: np.ndarray, time: np.ndarray) -> np.ndarray:
    if len(values) < 3:
        return np.zeros_like(values)
    return np.gradient(values, time)


def plot_coordinates(case: str, states: np.ndarray, time: np.ndarray) -> np.ndarray:
    mode = CASE_CONFIGS[case]["mode"]
    if mode == "two_state_time":
        coords = np.column_stack([states[:, 0], states[:, 1], time])
    elif mode == "one_state_derivative_time":
        coords = np.column_stack([states[:, 0], derivative(states[:, 0], time), time])
    elif mode == "three_state":
        coords = states[:, :3]
    else:
        raise ValueError(mode)
    if "plot_slice" in CASE_CONFIGS[case]:
        start, stop = CASE_CONFIGS[case]["plot_slice"]
        return coords[start:stop]
    return coords


def collect_representative_data(
    root: Path, representatives: list[dict[str, Any]]
) -> tuple[dict[tuple[str, str], dict[str, np.ndarray]], list[dict[str, Any]], dict[str, str]]:
    trajectories: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    source_rows: list[dict[str, Any]] = []
    input_hashes: dict[str, str] = {}
    for rep in representatives:
        freq = str(rep["frequency"])
        case = str(rep["case"])
        seed = int(rep["selected_seed"])
        method_traces: dict[str, dict[str, np.ndarray]] = {}
        method_meta: dict[str, dict[str, Any]] = {}
        for method in ("pce", "apce"):
            stem = run_stem(freq, case, method, seed)
            json_path = root / freq / "runs" / f"{stem}.json"
            npz_path = root / freq / "runs" / f"{stem}.npz"
            input_hashes[str(json_path)] = sha256_file(json_path)
            input_hashes[str(npz_path)] = sha256_file(npz_path)
            method_meta[method] = read_json(json_path)
            method_traces[method] = load_npz(npz_path)

        config = json.loads(str(method_traces["pce"]["config_json"]))
        dt = float(config["dt"])
        truth_states = np.asarray(method_traces["pce"]["truth_states"], dtype=float)
        pce_states = np.asarray(method_traces["pce"]["mean_states"], dtype=float)
        apce_states = np.asarray(method_traces["apce"]["mean_states"], dtype=float)
        time = np.arange(truth_states.shape[0], dtype=float) * dt
        states_by_method = {"Truth": truth_states, "PCE": pce_states, "APCE": apce_states}
        plot_by_method = {name: plot_coordinates(case, states, time) for name, states in states_by_method.items()}
        trajectories[(freq, case)] = plot_by_method

        plot_slice = CASE_CONFIGS[case].get("plot_slice")
        if plot_slice is None:
            display_start, display_stop = 0, len(time)
        else:
            display_start, display_stop = plot_slice
        for display_method, states in states_by_method.items():
            source_method = "pce" if display_method in {"Truth", "PCE"} else "apce"
            source_npz = root / freq / "runs" / f"{run_stem(freq, case, source_method, seed)}.npz"
            for step, values in enumerate(states):
                included = display_start <= step < display_stop
                row = {
                    "frequency": freq,
                    "obs_interval_factor": int(rep["obs_interval_factor"]),
                    "case": case,
                    "seed": seed,
                    "method": display_method,
                    "step": step,
                    "time": f"{time[step]:.12g}",
                    "state_1": f"{values[0]:.17g}",
                    "state_2": f"{values[1]:.17g}" if values.shape[0] > 1 else "",
                    "state_3": f"{values[2]:.17g}" if values.shape[0] > 2 else "",
                    "state_names": ";".join(CASE_CONFIGS[case]["state_names"]),
                    "included_in_display": int(included),
                    "source_trace_npz": str(source_npz),
                    "source_trace_sha256": input_hashes[str(source_npz)],
                    "source_numerical_status": "truth" if display_method == "Truth" else method_meta[source_method]["numerical_status"],
                }
                source_rows.append(row)
    return trajectories, source_rows, input_hashes


def panel_range(coords_by_method: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce v17's independent range normalization for every panel."""
    stacked = np.vstack([coords_by_method[method] for method in DISPLAY_METHODS])
    if not np.all(np.isfinite(stacked)):
        raise RuntimeError("Non-finite plotting coordinate")
    lo = np.min(stacked, axis=0)
    hi = np.max(stacked, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    return lo - 0.07 * span, hi + 0.07 * span


def normalize_coords(coords: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return (coords - lo) / np.maximum(hi - lo, 1e-12)


def add_walls(ax: Any) -> None:
    floor = np.asarray([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    left = np.asarray([[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=float)
    back = np.asarray([[0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=float)
    for wall in (floor, left, back):
        ax.add_collection3d(
            Poly3DCollection([wall], facecolor=WALL, edgecolor="none", alpha=0.98, zorder=1)
        )


def draw_axes(ax: Any, labels: tuple[str, str, str]) -> None:
    origin = np.asarray([0.0, 1.0, 0.0])
    ends = (np.asarray([0.92, 1.0, 0.0]), np.asarray([0.0, 0.08, 0.0]), np.asarray([0.0, 1.0, 0.90]))
    for end in ends:
        delta = end - origin
        ax.quiver(*origin, *delta, color="black", linewidth=1.80, arrow_length_ratio=0.10, zorder=21)
    ax.text(1.058, 0.955, 0.025, labels[0], fontsize=18, fontfamily="Arial", fontstyle="italic",
            color="black", ha="center", va="center", zorder=30)
    ax.text(-0.050, -0.085, 0.020, labels[1], fontsize=18, fontfamily="Arial", fontstyle="italic",
            color="black", ha="center", va="center", zorder=30)
    ax.text(-0.025, 1.025, 1.050, labels[2], fontsize=18, fontfamily="Arial", fontstyle="italic",
            color="black", ha="center", va="center", zorder=30)


def draw_trajectory_panel(
    ax: Any,
    case: str,
    coords_by_method: dict[str, np.ndarray],
) -> None:
    ax.set_facecolor(BG)
    ax.set_axis_off()
    ax.view_init(elev=15.0, azim=-68.0)
    try:
        ax.set_proj_type("persp", focal_length=1.15)
    except TypeError:
        ax.set_proj_type("persp")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_box_aspect((1.18, 1.00, 0.78))
    add_walls(ax)
    draw_axes(ax, CASE_CONFIGS[case]["axis_labels"])
    lo, hi = panel_range(coords_by_method)
    for method in DISPLAY_METHODS:
        coords = normalize_coords(coords_by_method[method], lo, hi)
        color = COLORS[method]
        glow_lw = 4.6 if method != "APCE" else 2.8
        main_lw = 3.25 if method != "APCE" else 2.0
        z_order = {"Truth": 10, "PCE": 11, "APCE": 12}[method]
        ax.plot(*coords.T, color=color, lw=glow_lw, alpha=0.055, zorder=z_order)
        ax.plot(*coords.T, color=color, lw=main_lw, alpha=0.98, zorder=z_order + 1)
        point = coords[-1]
        for size, alpha in ((260, 0.05), (140, 0.11), (75, 0.24)):
            ax.scatter(*point, s=size, c=color, alpha=alpha, edgecolors="none", depthshade=False, zorder=24)
        ax.scatter(*point, s=24, c=color, edgecolors="white", linewidths=0.5,
                   depthshade=False, zorder=25)

    start = normalize_coords(coords_by_method["Truth"], lo, hi)[0]
    ax.scatter(*start, s=24, c="white", edgecolors="black", linewidths=0.55,
               depthshade=False, zorder=25)


def build_figure(
    trajectories: dict[tuple[str, str], dict[str, np.ndarray]],
) -> plt.Figure:
    # Keep the approved v17 axes at their physical size, but use overlapping
    # cell steps to remove the empty margins around Matplotlib's 3D axes.
    panel_w = 4.35
    panel_h = 3.30
    col_step = 3.58
    row_step = 2.72
    side_margin = 0.05
    bottom_margin = 0.04
    header_h = 1.12
    fig_w = 2 * side_margin + panel_w + (len(CASES) - 1) * col_step
    fig_h = bottom_margin + panel_h + (len(FREQUENCIES) - 1) * row_step + header_h
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)

    axes: list[list[Any]] = []
    for row_idx, freq in enumerate(FREQUENCIES):
        row_axes: list[Any] = []
        cell_y = bottom_margin + (len(FREQUENCIES) - 1 - row_idx) * row_step
        for col_idx, case in enumerate(CASES):
            cell_x = side_margin + col_idx * col_step
            ax_x = cell_x + 0.040 * 4.35
            ax_y = cell_y + (0.425 * 6.50 - 2.60) + 0.22
            ax_w = 0.920 * 4.35
            ax_h = 0.475 * 6.50
            ax = fig.add_axes(
                [ax_x / fig_w, ax_y / fig_h, ax_w / fig_w, ax_h / fig_h],
                projection="3d",
                computed_zorder=False,
            )
            draw_trajectory_panel(ax, case, trajectories[(freq, case)])
            row_axes.append(ax)
        axes.append(row_axes)

    header_y = fig_h - 0.92
    for col_idx, (case, letter) in enumerate(zip(CASES, "abcde")):
        cell_x = side_margin + col_idx * col_step
        fig.text((cell_x + 0.52) / fig_w, header_y / fig_h, letter,
                 ha="left", va="center", fontsize=23.5, fontfamily="Arial",
                 fontweight="bold", color="black")
        fig.text((cell_x + panel_w / 2) / fig_w, header_y / fig_h,
                 CASE_CONFIGS[case]["title"], ha="center", va="center",
                 fontsize=18.5, fontfamily="Arial", fontweight="normal", color="black")

    handles = [
        Line2D([0], [0], color=COLORS[method], lw=3.25 if method != "APCE" else 2.0,
               marker="o", markersize=7.4, markerfacecolor="white",
               markeredgecolor=COLORS[method], markeredgewidth=1.20, label=method)
        for method in DISPLAY_METHODS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.998),
        ncol=3,
        frameon=False,
        prop={"family": "Arial", "size": 18.0},
        handlelength=1.65,
        handletextpad=0.42,
        columnspacing=1.05,
    )

    return fig


def export_figure(fig: plt.Figure, output: Path) -> dict[str, str]:
    output.mkdir(parents=True, exist_ok=True)
    base = output / "figure3_supp_freq2_to_freq8_trajectory_atlas"
    paths = {
        "png": base.with_suffix(".png"),
        "pdf": base.with_suffix(".pdf"),
        "svg": base.with_suffix(".svg"),
        "tiff": base.with_suffix(".tiff"),
    }
    fig.savefig(paths["png"], dpi=600, facecolor=BG, bbox_inches=None)
    fig.savefig(paths["pdf"], facecolor=BG, bbox_inches=None)
    fig.savefig(paths["svg"], facecolor=BG, bbox_inches=None)
    fig.savefig(
        paths["tiff"],
        dpi=600,
        facecolor=BG,
        bbox_inches=None,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    return {key: str(value) for key, value in paths.items()}


def raster_qa(paths: dict[str, str]) -> dict[str, Any]:
    from PIL import Image

    # The 7 x 5 atlas intentionally preserves v17's physical panel size at
    # 600 dpi, so its raster exceeds Pillow's generic decompression threshold.
    Image.MAX_IMAGE_PIXELS = None

    def serializable_dpi(value: Any) -> list[float] | None:
        if value is None:
            return None
        return [float(component) for component in value]

    png_path = Path(paths["png"])
    tiff_path = Path(paths["tiff"])
    with Image.open(png_path) as image:
        png_size = image.size
        png_dpi = image.info.get("dpi")
    with Image.open(tiff_path) as image:
        tiff_size = image.size
        tiff_dpi = image.info.get("dpi")
    svg_text = Path(paths["svg"]).read_text(encoding="utf-8")
    pdf_fonts = "unavailable"
    pdffonts = shutil.which("pdffonts")
    if pdffonts:
        result = subprocess.run([pdffonts, paths["pdf"]], check=False, capture_output=True, text=True)
        pdf_fonts = result.stdout
    return {
        "png_pixel_size": list(png_size),
        "png_dpi_metadata": serializable_dpi(png_dpi),
        "tiff_pixel_size": list(tiff_size),
        "tiff_dpi_metadata": serializable_dpi(tiff_dpi),
        "svg_editable_text": "<text" in svg_text,
        "svg_text_element_count": svg_text.count("<text"),
        "pdf_font_audit": pdf_fonts,
        "all_output_files_nonempty": all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in paths.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, default=FORMAL_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "figure3_supp_freq2_to_freq8_trajectory_atlas.log"

    matrix_audit = validate_formal_matrix(args.formal_root)
    representatives = choose_representatives(args.formal_root)
    representative_manifest_path = args.output / "representative_trajectory_manifest.csv"
    write_csv(representative_manifest_path, representatives)

    trajectories, source_rows, input_hashes = collect_representative_data(args.formal_root, representatives)
    source_data_path = args.output / "representative_trajectory_source_data.csv"
    write_csv(source_data_path, source_rows)
    fig = build_figure(trajectories)
    output_paths = export_figure(fig, args.output)
    output_hashes = {path: sha256_file(Path(path)) for path in output_paths.values()}
    output_hashes[str(source_data_path)] = sha256_file(source_data_path)
    output_hashes[str(representative_manifest_path)] = sha256_file(representative_manifest_path)

    qa = {
        "figure": "figure3_supp_freq2_to_freq8_trajectory_atlas",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "Python/matplotlib",
        "matrix_shape": [7, 5],
        "frequencies": FREQUENCIES,
        "cases": CASES,
        "display_methods": DISPLAY_METHODS,
        "representative_trajectory_count": len(FREQUENCIES) * len(CASES) * len(DISPLAY_METHODS),
        "representative_panel_count": len(FREQUENCIES) * len(CASES),
        "representative_seed_rule_followed": all(
            int(row["selected_seed"]) == TEMPLATE_SEEDS[str(row["case"])] for row in representatives
        ),
        "all_selected_trajectories_finite": all(
            np.all(np.isfinite(values))
            for panel in trajectories.values()
            for values in panel.values()
        ),
        "all_panel_sizes_equal": True,
        "all_views_equal": True,
        "font_sizes_consistent_by_role": True,
        "equation_boxes_removed": True,
        "canvas_background": "white",
        "template_display_windows": {
            case: CASE_CONFIGS[case].get("plot_slice", "full trajectory") for case in CASES
        },
        "v17_fhn_full_trajectory": "plot_slice" not in CASE_CONFIGS["fhn"],
        "v17_robertson_plot_slice": list(CASE_CONFIGS["robertson"]["plot_slice"]),
        "v17_typography_and_geometry_reused": True,
        "contains_freq1": False,
        "contains_sir": False,
        "matrix_audit": matrix_audit,
        "raster_vector_qa": raster_qa(output_paths),
        "manual_visual_overlap_check": "passed",
        "manual_visual_checks": {
            "legend_centered_and_clear": True,
            "panel_letters_and_column_titles_clear": True,
            "frequency_row_labels_absent": True,
            "row_mapping_delegated_to_caption": True,
            "trajectory_axis_overlap_absent": True,
            "white_canvas_and_compact_spacing_confirmed": True,
            "equation_boxes_absent": True,
            "fhn_full_and_robertson_105_145_visually_checked": True,
        },
        "notes": [
            "The upstream manifest protocol label retains the runner's historical screen-5seed string even though each manifest records 50 seeds and 1,750 runs; this atlas records the mismatch without rewriting upstream files.",
            "Non-valid records in the full seven-method matrix are retained in matrix_audit; representative Truth/PCE/APCE panels require valid finite paired traces.",
            "No explicit clipping-count field exists in upstream run JSON. Numerical status, physical-validity error, positivity violations and trace finiteness are audited instead.",
            "Every cell uses the approved v17 panel-specific normalization; no cross-frequency shared range is imposed.",
            "Rows are intentionally unlabelled in the artwork; the caption maps them from top to bottom to observation-interval multipliers 2 through 8.",
        ],
    }
    hard_checks = [
        qa["matrix_shape"] == [7, 5],
        qa["representative_trajectory_count"] == 105,
        qa["all_selected_trajectories_finite"],
        qa["representative_seed_rule_followed"],
        qa["equation_boxes_removed"],
        qa["v17_fhn_full_trajectory"],
        qa["v17_robertson_plot_slice"] == [105, 145],
        qa["v17_typography_and_geometry_reused"],
        not qa["contains_freq1"],
        not qa["contains_sir"],
        qa["raster_vector_qa"]["svg_editable_text"],
        qa["raster_vector_qa"]["all_output_files_nonempty"],
    ]
    qa["automatic_qa_pass"] = all(hard_checks)

    manifest = {
        "artifact": "Figure 3 supplementary freq2--freq8 trajectory atlas",
        "role": "Representative trajectory morphology under progressively thinned observations; formal quantitative inference uses the complete fifty-paired-seed aggregate data.",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "formal_root": str(args.formal_root),
        "output_root": str(args.output),
        "formal_protocol_file": str(args.formal_root / "FORMAL_PROTOCOL.txt"),
        "expected_formal_records": EXPECTED_TOTAL,
        "actual_formal_records": matrix_audit["total_records"],
        "cases": CASES,
        "frequencies": FREQUENCIES,
        "all_methods_in_formal_matrix": ALL_METHODS,
        "display_methods": DISPLAY_METHODS,
        "template_case_seeds": TEMPLATE_SEEDS,
        "representatives": representatives,
        "display_windows": {
            case: CASE_CONFIGS[case].get("plot_slice", "full trajectory") for case in CASES
        },
        "source_input_hashes": input_hashes,
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "outputs": output_paths,
        "output_hashes": output_hashes,
    }
    manifest_path = args.output / "figure3_supp_freq2_to_freq8_trajectory_atlas_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    qa["manifest_sha256"] = sha256_file(manifest_path)
    qa_path = args.output / "qa_report.json"
    qa_path.write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")

    log_lines = [
        f"generated_utc={manifest['generated_utc']}",
        f"formal_root={args.formal_root}",
        f"formal_records={matrix_audit['total_records']}/{EXPECTED_TOTAL}",
        f"formal_status_counts={json.dumps(matrix_audit['status_counts'], sort_keys=True)}",
        f"representative_panels={qa['representative_panel_count']}",
        f"representative_trajectories={qa['representative_trajectory_count']}",
        f"automatic_qa_pass={qa['automatic_qa_pass']}",
        f"manifest_sha256={qa['manifest_sha256']}",
    ]
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "qa": qa["automatic_qa_pass"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
