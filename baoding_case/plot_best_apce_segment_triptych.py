#!/usr/bin/env python
"""Draw a one-row triptych for the Baoding single-turn APCE segment.

This script is intentionally display-only: it reads the already generated
segment-level files in the project tmp directory, writes a candidate figure
back to tmp, and does not modify the manuscript package.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TMP = ROOT / "tmp"

OPT_SOURCE = TMP / "baoding_nearfield_best_apce_segment_optimized_strong_source.csv"
OUT_BASE = TMP / "baoding_nearfield_best_apce_segment_triptych"

METHOD_FILES = {
    "DEnKF": TMP / "denkf_seed_2026082001.json",
    "Aug-EnKF": TMP / "aug_enkf_seed_2026082001.json",
    "PCE": TMP / "pce_seed_2026082001.json",
    "APCE": TMP / "apce_seed_2026082001.json",
}

METHOD_COLORS = {
    "DEnKF": "#9AA0A6",
    "Aug-EnKF": "#6F7D8C",
    "PCE": "#155A9C",
    "APCE": "#D55E00",
}


def _read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"empty source CSV: {path}")
    cols: dict[str, list[float]] = {}
    for row in rows:
        for k, v in row.items():
            if v is None or v == "":
                continue
            try:
                cols.setdefault(k, []).append(float(v))
            except ValueError:
                pass
    return {k: np.asarray(v, dtype=float) for k, v in cols.items()}


def _read_method_records(path: Path, segment_id: int = 3) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    records = [
        r for r in data.get("records", [])
        if int(r.get("observation_segment_id", -1)) == segment_id
    ]
    if not records:
        raise RuntimeError(f"no segment {segment_id} records in {path}")
    return records


def _records_to_arrays(records: list[dict]) -> dict[str, np.ndarray]:
    keys = [
        "time_s",
        "truth_x", "truth_y", "truth_z",
        "px", "py", "pz",
        "position_error_m",
        "crps_position_m",
    ]
    out = {}
    for k in keys:
        out[k] = np.asarray([float(r[k]) for r in records], dtype=float)
    return out


def _trajectory_scale(truth_xyz_m: np.ndarray) -> float:
    centred = truth_xyz_m - truth_xyz_m.mean(axis=0, keepdims=True)
    scale = float(np.sqrt(np.mean(np.sum(centred * centred, axis=1))))
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("invalid trajectory normalization scale")
    return scale


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E7E7E7", linewidth=0.55, zorder=0)
    ax.tick_params(axis="both", labelsize=6.8, width=0.7, length=2.5, pad=2)
    return ax


def _format_no_sci(ax):
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(mticker.StrMethodFormatter("{x:g}"))
        axis.get_offset_text().set_visible(False)


def _equalize_3d(ax, xs, ys, zs):
    xmid, ymid, zmid = map(float, [np.mean(xs), np.mean(ys), np.mean(zs)])
    span = max(float(np.ptp(xs)), float(np.ptp(ys)), float(np.ptp(zs)), 0.12)
    pad = span * 0.10
    half = span / 2 + pad
    ax.set_xlim(xmid - half, xmid + half)
    ax.set_ylim(ymid - half, ymid + half)
    z_half = max(float(np.ptp(zs)) / 2 + pad * 0.35, 0.08)
    ax.set_zlim(zmid - z_half, zmid + z_half)
    try:
        ax.set_box_aspect((1.0, 1.0, 0.44))
    except Exception:
        pass


def _panel_label(ax, label: str, x: float = -0.12, y: float = 1.03):
    text_kwargs = dict(
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        va="bottom",
        ha="left",
        color="#202020",
    )
    if hasattr(ax, "text2D"):
        ax.text2D(x, y, label, **text_kwargs)
    else:
        ax.text(x, y, label, **text_kwargs)


def main() -> None:
    # Mandatory publication export settings.
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 7
    plt.rcParams["axes.linewidth"] = 0.7
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["legend.frameon"] = False

    seg = _read_csv(OPT_SOURCE)
    time_rel = seg["time_rel_s"]
    truth = np.column_stack([seg["truth_x"], seg["truth_y"], seg["truth_z"]])
    apce_opt = np.column_stack([seg["apce_opt_x"], seg["apce_opt_y"], seg["apce_opt_z"]])

    method = {}
    reference_times = None
    for name, path in METHOD_FILES.items():
        arr = _records_to_arrays(_read_method_records(path))
        method[name] = arr
        if reference_times is None:
            reference_times = arr["time_s"]
        elif not np.array_equal(reference_times, arr["time_s"]):
            raise RuntimeError(f"time mismatch for {name}")

    if len(reference_times) != len(time_rel):
        raise RuntimeError("optimized segment and method records have different lengths")
    if not np.array_equal(reference_times, seg["time_s"].astype(reference_times.dtype)):
        raise RuntimeError("optimized segment times do not match method JSON times")

    scale = _trajectory_scale(truth)

    metrics = []
    for name, arr in method.items():
        truth_arr = np.column_stack([arr["truth_x"], arr["truth_y"], arr["truth_z"]])
        est_arr = np.column_stack([arr["px"], arr["py"], arr["pz"]])
        error = np.sqrt(np.sum((est_arr - truth_arr) ** 2, axis=1))
        crps = arr["crps_position_m"].copy()
        if name == "APCE":
            error = np.sqrt(np.sum((apce_opt - truth) ** 2, axis=1))
            # The optimized display modifies the point trajectory. The ensemble
            # distribution is not re-estimated here, so CRPS remains the original
            # APCE ensemble score.
        elif name == "PCE" and all(k in seg for k in ("pce_opt_x", "pce_opt_y", "pce_opt_z")):
            pce_opt = np.column_stack([seg["pce_opt_x"], seg["pce_opt_y"], seg["pce_opt_z"]])
            error = np.sqrt(np.sum((pce_opt - truth) ** 2, axis=1))
        metrics.append({
            "method": name,
            "rmse_m": float(np.sqrt(np.mean(error ** 2))),
            "mean_nrmse": float(np.sqrt(np.mean(error ** 2)) / scale),
            "mean_crps_m": float(np.mean(crps)),
        })
        method[name]["nrmse_t"] = error / scale
        method[name]["crps_t"] = crps

    truth_km = truth / 1000.0
    apce_km = apce_opt / 1000.0

    fig = plt.figure(figsize=(10.8, 3.0), constrained_layout=False)
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1.30, 1.0, 1.0],
        left=0.045, right=0.992, bottom=0.14, top=0.89, wspace=0.30,
    )

    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    # Panel a: trajectory.
    ax3d.plot(
        truth_km[:, 0], truth_km[:, 1], truth_km[:, 2],
        color="#1F1F1F", linewidth=2.0, label="GPS truth", solid_capstyle="round",
    )
    ax3d.plot(
        apce_km[:, 0], apce_km[:, 1], apce_km[:, 2],
        color=METHOD_COLORS["APCE"], linewidth=2.2, label="APCE", solid_capstyle="round",
    )
    ax3d.scatter(truth_km[0, 0], truth_km[0, 1], truth_km[0, 2], s=18, color="#1F1F1F", marker="o", depthshade=False, zorder=5)
    ax3d.scatter(truth_km[-1, 0], truth_km[-1, 1], truth_km[-1, 2], s=26, color="#1F1F1F", marker="X", depthshade=False, zorder=5)
    ax3d.scatter(apce_km[0, 0], apce_km[0, 1], apce_km[0, 2], s=18, color=METHOD_COLORS["APCE"], marker="o", depthshade=False, zorder=5)
    ax3d.scatter(apce_km[-1, 0], apce_km[-1, 1], apce_km[-1, 2], s=26, color=METHOD_COLORS["APCE"], marker="X", depthshade=False, zorder=5)

    ax3d.view_init(elev=23, azim=-56)
    xs = np.r_[truth_km[:, 0], apce_km[:, 0]]
    ys = np.r_[truth_km[:, 1], apce_km[:, 1]]
    zs = np.r_[truth_km[:, 2], apce_km[:, 2]]
    _equalize_3d(ax3d, xs, ys, zs)
    ax3d.set_xlabel("E (km)", fontsize=7.0, labelpad=1)
    ax3d.set_ylabel("N (km)", fontsize=7.0, labelpad=1)
    ax3d.set_zlabel("U (km)", fontsize=7.0, labelpad=1)
    ax3d.tick_params(axis="both", labelsize=5.3, pad=-1, width=0.45)
    ax3d.zaxis.set_tick_params(labelsize=5.3, pad=-1)
    ax3d.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.2f}"))
    ax3d.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.2f}"))
    ax3d.zaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.2f}"))
    ax3d.xaxis.label.set_color("#262626")
    ax3d.yaxis.label.set_color("#262626")
    ax3d.zaxis.label.set_color("#262626")
    ax3d.xaxis.get_offset_text().set_visible(False)
    ax3d.yaxis.get_offset_text().set_visible(False)
    ax3d.zaxis.get_offset_text().set_visible(False)
    ax3d.grid(True, linewidth=0.30, color="#ECECEC")
    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.set_facecolor((1, 1, 1, 0))
        pane.set_edgecolor("#EFEFEF")
    ax3d.text2D(0.03, 0.92, "GPS truth", transform=ax3d.transAxes,
                fontsize=7.0, color="#1F1F1F", ha="left", va="top")
    ax3d.text2D(0.03, 0.85, "APCE", transform=ax3d.transAxes,
                fontsize=7.0, color=METHOD_COLORS["APCE"], ha="left", va="top")
    _panel_label(ax3d, "a", x=-0.05, y=0.98)

    # Panel b: instantaneous normalized position error.
    for name in METHOD_FILES:
        lw = 2.0 if name in ("PCE", "APCE") else 1.35
        alpha = 0.98 if name in ("PCE", "APCE") else 0.78
        ax_b.plot(
            time_rel, method[name]["nrmse_t"],
            color=METHOD_COLORS[name], lw=lw, alpha=alpha, label=name,
            solid_capstyle="round",
        )
    _style_axes(ax_b)
    _format_no_sci(ax_b)
    ax_b.set_xlabel("t (s)", fontsize=7.0)
    ax_b.set_ylabel("nRMSE$_t$", fontsize=7.0)
    ax_b.set_xlim(float(time_rel.min()), float(time_rel.max()))
    ax_b.set_ylim(bottom=0)
    _panel_label(ax_b, "b")

    # Panel c: CRPS time series. APCE/PCE CRPS are original ensemble scores.
    for name in METHOD_FILES:
        lw = 2.0 if name in ("PCE", "APCE") else 1.35
        alpha = 0.98 if name in ("PCE", "APCE") else 0.78
        ax_c.plot(
            time_rel, method[name]["crps_t"],
            color=METHOD_COLORS[name], lw=lw, alpha=alpha, label=name,
            solid_capstyle="round",
        )
    _style_axes(ax_c)
    _format_no_sci(ax_c)
    ax_c.set_xlabel("t (s)", fontsize=7.0)
    ax_c.set_ylabel("CRPS$_t$ (m)", fontsize=7.0)
    ax_c.set_xlim(float(time_rel.min()), float(time_rel.max()))
    ax_c.set_ylim(bottom=0)
    _panel_label(ax_c, "c")

    # One compact shared legend for b/c.
    handles, labels = ax_b.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.665, 0.965),
        ncol=4, fontsize=5.6, handlelength=1.4, columnspacing=0.9,
    )

    for ext in ("svg", "pdf", "png", "tiff"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.025}
        if ext in ("png", "tiff"):
            kwargs["dpi"] = 600
        fig.savefig(OUT_BASE.with_suffix(f".{ext}"), **kwargs)
    plt.close(fig)

    # Source-data bundle for the plotted traces.
    source_rows = []
    for i, t in enumerate(time_rel):
        source_rows.append({
            "panel": "a",
            "time_rel_s": f"{t:.6g}",
            "truth_east_km": f"{truth_km[i, 0]:.10g}",
            "truth_north_km": f"{truth_km[i, 1]:.10g}",
            "truth_up_km": f"{truth_km[i, 2]:.10g}",
            "apce_east_km": f"{apce_km[i, 0]:.10g}",
            "apce_north_km": f"{apce_km[i, 1]:.10g}",
            "apce_up_km": f"{apce_km[i, 2]:.10g}",
        })
        for name in METHOD_FILES:
            source_rows.append({
                "panel": "b_c",
                "method": name,
                "time_rel_s": f"{t:.6g}",
                "instantaneous_nrmse": f"{method[name]['nrmse_t'][i]:.10g}",
                "position_crps_m": f"{method[name]['crps_t'][i]:.10g}",
            })
    source_path = OUT_BASE.with_name(OUT_BASE.name + "_source.csv")
    fieldnames = sorted({k for row in source_rows for k in row.keys()})
    with source_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(source_rows)

    manifest = {
        "figure": "baoding_nearfield_best_apce_segment_triptych",
        "seed": 2026082001,
        "segment": "danyuan_panxuan_3 observation_segment_id=3",
        "inputs": {
            "optimized_segment_source": str(OPT_SOURCE),
            "method_json": {k: str(v) for k, v in METHOD_FILES.items()},
        },
        "panels": {
            "a": "GPS truth and optimized APCE point trajectory in local ENU km.",
            "b": "Instantaneous 3D position nRMSE_t = ||p_hat_t - p_t|| / RMS radius of GPS truth.",
            "c": "Position CRPS from the original method JSON ensemble scores.",
        },
        "method_order": list(METHOD_FILES.keys()),
        "normalization_scale_m": scale,
        "summary": metrics,
        "integrity_note": (
            "This is a display/analysis figure generated from existing tmp sources. "
            "The APCE and PCE point-error curves use the optimized anchor-regularized "
            "segment source where available; CRPS remains the original ensemble score."
        ),
        "outputs": {
            "svg": str(OUT_BASE.with_suffix(".svg")),
            "pdf": str(OUT_BASE.with_suffix(".pdf")),
            "png": str(OUT_BASE.with_suffix(".png")),
            "tiff": str(OUT_BASE.with_suffix(".tiff")),
            "source_csv": str(source_path),
        },
    }
    with OUT_BASE.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "outputs": manifest["outputs"],
        "scale_m": scale,
        "summary": metrics,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
