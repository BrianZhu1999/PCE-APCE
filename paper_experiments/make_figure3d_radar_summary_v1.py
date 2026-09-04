from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(r".\hybrid_uncertain_wave")
SOURCE_CSV = PROJECT_ROOT / "source_data" / "figure3_selected5_freq1to8_formal_50seeds_allmethods_20260813" / "combined" / "source_data" / "figure3_freq1to8_formal_authoritative_run_source_data.csv"
OUT_DIR = Path(r"figures")
OUT_STEM = "figure3d_radar_summary_v10_labelup"

PANEL_LABEL_SIZE = 22
CASE_TITLE_SIZE = 16
LEGEND_SIZE = 16
AXIS_LABEL_SIZE = 14
TICK_SIZE = 11
RADAR_TEXT_SIZE = AXIS_LABEL_SIZE

CASES = ["pendulum", "fhn", "robertson"]
CASE_LABELS = {"pendulum": "Forced", "fhn": "FHN", "robertson": "Robertson"}
METHODS = ["aug_enkf", "bma", "pce", "apce"]
METHOD_LABELS = {"aug_enkf": "Aug-EnKF", "bma": "BMA", "pce": "PCE", "apce": "APCE"}
COLORS = {"aug_enkf": "#5b6675", "bma": "#9b75b6", "pce": "#55b7e8", "apce": "#ff8c00"}
LINEWIDTHS = {"aug_enkf": 1.25, "bma": 1.25, "pce": 2.25, "apce": 2.45}
ALPHAS = {"aug_enkf": 0.68, "bma": 0.68, "pce": 0.95, "apce": 0.98}

RADAR_DIMS = ["Acc.", "Dist.", "S-Acc.", "S-Dist.", "Sharp", "Win"]
GROUPS = [
    ("Error tests", 0, 1, "#dce8f5"),
    ("Sparse tests", 2, 3, "#fde8c8"),
    ("Summary tests", 4, 5, "#e8dfef"),
]


plt.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 600,
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.9,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.unicode_minus": False,
})


def method_key(raw: str) -> str:
    s = raw.strip().lower().replace("-", "_")
    if s in {"augenkf", "aug_enkf"}:
        return "aug_enkf"
    if s in {"bma_static", "static_bma"}:
        return "bma"
    return s


def read_rows() -> list[dict[str, str]]:
    rows = []
    with SOURCE_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("status", "").strip().lower() != "completed":
                continue
            case = row["case"].strip().lower()
            method = method_key(row["method"])
            if case in CASES and method in METHODS:
                row = dict(row)
                row["case"] = case
                row["method"] = method
                row["freq"] = str(int(float(row["obs_interval_factor"])))
                rows.append(row)
    return rows


def low_score(values: dict[str, float]) -> dict[str, float]:
    arr = np.asarray([values[m] for m in METHODS], dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return {m: 0.5 for m in METHODS}
    lo = float(np.min(arr[finite]))
    hi = float(np.max(arr[finite]))
    if hi - lo <= 1e-14:
        return {m: 1.0 for m in METHODS}
    return {m: float(np.clip(1.0 - (values[m] - lo) / (hi - lo), 0.0, 1.0)) for m in METHODS}


def normalized_quality(quality: dict[str, float]) -> dict[str, float]:
    arr = np.asarray([quality[m] for m in METHODS], dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return {m: 0.5 for m in METHODS}
    lo = float(np.min(arr[finite]))
    hi = float(np.max(arr[finite]))
    if hi - lo <= 1e-14:
        return {m: 1.0 for m in METHODS}
    return {m: float(np.clip((quality[m] - lo) / (hi - lo), 0.0, 1.0)) for m in METHODS}


def paired_win_scores(rows: list[dict[str, str]], case: str) -> dict[str, float]:
    case_rows = [r for r in rows if r["case"] == case]
    by = {(r["method"], int(r["freq"]), r["seed"]): r for r in case_rows}
    out = {}
    for method in METHODS:
        rates = []
        for opponent in [m for m in METHODS if m != method]:
            keys_m = {(freq, seed) for (m, freq, seed) in by if m == method}
            keys_o = {(freq, seed) for (m, freq, seed) in by if m == opponent}
            for freq, seed in sorted(keys_m & keys_o):
                rm = by[(method, freq, seed)]
                ro = by[(opponent, freq, seed)]
                for metric in ("nrmse", "crps"):
                    try:
                        rates.append(float(rm[metric]) < float(ro[metric]))
                    except Exception:
                        pass
        out[method] = float(np.mean(rates)) if rates else 0.0
    return normalized_quality(out)


def compute_scores(rows: list[dict[str, str]], case: str) -> dict[str, np.ndarray]:
    case_rows = [r for r in rows if r["case"] == case]

    def mean_metric(method: str, metric: str, sparse_only: bool = False) -> float:
        rr = [r for r in case_rows if r["method"] == method and ((int(r["freq"]) >= 6) if sparse_only else True)]
        scale = 100.0 if metric == "nrmse" else 1.0
        vals = []
        for r in rr:
            try:
                vals.append(float(r[metric]) * scale)
            except Exception:
                pass
        return float(np.mean(vals)) if vals else np.nan

    qualities_by_dim = {
        "Acc.": low_score({m: mean_metric(m, "nrmse") for m in METHODS}),
        "Dist.": low_score({m: mean_metric(m, "crps") for m in METHODS}),
        "S-Acc.": low_score({m: mean_metric(m, "nrmse", sparse_only=True) for m in METHODS}),
        "S-Dist.": low_score({m: mean_metric(m, "crps", sparse_only=True) for m in METHODS}),
        "Sharp": low_score({m: mean_metric(m, "interval_width_90") for m in METHODS}),
        "Win": paired_win_scores(rows, case),
    }
    return {
        method: np.asarray([qualities_by_dim[dim][method] for dim in RADAR_DIMS], dtype=float)
        for method in METHODS
    }


def draw_radar(ax: plt.Axes, scores: dict[str, np.ndarray], case_label: str) -> None:
    n = len(RADAR_DIMS)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    closed = np.r_[angles, angles[0]]
    step = 2 * np.pi / n

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.44)
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    ax.set_facecolor("#FFFFFF")
    ax.set_xticks(angles)
    ax.set_xticklabels([])
    ax.set_yticks([])

    ax.fill(closed, np.full_like(closed, 1.0), color="#DDE1E4", alpha=0.70, zorder=-4)
    for group_label, start, end, color in GROUPS:
        theta_start = angles[start] - step * 0.48
        theta_end = angles[end] + step * 0.48
        theta_center = 0.5 * (theta_start + theta_end)
        ax.bar(theta_center, 0.22, width=theta_end - theta_start, bottom=1.030,
               color=color, edgecolor="none", alpha=0.94, align="center", zorder=-3, clip_on=False)
        rotation = -np.degrees(theta_center)
        if rotation < -90:
            rotation += 180
        if rotation > 90:
            rotation -= 180
        if group_label == "Summary tests":
            rotation += 180
        ax.text(theta_center, 1.115, group_label, fontsize=RADAR_TEXT_SIZE,
                rotation=rotation, rotation_mode="anchor", ha="center", va="center",
                color="#111111", clip_on=False)

    for radius in (0.2, 0.4, 0.6, 0.8, 1.0):
        ax.plot(closed, np.full_like(closed, radius), color="#B8C0C5" if radius < 1 else "#0F77A8",
                lw=0.55 if radius < 1 else 0.9, zorder=0)
    for angle in angles:
        ax.plot([angle, angle], [0, 1], color="#9FA9AF", lw=0.55, zorder=0)

    for angle, label in zip(angles, RADAR_DIMS, strict=True):
        deg = np.degrees(angle)
        ha = "center"
        if 8 < deg < 172:
            ha = "left"
        elif 188 < deg < 352:
            ha = "right"
        ax.text(angle, 1.355, label, fontsize=RADAR_TEXT_SIZE, ha=ha, va="center", clip_on=False)

    for radius, label in zip((0.0, 0.2, 0.4, 0.6, 0.8, 1.0), ("0", "0.2", "0.4", "0.6", "0.8", "1"), strict=True):
        ax.text(np.deg2rad(0.0), radius, label, fontsize=RADAR_TEXT_SIZE,
                ha="center", va="bottom" if radius > 0 else "center", color="#111111", zorder=2)

    area = {m: float(np.mean(scores[m])) for m in METHODS}
    fill_order = sorted(METHODS, key=lambda m: area[m], reverse=True)
    line_order = sorted(METHODS, key=lambda m: area[m])
    for m in fill_order:
        vals = np.r_[scores[m], scores[m][0]]
        ax.fill(closed, vals, color=COLORS[m], alpha=0.13 if m in {"pce", "apce"} else 0.075, zorder=2)
    for m in line_order:
        vals = np.r_[scores[m], scores[m][0]]
        ax.plot(closed, vals, color=COLORS[m], lw=LINEWIDTHS[m], alpha=ALPHAS[m], zorder=7 if m not in {"pce", "apce"} else 9)

    ax.text(0.5, -0.020, case_label, transform=ax.transAxes, fontsize=CASE_TITLE_SIZE,
            ha="center", va="top", clip_on=False)


def write_source(scores_by_case: dict[str, dict[str, np.ndarray]]) -> None:
    out = OUT_DIR / f"{OUT_STEM}_scores.csv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "method", "dimension", "score"])
        writer.writeheader()
        for case, scores in scores_by_case.items():
            for method, vals in scores.items():
                for dim, val in zip(RADAR_DIMS, vals, strict=True):
                    writer.writerow({"case": case, "method": method, "dimension": dim, "score": f"{float(val):.10g}"})


def main() -> None:
    rows = read_rows()
    scores_by_case = {case: compute_scores(rows, case) for case in CASES}
    fig = plt.figure(figsize=(15.56, 5.35), facecolor="white")
    left, right = 0.002, 0.998
    gap = 0.018
    width = (right - left - 2 * gap) / 3.0
    axes = []
    for i, case in enumerate(CASES):
        ax = fig.add_axes([left + i * (width + gap), 0.070, width, 0.810], projection="polar")
        draw_radar(ax, scores_by_case[case], CASE_LABELS[case])
        axes.append(ax)

    fig.text(0.006, 0.985, "d", fontsize=PANEL_LABEL_SIZE, fontweight="bold", ha="left", va="top")
    handles = [
        Line2D([0], [0], color=COLORS[m], lw=LINEWIDTHS[m], marker="s", ms=7.0,
               mfc="white", mec=COLORS[m], mew=1.4, label=METHOD_LABELS[m], alpha=ALPHAS[m])
        for m in METHODS
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.500, 0.990), ncol=4,
               frameon=False, prop={"family": "Arial", "size": LEGEND_SIZE},
               handlelength=1.15, handletextpad=0.36, columnspacing=0.85)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_source(scores_by_case)
    for ext, kwargs in [("png", {"dpi": 600}), ("pdf", {}), ("svg", {}), ("tiff", {"dpi": 600})]:
        fig.savefig(OUT_DIR / f"{OUT_STEM}.{ext}", facecolor="white", bbox_inches=None, pad_inches=0, **kwargs)
    plt.close(fig)
    print(OUT_DIR / f"{OUT_STEM}.png")


if __name__ == "__main__":
    main()
